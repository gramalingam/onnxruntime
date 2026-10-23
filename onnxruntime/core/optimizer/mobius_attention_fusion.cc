// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/mobius_attention_fusion.h"

#include <cmath>
#include <limits>

namespace onnxruntime {
namespace rewrite {
#ifndef DISABLE_CONTRIB_OPS
namespace {

// Mobius d69efe03a9f4c1ada744bc31dd01077a710b3c39:
// _pixtral_vision.py, _qwen3_vl_vision.py, and _attention.py.
constexpr const char* kAttentionPatterns = R"(
  pattern Vision(q,k,v) {
    y=Attention(q,k,v,None,None,None,None)
    return y
  }
  pattern FuseVision(q,k,v) {
    attention=Vision(q,k,v)
    return (y=attention.y)
  }

  interface ScalarLike(c) -> (y)
  pattern plain implements ScalarLike { return (y=c) }
  pattern cast_like implements ScalarLike: (like) -> { y=CastLike(c,like); return y }
  pattern ScalarForms=plain|cast_like
  pattern BlockMask(hidden,cu,zero,negative) {
    n=Shape(hidden,start=0,end=1)
    limit=Squeeze(n)
    positions=Range(0,limit,1)
    rows=Unsqueeze(positions,[1])
    boundaries=Unsqueeze(cu,[0])
    ge=GreaterOrEqual(rows,boundaries)
    integers=Cast(ge,to=7)
    counts=ReduceSum(integers,[1],keepdims=0)
    segments=Sub(counts,1)
    r=Unsqueeze(segments,[1])
    c=Unsqueeze(segments,[0])
    same=Equal(r,c)
    yes=ScalarForms(zero)
    no=ScalarForms(negative)
    bias=Where(same,yes.y,no.y)
    y=Unsqueeze(bias,[0,1])
    return y
  }
  pattern Packed(q,k,v,bias) {
    qb=Unsqueeze(q,[0])
    kb=Unsqueeze(k,[0])
    vb=Unsqueeze(v,[0])
    y=Attention(qb,kb,vb,bias,None,None,None)
    return y
  }
  pattern FusePacked(q,k,v,hidden,cu,zero,negative) {
    mask=BlockMask(hidden,cu,zero,negative)
    attention=Packed(q,k,v,mask.y)
    y=Squeeze(attention.y,[0])
    return y
  }

  pattern Decoder(q,k,v,pk,pv) {
    y,present_k,present_v=Attention(q,k,v,None,pk,pv,None)
    return y,present_k,present_v
  }
  pattern FuseDecoder(q,k,v,pk,pv) {
    attention=Decoder(q,k,v,pk,pv)
    return (y=attention.y,present_k=attention.present_k,present_v=attention.present_v)
  }
)";

struct AttentionView {
  const NodeArg* q;
  const NodeArg* k;
  const NodeArg* v;
  const NodeArg* y;
  explicit AttentionView(BindingView view)
      : q(view.valueOf("q")), k(view.valueOf("k")), v(view.valueOf("v")), y(view.valueOf("y")) {}
};

struct ScalarView {
  const NodeArg* value;
  const NodeArg* like;
  explicit ScalarView(BindingView view)
      : value(view.valueOf("c")),
        like(view.alternative() == "cast_like" ? view.alternativeView().valueOf("like") : nullptr) {
    ORT_ENFORCE(view.alternative() == "plain" || view.alternative() == "cast_like");
  }
};

struct PackedView {
  AttentionView attention;
  const NodeArg* hidden;
  const NodeArg* cu;
  ScalarView zero;
  ScalarView negative;
  explicit PackedView(const Match& match)
      : attention(match.instance("attention")),
        hidden(match.instance("mask").valueOf("hidden")),
        cu(match.instance("mask").valueOf("cu")),
        zero(match.instance("mask").instance("yes")),
        negative(match.instance("mask").instance("no")) {}
};

struct DecoderView {
  AttentionView attention;
  const NodeArg* pk;
  const NodeArg* pv;
  explicit DecoderView(const Match& match)
      : attention(match.instance("attention")),
        pk(match.instance("attention").valueOf("pk")),
        pv(match.instance("attention").valueOf("pv")) {}
};

bool Float32(const NodeArg* value) {
  const auto* type = value ? value->TypeAsProto() : nullptr;
  return type && type->has_tensor_type() &&
         type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
}

bool Shape(const NodeArg* value, int rank, InlinedVector<int64_t>& dimensions, bool allow_empty = false) {
  const auto* shape = value ? value->Shape() : nullptr;
  if (!shape || shape->dim_size() != rank) return false;
  dimensions.clear();
  for (const auto& dim : shape->dim()) {
    if (!dim.has_dim_value() || dim.dim_value() < (allow_empty ? 0 : 1) ||
        dim.dim_value() > std::numeric_limits<int32_t>::max())
      return false;
    dimensions.push_back(dim.dim_value());
  }
  return true;
}

int64_t Integer(const Node& node, const char* name, int64_t fallback) {
  const auto it = node.GetAttributes().find(name);
  return it == node.GetAttributes().end() ? fallback : it->second.i();
}

float Floating(const Node& node, const char* name, float fallback) {
  const auto it = node.GetAttributes().find(name);
  return it == node.GetAttributes().end() ? fallback : it->second.f();
}

bool Attributes(const Node& attention, Match& match, bool causal) {
  for (const auto& [name, value] : attention.GetAttributes()) {
    ORT_UNUSED_PARAMETER(value);
    if (name != "q_num_heads" && name != "kv_num_heads" && name != "scale" && name != "is_causal" &&
        name != "softcap" && name != "softmax_precision" && name != "qk_matmul_output_mode")
      return match.Reject("Unsupported Attention attribute.");
  }
  const auto q_heads = Integer(attention, "q_num_heads", 0);
  const auto kv_heads = Integer(attention, "kv_num_heads", 0);
  const float scale = Floating(attention, "scale", 0.0f);
  const auto precision = Integer(attention, "softmax_precision", 0);
  if (q_heads <= 0 || q_heads > std::numeric_limits<int32_t>::max() ||
      kv_heads <= 0 || q_heads % kv_heads != 0 || !std::isfinite(scale) || scale <= 0 ||
      Integer(attention, "is_causal", 0) != (causal ? 1 : 0) ||
      Floating(attention, "softcap", 0.0f) != 0.0f ||
      (precision != 0 && precision != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
      Integer(attention, "qk_matmul_output_mode", 0) != 0)
    return match.Reject("Unsupported heads, scale, causality, softcap or softmax precision.");
  for (const char* name : {"q_num_heads", "kv_num_heads", "scale"})
    match.update(std::string("fused_") + name, attention.GetAttributes().at(name));
  return true;
}

bool Qkv(const AttentionView& view, int rank, int64_t heads, int64_t kv_heads,
         InlinedVector<int64_t>& q, InlinedVector<int64_t>& k, InlinedVector<int64_t>& v) {
  if (!Float32(view.q) || !Float32(view.k) || !Float32(view.v) ||
      !Shape(view.q, rank, q) || !Shape(view.k, rank, k) || !Shape(view.v, rank, v))
    return false;
  for (int i = 0; i < rank - 1; ++i)
    if (q[i] != k[i] || k[i] != v[i]) return false;
  return q.back() % heads == 0 && k.back() % kv_heads == 0 && k.back() == v.back() &&
         q.back() / heads == k.back() / kv_heads;
}

bool Constant(const Graph& graph, const NodeArg* value, float expected) {
  InlinedVector<float> data;
  bool available;
  ORT_THROW_IF_ERROR(ReadConstantTensor(graph, value, data, available));
  return available && value->Shape() && value->Shape()->dim_size() == 0 && data.size() == 1 &&
         data[0] == expected;
}

}  // namespace
#endif

Rules MakeMobiusAttentionRules(bool include_cuda_packed_attention) {
  Rules rules;
#ifndef DISABLE_CONTRIB_OPS
  const PatternLibrary library(kAttentionPatterns);
  const InlinedVector<OpsetRequirement, 1> opsets{{kOnnxDomain, 23, 24}, {kMSDomain, 1, 1}};
  rules.emplace_back(
      "MobiusVisionAttention", library.GetPattern("FuseVision"),
      "(q,k,v,fused_q_num_heads,fused_scale)=>"
      "com.microsoft.MultiHeadAttention(q,k,v,num_heads=fused_q_num_heads,scale=fused_scale)",
      [](const Graph& graph, Match& match) {
        const AttentionView view(match.instance("attention"));
        const Node& node = *graph.GetProducerNode(view.y->Name());
        if (!Attributes(node, match, false)) return false;
        const auto heads = Integer(node, "q_num_heads", 0);
        InlinedVector<int64_t> q, k, v;
        if (heads != Integer(node, "kv_num_heads", 0) || !Qkv(view, 3, heads, heads, q, k, v))
          return match.Reject("Vision attention requires static same-shaped FP32 equal-head Q/K/V.");
        return true;
      },
      opsets);
  rules.emplace_back(
      "MobiusDecoderAttention", library.GetPattern("FuseDecoder"),
      [](const Graph&, const Match& match) {
        const DecoderView view(match);
        const auto total = view.attention.q->Shape()->dim(1).dim_value() + view.pk->Shape()->dim(2).dim_value();
        const auto batch = view.attention.q->Shape()->dim(0).dim_value();
        return Replacement(
            "(q,k,v,pk,pv,fused_q_num_heads,fused_kv_num_heads,fused_scale)=>{"
            "length=Cast(" +
            std::to_string(total - 1) +
            ",to=6);lengths=Expand(length,[" + std::to_string(batch) +
            "]);"
            "total=Cast(" +
            std::to_string(total) +
            ",to=6);"
            "y,pk_out,pv_out=com.microsoft.GroupQueryAttention(q,k,v,pk,pv,lengths,total,"
            "num_heads=fused_q_num_heads,kv_num_heads=fused_kv_num_heads,scale=fused_scale,"
            "causal=1,do_rotary=0);return y,pk_out,pv_out}");
      },
      [](const Graph& graph, Match& match) {
        const DecoderView view(match);
        const Node& node = *graph.GetProducerNode(view.attention.y->Name());
        if (!node.GetExecutionProviderType().empty() &&
            node.GetExecutionProviderType() != kCpuExecutionProvider)
          return match.Reject("The admitted FP32 GQA form requires the CPU kernel.");
        if (!Attributes(node, match, true)) return false;
        const auto heads = Integer(node, "q_num_heads", 0);
        const auto kv_heads = Integer(node, "kv_num_heads", 0);
        InlinedVector<int64_t> q, k, v, pk, pv;
        if (!Qkv(view.attention, 3, heads, kv_heads, q, k, v) ||
            !Float32(view.pk) || !Float32(view.pv) ||
            !Shape(view.pk, 4, pk, true) || !Shape(view.pv, 4, pv, true) ||
            pk != pv || pk[0] != q[0] || pk[1] != kv_heads || pk[3] != k[2] / kv_heads || pk[3] % 8 != 0 ||
            pk[2] > std::numeric_limits<int32_t>::max() - q[1] ||
            (pk[2] > 0 && q[1] > 1 && q[0] != 1))
          return match.Reject("Decoder requires unpadded static FP32 Q/K/V and compatible dynamic BNSH caches.");
        return true;
      },
      opsets);
  if (include_cuda_packed_attention) {
    rules.emplace_back(
        "MobiusSingleSequencePackedAttention", library.GetPattern("FusePacked"),
        [](const Graph&, const Match& match) {
          const PackedView view(match);
          const auto tokens = view.attention.q->Shape()->dim(0).dim_value();
          return Replacement(
              "(q,k,v,cu,fused_q_num_heads,fused_scale)=>{"
              "positions=Range(0," +
              std::to_string(tokens) +
              ",1);"
              "positions_i32=Cast(positions,to=6);offset=Unsqueeze(positions_i32,[0]);"
              "cu_i32=Cast(cu,to=6);"
              "return com.microsoft.PackedMultiHeadAttention(q,k,v,None,offset,cu_i32,"
              "num_heads=fused_q_num_heads,scale=fused_scale)}");
        },
        [](const Graph& graph, Match& match) {
          const PackedView view(match);
          const Node& node = *graph.GetProducerNode(view.attention.y->Name());
          if (!node.GetExecutionProviderType().empty() &&
              node.GetExecutionProviderType() != kCudaExecutionProvider)
            return match.Reject("PackedMHA requires CUDA deployment, not an assigned non-CUDA provider.");
          if (!Attributes(node, match, false)) return false;
          const auto heads = Integer(node, "q_num_heads", 0);
          InlinedVector<int64_t> q, k, v, hidden, cu_shape, cu;
          bool available;
          ORT_THROW_IF_ERROR(ReadConstantTensor(graph, view.cu, cu, available));
          if (heads != Integer(node, "kv_num_heads", 0) || !Qkv(view.attention, 2, heads, heads, q, k, v) ||
              !Shape(view.hidden, 2, hidden) || hidden[0] != q[0] ||
              !Shape(view.cu, 1, cu_shape) || cu_shape[0] != 2 || !available ||
              cu.size() != 2 || cu[0] != 0 || cu[1] != q[0] ||
              (view.zero.like && view.zero.like != view.attention.q) ||
              (view.negative.like && view.negative.like != view.attention.q) ||
              !Constant(graph, view.zero.value, 0.0f) || !Constant(graph, view.negative.value, -10000.0f))
            return match.Reject("Packed attention requires FP32 and immutable cu_seqlens=[0,token_count].");
          return true;
        },
        opsets);
  }
#else
  ORT_UNUSED_PARAMETER(include_cuda_packed_attention);
#endif
  return rules;
}

}  // namespace rewrite
}  // namespace onnxruntime
