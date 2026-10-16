// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/pattern_rewriter.h"

#include <algorithm>
#include <limits>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime::rewrite {
namespace {
#ifndef DISABLE_CONTRIB_OPS
bool ShapeIs(const NodeArg& value, gsl::span<const int64_t> expected) {
  const auto* shape = value.Shape();
  if (!shape || shape->dim_size() != static_cast<int>(expected.size())) return false;
  for (int i = 0; i < shape->dim_size(); ++i) {
    if (!shape->dim(i).has_dim_value() || shape->dim(i).dim_value() != expected[i]) return false;
  }
  return true;
}

bool IntegersAre(const Graph& graph, const NodeArg& value, gsl::span<const int64_t> expected, bool scalar = false) {
  const auto* tensor = graph_utils::GetConstantInitializer(graph, value.Name());
  if (!tensor || tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
      tensor->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL ||
      tensor->dims_size() != (scalar ? 0 : 1) ||
      (!scalar && tensor->dims(0) != static_cast<int64_t>(expected.size())))
    return false;
  Initializer data(graph, *tensor, graph.ModelPath());
  return data.size() == expected.size() && std::equal(expected.begin(), expected.end(), data.data<int64_t>());
}

void BindInt(Match& match, const char* name, int64_t value) {
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_name(name);
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr.set_i(value);
  match.update(name, std::move(attr));
}

std::optional<int64_t> StaticDimension(const NodeArg& value, int index) {
  const auto* shape = value.Shape();
  if (!shape || index < 0 || index >= shape->dim_size() || !shape->dim(index).has_dim_value() ||
      shape->dim(index).dim_value() <= 0)
    return std::nullopt;
  return shape->dim(index).dim_value();
}

bool BlockSize(int64_t size) {
  return size >= 16 && (size & (size - 1)) == 0;
}

bool SameTokenDimension(const NodeArg& x, const NodeArg& logits) {
  const auto& a = x.Shape()->dim(0);
  const auto& b = logits.Shape()->dim(0);
  return (a.has_dim_value() && b.has_dim_value() && a.dim_value() > 0 && a.dim_value() == b.dim_value()) ||
         (a.has_dim_param() && !a.dim_param().empty() && b.has_dim_param() && a.dim_param() == b.dim_param());
}

enum class ZeroPoint { Symmetric,
                       Float,
                       Packed };

Rule MatMulRule(ZeroPoint zero_point) {
  std::string pattern = "(a:float[], w:uint8[N,B,P], s:float[N,B], count:int64, shape:int64[2]";
  if (zero_point == ZeroPoint::Float) pattern += ", z:float[N,B]";
  if (zero_point == ZeroPoint::Packed) pattern += ", z:uint8[N,Z], zcount:int64";
  pattern += R"() => {
    codes = local.quant.Unpack(w, count, bits=4)
    q = Cast(codes, to=1)
  )";
  if (zero_point == ZeroPoint::Symmetric) {
    pattern += "centered = Sub(q, 8.0)\n";
  } else {
    if (zero_point == ZeroPoint::Packed)
      pattern += "zcodes = local.quant.Unpack(z, zcount, bits=4)\nzfloat = Cast(zcodes, to=1)\n";
    pattern += "zp = Unsqueeze(";
    pattern += zero_point == ZeroPoint::Float ? "z" : "zfloat";
    pattern += ", [-1])\ncentered = Sub(q, zp)\n";
  }
  pattern += R"(
    scales = Unsqueeze(s, [-1])
    blocks = Mul(centered, scales)
    rows = Reshape(blocks, shape, allowzero=0)
    weights = Transpose(rows, perm=[1,0])
    return MatMul(a, weights)
  })";
  auto check = [zero_point](const Graph& graph, Match& m) {
    const auto n = StaticDimension(*m.valueOf("w"), 0);
    const auto blocks = StaticDimension(*m.valueOf("w"), 1);
    const auto bytes = StaticDimension(*m.valueOf("w"), 2);
    const auto* a_shape = m.valueOf("a")->Shape();
    if (!n || !blocks || !bytes || !a_shape || a_shape->dim_size() < 1)
      return m.Reject("MatMulNBits requires static packed weight dimensions and known activation rank.");
    const auto k = StaticDimension(*m.valueOf("a"), a_shape->dim_size() - 1);
    // Divide before multiplying: these checks also bound all derived dimensions.
    if (!k || *bytes > *k / 2 || !BlockSize(*bytes * 2) ||
        *k % (*bytes * 2) != 0 || *blocks != *k / (*bytes * 2) ||
        !ShapeIs(*m.valueOf("s"), InlinedVector<int64_t>{*n, *blocks}) ||
        !IntegersAre(graph, *m.valueOf("count"), InlinedVector<int64_t>{*bytes * 2}, true) ||
        !IntegersAre(graph, *m.valueOf("shape"), InlinedVector<int64_t>{*n, *k}))
      return m.Reject("MatMulNBits requires complete power-of-two blocks and canonical row reshaping.");
    if (zero_point == ZeroPoint::Float &&
        !ShapeIs(*m.valueOf("z"), InlinedVector<int64_t>{*n, *blocks}))
      return m.Reject("MatMulNBits floating zero points must be per block.");
    if (zero_point == ZeroPoint::Packed &&
        (!ShapeIs(*m.valueOf("z"), InlinedVector<int64_t>{*n, *blocks / 2 + *blocks % 2}) ||
         !IntegersAre(graph, *m.valueOf("zcount"), InlinedVector<int64_t>{*blocks}, true)))
      return m.Reject("MatMulNBits packed zero points must unpack exactly the block count.");
    BindInt(m, "k_attr", *k);
    BindInt(m, "n_attr", *n);
    BindInt(m, "block_size", *bytes * 2);
    return true;
  };
  std::string replacement = zero_point == ZeroPoint::Symmetric
                                ? "(a,w,s,k_attr,n_attr,block_size) => com.microsoft.MatMulNBits(a,w,s,"
                                : "(a,w,s,z,k_attr,n_attr,block_size) => com.microsoft.MatMulNBits(a,w,s,z,";
  replacement += "K=k_attr,N=n_attr,block_size=block_size,bits=4,accuracy_level=0)";
  const char* name = zero_point == ZeroPoint::Symmetric ? "UnpackMatMulNBitsSymmetric"
                     : zero_point == ZeroPoint::Float   ? "UnpackMatMulNBitsFloatZeroPoint"
                                                        : "UnpackMatMulNBitsPackedZeroPoint";
  return Rule{name, pattern, replacement, check, {{kOnnxDomain, 21, 21}, {"local.quant", 1, 1}, {kMSDomain, 1, 1}}};
}

// Expert storage [E,N,K/2] is reshaped into blocks, dequantized, then transposed to [E,K,N].
std::string ExpertPattern(const std::string& p) {
  return p + "b = Reshape(" + p + "w, " + p + "blocked, allowzero=0)\n" +
         p + "q = local.quant.Unpack(" + p + "b, count, bits=4)\n" +
         p + "f = Cast(" + p + "q, to=1)\n" +
         p + "c = Sub(" + p + "f, 8.0)\n" +
         p + "s4 = Unsqueeze(" + p + "s, [-1])\n" +
         p + "d = Mul(" + p + "c, " + p + "s4)\n" +
         p + "r = Reshape(" + p + "d, " + p + "rows, allowzero=0)\n" +
         p + " = Transpose(" + p + "r, perm=[0,2,1])\n";
}

enum class Routing { TopKSoftmax,
                     SoftmaxTopK,
                     SoftmaxTopKNormalized };

Rule QMoERule(Routing routing) {
  std::string pattern = R"(
    (x:float[M,H], logits:float[M,E], top_k:int64[1], count:int64,
     gw:uint8[E,I,HP], gs:float[E,I,HB], gblocked:int64[4], grows:int64[3],
     uw:uint8[E,I,HP], us:float[E,I,HB], ublocked:int64[4], urows:int64[3],
     dw:uint8[E,H,IP], ds:float[E,H,IB], dblocked:int64[4], drows:int64[3],
     flat_shape:int64[2], output_shape:int64[3]) => {
  )";
  pattern += ExpertPattern("g") + ExpertPattern("u") + ExpertPattern("d");
  if (routing == Routing::TopKSoftmax) {
    pattern += R"(
      selected, indices:int64[M,T] = TopK(logits, top_k, axis=-1, largest=1)
      weights = Softmax(selected, axis=-1)
    )";
  } else {
    pattern += R"(
      probs = Softmax(logits, axis=-1)
      selected, indices:int64[M,T] = TopK(probs, top_k, axis=-1, largest=1)
    )";
    if (routing == Routing::SoftmaxTopKNormalized)
      pattern += R"(
        sum = ReduceSum(selected, [-1], keepdims=1, noop_with_empty_axes=0)
        weights = Div(selected, sum)
      )";
  }
  pattern += R"(
    gate = local.moe.GroupedMatMul(x, g, indices)
    up = local.moe.GroupedMatMul(x, u, indices)
    sigmoid = Sigmoid(gate)
    activated = Mul(gate, sigmoid)
    product = Mul(activated, up)
    flat = Reshape(product, flat_shape, allowzero=0)
    flat_indices = Reshape(indices, [-1,1], allowzero=0)
    projected = local.moe.GroupedMatMul(flat, d, flat_indices)
    regrouped = Reshape(projected, output_shape, allowzero=0)
  )";
  pattern += std::string("expanded = Unsqueeze(") +
             (routing == Routing::SoftmaxTopK ? "selected" : "weights") + R"(, [-1])
    weighted = Mul(regrouped, expanded)
    return ReduceSum(weighted, [1], keepdims=0, noop_with_empty_axes=0)
  })";
  auto check = [](const Graph& graph, Match& m) {
    const auto experts = StaticDimension(*m.valueOf("gw"), 0);
    const auto intermediate = StaticDimension(*m.valueOf("gw"), 1);
    const auto hidden = StaticDimension(*m.valueOf("dw"), 1);
    const auto hb = StaticDimension(*m.valueOf("gs"), 2);
    const auto ib = StaticDimension(*m.valueOf("ds"), 2);
    const auto top_k = StaticDimension(*m.valueOf("indices"), 1);
    if (!experts || !intermediate || !hidden || !hb || !ib || !top_k ||
        *top_k > *experts || *hidden % *hb != 0 || *intermediate > std::numeric_limits<int64_t>::max() / 2)
      return m.Reject("QMoE requires positive static expert, hidden, block, and top-k dimensions.");
    const int64_t block = *hidden / *hb;
    if (!BlockSize(block) || *hidden % 2 != 0 || *intermediate % block != 0 ||
        *ib != *intermediate / block)
      return m.Reject("QMoE requires a common power-of-two block size dividing both feature dimensions.");
    for (const char* prefix : {"g", "u", "d"}) {
      const std::string p(prefix);
      const int64_t n = p == "d" ? *hidden : *intermediate;
      const int64_t k = p == "d" ? *intermediate : *hidden;
      if (!ShapeIs(*m.valueOf(p + "w"), InlinedVector<int64_t>{*experts, n, k / 2}) ||
          !ShapeIs(*m.valueOf(p + "s"), InlinedVector<int64_t>{*experts, n, k / block}) ||
          !IntegersAre(graph, *m.valueOf(p + "blocked"), InlinedVector<int64_t>{*experts, n, k / block, block / 2}) ||
          !IntegersAre(graph, *m.valueOf(p + "rows"), InlinedVector<int64_t>{*experts, n, k}))
        return m.Reject("QMoE requires canonical expert-major, output-row-major packed weights and scales.");
    }
    const auto x_hidden = StaticDimension(*m.valueOf("x"), 1);
    const auto logit_experts = StaticDimension(*m.valueOf("logits"), 1);
    if (!x_hidden || *x_hidden != *hidden || !logit_experts || *logit_experts != *experts ||
        !SameTokenDimension(*m.valueOf("x"), *m.valueOf("logits")) ||
        !IntegersAre(graph, *m.valueOf("count"), InlinedVector<int64_t>{block}, true) ||
        !IntegersAre(graph, *m.valueOf("top_k"), InlinedVector<int64_t>{*top_k}) ||
        !IntegersAre(graph, *m.valueOf("flat_shape"), InlinedVector<int64_t>{-1, *intermediate}) ||
        !IntegersAre(graph, *m.valueOf("output_shape"), InlinedVector<int64_t>{-1, *top_k, *hidden}))
      return m.Reject("QMoE requires canonical token/slot-preserving shapes and constant counts.");
    BindInt(m, "k", *top_k);
    BindInt(m, "block_size", block);
    return true;
  };
  // Interleave entire output rows (including their block scales), never individual nibbles.
  std::string replacement = R"(
    (x,logits,top_k,gw,gs,uw,us,dw,ds,k,block_size) => {
  )";
  if (routing == Routing::TopKSoftmax) {
    replacement += R"(
      selected, indices = TopK(logits, top_k, axis=-1, largest=1)
      mixing = Softmax(selected, axis=-1)
    )";
  } else {
    replacement += R"(
      probs = Softmax(logits, axis=-1)
      selected, indices = TopK(probs, top_k, axis=-1, largest=1)
    )";
    if (routing == Routing::SoftmaxTopKNormalized)
      replacement += R"(
        sum = ReduceSum(selected, [-1], keepdims=1, noop_with_empty_axes=0)
        mixing = Div(selected, sum)
      )";
  }
  // QMoE's CPU top-k tie-break differs from ONNX TopK. A mask fixes the selected set;
  // router_weights also avoids the kernel's implicit top-k-softmax aggregation.
  replacement += R"(
      router_shape = Shape(logits)
      zeros = Expand(0.0, router_shape)
      index_shape = Shape(indices)
      ones = Expand(1.0, index_shape)
      selection = ScatterElements(zeros, indices, ones, axis=1, reduction="none")
      router_weights = ScatterElements(zeros, indices,
  )";
  replacement += routing == Routing::SoftmaxTopK ? "selected" : "mixing";
  replacement += R"(, axis=1, reduction="none")
      g = Unsqueeze(gw, [2])
      u = Unsqueeze(uw, [2])
      gu = Concat(g, u, axis=2)
      ws = Shape(gw)
      wshape = Mul(ws, [1,2,1])
      fc1 = Reshape(gu, wshape)
      sg = Unsqueeze(gs, [2])
      su = Unsqueeze(us, [2])
      scales = Concat(sg, su, axis=2)
      ss = Shape(gs)
      sshape = Mul(ss, [1,2,1])
      fc1_scales = Reshape(scales, sshape)
      return com.microsoft.QMoE(x,selection,fc1,fc1_scales,None,dw,ds,
        None,None,None,None,None,None,None,router_weights,
        activation_type="swiglu",swiglu_fusion=1,expert_weight_bits=4,
        quant_type="int",weights_prepacked=0,accuracy_level=0,
        block_size=block_size,k=k,normalize_routing_weights=0)
    }
  )";
  const char* name = routing == Routing::TopKSoftmax   ? "UnpackQMoETopKSoftmax"
                     : routing == Routing::SoftmaxTopK ? "UnpackQMoESoftmaxTopK"
                                                       : "UnpackQMoESoftmaxTopKNormalized";
  return Rule{name, pattern, replacement, check, {{kOnnxDomain, 21, 21}, {"local.quant", 1, 1}, {"local.moe", 1, 1}, {kMSDomain, 1, 1}}};
}
#endif
}  // namespace

Rules MakeQuantizedFusionRules() {
  Rules rules;
#ifndef DISABLE_CONTRIB_OPS
  for (auto zp : {ZeroPoint::Symmetric, ZeroPoint::Float, ZeroPoint::Packed})
    rules.push_back(MatMulRule(zp));
  for (auto routing : {Routing::TopKSoftmax, Routing::SoftmaxTopK, Routing::SoftmaxTopKNormalized})
    rules.push_back(QMoERule(routing));
#endif
  return rules;
}
}  // namespace onnxruntime::rewrite
