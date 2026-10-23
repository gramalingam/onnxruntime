// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/pattern_rewriter.h"

#include <algorithm>
#include <cmath>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime::rewrite {
namespace {
#ifndef DISABLE_CONTRIB_OPS
// Mobius d69efe03a9f4c1ada744bc31dd01077a710b3c39, src/mobius/functions/
// matmul_nbits.py and matmul_block_quantized_fp4_weight.py (opset 24).
constexpr const char* kProjectionPatterns = R"(
  interface PackedNibbles(packed) -> (interleaved)
  pattern Nibbles3 implements PackedNibbles: (mask:uint8, shift:uint8) -> {
    lo = BitwiseAnd(packed,mask)
    hi = BitShift(packed,shift,direction="RIGHT")
    low = Unsqueeze(lo,[3])
    high = Unsqueeze(hi,[3])
    interleaved = Concat(low,high,axis=3)
    return interleaved
  }
  pattern Nibbles2 implements PackedNibbles: (mask:uint8, shift:uint8) -> {
    lo = BitwiseAnd(packed,mask)
    hi = BitShift(packed,shift,direction="RIGHT")
    low = Unsqueeze(lo,[2])
    high = Unsqueeze(hi,[2])
    interleaved = Concat(low,high,axis=2)
    return interleaved
  }
  interface ZeroPoint(scales) -> (q)
  pattern PackedZero implements ZeroPoint: (z:uint8[_,_]) -> {
    unpack = Nibbles2(z)
    rows = Reshape(unpack.interleaved,[0,-1],allowzero=0)
    shape = Shape(scales)
    end = Slice(shape,[1],[2])
    trimmed = Slice(rows,[0],end,[1])
    q = Cast(trimmed,to=21)
    return q
  }
  pattern MidpointZero implements ZeroPoint: (midpoint:tensor) -> {
    shape = Shape(scales)
    zeros = ConstantOfShape(shape,value=midpoint)
    q = Cast(zeros,to=21)
    return q
  }
  pattern Zeros = PackedZero | MidpointZero
  pattern Int4(a:tensor<T>[], w:uint8[N,B,P], s:tensor<T>[N,B], block:int) {
    unpack = Nibbles3(w)
    blocks = Reshape(unpack.interleaved,[0,0,-1],allowzero=0)
    rows = Reshape(blocks,[0,-1],allowzero=0)
    codes = Cast(rows,to=21)
    zero = Zeros(s)
    dequant = DequantizeLinear(codes,s,zero.q,axis=1,block_size=block)
    weight = Transpose(dequant,perm=[1,0])
    return MatMul(a,weight)
  }
  pattern NvFp4(a:float16[], w:uint8[N,P], s:uint8[N,B], g:float[],
      fp4:float[16], fp8:float[256]) {
    unpack = Nibbles2(w)
    codes = Reshape(unpack.interleaved,[0,-1],allowzero=0)
    indices = Cast(codes,to=7)
    values = Gather(fp4,indices,axis=0)
    scale_indices = Cast(s,to=7)
    scales = Gather(fp8,scale_indices,axis=0)
    unsqueezed = Unsqueeze(scales,[2])
    scale_shape = Shape(unsqueezed)
    prefix = Slice(scale_shape,[0],[2])
    expanded_shape = Concat(prefix,[16],axis=0)
    expanded = Expand(unsqueezed,expanded_shape)
    block_scales = Reshape(expanded,[0,-1],allowzero=0)
    scaled = Mul(values,block_scales)
    weight_float = Mul(scaled,g)
    weight = CastLike(weight_float,a)
    transposed = Transpose(weight,perm=[1,0])
    return MatMul(a,transposed)
  }
)";

bool ShapeIs(const NodeArg& value, gsl::span<const int64_t> expected) {
  const auto* shape = value.Shape();
  if (!shape || shape->dim_size() != static_cast<int>(expected.size())) return false;
  for (int i = 0; i < shape->dim_size(); ++i) {
    if (!shape->dim(i).has_dim_value() || shape->dim(i).dim_value() != expected[i]) return false;
  }
  return true;
}

std::optional<int64_t> PositiveDimension(const NodeArg& value, int index) {
  const auto* shape = value.Shape();
  if (!shape || index < 0 || index >= shape->dim_size() || !shape->dim(index).has_dim_value() ||
      shape->dim(index).dim_value() <= 0)
    return std::nullopt;
  return shape->dim(index).dim_value();
}

const ONNX_NAMESPACE::TensorProto* Constant(const Graph& graph, const NodeArg& value) {
  const auto* tensor = graph_utils::GetConstantInitializer(graph, value.Name());
  if (!tensor || tensor->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) return nullptr;
  return tensor;
}

bool NibbleConstants(const Graph& graph, BindingView unpack) {
  for (const auto& [name, expected] : {std::pair{"mask", uint8_t{15}}, {"shift", uint8_t{4}}}) {
    const auto* tensor = Constant(graph, *unpack.valueOf(name));
    if (!tensor || tensor->dims_size() != 0 ||
        tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8)
      return false;
    Initializer data(graph, *tensor, graph.ModelPath());
    if (data.size() != 1 || data.data<uint8_t>()[0] != expected) return false;
  }
  return true;
}

void BindInt(Match& match, const char* name, int64_t value) {
  ONNX_NAMESPACE::AttributeProto attribute;
  attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attribute.set_i(value);
  match.update(name, std::move(attribute));
}

bool ProviderIs(const Match& match, bool nvfp4) {
  const auto& provider = match.nodes.back()->GetExecutionProviderType();
  return provider.empty() || provider == kCudaExecutionProvider || (!nvfp4 && provider == kCpuExecutionProvider);
}

bool FullShape(const Node& node) {
  // Shape(start/end) is not equivalent to the full Shape used by these bodies.
  const auto& attributes = node.GetAttributes();
  const auto start = attributes.find("start");
  return (start == attributes.end() || start->second.i() == 0) && attributes.count("end") == 0;
}

bool ShapesAreFull(const Match& match) {
  return std::all_of(match.nodes.begin(), match.nodes.end(), [](const Node* node) {
    return node->OpType() != "Shape" || FullShape(*node);
  });
}

bool CheckInt4(const Graph& graph, Match& match) {
  const auto* a = match.valueOf("a");
  const auto* w = match.valueOf("w");
  const auto* s = match.valueOf("s");
  const auto n = PositiveDimension(*w, 0);
  const auto blocks = PositiveDimension(*w, 1);
  const auto bytes = PositiveDimension(*w, 2);
  const auto* a_shape = a->Shape();
  const auto type = match.elementTypeOf("T");
  const int64_t block = match.attributeOf("block").i();
  // FP16 MatMulNBits can remove the source's FP16 dequantized-weight rounding.
  // Use the CPU-supported block range even before provider assignment.
  if (!ProviderIs(match, false) || !ShapesAreFull(match) ||
      type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      !n || !blocks || !bytes || !a_shape || a_shape->dim_size() < 1 ||
      block < 16 || block > 256 || (block & (block - 1)) != 0 || *bytes != block / 2)
    return match.Reject("INT4 requires FP32, full shapes and packed blocks of size 16/32/64/128/256.");

  const auto k = PositiveDimension(*a, a_shape->dim_size() - 1);
  // Division avoids overflowing while also rejecting padded/trimmed logical K.
  if (!k || *k % block != 0 || *blocks != *k / block ||
      !ShapeIs(*s, InlinedVector<int64_t>{*n, *blocks}) ||
      !graph_utils::GetConstantInitializer(graph, w->Name()) ||
      !graph_utils::GetConstantInitializer(graph, s->Name()) ||
      !NibbleConstants(graph, match.instance("unpack")))
    return match.Reject("INT4 requires immutable weights/scales and complete unpadded K blocks.");

  auto zero = match.instance("zero");
  if (zero.alternative() == "PackedZero") {
    const auto body = zero.alternativeView();
    const auto* z = body.valueOf("z");
    if (!ShapeIs(*z, InlinedVector<int64_t>{*n, *blocks / 2 + *blocks % 2}) ||
        !graph_utils::GetConstantInitializer(graph, z->Name()) ||
        !NibbleConstants(graph, body.instance("unpack")))
      return match.Reject("INT4 zero points must be low-nibble-first packed UINT8 per block.");
  } else {
    ORT_ENFORCE(zero.alternative() == "MidpointZero");
    const auto& tensor = zero.alternativeView().attributeOf("midpoint").t();
    if (tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
        tensor.dims_size() != 1 || tensor.dims(0) != 1 ||
        tensor.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL)
      return match.Reject("INT4 default zero point must be UINT8 [8].");
    uint8_t midpoint;
    const auto& raw = tensor.raw_data();
    ORT_THROW_IF_ERROR(utils::UnpackTensor<uint8_t>(
        tensor, utils::HasRawData(tensor) ? raw.data() : nullptr, raw.size(), &midpoint, 1));
    if (midpoint != 8)
      return match.Reject("INT4 default zero point must be midpoint 8.");
  }
  if (match.valueOf("dequant")->TypeAsProto()->tensor_type().elem_type() != *type)
    return match.Reject("INT4 dequantization must preserve the scale dtype.");
  BindInt(match, "logical_k", *k);
  BindInt(match, "logical_n", *n);
  return true;
}

float DecodeE4M3(int code) {
  const int magnitude = code & 127;
  const int exponent = magnitude >> 3;
  const int mantissa = magnitude & 7;
  const float value = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                                    : std::ldexp(1.0f + static_cast<float>(mantissa) / 8, exponent - 7);
  return code & 128 ? -value : value;
}

bool LookupTables(const Graph& graph, const Match& match) {
  constexpr float fp4_values[] = {0.0f, 0.5f, 1, 1.5f, 2, 3, 4, 6,
                                  -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
  for (const char* name : {"fp4", "fp8"}) {
    const auto* tensor = Constant(graph, *match.valueOf(name));
    if (!tensor || tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) return false;
    Initializer data(graph, *tensor, graph.ModelPath());
    const size_t length = std::string_view(name) == "fp4" ? 16 : 256;
    if (data.size() != length) return false;
    for (size_t i = 0; i < length; ++i) {
      const float actual = data.data<float>()[i];
      if (length == 256 && (i & 127) == 127) {
        if (!std::isnan(actual)) return false;
      } else {
        const float expected = length == 16 ? fp4_values[i] : DecodeE4M3(static_cast<int>(i));
        if (actual != expected || std::signbit(actual) != std::signbit(expected)) return false;
      }
    }
  }
  return true;
}

bool CheckNvFp4(const Graph& graph, Match& match) {
  const auto* a = match.valueOf("a");
  const auto* w = match.valueOf("w");
  const auto* s = match.valueOf("s");
  const auto n = PositiveDimension(*w, 0);
  const auto bytes = PositiveDimension(*w, 1);
  const auto* a_shape = a->Shape();
  if (!ProviderIs(match, true) || !ShapesAreFull(match) || !n || !bytes ||
      !a_shape || a_shape->dim_size() < 1)
    return match.Reject("NVFP4 requires FP16 activations, a CUDA destination and static packed geometry.");
  const auto k = PositiveDimension(*a, a_shape->dim_size() - 1);
  if (!k || *k % 16 != 0 || *bytes != *k / 2 ||
      !ShapeIs(*s, InlinedVector<int64_t>{*n, *k / 16}) ||
      !graph_utils::GetConstantInitializer(graph, w->Name()) ||
      !NibbleConstants(graph, match.instance("unpack")) || !LookupTables(graph, match))
    return match.Reject("NVFP4 requires block16 low-first storage and exact E2M1/E4M3FN decode tables.");

  const auto* scale = Constant(graph, *s);
  const auto* global = Constant(graph, *match.valueOf("g"));
  if (!scale || !global || scale->data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
      global->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      (global->dims_size() != 0 && (global->dims_size() != 1 || global->dims(0) != 1)))
    return match.Reject("NVFP4 requires immutable raw scale bytes and a scalar FP32 global multiplier.");
  Initializer scale_data(graph, *scale, graph.ModelPath());
  Initializer global_data(graph, *global, graph.ModelPath());
  if (global_data.size() != 1 || !std::isfinite(global_data.data<float>()[0]) ||
      global_data.data<float>()[0] <= 0 ||
      std::any_of(scale_data.data<uint8_t>(), scale_data.data<uint8_t>() + scale_data.size(),
                  [](uint8_t code) { return (code & 127) == 127; }))
    return match.Reject("NVFP4 requires finite E4M3 scales and a finite positive global multiplier.");
  return true;
}
#endif  // DISABLE_CONTRIB_OPS
}  // namespace

Rules MakeMobiusMatMulNBitsRules() {
#ifndef DISABLE_CONTRIB_OPS
  const PatternLibrary library(kProjectionPatterns);
  const Replacement packed(R"(
    (a,w,s,zero.z,logical_k,logical_n,block) =>
      com.microsoft.MatMulNBits(a,w,s,zero.z,K=logical_k,N=logical_n,bits=4,block_size=block,accuracy_level=0)
  )");
  const Replacement midpoint(R"(
    (a,w,s,logical_k,logical_n,block) =>
      com.microsoft.MatMulNBits(a,w,s,K=logical_k,N=logical_n,bits=4,block_size=block,accuracy_level=0)
  )");
  auto replacement = [packed, midpoint](const Graph&, const Match& match) {
    return match.instance("zero").alternative() == "PackedZero" ? packed : midpoint;
  };
  return {Rule{"MobiusInt4Projection", library.GetPattern("Int4"), replacement, CheckInt4, {{kOnnxDomain, 24, 24}, {kMSDomain, 1, 1}}}};
#else
  return {};
#endif
}

Rules MakeMobiusNvFp4Rules() {
#ifndef DISABLE_CONTRIB_OPS
  const PatternLibrary library(kProjectionPatterns);
  return {Rule{"MobiusNvFp4Projection", library.GetPattern("NvFp4"), "(a,w,s,g)=>com.microsoft.MatMulBlockQuantizedFp4Weight(a,w,s,g,block_size=16)", CheckNvFp4, {{kOnnxDomain, 24, 24}, {kMSDomain, 1, 1}}}};
#else
  return {};
#endif
}
}  // namespace onnxruntime::rewrite
