// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/pattern_rewriter.h"

#include <algorithm>
#include <array>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime::rewrite {
namespace {
#ifndef DISABLE_CONTRIB_OPS
enum class Routing { TopKSoftmax,
                     SoftmaxTopK,
                     SoftmaxTopKNormalized };

bool HasInt64Values(const Graph& graph, const NodeArg& value, gsl::span<const int64_t> expected) {
  const auto* tensor = graph_utils::GetConstantInitializer(graph, value.Name());
  if (!tensor || tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
      tensor->dims_size() != 1 || tensor->dims(0) != static_cast<int64_t>(expected.size()) ||
      tensor->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL)
    return false;
  Initializer initializer(graph, *tensor, graph.ModelPath());
  return std::equal(expected.begin(), expected.end(), initializer.data<int64_t>());
}

bool CheckMoE(const Graph& graph, Match& match) {
  const auto* gate = match.valueOf("gate")->Shape();
  const auto* indices = match.valueOf("indices")->Shape();
  // Pattern annotations establish ranks and common dimensions; kernel dimensions must be static.
  for (int i = 0; i < 3; ++i) {
    if (!gate->dim(i).has_dim_value() || gate->dim(i).dim_value() <= 0)
      return match.Reject("MoE expert dimensions must be positive and static.");
  }
  if (!indices->dim(1).has_dim_value()) return match.Reject("MoE requires a static top-k.");
  const int64_t experts = gate->dim(0).dim_value();
  const int64_t hidden = gate->dim(1).dim_value();
  const int64_t intermediate = gate->dim(2).dim_value();
  const int64_t k = indices->dim(1).dim_value();
  if (k < 1 || k > experts ||
      !HasInt64Values(graph, *match.valueOf("top_k"), std::array<int64_t, 1>{k}) ||
      !HasInt64Values(graph, *match.valueOf("flat_shape"), std::array<int64_t, 2>{-1, intermediate}) ||
      !HasInt64Values(graph, *match.valueOf("output_shape"), std::array<int64_t, 3>{-1, k, hidden}))
    return match.Reject("MoE requires canonical token/slot-preserving shapes and constant top-k.");
  ONNX_NAMESPACE::AttributeProto attribute;
  attribute.set_name("k");
  attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attribute.set_i(k);
  match.update("k", std::move(attribute));
  return true;
}

Rule MakeMoERule(Routing routing) {
  std::string pattern = R"(
    (x:float[M,H], logits:float[M,E], top_k:int64[1],
     gate:float[E,H,I], up:float[E,H,I], down:float[E,I,H],
     flat_shape:int64[2], output_shape:int64[3]) => {
  )";
  if (routing == Routing::TopKSoftmax) {
    pattern += R"(
      selected, indices:int64[M,K] = TopK(logits, top_k, axis=-1, largest=1)
      weights = Softmax(selected, axis=-1)
    )";
  } else {
    pattern += R"(
      probs = Softmax(logits, axis=-1)
      selected, indices:int64[M,K] = TopK(probs, top_k, axis=-1, largest=1)
    )";
    if (routing == Routing::SoftmaxTopKNormalized) {
      pattern += R"(
        sum = ReduceSum(selected, [-1], keepdims=1, noop_with_empty_axes=0)
        weights = Div(selected, sum)
      )";
    }
  }
  pattern += R"(
      g = local.moe.GroupedMatMul(x, gate, indices)
      u = local.moe.GroupedMatMul(x, up, indices)
      activated = Swish(g, alpha=1.0)
      product = Mul(activated, u)
      flat = Reshape(product, flat_shape, allowzero=0)
      flat_indices = Reshape(indices, [-1,1], allowzero=0)
      projected = local.moe.GroupedMatMul(flat, down, flat_indices)
      regrouped = Reshape(projected, output_shape, allowzero=0)
  )";
  pattern += "expanded = Unsqueeze(";
  pattern += routing == Routing::SoftmaxTopK ? "selected" : "weights";
  pattern += R"(, [-1])
      weighted = Mul(regrouped, expanded)
      return ReduceSum(weighted, [1], keepdims=0, noop_with_empty_axes=0)
    }
  )";
  // [E,H,I] -> [E,I,2,H] -> [E,2*I,H] packs [g0,u0,g1,u1,...] for CPU SwiGLU.
  std::string replacement = R"(
    (x, logits, gate, up, down, k) => {
      gt = Transpose(gate, perm=[0,2,1])
      ut = Transpose(up, perm=[0,2,1])
      g = Unsqueeze(gt, [2])
      u = Unsqueeze(ut, [2])
      interleaved = Concat(g, u, axis=2)
      shape = Shape(gt)
      packed_shape = Mul(shape, [1,2,1])
      fc1 = Reshape(interleaved, packed_shape)
      fc2 = Transpose(down, perm=[0,2,1])
      return com.microsoft.MoE(x, logits, fc1, None, fc2,
        activation_type="swiglu", swiglu_fusion=1, k=k, normalize_routing_weights=
  )";
  replacement += routing == Routing::SoftmaxTopK ? "0" : "1";
  replacement += ") }";
  const char* name = routing == Routing::TopKSoftmax   ? "GroupedMoETopKSoftmax"
                     : routing == Routing::SoftmaxTopK ? "GroupedMoESoftmaxTopK"
                                                       : "GroupedMoESoftmaxTopKNormalized";
  return Rule{name, pattern, replacement, CheckMoE, {{kOnnxDomain, 24, 24}, {"local.moe", 1, 1}, {kMSDomain, 1, 1}}};
}
#endif
}  // namespace

Rules MakeMoEFusionRules() {
  Rules rules;
#ifndef DISABLE_CONTRIB_OPS
  for (auto routing : {Routing::TopKSoftmax, Routing::SoftmaxTopK, Routing::SoftmaxTopKNormalized})
    rules.push_back(MakeMoERule(routing));
#endif
  return rules;
}
}  // namespace onnxruntime::rewrite
