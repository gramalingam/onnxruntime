// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/pattern_rewriter.h"

#include <algorithm>
#include <array>

namespace onnxruntime::rewrite {
namespace {
#ifndef DISABLE_CONTRIB_OPS
constexpr const char* kMoELibrary = R"(
  interface Routing(logits:float[M,E], top_k:const int64[1]) -> (weights:float[M,K], indices:int64[M,K])

  pattern TopKSoftmax implements Routing {
    selected, ids = TopK(logits, top_k, axis=-1, largest=1)
    probabilities = Softmax(selected, axis=-1)
    return (weights=probabilities, indices=ids)
  }

  pattern SoftmaxTopK implements Routing {
    probabilities = Softmax(logits, axis=-1)
    selected, ids = TopK(probabilities, top_k, axis=-1, largest=1)
    return (weights=selected, indices=ids)
  }

  pattern NormalizedSoftmaxTopK implements Routing {
    selected = SoftmaxTopK(logits, top_k)
    sum = ReduceSum(selected.weights, [-1], keepdims=1, noop_with_empty_axes=0)
    weights = Div(selected.weights, sum)
    return (weights=weights, indices=selected.indices)
  }

  pattern RoutingForms = TopKSoftmax | SoftmaxTopK | NormalizedSoftmaxTopK

  pattern SwiGLU(g,u) {
    activated = Swish(g, alpha=1.0)
    product = Mul(activated,u)
    return product
  }

  interface Experts(x:float[M,H], indices:int64[M,K],
      gate:float[E,H,I], up:float[E,H,I], down:float[E,I,H],
      flat_shape:const int64[2], output_shape:const int64[3]) -> (output:float[M,K,H])

  pattern GroupedExperts implements Experts {
    g = local.moe.GroupedMatMul(x, gate, indices)
    u = local.moe.GroupedMatMul(x, up, indices)
    activation = SwiGLU(g,u)
    flat = Reshape(activation.product, flat_shape, allowzero=0)
    flat_indices = Reshape(indices, [-1,1], allowzero=0)
    projected = local.moe.GroupedMatMul(flat, down, flat_indices)
    output = Reshape(projected, output_shape, allowzero=0)
    return output
  }

  pattern MoEPattern(x:float[M,H], logits:float[M,E], top_k:const int64[1],
      gate:float[E,H,I], up:float[E,H,I], down:float[E,I,H],
      flat_shape:const int64[2], output_shape:const int64[3]) {
    route = RoutingForms(logits, top_k)
    experts = GroupedExperts(x, route.indices, gate, up, down, flat_shape, output_shape)
    expanded = Unsqueeze(route.weights, [-1])
    weighted = Mul(experts.output, expanded)
    return ReduceSum(weighted, [1], keepdims=0, noop_with_empty_axes=0)
  }
)";

struct RoutingView {
  const NodeArg* logits;
  const NodeArg* indices;
  const NodeArg* top_k;
  bool normalize;

  explicit RoutingView(BindingView view)
      : logits(view.valueOf("logits")), indices(view.valueOf("indices")), top_k(view.valueOf("top_k")), normalize(view.alternative() != "SoftmaxTopK") {
    ORT_ENFORCE(view.interfaceName() == "Routing");
    const auto member = view.alternative();
    ORT_ENFORCE(member == "TopKSoftmax" || member == "SoftmaxTopK" || member == "NormalizedSoftmaxTopK");
  }
};

struct ExpertsView {
  const NodeArg* x;
  const NodeArg* gate;
  const NodeArg* up;
  const NodeArg* down;
  const NodeArg* flat_shape;
  const NodeArg* output_shape;

  explicit ExpertsView(BindingView view)
      : x(view.valueOf("x")), gate(view.valueOf("gate")), up(view.valueOf("up")), down(view.valueOf("down")), flat_shape(view.valueOf("flat_shape")), output_shape(view.valueOf("output_shape")) {
    ORT_ENFORCE(view.interfaceName() == "Experts");
  }
};

bool HasInt64Values(const Graph& graph, const NodeArg* value, gsl::span<const int64_t> expected) {
  InlinedVector<int64_t> data;
  bool available = false;
  ORT_THROW_IF_ERROR(ReadConstantTensor(graph, value, data, available));
  return available && data.size() == expected.size() && std::equal(data.begin(), data.end(), expected.begin());
}

bool HasDimensions(const NodeArg* value, gsl::span<const int64_t> expected, int offset = 0) {
  const auto* shape = value->Shape();
  if (!shape || shape->dim_size() != offset + static_cast<int>(expected.size())) return false;
  for (int i = 0; i < static_cast<int>(expected.size()); ++i) {
    const auto& dim = shape->dim(offset + i);
    if (!dim.has_dim_value() || dim.dim_value() != expected[i]) return false;
  }
  return true;
}

bool SameTokens(const NodeArg* x, const NodeArg* y) {
  const auto* x_shape = x->Shape();
  const auto* y_shape = y->Shape();
  if (!x_shape || !y_shape || x_shape->dim_size() != 2 || y_shape->dim_size() != 2) return false;
  const auto& a = x_shape->dim(0);
  const auto& b = y_shape->dim(0);
  if (a.has_dim_value()) return a.dim_value() > 0 && b.has_dim_value() && a.dim_value() == b.dim_value();
  return a.has_dim_param() && !a.dim_param().empty() && b.has_dim_param() && a.dim_param() == b.dim_param();
}

void BindInteger(Match& match, const char* name, int64_t value) {
  ONNX_NAMESPACE::AttributeProto attribute;
  attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attribute.set_i(value);
  match.update(name, std::move(attribute));
}

bool CheckMoE(const Graph& graph, Match& match) {
  const RoutingView route(match.instance("route"));
  const ExpertsView experts(match.instance("experts"));
  const auto* gate = experts.gate->Shape();
  std::array<int64_t, 3> dims;
  for (int i = 0; i < 3; ++i) {
    if (!gate->dim(i).has_dim_value() || gate->dim(i).dim_value() <= 0)
      return match.Reject("MoE expert dimensions must be positive and static.");
    dims[i] = gate->dim(i).dim_value();
  }
  const auto [e, h, intermediate] = dims;
  // Annotations unify known dimensions, but anonymous unknowns are not equality evidence.
  if (!HasDimensions(experts.up, dims) ||
      !HasDimensions(experts.down, std::array<int64_t, 3>{e, intermediate, h}) ||
      !HasDimensions(experts.x, std::array<int64_t, 1>{h}, 1) ||
      !HasDimensions(route.logits, std::array<int64_t, 1>{e}, 1) ||
      !SameTokens(experts.x, route.logits) || !SameTokens(experts.x, route.indices))
    return match.Reject("MoE requires proven common token and expert dimensions.");
  const auto& k_dim = route.indices->Shape()->dim(1);
  if (!k_dim.has_dim_value()) return match.Reject("MoE requires a static top-k.");
  const int64_t k = k_dim.dim_value();
  if (k < 1 || k > e ||
      !HasInt64Values(graph, route.top_k, std::array<int64_t, 1>{k}) ||
      !HasInt64Values(graph, experts.flat_shape, std::array<int64_t, 2>{-1, intermediate}) ||
      !HasInt64Values(graph, experts.output_shape, std::array<int64_t, 3>{-1, k, h}))
    return match.Reject("MoE requires canonical token/slot-preserving shapes and constant top-k.");
  BindInteger(match, "k", k);
  BindInteger(match, "normalize", route.normalize ? 1 : 0);
  return true;
}
#endif
}  // namespace

Rules MakeMoEFusionRules() {
  Rules rules;
#ifndef DISABLE_CONTRIB_OPS
  const PatternLibrary library(kMoELibrary);
  // [E,H,I] -> [E,I,2,H] -> [E,2*I,H] interleaves gate/up rows for CPU SwiGLU.
  const Replacement recipe(R"(
    (x, logits, experts.gate, experts.up, experts.down, k, normalize) => {
      gt = Transpose(experts.gate, perm=[0,2,1])
      ut = Transpose(experts.up, perm=[0,2,1])
      g = Unsqueeze(gt, [2])
      u = Unsqueeze(ut, [2])
      interleaved = Concat(g, u, axis=2)
      shape = Shape(gt)
      packed_shape = Mul(shape, [1,2,1])
      fc1 = Reshape(interleaved, packed_shape)
      fc2 = Transpose(experts.down, perm=[0,2,1])
      return com.microsoft.MoE(x, logits, fc1, None, fc2,
        activation_type="swiglu", swiglu_fusion=1, k=k, normalize_routing_weights=normalize)
    }
  )");
  rules.push_back(Rule{"GroupedMoE", library.GetPattern("MoEPattern"), recipe, CheckMoE, {{kOnnxDomain, 24, 24}, {"local.moe", 1, 1}, {kMSDomain, 1, 1}}});
#endif
  return rules;
}
}  // namespace onnxruntime::rewrite
