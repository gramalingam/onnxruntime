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
    selected, indices = SoftmaxTopK(logits, top_k)
    sum = ReduceSum(selected, [-1], keepdims=1, noop_with_empty_axes=0)
    weights = Div(selected, sum)
    return weights, indices
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
    weights, indices = RoutingForms(logits, top_k)
    experts = GroupedExperts(x, indices, gate, up, down, flat_shape, output_shape)
    expanded = Unsqueeze(weights, [-1])
    weighted = Mul(experts.output, expanded)
    return ReduceSum(weighted, [1], keepdims=0, noop_with_empty_axes=0)
  }

  pattern DenseExpert(x:float[M,H], weights:float[M,K], indices:int64[M,K],
      expert:const int64, gate:const float[I,H], up:const float[I,H], down:const float[H,I]) {
    gt = Transpose(gate, perm=[1,0])
    ut = Transpose(up, perm=[1,0])
    dt = Transpose(down, perm=[1,0])
    g = MatMul(x, gt)
    u = MatMul(x, ut)
    activation = SwiGLU(g,u)
    projected = MatMul(activation.product, dt)
    mask = Equal(indices, expert)
    float_mask = CastLike(mask, weights)
    masked_weights = Mul(weights, float_mask)
    weight = ReduceSum(masked_weights, [-1], keepdims=1, noop_with_empty_axes=0)
    output = Mul(projected, weight)
    return output
  }

  pattern DenseMoEPattern(x:float[M,H], logits:float[M,E], top_k:const int64[1],
      expert0:const int64, expert1:const int64, gate0, up0, down0, gate1, up1, down1) {
    weights, indices = TopKSoftmax(logits, top_k)
    e0 = DenseExpert(x, weights, indices, expert0, gate0, up0, down0)
    e1 = DenseExpert(x, weights, indices, expert1, gate1, up1, down1)
    return Add(e0.output, e1.output)
  }
)";

struct RoutingView {
  const NodeArg* logits;
  const NodeArg* indices;
  const NodeArg* top_k;
  bool normalize;

  explicit RoutingView(const Graph& graph, const Match& match)
      : logits(match.valueOf("logits")), indices(match.valueOf("indices")), top_k(match.valueOf("top_k")) {
    const auto* producer = graph.GetProducerNode(match.valueOf("weights")->Name());
    ORT_ENFORCE(producer && producer->Domain() == kOnnxDomain &&
                    (producer->OpType() == "TopK" || producer->OpType() == "Softmax" || producer->OpType() == "Div"),
                "Unexpected matched routing weights producer.");
    normalize = producer->OpType() != "TopK";
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
  const RoutingView route(graph, match);
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

bool CheckDenseMoE(const Graph& graph, Match& match) {
  const auto first = match.instance("e0");
  const auto second = match.instance("e1");
  const auto* gate_shape = first.valueOf("gate")->Shape();
  if (!gate_shape || gate_shape->dim_size() != 2 ||
      !gate_shape->dim(0).has_dim_value() || !gate_shape->dim(1).has_dim_value())
    return match.Reject("Dense MoE requires static expert dimensions.");
  const int64_t intermediate = gate_shape->dim(0).dim_value();
  const int64_t hidden = gate_shape->dim(1).dim_value();
  if (intermediate <= 0 || hidden <= 0)
    return match.Reject("Dense MoE requires positive expert dimensions.");
  for (const auto& expert : {first, second}) {
    if (!HasDimensions(expert.valueOf("gate"), std::array<int64_t, 2>{intermediate, hidden}) ||
        !HasDimensions(expert.valueOf("up"), std::array<int64_t, 2>{intermediate, hidden}) ||
        !HasDimensions(expert.valueOf("down"), std::array<int64_t, 2>{hidden, intermediate}))
      return match.Reject("Dense MoE requires uniform gate/up/down geometry.");
  }
  const auto* x = match.valueOf("x");
  const auto* logits = match.valueOf("logits");
  const auto* indices = match.valueOf("indices");
  if (!HasDimensions(x, std::array<int64_t, 1>{hidden}, 1) ||
      !HasDimensions(logits, std::array<int64_t, 1>{2}, 1) ||
      !HasDimensions(indices, std::array<int64_t, 1>{1}, 1) ||
      !SameTokens(x, logits) || !SameTokens(x, indices) ||
      !HasInt64Values(graph, match.valueOf("top_k"), std::array<int64_t, 1>{1}) ||
      !HasInt64Values(graph, match.valueOf("expert0"), std::array<int64_t, 1>{0}) ||
      !HasInt64Values(graph, match.valueOf("expert1"), std::array<int64_t, 1>{1}))
    return match.Reject("Dense MoE supports exactly two experts and top-k=1 with proven token dimensions.");

  // TODO: Investigate dense inactive-expert NaN/Inf propagation versus MoE skipping.
  // This opt-in rule assumes finite intermediates and tolerance-based equivalence.
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
  const Replacement dense_recipe(R"(
    (x, logits, top_k, e0.gate, e0.up, e0.down, e1.gate, e1.up, e1.down) => {
      selected, indices = TopK(logits, top_k, axis=-1, largest=1)
      mask = Equal(indices, [0,1])
      router = Cast(mask, to=1)
      g0 = Unsqueeze(e0.gate, [1])
      u0 = Unsqueeze(e0.up, [1])
      g1 = Unsqueeze(e1.gate, [1])
      u1 = Unsqueeze(e1.up, [1])
      expert0 = Concat(g0, u0, axis=1)
      expert1 = Concat(g1, u1, axis=1)
      expanded0 = Unsqueeze(expert0, [0])
      expanded1 = Unsqueeze(expert1, [0])
      stacked = Concat(expanded0, expanded1, axis=0)
      shape = Shape(e0.gate)
      doubled = Mul(shape, [2,1])
      packed_shape = Concat([2], doubled, axis=0)
      fc1 = Reshape(stacked, packed_shape)
      down0 = Unsqueeze(e0.down, [0])
      down1 = Unsqueeze(e1.down, [0])
      fc2 = Concat(down0, down1, axis=0)
      return com.microsoft.MoE(x, router, fc1, None, fc2,
        activation_type="swiglu", swiglu_fusion=1, k=1, normalize_routing_weights=1)
    }
  )");
  rules.push_back(Rule{"DenseTwoExpertMoE", library.GetPattern("DenseMoEPattern"), dense_recipe, CheckDenseMoE, {{kOnnxDomain, 24, 24}, {kMSDomain, 1, 1}}});
#endif
  return rules;
}
}  // namespace onnxruntime::rewrite
