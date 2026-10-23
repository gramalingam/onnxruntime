// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/pattern_rewriter.h"

#include <cmath>
#include <limits>

namespace onnxruntime {
namespace rewrite {
#ifndef DISABLE_CONTRIB_OPS
namespace {

// Source: onnxruntime/mobius@d69efe03a9f4c1ada744bc31dd01077a710b3c39,
// functions/skip_layer_normalization.py and rewrite_rules/_bias_gelu.py.
constexpr const char* kPatterns = R"(
  pattern residual(x:float[], skip:float[]) {
    sum = Add(x, skip)
    return sum
  }
  pattern layer(x:float[], scale:float[], beta:float[] | None) {
    y = LayerNormalization(x, scale, beta, axis=axis, epsilon=epsilon, stash_type=stash)
    return y
  }
  pattern layer_stats(x:float[], scale:float[], beta:float[] | None) {
    y, mean, inv = LayerNormalization(x, scale, beta, axis=axis, epsilon=epsilon, stash_type=stash)
    return y, mean, inv
  }
  pattern rms(x:float[], scale:float[]) {
    y = RMSNormalization(x, scale, axis=axis, epsilon=epsilon, stash_type=stash)
    return y
  }
  pattern SkipLayer(x:float[], skip:float[], scale:float[], beta:float[] | None) {
    add = residual(x, skip)
    norm = layer(add.sum, scale, beta)
    return norm.y, add.sum
  }
  pattern SkipLayerStats(x:float[], skip:float[], scale:float[], beta:float[] | None) {
    add = residual(x, skip)
    norm = layer_stats(add.sum, scale, beta)
    return norm.y, norm.mean, norm.inv, add.sum
  }
  pattern SkipRms(x:float[], skip:float[], scale:float[]) {
    add = residual(x, skip)
    norm = rms(add.sum, scale)
    return norm.y, add.sum
  }
  interface BiasAdd(x:float[], bias:float[]) -> (sum:float[])
  pattern bias_last implements BiasAdd {
    sum = Add(x, bias)
    return sum
  }
  pattern bias_first implements BiasAdd {
    sum = Add(bias, x)
    return sum
  }
  pattern BiasForms = bias_last | bias_first
  pattern BiasGelu(x:float[], bias:float[]) {
    add = BiasForms(x, bias)
    y = Gelu(add.sum, approximate=approximate)
    return y
  }
)";

struct ResidualView {
  const NodeArg* x;
  const NodeArg* skip;
  explicit ResidualView(BindingView view) : x(view.valueOf("x")), skip(view.valueOf("skip")) {}
};

struct NormalizationView {
  const NodeArg* scale;
  const ONNX_NAMESPACE::AttributeProto& axis;
  const ONNX_NAMESPACE::AttributeProto& epsilon;
  const ONNX_NAMESPACE::AttributeProto& stash;
  explicit NormalizationView(BindingView view)
      : scale(view.valueOf("scale")),
        axis(view.attributeOf("axis")),
        epsilon(view.attributeOf("epsilon")),
        stash(view.attributeOf("stash")) {}
};

struct BiasView {
  const NodeArg* x;
  const NodeArg* bias;
  explicit BiasView(BindingView view) : x(view.valueOf("x")), bias(view.valueOf("bias")) {
    ORT_ENFORCE(view.interfaceName() == "BiasAdd");
  }
};

bool CpuOnly(const Match& match) {
  for (const Node* node : match.nodes) {
    const auto& provider = node->GetExecutionProviderType();
    if (!provider.empty() && provider != kCpuExecutionProvider) return false;
  }
  return true;
}

bool SameDimension(const ONNX_NAMESPACE::TensorShapeProto_Dimension& a,
                   const ONNX_NAMESPACE::TensorShapeProto_Dimension& b) {
  if (a.has_dim_value() && b.has_dim_value()) return a.dim_value() > 0 && a.dim_value() == b.dim_value();
  return a.has_dim_param() && !a.dim_param().empty() && b.has_dim_param() && a.dim_param() == b.dim_param();
}

bool VectorOfWidth(const NodeArg* value, int64_t width) {
  const auto* shape = value ? value->Shape() : nullptr;
  return shape && shape->dim_size() == 1 && shape->dim(0).has_dim_value() && shape->dim(0).dim_value() == width;
}

bool CheckNormalization(const Graph& graph, Match& match, bool simplified) {
  // CUDA's skip kernel does not produce statistics; do not leave their placement open.
  const auto& node = *match.nodes.back();
  if (!simplified && node.OutputDefs().size() == 3 && node.GetExecutionProviderType() != kCpuExecutionProvider) {
    for (const NodeArg* output : graph.GetOutputs()) {
      if (output == node.OutputDefs()[1] || output == node.OutputDefs()[2])
        return match.Reject("Live LayerNormalization statistics require explicit CPU assignment.");
    }
    for (auto edge = node.OutputEdgesBegin(); edge != node.OutputEdgesEnd(); ++edge) {
      if (edge->GetSrcArgIndex() == 1 || edge->GetSrcArgIndex() == 2)
        return match.Reject("Live LayerNormalization statistics require explicit CPU assignment.");
    }
  }
  const ResidualView residual(match.instance("add"));
  const NormalizationView norm(match.instance("norm"));
  const auto* shape = residual.x->Shape();
  const auto* skip_shape = residual.skip->Shape();
  if (!CpuOnly(match) || !shape || !skip_shape || (shape->dim_size() != 2 && shape->dim_size() != 3) ||
      skip_shape->dim_size() != shape->dim_size())
    return match.Reject("Skip normalization requires float32 CPU inputs of known rank 2 or 3.");
  for (int i = 0; i < shape->dim_size(); ++i) {
    if (!SameDimension(shape->dim(i), skip_shape->dim(i)))
      return match.Reject("Residual inputs must have provably equal, nonanonymous dimensions.");
  }
  const auto& width = shape->dim(shape->dim_size() - 1);
  if (!width.has_dim_value() || width.dim_value() <= 0 || width.dim_value() > std::numeric_limits<int>::max() ||
      !VectorOfWidth(norm.scale, width.dim_value()) ||
      (!simplified && match.valueOf("beta") && !VectorOfWidth(match.valueOf("beta"), width.dim_value())))
    return match.Reject("Scale and optional beta must be vectors of the known positive hidden width.");
  if (norm.axis.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INT ||
      (norm.axis.i() != -1 && norm.axis.i() != shape->dim_size() - 1) ||
      norm.stash.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INT ||
      norm.stash.i() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      norm.epsilon.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT ||
      !std::isfinite(norm.epsilon.f()) || norm.epsilon.f() <= 0)
    return match.Reject("Only last-axis float32 accumulation with finite positive epsilon is supported.");
  match.update("fused_epsilon", norm.epsilon);
  return true;
}

bool CheckBiasGelu(const Graph&, Match& match) {
  const BiasView add(match.instance("add"));
  const auto& approximate = match.attributeOf("approximate");
  const auto* shape = add.x->Shape();
  if (!CpuOnly(match) || approximate.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_STRING ||
      approximate.s() != "none")
    return match.Reject("BiasGelu supports only exact Gelu on CPU.");
  if (!shape || shape->dim_size() < 1 || !shape->dim(shape->dim_size() - 1).has_dim_value() ||
      shape->dim(shape->dim_size() - 1).dim_value() <= 0 ||
      !VectorOfWidth(add.bias, shape->dim(shape->dim_size() - 1).dim_value()))
    return match.Reject("Bias must be a 1D tensor matching the known positive last input dimension.");
  return true;
}

}  // namespace
#endif

Rules MakeMobiusNormalizationRules() {
  Rules rules;
#ifndef DISABLE_CONTRIB_OPS
  const PatternLibrary library(kPatterns);
  const Replacement layer(R"(
    (x, skip, scale, beta, fused_epsilon) => {
      y, mean, inv, sum = com.microsoft.SkipLayerNormalization(x, skip, scale, beta, epsilon=fused_epsilon)
      return y, sum
    }
  )");
  const Replacement layer_stats(R"(
    (x, skip, scale, beta, fused_epsilon) => {
      y, mean, inv, sum = com.microsoft.SkipLayerNormalization(x, skip, scale, beta, epsilon=fused_epsilon)
      return y, mean, inv, sum
    }
  )");
  for (bool stats : {false, true}) {
    rules.emplace_back(
        stats ? "MobiusSkipLayerNormalizationStats" : "MobiusSkipLayerNormalization",
        library.GetPattern(stats ? "SkipLayerStats" : "SkipLayer"),
        [replacement = stats ? layer_stats : layer](const Graph&, const Match&) { return replacement; },
        [](const Graph& graph, Match& match) { return CheckNormalization(graph, match, false); },
        InlinedVector<OpsetRequirement, 1>{{kOnnxDomain, 17, 24}, {kMSDomain, 1, 1}});
  }
  const Replacement rms(R"(
    (x, skip, scale, fused_epsilon) => {
      y, mean, inv, sum = com.microsoft.SkipSimplifiedLayerNormalization(x, skip, scale, epsilon=fused_epsilon)
      return y, sum
    }
  )");
  rules.emplace_back(
      "MobiusSkipSimplifiedLayerNormalization", library.GetPattern("SkipRms"),
      [rms](const Graph&, const Match&) { return rms; },
      [](const Graph& graph, Match& match) { return CheckNormalization(graph, match, true); },
      InlinedVector<OpsetRequirement, 1>{{kOnnxDomain, 23, 24}, {kMSDomain, 1, 1}});
  const Replacement gelu("(x, bias) => com.microsoft.BiasGelu(x, bias)");
  rules.emplace_back(
      "MobiusBiasGelu", library.GetPattern("BiasGelu"),
      [gelu](const Graph&, const Match&) { return gelu; }, CheckBiasGelu,
      InlinedVector<OpsetRequirement, 1>{{kOnnxDomain, 20, 24}, {kMSDomain, 1, 1}});
#endif
  return rules;
}

}  // namespace rewrite
}  // namespace onnxruntime
