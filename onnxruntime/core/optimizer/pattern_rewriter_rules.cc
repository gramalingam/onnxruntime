// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/pattern_rewriter.h"

#include <cmath>
#include <optional>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {
namespace rewrite {
namespace {
Operator Op(const char* name, std::initializer_list<int> versions) {
  return {name, kOnnxDomain, InlinedVector<int>(versions)};
}

ONNX_NAMESPACE::AttributeProto FloatAttribute(const char* name, float value) {
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_name(name);
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  attr.set_f(value);
  return attr;
}

ONNX_NAMESPACE::AttributeProto IntAttribute(const char* name, int64_t value) {
  ONNX_NAMESPACE::AttributeProto attr;
  attr.set_name(name);
  attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr.set_i(value);
  return attr;
}

bool FloatCpu(const Match& match) {
  const auto& provider = match.nodes.back()->GetExecutionProviderType();
  if (!provider.empty() && provider != kCpuExecutionProvider) return false;
  for (const NodeArg* value : match.values) {
    const auto* type = value->TypeAsProto();
    if (!type || !type->has_tensor_type() ||
        type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      return false;
  }
  return true;
}

bool HasMicrosoftDomain(const Graph& graph) {
  auto it = graph.DomainToVersionMap().find(kMSDomain);
  return it != graph.DomainToVersionMap().end() && it->second == 1;
}

std::optional<float> Scalar(const Graph& graph, const NodeArg& value) {
  const auto* tensor = graph_utils::GetConstantInitializer(graph, value.Name());
  if (!tensor || tensor->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      tensor->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL || tensor->dims_size() != 0)
    return std::nullopt;
  Initializer initializer(graph, *tensor, graph.ModelPath());
  if (initializer.size() != 1) return std::nullopt;
  float scalar = *initializer.data<float>();
  if (!std::isfinite(scalar)) return std::nullopt;
  return scalar;
}

bool Integer(const Node& node, const char* name, int64_t value) {
  auto it = node.GetAttributes().find(name);
  return it != node.GetAttributes().end() && it->second.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INT &&
         it->second.i() == value;
}

bool Float(const Node& node, const char* name, float value) {
  auto it = node.GetAttributes().find(name);
  return it != node.GetAttributes().end() && it->second.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT &&
         it->second.f() == value;
}
}  // namespace

Rules MakeExampleRules() {
  Rules rules;
  {
    Pattern p;
    auto x = p.Input(), bias = p.Input(), low = p.Input(), high = p.Input(), divisor = p.Input();
    auto add = p.Op(Op("Add", {7, 13, 14}), {x, bias});
    auto clip = p.Op(Op("Clip", {11, 12, 13}), {add, low, high});
    auto mul = p.Op(Op("Mul", {7, 13, 14}), {clip, x});
    p.Op(Op("Div", {7, 13, 14}), {mul, divisor});
    rules.push_back({"HardSwish", p,
                     [=](const Graph& graph, const Match& m) {
                       auto version = graph.DomainToVersionMap().at(kOnnxDomain);
                       return version >= 14 && version < 22 && FloatCpu(m) && Scalar(graph, *m.values[bias]) == 3.0f &&
                              Scalar(graph, *m.values[low]) == 0.0f && Scalar(graph, *m.values[high]) == 6.0f &&
                              Scalar(graph, *m.values[divisor]) == 6.0f;
                     },
                     [=](const Graph&, const Match&) { return Replacement{"HardSwish", kOnnxDomain, 14, {x}, {}}; }});
  }
  {
    Pattern p;
    auto x = p.Input();
    auto sigmoid = p.Op(Op("HardSigmoid", {6}), {x});
    p.Op(Op("Mul", {7, 13, 14}), {sigmoid, x});
    rules.push_back({"HardSigmoidMul", p,
                     [](const Graph& graph, const Match& m) {
                       auto version = graph.DomainToVersionMap().at(kOnnxDomain);
                       return version >= 14 && version < 22 && FloatCpu(m) &&
                              Float(*m.nodes[0], "alpha", 1.0f / 6.0f) && Float(*m.nodes[0], "beta", 0.5f);
                     },
                     [=](const Graph&, const Match&) { return Replacement{"HardSwish", kOnnxDomain, 14, {x}, {}}; }});
  }
#ifndef DISABLE_CONTRIB_OPS
  {
    Pattern p;
    auto x = p.Input(), y = p.Input(), divisor = p.Input();
    auto mm = p.Op(Op("MatMul", {9, 13}), {x, y});
    p.Op(Op("Div", {7, 13, 14}), {mm, divisor});
    rules.push_back({"MatMulDiv", p,
                     [=](const Graph& graph, const Match& m) {
                       auto scalar = Scalar(graph, *m.values[divisor]);
                       return HasMicrosoftDomain(graph) && FloatCpu(m) && scalar && *scalar != 0.0f &&
                              std::isfinite(1.0f / *scalar);
                     },
                     [=](const Graph& graph, const Match& m) {
                       NodeAttributes attributes;
                       attributes.emplace("alpha", FloatAttribute("alpha", 1.0f / *Scalar(graph, *m.values[divisor])));
                       return Replacement{"FusedMatMul", kMSDomain, 1, {x, y}, std::move(attributes)};
                     }});
  }
  {
    Pattern p;
    auto x = p.Input(), y = p.Input();
    auto transpose = p.Op(Op("Transpose", {1, 13}), {y});
    p.Op(Op("MatMul", {9, 13}), {x, transpose});
    rules.push_back({"TransposeMatMul", p,
                     [=](const Graph& graph, const Match& m) {
                       if (!HasMicrosoftDomain(graph) || !FloatCpu(m)) return false;
                       const auto* shape = m.values[y]->Shape();
                       if (!shape || shape->dim_size() < 2) return false;
                       auto it = m.nodes[0]->GetAttributes().find("perm");
                       if (it == m.nodes[0]->GetAttributes().end()) return shape->dim_size() == 2;
                       const auto& perm = it->second;
                       if (perm.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INTS ||
                           perm.ints_size() != shape->dim_size())
                         return false;
                       int rank = shape->dim_size();
                       for (int i = 0; i < rank; ++i) {
                         int expected = i < rank - 2 ? i : (i == rank - 2 ? rank - 1 : rank - 2);
                         if (perm.ints(i) != expected) return false;
                       }
                       return true;
                     },
                     [=](const Graph&, const Match&) {
                       return Replacement{"FusedMatMul", kMSDomain, 1, {x, y}, {{"transB", IntAttribute("transB", 1)}}};
                     }});
  }
  {
    Pattern p;
    auto x = p.Input(), y = p.Input(), divisor = p.Input();
    auto mm = p.Op({"FusedMatMul", kMSDomain, {1}}, {x, y});
    p.Op(Op("Div", {7, 13, 14}), {mm, divisor});
    rules.push_back({"FusedMatMulDiv", p,
                     [=](const Graph& graph, const Match& m) {
                       auto scalar = Scalar(graph, *m.values[divisor]);
                       auto it = m.nodes[0]->GetAttributes().find("alpha");
                       float alpha = it == m.nodes[0]->GetAttributes().end() ? 1.0f : it->second.f();
                       return HasMicrosoftDomain(graph) && FloatCpu(m) && scalar && *scalar != 0.0f &&
                              std::isfinite(alpha / *scalar);
                     },
                     [=](const Graph& graph, const Match& m) {
                       auto attributes = m.nodes[0]->GetAttributes();
                       auto it = attributes.find("alpha");
                       float alpha = it == attributes.end() ? 1.0f : it->second.f();
                       attributes["alpha"] = FloatAttribute("alpha", alpha / *Scalar(graph, *m.values[divisor]));
                       return Replacement{"FusedMatMul", kMSDomain, 1, {x, y}, std::move(attributes)};
                     }});
  }
  {
    Pattern p;
    auto x = p.Input(), scale = p.Input(), exponent = p.Input(), epsilon = p.Input();
    auto square = p.Op(Op("Pow", {7, 12, 13, 15}), {x, exponent});
    auto mean = p.Op(Op("ReduceMean", {1, 13}), {square});
    auto add = p.Op(Op("Add", {7, 13, 14}), {mean, epsilon});
    auto sqrt = p.Op(Op("Sqrt", {6, 13}), {add});
    auto reciprocal = p.Op(Op("Reciprocal", {6, 13}), {sqrt});
    auto normalize = p.Op(Op("Mul", {7, 13, 14}), {x, reciprocal});
    p.Op(Op("Mul", {7, 13, 14}), {normalize, scale});
    rules.push_back(
        {"RmsNormalization", p,
         [=](const Graph& graph, const Match& m) {
           if (!FloatCpu(m) || Scalar(graph, *m.values[exponent]) != 2.0f) return false;
           auto eps = Scalar(graph, *m.values[epsilon]);
           const auto* shape = m.values[x]->Shape();
           const auto* scale_shape = m.values[scale]->Shape();
           if (!eps || *eps <= 0 || !shape || shape->dim_size() < 1 || !scale_shape || scale_shape->dim_size() != 1)
             return false;
           const auto& dim = shape->dim(shape->dim_size() - 1);
           const auto& scale_dim = scale_shape->dim(0);
           if (!dim.has_dim_value() || !scale_dim.has_dim_value() || dim.dim_value() <= 0 ||
               dim.dim_value() != scale_dim.dim_value())
             return false;
           const auto& mean_node = *m.nodes[1];
           auto axes = mean_node.GetAttributes().find("axes");
           return Integer(mean_node, "keepdims", 1) && axes != mean_node.GetAttributes().end() &&
                  axes->second.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INTS &&
                  axes->second.ints_size() == 1 && axes->second.ints(0) == -1;
         },
         [=](const Graph& graph, const Match& m) {
           NodeAttributes attributes;
           attributes.emplace("axis", IntAttribute("axis", -1));
           attributes.emplace("epsilon", FloatAttribute("epsilon", *Scalar(graph, *m.values[epsilon])));
           attributes.emplace("stash_type", IntAttribute("stash_type", ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
           return Replacement{"SimplifiedLayerNormalization", kOnnxDomain, 1, {x, scale}, std::move(attributes)};
         }});
  }
#endif
  return rules;
}

}  // namespace rewrite
}  // namespace onnxruntime
