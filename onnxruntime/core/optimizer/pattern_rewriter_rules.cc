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

bool Float32(const Match& match) {
  for (const NodeArg* value : match.values) {
    const auto* type = value->TypeAsProto();
    if (!type || !type->has_tensor_type() ||
        type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT)
      return false;
  }
  return true;
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
}  // namespace

Rules MakeExampleRules() {
  Rules rules;
  {
    Pattern p(R"(
      (x, bias, low, high, divisor) =>
        add = Add(x, bias)
        clip = Clip(add, low, high)
        mul = Mul(clip, x)
        return Div(mul, divisor)
    )");
    auto x = p.NamedValue("x"), bias = p.NamedValue("bias"), low = p.NamedValue("low"),
         high = p.NamedValue("high"), divisor = p.NamedValue("divisor");
    rules.push_back({"HardSwish",
                     p,
                     {{kOnnxDomain, 14, 21}},
                     [=](const Graph& graph, const Match& m) {
                       return Float32(m) && Scalar(graph, *m.values[bias]) == 3.0f &&
                              Scalar(graph, *m.values[low]) == 0.0f && Scalar(graph, *m.values[high]) == 6.0f &&
                              Scalar(graph, *m.values[divisor]) == 6.0f;
                     },
                     [=](const Graph&, const Match&) { return Replacement{"HardSwish", kOnnxDomain, {x}, {}}; }});
  }
  {
    Pattern p(R"(
      (x) =>
        sigmoid = HardSigmoid(x, alpha=0.16666667, beta=0.5)
        return Mul(sigmoid, x)
    )");
    auto x = p.NamedValue("x");
    rules.push_back({"HardSigmoidMul",
                     p,
                     {{kOnnxDomain, 14, 21}},
                     [](const Graph&, const Match& m) { return Float32(m); },
                     [=](const Graph&, const Match&) { return Replacement{"HardSwish", kOnnxDomain, {x}, {}}; }});
  }
#ifndef DISABLE_CONTRIB_OPS
  {
    Pattern p("(x, y, divisor) => mm = MatMul(x, y); return Div(mm, divisor)");
    auto x = p.NamedValue("x"), y = p.NamedValue("y"), divisor = p.NamedValue("divisor");
    rules.push_back({"MatMulDiv",
                     p,
                     {{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}},
                     [=](const Graph& graph, const Match& m) {
                       auto scalar = Scalar(graph, *m.values[divisor]);
                       return Float32(m) && scalar && *scalar != 0.0f && std::isfinite(1.0f / *scalar);
                     },
                     [=](const Graph& graph, const Match& m) {
                       NodeAttributes attributes;
                       attributes.emplace("alpha", FloatAttribute("alpha", 1.0f / *Scalar(graph, *m.values[divisor])));
                       return Replacement{"FusedMatMul", kMSDomain, {x, y}, std::move(attributes)};
                     }});
  }
  {
    Pattern p("(x, y) => transpose = Transpose(y); return MatMul(x, transpose)");
    auto x = p.NamedValue("x"), y = p.NamedValue("y");
    rules.push_back({"TransposeMatMul",
                     p,
                     {{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}},
                     [=](const Graph&, const Match& m) {
                       if (!Float32(m)) return false;
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
                       return Replacement{"FusedMatMul", kMSDomain, {x, y}, {{"transB", IntAttribute("transB", 1)}}};
                     }});
  }
  {
    Pattern p("(x, y, divisor) => mm = com.microsoft::FusedMatMul(x, y); return Div(mm, divisor)");
    auto x = p.NamedValue("x"), y = p.NamedValue("y"), divisor = p.NamedValue("divisor");
    rules.push_back({"FusedMatMulDiv",
                     p,
                     {{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}},
                     [=](const Graph& graph, const Match& m) {
                       auto scalar = Scalar(graph, *m.values[divisor]);
                       auto it = m.nodes[0]->GetAttributes().find("alpha");
                       float alpha = it == m.nodes[0]->GetAttributes().end() ? 1.0f : it->second.f();
                       return Float32(m) && scalar && *scalar != 0.0f && std::isfinite(alpha / *scalar);
                     },
                     [=](const Graph& graph, const Match& m) {
                       auto attributes = m.nodes[0]->GetAttributes();
                       auto it = attributes.find("alpha");
                       float alpha = it == attributes.end() ? 1.0f : it->second.f();
                       attributes["alpha"] = FloatAttribute("alpha", alpha / *Scalar(graph, *m.values[divisor]));
                       return Replacement{"FusedMatMul", kMSDomain, {x, y}, std::move(attributes)};
                     }});
  }
  {
    Pattern p(R"(
      (x, scale, exponent, epsilon) =>
        square = Pow(x, exponent)
        mean = ReduceMean(square)
        add = Add(mean, epsilon)
        sqrt = Sqrt(add)
        reciprocal = Reciprocal(sqrt)
        normalize = Mul(x, reciprocal)
        return Mul(normalize, scale)
    )");
    auto x = p.NamedValue("x"), scale = p.NamedValue("scale"), exponent = p.NamedValue("exponent"),
         epsilon = p.NamedValue("epsilon");
    rules.push_back(
        {"RmsNormalization",
         p,
         {{kOnnxDomain, 7, 17}},
         [=](const Graph& graph, const Match& m) {
           if (!Float32(m) || Scalar(graph, *m.values[exponent]) != 2.0f) return false;
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
           return Replacement{"SimplifiedLayerNormalization", kOnnxDomain, {x, scale}, std::move(attributes)};
         }});
  }
#endif
  return rules;
}

}  // namespace rewrite
}  // namespace onnxruntime
