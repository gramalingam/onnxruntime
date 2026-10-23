// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/mobius_stateful_fusion.h"

#include <algorithm>
#include <limits>

namespace onnxruntime::rewrite {
#ifndef DISABLE_CONTRIB_OPS
namespace {

const char* ConvLibrary = R"(
  interface History(data:float[B,C,L]) -> (present:float[B,C,P])
  pattern fixed implements History:
      (starts:int64[1], ends:int64[1], steps:int64[1] | None) -> {
    result = Slice(data,starts,ends,[2],steps)
    return (present=result)
  }
  pattern computed implements History: (width:int64) -> {
    shape = Shape(data)
    length = Gather(shape,2,axis=0)
    start = Sub(length,width)
    starts = Reshape(start,[1])
    ends = Reshape(length,[1])
    result = Slice(data,starts,ends,[2])
    return (present=result)
  }
  pattern Histories = fixed | computed
  interface StatefulConv(x:float[B,C,T], w:float[C,1,K], bias:float[C], past:float[B,C,P])
      -> (y:float[B,C,T], present:float[B,C,P])
  pattern conv_bias implements StatefulConv {
    padded = Concat(past,x,axis=2)
    history = Histories(padded)
    conv = Conv(padded,w)
    reshaped_bias = Reshape(bias,[1,-1,1])
    biased = Add(conv,reshaped_bias)
    return (y=biased,present=history.present)
  }
  pattern Plain(x,w,bias,past) {
    body = conv_bias(x,w,bias,past)
    return body.y,body.present
  }
  pattern Silu(x,w,bias,past) {
    body = conv_bias(x,w,bias,past)
    sigmoid = Sigmoid(body.y)
    output = Mul(body.y,sigmoid)
    return output,body.present
  }
  pattern SwishForm(x,w,bias,past) {
    body = conv_bias(x,w,bias,past)
    output = Swish(body.y)
    return output,body.present
  }
)";

struct ConvView {
  BindingView body;
  BindingView history;
  const NodeArg* x;
  const NodeArg* weight;
  const NodeArg* bias;
  const NodeArg* past;
  explicit ConvView(const Match& match)
      : body(match.instance("body")),
        history(body.instance("history").alternativeView()),
        x(body.valueOf("x")),
        weight(body.valueOf("w")),
        bias(body.valueOf("bias")),
        past(body.valueOf("past")) {}
};

bool Shape(const NodeArg* value, InlinedVector<int64_t>& dimensions) {
  const auto* shape = value->Shape();
  if (!shape) return false;
  for (const auto& dim : shape->dim()) {
    if (!dim.has_dim_value() || dim.dim_value() < 0) return false;
    dimensions.push_back(dim.dim_value());
  }
  return true;
}

const Node& Producer(const Match& match, const NodeArg* output) {
  for (const auto* node : match.nodes) {
    for (const auto* value : node->OutputDefs()) {
      if (value == output) return *node;
    }
  }
  ORT_THROW("Missing matched producer for ", output->Name());
}

bool IntAttribute(const Node& node, const char* name, int64_t expected, int64_t default_value) {
  const auto it = node.GetAttributes().find(name);
  return it == node.GetAttributes().end()
             ? expected == default_value
             : it->second.type() == ONNX_NAMESPACE::AttributeProto::INT && it->second.i() == expected;
}

bool IntsAttribute(const Node& node, const char* name, gsl::span<const int64_t> expected) {
  const auto it = node.GetAttributes().find(name);
  if (it == node.GetAttributes().end()) return true;
  const auto& attr = it->second;
  return attr.type() == ONNX_NAMESPACE::AttributeProto::INTS &&
         static_cast<size_t>(attr.ints_size()) == expected.size() &&
         std::equal(expected.begin(), expected.end(), attr.ints().begin());
}

bool Constant(const Graph& graph, const NodeArg* value, int64_t expected) {
  InlinedVector<int64_t> data;
  bool available = false;
  ORT_THROW_IF_ERROR(ReadConstantTensor(graph, value, data, available));
  return available && data.size() == 1 && data[0] == expected;
}

bool CheckConv(const Graph& graph, Match& match) {
  const ConvView view(match);
  InlinedVector<int64_t> x, weight, bias, past;
  if (!Shape(view.x, x) || !Shape(view.weight, weight) || !Shape(view.bias, bias) || !Shape(view.past, past) ||
      x.size() != 3 || weight.size() != 3 || bias.size() != 1 || past.size() != 3 ||
      x[0] <= 0 || x[1] <= 0 || x[2] <= 0 || weight[2] <= 0 ||
      weight[0] != x[1] || weight[1] != 1 || bias[0] != x[1] ||
      past[0] != x[0] || past[1] != x[1] || past[2] != weight[2] - 1 ||
      x[2] > std::numeric_limits<int64_t>::max() - past[2])
    return match.Reject("Expected fixed positive B,C,T,K and depthwise weights with K-1 history.");

  const auto& conv = Producer(match, view.body.valueOf("conv"));
  const InlinedVector<int64_t> kernel{weight[2]}, ones{1}, zeros{0, 0};
  if (!IntAttribute(conv, "group", x[1], 1) || !IntsAttribute(conv, "kernel_shape", kernel) ||
      !IntsAttribute(conv, "strides", ones) || !IntsAttribute(conv, "dilations", ones) ||
      !IntsAttribute(conv, "pads", zeros))
    return match.Reject("Expected valid, unit-stride, undilated depthwise Conv.");
  const auto auto_pad = conv.GetAttributes().find("auto_pad");
  if (auto_pad != conv.GetAttributes().end() &&
      (auto_pad->second.type() != ONNX_NAMESPACE::AttributeProto::STRING ||
       auto_pad->second.s() != "NOTSET"))
    return match.Reject("Conv auto_pad must be NOTSET.");
  const auto& bias_reshape = Producer(match, view.body.valueOf("reshaped_bias"));
  if (!IntAttribute(bias_reshape, "allowzero", 0, 0))
    return match.Reject("Bias Reshape must use ordinary dimension semantics.");

  if (view.history.definition() == "computed") {
    if (!Constant(graph, view.history.valueOf("width"), past[2]))
      return match.Reject("Computed Slice must retain exactly K-1 positions.");
    const auto& shape = Producer(match, view.history.valueOf("shape"));
    const auto end = shape.GetAttributes().find("end");
    if (!IntAttribute(shape, "start", 0, 0) ||
        (end != shape.GetAttributes().end() &&
         (end->second.type() != ONNX_NAMESPACE::AttributeProto::INT || end->second.i() < 3)))
      return match.Reject("History Shape must include all three dimensions.");
  } else {
    InlinedVector<int64_t> starts, ends;
    bool available = false;
    ORT_THROW_IF_ERROR(ReadConstantTensor(graph, view.history.valueOf("starts"), starts, available));
    if (!available || starts.size() != 1) return match.Reject("Slice start must be immutable.");
    ORT_THROW_IF_ERROR(ReadConstantTensor(graph, view.history.valueOf("ends"), ends, available));
    if (!available || ends.size() != 1) return match.Reject("Slice end must be immutable.");
    const auto* steps = view.history.valueOf("steps");
    if (steps && !Constant(graph, steps, 1)) return match.Reject("Slice step must be one.");
    const int64_t length = x[2] + past[2];
    const auto normalize = [length](int64_t index) {
      return index < 0 ? std::max(int64_t{0}, length + index) : std::min(length, index);
    };
    if (normalize(starts[0]) != x[2] || normalize(ends[0]) != length)
      return match.Reject("Slice must return precisely the trailing K-1 positions, including empty K=1.");
  }
  // Swish's alpha is independent of its input topology.
  for (const auto* node : match.nodes) {
    if (node->OpType() == "Swish") {
      const auto alpha = node->GetAttributes().find("alpha");
      if (alpha != node->GetAttributes().end() &&
          (alpha->second.type() != ONNX_NAMESPACE::AttributeProto::FLOAT || alpha->second.f() != 1.0f))
        return match.Reject("Swish alpha must be one.");
    }
  }
  return true;
}
}  // namespace
#endif

Rules MakeMobiusStatefulRules() {
  Rules rules;
#ifndef DISABLE_CONTRIB_OPS
  const PatternLibrary library(ConvLibrary);
  for (const auto* form : {"Plain", "Silu", "SwishForm"}) {
    const bool plain = std::string_view(form) == "Plain";
    const std::string replacement = std::string(R"(
      (x,w,bias,past) => {
        y,present = com.microsoft.CausalConvWithState(x,w,bias,past,
          ndim=1,dilation=1,channels_last=0,state_window=0,activation=")") +
                                    (plain ? "none" : "silu") + R"(")
        return y,present
      })";
    rules.emplace_back(std::string("MobiusCausalConv") + form, library.GetPattern(form, true), replacement, CheckConv,
                       InlinedVector<OpsetRequirement, 1>{{kOnnxDomain, 17, 24}, {kMSDomain, 1, 1}});
  }
#endif
  return rules;
}

}  // namespace onnxruntime::rewrite
