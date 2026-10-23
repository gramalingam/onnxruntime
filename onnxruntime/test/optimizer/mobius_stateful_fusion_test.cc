// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <limits>
#include <string>
#include <unordered_map>

#include "core/optimizer/mobius_stateful_fusion.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
namespace {
using namespace rewrite;

#ifndef DISABLE_CONTRIB_OPS
struct ConvCase {
  int64_t kernel = 3;
  int64_t sequence = 5;
  std::string activation = "none";
  bool computed_slice = false;
  bool explicit_step = false;
  bool output_consumers = false;
  bool internal_consumer = false;
  bool duplicate_state = false;
  std::string mutation;
};

// Faithful standard-ONNX expansion of Mobius causal_conv_nd_with_state at
// d69efe03a9f4c1ada744bc31dd01077a710b3c39, with an optional folded Slice.
std::pair<const NodeArg*, const NodeArg*> BuildConv(ModelTestBuilder& b, const ConvCase& c) {
  constexpr int64_t batch = 2, channels = 4;
  auto* x = b.MakeInput<float>({batch, channels, c.sequence}, -0.5f, 0.5f);
  auto* past = b.MakeInput<float>({batch, channels, c.kernel - 1 + (c.mutation == "past_shape" ? 1 : 0)},
                                  -0.5f, 0.5f);
  auto* w = b.MakeInitializer<float>({channels, c.mutation == "group" ? channels : 1, c.kernel}, -0.3f, 0.3f);
  auto* bias = b.MakeInput<float>({channels}, -0.2f, 0.2f);
  auto* padded = b.MakeIntermediate();
  b.AddNode("Concat", {past, x}, {padded}).AddAttribute("axis", int64_t{2});
  auto* present = b.MakeOutput();
  auto vector = [&](int64_t value) { return b.MakeInitializer<int64_t>({1}, {value}); };
  NodeArg* starts;
  NodeArg* ends;
  if (c.computed_slice) {
    auto* shape = b.MakeIntermediate();
    auto& shape_node = b.AddNode("Shape", {padded}, {shape});
    if (c.mutation == "shape_start") shape_node.AddAttribute("start", int64_t{1});
    auto* length = b.MakeIntermediate();
    b.AddNode("Gather", {shape, b.MakeScalarInitializer<int64_t>(c.mutation == "gather" ? 1 : 2)}, {length})
        .AddAttribute("axis", int64_t{0});
    auto* start = b.MakeIntermediate();
    b.AddNode("Sub",
              {length, b.MakeScalarInitializer<int64_t>(c.kernel - 1 + (c.mutation == "width" ? 1 : 0))},
              {start});
    starts = b.MakeIntermediate();
    ends = b.MakeIntermediate();
    b.AddNode("Reshape", {start, vector(1)}, {starts});
    b.AddNode("Reshape", {length, vector(1)}, {ends});
  } else {
    starts = vector(c.mutation == "start" ? 0 : c.sequence);
    ends = vector(c.mutation == "end" ? c.sequence : std::numeric_limits<int64_t>::max());
    if (c.mutation == "negative_start") starts = vector(1 - c.kernel);
    if (c.mutation == "runtime_start") starts = b.MakeInput<int64_t>({1}, {c.sequence});
  }
  std::vector<NodeArg*> slice_inputs{padded, starts, ends, vector(c.mutation == "axis" ? 1 : 2)};
  if (c.explicit_step || c.mutation == "step") slice_inputs.push_back(vector(c.mutation == "step" ? 2 : 1));
  b.AddNode("Slice", slice_inputs, {present});
  if (c.duplicate_state) b.AddNode("Slice", slice_inputs, {b.MakeOutput()});
  auto* conv = b.MakeIntermediate();
  auto& conv_node = b.AddNode("Conv", {padded, w}, {conv});
  conv_node.AddAttribute("group", c.mutation == "group" ? int64_t{1} : channels);
  conv_node.AddAttribute("kernel_shape", std::vector<int64_t>{c.kernel});
  conv_node.AddAttribute("dilations", std::vector<int64_t>{c.mutation == "dilation" ? 2 : 1});
  conv_node.AddAttribute("strides", std::vector<int64_t>{c.mutation == "stride" ? 2 : 1});
  conv_node.AddAttribute("pads", std::vector<int64_t>{c.mutation == "pads" ? 1 : 0, 0});
  auto* reshaped_bias = b.MakeIntermediate();
  b.AddNode("Reshape",
            {bias, b.MakeInitializer<int64_t>({3}, c.mutation == "bias" ? std::vector<int64_t>{1, 1, -1}
                                                                        : std::vector<int64_t>{1, -1, 1})},
            {reshaped_bias});
  auto* output = b.MakeOutput();
  auto* biased = c.activation == "none" ? output : b.MakeIntermediate();
  b.AddNode("Add", {conv, reshaped_bias}, {biased});
  if (c.activation == "silu") {
    auto* sigmoid = b.MakeIntermediate();
    b.AddNode("Sigmoid", {biased}, {sigmoid});
    b.AddNode("Mul", {biased, sigmoid}, {output});
  } else if (c.activation == "swish") {
    auto& swish = b.AddNode("Swish", {biased}, {output});
    if (c.mutation == "alpha") swish.AddAttribute("alpha", 0.5f);
  }
  if (c.internal_consumer) b.AddNode("Identity", {padded}, {b.MakeOutput()});
  if (c.output_consumers) {
    b.AddNode("Identity", {output}, {b.MakeOutput()});
    b.AddNode("Identity", {present}, {b.MakeOutput()});
  }
  return {output, present};
}

struct TestGraph {
  Model model{"stateful", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 24}, {kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger()};
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder{graph};
};

size_t Count(const Graph& graph, const char* op) {
  size_t count = 0;
  for (const auto& node : graph.Nodes()) count += node.OpType() == op;
  return count;
}

TEST(MobiusStatefulFusionTests, StructuralFormsAndOutputIdentity) {
  for (const auto* activation : {"none", "silu", "swish"}) {
    for (bool computed : {false, true}) {
      for (int64_t kernel : {1, 3}) {
        SCOPED_TRACE(MakeString(activation, " computed=", computed, " K=", kernel));
        TestGraph test;
        ConvCase c;
        c.activation = activation;
        c.computed_slice = computed;
        c.kernel = kernel;
        c.explicit_step = !computed;
        c.output_consumers = true;
        const auto original = BuildConv(test.builder, c);
        ASSERT_STATUS_OK(test.graph.Resolve());
        const auto outputs = test.graph.GetOutputs();
        PatternRewriter rewriter(MakeMobiusStatefulRules());
        Statistics stats;
        ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
        EXPECT_EQ(stats.rewrites, 1u);
        EXPECT_EQ(Count(test.graph, "CausalConvWithState"), 1u);
        EXPECT_EQ(Count(test.graph, "Conv"), 0u);
        EXPECT_EQ(Count(test.graph, "Slice"), 0u);
        EXPECT_EQ(Count(test.graph, "Concat"), 0u);
        EXPECT_EQ(Count(test.graph, "Identity"), 2u);
        EXPECT_EQ(Count(test.graph, "Sigmoid"), 0u);
        EXPECT_EQ(Count(test.graph, "Swish"), 0u);
        for (const auto& node : test.graph.Nodes()) {
          if (node.OpType() != "CausalConvWithState") continue;
          ASSERT_EQ(node.OutputDefs().size(), 2u);
          EXPECT_EQ(node.OutputDefs()[0], original.first);
          EXPECT_EQ(node.OutputDefs()[1], original.second);
          EXPECT_EQ(node.GetAttributes().at("activation").s(), c.activation == "none" ? "none" : "silu");
        }
        EXPECT_EQ(test.graph.GetOutputs(), outputs);
        ASSERT_STATUS_OK(test.graph.Resolve());
        ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
        EXPECT_EQ(stats.rewrites, 0u);
      }
    }
  }
}

TEST(MobiusStatefulFusionTests, RejectsMutatedConstantsAndBoundaries) {
  for (const auto* mutation : {"start", "end", "step", "axis", "runtime_start", "bias", "group", "past_shape",
                               "stride", "dilation", "pads", "width", "gather", "alpha"}) {
    SCOPED_TRACE(mutation);
    TestGraph test;
    ConvCase c;
    c.mutation = mutation;
    c.sequence = 4;  // Also keeps the incorrect [1,1,C] bias broadcast schema-valid.
    c.computed_slice = c.mutation == "width" || c.mutation == "gather";
    if (c.mutation == "alpha") c.activation = "swish";
    BuildConv(test.builder, c);
    ASSERT_STATUS_OK(test.graph.Resolve());
    PatternRewriter rewriter(MakeMobiusStatefulRules());
    Statistics stats;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 0u);
    EXPECT_EQ(Count(test.graph, "CausalConvWithState"), 0u);
  }
  TestGraph test;
  ConvCase c;
  c.internal_consumer = true;
  BuildConv(test.builder, c);
  ASSERT_STATUS_OK(test.graph.Resolve());
  PatternRewriter rewriter(MakeMobiusStatefulRules());
  Statistics stats;
  ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
}

TEST(MobiusStatefulFusionTests, CpuNumericalBothOutputs) {
  for (const auto* activation : {"none", "silu"}) {
    for (int64_t sequence : {1, 2, 5}) {
      for (int64_t kernel : {1, 3}) {
        SCOPED_TRACE(MakeString(activation, " T=", sequence, " K=", kernel));
        ConvCase c;
        c.activation = activation;
        c.kernel = kernel;
        c.sequence = sequence;
        c.computed_slice = true;
        c.output_consumers = true;
        TransformerTester(
            [c](ModelTestBuilder& b) { BuildConv(b, c); },
            [c](InferenceSessionWrapper& session) {
              EXPECT_EQ(Count(session.GetGraph(), "CausalConvWithState"), 1u);
              EXPECT_EQ(Count(session.GetGraph(), "Conv"), 0u);
              EXPECT_EQ(Count(session.GetGraph(), "Sigmoid"), 0u);
              for (const auto& node : session.GetGraph().Nodes()) {
                if (node.OpType() == "CausalConvWithState")
                  EXPECT_EQ(node.GetAttributes().at("activation").s(), c.activation);
              }
            },
            TransformerLevel::Default, TransformerLevel::Level1, 17, 1e-6, 1e-5,
            std::make_unique<PatternRewriter>(MakeMobiusStatefulRules()));
      }
    }
  }
}

TEST(MobiusStatefulFusionTests, NegativeStartAndProviderGuard) {
  for (bool cuda : {false, true}) {
    TestGraph test;
    ConvCase c;
    c.mutation = "negative_start";
    BuildConv(test.builder, c);
    ASSERT_STATUS_OK(test.graph.Resolve());
    if (cuda) {
      for (auto& node : test.graph.Nodes()) node.SetExecutionProviderType(kCudaExecutionProvider);
    }
    PatternRewriter rewriter(MakeMobiusStatefulRules());
    Statistics stats;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, cuda ? 0u : 1u);
  }
}

TEST(MobiusStatefulFusionTests, AmbiguousStateBranchIsRejected) {
  for (bool indexed : {false, true}) {
    TestGraph test;
    ConvCase c;
    c.duplicate_state = true;
    BuildConv(test.builder, c);
    ASSERT_STATUS_OK(test.graph.Resolve());
    Options options;
    options.indexed = indexed;
    PatternRewriter rewriter(MakeMobiusStatefulRules(), options);
    Statistics stats;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 0u);
    EXPECT_EQ(Count(test.graph, "Conv"), 1u);
  }
}

TEST(MobiusStatefulFusionTests, MultipleFusionsAgreeWithReference) {
  for (bool indexed : {false, true}) {
    TestGraph test;
    ConvCase c;
    BuildConv(test.builder, c);
    c.activation = "silu";
    c.computed_slice = true;
    BuildConv(test.builder, c);
    ASSERT_STATUS_OK(test.graph.Resolve());
    Options options;
    options.indexed = indexed;
    PatternRewriter rewriter(MakeMobiusStatefulRules(), options);
    Statistics stats;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 2u);
    EXPECT_EQ(Count(test.graph, "CausalConvWithState"), 2u);
    EXPECT_EQ(Count(test.graph, "Slice"), 0u);
    EXPECT_EQ(Count(test.graph, "Sigmoid"), 0u);
    ASSERT_STATUS_OK(test.graph.Resolve());
    ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 0u);
  }
}

TEST(MobiusStatefulFusionTests, SiblingMatchingIsExplicitAndReferenceAgrees) {
  const PatternLibrary library(R"(
    pattern Siblings(x) {
      a=Identity(x)
      l=Abs(a)
      r=Neg(a)
      return l,r
    }
  )");
  EXPECT_THROW(library.GetPattern("Siblings"), OnnxRuntimeException);
  for (bool indexed : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2, 4}, -1.0f, 1.0f);
    auto* a = b.MakeIntermediate();
    b.AddNode("Identity", {x}, {a});
    b.AddNode("Abs", {a}, {b.MakeOutput()});
    b.AddNode("Neg", {a}, {b.MakeOutput()});
    ASSERT_STATUS_OK(test.graph.Resolve());
    Rules rules;
    rules.emplace_back("siblings", library.GetPattern("Siblings", true),
                       "(x) => { l,r=Split(x,axis=0); return l,r; }", Rule::CheckAndBind{},
                       InlinedVector<OpsetRequirement, 1>{{kOnnxDomain, 17, 24}});
    Options options;
    options.indexed = indexed;
    options.match_only = true;
    PatternRewriter rewriter(std::move(rules), options);
    Statistics stats;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
    EXPECT_EQ(stats.matches, 1u);
    EXPECT_EQ(stats.rewrites, 0u);
  }
}

TEST(MobiusStatefulFusionTests, ScanBodyRemainsIneligibleBeforeCallback) {
  TestGraph test;
  auto& b = test.builder;
  auto* state = b.MakeInput<float>({2}, -1.0f, 1.0f);
  auto* input = b.MakeInput<float>({3, 2}, -1.0f, 1.0f);
  ONNX_NAMESPACE::GraphProto body;
  body.set_name("unrelated_recurrence");
  const auto value = [](ONNX_NAMESPACE::ValueInfoProto* info, const char* name) {
    info->set_name(name);
    auto* tensor = info->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
    tensor->mutable_shape()->add_dim()->set_dim_value(2);
  };
  value(body.add_input(), "state");
  value(body.add_input(), "input");
  value(body.add_output(), "present");
  value(body.add_output(), "output");
  for (const auto& names : {std::pair{"state", "present"}, std::pair{"input", "output"}}) {
    auto* node = body.add_node();
    node->set_op_type("Identity");
    node->add_input(names.first);
    node->add_output(names.second);
  }
  auto& scan = b.AddNode("Scan", {state, input}, {b.MakeOutput(), b.MakeOutput()});
  scan.AddAttribute("num_scan_inputs", int64_t{1});
  scan.AddAttribute("body", body);
  ASSERT_STATUS_OK(test.graph.Resolve());
  const PatternLibrary library(R"(
    pattern Recurrence(state,x) {
      s,y=Scan(state,x)
      return s,y
    }
  )");
  bool called = false, rejected_owner = false;
  Rules rules;
  rules.emplace_back(
      "not_a_linear_attention_rule", library.GetPattern("Recurrence", true),
      "(state,x) => { s,y=Scan(state,x); return s,y; }",
      [&called](const Graph&, Match&) {
        called = true;
        return true;
      },
      InlinedVector<OpsetRequirement, 1>{{kOnnxDomain, 17, 24}});
  Options options;
  options.match_only = true;
  options.on_failure = [&rejected_owner](const MatchFailure& failure) {
    if (failure.stage == FailureStage::Structure &&
        failure.reason == "Matched operators cannot own subgraphs or implicit inputs.")
      rejected_owner = true;
  };
  PatternRewriter rewriter(std::move(rules), options);
  Statistics stats;
  ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
  EXPECT_FALSE(called);
  EXPECT_TRUE(rejected_owner);
  EXPECT_EQ(stats.matches, 0u);
  EXPECT_EQ(Count(test.graph, "Scan"), 1u);
}
#else
TEST(MobiusStatefulFusionTests, ContribDisabled) {
  EXPECT_TRUE(MakeMobiusStatefulRules().empty());
}
#endif
}  // namespace
}  // namespace onnxruntime::test
