// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <limits>
#include <sstream>

#include "core/optimizer/mobius_attention_fusion.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

#ifndef DISABLE_CONTRIB_OPS
namespace onnxruntime {
namespace test {
namespace {
using namespace rewrite;

struct AttentionGraph {
  explicit AttentionGraph(int opset = 24, bool microsoft_import = true)
      : model("mobius_attention", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(),
              microsoft_import ? std::unordered_map<std::string, int>{{kOnnxDomain, opset}, {kMSDomain, 1}}
                               : std::unordered_map<std::string, int>{{kOnnxDomain, opset}},
              {}, DefaultLoggingManager().DefaultLogger()) {}
  Model model;
  Graph& graph = model.MainGraph();
  ModelTestBuilder b{graph};

  void Resolve() {
    b.SetGraphOutputs();
    ORT_THROW_IF_ERROR(graph.Resolve());
  }

  std::string Serialize() {
    std::string result;
    model.ToProto().SerializeToString(&result);
    return result;
  }
};

size_t Count(const Graph& graph, std::string_view op, std::string_view domain = kMSDomain) {
  size_t count = 0;
  for (const auto& node : graph.Nodes()) count += node.OpType() == op && node.Domain() == domain;
  return count;
}

void StandardOnly(const Graph& graph) {
  for (const auto& node : graph.Nodes()) EXPECT_EQ(node.Domain(), kOnnxDomain);
}

void AttentionAttributes(Node& node, int64_t heads, int64_t kv_heads, bool causal, float scale = 0.375f) {
  node.AddAttribute("q_num_heads", heads);
  node.AddAttribute("kv_num_heads", kv_heads);
  node.AddAttribute("scale", scale);
  node.AddAttribute("is_causal", int64_t{causal ? 1 : 0});
}

Status RunCpu(AttentionGraph& graph, const std::string& serialized, std::vector<OrtValue>& outputs) {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.intra_op_param.thread_pool_size = 1;
  InferenceSessionWrapper session{options, GetEnvironment()};
  std::istringstream stream(serialized);
  ORT_RETURN_IF_ERROR(session.Load(stream));
  ORT_RETURN_IF_ERROR(session.Initialize());
  return session.Run(RunOptions{}, graph.b.feeds_, graph.b.output_names_, &outputs);
}

void CompareCpu(AttentionGraph& graph, const std::string& original, float tolerance = 2e-6f) {
  std::vector<OrtValue> expected, actual;
  ASSERT_STATUS_OK(RunCpu(graph, original, expected));
  ASSERT_STATUS_OK(RunCpu(graph, graph.Serialize(), actual));
  ASSERT_EQ(actual.size(), expected.size());
  ASSERT_FALSE(actual.empty());
  for (size_t output = 0; output < actual.size(); ++output) {
    const auto& e = expected[output].Get<Tensor>();
    const auto& a = actual[output].Get<Tensor>();
    ASSERT_EQ(a.Shape(), e.Shape());
    for (int64_t i = 0; i < a.Shape().Size(); ++i)
      EXPECT_NEAR(a.Data<float>()[i], e.Data<float>()[i], tolerance) << "output " << output << " element " << i;
  }
}

// Faithful boundary generator for Mobius d69efe03a9f4c1ada744bc31dd01077a710b3c39,
// components/_pixtral_vision.py:147-202. Projections and 2D RoPE remain outside the boundary.
Node& Vision(AttentionGraph& test, int64_t batch = 1, int64_t sequence = 3, int64_t heads = 2) {
  auto& b = test.b;
  auto* raw_q = b.MakeInput<float>({batch, sequence, heads * 4}, -0.7f, 0.7f);
  auto* q = b.MakeIntermediate();
  b.AddNode("Reshape", {raw_q, b.MakeInitializer<int64_t>({3}, {batch, sequence, heads * 4})}, {q});
  auto* k = b.MakeInput<float>({batch, sequence, heads * 4}, -0.7f, 0.7f);
  auto* v = b.MakeInput<float>({batch, sequence, heads * 4}, -0.7f, 0.7f);
  auto* y = b.MakeIntermediate();
  auto& node = b.AddNode("Attention", {q, k, v}, {y});
  AttentionAttributes(node, heads, heads, false);
  b.AddNode("Identity", {y}, {b.MakeOutput()});
  b.AddNode("Add", {y, y}, {b.MakeOutput()});
  b.AddNode("Identity", {q}, {b.MakeOutput()});
  return node;
}

// components/_attention.py:77-231, dynamic-cache, no-padding-mask branch.
Node& Decoder(AttentionGraph& test, int64_t batch, int64_t sequence, int64_t past, int64_t head_size = 8) {
  auto& b = test.b;
  auto* q = b.MakeInput<float>({batch, sequence, 4 * head_size}, -0.5f, 0.5f);
  auto* k = b.MakeInput<float>({batch, sequence, 2 * head_size}, -0.5f, 0.5f);
  auto* v = b.MakeInput<float>({batch, sequence, 2 * head_size}, -0.5f, 0.5f);
  auto* pk = b.MakeInput<float>({batch, 2, past, head_size}, -0.5f, 0.5f);
  auto* pv = b.MakeInput<float>({batch, 2, past, head_size}, -0.5f, 0.5f);
  auto* y = b.MakeOutput();
  auto* present_k = b.MakeOutput();
  auto* present_v = b.MakeOutput();
  auto& node = b.AddNode("Attention", {q, k, v, b.MakeOptionalTensor(), pk, pv}, {y, present_k, present_v});
  AttentionAttributes(node, 4, 2, true);
  b.AddNode("Identity", {present_k}, {b.MakeOutput()});
  b.AddNode("Identity", {present_v}, {b.MakeOutput()});
  b.AddNode("Identity", {pk}, {b.MakeOutput()});
  b.AddNode("Identity", {pv}, {b.MakeOutput()});
  b.AddNode("Add", {y, y}, {b.MakeOutput()});
  return node;
}

struct PackedMutation {
  std::vector<int64_t> cumulative{0, 3};
  bool mutable_cu = false;
  bool cast_like = true;
  bool live_mask = false;
  bool live_attention = false;
  bool live_positions = false;
  int64_t subtract = 1;
  int64_t cast_to = 7;
  float negative = -10000.0f;
  float zero = 0.0f;
};

// components/_qwen3_vl_vision.py:271-340, including the actual segment-mask chain.
Node& Packed(AttentionGraph& test, const PackedMutation& mutation = {}) {
  auto& b = test.b;
  auto unary = [&](const char* op, NodeArg* input) {
    auto* out = b.MakeIntermediate();
    b.AddNode(op, {input}, {out});
    return out;
  };
  auto binary = [&](const char* op, NodeArg* a, NodeArg* c) {
    auto* out = b.MakeIntermediate();
    b.AddNode(op, {a, c}, {out});
    return out;
  };
  auto unsqueeze = [&](NodeArg* input, std::vector<int64_t> axes) {
    const auto rank = static_cast<int64_t>(axes.size());
    return binary("Unsqueeze", input, b.MakeInitializer<int64_t>({rank}, axes));
  };
  auto* q = b.MakeInput<float>({3, 8}, -0.5f, 0.5f);
  auto* k = b.MakeInput<float>({3, 8}, -0.5f, 0.5f);
  auto* v = b.MakeInput<float>({3, 8}, -0.5f, 0.5f);
  auto* hidden = b.MakeInput<float>({3, 8}, -0.5f, 0.5f);
  const auto cu_size = static_cast<int64_t>(mutation.cumulative.size());
  auto* cu = mutation.mutable_cu ? b.MakeInput<int64_t>({cu_size}, mutation.cumulative)
                                 : b.MakeInitializer<int64_t>({cu_size}, mutation.cumulative);
  auto* shape = b.MakeIntermediate();
  auto& shape_node = b.AddNode("Shape", {hidden}, {shape});
  shape_node.AddAttribute("start", int64_t{0});
  shape_node.AddAttribute("end", int64_t{1});
  auto* limit = unary("Squeeze", shape);
  auto* positions = b.MakeIntermediate();
  b.AddNode("Range", {b.MakeScalarInitializer<int64_t>(0), limit, b.MakeScalarInitializer<int64_t>(1)}, {positions});
  auto* ge = binary("GreaterOrEqual", unsqueeze(positions, {1}), unsqueeze(cu, {0}));
  auto* integers = b.MakeIntermediate();
  b.AddNode("Cast", {ge}, {integers}).AddAttribute("to", mutation.cast_to);
  auto* counts = b.MakeIntermediate();
  b.AddNode("ReduceSum", {integers, b.MakeInitializer<int64_t>({1}, {1})}, {counts})
      .AddAttribute("keepdims", int64_t{0});
  auto* segments = binary("Sub", counts, b.MakeScalarInitializer<int64_t>(mutation.subtract));
  auto* same = binary("Equal", unsqueeze(segments, {1}), unsqueeze(segments, {0}));
  NodeArg* zero = b.MakeScalarInitializer<float>(mutation.zero);
  NodeArg* negative = b.MakeScalarInitializer<float>(mutation.negative);
  if (mutation.cast_like) {
    zero = binary("CastLike", zero, q);
    negative = binary("CastLike", negative, q);
  }
  auto* bias = b.MakeIntermediate();
  b.AddNode("Where", {same, zero, negative}, {bias});
  auto* bias4 = unsqueeze(bias, {0, 1});
  auto* y = b.MakeIntermediate();
  auto& node = b.AddNode("Attention", {unsqueeze(q, {0}), unsqueeze(k, {0}), unsqueeze(v, {0}), bias4}, {y});
  AttentionAttributes(node, 2, 2, false);
  auto* flat = b.MakeIntermediate();
  b.AddNode("Squeeze", {y, b.MakeInitializer<int64_t>({1}, {0})}, {flat});
  b.AddNode("Identity", {flat}, {b.MakeOutput()});
  b.AddNode("Add", {flat, flat}, {b.MakeOutput()});
  if (mutation.live_mask) b.AddNode("Identity", {bias4}, {b.MakeOutput()});
  if (mutation.live_attention) b.AddNode("Identity", {y}, {b.MakeOutput()});
  if (mutation.live_positions) b.AddNode("Identity", {positions}, {b.MakeOutput()});
  return node;
}

TEST(MobiusAttentionFusionTests, VisionCpuNumericsAndConsumers) {
  for (int opset : {23, 24}) {
    for (bool indexed : {false, true}) {
      for (int64_t batch : {1, 2}) {
        AttentionGraph test(opset);
        auto& attention = Vision(test, batch);
        if (batch == 2) attention.AddAttribute("softmax_precision", int64_t{1});
        test.Resolve();
        StandardOnly(test.graph);
        const auto original = test.Serialize();
        Options options;
        options.indexed = indexed;
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter(MakeMobiusAttentionRules(), options).Run(test.graph, stats));
        EXPECT_EQ(stats.rewrites, 1u);
        EXPECT_EQ(Count(test.graph, "MultiHeadAttention"), 1u);
        EXPECT_EQ(Count(test.graph, "Attention", kOnnxDomain), 0u);
        EXPECT_EQ(Count(test.graph, "Reshape", kOnnxDomain), 1u);
        ASSERT_STATUS_OK(test.graph.Resolve());
        CompareCpu(test, original);
      }
    }
  }
}

TEST(MobiusAttentionFusionTests, VisionRejectsSemanticMutations) {
  for (int scenario = 0; scenario < 11; ++scenario) {
    SCOPED_TRACE(scenario);
    AttentionGraph test;
    auto& node = Vision(test);
    switch (scenario) {
      case 0:
        node.AddAttribute("is_causal", int64_t{1});
        break;
      case 1:
        node.AddAttribute("softcap", 1.0f);
        break;
      case 2:
        node.AddAttribute("softmax_precision", int64_t{11});
        break;
      case 3:
        node.AddAttribute("scale", 0.0f);
        break;
      case 4:
        node.AddAttribute("scale", std::numeric_limits<float>::quiet_NaN());
        break;
      case 5:
        node.ClearAttribute("scale");
        break;
      case 6:
        node.MutableInputDefs().push_back(test.b.MakeInput<bool>({3, 3}, std::vector<bool>(9, true)));
        node.MutableInputArgsCount().push_back(1);
        break;
      case 7:
        node.MutableInputDefs()[1] = test.b.MakeInput<float>({1, 2, 8}, -0.5f, 0.5f);
        node.MutableInputDefs()[2] = test.b.MakeInput<float>({1, 2, 8}, -0.5f, 0.5f);
        break;
      case 8:
        node.MutableInputDefs()[2] = test.b.MakeInput<float>({1, 3, 12}, -0.5f, 0.5f);
        break;
      case 9:
        node.MutableInputDefs()[1] = test.b.MakeInput<float>({1, 3, 4}, -0.5f, 0.5f);
        node.MutableInputDefs()[2] = test.b.MakeInput<float>({1, 3, 4}, -0.5f, 0.5f);
        node.AddAttribute("kv_num_heads", int64_t{1});
        break;
      case 10:
        for (int i = 0; i < 3; ++i) node.MutableOutputDefs().push_back(test.b.MakeOutput());
        break;
    }
    test.Resolve();
    const auto before = test.Serialize();
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeMobiusAttentionRules()).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 0u);
    EXPECT_EQ(Count(test.graph, "Attention", kOnnxDomain), 1u);
    EXPECT_EQ(before, test.Serialize());
  }
}

TEST(MobiusAttentionFusionTests, DecoderCpuNumericsAndAllCacheConsumers) {
  for (int opset : {23, 24}) {
    for (bool indexed : {false, true}) {
      for (const auto& geometry : {std::vector<int64_t>{2, 1, 3}, {1, 2, 3}, {2, 3, 0}}) {
        AttentionGraph test(opset);
        auto& attention = Decoder(test, geometry[0], geometry[1], geometry[2]);
        if (geometry[0] == 2) attention.AddAttribute("softmax_precision", int64_t{1});
        test.Resolve();
        StandardOnly(test.graph);
        const auto original = test.Serialize();
        Options options;
        options.indexed = indexed;
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter(MakeMobiusAttentionRules(), options).Run(test.graph, stats));
        EXPECT_EQ(stats.rewrites, 1u);
        EXPECT_EQ(Count(test.graph, "GroupQueryAttention"), 1u);
        ASSERT_STATUS_OK(test.graph.Resolve());
        for (size_t output = 1; output <= 4; ++output) {
          const auto* shape = test.graph.GetOutputs()[output]->Shape();
          ASSERT_NE(shape, nullptr);
          ASSERT_EQ(shape->dim_size(), 4);
          const std::vector<int64_t> expected{geometry[0], 2, geometry[1] + geometry[2], 8};
          for (int dimension = 0; dimension < 4; ++dimension) {
            EXPECT_TRUE(shape->dim(dimension).has_dim_value());
            EXPECT_EQ(shape->dim(dimension).dim_value(), expected[dimension]);
          }
        }
        CompareCpu(test, original);
      }
    }
  }
}

TEST(MobiusAttentionFusionTests, VisionRejectsRanksDtypesDynamicShapesAndProviders) {
  for (int scenario = 0; scenario < 5; ++scenario) {
    SCOPED_TRACE(scenario);
    AttentionGraph test(scenario == 4 ? 25 : 24);
    auto& node = Vision(test);
    if (scenario == 0) {
      for (int i = 0; i < 3; ++i)
        node.MutableInputDefs()[i] = test.b.MakeInput<MLFloat16>({1, 3, 8}, MLFloat16(-0.5f), MLFloat16(0.5f));
    } else if (scenario == 1) {
      for (int i = 0; i < 3; ++i)
        node.MutableInputDefs()[i] = test.b.MakeInput<float>({1, 2, 3, 4}, -0.5f, 0.5f);
    } else if (scenario == 2) {
      for (int i = 0; i < 3; ++i)
        node.MutableInputDefs()[i] =
            test.b.MakeInput<float>(std::optional<std::vector<int64_t>>{{-1, 3, 8}});
    } else if (scenario == 3) {
      node.SetExecutionProviderType(kCudaExecutionProvider);
    }
    test.Resolve();
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeMobiusAttentionRules()).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 0u);
    EXPECT_EQ(Count(test.graph, "MultiHeadAttention"), 0u);
  }
}

TEST(MobiusAttentionFusionTests, DecoderRejectsMasksPrecisionAndUnsupportedGeometry) {
  for (int scenario = 0; scenario < 10; ++scenario) {
    SCOPED_TRACE(scenario);
    AttentionGraph test;
    auto& node = Decoder(test, scenario == 5 ? 2 : 1, scenario == 5 ? 2 : 1, 3, scenario == 9 ? 4 : 8);
    switch (scenario) {
      case 0:
        node.AddAttribute("is_causal", int64_t{0});
        break;
      case 1:
        node.AddAttribute("softcap", 2.0f);
        break;
      case 2:
        node.AddAttribute("softmax_precision", int64_t{11});
        break;
      case 3:
        node.MutableInputDefs()[3] = test.b.MakeInput<bool>({1, 4}, std::vector<bool>{true, false, true, true});
        break;
      case 4:
        node.MutableInputDefs()[2] = test.b.MakeInput<float>({1, 1, 12}, -0.5f, 0.5f);
        node.MutableInputDefs()[5] = test.b.MakeInput<float>({1, 2, 3, 6}, -0.5f, 0.5f);
        break;
      case 5:
        break;
      case 6:
        node.AddAttribute("scale", -0.5f);
        break;
      case 7:
        node.MutableOutputDefs().push_back(test.b.MakeOutput());
        break;
      case 8:
        node.MutableInputDefs()[4] = test.b.MakeOptionalTensor();
        node.MutableInputDefs()[5] = test.b.MakeOptionalTensor();
        node.MutableInputDefs().push_back(test.b.MakeInitializer<int64_t>({1}, {1}));
        node.MutableInputArgsCount().push_back(1);
        break;
      case 9:
        break;
    }
    test.Resolve();
    const auto before = test.Serialize();
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeMobiusAttentionRules()).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 0u);
    EXPECT_EQ(Count(test.graph, "Attention", kOnnxDomain), 1u);
    EXPECT_EQ(before, test.Serialize());
  }
}

TEST(MobiusAttentionFusionTests, AssignedCudaFp32DecoderIsRejectedEvenWhenProviderIsAllowed) {
  AttentionGraph test;
  auto& node = Decoder(test, 1, 1, 3);
  node.SetExecutionProviderType(kCudaExecutionProvider);
  test.Resolve();
  Options options;
  options.compatible_execution_providers.insert(kCudaExecutionProvider);
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter(MakeMobiusAttentionRules(), options).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
  EXPECT_EQ(Count(test.graph, "GroupQueryAttention"), 0u);
}

TEST(MobiusAttentionFusionTests, DecoderSchemaInfersOnlyKnownCacheLength) {
  for (bool constant_total : {false, true}) {
    for (bool sliding_window : {false, true}) {
      AttentionGraph test;
      auto& b = test.b;
      auto* q = b.MakeInput<float>({1, 1, 32}, -0.5f, 0.5f);
      auto* k = b.MakeInput<float>({1, 1, 16}, -0.5f, 0.5f);
      auto* v = b.MakeInput<float>({1, 1, 16}, -0.5f, 0.5f);
      auto* pk = b.MakeInput<float>({1, 2, 3, 8}, -0.5f, 0.5f);
      auto* pv = b.MakeInput<float>({1, 2, 3, 8}, -0.5f, 0.5f);
      auto* lengths = b.MakeInput<int32_t>({1}, {3});
      auto* total = constant_total ? b.MakeScalarInitializer<int32_t>(4) : b.MakeInput<int32_t>({}, {4});
      auto* present_k = b.MakeOutput();
      auto* present_v = b.MakeOutput();
      auto& node = b.AddNode("GroupQueryAttention", {q, k, v, pk, pv, lengths, total},
                             {b.MakeOutput(), present_k, present_v}, kMSDomain);
      node.AddAttribute("num_heads", int64_t{4});
      node.AddAttribute("kv_num_heads", int64_t{2});
      if (sliding_window) node.AddAttribute("sliding_window_cache", int64_t{1});
      test.Resolve();
      for (const auto* output : {present_k, present_v}) {
        const auto* shape = output->Shape();
        ASSERT_NE(shape, nullptr);
        ASSERT_EQ(shape->dim_size(), 4);
        EXPECT_EQ(shape->dim(0).dim_value(), 1);
        EXPECT_EQ(shape->dim(1).dim_value(), 2);
        EXPECT_EQ(shape->dim(3).dim_value(), 8);
        EXPECT_EQ(shape->dim(2).has_dim_value(), constant_total || sliding_window);
        if (constant_total || sliding_window)
          EXPECT_EQ(shape->dim(2).dim_value(), sliding_window ? 3 : 4);
      }
    }
  }
}

TEST(MobiusAttentionFusionTests, PackedSingleSequenceCanonicalAndFoldedScalarForms) {
  for (bool indexed : {false, true}) {
    for (bool cast_like : {false, true}) {
      AttentionGraph test;
      PackedMutation mutation;
      mutation.cast_like = cast_like;
      Packed(test, mutation);
      test.Resolve();
      StandardOnly(test.graph);
      Options options;
      options.indexed = indexed;
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter(MakeMobiusAttentionRules(true), options).Run(test.graph, stats));
      EXPECT_EQ(stats.rewrites, 1u);
      EXPECT_EQ(Count(test.graph, "PackedMultiHeadAttention"), 1u);
      EXPECT_EQ(Count(test.graph, "Attention", kOnnxDomain), 0u);
      EXPECT_EQ(Count(test.graph, "Where", kOnnxDomain), 0u);
      ASSERT_STATUS_OK(test.graph.Resolve());
      for (const auto& node : test.graph.Nodes()) {
        if (node.OpType() != "PackedMultiHeadAttention") continue;
        ASSERT_EQ(node.InputDefs().size(), 6u);
        ASSERT_EQ(node.OutputDefs().size(), 1u);
        EXPECT_EQ(node.GetAttributes().at("num_heads").i(), 2);
        EXPECT_FLOAT_EQ(node.GetAttributes().at("scale").f(), 0.375f);
        const auto* offset_shape = node.InputDefs()[4]->Shape();
        ASSERT_NE(offset_shape, nullptr);
        ASSERT_EQ(offset_shape->dim_size(), 2);
        EXPECT_EQ(offset_shape->dim(0).dim_value(), 1);
        EXPECT_EQ(offset_shape->dim(1).dim_value(), 3);
      }
    }
  }
}

TEST(MobiusAttentionFusionTests, PackedRejectsMultipleSequencesAndSemanticMutations) {
  for (int scenario = 0; scenario < 12; ++scenario) {
    SCOPED_TRACE(scenario);
    AttentionGraph test;
    PackedMutation mutation;
    switch (scenario) {
      case 0:
        mutation.cumulative = {0, 1, 3};
        break;
      case 1:
        mutation.cumulative = {1, 3};
        break;
      case 2:
        mutation.cumulative = {0, 2};
        break;
      case 3:
        mutation.mutable_cu = true;
        break;
      case 4:
        mutation.live_mask = true;
        break;
      case 5:
        mutation.live_attention = true;
        break;
      case 6:
        mutation.live_positions = true;
        break;
      case 7:
        mutation.subtract = 2;
        break;
      case 8:
        mutation.negative = -100.0f;
        break;
      case 9:
        mutation.zero = 1.0f;
        break;
      default:
        break;
    }
    auto& node = Packed(test, mutation);
    if (scenario == 10) node.AddAttribute("is_causal", int64_t{1});
    if (scenario == 11) node.AddAttribute("kv_num_heads", int64_t{1});
    test.Resolve();
    const auto before = test.Serialize();
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeMobiusAttentionRules(true)).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 0u);
    EXPECT_EQ(before, test.Serialize());
  }
}

TEST(MobiusAttentionFusionTests, PackedRequiresDeploymentOptIn) {
  AttentionGraph test;
  Packed(test);
  test.Resolve();
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter(MakeMobiusAttentionRules()).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
}

TEST(MobiusAttentionFusionTests, PackedRejectsAssignedCpuEvenWithDeploymentOptIn) {
  AttentionGraph test;
  Packed(test);
  for (auto& node : test.graph.Nodes()) node.SetExecutionProviderType(kCpuExecutionProvider);
  test.Resolve();
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter(MakeMobiusAttentionRules(true)).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
}

TEST(MobiusAttentionFusionTests, FinitePackedMaskIsNotAHardBlockMask) {
  AttentionGraph test;
  auto& b = test.b;
  auto* q = b.MakeInput<float>({1, 2, 1}, {1.0f, 1.0f});
  auto* k = b.MakeInput<float>({1, 2, 1}, {0.0f, 20000.0f});
  auto* v = b.MakeInput<float>({1, 2, 1}, {0.0f, 1.0f});
  auto* mask = b.MakeInitializer<float>({1, 1, 2, 2}, {0.0f, -10000.0f, -10000.0f, 0.0f});
  auto& attention = b.AddNode("Attention", {q, k, v, mask}, {b.MakeOutput()});
  AttentionAttributes(attention, 1, 1, false, 1.0f);
  test.Resolve();
  std::vector<OrtValue> actual;
  ASSERT_STATUS_OK(RunCpu(test, test.Serialize(), actual));
  ASSERT_EQ(actual.size(), 1u);
  // Hard packed attention with lengths [1,1] must return V[0]=0 for query zero.
  EXPECT_GT(actual[0].Get<Tensor>().Data<float>()[0], 0.99f);
}

TEST(MobiusAttentionFusionTests, MissingMicrosoftImportIsNotAddedImplicitly) {
  AttentionGraph test(24, false);
  Vision(test);
  test.Resolve();
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter(MakeMobiusAttentionRules()).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
  EXPECT_EQ(test.graph.DomainToVersionMap().count(kMSDomain), 0u);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
#endif
