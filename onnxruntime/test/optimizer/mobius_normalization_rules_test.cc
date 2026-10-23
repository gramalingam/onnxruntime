// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_map>

#include "core/optimizer/pattern_rewriter.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {
using namespace rewrite;

struct Fixture {
  explicit Fixture(int opset = 24, bool import_contrib = true)
      : model{"mobius_normalization", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), import_contrib ? std::unordered_map<std::string, int>{{kOnnxDomain, opset}, {kMSDomain, 1}} : std::unordered_map<std::string, int>{{kOnnxDomain, opset}}, {}, DefaultLoggingManager().DefaultLogger()} {}
  Model model;
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder{graph};
};

#ifndef DISABLE_CONTRIB_OPS
// Faithful reproducible Mobius bodies, not exports from an unpinned installed package:
// https://github.com/onnxruntime/mobius/blob/d69efe03a9f4c1ada744bc31dd01077a710b3c39/src/mobius/functions/skip_layer_normalization.py
// The RMS body's dummy constants at outputs 1/2 are independent, not RMS statistics.
struct NormConfig {
  bool rms = false;
  bool beta = true;
  bool empty_beta = false;
  int outputs = 1;
  bool stats_output = true;
  bool stats_consumer = false;
  std::vector<int64_t> shape{2, 3, 4};
  std::vector<int64_t> skip_shape{2, 3, 4};
  int64_t scale_width = 4;
  int64_t beta_width = 4;
  bool defaults = false;
  bool positive_axis = false;
  bool sum_output = true;
  bool sum_consumer = true;
};

struct NormNodes {
  NodeArg* x;
  NodeArg* skip;
  NodeArg* sum;
  Node* norm;
};

template <typename T = float>
NormNodes BuildNorm(ModelTestBuilder& b, const NormConfig& config = {}) {
  auto* x = b.MakeInput<T>(config.shape, T(-1.0f), T(1.0f));
  auto* skip = b.MakeInput<T>(config.skip_shape, T(-0.5f), T(0.5f));
  auto* sum = config.sum_output ? b.MakeOutput() : b.MakeIntermediate();
  b.AddNode("Add", {x, skip}, {sum});
  std::vector<NodeArg*> inputs{x, b.MakeInitializer<T>({config.scale_width}, T(0.8f), T(1.2f))};
  inputs[0] = sum;
  if (!config.rms && config.beta)
    inputs.push_back(b.MakeInitializer<T>({config.beta_width}, T(-0.1f), T(0.1f)));
  else if (!config.rms && config.empty_beta)
    inputs.push_back(b.MakeEmptyInput());
  std::vector<NodeArg*> outputs{b.MakeOutput()};
  for (int i = 1; i < config.outputs; ++i) {
    auto* statistic = config.stats_output ? b.MakeOutput() : b.MakeIntermediate();
    outputs.push_back(statistic);
    if (config.stats_consumer) b.AddNode("Identity", {statistic}, {b.MakeOutput()});
  }
  auto& norm = b.AddNode(config.rms ? "RMSNormalization" : "LayerNormalization", inputs, outputs);
  if (!config.defaults) {
    norm.AddAttribute("epsilon", 3e-4f);
    norm.AddAttribute("axis", config.positive_axis ? static_cast<int64_t>(config.shape.size() - 1) : int64_t{-1});
    norm.AddAttribute("stash_type", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
  }
  if (config.sum_consumer) b.AddNode("Identity", {sum}, {b.MakeOutput()});
  return {x, skip, sum, &norm};
}

// Mobius rewrite_rules/_bias_gelu.py at the same SHA matches this topology but
// incorrectly accepts tanh. These fixtures deliberately exercise the exact mode.
struct GeluConfig {
  bool bias_first = false;
  bool defaults = false;
  std::string approximate = "none";
  std::vector<int64_t> shape{2, 3, 4};
  std::vector<int64_t> bias_shape{4};
  bool sum_output = false;
  bool sum_consumer = false;
};

template <typename T = float>
void BuildGelu(ModelTestBuilder& b, const GeluConfig& config = {}) {
  auto* x = b.MakeInput<T>(config.shape, T(-3.0f), T(3.0f));
  auto* bias = b.MakeInput<T>(config.bias_shape, T(-0.25f), T(0.25f));
  auto* sum = config.sum_output ? b.MakeOutput() : b.MakeIntermediate();
  b.AddNode("Add", config.bias_first ? std::vector<NodeArg*>{bias, x} : std::vector<NodeArg*>{x, bias}, {sum});
  auto& gelu = b.AddNode("Gelu", {sum}, {b.MakeOutput()});
  if (!config.defaults) gelu.AddAttribute("approximate", config.approximate);
  if (config.sum_consumer) b.AddNode("Identity", {sum}, {b.MakeOutput()});
}

Status Execute(const std::string& bytes, const ModelTestBuilder& builder, std::vector<OrtValue>& outputs) {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.intra_op_param.thread_pool_size = 1;
  InferenceSessionWrapper session{options, GetEnvironment()};
  std::istringstream stream(bytes);
  ORT_RETURN_IF_ERROR(session.Load(stream));
  ORT_RETURN_IF_ERROR(session.Initialize());
  return session.Run(RunOptions{}, builder.feeds_, builder.output_names_, &outputs);
}

void CheckFixture(Fixture& fixture, const char* target, bool accepted, bool numerical = false) {
  fixture.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  const Node* source_norm = nullptr;
  InlinedHashMap<std::string, std::string> providers;
  for (const auto& node : fixture.graph.Nodes()) {
    providers.emplace(node.Name(), node.GetExecutionProviderType());
    if (node.OpType() == "LayerNormalization" || node.OpType() == "RMSNormalization") source_norm = &node;
  }
  const auto source = fixture.model.ToProto();
  std::string original;
  ASSERT_TRUE(source.SerializeToString(&original));
  std::vector<OrtValue> expected;
  if (numerical) ASSERT_STATUS_OK(Execute(original, fixture.builder, expected));
  std::string reference;
  for (bool indexed : {false, true}) {
    SCOPED_TRACE(indexed);
    Model model(source, {}, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    ASSERT_STATUS_OK(graph.Resolve());
    for (auto& node : graph.Nodes())
      node.SetExecutionProviderType(providers.at(node.Name()));
    std::string before;
    ASSERT_TRUE(model.ToProto().SerializeToString(&before));
    Options options;
    options.indexed = indexed;
    // The factory must still reject CUDA when the caller allows it globally.
    options.compatible_execution_providers.insert(kCudaExecutionProvider);
    PatternRewriter rewriter(MakeMobiusNormalizationRules(), options);
    Statistics statistics;
    ASSERT_STATUS_OK(rewriter.Run(graph, statistics));
    EXPECT_EQ(statistics.rewrites, accepted ? 1u : 0u);
    size_t count = 0;
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() != target) continue;
      ++count;
      EXPECT_EQ(node.Domain(), kMSDomain);
      if (node.OpType() != "BiasGelu") {
        ASSERT_NE(source_norm, nullptr);
        ASSERT_EQ(node.OutputDefs().size(), 4u);
        EXPECT_EQ(node.OutputDefs()[3]->Name(), source_norm->InputDefs()[0]->Name());
        EXPECT_EQ(node.GetAttributes().at("epsilon").f(), source_norm->GetAttributes().at("epsilon").f());
        EXPECT_EQ(node.InputDefs().size(), node.OpType() == "SkipLayerNormalization" ? 4u : 3u);
        if (node.OpType() == "SkipLayerNormalization") {
          const auto& inputs = source_norm->InputDefs();
          EXPECT_EQ(node.InputDefs()[3]->Name(), inputs.size() > 2 ? inputs[2]->Name() : "");
          if (inputs.size() > 2 && inputs[2]->Exists()) EXPECT_EQ(node.InputDefs()[3], graph.GetNodeArg(inputs[2]->Name()));
          if (source_norm->OutputDefs().size() == 3) {
            EXPECT_EQ(node.OutputDefs()[1]->Name(), source_norm->OutputDefs()[1]->Name());
            EXPECT_EQ(node.OutputDefs()[2]->Name(), source_norm->OutputDefs()[2]->Name());
          }
        }
      }
    }
    EXPECT_EQ(count, accepted ? 1u : 0u);
    ASSERT_STATUS_OK(graph.Resolve());
    std::string rewritten;
    ASSERT_TRUE(model.ToProto().SerializeToString(&rewritten));
    if (!accepted) EXPECT_EQ(before, rewritten);
    if (!indexed)
      reference = rewritten;
    else
      EXPECT_EQ(reference, rewritten);
    ASSERT_STATUS_OK(rewriter.Run(graph, statistics));
    EXPECT_EQ(statistics.rewrites, 0u);
    ONNX_NAMESPACE::ModelProto parsed;
    ASSERT_TRUE(parsed.ParseFromString(rewritten));
    Model reloaded(parsed, {}, DefaultLoggingManager().DefaultLogger());
    ASSERT_STATUS_OK(reloaded.MainGraph().Resolve());
    if (!accepted) {
      for (auto& node : reloaded.MainGraph().Nodes()) node.SetExecutionProviderType(providers.at(node.Name()));
    }
    ASSERT_STATUS_OK(rewriter.Run(reloaded.MainGraph(), statistics));
    EXPECT_EQ(statistics.rewrites, 0u);
    if (numerical) {
      std::vector<OrtValue> actual;
      ASSERT_STATUS_OK(Execute(rewritten, fixture.builder, actual));
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t output = 0; output < expected.size(); ++output) {
        const auto& a = expected[output].Get<Tensor>();
        const auto& b = actual[output].Get<Tensor>();
        ASSERT_EQ(a.Shape(), b.Shape());
        for (int64_t i = 0; i < a.Shape().Size(); ++i) {
          const float value = a.Data<float>()[i];
          EXPECT_NEAR(value, b.Data<float>()[i], 2e-5f * (1.0f + std::abs(value)));
        }
      }
    }
  }
}
#endif
}  // namespace

TEST(MobiusNormalizationRulesTests, FactoryIsExplicitOptIn) {
#ifdef DISABLE_CONTRIB_OPS
  EXPECT_TRUE(MakeMobiusNormalizationRules().empty());
#else
  EXPECT_EQ(MakeMobiusNormalizationRules().size(), 4u);
  Fixture fixture;
  BuildGelu(fixture.builder);
  fixture.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(fixture.graph.Resolve());
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter(MakeExampleRules()).Run(fixture.graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
#endif
}

#ifndef DISABLE_CONTRIB_OPS
TEST(MobiusNormalizationRulesTests, SourceBackedLayerNormAndRmsFixtures) {
  for (bool rms : {false, true}) {
    for (int variant = 0; variant < 8; ++variant) {
      SCOPED_TRACE(rms);
      SCOPED_TRACE(variant);
      Fixture fixture(rms ? (variant == 0 ? 23 : 24) : (variant == 0 ? 17 : 24));
      NormConfig config;
      config.rms = rms;
      config.beta = variant % 2 == 0;
      config.empty_beta = variant == 1;
      config.outputs = !rms && variant >= 4 ? 3 : 1;
      config.defaults = variant == 0;
      config.positive_axis = variant == 2;
      config.sum_output = variant % 2 == 0;
      config.sum_consumer = variant % 3 != 0;
      if (variant == 3) config.shape = config.skip_shape = {6, 4};
      const auto nodes = BuildNorm(fixture.builder, config);
      if (variant == 7) {
        ONNX_NAMESPACE::TensorShapeProto shape = *nodes.x->Shape();
        shape.mutable_dim(0)->set_dim_param("batch");
        nodes.x->SetShape(shape);
        nodes.skip->SetShape(shape);
      }
      if (variant == 6 || config.outputs == 3) {
        for (auto& node : fixture.graph.Nodes()) node.SetExecutionProviderType(kCpuExecutionProvider);
      }
      CheckFixture(fixture, rms ? "SkipSimplifiedLayerNormalization" : "SkipLayerNormalization", true, true);
    }
  }
}

TEST(MobiusNormalizationRulesTests, WideHighOffsetRowsPreserveNormalizationAndLiveStatistics) {
  for (bool rms : {false, true}) {
    for (int64_t width : {8, 16, 33, 256}) {
      for (bool beta : {false, true}) {
        for (bool statistics : {false, true}) {
          if (rms && statistics) continue;
          SCOPED_TRACE(rms);
          SCOPED_TRACE(width);
          SCOPED_TRACE(beta);
          SCOPED_TRACE(statistics);
          Fixture fixture;
          NormConfig config;
          config.rms = rms;
          config.beta = beta;
          config.outputs = statistics ? 3 : 1;
          config.sum_output = config.sum_consumer = statistics;
          config.shape = config.skip_shape = {2, width};
          config.scale_width = config.beta_width = width;
          auto nodes = BuildNorm(fixture.builder, config);
          auto* input = fixture.builder.feeds_.at(nodes.x->Name()).GetMutable<Tensor>()->MutableData<float>();
          auto* skip = fixture.builder.feeds_.at(nodes.skip->Name()).GetMutable<Tensor>()->MutableData<float>();
          for (int64_t row = 0; row < 2; ++row) {
            for (int64_t i = 0; i < width; ++i) {
              input[row * width + i] = 100000000.0f + 8.0f * static_cast<float>(i);
              skip[row * width + i] = 0.0f;
            }
          }
          nodes.norm->MutableInputDefs()[1] = fixture.builder.MakeInitializer<float>(
              {width}, std::vector<float>(static_cast<size_t>(width), 1.0f));
          if (!rms && beta) {
            nodes.norm->MutableInputDefs()[2] = fixture.builder.MakeInitializer<float>(
                {width}, std::vector<float>(static_cast<size_t>(width), 0.0f));
          }
          for (auto& node : fixture.graph.Nodes()) node.SetExecutionProviderType(kCpuExecutionProvider);
          CheckFixture(fixture, rms ? "SkipSimplifiedLayerNormalization" : "SkipLayerNormalization", true, true);
        }
      }
    }
  }
}

TEST(MobiusNormalizationRulesTests, RejectsUnprovedResidualAndUnsupportedNormalization) {
  for (bool rms : {false, true}) {
    for (int variant = 0; variant < 17; ++variant) {
      SCOPED_TRACE(rms);
      SCOPED_TRACE(variant);
      Fixture fixture(variant == 12 ? 25 : 24);
      NormConfig config;
      config.rms = rms;
      if (variant == 0) config.skip_shape = {1, 3, 4};
      if (variant == 1) config.skip_shape = {4};
      if (variant == 2) config.shape = config.skip_shape = {1, 2, 3, 4};
      if (variant == 3) config.scale_width = 1;
      if (variant == 4 && !rms) config.beta_width = 1;
      const auto nodes = BuildNorm(fixture.builder, config);
      if (variant == 4 && rms) nodes.norm->AddAttribute("epsilon", std::numeric_limits<float>::infinity());
      if (variant == 5) nodes.norm->AddAttribute("axis", int64_t{-2});
      if (variant == 6) nodes.norm->AddAttribute("stash_type", int64_t{10});
      if (variant == 7) nodes.norm->AddAttribute("epsilon", -1e-5f);
      if (variant == 8) nodes.norm->AddAttribute("epsilon", std::numeric_limits<float>::quiet_NaN());
      if (variant == 9) {
        ONNX_NAMESPACE::TensorShapeProto shape = *nodes.x->Shape();
        shape.mutable_dim(0)->Clear();
        nodes.x->SetShape(shape);
        nodes.skip->SetShape(shape);
      }
      if (variant == 10) nodes.x->ClearShape();
      if (variant == 11) {
        for (auto& node : fixture.graph.Nodes()) node.SetExecutionProviderType(kCudaExecutionProvider);
      }
      if (variant == 14) nodes.norm->SetExecutionProviderType(kCpuExecutionProvider);
      if (variant == 13) nodes.norm->AddAttribute("epsilon", std::numeric_limits<float>::infinity());
      if (variant == 15) nodes.norm->AddAttribute("epsilon", 0.0f);
      if (variant == 16) {
        ONNX_NAMESPACE::TensorShapeProto shape = *nodes.x->Shape();
        shape.mutable_dim(0)->set_dim_param("batch_a");
        nodes.x->SetShape(shape);
        shape.mutable_dim(0)->set_dim_param("batch_b");
        nodes.skip->SetShape(shape);
      }
      CheckFixture(fixture, rms ? "SkipSimplifiedLayerNormalization" : "SkipLayerNormalization", false);
    }
    Fixture doubles;
    NormConfig config;
    config.rms = rms;
    BuildNorm<double>(doubles.builder, config);
    CheckFixture(doubles, rms ? "SkipSimplifiedLayerNormalization" : "SkipLayerNormalization", false);
  }
}

TEST(MobiusNormalizationRulesTests, MobiusRmsDummyOutputsRemainConstants) {
  Fixture fixture;
  NormConfig config;
  config.rms = true;
  BuildNorm(fixture.builder, config);
  for (int i = 0; i < 2; ++i) {
    auto& dummy = fixture.builder.AddNode("Constant", {}, {fixture.builder.MakeOutput()});
    dummy.AddAttribute("value_floats", std::vector<float>{0.0f});
  }
  CheckFixture(fixture, "SkipSimplifiedLayerNormalization", true, true);
}

TEST(MobiusNormalizationRulesTests, RejectsTwoOutputLayerNormInsteadOfDroppingStatistics) {
  Fixture fixture;
  NormConfig config;
  config.outputs = 2;
  BuildNorm(fixture.builder, config);
  CheckFixture(fixture, "SkipLayerNormalization", false);
}

TEST(MobiusNormalizationRulesTests, RejectsUnassignedStatisticsThatCouldBePlacedOnCuda) {
  for (bool graph_output : {false, true}) {
    Fixture fixture;
    NormConfig config;
    config.outputs = 3;
    config.stats_output = graph_output;
    config.stats_consumer = !graph_output;
    BuildNorm(fixture.builder, config);
    CheckFixture(fixture, "SkipLayerNormalization", false);
  }
}

TEST(MobiusNormalizationRulesTests, UnassignedMobiusLayerNormWithUnusedStatistics) {
  Fixture fixture;
  NormConfig config;
  config.outputs = 3;
  config.stats_output = false;
  BuildNorm(fixture.builder, config);
  CheckFixture(fixture, "SkipLayerNormalization", true, true);
}

TEST(MobiusNormalizationRulesTests, ExactBiasGeluBothAddOrdersAndDefaultAttribute) {
  for (bool first : {false, true}) {
    for (bool defaults : {false, true}) {
      Fixture fixture(defaults ? 20 : 24);
      GeluConfig config;
      config.bias_first = first;
      config.defaults = defaults;
      if (defaults) config.shape = {4};
      BuildGelu(fixture.builder, config);
      if (first && defaults) {
        for (auto& node : fixture.graph.Nodes()) node.SetExecutionProviderType(kCpuExecutionProvider);
      }
      CheckFixture(fixture, "BiasGelu", true, true);
    }
  }
}

TEST(MobiusNormalizationRulesTests, BiasGeluRejectsApproximationBroadcastsAndBoundaries) {
  for (int variant = 0; variant < 10; ++variant) {
    SCOPED_TRACE(variant);
    Fixture fixture(variant == 8 ? 25 : 24);
    GeluConfig config;
    if (variant == 0) config.approximate = "tanh";
    if (variant == 1) config.bias_shape = {1};
    if (variant == 2) config.bias_shape = {1, 4};
    if (variant == 3) config.bias_shape = {};
    if (variant == 4) config.sum_output = true;
    if (variant == 5) config.sum_consumer = true;
    if (variant == 9) config.bias_shape = config.shape;
    BuildGelu(fixture.builder, config);
    if (variant == 6) {
      for (auto& node : fixture.graph.Nodes()) node.SetExecutionProviderType(kCudaExecutionProvider);
    }
    if (variant == 7) fixture.graph.GetNode(0)->MutableInputDefs()[0]->ClearShape();
    CheckFixture(fixture, "BiasGelu", false);
  }
  Fixture doubles;
  BuildGelu<double>(doubles.builder);
  CheckFixture(doubles, "BiasGelu", false);
}

TEST(MobiusNormalizationRulesTests, MissingContribImportIsNotAddedByRewriter) {
  // Model's protobuf-loading constructor adds registered imports. Test this
  // pre-load graph directly instead of manufacturing a missing-import reload.
  for (int variant = 0; variant < 3; ++variant) {
    Fixture fixture(24, false);
    if (variant == 2) {
      BuildGelu(fixture.builder);
    } else {
      NormConfig config;
      config.rms = variant == 1;
      BuildNorm(fixture.builder, config);
    }
    fixture.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(fixture.graph.Resolve());
    ASSERT_EQ(fixture.graph.DomainToVersionMap().count(kMSDomain), 0u);
    for (bool indexed : {false, true}) {
      Options options;
      options.indexed = indexed;
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter(MakeMobiusNormalizationRules(), options).Run(fixture.graph, stats));
      EXPECT_EQ(stats.rewrites, 0u);
      EXPECT_EQ(fixture.graph.DomainToVersionMap().count(kMSDomain), 0u);
    }
  }
}
#endif
}  // namespace test
}  // namespace onnxruntime
