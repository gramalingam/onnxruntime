// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/pattern_rewriter.h"

#include <array>

#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/initializer.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
#ifndef DISABLE_CONTRIB_OPS
namespace {
size_t Count(const Graph& graph, const std::string& op, const std::string& domain) {
  size_t count = 0;
  for (const auto& node : graph.Nodes()) count += node.OpType() == op && node.Domain() == domain;
  return count;
}

void CheckGroupedModel(const PathString& path, bool indexed, int64_t k = 2, int64_t normalize = 1) {
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(path, model, nullptr, DefaultLoggingManager().DefaultLogger()));
  auto& graph = model->MainGraph();
  ASSERT_EQ(Count(graph, "MoE", kMSDomain), 0u);
  ASSERT_EQ(Count(graph, "GroupedMatMul", "local.moe"), 3u);
  const NodeArg* logits = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "MatMul" && node.Domain() == kOnnxDomain) logits = node.OutputDefs()[0];
  }
  ASSERT_NE(logits, nullptr);
  rewrite::Statistics stats;
  rewrite::Options options;
  options.indexed = indexed;
  std::string diagnostics;
  options.on_failure = [&diagnostics](const rewrite::MatchFailure& failure) {
    if (failure.stage != rewrite::FailureStage::Structure)
      diagnostics += failure.rule + ": " + failure.reason + "\n";
  };
  rewrite::PatternRewriter rewriter(rewrite::MakeMoEFusionRules(), options);
  ASSERT_STATUS_OK(rewriter.Run(graph, stats));
  ASSERT_EQ(stats.rewrites, 1u) << diagnostics;
  ASSERT_STATUS_OK(graph.Resolve());
  EXPECT_EQ(Count(graph, "GroupedMatMul", "local.moe"), 0u);
  ASSERT_EQ(Count(graph, "MoE", kMSDomain), 1u);
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() != "MoE") continue;
    EXPECT_EQ(node.GetAttributes().at("k").i(), k);
    EXPECT_EQ(node.GetAttributes().at("activation_type").s(), "swiglu");
    EXPECT_EQ(node.GetAttributes().at("swiglu_fusion").i(), 1);
    EXPECT_EQ(node.GetAttributes().at("normalize_routing_weights").i(), normalize);
    EXPECT_EQ(node.InputDefs()[1], logits);
    ASSERT_NE(node.OutputDefs()[0]->Shape(), nullptr);
    EXPECT_EQ(node.OutputDefs()[0]->Shape()->dim(1).dim_value(), 8);
  }
  ASSERT_STATUS_OK(rewriter.Run(graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
  std::shared_ptr<Model> reloaded;
  ASSERT_STATUS_OK(Model::Load(model->ToProto(), reloaded, nullptr, DefaultLoggingManager().DefaultLogger()));
  EXPECT_EQ(Count(reloaded->MainGraph(), "MoE", kMSDomain), 1u);
  EXPECT_EQ(Count(reloaded->MainGraph(), "GroupedMatMul", "local.moe"), 0u);
}
}  // namespace

TEST(MoEFusionRulesTests, GroupedTopKSoftmax) {
  for (bool indexed : {true, false})
    CheckGroupedModel(ORT_TSTR("testdata/transform/fusion/mobius_moe_topk.onnx"), indexed);
}

TEST(MoEFusionRulesTests, GroupedSoftmaxTopKNormalized) {
  for (bool indexed : {true, false})
    CheckGroupedModel(ORT_TSTR("testdata/transform/fusion/mobius_moe_softmax_topk_normalized.onnx"), indexed);
}

TEST(MoEFusionRulesTests, GroupedSoftmaxTopKDifferentExpertCount) {
  for (bool indexed : {true, false})
    CheckGroupedModel(ORT_TSTR("testdata/transform/fusion/mobius_moe_softmax_topk.onnx"), indexed, 1, 0);
}

TEST(MoEFusionRulesTests, PacksInterleavedCPUWeights) {
  constexpr int64_t experts = 3, hidden = 8, intermediate = 12;
  for (bool indexed : {true, false}) {
    SCOPED_TRACE(indexed);
    std::shared_ptr<Model> model;
    const auto& logger = DefaultLoggingManager().DefaultLogger();
    ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/transform/fusion/mobius_moe_topk.onnx"),
                                 model, nullptr, logger));
    auto& graph = model->MainGraph();
    auto replace_weights = [&](const char* name, const std::array<int64_t, 3>& dims, float base) {
      ONNX_NAMESPACE::TensorProto tensor;
      tensor.set_name(name);
      tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      for (auto dim : dims) tensor.add_dims(dim);
      for (int64_t i = 0; i < experts * hidden * intermediate; ++i)
        tensor.add_float_data(base + static_cast<float>(i));
      graph.RemoveInitializedTensor(name);
      graph.AddInitializedTensor(tensor);
    };
    replace_weights("expert_gate", {experts, hidden, intermediate}, 1000.0f);
    replace_weights("expert_up", {experts, hidden, intermediate}, 2000.0f);
    replace_weights("expert_down", {experts, intermediate, hidden}, 3000.0f);
    ASSERT_STATUS_OK(graph.Resolve());
    rewrite::Statistics stats;
    ASSERT_STATUS_OK(rewrite::PatternRewriter(rewrite::MakeMoEFusionRules(), {.indexed = indexed}).Run(graph, stats));
    ASSERT_EQ(stats.rewrites, 1u);
    ASSERT_STATUS_OK(graph.Resolve());

    auto cpu = DefaultCpuExecutionProvider();
    const ConfigOptions config;
    ConstantFolding folding(*cpu, false, config);
    bool modified = false;
    ASSERT_STATUS_OK(folding.Apply(graph, modified, logger));
    ASSERT_TRUE(modified);
    ASSERT_STATUS_OK(graph.Resolve());
    ASSERT_EQ(Count(graph, "MoE", kMSDomain), 1u);
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() != "MoE" || node.Domain() != kMSDomain) continue;
      const auto* fc1 = node.InputDefs()[2];
      const auto* fc2 = node.InputDefs()[4];
      ASSERT_NE(fc1->Shape(), nullptr);
      ASSERT_NE(fc2->Shape(), nullptr);
      ASSERT_EQ(fc1->Shape()->dim_size(), 3);
      ASSERT_EQ(fc2->Shape()->dim_size(), 3);
      EXPECT_EQ(fc1->Shape()->dim(0).dim_value(), experts);
      EXPECT_EQ(fc1->Shape()->dim(1).dim_value(), 2 * intermediate);
      EXPECT_EQ(fc1->Shape()->dim(2).dim_value(), hidden);
      EXPECT_EQ(fc2->Shape()->dim(0).dim_value(), experts);
      EXPECT_EQ(fc2->Shape()->dim(1).dim_value(), hidden);
      EXPECT_EQ(fc2->Shape()->dim(2).dim_value(), intermediate);
      const auto* fc1_tensor = graph.GetInitializer(fc1->Name(), false);
      const auto* fc2_tensor = graph.GetInitializer(fc2->Name(), false);
      ASSERT_NE(fc1_tensor, nullptr);
      ASSERT_NE(fc2_tensor, nullptr);
      Initializer fc1_data(graph, *fc1_tensor, graph.ModelPath());
      Initializer fc2_data(graph, *fc2_tensor, graph.ModelPath());
      ASSERT_EQ(fc1_data.size(), experts * 2 * intermediate * hidden);
      ASSERT_EQ(fc2_data.size(), experts * hidden * intermediate);
      const auto* packed = fc1_data.data<float>();
      const auto* projected = fc2_data.data<float>();
      for (int64_t e = 0; e < experts; ++e) {
        for (int64_t i = 0; i < intermediate; ++i) {
          for (int64_t h = 0; h < hidden; ++h) {
            const auto source = static_cast<float>((e * hidden + h) * intermediate + i);
            EXPECT_EQ(packed[(e * 2 * intermediate + 2 * i) * hidden + h], 1000.0f + source);
            EXPECT_EQ(packed[(e * 2 * intermediate + 2 * i + 1) * hidden + h], 2000.0f + source);
            EXPECT_EQ(projected[(e * hidden + h) * intermediate + i],
                      3000.0f + static_cast<float>((e * intermediate + i) * hidden + h));
          }
        }
      }
    }
  }
}

TEST(MoEFusionRulesTests, RejectsUnsupportedVariantsAndBoundaries) {
  for (int scenario = 0; scenario < 8; ++scenario) {
    SCOPED_TRACE(scenario);
    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/transform/fusion/mobius_moe_topk.onnx"),
                                 model, nullptr, DefaultLoggingManager().DefaultLogger()));
    auto& graph = model->MainGraph();
    bool changed = false;
    for (auto& node : graph.Nodes()) {
      if (scenario == 0 && node.OpType() == "Swish") {
        node.AddAttribute("alpha", 2.0f);
        changed = true;
        break;
      }
      if (scenario == 1 && node.OpType() == "TopK") {
        node.AddAttribute("largest", int64_t{0});
        changed = true;
        break;
      }
      if (scenario == 2 && node.OpType() == "GroupedMatMul") {
        auto outputs = graph.GetOutputs();
        outputs.push_back(node.OutputDefs()[0]);
        graph.SetOutputs(outputs);
        changed = true;
        break;
      }
      if (scenario == 3 && node.OpType() == "GroupedMatMul") {
        node.SetExecutionProviderType(kCudaExecutionProvider);
        changed = true;
        break;
      }
      if (scenario == 4 && node.OpType() == "Reshape" &&
          node.OutputDefs()[0]->Shape()->dim_size() == 3) {
        ONNX_NAMESPACE::TensorProto shape;
        shape.set_name("noncanonical_shape");
        shape.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
        shape.add_dims(3);
        for (int64_t dim : {6, 2, 8}) shape.add_int64_data(dim);
        graph.AddInitializedTensor(shape);
        auto& value = graph.GetOrCreateNodeArg(shape.name(), node.InputDefs()[1]->TypeAsProto());
        node.MutableInputDefs()[1] = &value;
        changed = true;
        break;
      }
      if (scenario == 5 && node.OpType() == "TopK") {
        auto inputs = graph.GetInputsIncludingInitializers();
        inputs.push_back(node.InputDefs()[1]);
        graph.SetInputs(inputs);
        changed = true;
        break;
      }
      if (scenario == 6 && node.OpType() == "Swish") {
        auto& exposed = graph.GetOrCreateNodeArg("exposed", node.OutputDefs()[0]->TypeAsProto());
        graph.AddNode("consumer", "Identity", "", {node.MutableOutputDefs()[0]}, {&exposed});
        auto outputs = graph.GetOutputs();
        outputs.push_back(&exposed);
        graph.SetOutputs(outputs);
        changed = true;
        break;
      }
      if (scenario == 7 && node.OpType() == "Reshape" &&
          node.OutputDefs()[0]->Shape()->dim_size() == 2 &&
          node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type() ==
              ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        ONNX_NAMESPACE::TensorProto shape;
        shape.set_name("noncanonical_flat_shape");
        shape.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
        shape.add_dims(2);
        shape.add_int64_data(12);
        shape.add_int64_data(12);
        graph.AddInitializedTensor(shape);
        auto& value = graph.GetOrCreateNodeArg(shape.name(), node.InputDefs()[1]->TypeAsProto());
        node.MutableInputDefs()[1] = &value;
        changed = true;
        break;
      }
    }
    ASSERT_TRUE(changed);
    ASSERT_STATUS_OK(graph.Resolve());
    for (bool indexed : {true, false}) {
      rewrite::Statistics stats;
      ASSERT_STATUS_OK(rewrite::PatternRewriter(rewrite::MakeMoEFusionRules(), {.indexed = indexed}).Run(graph, stats));
      EXPECT_EQ(stats.rewrites, 0u);
      EXPECT_EQ(Count(graph, "GroupedMatMul", "local.moe"), 3u);
      EXPECT_EQ(Count(graph, "MoE", kMSDomain), 0u);
    }
  }
}

TEST(MoEFusionRulesTests, ExplicitRoutingOutputBindings) {
  const std::array<PathString, 3> paths{
      ORT_TSTR("testdata/transform/fusion/mobius_moe_topk.onnx"),
      ORT_TSTR("testdata/transform/fusion/mobius_moe_softmax_topk.onnx"),
      ORT_TSTR("testdata/transform/fusion/mobius_moe_softmax_topk_normalized.onnx")};
  const std::array<std::string_view, 3> producers{"Softmax", "TopK", "Div"};
  for (size_t i = 0; i < paths.size(); ++i) {
    for (bool indexed : {true, false}) {
      std::shared_ptr<Model> model;
      ASSERT_STATUS_OK(Model::Load(paths[i], model, nullptr, DefaultLoggingManager().DefaultLogger()));
      auto rules = rewrite::MakeMoEFusionRules();
      ASSERT_EQ(rules.size(), 1u);
      auto check = rules[0].check_and_bind;
      size_t callbacks = 0;
      rules[0].check_and_bind = [&](const Graph& graph, rewrite::Match& match) {
        ++callbacks;
        EXPECT_EQ(match.bindings().definition(), "MoEPattern");
        const auto* weights = match.valueOf("weights");
        const auto* indices = match.valueOf("indices");
        EXPECT_EQ(graph.GetProducerNode(weights->Name())->OpType(), producers[i]);
        for (const auto& node : graph.Nodes()) {
          if (node.OpType() == "TopK") {
            EXPECT_EQ(node.InputDefs()[1], match.valueOf("top_k"));
            EXPECT_EQ(node.OutputDefs()[1], indices);
            if (i == 1) EXPECT_EQ(node.OutputDefs()[0], weights);
          }
        }
        EXPECT_EQ(indices, match.instance("experts").valueOf("indices"));
        for (const auto* port : {"gate", "up", "down", "flat_shape", "output_shape"})
          EXPECT_EQ(match.instance("experts").valueOf(port), match.valueOf(port));
        const auto activation = match.instance("experts").instance("activation");
        EXPECT_EQ(activation.definition(), "SwiGLU");
        EXPECT_NE(activation.valueOf("product"), nullptr);
        return check(graph, match);
      };
      rewrite::Statistics stats;
      ASSERT_STATUS_OK(rewrite::PatternRewriter(std::move(rules), {.match_only = true, .indexed = indexed})
                           .Run(model->MainGraph(), stats));
      EXPECT_EQ(callbacks, 1u);
      EXPECT_EQ(stats.matches, 1u);
      EXPECT_EQ(stats.rewrites, 0u);
    }
  }
}

TEST(MoEFusionRulesTests, RejectsUnprovenDimensions) {
  for (bool indexed : {true, false}) {
    for (const auto* parameter : {"hidden", "expert_up", "expert_down"}) {
      SCOPED_TRACE(parameter);
      std::shared_ptr<Model> model;
      ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/transform/fusion/mobius_moe_topk.onnx"),
                                   model, nullptr, DefaultLoggingManager().DefaultLogger()));
      auto& graph = model->MainGraph();
      auto* value = graph.GetNodeArg(parameter);
      ASSERT_NE(value, nullptr);
      auto shape = *value->Shape();
      shape.mutable_dim(0)->Clear();
      value->SetShape(shape);
      rewrite::Statistics stats;
      ASSERT_STATUS_OK(rewrite::PatternRewriter(rewrite::MakeMoEFusionRules(), {.indexed = indexed})
                           .Run(graph, stats));
      EXPECT_EQ(stats.check_and_bind_calls, 1u);
      EXPECT_EQ(stats.rewrites, 0u);
      EXPECT_EQ(Count(graph, "GroupedMatMul", "local.moe"), 3u);
    }
  }
}

TEST(MoEFusionRulesTests, AcceptsSharedSymbolicTokenMetadata) {
  for (bool indexed : {true, false}) {
    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/transform/fusion/mobius_moe_topk.onnx"),
                                 model, nullptr, DefaultLoggingManager().DefaultLogger()));
    auto& graph = model->MainGraph();
    auto make_symbolic = [](NodeArg* value) {
      const auto* original = value->Shape();
      if (original && original->dim_size() > 0 && original->dim(0).has_dim_value() &&
          original->dim(0).dim_value() == 6) {
        auto shape = *original;
        shape.mutable_dim(0)->set_dim_param("tokens");
        value->SetShape(shape);
      }
    };
    for (auto& node : graph.Nodes()) {
      for (auto* value : node.MutableInputDefs()) make_symbolic(value);
      for (auto* value : node.MutableOutputDefs()) make_symbolic(value);
    }
    rewrite::Statistics stats;
    ASSERT_STATUS_OK(rewrite::PatternRewriter(rewrite::MakeMoEFusionRules(), {.indexed = indexed}).Run(graph, stats));
    EXPECT_EQ(stats.rewrites, 1u);
    ASSERT_STATUS_OK(graph.Resolve());
    ASSERT_EQ(graph.GetOutputs().size(), 1u);
    EXPECT_EQ(graph.GetOutputs()[0]->Shape()->dim(0).dim_param(), "tokens");
  }
}

TEST(MoEFusionRulesTests, RejectsUnknownRankBeforeSemanticChecks) {
  for (bool indexed : {true, false}) {
    for (const auto* target : {"hidden", "logits", "indices", "expert_gate"}) {
      SCOPED_TRACE(target);
      std::shared_ptr<Model> model;
      ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/transform/fusion/mobius_moe_topk.onnx"),
                                   model, nullptr, DefaultLoggingManager().DefaultLogger()));
      auto& graph = model->MainGraph();
      NodeArg* value = nullptr;
      if (std::string_view(target) == "logits" || std::string_view(target) == "indices") {
        for (auto& node : graph.Nodes()) {
          if (node.OpType() != "TopK") continue;
          value = std::string_view(target) == "logits" ? node.MutableInputDefs()[0] : node.MutableOutputDefs()[1];
        }
      } else {
        value = graph.GetNodeArg(target);
      }
      ASSERT_NE(value, nullptr);
      value->ClearShape();
      ASSERT_EQ(value->Shape(), nullptr);
      rewrite::Statistics stats;
      ASSERT_STATUS_OK(rewrite::PatternRewriter(rewrite::MakeMoEFusionRules(), {.indexed = indexed}).Run(graph, stats));
      EXPECT_EQ(stats.check_and_bind_calls, 0u);
      EXPECT_EQ(stats.rewrites, 0u);
      EXPECT_EQ(Count(graph, "GroupedMatMul", "local.moe"), 3u);
      EXPECT_EQ(Count(graph, "MoE", kMSDomain), 0u);
    }
  }
}

TEST(MoEFusionRulesTests, RejectsMismatchedTopKAndPropagatesMalformedConstants) {
  for (bool indexed : {true, false}) {
    for (bool malformed : {true, false}) {
      std::shared_ptr<Model> model;
      ASSERT_STATUS_OK(Model::Load(ORT_TSTR("testdata/transform/fusion/mobius_moe_topk.onnx"),
                                   model, nullptr, DefaultLoggingManager().DefaultLogger()));
      auto& graph = model->MainGraph();
      std::shared_ptr<Model> constants_model;
      ASSERT_STATUS_OK(Model::Load(model->ToProto(), constants_model, nullptr,
                                   DefaultLoggingManager().DefaultLogger()));
      auto& constants = constants_model->MainGraph();
      const NodeArg* top_k = nullptr;
      for (const auto& node : graph.Nodes()) {
        if (node.OpType() == "TopK") top_k = node.InputDefs()[1];
      }
      ASSERT_NE(top_k, nullptr);
      ONNX_NAMESPACE::TensorProto tensor;
      tensor.set_name(top_k->Name());
      tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
      tensor.add_dims(1);
      if (malformed)
        tensor.set_raw_data("bad");
      else
        tensor.add_int64_data(1);
      constants.RemoveInitializedTensor(top_k->Name());
      constants.AddInitializedTensor(tensor);
      auto rules = rewrite::MakeMoEFusionRules();
      const auto check = rules[0].check_and_bind;
      size_t callbacks = 0;
      // Supply damaged payloads to the checker without re-running ONNX TopK inference on them.
      rules[0].check_and_bind = [&](const Graph&, rewrite::Match& match) {
        ++callbacks;
        return check(constants, match);
      };
      rewrite::Statistics stats;
      const rewrite::PatternRewriter rewriter(std::move(rules), {.match_only = true, .indexed = indexed});
      if (malformed) {
        EXPECT_THROW(ORT_THROW_IF_ERROR(rewriter.Run(graph, stats)), OnnxRuntimeException);
      } else {
        ASSERT_STATUS_OK(rewriter.Run(graph, stats));
      }
      EXPECT_EQ(callbacks, 1u);
      EXPECT_EQ(stats.matches, 0u);
      EXPECT_EQ(stats.rewrites, 0u);
      EXPECT_EQ(Count(graph, "GroupedMatMul", "local.moe"), 3u);
      EXPECT_EQ(Count(graph, "MoE", kMSDomain), 0u);
    }
  }
}
#endif
}  // namespace onnxruntime::test
