// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/pattern_rewriter.h"

#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
#ifndef DISABLE_CONTRIB_OPS
namespace {
size_t Count(const Graph& graph, const std::string& op, const std::string& domain) {
  size_t count = 0;
  for (const auto& node : graph.Nodes()) count += node.OpType() == op && node.Domain() == domain;
  return count;
}

void CheckGroupedModel(const PathString& path, int64_t k = 2, int64_t normalize = 1) {
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
  CheckGroupedModel(ORT_TSTR("testdata/transform/fusion/mobius_moe_topk.onnx"));
}

TEST(MoEFusionRulesTests, GroupedSoftmaxTopKNormalized) {
  CheckGroupedModel(ORT_TSTR("testdata/transform/fusion/mobius_moe_softmax_topk_normalized.onnx"));
}

TEST(MoEFusionRulesTests, GroupedSoftmaxTopKDifferentExpertCount) {
  CheckGroupedModel(ORT_TSTR("testdata/transform/fusion/mobius_moe_softmax_topk.onnx"), 1, 0);
}

TEST(MoEFusionRulesTests, RejectsUnsupportedVariantsAndBoundaries) {
  for (int scenario = 0; scenario < 5; ++scenario) {
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
#endif
}  // namespace onnxruntime::test
