// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/pattern_rewriter.h"

#include <cmath>
#include <cstring>
#include <sstream>

#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime::test {
#ifndef DISABLE_CONTRIB_OPS
namespace {
const PathString kMatMulModels[] = {
    ORT_TSTR("testdata/transform/fusion/unpack_matmul_symmetric.onnx"),
    ORT_TSTR("testdata/transform/fusion/unpack_matmul_float_zp.onnx"),
    ORT_TSTR("testdata/transform/fusion/unpack_matmul_packed_zp.onnx")};
const PathString kMoEModels[] = {
    ORT_TSTR("testdata/transform/fusion/unpack_qmoe_topk.onnx"),
    ORT_TSTR("testdata/transform/fusion/unpack_qmoe_normalized.onnx"),
    ORT_TSTR("testdata/transform/fusion/unpack_qmoe_unnormalized.onnx")};

size_t Count(const Graph& graph, const char* op, const char* domain) {
  size_t count = 0;
  for (const auto& node : graph.Nodes()) count += node.OpType() == op && node.Domain() == domain;
  return count;
}

Status Load(const PathString& path, std::shared_ptr<Model>& model) {
  return Model::Load(path, model, nullptr, DefaultLoggingManager().DefaultLogger());
}

std::string Serialize(const Model& model) {
  std::string serialized;
  model.ToProto().SerializeToString(&serialized);
  return serialized;
}

NameMLValMap Feeds(const Graph& graph, int seed) {
  NameMLValMap feeds;
  for (const auto* input : graph.GetInputs()) {
    std::vector<int64_t> dims;
    int64_t size = 1;
    for (const auto& dim : input->Shape()->dim()) {
      const int64_t value = dim.has_dim_value() ? dim.dim_value() : 4;
      dims.push_back(value);
      size *= value;
    }
    std::vector<float> data(static_cast<size_t>(size));
    for (size_t i = 0; i < data.size(); ++i)
      data[i] = seed == 0 && input->Name() == "logits"
                    ? 0.0f
                    : std::sin(static_cast<float>(i * 7 + seed * 13)) * 0.7f;
    OrtValue value;
    CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], dims, data, &value);
    feeds.emplace(input->Name(), std::move(value));
  }
  return feeds;
}

Status Execute(const std::string& model, const NameMLValMap& feeds, std::vector<OrtValue>& outputs) {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.intra_op_param.thread_pool_size = 1;
  InferenceSessionWrapper session{options, GetEnvironment()};
  std::istringstream stream(model);
  ORT_RETURN_IF_ERROR(session.Load(stream));
  ORT_RETURN_IF_ERROR(session.Initialize());
  const InlinedVector<std::string> output_names{"output"};
  return session.Run(RunOptions{}, feeds, output_names, &outputs);
}

void CheckFusion(const PathString& path, bool moe, bool indexed, int variant, bool symbolic_tokens = false) {
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Load(path, model));
  if (symbolic_tokens) {
    auto proto = model->ToProto();
    proto.mutable_graph()->clear_value_info();
    for (auto& input : *proto.mutable_graph()->mutable_input())
      input.mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_param("tokens");
    proto.mutable_graph()->mutable_output(0)->mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->set_dim_param("tokens");
    ASSERT_STATUS_OK(Model::Load(proto, model, nullptr, DefaultLoggingManager().DefaultLogger()));
  }
  auto& graph = model->MainGraph();
  const auto baseline = Serialize(*model);
  rewrite::Statistics stats;
  rewrite::Options options;
  options.indexed = indexed;
  std::string diagnostics;
  options.on_failure = [&](const rewrite::MatchFailure& failure) {
    if (failure.stage != rewrite::FailureStage::Structure)
      diagnostics += failure.rule + ": " + failure.reason + "\n";
  };
  rewrite::PatternRewriter rewriter(rewrite::MakeQuantizedFusionRules(), options);
  ASSERT_STATUS_OK(rewriter.Run(graph, stats));
  ASSERT_EQ(stats.rewrites, 1u) << diagnostics;
  ASSERT_STATUS_OK(graph.Resolve());
  EXPECT_EQ(Count(graph, "Unpack", "local.quant"), 0u);
  EXPECT_EQ(Count(graph, "GroupedMatMul", "local.moe"), 0u);
  const char* op = moe ? "QMoE" : "MatMulNBits";
  ASSERT_EQ(Count(graph, op, kMSDomain), 1u);
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() != op || node.Domain() != kMSDomain) continue;
    EXPECT_EQ(node.GetAttributes().at("accuracy_level").i(), 0);
    EXPECT_EQ(node.InputDefs()[0]->Name(), "input");
    if (moe) {
      EXPECT_EQ(node.GetAttributes().at("expert_weight_bits").i(), 4);
      EXPECT_EQ(node.GetAttributes().at("weights_prepacked").i(), 0);
      EXPECT_EQ(node.GetAttributes().at("swiglu_fusion").i(), 1);
      EXPECT_EQ(node.GetAttributes().at("k").i(), variant == 2 ? 1 : 2);
      EXPECT_EQ(node.GetAttributes().at("normalize_routing_weights").i(), 0);
      ASSERT_TRUE(node.InputDefs()[14]->Exists());
      EXPECT_EQ(node.InputDefs()[5]->Name(), "dpacked");
      EXPECT_EQ(node.InputDefs()[6]->Name(), "dscale");
    } else {
      EXPECT_EQ(node.GetAttributes().at("bits").i(), 4);
      EXPECT_EQ(node.GetAttributes().at("K").i(), variant == 1 ? 64 : 48);
      EXPECT_EQ(node.GetAttributes().at("N").i(), 5);
      EXPECT_EQ(node.InputDefs()[1]->Name(), "packed");
      EXPECT_EQ(node.InputDefs()[2]->Name(), "scales");
      if (variant != 0) EXPECT_EQ(node.InputDefs()[3]->Name(), "zero_points");
    }
    EXPECT_EQ(node.GetAttributes().at("block_size").i(), variant == 1 ? 32 : 16);
  }
  const auto fused = Serialize(*model);
  std::shared_ptr<Model> reloaded;
  ASSERT_STATUS_OK(Model::Load(model->ToProto(), reloaded, nullptr, DefaultLoggingManager().DefaultLogger()));
  EXPECT_EQ(Count(reloaded->MainGraph(), op, kMSDomain), 1u);
  ASSERT_STATUS_OK(rewriter.Run(graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
  for (int seed : {0, 1, 2, 3}) {
    SCOPED_TRACE(seed);
    const auto feeds = Feeds(graph, seed);
    std::vector<OrtValue> expected, actual;
    ASSERT_STATUS_OK(Execute(baseline, feeds, expected));
    ASSERT_STATUS_OK(Execute(fused, feeds, actual));
    ASSERT_EQ(expected.size(), 1u);
    ASSERT_EQ(actual.size(), 1u);
    const auto& a = expected[0].Get<Tensor>();
    const auto& b = actual[0].Get<Tensor>();
    ASSERT_EQ(a.Shape(), b.Shape());
    for (int64_t i = 0; i < a.Shape().Size(); ++i) {
      ASSERT_TRUE(std::isfinite(a.Data<float>()[i]));
      ASSERT_TRUE(std::isfinite(b.Data<float>()[i]));
      EXPECT_NEAR(a.Data<float>()[i], b.Data<float>()[i], 2e-5f + 2e-4f * std::abs(a.Data<float>()[i]));
    }
  }
}

void RejectVariant(const PathString& path, bool moe, int scenario, bool indexed) {
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Load(path, model));
  auto proto = model->ToProto();
  if (scenario == 0) {
    for (auto& tensor : *proto.mutable_graph()->mutable_initializer()) {
      if (tensor.name() != (moe ? "goffset" : "woffset")) continue;
      tensor.clear_raw_data();
      tensor.add_float_data(7.0f);
    }
  } else if (scenario == 1) {
    for (auto& node : *proto.mutable_graph()->mutable_node()) {
      if (node.op_type() != "Unpack") continue;
      for (auto& attr : *node.mutable_attribute()) {
        if (attr.name() == "bits") attr.set_i(3);
      }
    }
  } else if (scenario == 2) {
    // Same shape and values, but not the canonical static shape spelling accepted by the rule.
    for (auto& tensor : *proto.mutable_graph()->mutable_initializer()) {
      if (tensor.name() != (moe ? "gblocked" : "wrows")) continue;
      const int64_t count = tensor.dims(0);
      std::vector<int64_t> values(static_cast<size_t>(count));
      std::memcpy(values.data(), tensor.raw_data().data(), values.size() * sizeof(int64_t));
      values[0] = -1;
      tensor.set_raw_data(values.data(), values.size() * sizeof(int64_t));
    }
  } else if (scenario == 3) {
    for (auto& import : *proto.mutable_opset_import()) {
      if (import.domain().empty()) import.set_version(20);
    }
  } else if (scenario == 7) {
    auto* input = proto.mutable_graph()->add_input();
    input->set_name("count");
    auto* type = input->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    type->mutable_shape();
    auto* initializers = proto.mutable_graph()->mutable_initializer();
    for (int i = 0; i < initializers->size(); ++i) {
      if (initializers->Get(i).name() == "count") {
        initializers->DeleteSubrange(i, 1);
        break;
      }
    }
  } else if (scenario == 8) {
    if (moe) {
      for (auto& input : *proto.mutable_graph()->mutable_input())
        input.mutable_type()->mutable_tensor_type()->mutable_shape()->mutable_dim(0)->Clear();
    } else {
      proto.mutable_graph()->clear_value_info();
      for (auto& tensor : *proto.mutable_graph()->mutable_initializer()) {
        if (tensor.name() == "packed") {
          tensor.set_dims(1, 2);
          tensor.set_dims(2, 12);
        } else if (tensor.name() == "scales") {
          tensor.set_dims(1, 2);
          tensor.mutable_raw_data()->resize(10 * sizeof(float));
        } else if (tensor.name() == "count") {
          const int64_t count = 24;
          tensor.set_raw_data(&count, sizeof(count));
        }
      }
    }
  }
  ASSERT_STATUS_OK(Model::Load(proto, model, nullptr, DefaultLoggingManager().DefaultLogger()));
  auto& graph = model->MainGraph();
  if (scenario == 4 || scenario == 5 || scenario == 6) {
    for (auto& node : graph.Nodes()) {
      if (node.OpType() != "Unpack") continue;
      if (scenario == 4) {
        auto outputs = graph.GetOutputs();
        outputs.push_back(node.OutputDefs()[0]);
        graph.SetOutputs(outputs);
      } else if (scenario == 5) {
        auto& outside = graph.GetOrCreateNodeArg("outside", node.OutputDefs()[0]->TypeAsProto());
        graph.AddNode("outside", "Identity", "external consumer",
                      {node.MutableOutputDefs()[0]}, {&outside});
        auto outputs = graph.GetOutputs();
        outputs.push_back(&outside);
        graph.SetOutputs(outputs);
      } else {
        node.SetExecutionProviderType(kCudaExecutionProvider);
      }
      break;
    }
    ASSERT_STATUS_OK(graph.Resolve());
  }
  const auto before = graph.NumberOfNodes();
  rewrite::Statistics stats;
  ASSERT_STATUS_OK(rewrite::PatternRewriter(rewrite::MakeQuantizedFusionRules(), {.indexed = indexed}).Run(graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
  EXPECT_EQ(graph.NumberOfNodes(), before);
  EXPECT_EQ(Count(graph, "MatMulNBits", kMSDomain), 0u);
  EXPECT_EQ(Count(graph, "QMoE", kMSDomain), 0u);
}
}  // namespace

TEST(QuantizedFusionRulesTests, MatMulNBitsNumericalAndRoundTrip) {
  for (bool indexed : {true, false}) {
    for (int i = 0; i < 3; ++i) {
      SCOPED_TRACE(i);
      SCOPED_TRACE(indexed);
      CheckFusion(kMatMulModels[i], false, indexed, i);
    }
  }
}

TEST(QuantizedFusionRulesTests, QMoENumericalAndRoundTrip) {
  for (bool indexed : {true, false}) {
    for (int i = 0; i < 3; ++i) {
      SCOPED_TRACE(i);
      SCOPED_TRACE(indexed);
      CheckFusion(kMoEModels[i], true, indexed, i);
    }
    CheckFusion(kMoEModels[0], true, indexed, 0, true);
  }
}

TEST(QuantizedFusionRulesTests, RejectsUnsupportedSemanticsAndBoundaries) {
  for (bool indexed : {true, false}) {
    for (bool moe : {true, false}) {
      for (int scenario = 0; scenario < 9; ++scenario) {
        SCOPED_TRACE(indexed);
        SCOPED_TRACE(moe);
        SCOPED_TRACE(scenario);
        RejectVariant(moe ? kMoEModels[0] : kMatMulModels[0], moe, scenario, indexed);
      }
    }
  }
}
#else
TEST(QuantizedFusionRulesTests, NoRulesWithoutContribOps) {
  EXPECT_TRUE(rewrite::MakeQuantizedFusionRules().empty());
}
#endif
}  // namespace onnxruntime::test
