// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

#include "core/optimizer/pattern_rewriter.h"
#include "core/platform/env.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {
using namespace rewrite;

struct TestGraph {
  Model model{"rewrite",
              false,
              ModelMetaData(),
              PathString(),
              IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 17}, {kMSDomain, 1}},
              {},
              DefaultLoggingManager().DefaultLogger()};
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder{graph};
};

void BuildHardSwish(ModelTestBuilder& b, bool mismatch = false, bool intermediate_output = false, float divisor = 6.0f,
                    bool external_consumer = false) {
  auto* x = b.MakeInput<float>({2, 4}, -5.0f, 5.0f);
  auto* y = mismatch ? b.MakeInput<float>({2, 4}, -5.0f, 5.0f) : x;
  auto* add = b.MakeIntermediate();
  auto* clip = intermediate_output ? b.MakeOutput() : b.MakeIntermediate();
  auto* mul = b.MakeIntermediate();
  b.AddNode("Add", {x, b.MakeScalarInitializer<float>(3.0f)}, {add});
  b.AddNode("Clip", {add, b.MakeScalarInitializer<float>(0.0f), b.MakeScalarInitializer<float>(6.0f)}, {clip});
  b.AddNode("Mul", {clip, y}, {mul});
  b.AddNode("Div", {mul, b.MakeScalarInitializer<float>(divisor)}, {b.MakeOutput()});
  if (external_consumer) b.AddNode("Identity", {clip}, {b.MakeOutput()});
}

void BuildRms(ModelTestBuilder& b, NodeArg* x, NodeArg* output, int64_t width, float epsilon = 1e-5f,
              bool wrong_axis = false) {
  auto* square = b.MakeIntermediate();
  auto* mean = b.MakeIntermediate();
  auto* add = b.MakeIntermediate();
  auto* sqrt = b.MakeIntermediate();
  auto* reciprocal = b.MakeIntermediate();
  auto* normalized = b.MakeIntermediate();
  b.AddNode("Pow", {x, b.MakeScalarInitializer<float>(2.0f)}, {square});
  auto& reduce = b.AddNode("ReduceMean", {square}, {mean});
  reduce.AddAttribute("axes", std::vector<int64_t>{wrong_axis ? -2 : -1});
  reduce.AddAttribute("keepdims", int64_t{1});
  b.AddNode("Add", {mean, b.MakeScalarInitializer<float>(epsilon)}, {add});
  b.AddNode("Sqrt", {add}, {sqrt});
  b.AddNode("Reciprocal", {sqrt}, {reciprocal});
  b.AddNode("Mul", {x, reciprocal}, {normalized});
  b.AddNode("Mul", {normalized, b.MakeInitializer<float>({width}, 0.8f, 1.2f)}, {output});
}

size_t Count(const Graph& graph, const std::string& op) {
  size_t count = 0;
  for (const auto& node : graph.Nodes()) count += node.OpType() == op;
  return count;
}

Rule IdentityPair(const char* name = "identity", bool accept = true) {
  Pattern p;
  auto x = p.Input();
  auto a = p.Op({"Identity", kOnnxDomain, {16}}, {x});
  p.Op({"Identity", kOnnxDomain, {16}}, {a});
  return {name, p, [accept](const Graph&, const Match&) { return accept; },
          [x](const Graph&, const Match&) { return Replacement{"Identity", kOnnxDomain, 16, {x}, {}}; }};
}

// Small-weight eager decoder: B=1, Hq=4, Hkv=2, D=4. Cache length is 3.
// See the design document for the model/export sources and deliberate simplifications.
void BuildDecoder(ModelTestBuilder& b, size_t layers, int64_t sequence = 1) {
  constexpr int64_t width = 16, kv_width = 8, head = 4, past = 3;
  auto* x = b.MakeInput<float>({1, sequence, width}, -0.5f, 0.5f);
  auto unary = [&](const char* op, NodeArg* input) {
    auto* output = b.MakeIntermediate();
    b.AddNode(op, {input}, {output});
    return output;
  };
  auto binary = [&](const char* op, NodeArg* a, NodeArg* c) {
    auto* output = b.MakeIntermediate();
    b.AddNode(op, {a, c}, {output});
    return output;
  };
  auto reshape = [&](NodeArg* input, std::vector<int64_t> shape) {
    auto* output = b.MakeIntermediate();
    auto rank = static_cast<int64_t>(shape.size());
    b.AddNode("Reshape", {input, b.MakeInitializer<int64_t>({rank}, shape)}, {output});
    return output;
  };
  auto transpose = [&](NodeArg* input, std::vector<int64_t> perm) {
    auto* output = b.MakeIntermediate();
    b.AddNode("Transpose", {input}, {output}).AddAttribute("perm", perm);
    return output;
  };
  auto projection = [&](NodeArg* input, int64_t in, int64_t out) {
    return binary("MatMul", input, b.MakeInitializer<float>({in, out}, -0.08f, 0.08f));
  };
  auto slice = [&](NodeArg* input, int64_t start, int64_t end) {
    auto* output = b.MakeIntermediate();
    b.AddNode("Slice",
              {input, b.MakeInitializer<int64_t>({1}, {start}), b.MakeInitializer<int64_t>({1}, {end}),
               b.MakeInitializer<int64_t>({1}, {3})},
              {output});
    return output;
  };
  auto concat = [&](NodeArg* a, NodeArg* c, int64_t axis, NodeArg* output) {
    b.AddNode("Concat", {a, c}, {output}).AddAttribute("axis", axis);
    return output;
  };
  auto rope = [&](NodeArg* input, int64_t heads) {
    auto* shaped = transpose(reshape(input, {1, sequence, heads, head}), {0, 2, 1, 3});
    auto* first = slice(shaped, 0, head / 2);
    auto* second = slice(shaped, head / 2, head);
    auto* rotated = concat(unary("Neg", second), first, 3, b.MakeIntermediate());
    // Nontrivial position-dependent cache entries, gathered at decode positions.
    std::vector<int64_t> positions;
    for (int64_t s = 0; s < sequence; ++s) positions.push_back(past + s);
    auto* indices = b.MakeInitializer<int64_t>({sequence}, positions);
    std::vector<float> cosine(32 * head), sine(32 * head);
    for (int64_t position = 0; position < 32; ++position) {
      for (int64_t d = 0; d < head; ++d) {
        float angle = static_cast<float>(position) * std::pow(10000.0f, -2.0f * (d % (head / 2)) / head);
        cosine[static_cast<size_t>(position * head + d)] = std::cos(angle);
        sine[static_cast<size_t>(position * head + d)] = std::sin(angle);
      }
    }
    auto* cos = binary("Gather", b.MakeInitializer<float>({32, head}, cosine), indices);
    auto* sin = binary("Gather", b.MakeInitializer<float>({32, head}, sine), indices);
    return binary("Add", binary("Mul", shaped, reshape(cos, {1, 1, sequence, head})),
                  binary("Mul", rotated, reshape(sin, {1, 1, sequence, head})));
  };
  auto repeat_kv = [&](NodeArg* input) {
    auto* expanded = b.MakeIntermediate();
    b.AddNode("Expand",
              {reshape(input, {1, 2, 1, past + sequence, head}),
               b.MakeInitializer<int64_t>({5}, {1, 2, 2, past + sequence, head})},
              {expanded});
    return reshape(expanded, {1, 4, past + sequence, head});
  };
  for (size_t layer = 0; layer < layers; ++layer) {
    auto* normalized = b.MakeIntermediate();
    BuildRms(b, x, normalized, width);
    auto* q = rope(projection(normalized, width, width), 4);
    auto* k = rope(projection(normalized, width, kv_width), 2);
    auto* v = transpose(reshape(projection(normalized, width, kv_width), {1, sequence, 2, head}), {0, 2, 1, 3});
    auto* present_k = concat(b.MakeInput<float>({1, 2, past, head}, -0.1f, 0.1f), k, 2, b.MakeOutput());
    auto* present_v = concat(b.MakeInput<float>({1, 2, past, head}, -0.1f, 0.1f), v, 2, b.MakeOutput());
    auto* scores = binary("MatMul", q, transpose(repeat_kv(present_k), {0, 1, 3, 2}));
    auto* scaled = binary("Div", scores, b.MakeScalarInitializer<float>(2.0f));
    std::vector<float> mask(static_cast<size_t>(sequence * (past + sequence)), 0.0f);
    for (int64_t s = 0; s < sequence; ++s) {
      for (int64_t t = past + s + 1; t < past + sequence; ++t) {
        mask[static_cast<size_t>(s * (past + sequence) + t)] = -10000.0f;
      }
    }
    auto* masked = binary("Add", scaled, b.MakeInitializer<float>({1, 1, sequence, past + sequence}, mask));
    auto* probabilities = b.MakeIntermediate();
    b.AddNode("Softmax", {masked}, {probabilities}).AddAttribute("axis", int64_t{-1});
    auto* attention = binary("MatMul", probabilities, repeat_kv(present_v));
    auto* context = reshape(transpose(attention, {0, 2, 1, 3}), {1, sequence, width});
    auto* residual = binary("Add", x, projection(context, width, width));
    auto* norm2 = b.MakeIntermediate();
    BuildRms(b, residual, norm2, width);
    auto* gate = projection(norm2, width, width * 4);
    auto* silu = binary("Mul", gate, unary("Sigmoid", gate));
    auto* mlp = binary("Mul", silu, projection(norm2, width, width * 4));
    x = binary("Add", residual, projection(mlp, width * 4, width));
  }
  b.AddNode("Identity", {x}, {b.MakeOutput()});
}
}  // namespace

TEST(PatternRewriterTests, HardSwishPositiveAndBoundaries) {
  for (int scenario = 0; scenario < 6; ++scenario) {
    TestGraph test;
    BuildHardSwish(test.builder, scenario == 1, scenario == 2, scenario == 3 ? 7.0f : 6.0f, scenario == 4);
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    if (scenario == 5) test.graph.Nodes().begin()->SetExecutionProviderType(kCudaExecutionProvider);
    PatternRewriter rewriter(MakeExampleRules());
    Statistics statistics;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, statistics));
    EXPECT_EQ(statistics.rewrites, scenario == 0 ? 1u : 0u);
    ASSERT_STATUS_OK(test.graph.Resolve());
    EXPECT_EQ(Count(test.graph, "HardSwish"), scenario == 0 ? 1u : 0u);
  }
}

TEST(PatternRewriterTests, NumericalHardSwish) {
  TransformerTester([](ModelTestBuilder& b) { BuildHardSwish(b); },
                    [](InferenceSessionWrapper& s) {
                      // CPU executes HardSwish's ONNX function body (HardSigmoid + Mul).
                      EXPECT_EQ(Count(s.GetGraph(), "HardSigmoid"), 1u);
                      EXPECT_EQ(Count(s.GetGraph(), "Clip"), 0u);
                    },
                    TransformerLevel::Default, TransformerLevel::Level1, 17, 1e-6, 1e-5,
                    std::make_unique<PatternRewriter>(MakeExampleRules()));
}

TEST(PatternRewriterTests, HardSigmoidAttributeGuard) {
  for (bool correct : {true, false}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2, 4}, -5.0f, 5.0f);
    auto* h = b.MakeIntermediate();
    auto& sigmoid = b.AddNode("HardSigmoid", {x}, {h});
    sigmoid.AddAttribute("alpha", correct ? 1.0f / 6.0f : 0.2f);
    sigmoid.AddAttribute("beta", 0.5f);
    b.AddNode("Mul", {h, x}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeExampleRules()).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, correct ? 1u : 0u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, PriorityOverlapTerminationAndMatchOnly) {
  for (bool indexed : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    for (int i = 0; i < 8; ++i) {
      auto* output = i == 7 ? b.MakeOutput() : b.MakeIntermediate();
      b.AddNode("Identity", {x}, {output});
      x = output;
    }
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    int first = 0, second = 0;
    Rule a = IdentityPair("first"), c = IdentityPair("second");
    a.guard = [&first](const Graph&, const Match&) {
      ++first;
      return true;
    };
    c.guard = [&second](const Graph&, const Match&) {
      ++second;
      return true;
    };
    PatternRewriter rewriter({a, c}, {.indexed = indexed});
    Statistics match_stats;
    ASSERT_STATUS_OK(PatternRewriter({a, c}, {.match_only = true, .indexed = indexed}).Run(test.graph, match_stats));
    EXPECT_EQ(test.graph.NumberOfNodes(), 8);
    EXPECT_EQ(match_stats.matches, 14u);
    first = second = 0;
    Statistics stats;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, 7u);
    EXPECT_EQ(first, 7);
    EXPECT_EQ(second, 0);
    ASSERT_STATUS_OK(test.graph.Resolve());
    Statistics repeated;
    ASSERT_STATUS_OK(rewriter.Run(test.graph, repeated));
    EXPECT_EQ(repeated.rewrites, 0u);
  }
}

TEST(PatternRewriterTests, InvalidRuleAndRecipe) {
  Pattern disconnected;
  auto x = disconnected.Input();
  disconnected.Op({"Identity", "", {16}}, {x});
  disconnected.Op({"Identity", "", {16}}, {x});
  auto rule = IdentityPair();
  rule.pattern = disconnected;
  EXPECT_THROW(PatternRewriter({rule}), OnnxRuntimeException);
  TestGraph test;
  auto* input = test.builder.MakeInput<float>({1}, -1.0f, 1.0f);
  auto* a = test.builder.MakeIntermediate();
  test.builder.AddNode("Identity", {input}, {a});
  test.builder.AddNode("Identity", {a}, {test.builder.MakeOutput()});
  test.builder.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  rule = IdentityPair();
  for (int scenario = 0; scenario < 4; ++scenario) {
    rule.replacement = [scenario](const Graph&, const Match&) {
      Replacement recipe{"Identity", "", 16, {0}, {}};
      if (scenario == 0) recipe.version = 999;
      if (scenario == 1) recipe.domain = "not.imported";
      if (scenario == 2) recipe.inputs = {1};  // Deleted intermediate, not an external input.
      if (scenario == 3) {
        ONNX_NAMESPACE::AttributeProto attr;
        attr.set_name("invalid");
        attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
        attr.set_i(1);
        recipe.attributes.emplace("invalid", attr);
      }
      return recipe;
    };
    Statistics stats;
    EXPECT_FALSE(PatternRewriter({rule}).Run(test.graph, stats).IsOK());
    EXPECT_EQ(test.graph.NumberOfNodes(), 2);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, PriorityAcrossInterleavedStructuralGroups) {
  for (bool indexed : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    for (int i = 0; i < 3; ++i) {
      auto* next = i == 2 ? b.MakeOutput() : b.MakeIntermediate();
      b.AddNode("Identity", {x}, {next});
      x = next;
    }
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto rejected = IdentityPair("rejected", false);
    auto lower_priority = IdentityPair("lower_priority");
    lower_priority.guard = [](const Graph& graph, const Match& m) {
      return graph.NodeProducesGraphOutput(*m.nodes.back());
    };
    Pattern p;
    auto input = p.Input();
    auto first = p.Op({"Identity", "", {16}}, {input});
    auto second = p.Op({"Identity", "", {16}}, {first});
    p.Op({"Identity", "", {16}}, {second});
    Rule higher_priority{"higher_priority", p, [](const Graph&, const Match&) { return true; },
                         [input](const Graph&, const Match&) { return Replacement{"Identity", "", 16, {input}, {}}; }};
    Statistics stats;
    ASSERT_STATUS_OK(
        PatternRewriter({rejected, higher_priority, lower_priority}, {.indexed = indexed}).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, 1u);
    EXPECT_EQ(stats.rewrites, 1u);
    EXPECT_EQ(test.graph.NumberOfNodes(), 1);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, LayerAnnotationsAndRootConsumers) {
  for (bool same_layer : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* first = b.MakeIntermediate();
    auto* root = b.MakeOutput();
    auto& a = b.AddNode("Identity", {x}, {first});
    auto& c = b.AddNode("Identity", {first}, {root});
    a.SetLayeringAnnotation(same_layer ? "layer0" : "layer1");
    c.SetLayeringAnnotation("layer0");
    b.AddNode("Neg", {root}, {b.MakeOutput()});
    b.AddNode("Abs", {root}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter({IdentityPair()}).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, same_layer ? 1u : 0u);
    ASSERT_STATUS_OK(test.graph.Resolve());
    const Node* producer = test.graph.GetProducerNode(root->Name());
    ASSERT_NE(producer, nullptr);
    EXPECT_EQ(producer->GetLayeringAnnotation(), "layer0");
    EXPECT_EQ(producer->GetOutputEdgesCount(), 2);
  }
}

TEST(PatternRewriterTests, MultiOutputAndVersionRejection) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
  auto* a = b.MakeIntermediate();
  b.AddNode("Identity", {x}, {a});
  b.AddNode("Dropout", {a}, {b.MakeOutput(), b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  Pattern p;
  auto input = p.Input();
  auto identity = p.Op({"Identity", "", {16}}, {input});
  p.Op({"Dropout", "", {13}}, {identity});
  auto rule = IdentityPair();
  rule.pattern = p;
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter({rule}).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
  for (bool indexed : {false, true}) {
    for (int scenario = 0; scenario < 3; ++scenario) {
      TestGraph versions;
      auto* input_value = versions.builder.MakeInput<float>({2}, -1.0f, 1.0f);
      auto* middle = versions.builder.MakeIntermediate();
      versions.builder.AddNode("Identity", {input_value}, {middle});
      versions.builder.AddNode("Identity", {middle}, {versions.builder.MakeOutput()});
      versions.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(versions.graph.Resolve());
      Pattern pattern;
      auto value = pattern.Input();
      auto first = pattern.Op({"Identity", "", {scenario == 0 ? 1 : 16}}, {value});
      pattern.Op({"Identity", "", {scenario == 1 ? 1 : 16}}, {first});
      rule = IdentityPair();
      rule.pattern = pattern;
      ASSERT_STATUS_OK(PatternRewriter({rule}, {.indexed = indexed}).Run(versions.graph, stats));
      EXPECT_EQ(stats.rewrites, scenario == 2 ? 1u : 0u);
      ASSERT_STATUS_OK(versions.graph.Resolve());
    }
  }
}

TEST(PatternRewriterTests, NumericalMultiOutputLeafPreservation) {
  TransformerTester(
      [](ModelTestBuilder& b) {
        auto* x = b.MakeInput<float>({2, 4}, -1.0f, 1.0f);
        auto* dropout = b.MakeIntermediate();
        auto* first = b.MakeIntermediate();
        b.AddNode("Dropout", {x}, {dropout, b.MakeOutput()});
        b.AddNode("Identity", {dropout}, {first});
        b.AddNode("Identity", {first}, {b.MakeOutput()});
      },
      [](InferenceSessionWrapper& session) {
        EXPECT_EQ(Count(session.GetGraph(), "Dropout"), 1u);
        EXPECT_EQ(Count(session.GetGraph(), "Identity"), 1u);
      },
      TransformerLevel::Default, TransformerLevel::Level1, 17, 0.0, 0.0,
      std::make_unique<PatternRewriter>(Rules{IdentityPair()}));
}

TEST(PatternRewriterTests, DoubleTypeRejection) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<double>({2}, -1.0, 1.0);
  auto* h = b.MakeIntermediate();
  auto& sigmoid = b.AddNode("HardSigmoid", {x}, {h});
  sigmoid.AddAttribute("alpha", 1.0f / 6.0f);
  sigmoid.AddAttribute("beta", 0.5f);
  b.AddNode("Mul", {h, x}, {b.MakeOutput()});
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  Statistics stats;
  ASSERT_STATUS_OK(PatternRewriter(MakeExampleRules()).Run(test.graph, stats));
  EXPECT_EQ(stats.rewrites, 0u);
}

TEST(PatternRewriterTests, BranchedSharedNodeBindings) {
  for (bool external : {false, true}) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* y = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* sum = b.MakeIntermediate();
    auto* product = b.MakeIntermediate();
    b.AddNode("Add", {x, y}, {sum});
    b.AddNode("Mul", {sum, x}, {product});
    b.AddNode("Add", {sum, product}, {b.MakeOutput()});
    if (external) b.AddNode("Identity", {sum}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Pattern p;
    auto px = p.Input(), py = p.Input();
    auto psum = p.Op({"Add", "", {14}}, {px, py});
    auto pproduct = p.Op({"Mul", "", {14}}, {psum, px});
    p.Op({"Add", "", {14}}, {psum, pproduct});
    Rule rule{"diamond", p, [](const Graph&, const Match&) { return true; },
              [](const Graph&, const Match&) { return Replacement{"Identity", "", 16, {0}, {}}; }};
    Statistics stats;
    // Structural-only test: this deliberately has no semantically equivalent replacement.
    ASSERT_STATUS_OK(PatternRewriter({rule}, {.match_only = true}).Run(test.graph, stats));
    EXPECT_EQ(stats.matches, external ? 0u : 1u);
    EXPECT_EQ(stats.rewrites, 0u);
  }
}

TEST(PatternRewriterTests, SubgraphsAndImplicitCaptureBoundary) {
  TestGraph test;
  auto& b = test.builder;
  auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
  auto* captured = b.MakeIntermediate();
  b.AddNode("Identity", {x}, {captured});
  b.AddNode("Identity", {captured}, {b.MakeOutput()});
  ONNX_NAMESPACE::GraphProto branch;
  branch.set_name("branch");
  auto* first = branch.add_node();
  first->set_op_type("Identity");
  first->add_input(captured->Name());
  first->add_output("branch_mid");
  auto* second = branch.add_node();
  second->set_op_type("Identity");
  second->add_input("branch_mid");
  second->add_output("branch_out");
  auto* output = branch.add_output();
  output->set_name("branch_out");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  output->mutable_type()->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
  auto* condition = b.MakeInput<bool>({}, std::vector<bool>{true});
  auto& if_node = b.AddNode("If", {condition}, {b.MakeOutput()});
  if_node.AddAttribute("then_branch", branch);
  if_node.AddAttribute("else_branch", branch);
  b.SetGraphOutputs();
  ASSERT_STATUS_OK(test.graph.Resolve());
  bool modified = false;
  ASSERT_STATUS_OK(
      PatternRewriter({IdentityPair()}).Apply(test.graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_TRUE(modified);
  // The captured intermediate keeps both outer Identity nodes alive.
  EXPECT_EQ(Count(test.graph, "Identity"), 2u);
  for (const auto& entry : if_node.GetAttributeNameToSubgraphMap()) {
    EXPECT_EQ(Count(*entry.second, "Identity"), 1u);
  }
}

#ifndef DISABLE_CONTRIB_OPS
TEST(PatternRewriterTests, MatMulConstantGuardsAndBroadcastRank) {
  for (int scenario = 0; scenario < 6; ++scenario) {
    TestGraph test;
    auto& b = test.builder;
    auto* x = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* y = b.MakeInput<float>({2}, -1.0f, 1.0f);
    auto* mm = b.MakeIntermediate();
    b.AddNode("MatMul", {x, y}, {mm});
    float value = 2.0f;
    if (scenario == 1) value = 0.0f;
    if (scenario == 2) value = std::numeric_limits<float>::infinity();
    NodeArg* divisor;
    if (scenario == 3) {
      divisor = b.MakeInitializer<float>({1}, {value});
    } else if (scenario == 4) {
      divisor = b.MakeInput<float>({}, std::vector<float>{value});
    } else {
      divisor = b.MakeScalarInitializer<float>(value);
    }
    b.AddNode("Div", {mm, divisor}, {b.MakeOutput()});
    b.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    if (scenario == 5) {
      auto inputs = test.graph.GetInputsIncludingInitializers();
      inputs.push_back(divisor);
      test.graph.SetInputs(inputs);
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeExampleRules()).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, NumericalMatMulDivAndTranspose) {
  for (int scenario = 0; scenario < 3; ++scenario) {
    bool transpose = scenario != 0;
    bool divide = scenario != 1;
    TransformerTester(
        [transpose, divide](ModelTestBuilder& b) {
          auto* x = b.MakeInput<float>({2, 3}, -1.0f, 1.0f);
          auto* y =
              b.MakeInput<float>(transpose ? std::vector<int64_t>{4, 3} : std::vector<int64_t>{3, 4}, -1.0f, 1.0f);
          auto* mm = divide ? b.MakeIntermediate() : b.MakeOutput();
          if (transpose) {
            auto* t = b.MakeIntermediate();
            b.AddNode("Transpose", {y}, {t}).AddAttribute("perm", std::vector<int64_t>{1, 0});
            y = t;
          }
          b.AddNode("MatMul", {x, y}, {mm});
          if (divide) b.AddNode("Div", {mm, b.MakeScalarInitializer<float>(2.0f)}, {b.MakeOutput()});
        },
        [](InferenceSessionWrapper& s) { EXPECT_EQ(Count(s.GetGraph(), "FusedMatMul"), 1u); },
        TransformerLevel::Default, TransformerLevel::Level1, 17, 1e-6, 1e-5,
        std::make_unique<PatternRewriter>(MakeExampleRules()));
  }
}

TEST(PatternRewriterTests, RmsConstantAndAttributeGuards) {
  for (int scenario = 0; scenario < 4; ++scenario) {
    TestGraph test;
    auto* x = test.builder.MakeInput<float>({2, 4}, -1.0f, 1.0f);
    BuildRms(test.builder, x, test.builder.MakeOutput(), 4, scenario == 1 ? -1e-5f : 1e-5f, scenario == 2);
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    if (scenario == 3) {
      for (auto& node : test.graph.Nodes()) {
        if (node.OpType() == "Pow") node.MutableInputDefs()[1] = x;
      }
      ASSERT_STATUS_OK(test.graph.Resolve());
    }
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeExampleRules()).Run(test.graph, stats));
    EXPECT_EQ(stats.rewrites, scenario == 0 ? 1u : 0u);
    ASSERT_STATUS_OK(test.graph.Resolve());
  }
}

TEST(PatternRewriterTests, NumericalDecoderPrefillAndDecode) {
  for (int64_t sequence : {1, 4}) {
    TestGraph test;
    BuildDecoder(test.builder, 2, sequence);
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    std::string original;
    test.model.ToProto().SerializeToString(&original);
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeExampleRules()).Run(test.graph, stats));
    ASSERT_STATUS_OK(test.graph.Resolve());
    EXPECT_EQ(Count(test.graph, "SimplifiedLayerNormalization"), 4u);
    EXPECT_EQ(Count(test.graph, "FusedMatMul"), 2u);
    EXPECT_EQ(stats.rewrites, 8u);
    std::string rewritten;
    test.model.ToProto().SerializeToString(&rewritten);
    auto execute = [&](const std::string& serialized, std::vector<OrtValue>& outputs) {
      SessionOptions options;
      options.graph_optimization_level = TransformerLevel::Default;
      options.intra_op_param.thread_pool_size = 1;
      InferenceSessionWrapper session{options, GetEnvironment()};
      std::istringstream stream(serialized);
      ORT_RETURN_IF_ERROR(session.Load(stream));
      ORT_RETURN_IF_ERROR(session.Initialize());
      return session.Run(RunOptions{}, test.builder.feeds_, test.builder.output_names_, &outputs);
    };
    std::vector<OrtValue> baseline, fused;
    ASSERT_STATUS_OK(execute(original, baseline));
    ASSERT_STATUS_OK(execute(rewritten, fused));
    ASSERT_EQ(baseline.size(), fused.size());
    float max_absolute = 0.0f, max_relative = 0.0f;
    for (size_t output = 0; output < baseline.size(); ++output) {
      const auto& a = baseline[output].Get<Tensor>();
      const auto& c = fused[output].Get<Tensor>();
      ASSERT_EQ(a.Shape(), c.Shape());
      for (int64_t i = 0; i < a.Shape().Size(); ++i) {
        float difference = std::abs(a.Data<float>()[i] - c.Data<float>()[i]);
        float magnitude = std::abs(a.Data<float>()[i]);
        max_absolute = std::max(max_absolute, difference);
        max_relative = std::max(max_relative, difference / std::max(magnitude, 1e-6f));
        ASSERT_LE(difference, 2e-5f + 2e-4f * magnitude);
      }
    }
    std::cout << "decoder sequence=" << sequence << " outputs=" << fused.size() << " rewrites=" << stats.rewrites
              << " max_absolute=" << max_absolute << " max_relative_floor_1e-6=" << max_relative << '\n';
  }
}
#endif

// Explicitly enabled benchmark, not part of regular correctness runs.
TEST(PatternRewriterBenchmark, DISABLED_Decoder) {
#ifdef DISABLE_CONTRIB_OPS
  GTEST_SKIP() << "The active decoder benchmark requires contrib operators.";
#else
  using Clock = std::chrono::steady_clock;
  auto ms = [](auto begin, auto end) { return std::chrono::duration<double, std::milli>(end - begin).count(); };
  auto logger = DefaultLoggingManager().CreateLogger("pattern_rewriter_benchmark", logging::Severity::kERROR, false);
  const auto destination = Env::Default().GetEnvironmentVar("ORT_REWRITER_BENCHMARK_CSV");
  ASSERT_FALSE(destination.empty()) << "Set ORT_REWRITER_BENCHMARK_CSV to an output CSV path.";
  std::ofstream csv(destination);
  ASSERT_TRUE(csv.is_open());
  csv << "layers,nodes,rules,workload,indexed,repeat,prepare_ms,engine_ms,resolve_ms,visited,candidates,"
         "structural_checks,guards,matches,rewrites,output_nodes\n";
  for (size_t layers : {1u, 8u, 32u}) {
    TestGraph source;
    BuildDecoder(source.builder, layers);
    source.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(source.graph.Resolve());
    InlinedHashMap<std::string, size_t> histogram;
    InlinedVector<std::string> operators;
    for (const auto& node : source.graph.Nodes()) {
      if (++histogram[node.OpType()] == 1) operators.push_back(node.OpType());
    }
    std::sort(operators.begin(), operators.end());
    std::cout << "decoder layers=" << layers << " nodes=" << source.graph.NumberOfNodes()
              << " inputs=" << source.graph.GetInputs().size() << " outputs=" << source.graph.GetOutputs().size()
              << " operators=";
    for (const auto& op : operators) std::cout << op << ':' << histogram.at(op) << ' ';
    std::cout << '\n';
    const auto proto = source.model.ToProto();
    for (size_t count : {1u, 10u, 100u, 300u, 500u}) {
      for (int workload = 0; workload < 4; ++workload) {
        for (bool indexed : {false, true}) {
          for (int repeat = -2; repeat < 9; ++repeat) {
            // Loading, generation, and copying are deliberately outside timing.
            Model model(proto, nullptr, *logger);
            Graph& graph = model.MainGraph();
            ASSERT_STATUS_OK(graph.Resolve());
            auto begin_prepare = Clock::now();
            auto examples = MakeExampleRules();
            // Put the decoder's active RMSNorm pattern first, including in the one-rule case.
            std::rotate(examples.begin(), examples.end() - 1, examples.end());
            Rules rules;
            for (size_t i = 0; i < count; ++i) {
              Rule rule = examples[i % examples.size()];
              rule.name += std::to_string(i);
              // Same-root, same-structure adversarial rejection after full matching.
              // Active rules are placed last to force evaluation of all preceding guards.
              if (workload == 1 || i + examples.size() < count) {
                auto original = rule.guard;
                rule.guard = [original](const Graph& g, const Match& m) {
                  bool supported = original(g, m);
                  return supported && m.nodes.back()->Name() == "deliberately_absent_benchmark_name";
                };
              }
              if (workload == 3) {
                // Distinct, synthetic same-Mul-root structural nonmatches, with common suffixes.
                Pattern pattern;
                auto input = pattern.Input();
                auto value = input;
                for (size_t bit = 0; bit < 9; ++bit) {
                  value = pattern.Op({(i & (size_t{1} << bit)) ? "Tanh" : "Sigmoid", "", {6, 13}}, {value});
                }
                pattern.Op({"Mul", "", {7, 13, 14}}, {input, value});
                rule.pattern = std::move(pattern);
                rule.guard = [](const Graph&, const Match&) { return false; };
              }
              rules.push_back(std::move(rule));
            }
            PatternRewriter rewriter(std::move(rules), {.match_only = workload != 2, .indexed = indexed});
            auto end_prepare = Clock::now();
            Statistics stats;
            auto begin_engine = Clock::now();
            ASSERT_STATUS_OK(rewriter.Run(graph, stats));
            auto end_engine = Clock::now();
            auto begin_resolve = Clock::now();
            ASSERT_STATUS_OK(graph.Resolve());
            auto end_resolve = Clock::now();
            if (repeat < 0) continue;
            csv << layers << ',' << source.graph.NumberOfNodes() << ',' << count << ',' << workload << ',' << indexed
                << ',' << repeat << ',' << ms(begin_prepare, end_prepare) << ',' << ms(begin_engine, end_engine) << ','
                << ms(begin_resolve, end_resolve) << ',' << stats.visited << ',' << stats.candidates << ','
                << stats.structural_checks << ',' << stats.guards << ',' << stats.matches << ',' << stats.rewrites
                << ',' << graph.NumberOfNodes() << '\n';
          }
        }
      }
    }
  }
  csv.flush();
  ASSERT_TRUE(csv.good()) << "Failed to write benchmark results.";
#endif
}

}  // namespace test
}  // namespace onnxruntime
