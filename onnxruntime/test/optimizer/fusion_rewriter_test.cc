// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <type_traits>

#include "gtest/gtest.h"

#include "core/common/logging/logging.h"
#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/function_extractor.h"
#include "core/optimizer/fusion_rewriter.h"
#include "core/optimizer/fusion_rewriter_diagnostics.h"
#include "core/optimizer/fusion_rewriter_matcher.h"
#include "onnx/defs/function.h"
#include "onnx/defs/parser.h"

namespace onnxruntime {
namespace test {

namespace {

using FunctionProto = ONNX_NAMESPACE::FunctionProto;
using fusion_rewriter_internal::FusionDiagnosticsTestAccess;
using fusion_rewriter_internal::FusionExecutionControls;
using fusion_rewriter_internal::FusionRuleSetTestAccess;
using fusion_rewriter_internal::FusionTestPlan;
using fusion_rewriter_internal::ObservedDependencyKind;

constexpr int kOnnxOpset = 13;
constexpr int kFastGeluVersion = 1;
constexpr const char* kPatternDomain = "ort.pattern";

FunctionProto ParseFunction(std::string_view source) {
  const std::string source_string{source};
  ONNX_NAMESPACE::OnnxParser parser(source_string.c_str());
  FunctionProto function_proto;
  const auto status = parser.Parse(function_proto);
  ORT_ENFORCE(status.IsOK(), "Failed to parse function text: ",
              status.ErrorMessage());
  ORT_ENFORCE(parser.EndOfInput(), "Extra unparsed function input.");
  return function_proto;
}

ONNX_NAMESPACE::ModelProto ParseModel(std::string_view source) {
  const std::string source_string{source};
  ONNX_NAMESPACE::OnnxParser parser(source_string.c_str());
  ONNX_NAMESPACE::ModelProto model_proto;
  const auto status = parser.Parse(model_proto);
  ORT_ENFORCE(status.IsOK(), "Failed to parse model text: ",
              status.ErrorMessage());
  ORT_ENFORCE(parser.EndOfInput(), "Extra unparsed model input.");
  return model_proto;
}

std::shared_ptr<Model> MakeModelFromText(
    std::string_view source,
    gsl::span<const FunctionProto> function_protos = {}) {
  ONNX_NAMESPACE::ModelProto model_proto = ParseModel(source);
  for (const auto& function_proto : function_protos) {
    model_proto.add_functions()->CopyFrom(function_proto);
  }

  std::shared_ptr<Model> model;
  ORT_THROW_IF_ERROR(Model::Load(
      std::move(model_proto), model, nullptr,
      DefaultLoggingManager().DefaultLogger()));
  ORT_ENFORCE(model != nullptr, "Parsed model load returned null.");
  return model;
}

FunctionProto MakeTwoIdentityPattern(std::string_view name = "TwoIdentity") {
  return ParseFunction(
      "<opset_import: [\"\" : 13], domain: \"ort.pattern\">\n" +
      std::string{name} + R"( (x) => (out) {
  intermediate = Identity(x)
  out = Identity(intermediate)
})");
}

FunctionProto MakeIdentityReluPattern(std::string_view name = "IdentityRelu") {
  return ParseFunction(
      "<opset_import: [\"\" : 13], domain: \"ort.pattern\">\n" +
      std::string{name} + R"( (x) => (out) {
  intermediate = Identity(x)
  out = Relu(intermediate)
})");
}

FunctionProto MakeAddIdentityPattern(std::string_view name = "AddIdentity") {
  return ParseFunction(
      "<opset_import: [\"\" : 13], domain: \"ort.pattern\">\n" +
      std::string{name} + R"( (x, y) => (out) {
  sum = Add(x, y)
  out = Identity(sum)
})");
}

FunctionProto MakeAttributedPattern(std::string_view name = "Attributed") {
  return ParseFunction(
      "<opset_import: [\"\" : 13], domain: \"ort.pattern\">\n" +
      std::string{name} + R"( <slope> (x) => (out) {
  activated = LeakyRelu <alpha : float = @slope> (x)
  out = Identity(activated)
})");
}

std::string GeluBody(std::string_view tensor_type,
                     std::string_view coefficient) {
  const std::string scalar_type{
      tensor_type == "double" ? "double" : "float"};
  return "  three = Constant <value = " + scalar_type +
         " {3.0}> ()\n"
         "  x3 = Pow(x, three)\n"
         "  cubic_coefficient = Constant <value = " +
         scalar_type + " {" + std::string{coefficient} +
         "}> ()\n"
         "  cubic = Mul(cubic_coefficient, x3)\n"
         "  shifted = Add(x, cubic)\n"
         "  sqrt_two_over_pi = Constant <value = " +
         scalar_type +
         " {0.7978845608028654}> ()\n"
         "  scaled = Mul(sqrt_two_over_pi, shifted)\n"
         "  tanh = Tanh(scaled)\n"
         "  one = Constant <value = " +
         scalar_type +
         " {1.0}> ()\n"
         "  plus_one = Add(tanh, one)\n"
         "  half = Constant <value = " +
         scalar_type +
         " {0.5}> ()\n"
         "  halved = Mul(half, plus_one)\n"
         "  out = Mul(x, halved)\n";
}

FunctionProto MakeGeluPattern(
    std::string_view tensor_type = "float",
    std::string_view name = "TanhGelu") {
  return ParseFunction(
      "<opset_import: [\"\" : 13], domain: \"ort.pattern\">\n" +
      std::string{name} + " (x) => (out) {\n" +
      GeluBody(tensor_type, "0.044715") + "}");
}

std::shared_ptr<Model> MakeGeluModel(
    std::string_view tensor_type = "float",
    std::string_view coefficient = "0.044715") {
  return MakeModelFromText(
      "<ir_version: 8, opset_import: [\"\" : 13, "
      "\"com.microsoft\" : 1]>\n"
      "target_graph (" +
      std::string{tensor_type} + "[2, 3] x) => (" +
      std::string{tensor_type} + "[2, 3] out) {\n" +
      GeluBody(tensor_type, coefficient) + "}");
}

std::shared_ptr<Model> MakeTwoIdentityModel(
    std::string_view type = "float[2]",
    std::string_view input_name = "x") {
  std::string source =
      "<ir_version: 8, opset_import: [\"\" : 13, "
      "\"com.microsoft\" : 1]>\n"
      "target_graph (" +
      std::string{type} + " " + std::string{input_name} + ") => (" +
      std::string{type} +
      " out) {\n"
      "  intermediate = Identity(" +
      std::string{input_name} +
      ")\n"
      "  out = Identity(intermediate)\n"
      "}";
  return MakeModelFromText(source);
}

std::shared_ptr<Model> MakeIdentityReluModel(
    std::string_view type = "float[2]") {
  return MakeModelFromText(
      "<ir_version: 8, opset_import: [\"\" : 13, "
      "\"com.microsoft\" : 1]>\n"
      "target_graph (" +
      std::string{type} + " x) => (" + std::string{type} + R"( out) {
  intermediate = Identity(x)
  out = Relu(intermediate)
})");
}

std::shared_ptr<Model> MakeIndependentRegionsModel(size_t region_count) {
  std::string source =
      "<ir_version: 8, opset_import: [\"\" : 13, "
      "\"com.microsoft\" : 1]>\n"
      "target_graph (";
  for (size_t i = 0; i < region_count; ++i) {
    if (i != 0) {
      source += ", ";
    }
    source += "float[2] x" + std::to_string(i);
  }
  source += ") => (";
  for (size_t i = 0; i < region_count; ++i) {
    if (i != 0) {
      source += ", ";
    }
    source += "float[2] out" + std::to_string(i);
  }
  source += ") {\n";
  for (size_t i = 0; i < region_count; ++i) {
    source += "  intermediate" + std::to_string(i) +
              " = Identity(x" + std::to_string(i) + ")\n";
    source += "  out" + std::to_string(i) +
              " = Identity(intermediate" + std::to_string(i) + ")\n";
  }
  source += "}";
  return MakeModelFromText(source);
}

FusionConstraintProgram MakeConstraints(
    FusionConstraint predicate = FusionConstraint::AllOf({}),
    std::vector<FusionDimensionEquivalenceClass> dimension_classes = {}) {
  return FusionConstraintProgram(std::move(dimension_classes),
                                 std::move(predicate));
}

FusionReplacementCall MakeReplacementCall(
    std::string domain, std::string op_type, int since_version,
    std::initializer_list<size_t> input_indices = {0},
    std::initializer_list<size_t> output_indices = {0}) {
  FusionReplacementCall replacement;
  replacement.domain = std::move(domain);
  replacement.op_type = std::move(op_type);
  replacement.since_version = since_version;
  for (const size_t index : input_indices) {
    replacement.inputs.push_back(
        FusionReplacementInput{std::optional<size_t>{index}});
  }
  for (const size_t index : output_indices) {
    replacement.outputs.push_back(FusionReplacementOutput{index});
  }
  return replacement;
}

FusionReplacementCall MakeFastGeluReplacement() {
  return MakeReplacementCall(kMSDomain, "FastGelu", kFastGeluVersion);
}

FusionRule MakeRule(
    const FunctionProto& pattern,
    FusionReplacementCall replacement,
    FusionConstraintProgram constraints = MakeConstraints(),
    FusionMatchPredicate predicate = {},
    FusionRuleId id = 1,
    int32_t priority = 0,
    std::string name = "test-rule") {
  return FusionRule(
      pattern, std::move(replacement), std::move(constraints),
      std::move(predicate),
      FusionRuleOptions{id, std::move(name), priority});
}

std::unique_ptr<FusionRuleSet> MakeRuleSet(
    std::vector<FusionRule> rules,
    FusionRuleSetOptions options = {}) {
  return std::make_unique<FusionRuleSet>(
      std::move(rules), std::move(options));
}

std::unique_ptr<FusionRuleSet> MakeIdentityRuleSet(
    FusionConstraintProgram constraints = MakeConstraints(),
    FusionMatchPredicate predicate = {},
    FusionRuleSetOptions options = {},
    FusionRuleId id = 1) {
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      MakeTwoIdentityPattern(), MakeFastGeluReplacement(),
      std::move(constraints), std::move(predicate), id));
  return MakeRuleSet(std::move(rules), std::move(options));
}

size_t CountOp(
    const Graph& graph, std::string_view domain, std::string_view op_type) {
  return static_cast<size_t>(std::count_if(
      graph.Nodes().begin(), graph.Nodes().end(),
      [&](const Node& node) {
        return node.Domain() == domain && node.OpType() == op_type;
      }));
}

Node& FindOnlyOp(
    Graph& graph, std::string_view domain, std::string_view op_type) {
  Node* result = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.Domain() == domain && node.OpType() == op_type) {
      EXPECT_EQ(result, nullptr);
      result = &node;
    }
  }
  EXPECT_NE(result, nullptr);
  return *result;
}

void AssertResolved(const Graph& graph) {
  ASSERT_FALSE(graph.GraphResolveNeeded());
  for (const auto& node : graph.Nodes()) {
    ASSERT_NE(node.Op(), nullptr) << node.Name();
  }
}

void AssertCallIO(
    const Node& node,
    gsl::span<const std::string> expected_inputs,
    gsl::span<const std::string> expected_outputs) {
  ASSERT_EQ(node.InputDefs().size(), expected_inputs.size());
  ASSERT_EQ(node.OutputDefs().size(), expected_outputs.size());
  for (size_t i = 0; i < expected_inputs.size(); ++i) {
    EXPECT_EQ(node.InputDefs()[i]->Name(), expected_inputs[i]);
  }
  for (size_t i = 0; i < expected_outputs.size(); ++i) {
    EXPECT_EQ(node.OutputDefs()[i]->Name(), expected_outputs[i]);
  }
}

std::string SerializeGraph(const Graph& graph) {
  return graph.ToGraphProto().SerializeAsString();
}

void ExpectSingleIdentityFusion(
    FusionConstraintProgram constraints,
    std::string_view type,
    bool should_fuse) {
  auto model = MakeTwoIdentityModel(type);
  auto rule_set = MakeIdentityRuleSet(std::move(constraints));
  const FusionRewriteResult result = rule_set->Apply(*model);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, should_fuse ? 1u : 0u);
  EXPECT_EQ(CountOp(model->MainGraph(), kMSDomain, "FastGelu"),
            should_fuse ? 1u : 0u);
  AssertResolved(model->MainGraph());
}

class FusionRewriterTest : public ::testing::Test {};

TEST_F(FusionRewriterTest, PatternIdentityNeedNotBeRegistered) {
  const FunctionProto pattern = MakeTwoIdentityPattern();
  auto model = MakeTwoIdentityModel();
  ASSERT_EQ(model->ToProto().functions_size(), 0);

  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      pattern, MakeFastGeluReplacement()));
  auto rule_set = MakeRuleSet(std::move(rules));
  const FusionRewriteResult result = rule_set->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(CountOp(model->MainGraph(), kMSDomain, "FastGelu"), 1u);
  AssertResolved(model->MainGraph());
}

TEST_F(FusionRewriterTest, RejectsFewerThanTwoPatternOperationNodes) {
  const FunctionProto pattern = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "ort.pattern">
OneIdentity (x) => (out) {
  out = Identity(x)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[2] x) => (float[2] out) {
  out = Identity(x)
})");
  const std::string before = SerializeGraph(model->MainGraph());

  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(pattern, MakeFastGeluReplacement()));
  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);

  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
}

TEST_F(FusionRewriterTest, RejectsMissingOrMismatchedReplacementSchema) {
  for (auto replacement : {
           MakeReplacementCall("missing.domain", "Missing", 1),
           MakeReplacementCall(kMSDomain, "FastGelu", 2)}) {
    auto model = MakeTwoIdentityModel();
    const std::string before = SerializeGraph(model->MainGraph());
    std::vector<FusionRule> rules;
    rules.push_back(MakeRule(
        MakeTwoIdentityPattern(), std::move(replacement)));

    const FusionRewriteResult result =
        MakeRuleSet(std::move(rules))->Apply(*model);
    EXPECT_FALSE(result.status.IsOK());
    EXPECT_EQ(result.replacements_applied, 0u);
    EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
  }
}

TEST_F(FusionRewriterTest, RejectsInvalidReplacementBoundaryMappings) {
  std::vector<FusionReplacementCall> replacements;
  replacements.push_back(
      MakeReplacementCall(kMSDomain, "FastGelu", 1, {1}, {0}));
  replacements.push_back(
      MakeReplacementCall(kMSDomain, "FastGelu", 1, {0}, {}));
  replacements.push_back(
      MakeReplacementCall(kMSDomain, "FastGelu", 1, {0}, {0, 0}));

  for (auto& replacement : replacements) {
    auto model = MakeTwoIdentityModel();
    const std::string before = SerializeGraph(model->MainGraph());
    std::vector<FusionRule> rules;
    rules.push_back(MakeRule(
        MakeTwoIdentityPattern(), std::move(replacement)));

    const FusionRewriteResult result =
        MakeRuleSet(std::move(rules))->Apply(*model);
    EXPECT_FALSE(result.status.IsOK());
    EXPECT_EQ(result.replacements_applied, 0u);
    EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
  }
}

TEST_F(FusionRewriterTest, PositiveConstraintsDoNotAcceptPermissiveUnknownPolicy) {
  using RankFactory = FusionConstraint (*)(FusionValueRef, size_t);
  static_assert(std::is_same_v<
                decltype(&FusionConstraint::RankIs), RankFactory>);
  static_assert(!std::is_convertible_v<
                FusionDimensionEquivalenceClass, FusionConstraint>);
}

TEST_F(FusionRewriterTest, MapsOrderedReplacementInputsAndOutputs) {
  const FunctionProto pattern = MakeAddIdentityPattern();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13]>
target_graph (float[2] x, float[2] y) => (float[2] out) {
  sum = Add(x, y)
  out = Identity(sum)
})");
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      pattern, MakeReplacementCall(kOnnxDomain, "Add", 13, {1, 0}, {0})));

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  Node& call = FindOnlyOp(model->MainGraph(), kOnnxDomain, "Add");
  const std::vector<std::string> expected_inputs{"y", "x"};
  const std::vector<std::string> expected_outputs{"out"};
  AssertCallIO(call, expected_inputs, expected_outputs);
}

TEST_F(FusionRewriterTest, EmitsLiteralAndBoundReplacementAttributes) {
  const FunctionProto pattern = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "ort.pattern">
AttributedGemm <scale> (x, y) => (out) {
  activated = LeakyRelu <alpha : float = @scale> (x)
  out = MatMul(activated, y)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13]>
target_graph (float[2, 3] x, float[3, 4] y) => (float[2, 4] out) {
  activated = LeakyRelu <alpha = 0.2> (x)
  out = MatMul(activated, y)
})");
  FusionReplacementCall replacement;
  replacement.domain = kOnnxDomain;
  replacement.op_type = "Gemm";
  replacement.since_version = 13;
  replacement.inputs = {
      FusionReplacementInput{std::optional<size_t>{0}},
      FusionReplacementInput{std::optional<size_t>{1}},
      FusionReplacementInput{std::nullopt},
  };
  replacement.outputs = {FusionReplacementOutput{0}};
  replacement.attributes = {
      FusionReplacementAttribute{
          "alpha", FusionReplacementAttributeSource::kLiteral, 0,
          ONNX_NAMESPACE::MakeAttribute("ignored", 0.5f)},
      FusionReplacementAttribute{
          "beta", FusionReplacementAttributeSource::kFormalAttribute, 0, {}},
  };
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(pattern, std::move(replacement)));

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const Node& call = FindOnlyOp(
      model->MainGraph(), kOnnxDomain, "Gemm");
  EXPECT_GE(call.GetAttributes().size(), 2u);
  EXPECT_FLOAT_EQ(call.GetAttributes().at("alpha").f(), 0.5f);
  EXPECT_FLOAT_EQ(call.GetAttributes().at("beta").f(), 0.2f);
}

TEST_F(FusionRewriterTest, VirtualCallRejectsTypeContradictionBeforeMutation) {
  auto model = MakeTwoIdentityModel("int32[2]");
  const std::string before = SerializeGraph(model->MainGraph());
  auto rule_set = MakeIdentityRuleSet();

  const FusionRewriteResult result = rule_set->Apply(*model);

  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
}

TEST_F(FusionRewriterTest, VirtualCallRejectsOutputShapeContradictionBeforeMutation) {
  auto model = MakeTwoIdentityModel("float[2, 3]");
  const std::string before = SerializeGraph(model->MainGraph());
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      MakeTwoIdentityPattern(),
      MakeReplacementCall(kOnnxDomain, "Transpose", 13)));

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);

  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
}

TEST_F(FusionRewriterTest, AllowsUnknownCompatibleReplacementShape) {
  auto model = MakeTwoIdentityModel("float[N, M]");
  Graph& graph = model->MainGraph();
  graph.GetNodeArg("x")->ClearShape();
  graph.GetNodeArg("out")->ClearShape();
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      MakeTwoIdentityPattern(),
      MakeReplacementCall(kOnnxDomain, "Transpose", 13)));

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
  AssertResolved(graph);
}

TEST_F(FusionRewriterTest, RankIsPassFailUnknown) {
  const FusionValueRef input = FusionValueRef::FormalInput(0);
  ExpectSingleIdentityFusion(
      MakeConstraints(FusionConstraint::RankIs(input, 2)),
      "float[2, 3]", true);
  ExpectSingleIdentityFusion(
      MakeConstraints(FusionConstraint::RankIs(input, 2)),
      "float[2]", false);

  auto model = MakeTwoIdentityModel("float[N, M]");
  model->MainGraph().GetNodeArg("x")->ClearShape();
  ASSERT_STATUS_OK(model->MainGraph().Resolve());
  auto rule_set = MakeIdentityRuleSet(
      MakeConstraints(FusionConstraint::RankIs(input, 2)));
  const FusionRewriteResult result = rule_set->Apply(*model);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FusionRewriterTest, ElementTypeIsAndIn) {
  const FusionValueRef input = FusionValueRef::FormalInput(0);
  ExpectSingleIdentityFusion(
      MakeConstraints(FusionConstraint::ElementTypeIs(
          input, ONNX_NAMESPACE::TensorProto_DataType_FLOAT)),
      "float[2]", true);
  ExpectSingleIdentityFusion(
      MakeConstraints(FusionConstraint::ElementTypeIs(
          input, ONNX_NAMESPACE::TensorProto_DataType_FLOAT)),
      "float16[2]", false);
  ExpectSingleIdentityFusion(
      MakeConstraints(FusionConstraint::ElementTypeIn(
          input, {ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                  ONNX_NAMESPACE::TensorProto_DataType_FLOAT16})),
      "float16[2]", true);
  ExpectSingleIdentityFusion(
      MakeConstraints(FusionConstraint::ElementTypeIn(
          input, {ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                  ONNX_NAMESPACE::TensorProto_DataType_FLOAT16})),
      "int32[2]", false);
}

TEST_F(FusionRewriterTest, ShapeEqualsConcreteAndSymbolic) {
  const FunctionProto pattern = MakeAddIdentityPattern();
  const auto apply = [&](std::string_view x_type,
                         std::string_view y_type,
                         FusionUnknownPolicy policy) {
    auto model = MakeModelFromText(
        "<ir_version: 8, opset_import: [\"\" : 13]>\n"
        "target_graph (" +
        std::string{x_type} + " x, " + std::string{y_type} +
        R"( y) => (float[2, 3] out) {
  sum = Add(x, y)
  out = Identity(sum)
})");
    std::vector<FusionRule> rules;
    rules.push_back(MakeRule(
        pattern, MakeReplacementCall(kOnnxDomain, "Add", 13, {0, 1}, {0}),
        MakeConstraints(FusionConstraint::ShapeEquals(
            FusionValueRef::FormalInput(0),
            FusionValueRef::FormalInput(1), policy))));
    const FusionRewriteResult result =
        MakeRuleSet(std::move(rules))->Apply(*model);
    EXPECT_STATUS_OK(result.status);
    return result.replacements_applied;
  };

  EXPECT_EQ(apply("float[2, 3]", "float[2, 3]",
                  FusionUnknownPolicy::kReject),
            1u);
  EXPECT_EQ(apply("float[N, D]", "float[N, D]",
                  FusionUnknownPolicy::kReject),
            1u);
  EXPECT_EQ(apply("float[N, D]", "float[M, D]",
                  FusionUnknownPolicy::kReject),
            0u);
  EXPECT_EQ(apply("float[N, D]", "float[M, D]",
                  FusionUnknownPolicy::kNotContradicted),
            1u);
  EXPECT_EQ(apply("float[2, 3]", "float[1, 3]",
                  FusionUnknownPolicy::kNotContradicted),
            0u);
}

TEST_F(FusionRewriterTest, NegativeAxisRequiresKnownRank) {
  const FusionConstraint dim_constraint = FusionConstraint::DimValueIs(
      FusionDimRef{FusionValueRef::FormalInput(0), -1}, 3);
  ExpectSingleIdentityFusion(
      MakeConstraints(dim_constraint), "float[2, 3]", true);

  auto model = MakeTwoIdentityModel("float[N, M]");
  model->MainGraph().GetNodeArg("x")->ClearShape();
  ASSERT_STATUS_OK(model->MainGraph().Resolve());
  auto rule_set = MakeIdentityRuleSet(MakeConstraints(dim_constraint));
  const FusionRewriteResult result = rule_set->Apply(*model);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FusionRewriterTest, DimensionEquivalenceIsOrderIndependent) {
  const FunctionProto pattern = MakeAddIdentityPattern();
  auto apply = [&](std::vector<FusionDimRef> dimensions) {
    auto model = MakeModelFromText(
        R"(<ir_version: 8, opset_import: ["" : 13]>
target_graph (float[B, H] x, float[B, H] y) => (float[B, H] out) {
  sum = Add(x, y)
  out = Identity(sum)
})");
    FusionDimensionEquivalenceClass dimensions_equal{
        "same", std::move(dimensions), FusionUnknownPolicy::kReject};
    std::vector<FusionRule> rules;
    rules.push_back(MakeRule(
        pattern, MakeReplacementCall(kOnnxDomain, "Add", 13, {0, 1}, {0}),
        MakeConstraints(FusionConstraint::AllOf({}),
                        {std::move(dimensions_equal)})));
    const FusionRewriteResult result =
        MakeRuleSet(std::move(rules))->Apply(*model);
    EXPECT_STATUS_OK(result.status);
    return result.replacements_applied;
  };
  const FusionDimRef x_h{FusionValueRef::FormalInput(0), 1};
  const FusionDimRef y_h{FusionValueRef::FormalInput(1), 1};
  EXPECT_EQ(apply({x_h, y_h}), 1u);
  EXPECT_EQ(apply({y_h, x_h}), 1u);
}

TEST_F(FusionRewriterTest, DimensionEquivalenceRejectsMismatch) {
  const FunctionProto pattern = MakeAddIdentityPattern();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13]>
target_graph (float[2, 3] x, float[2, 1] y) => (float[2, 3] out) {
  sum = Add(x, y)
  out = Identity(sum)
})");
  FusionDimensionEquivalenceClass dimensions_equal{
      "H",
      {{FusionValueRef::FormalInput(0), 1},
       {FusionValueRef::FormalInput(1), 1}},
      FusionUnknownPolicy::kReject};
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      pattern, MakeReplacementCall(kOnnxDomain, "Add", 13, {0, 1}, {0}),
      MakeConstraints(FusionConstraint::AllOf({}),
                      {std::move(dimensions_equal)})));
  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FusionRewriterTest, AttributePredicatesUseExplicitAndSchemaDefaultValues) {
  const FunctionProto pattern = MakeAttributedPattern();
  const auto apply = [&](std::string_view alpha_text,
                         FusionConstraint constraint) {
    auto model = MakeModelFromText(
        "<ir_version: 8, opset_import: [\"\" : 13, "
        "\"com.microsoft\" : 1]>\n"
        "target_graph (float[2] x) => (float[2] out) {\n"
        "  activated = LeakyRelu " +
        std::string{alpha_text} +
        "(x)\n"
        "  out = Identity(activated)\n}");
    std::vector<FusionRule> rules;
    rules.push_back(MakeRule(
        pattern, MakeFastGeluReplacement(),
        MakeConstraints(std::move(constraint))));
    const FusionRewriteResult result =
        MakeRuleSet(std::move(rules))->Apply(*model);
    EXPECT_STATUS_OK(result.status);
    return result.replacements_applied;
  };
  const FusionAttributeRef explicit_alpha =
      FusionAttributeRef::Effective({0}, "alpha");
  EXPECT_EQ(apply(
                "<alpha = 0.2> ",
                FusionConstraint::AttributeEquals(
                    explicit_alpha,
                    ONNX_NAMESPACE::MakeAttribute("alpha", 0.2f))),
            1u);

  const FusionAttributeRef default_alpha =
      FusionAttributeRef::Effective({0}, "alpha");
  EXPECT_EQ(apply(
                "",
                FusionConstraint::AttributeEquals(
                    default_alpha,
                    ONNX_NAMESPACE::MakeAttribute("alpha", 0.01f))),
            1u);
}

TEST_F(FusionRewriterTest, AttributePredicateUsesFormalBinding) {
  const FunctionProto pattern = MakeAttributedPattern();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[2] x) => (float[2] out) {
  activated = LeakyRelu <alpha = 0.2> (x)
  out = Identity(activated)
})");
  const FusionAttributeRef slope = FusionAttributeRef::Formal(0);
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      pattern, MakeFastGeluReplacement(),
      MakeConstraints(FusionConstraint::AllOf({
          FusionConstraint::AttributePresent(slope),
          FusionConstraint::AttributeTypeIs(
              slope, ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT),
          FusionConstraint::FloatAttributeInRange(slope, 0.1f, 0.3f),
      }))));
  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
}

TEST_F(FusionRewriterTest, EvaluatesTypeRankAndDimensionPredicates) {
  const FusionValueRef input = FusionValueRef::FormalInput(0);
  const FusionValueRef output = FusionValueRef::FormalOutput(0);
  ExpectSingleIdentityFusion(
      MakeConstraints(FusionConstraint::AllOf({
          FusionConstraint::TypeEquals(input, output),
          FusionConstraint::SameElementType(input, output),
          FusionConstraint::SameRank(input, output),
          FusionConstraint::DimEquals(
              FusionDimRef{input, 0},
              FusionDimRef{output, 0}),
      })),
      "float[2, 3]", true);

  const FunctionProto reshape_pattern = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "ort.pattern">
ReshapeIdentity (x, shape) => (out) {
  reshaped = Reshape(x, shape)
  out = Identity(reshaped)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[2, 3] x) => (float[6] out) {
  shape = Constant <value = int64 {6}> ()
  reshaped = Reshape(x, shape)
  out = Identity(reshaped)
})");
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      reshape_pattern, MakeFastGeluReplacement(),
      MakeConstraints(FusionConstraint::SameRank(
          FusionValueRef::FormalInput(0),
          FusionValueRef::FormalOutput(0)))));
  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FusionRewriterTest, EvaluatesIntegerAndSetAttributePredicates) {
  const FunctionProto pattern = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "ort.pattern">
CastIdentity <dtype> (x) => (out) {
  cast = Cast <to : int = @dtype> (x)
  out = Identity(cast)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[2] x) => (float[2] out) {
  cast = Cast <to : int = 1> (x)
  out = Identity(cast)
})");
  const FusionAttributeRef dtype = FusionAttributeRef::Formal(0);
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      pattern, MakeFastGeluReplacement(),
      MakeConstraints(FusionConstraint::AllOf({
          FusionConstraint::AttributeTypeIs(
              dtype, ONNX_NAMESPACE::AttributeProto_AttributeType_INT),
          FusionConstraint::AttributeIn(
              dtype,
              {ONNX_NAMESPACE::MakeAttribute("to", int64_t{1}),
               ONNX_NAMESPACE::MakeAttribute("to", int64_t{10})}),
          FusionConstraint::IntAttributeInRange(dtype, 1, 10),
          FusionConstraint::SameAttributeValue(
              dtype, FusionAttributeRef::Effective({0}, "to")),
      }))));

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
}

TEST_F(FusionRewriterTest, EvaluatesStringAttributePredicate) {
  const FunctionProto pattern = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "ort.pattern">
MaxPoolIdentity <padding> (x) => (out) {
  pooled = MaxPool <auto_pad : string = @padding, kernel_shape = [1]> (x)
  out = Identity(pooled)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[1, 1, 2] x) => (float[1, 1, 2] out) {
  pooled = MaxPool <auto_pad = "NOTSET", kernel_shape = [1]> (x)
  out = Identity(pooled)
})");
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      pattern, MakeFastGeluReplacement(),
      MakeConstraints(FusionConstraint::StringAttributeIn(
          FusionAttributeRef::Formal(0),
          {"NOTSET", "SAME_UPPER"}))));

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
}

TEST_F(FusionRewriterTest, IsPresentAndIsMissingInspectSlotsOnly) {
  const FunctionProto pattern = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "ort.pattern">
OptionalClip (x, minimum) => (out) {
  clipped = Clip(x, minimum)
  out = Identity(clipped)
})");
  const auto run = [&](std::string_view minimum_input,
                       FusionConstraint condition) {
    auto model = MakeModelFromText(
        "<ir_version: 8, opset_import: [\"\" : 13, "
        "\"com.microsoft\" : 1]>\n"
        "target_graph (float[2] x) => (float[2] out) {\n" +
        std::string{minimum_input} +
        "  clipped = Clip(x" +
        (minimum_input.empty() ? std::string{} : ", minimum") +
        ")\n"
        "  out = Identity(clipped)\n"
        "}");
    std::vector<FusionRule> rules;
    rules.push_back(MakeRule(
        pattern, MakeFastGeluReplacement(),
        MakeConstraints(std::move(condition))));
    const FusionRewriteResult result =
        MakeRuleSet(std::move(rules))->Apply(*model);
    EXPECT_STATUS_OK(result.status);
    return result.replacements_applied;
  };

  EXPECT_EQ(
      run("", FusionConstraint::IsMissing(
                  FusionValueRef::FormalInput(1))),
      1u);
  EXPECT_EQ(
      run("  minimum = Constant <value = float {-1.0}> ()\n",
          FusionConstraint::IsPresent(
              FusionValueRef::FormalInput(1))),
      1u);
}

TEST_F(FusionRewriterTest, ConditionRunsBeforeClosure) {
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[2] x) => (float[2] out, float[2] side) {
  intermediate = Identity(x)
  out = Identity(intermediate)
  side = Relu(intermediate)
})");
  FusionRuleSetOptions options;
  options.diagnostic_mode = FusionDiagnosticMode::kBestFailure;
  auto rule_set = MakeIdentityRuleSet(
      MakeConstraints(FusionConstraint::ElementTypeIs(
          FusionValueRef::FormalInput(0),
          ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)),
      {}, options);
  FusionTraceCollector trace;

  const FusionRewriteResult result =
      rule_set->Apply(*model, &trace);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
  ASSERT_EQ(trace.BestFailures().size(), 1u);
  EXPECT_EQ(trace.BestFailures()[0].stage,
            FusionMatchStage::kCondition);
  EXPECT_EQ(trace.BestFailures()[0].code,
            FusionFailureCode::kConstraintFalse);
}

TEST_F(FusionRewriterTest, CallbackSeesCompleteOpaqueBinding) {
  auto model = MakeTwoIdentityModel("float[B, 3]", "input");
  size_t call_count = 0;
  FusionMatchPredicate predicate =
      [&](const FusionMatchContext& context,
          FusionConditionResult& result) -> common::Status {
    ++call_count;
    const FusionValueView input = context.BoundInput(0);
    const FusionValueView output = context.BoundOutput(0);
    EXPECT_EQ(input.Name(), "input");
    EXPECT_EQ(output.Name(), "out");
    EXPECT_TRUE(input.Type().IsTensor());
    ORT_RETURN_IF_NOT(
        input.Type().TensorElementType().has_value(),
        "Expected the test input to have an element type.");
    EXPECT_EQ(*input.Type().TensorElementType(),
              ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    EXPECT_TRUE(input.Shape().HasRank());
    EXPECT_EQ(input.Shape().Rank(), 2u);
    ORT_RETURN_IF_NOT(
        input.Shape().Dimension(0).has_value(),
        "Expected the test input to have a first dimension.");
    EXPECT_EQ(input.Shape().Dimension(0)->Kind(),
              FusionDimensionKind::kSymbol);
    EXPECT_EQ(context.MatchedNode(0).OpType(), "Identity");
    result.decision = FusionConditionDecision::kSatisfied;
    return Status::OK();
  };
  auto rule_set =
      MakeIdentityRuleSet(MakeConstraints(), std::move(predicate));

  const FusionRewriteResult result = rule_set->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(call_count, 1u);
}

TEST_F(FusionRewriterTest, CallbackRejectionAndErrorPreserveGraph) {
  const auto run = [](bool return_error) {
    auto model = MakeTwoIdentityModel();
    const std::string before = SerializeGraph(model->MainGraph());
    FusionMatchPredicate predicate =
        [=](const FusionMatchContext&,
            FusionConditionResult& result) -> common::Status {
      if (return_error) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, FAIL, "injected predicate failure");
      }
      result.decision = FusionConditionDecision::kNotSatisfied;
      result.failure = FusionConditionFailure{
          "rank rejected", std::nullopt,
          FusionPatternValueId{0}, std::nullopt};
      return Status::OK();
    };
    auto rule_set =
        MakeIdentityRuleSet(MakeConstraints(), std::move(predicate));
    FusionTraceCollector trace;
    const FusionRewriteResult result =
        rule_set->Apply(*model, return_error ? nullptr : &trace);
    EXPECT_EQ(result.status.IsOK(), !return_error);
    EXPECT_EQ(result.replacements_applied, 0u);
    EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
  };
  run(false);
  run(true);
}

TEST_F(FusionRewriterTest, OpaqueViewsExposeNoGraphTypes) {
  static_assert(std::is_same_v<
                decltype(std::declval<FusionNodeView>().Index()),
                NodeIndex>);
  static_assert(std::is_same_v<
                decltype(std::declval<FusionValueView>().Name()),
                std::string_view>);
  static_assert(std::is_same_v<
                decltype(std::declval<FusionMatchContext>().BoundInput(0)),
                FusionValueView>);
  static_assert(!std::is_convertible_v<FusionNodeView, Node*>);
  static_assert(!std::is_convertible_v<FusionValueView, NodeArg*>);
  static_assert(!std::is_convertible_v<FusionMatchContext, Graph*>);
  static_assert(!std::is_convertible_v<
                FusionAttributeView,
                const ONNX_NAMESPACE::AttributeProto*>);
}

TEST_F(FusionRewriterTest, ShortCircuitRecordsOnlyObservedDependencies) {
  auto model = MakeTwoIdentityModel("float[2, 3]");
  auto rule_set = MakeIdentityRuleSet(
      MakeConstraints(FusionConstraint::AnyOf({
          FusionConstraint::ElementTypeIs(
              FusionValueRef::FormalInput(0),
              ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
          FusionConstraint::RankIs(
              FusionValueRef::FormalInput(0), 2),
      })));
  std::vector<FusionTestPlan> plans;
  ASSERT_STATUS_OK(FusionRuleSetTestAccess::DiscoverPlans(
      *rule_set, model->MainGraph(), plans));
  ASSERT_EQ(plans.size(), 1u);

  const auto dependencies =
      FusionRuleSetTestAccess::ObservedDependencies(plans[0]);
  EXPECT_NE(std::find_if(
                dependencies.begin(), dependencies.end(),
                [](const auto& dependency) {
                  return dependency.kind ==
                         ObservedDependencyKind::kValueType;
                }),
            dependencies.end());
  EXPECT_EQ(std::find_if(
                dependencies.begin(), dependencies.end(),
                [](const auto& dependency) {
                  return dependency.kind ==
                         ObservedDependencyKind::kValueRank;
                }),
            dependencies.end());
}

TEST_F(FusionRewriterTest, StaleObservedDimensionRejectsWholeBatch) {
  auto model = MakeTwoIdentityModel("float[2, 3]");
  FusionMatchPredicate predicate =
      [](const FusionMatchContext& context,
         FusionConditionResult&) -> common::Status {
    const auto dimension =
        context.BoundInput(0).Shape().Dimension(0);
    ORT_RETURN_IF_NOT(
        dimension.has_value() && dimension->Value() == 2,
        "Unexpected test input dimension.");
    return Status::OK();
  };
  auto rule_set =
      MakeIdentityRuleSet(MakeConstraints(), std::move(predicate));
  std::vector<FusionTestPlan> plans;
  ASSERT_STATUS_OK(FusionRuleSetTestAccess::DiscoverPlans(
      *rule_set, model->MainGraph(), plans));
  ASSERT_EQ(plans.size(), 1u);
  const auto dependencies =
      FusionRuleSetTestAccess::ObservedDependencies(plans[0]);
  EXPECT_NE(std::find_if(
                dependencies.begin(), dependencies.end(),
                [](const auto& dependency) {
                  return dependency.kind ==
                         ObservedDependencyKind::kValueDimension;
                }),
            dependencies.end());

  ONNX_NAMESPACE::TensorShapeProto changed_shape;
  changed_shape.add_dim()->set_dim_value(7);
  changed_shape.add_dim()->set_dim_value(3);
  model->MainGraph().GetNodeArg("x")->SetShape(changed_shape);
  const std::string before = SerializeGraph(model->MainGraph());

  EXPECT_FALSE(FusionRuleSetTestAccess::PrevalidatePlans(
                   *rule_set, model->MainGraph(), plans)
                   .IsOK());
  EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
}

TEST_F(FusionRewriterTest, StaleObservedTypeRejectsWholeBatch) {
  auto model = MakeTwoIdentityModel();
  FusionMatchPredicate predicate =
      [](const FusionMatchContext& context,
         FusionConditionResult&) -> common::Status {
    ORT_RETURN_IF_NOT(
        context.BoundInput(0).Type().TensorElementType() ==
            ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
        "Unexpected test input type.");
    return Status::OK();
  };
  auto rule_set =
      MakeIdentityRuleSet(MakeConstraints(), std::move(predicate));
  std::vector<FusionTestPlan> plans;
  ASSERT_STATUS_OK(FusionRuleSetTestAccess::DiscoverPlans(
      *rule_set, model->MainGraph(), plans));
  ASSERT_EQ(plans.size(), 1u);
  const auto dependencies =
      FusionRuleSetTestAccess::ObservedDependencies(plans[0]);
  EXPECT_NE(std::find_if(
                dependencies.begin(), dependencies.end(),
                [](const auto& dependency) {
                  return dependency.kind ==
                         ObservedDependencyKind::kValueType;
                }),
            dependencies.end());

  auto* type = const_cast<ONNX_NAMESPACE::TypeProto*>(
      model->MainGraph().GetNodeArg("x")->TypeAsProto());
  ASSERT_NE(type, nullptr);
  type->mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  const std::string before = SerializeGraph(model->MainGraph());

  EXPECT_FALSE(FusionRuleSetTestAccess::PrevalidatePlans(
                   *rule_set, model->MainGraph(), plans)
                   .IsOK());
  EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
}

TEST_F(FusionRewriterTest, StaleObservedAttributeRejectsWholeBatch) {
  const FunctionProto pattern = MakeAttributedPattern();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[2] x) => (float[2] out) {
  activated = LeakyRelu <alpha = 0.2> (x)
  out = Identity(activated)
})");
  FusionMatchPredicate predicate =
      [](const FusionMatchContext& context,
         FusionConditionResult&) -> common::Status {
    const auto alpha = context.EffectiveAttribute(0, "alpha").Float();
    ORT_RETURN_IF_NOT(
        alpha.has_value() && *alpha == 0.2f,
        "Unexpected test alpha.");
    return Status::OK();
  };
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      pattern, MakeFastGeluReplacement(),
      MakeConstraints(), std::move(predicate)));
  auto rule_set = MakeRuleSet(std::move(rules));
  std::vector<FusionTestPlan> plans;
  ASSERT_STATUS_OK(FusionRuleSetTestAccess::DiscoverPlans(
      *rule_set, model->MainGraph(), plans));
  ASSERT_EQ(plans.size(), 1u);
  EXPECT_NE(std::find_if(
                FusionRuleSetTestAccess::ObservedDependencies(plans[0]).begin(),
                FusionRuleSetTestAccess::ObservedDependencies(plans[0]).end(),
                [](const auto& dependency) {
                  return dependency.kind ==
                         ObservedDependencyKind::kEffectiveAttribute;
                }),
            FusionRuleSetTestAccess::ObservedDependencies(plans[0]).end());

  FindOnlyOp(model->MainGraph(), kOnnxDomain, "LeakyRelu")
      .GetMutableAttributes()
      .at("alpha")
      .set_f(0.3f);
  const std::string before = SerializeGraph(model->MainGraph());

  EXPECT_FALSE(FusionRuleSetTestAccess::PrevalidatePlans(
                   *rule_set, model->MainGraph(), plans)
                   .IsOK());
  EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
}

TEST_F(FusionRewriterTest, FusesFloatTanhGeluToPrimitiveFastGelu) {
  const FunctionProto pattern =
      MakeGeluPattern("float", "TanhGeluFloat");
  auto model = MakeGeluModel();
  const size_t nodes_before = model->MainGraph().NumberOfNodes();
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      pattern, MakeFastGeluReplacement(),
      MakeConstraints(FusionConstraint::AllOf({
          FusionConstraint::IsTensor(
              FusionValueRef::FormalInput(0)),
          FusionConstraint::ElementTypeIs(
              FusionValueRef::FormalInput(0),
              ONNX_NAMESPACE::TensorProto_DataType_FLOAT),
          FusionConstraint::SameElementType(
              FusionValueRef::FormalInput(0),
              FusionValueRef::FormalOutput(0)),
          FusionConstraint::ShapeEquals(
              FusionValueRef::FormalInput(0),
              FusionValueRef::FormalOutput(0)),
      }))));

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(model->MainGraph().NumberOfNodes(), nodes_before - 7);
  Node& fast_gelu =
      FindOnlyOp(model->MainGraph(), kMSDomain, "FastGelu");
  const std::vector<std::string> expected_inputs{"x"};
  const std::vector<std::string> expected_outputs{"out"};
  AssertCallIO(fast_gelu, expected_inputs, expected_outputs);
  EXPECT_EQ(model->ToProto().functions_size(), 0);
  AssertResolved(model->MainGraph());
}

TEST_F(FusionRewriterTest, GeluPatternIdentityIsNotFastGeluIdentity) {
  const FunctionProto pattern =
      MakeGeluPattern("float", "TanhGeluFloat");
  EXPECT_EQ(pattern.domain(), kPatternDomain);
  EXPECT_EQ(pattern.name(), "TanhGeluFloat");
  EXPECT_NE(pattern.domain(), kMSDomain);
  EXPECT_NE(pattern.name(), "FastGelu");

  auto model = MakeGeluModel();
  EXPECT_EQ(model->ToProto().functions_size(), 0);
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(pattern, MakeFastGeluReplacement()));
  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
}

TEST_F(FusionRewriterTest, GeluRulesUseDtypeSpecificLiterals) {
  {
    auto model = MakeGeluModel("double");
    std::vector<FusionRule> rules;
    rules.push_back(MakeRule(
        MakeGeluPattern("float", "TanhGeluFloat"),
        MakeFastGeluReplacement()));
    const FusionRewriteResult result =
        MakeRuleSet(std::move(rules))->Apply(*model);
    ASSERT_STATUS_OK(result.status);
    EXPECT_EQ(result.replacements_applied, 0u);
  }
  {
    auto model = MakeGeluModel("double");
    std::vector<FusionRule> rules;
    rules.push_back(MakeRule(
        MakeGeluPattern("double", "TanhGeluDouble"),
        MakeFastGeluReplacement()));
    const FusionRewriteResult result =
        MakeRuleSet(std::move(rules))->Apply(*model);
    ASSERT_STATUS_OK(result.status);
    EXPECT_EQ(result.replacements_applied, 1u);
    EXPECT_EQ(CountOp(
                  model->MainGraph(), kMSDomain, "FastGelu"),
              1u);
  }
}

TEST_F(FusionRewriterTest, GeluCoefficientOneBitNearMiss) {
  auto model = MakeGeluModel("float", "0.04471500217914581");

  FusionRuleSetOptions options;
  options.diagnostic_mode = FusionDiagnosticMode::kBestFailure;
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      MakeGeluPattern(), MakeFastGeluReplacement(),
      MakeConstraints(), {}, 17, 0,
      "TanhGeluFloatToFastGelu"));
  FusionTraceCollector trace;
  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules), options)->Apply(*model, &trace);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
  ASSERT_EQ(trace.BestFailures().size(), 1u);
  EXPECT_EQ(trace.BestFailures()[0].stage,
            FusionMatchStage::kLiteral);
  EXPECT_EQ(trace.BestFailures()[0].code,
            FusionFailureCode::kLiteralMismatch);
  EXPECT_EQ(trace.BestFailures()[0].target_value_name,
            "cubic_coefficient");
  EXPECT_NE(trace.Format().find("cubic_coefficient"),
            std::string::npos);
}

TEST_F(FusionRewriterTest, GeluRankConditionNearMiss) {
  auto model = MakeGeluModel();
  FusionRuleSetOptions options;
  options.diagnostic_mode = FusionDiagnosticMode::kBestFailure;
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      MakeGeluPattern(), MakeFastGeluReplacement(),
      MakeConstraints(FusionConstraint::RankIn(
          FusionValueRef::FormalInput(0), 1, 1)),
      {}, 18, 0, "RankLimitedGelu"));
  FusionTraceCollector trace;

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules), options)->Apply(*model, &trace);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
  ASSERT_EQ(trace.BestFailures().size(), 1u);
  EXPECT_EQ(trace.BestFailures()[0].stage,
            FusionMatchStage::kCondition);
  EXPECT_TRUE(trace.BestFailures()[0].constraint.has_value());
}

FusionFailureRecord MakeFailure(
    FusionRuleId rule_id,
    FusionMatchStage stage,
    size_t pattern_nodes_matched,
    std::string detail = {}) {
  FusionFailureRecord record;
  record.rule_id = rule_id;
  record.stage = stage;
  record.code = FusionFailureCode::kConstraintFalse;
  record.anchor_node = static_cast<NodeIndex>(10 + rule_id);
  record.pattern_nodes_matched = pattern_nodes_matched;
  record.detail = std::move(detail);
  return record;
}

TEST_F(FusionRewriterTest, DiagnosticsOffRetainsNothing) {
  FusionTraceCollector trace;
  EXPECT_TRUE(trace.BestFailures().empty());
  EXPECT_TRUE(trace.Records().empty());
  EXPECT_TRUE(trace.Format().empty());
  EXPECT_EQ(trace.SuccessCount(1), 0u);
  EXPECT_FALSE(trace.Truncated());
}

TEST_F(FusionRewriterTest, BestFailureIsPerUnsuccessfulRule) {
  FusionTraceCollector trace;
  FusionDiagnosticsTestAccess::Configure(
      trace, FusionDiagnosticMode::kBestFailure, 10, 4096);
  FusionDiagnosticsTestAccess::RecordFailure(
      trace, MakeFailure(1, FusionMatchStage::kLiteral, 3), 0, 2, 0);
  FusionDiagnosticsTestAccess::RecordFailure(
      trace, MakeFailure(2, FusionMatchStage::kCondition, 2), 0, 1, 0);

  ASSERT_EQ(trace.BestFailures().size(), 2u);
  EXPECT_EQ(trace.BestFailures()[0].rule_id, 1u);
  EXPECT_EQ(trace.BestFailures()[1].rule_id, 2u);
}

TEST_F(FusionRewriterTest, BestFailurePrefersLaterStageThenProgress) {
  FusionTraceCollector trace;
  FusionDiagnosticsTestAccess::Configure(
      trace, FusionDiagnosticMode::kBestFailure, 10, 4096);
  FusionDiagnosticsTestAccess::RecordFailure(
      trace, MakeFailure(1, FusionMatchStage::kLiteral, 8, "early"), 0, 0, 0);
  FusionDiagnosticsTestAccess::RecordFailure(
      trace, MakeFailure(1, FusionMatchStage::kCondition, 2, "late"), 0, 2, 1);
  FusionDiagnosticsTestAccess::RecordFailure(
      trace, MakeFailure(2, FusionMatchStage::kLiteral, 2, "less"), 0, 0, 0);
  FusionDiagnosticsTestAccess::RecordFailure(
      trace, MakeFailure(2, FusionMatchStage::kLiteral, 4, "more"), 0, 1, 1);

  ASSERT_EQ(trace.BestFailures().size(), 2u);
  EXPECT_EQ(trace.BestFailures()[0].detail, "late");
  EXPECT_EQ(trace.BestFailures()[1].detail, "more");
}

TEST_F(FusionRewriterTest, SuccessfulRuleSuppressesBestFailure) {
  FusionTraceCollector trace;
  FusionDiagnosticsTestAccess::Configure(
      trace, FusionDiagnosticMode::kBestFailure, 10, 4096);
  FusionDiagnosticsTestAccess::RecordFailure(
      trace, MakeFailure(7, FusionMatchStage::kCondition, 2), 0, 0, 0);
  FusionDiagnosticsTestAccess::RecordSuccess(trace, 7);

  EXPECT_TRUE(trace.BestFailures().empty());
  EXPECT_EQ(trace.SuccessCount(7), 1u);
}

TEST_F(FusionRewriterTest, AllFailuresIsBounded) {
  FusionTraceCollector trace;
  FusionDiagnosticsTestAccess::Configure(
      trace, FusionDiagnosticMode::kAllFailures, 2, 4096);
  FusionDiagnosticsTestAccess::RecordFailure(
      trace, MakeFailure(1, FusionMatchStage::kLiteral, 1), 0, 0, 0);
  FusionDiagnosticsTestAccess::RecordFailure(
      trace, MakeFailure(1, FusionMatchStage::kLiteral, 2), 0, 1, 1);
  FusionDiagnosticsTestAccess::RecordFailure(
      trace, MakeFailure(1, FusionMatchStage::kCondition, 2), 0, 2, 2);

  EXPECT_EQ(trace.Records().size(), 2u);
  EXPECT_TRUE(trace.Truncated());
}

TEST_F(FusionRewriterTest, DryRunReportsSuccessWithoutMutation) {
  auto model = MakeTwoIdentityModel();
  const std::string before = SerializeGraph(model->MainGraph());
  FusionRuleSetOptions options;
  options.diagnostic_mode = FusionDiagnosticMode::kDryRun;
  auto rule_set = MakeIdentityRuleSet(
      MakeConstraints(), {}, options, 9);
  FusionTraceCollector trace;

  const FusionRewriteResult result =
      rule_set->Apply(*model, &trace);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(trace.SuccessCount(9), 1u);
  EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
}

TEST_F(FusionRewriterTest, DiagnosticsDoNotConsumeSemanticBudget) {
  for (const FusionDiagnosticMode mode :
       {FusionDiagnosticMode::kOff,
        FusionDiagnosticMode::kBestFailure}) {
    auto model = MakeTwoIdentityModel();
    const std::string before = SerializeGraph(model->MainGraph());
    FusionRuleSetOptions options;
    options.max_rule_attempts = 0;
    options.diagnostic_mode = mode;
    auto rule_set = MakeIdentityRuleSet(
        MakeConstraints(), {}, options);
    FusionTraceCollector trace;
    const FusionRewriteResult result = rule_set->Apply(
        *model, mode == FusionDiagnosticMode::kOff ? nullptr : &trace);
    EXPECT_FALSE(result.status.IsOK());
    EXPECT_EQ(result.replacements_applied, 0u);
    EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
  }
}

TEST_F(FusionRewriterTest, TriesAllApplicableRulesAtAnchor) {
  auto model = MakeTwoIdentityModel();
  FusionMatchPredicate reject =
      [](const FusionMatchContext&,
         FusionConditionResult& result) -> common::Status {
    result.decision = FusionConditionDecision::kNotSatisfied;
    return Status::OK();
  };
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      MakeTwoIdentityPattern("Rejecting"),
      MakeReplacementCall(kOnnxDomain, "Relu", 13),
      MakeConstraints(), std::move(reject), 1, -1));
  rules.push_back(MakeRule(
      MakeTwoIdentityPattern("Accepting"),
      MakeFastGeluReplacement(),
      MakeConstraints(), {}, 2, 0));

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(CountOp(
                model->MainGraph(), kMSDomain, "FastGelu"),
            1u);
}

TEST_F(FusionRewriterTest, AnchorLocalPriorityBreaksSameAnchorConflict) {
  auto model = MakeTwoIdentityModel();
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      MakeTwoIdentityPattern("LaterPriority"),
      MakeFastGeluReplacement(),
      MakeConstraints(), {}, 1, 10));
  rules.push_back(MakeRule(
      MakeTwoIdentityPattern("EarlierPriority"),
      MakeReplacementCall(kOnnxDomain, "Relu", 13),
      MakeConstraints(), {}, 2, -10));

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(CountOp(model->MainGraph(), kOnnxDomain, "Relu"), 1u);
  EXPECT_EQ(
      CountOp(model->MainGraph(), kMSDomain, "FastGelu"), 0u);
}

TEST_F(FusionRewriterTest, ConsumerAnchorWinsCrossAnchorOverlap) {
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[2] x) => (float[2] out) {
  first = Identity(x)
  second = Identity(first)
  out = Relu(second)
})");
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      MakeTwoIdentityPattern("Upstream"),
      MakeReplacementCall(kOnnxDomain, "Relu", 13),
      MakeConstraints(), {}, 1, -100));
  rules.push_back(MakeRule(
      MakeIdentityReluPattern("Downstream"),
      MakeFastGeluReplacement(),
      MakeConstraints(), {}, 2, 100));

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(
      CountOp(model->MainGraph(), kMSDomain, "FastGelu"), 1u);
  EXPECT_EQ(
      CountOp(model->MainGraph(), kOnnxDomain, "Identity"), 1u);
  EXPECT_EQ(CountOp(model->MainGraph(), kOnnxDomain, "Relu"), 0u);
}

TEST_F(FusionRewriterTest, SelectsDisjointRulesInOneBatch) {
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[2] x, float[2] y) =>
             (float[2] out_x, float[2] out_y) {
  x_mid = Identity(x)
  out_x = Identity(x_mid)
  y_mid = Identity(y)
  out_y = Relu(y_mid)
})");
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      MakeTwoIdentityPattern(),
      MakeFastGeluReplacement(), MakeConstraints(), {}, 1));
  rules.push_back(MakeRule(
      MakeIdentityReluPattern(),
      MakeFastGeluReplacement(), MakeConstraints(), {}, 2));

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 2u);
  EXPECT_EQ(result.epochs_completed, 1u);
  EXPECT_EQ(
      CountOp(model->MainGraph(), kMSDomain, "FastGelu"), 2u);
}

TEST_F(FusionRewriterTest, DefersBoundaryAdjacentPlans) {
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[2] x) => (float[2] out) {
  first = Identity(x)
  second = Identity(first)
  third = Identity(second)
  out = Identity(third)
})");
  auto rule_set = MakeIdentityRuleSet();

  const FusionRewriteResult result = rule_set->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 2u);
  EXPECT_EQ(result.epochs_completed, 2u);
  EXPECT_EQ(
      CountOp(model->MainGraph(), kMSDomain, "FastGelu"), 2u);
}

TEST_F(FusionRewriterTest, ReplacementEnablesRuleNextEpoch) {
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[2] x) => (float[2] out) {
  first = Identity(x)
  second = Identity(first)
  out = Identity(second)
})");
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      MakeTwoIdentityPattern(),
      MakeReplacementCall(kOnnxDomain, "Relu", 13),
      MakeConstraints(), {}, 1));
  rules.push_back(MakeRule(
      MakeIdentityReluPattern(),
      MakeFastGeluReplacement(), MakeConstraints(), {}, 2));

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules))->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 2u);
  EXPECT_EQ(result.epochs_completed, 2u);
  EXPECT_EQ(model->MainGraph().NumberOfNodes(), 1u);
  EXPECT_EQ(
      CountOp(model->MainGraph(), kMSDomain, "FastGelu"), 1u);
}

TEST_F(FusionRewriterTest, DoesNotUseStaleIteratorOrNodeHandle) {
  constexpr size_t kRegionCount = 32;
  auto model = MakeIndependentRegionsModel(kRegionCount);
  auto rule_set = MakeIdentityRuleSet();

  const FusionRewriteResult result = rule_set->Apply(*model);

  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, kRegionCount);
  EXPECT_EQ(model->MainGraph().NumberOfNodes(), kRegionCount);
  EXPECT_EQ(
      CountOp(model->MainGraph(), kMSDomain, "FastGelu"),
      kRegionCount);
  AssertResolved(model->MainGraph());
}

struct EpochObservation {
  size_t epoch;
  size_t nodes_before;
  size_t nodes_after;
};

void RecordEpoch(
    void* state, size_t epoch,
    size_t nodes_before, size_t nodes_after) {
  static_cast<std::vector<EpochObservation>*>(state)->push_back(
      {epoch, nodes_before, nodes_after});
}

TEST_F(FusionRewriterTest, StrictlyDecreasesNodeCountEachSuccessfulEpoch) {
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[2] x) => (float[2] out) {
  first = Identity(x)
  second = Identity(first)
  third = Identity(second)
  out = Identity(third)
})");
  auto rule_set = MakeIdentityRuleSet();
  std::vector<EpochObservation> observations;
  FusionExecutionControls controls;
  controls.epoch_observer = RecordEpoch;
  controls.epoch_observer_state = &observations;

  const FusionRewriteResult result =
      FusionRuleSetTestAccess::Apply(
          *rule_set, model->MainGraph(), controls);

  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(observations.size(), 2u);
  for (const auto& observation : observations) {
    EXPECT_LT(observation.nodes_after, observation.nodes_before)
        << observation.epoch;
  }
}

TEST_F(FusionRewriterTest, DeterministicAcrossUnrelatedRuleNoise) {
  auto apply = [](bool include_noise) {
    auto model = MakeTwoIdentityModel();
    std::vector<FusionRule> rules;
    if (include_noise) {
      rules.push_back(MakeRule(
          MakeIdentityReluPattern("Noise"),
          MakeReplacementCall(kOnnxDomain, "Relu", 13),
          MakeConstraints(), {}, 99, -100));
    }
    rules.push_back(MakeRule(
        MakeTwoIdentityPattern(),
        MakeFastGeluReplacement(), MakeConstraints(), {}, 1));
    const FusionRewriteResult result =
        MakeRuleSet(std::move(rules))->Apply(*model);
    EXPECT_STATUS_OK(result.status);
    return SerializeGraph(model->MainGraph());
  };

  EXPECT_EQ(apply(false), apply(true));
}

TEST_F(FusionRewriterTest, ZeroRuleAttemptBudgetDoesNoWork) {
  auto model = MakeTwoIdentityModel();
  const std::string before = SerializeGraph(model->MainGraph());
  FusionRuleSetOptions options;
  options.max_rule_attempts = 0;
  auto rule_set = MakeIdentityRuleSet(
      MakeConstraints(), {}, options);

  const FusionRewriteResult result = rule_set->Apply(*model);

  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
}

TEST_F(FusionRewriterTest, ReplacementBudgetFailsBeforeCurrentBatchMutates) {
  auto bounded_model = MakeIndependentRegionsModel(2);
  const std::string before = SerializeGraph(bounded_model->MainGraph());
  FusionRuleSetOptions bounded_options;
  bounded_options.max_replacements = 1;
  auto bounded_rule_set = MakeIdentityRuleSet(
      MakeConstraints(), {}, bounded_options);
  const FusionRewriteResult bounded_result =
      bounded_rule_set->Apply(*bounded_model);
  EXPECT_FALSE(bounded_result.status.IsOK());
  EXPECT_EQ(bounded_result.replacements_applied, 0u);
  EXPECT_EQ(SerializeGraph(bounded_model->MainGraph()), before);

  auto permissive_model = MakeIndependentRegionsModel(2);
  FusionRuleSetOptions permissive_options;
  permissive_options.max_replacements = 2;
  auto permissive_rule_set = MakeIdentityRuleSet(
      MakeConstraints(), {}, permissive_options);
  const FusionRewriteResult permissive_result =
      permissive_rule_set->Apply(*permissive_model);
  ASSERT_STATUS_OK(permissive_result.status);
  EXPECT_EQ(permissive_result.replacements_applied, 2u);
}

TEST_F(FusionRewriterTest, ConditionBudgetCountsRejectedCandidates) {
  auto model = MakeIndependentRegionsModel(3);
  const std::string before = SerializeGraph(model->MainGraph());
  FusionRuleSetOptions options;
  options.max_condition_evaluations = 2;
  FusionMatchPredicate reject =
      [](const FusionMatchContext&,
         FusionConditionResult& result) -> common::Status {
    result.decision = FusionConditionDecision::kNotSatisfied;
    return Status::OK();
  };
  auto rule_set = MakeIdentityRuleSet(
      MakeConstraints(), std::move(reject), options);

  const FusionRewriteResult result = rule_set->Apply(*model);

  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
}

TEST_F(FusionRewriterTest, ConstraintConstructionBudgetsRejectOversizedPrograms) {
  {
    auto model = MakeTwoIdentityModel();
    FusionRuleSetOptions options;
    options.max_constraint_nodes = 2;
    auto rule_set = MakeIdentityRuleSet(
        MakeConstraints(FusionConstraint::AllOf({
            FusionConstraint::IsTensor(
                FusionValueRef::FormalInput(0)),
            FusionConstraint::RankIs(
                FusionValueRef::FormalInput(0), 1),
        })),
        {}, options);
    const FusionRewriteResult result = rule_set->Apply(*model);
    EXPECT_FALSE(result.status.IsOK());
    EXPECT_EQ(result.replacements_applied, 0u);
  }
  {
    auto model = MakeTwoIdentityModel();
    FusionRuleSetOptions options;
    options.max_dimension_equivalence_operands = 1;
    FusionDimensionEquivalenceClass dimensions{
        "D",
        {{FusionValueRef::FormalInput(0), 0},
         {FusionValueRef::FormalOutput(0), 0}},
        FusionUnknownPolicy::kReject};
    auto rule_set = MakeIdentityRuleSet(
        MakeConstraints(
            FusionConstraint::AllOf({}), {std::move(dimensions)}),
        {}, options);
    const FusionRewriteResult result = rule_set->Apply(*model);
    EXPECT_FALSE(result.status.IsOK());
    EXPECT_EQ(result.replacements_applied, 0u);
  }
}

TEST_F(FusionRewriterTest, AggregatePatternBudgetsFailBeforeMutation) {
  struct BudgetCase {
    FunctionProto pattern;
    FusionRuleSetOptions options;
  };
  std::vector<BudgetCase> cases;
  {
    FusionRuleSetOptions options;
    options.max_pattern_nodes = 1;
    cases.push_back({MakeTwoIdentityPattern(), options});
  }
  {
    FusionRuleSetOptions options;
    options.max_literal_bytes = 0;
    cases.push_back({MakeGeluPattern(),
                     options});
  }
  {
    FusionRuleSetOptions options;
    options.max_attribute_bytes = 0;
    cases.push_back({MakeAttributedPattern(), options});
  }

  for (auto& budget_case : cases) {
    auto model = MakeTwoIdentityModel();
    const std::string before = SerializeGraph(model->MainGraph());
    std::vector<FusionRule> rules;
    rules.push_back(MakeRule(
        std::move(budget_case.pattern), MakeFastGeluReplacement()));
    const FusionRewriteResult result =
        MakeRuleSet(std::move(rules), budget_case.options)->Apply(*model);
    EXPECT_FALSE(result.status.IsOK());
    EXPECT_EQ(result.replacements_applied, 0u);
    EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
  }
}

TEST_F(FusionRewriterTest, EpochBudgetFailsBeforeNextBatch) {
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "com.microsoft" : 1]>
target_graph (float[2] x) => (float[2] out) {
  first = Identity(x)
  second = Identity(first)
  out = Identity(second)
})");
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      MakeTwoIdentityPattern(),
      MakeReplacementCall(kOnnxDomain, "Relu", 13),
      MakeConstraints(), {}, 1));
  rules.push_back(MakeRule(
      MakeIdentityReluPattern(),
      MakeFastGeluReplacement(), MakeConstraints(), {}, 2));
  FusionRuleSetOptions options;
  options.max_epochs = 1;

  const FusionRewriteResult result =
      MakeRuleSet(std::move(rules), options)->Apply(*model);

  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(
      CountOp(model->MainGraph(), kMSDomain, "FastGelu"), 0u);
}

common::Status InjectResolveFailure(
    Graph&, const Graph::ResolveOptions&) {
  return ORT_MAKE_STATUS(
      ONNXRUNTIME, FAIL, "injected resolve failure");
}

TEST_F(FusionRewriterTest, PostMutationResolveFailureReportsAppliedCount) {
  auto model = MakeTwoIdentityModel();
  auto rule_set = MakeIdentityRuleSet();
  FusionExecutionControls controls;
  controls.resolve_graph = InjectResolveFailure;

  const FusionRewriteResult result =
      FusionRuleSetTestAccess::Apply(
          *rule_set, model->MainGraph(), controls);

  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_TRUE(model->MainGraph().GraphResolveNeeded());
}

TEST_F(FusionRewriterTest, FunctionExtractorFacadeEmitsSameRegisteredFunctionCall) {
  const FunctionProto function = MakeTwoIdentityPattern("RegisteredIdentity");
  const auto make_model = [&]() {
    const std::array<FunctionProto, 1> functions{function};
    return MakeModelFromText(
        R"(<ir_version: 8, opset_import: ["" : 13, "ort.pattern" : 1]>
target_graph (float[2] x) => (float[2] out) {
  intermediate = Identity(x)
  out = Identity(intermediate)
})",
        functions);
  };

  auto extractor_model = make_model();
  const FunctionExtractionResult extractor_result =
      FunctionExtractor(function).Extract(
          extractor_model->MainGraph());
  ASSERT_STATUS_OK(extractor_result.status);
  ASSERT_EQ(extractor_result.replacements_applied, 1u);

  auto fusion_model = make_model();
  std::vector<FusionRule> rules;
  rules.push_back(MakeRule(
      function,
      MakeReplacementCall(
          function.domain(), function.name(), 1)));
  const FusionRewriteResult fusion_result =
      MakeRuleSet(std::move(rules))->Apply(*fusion_model);
  ASSERT_STATUS_OK(fusion_result.status);
  ASSERT_EQ(fusion_result.replacements_applied, 1u);

  const Node& extractor_call = FindOnlyOp(
      extractor_model->MainGraph(),
      function.domain(), function.name());
  const Node& fusion_call = FindOnlyOp(
      fusion_model->MainGraph(),
      function.domain(), function.name());
  EXPECT_EQ(fusion_call.Domain(), extractor_call.Domain());
  EXPECT_EQ(fusion_call.OpType(), extractor_call.OpType());
  EXPECT_EQ(fusion_call.Overload(), extractor_call.Overload());
  ASSERT_EQ(fusion_call.InputDefs().size(),
            extractor_call.InputDefs().size());
  ASSERT_EQ(fusion_call.OutputDefs().size(),
            extractor_call.OutputDefs().size());
  EXPECT_EQ(fusion_call.InputDefs()[0]->Name(),
            extractor_call.InputDefs()[0]->Name());
  EXPECT_EQ(fusion_call.OutputDefs()[0]->Name(),
            extractor_call.OutputDefs()[0]->Name());
}

TEST_F(FusionRewriterTest, FunctionExtractorStillRequiresRegisteredIdentity) {
  const FunctionProto function =
      MakeTwoIdentityPattern("UnregisteredIdentity");
  auto model = MakeTwoIdentityModel();
  const std::string before = SerializeGraph(model->MainGraph());

  const FunctionExtractionResult result =
      FunctionExtractor(function).Extract(model->MainGraph());

  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(SerializeGraph(model->MainGraph()), before);
}

}  // namespace

}  // namespace test
}  // namespace onnxruntime
