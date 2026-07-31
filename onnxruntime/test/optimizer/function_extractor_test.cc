// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/common/logging/logging.h"
#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/function_extractor.h"
#include "core/optimizer/function_extractor_matcher.h"
#include "core/optimizer/function_extractor_pattern.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/function.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"

namespace onnxruntime {
namespace test {

namespace {

using FunctionProto = ONNX_NAMESPACE::FunctionProto;
using NodeDef = ONNX_NAMESPACE::FunctionBodyHelper::NodeDef;

constexpr const char* kFunctionDomain = "test.function";
constexpr int kOnnxOpset = 13;

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
  for (const FunctionProto& function_proto : function_protos) {
    model_proto.add_functions()->CopyFrom(function_proto);
  }

  std::shared_ptr<Model> model;
  ORT_THROW_IF_ERROR(Model::Load(
      std::move(model_proto), model, nullptr,
      DefaultLoggingManager().DefaultLogger()));
  ORT_ENFORCE(model != nullptr, "Parsed model load returned null.");
  return model;
}

std::shared_ptr<Model> MakeModelFromText(
    std::string_view source,
    const FunctionProto& function_proto) {
  const std::vector<FunctionProto> function_protos{function_proto};
  return MakeModelFromText(source, function_protos);
}

std::string TensorTypeText(std::string_view element_type,
                           gsl::span<const int64_t> dimensions) {
  std::string result{element_type};
  result += "[";
  for (size_t i = 0; i < dimensions.size(); ++i) {
    if (i != 0) {
      result += ", ";
    }
    result += std::to_string(dimensions[i]);
  }
  result += "]";
  return result;
}

class FunctionExtractorGraphBuilder final : public ::onnxruntime::test::ModelTestBuilder {
 public:
  using ::onnxruntime::test::ModelTestBuilder::MakeIntermediate;
  using ::onnxruntime::test::ModelTestBuilder::MakeOutput;
  using ::onnxruntime::test::ModelTestBuilder::ModelTestBuilder;

  template <typename T>
  NodeArg* MakeIntermediate(std::initializer_list<int64_t> shape) {
    return ::onnxruntime::test::ModelTestBuilder::MakeIntermediate<T>(
        std::vector<int64_t>{shape});
  }

  template <typename T>
  NodeArg* MakeOutput(std::initializer_list<int64_t> shape) {
    return ::onnxruntime::test::ModelTestBuilder::MakeOutput<T>(
        std::vector<int64_t>{shape});
  }
};

FunctionProto MakeFunction(std::string name,
                           gsl::span<const NodeDef> node_defs,
                           gsl::span<const std::string> inputs,
                           gsl::span<const std::string> outputs) {
  // Retained for deliberately malformed protobufs and the generated high-arity
  // identity-snapshot case, where parsing would either reject the input first
  // or obscure the protobuf mutation being tested.
  FunctionProto function_proto;
  function_proto.set_domain(kFunctionDomain);
  function_proto.set_name(std::move(name));

  for (const auto& input : inputs) {
    function_proto.add_input(input);
  }

  for (const auto& output : outputs) {
    function_proto.add_output(output);
  }

  const std::vector<NodeDef> node_defs_copy{node_defs.begin(), node_defs.end()};
  for (const auto& node : ONNX_NAMESPACE::FunctionBodyHelper::BuildNodes(node_defs_copy)) {
    function_proto.add_node()->CopyFrom(node);
  }

  auto& opset_import = *function_proto.add_opset_import();
  opset_import.set_domain(kOnnxDomain);
  opset_import.set_version(kOnnxOpset);
  return function_proto;
}

FunctionProto MakeLinearFunction(std::string name = "Linear") {
  return ParseFunction("<opset_import: [\"\" : 13], domain: \"test.function\">\n" +
                       name + R"( (x, y) => (out) {
  sum = Add(x, y)
  out = Relu(sum)
})");
}

FunctionProto MakeLiteralFunction(std::string name = "Literal") {
  return ParseFunction("<opset_import: [\"\" : 13], domain: \"test.function\">\n" +
                       name + R"( (x) => (out) {
  one = Constant <value = float {1.0}> ()
  sum = Add(x, one)
  out = Relu(sum)
})");
}

std::shared_ptr<Model> MakeModel(std::vector<FunctionProto> function_protos) {
  return MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph () => () {
})",
      function_protos);
}

std::shared_ptr<Model> MakeModel(const FunctionProto& function_proto) {
  return MakeModel(std::vector<FunctionProto>{function_proto});
}

std::shared_ptr<Model> MakeLinearTargetModel(const FunctionProto& function_proto) {
  return MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y) => (float[2] out) {
  sum = Add(x, y)
  out = Relu(sum)
})",
      function_proto);
}

size_t CountOp(const Graph& graph, std::string_view domain, std::string_view op_type) {
  size_t count = 0;
  for (const auto& node : graph.Nodes()) {
    if (node.Domain() == domain && node.OpType() == op_type) {
      ++count;
    }
  }
  return count;
}

Node& FindOnlyOp(Graph& graph, std::string_view domain, std::string_view op_type) {
  Node* result = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.Domain() == domain && node.OpType() == op_type) {
      EXPECT_EQ(result, nullptr) << "Expected exactly one " << domain << "." << op_type;
      result = &node;
    }
  }

  EXPECT_NE(result, nullptr) << "Expected one " << domain << "." << op_type;
  return *result;
}

void AssertResolved(const Graph& graph) {
  ASSERT_FALSE(graph.GraphResolveNeeded());
  for (const auto& node : graph.Nodes()) {
    ASSERT_NE(node.Op(), nullptr) << node.Name();
  }
}

void AssertCallIO(const Node& node,
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

std::shared_ptr<Model> SerializeAndReload(Model& model) {
  std::string serialized_model;
  EXPECT_TRUE(model.ToProto().SerializeToString(&serialized_model));

  ONNX_NAMESPACE::ModelProto model_proto;
  EXPECT_TRUE(model_proto.ParseFromString(serialized_model));

  std::shared_ptr<Model> reloaded_model;
  EXPECT_STATUS_OK(Model::Load(std::move(model_proto), reloaded_model, nullptr,
                               DefaultLoggingManager().DefaultLogger()));
  return reloaded_model;
}

void BuildLinearTarget(Graph& graph,
                       NodeArg*& x,
                       NodeArg*& y,
                       NodeArg*& sum,
                       NodeArg*& output) {
  // Used only to keep the target graph deliberately unresolved.
  FunctionExtractorGraphBuilder builder(graph);
  x = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  y = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  sum = builder.MakeIntermediate<float>({2});
  output = builder.MakeOutput<float>({2});
  builder.AddNode("Add", {x, y}, {sum});
  builder.AddNode("Relu", {sum}, {output});
  builder.SetGraphOutputs();
}

common::Status FailGraphResolve(Graph&, const Graph::ResolveOptions&) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "injected resolve failure");
}

}  // namespace

class FunctionExtractorTest : public ::testing::Test {
 protected:
  static void ExpectConstructionRejected(FunctionProto function_proto) {
    Model model("InvalidFunctionExtractorTest", false,
                DefaultLoggingManager().DefaultLogger());
    ASSERT_STATUS_OK(model.MainGraph().Resolve());

    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(model.MainGraph());
    EXPECT_FALSE(result.status.IsOK());
    EXPECT_EQ(result.replacements_applied, 0u);
    EXPECT_EQ(model.MainGraph().NumberOfNodes(), 0u);
  }
};

// Pattern validation and registration.

TEST_F(FunctionExtractorTest, RejectsInvalidFormalNames) {
  const std::vector<NodeDef> nodes{
      {{"sum"}, "Add", {"x", "y"}},
      {{"out"}, "Relu", {"sum"}},
  };

  for (const auto& inputs : std::vector<std::vector<std::string>>{
           {"", "y"}, {"x", "x"}, {"x", "out"}}) {
    SCOPED_TRACE(::testing::PrintToString(inputs));
    ExpectConstructionRejected(MakeFunction("InvalidInputs", nodes, inputs, std::vector<std::string>{"out"}));
  }

  for (const auto& outputs : std::vector<std::vector<std::string>>{
           {""}, {"out", "out"}}) {
    SCOPED_TRACE(::testing::PrintToString(outputs));
    ExpectConstructionRejected(MakeFunction("InvalidOutputs", nodes, std::vector<std::string>{"x", "y"}, outputs));
  }

  FunctionProto internally_produced_input =
      MakeFunction("ProducedInput", nodes, std::vector<std::string>{"sum", "y"},
                   std::vector<std::string>{"out"});
  ExpectConstructionRejected(std::move(internally_produced_input));
}

TEST_F(FunctionExtractorTest, RejectsInvalidAttributes) {
  // Duplicate protobuf fields must be constructed directly because the parser
  // would normalize or reject them before FunctionExtractor sees them.
  FunctionProto duplicate_declaration = MakeLinearFunction("DuplicateAttribute");
  duplicate_declaration.add_attribute("axis");
  duplicate_declaration.add_attribute("axis");
  ExpectConstructionRejected(std::move(duplicate_declaration));

  FunctionProto duplicate_default = MakeLinearFunction("DuplicateDefault");
  duplicate_default.add_attribute("axis");
  duplicate_default.add_attribute_proto()->CopyFrom(
      ONNX_NAMESPACE::MakeAttribute("axis", int64_t{0}));
  duplicate_default.add_attribute_proto()->CopyFrom(
      ONNX_NAMESPACE::MakeAttribute("axis", int64_t{1}));
  ExpectConstructionRejected(std::move(duplicate_default));

  FunctionProto referenced_attribute = MakeLinearFunction("ReferencedAttribute");
  // The parser cannot encode both ref_attr_name and its required AttributeProto type.
  auto& attribute = *referenced_attribute.mutable_node(0)->add_attribute();
  attribute.set_name("axis");
  attribute.set_ref_attr_name("axis");
  attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  ExpectConstructionRejected(std::move(referenced_attribute));
}

TEST_F(FunctionExtractorTest, RejectsUnusedRequiredFunctionAttribute) {
  ExpectConstructionRejected(ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
RequiredAttribute <axis> (x, y) => (out) {
  sum = Add(x, y)
  out = Relu(sum)
})"));
}

TEST_F(FunctionExtractorTest, RejectsMalformedDataflow) {
  const std::vector<std::string> inputs{"x", "y"};
  const std::vector<std::string> outputs{"out"};

  const std::vector<std::vector<NodeDef>> invalid_bodies{
      {{{"sum"}, "Add", {"x", "undefined"}}, {{"out"}, "Relu", {"sum"}}},
      {{{"sum"}, "Add", {"x", "y"}}, {{"sum"}, "Mul", {"x", "y"}}, {{"out"}, "Relu", {"sum"}}},
      {{{"a"}, "Add", {"b", "x"}}, {{"b"}, "Mul", {"a", "y"}}, {{"out"}, "Relu", {"a"}}},
      {{{"sum"}, "Add", {"x", "y"}}, {{"dead"}, "Mul", {"x", "y"}}, {{"out"}, "Relu", {"sum"}}},
      {{{"sum"}, "Add", {"x", ""}}, {{"out"}, "Relu", {"sum"}}},
  };

  for (size_t i = 0; i < invalid_bodies.size(); ++i) {
    SCOPED_TRACE(i);
    ExpectConstructionRejected(MakeFunction("Malformed" + std::to_string(i),
                                            invalid_bodies[i], inputs, outputs));
  }
}

TEST_F(FunctionExtractorTest, RejectsDisconnectedMultiOutputBody) {
  ExpectConstructionRejected(ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
DisconnectedOutputs (x, y, z, w) => (left, right) {
  left = Add(x, y)
  right = Mul(z, w)
})"));
}

TEST_F(FunctionExtractorTest, RejectsOutputUnreachableOperations) {
  const std::vector<std::string_view> function_sources{
      R"(<opset_import: ["" : 13], domain: "test.function">
OutputConsumedInternally (x, y) => (out) {
  out = Add(x, y)
  after = Relu(out)
})",
      R"(<opset_import: ["" : 13], domain: "test.function">
DeadOperation (x, y) => (out) {
  sum = Add(x, y)
  out = Relu(sum)
  dead = Mul(x, y)
})",
  };

  for (const std::string_view function_source : function_sources) {
    SCOPED_TRACE(function_source);
    ExpectConstructionRejected(ParseFunction(function_source));
  }
}

TEST_F(FunctionExtractorTest, RejectsConstantFormalOutput) {
  const std::vector<std::string> no_inputs;
  const std::vector<std::string> outputs{"out"};
  const std::vector<NodeDef> constant_body{
      ONNX_NAMESPACE::FunctionBodyHelper::Const<float>("literal", 1.0f),
      {{"out"}, "Identity", {"literal"}},
  };
  FunctionProto function_proto = MakeFunction("ConstantOutput", constant_body, no_inputs, outputs);
  function_proto.mutable_output(0)->assign("literal");
  ExpectConstructionRejected(std::move(function_proto));
}

TEST_F(FunctionExtractorTest, RejectsSingleOperationPattern) {
  ExpectConstructionRejected(ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
Single (x) => (out) {
  out = Relu(x)
})"));
}

TEST_F(FunctionExtractorTest, RejectsUnsupportedBodyFeatures) {
  ExpectConstructionRejected(ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
ControlFlow (condition) => (out) {
  branch = If(condition) <
    then_branch = then_body () => () {},
    else_branch = else_body () => () {}
  >
  out = Identity(branch)
})"));
}

TEST_F(FunctionExtractorTest, RejectsUnregisteredFunction) {
  FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13]>
agraph (float[2] x, float[2] y) => (float[2] out) {
  sum = Add(x, y)
  out = Relu(sum)
})");
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());
  const size_t original_node_count = graph.NumberOfNodes();

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(graph.NumberOfNodes(), original_node_count);
}

TEST_F(FunctionExtractorTest, RejectsDifferentRegisteredDefinition) {
  const FunctionProto registered_function = MakeLinearFunction();
  const FunctionProto requested_function = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
Linear (x, y) => (out) {
  sum = Sub(x, y)
  out = Relu(sum)
})");
  auto model = MakeLinearTargetModel(registered_function);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(requested_function);
  const FunctionExtractionResult result = extractor.Extract(graph);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(graph.NumberOfNodes(), 2u);
}

TEST_F(FunctionExtractorTest, RejectsContextDependentSchemaFunction) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "">
Celu (x) => (out) {
  scaled = Mul(x, x)
  out = Relu(scaled)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13]>
agraph () => () {
})");
  ASSERT_STATUS_OK(model->MainGraph().Resolve());
  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(*model);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, ResolvesNestedLocalFunctionIdentity) {
  const FunctionProto nested_function = MakeLinearFunction("Nested");
  const FunctionProto outer_function = ParseFunction(
      R"(<opset_import: ["" : 13, "test.function" : 1], domain: "test.function">
Outer (x, y) => (out) {
  nested_out = test.function.Nested(x, y)
  out = Identity(nested_out)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y) => (float[2] out) {
  nested_out = test.function.Nested(x, y)
  out = Identity(nested_out)
})",
      std::vector<FunctionProto>{nested_function, outer_function});
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(outer_function);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, outer_function.name()), 1u);
}

TEST_F(FunctionExtractorTest, RejectsUnresolvedTargetGraph) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  // The target must remain unresolved, which cannot be preserved through Model::Load.
  NodeArg* x;
  NodeArg* y;
  NodeArg* sum;
  NodeArg* output;
  BuildLinearTarget(graph, x, y, sum, output);
  ASSERT_TRUE(graph.GraphResolveNeeded());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(graph.NumberOfNodes(), 2u);
}

TEST_F(FunctionExtractorTest, RejectsImpureOrUnknownOperation) {
  const std::vector<std::string_view> functions{
      R"(<opset_import: ["" : 13], domain: "test.function">
Impure0 (x, y) => (out) {
  random = RandomNormal()
  out = Add(x, random)
})",
      R"(<opset_import: ["" : 13], domain: "test.function">
Impure1 (x, y) => (out) {
  custom = Unknown(x)
  out = Add(custom, y)
})",
  };

  for (size_t i = 0; i < functions.size(); ++i) {
    SCOPED_TRACE(i);
    const FunctionProto function_proto = ParseFunction(functions[i]);
    auto model = MakeModel(function_proto);
    ASSERT_STATUS_OK(model->MainGraph().Resolve());
    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(model->MainGraph());
    EXPECT_FALSE(result.status.IsOK());
    EXPECT_EQ(result.replacements_applied, 0u);
  }
}

TEST_F(FunctionExtractorTest, EnforcesResourceBudgetsBeforeMutation) {
  const FunctionProto function_proto = MakeLinearFunction();
  struct BudgetCase {
    const char* name;
    FunctionExtractorOptions options;
  };

  FunctionExtractorOptions pattern_node_limit;
  pattern_node_limit.max_pattern_nodes = 1;
  FunctionExtractorOptions target_node_limit;
  target_node_limit.max_target_nodes = 1;
  FunctionExtractorOptions root_tuple_limit;
  root_tuple_limit.max_output_root_tuples = 0;
  FunctionExtractorOptions worklist_limit;
  worklist_limit.max_worklist_bindings = 0;
  const std::vector<BudgetCase> budget_cases{
      {"pattern node limit", pattern_node_limit},
      {"target node limit", target_node_limit},
      {"output root tuple limit", root_tuple_limit},
      {"worklist binding limit", worklist_limit},
  };

  for (const auto& budget_case : budget_cases) {
    SCOPED_TRACE(budget_case.name);
    auto model = MakeLinearTargetModel(function_proto);
    Graph& graph = model->MainGraph();
    ASSERT_STATUS_OK(graph.Resolve());

    FunctionExtractor extractor(function_proto, budget_case.options);
    const FunctionExtractionResult result = extractor.Extract(graph);
    EXPECT_FALSE(result.status.IsOK());
    EXPECT_EQ(result.replacements_applied, 0u);
    EXPECT_EQ(graph.NumberOfNodes(), 2u);
  }
}

TEST_F(FunctionExtractorTest, RejectsHighArityPatternBeforeMutation) {
  // Keep the generated 128-input graph programmatic so pointer/index identity
  // can be snapshotted across the rejected mutation attempt.
  constexpr size_t input_count = 128;
  std::vector<std::string> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    inputs.push_back("x" + std::to_string(i));
  }
  const std::vector<NodeDef> nodes{
      {{"joined"}, "Concat", inputs, {ONNX_NAMESPACE::MakeAttribute("axis", int64_t{0})}},
      {{"out"}, "Identity", {"joined"}},
  };
  const FunctionProto function_proto =
      MakeFunction("HighArity", nodes, inputs, std::vector<std::string>{"out"});
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  std::vector<NodeArg*> target_inputs;
  target_inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    target_inputs.push_back(builder.MakeInput<float>({1}, 0.0f, 1.0f));
  }
  NodeArg* joined = builder.MakeIntermediate<float>({static_cast<int64_t>(input_count)});
  NodeArg* output = builder.MakeOutput<float>({static_cast<int64_t>(input_count)});
  NodeAttributes attributes{{"axis", ONNX_NAMESPACE::MakeAttribute("axis", int64_t{0})}};
  builder.AddNode("Concat", target_inputs, {joined}, kOnnxDomain, &attributes);
  builder.AddNode("Identity", {joined}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  const size_t node_count_before = graph.NumberOfNodes();
  std::vector<std::pair<NodeIndex, const Node*>> node_identities_before;
  for (const Node& node : graph.Nodes()) {
    node_identities_before.emplace_back(node.Index(), &node);
  }
  const std::string graph_proto_before = graph.ToGraphProto().SerializeAsString();
  ASSERT_FALSE(graph.GraphResolveNeeded());

  FunctionExtractorOptions options;
  options.max_worklist_bindings = 16;
  FunctionExtractor extractor(function_proto, options);
  const FunctionExtractionResult result = extractor.Extract(graph);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(graph.NumberOfNodes(), node_count_before);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 0u);
  EXPECT_FALSE(graph.GraphResolveNeeded());
  EXPECT_EQ(graph.ToGraphProto().SerializeAsString(), graph_proto_before);

  std::vector<std::pair<NodeIndex, const Node*>> node_identities_after;
  for (const Node& node : graph.Nodes()) {
    node_identities_after.emplace_back(node.Index(), &node);
  }
  EXPECT_EQ(node_identities_after, node_identities_before);
}

// Deterministic structural matching.

TEST_F(FunctionExtractorTest, ExtractsLinearPattern) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeLinearTargetModel(function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* x = graph.GetNodeArg("x");
  NodeArg* y = graph.GetNodeArg("y");
  NodeArg* output = graph.GetNodeArg("out");
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_STATUS_OK(graph.Resolve());
  const std::vector<std::string> expected_inputs{x->Name(), y->Name()};
  const std::vector<std::string> expected_outputs{output->Name()};

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(*model);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Add"), 0u);
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Relu"), 0u);
  const Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  AssertCallIO(call, expected_inputs, expected_outputs);
  AssertResolved(graph);
}

TEST_F(FunctionExtractorTest, ExtractsBranchedMultiOutputPattern) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
Branched (x, y) => (scaled, activated) {
  sum = Add(x, y)
  scaled = Mul(sum, y)
  activated = Relu(sum)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y) => (float[2] scaled, float[2] activated) {
  sum = Add(x, y)
  scaled = Mul(sum, y)
  activated = Relu(sum)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* x = graph.GetNodeArg("x");
  NodeArg* y = graph.GetNodeArg("y");
  NodeArg* scaled = graph.GetNodeArg("scaled");
  NodeArg* activated = graph.GetNodeArg("activated");
  ASSERT_NE(x, nullptr);
  ASSERT_NE(y, nullptr);
  ASSERT_NE(scaled, nullptr);
  ASSERT_NE(activated, nullptr);
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  AssertCallIO(call, std::vector<std::string>{x->Name(), y->Name()},
               std::vector<std::string>{scaled->Name(), activated->Name()});
}

TEST_F(FunctionExtractorTest, ExtractsDiamondAndProcessesSharedValueOnce) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
Diamond (x, y) => (out) {
  sum = Add(x, y)
  left = Relu(sum)
  right = Identity(sum)
  out = Mul(left, right)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y) => (float[2] out) {
  sum = Add(x, y)
  left = Relu(sum)
  right = Identity(sum)
  out = Mul(left, right)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  using namespace function_extractor_internal;
  const NormalizedFunctionPattern normalized =
      NormalizeFunctionPattern(function_proto, FunctionExtractorOptions{});
  ASSERT_STATUS_OK(normalized.construction_status);
  CompiledFunctionPattern compiled;
  ASSERT_STATUS_OK(CompileFunctionPattern(normalized, graph, compiled));
  TargetGraphSnapshot snapshot;
  ASSERT_STATUS_OK(BuildTargetGraphSnapshot(graph, compiled, FunctionExtractorOptions{}, snapshot));
  std::vector<ReplacementPlan> plans;
  MatcherDiagnostics diagnostics;
  ASSERT_STATUS_OK(DiscoverReplacementPlans(
      compiled, snapshot, FunctionExtractorOptions{}, plans, &diagnostics));
  ASSERT_EQ(plans.size(), 1u);
  EXPECT_EQ(diagnostics.worklist_bindings_processed, normalized.values.size());
  EXPECT_EQ(diagnostics.worklist_bindings_scheduled, normalized.values.size());
}

TEST_F(FunctionExtractorTest, RejectsInconsistentMultiOutputRootTuple) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
TwoRoots (x, y) => (left, right) {
  sum = Add(x, y)
  left = Relu(sum)
  right = Identity(sum)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y, float[2] other) => (float[2] left, float[2] right) {
  sum_a = Add(x, y)
  sum_b = Add(x, other)
  left = Relu(sum_a)
  right = Identity(sum_b)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 0u);
}

TEST_F(FunctionExtractorTest, EnumeratesOutputRootsDeterministically) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y) => (float[2] first, float[2] second) {
  first_sum = Add(x, y)
  first = Relu(first_sum)
  second_sum = Add(x, y)
  second = Relu(second_sum)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  using namespace function_extractor_internal;
  const NormalizedFunctionPattern normalized =
      NormalizeFunctionPattern(function_proto, FunctionExtractorOptions{});
  CompiledFunctionPattern compiled;
  ASSERT_STATUS_OK(CompileFunctionPattern(normalized, graph, compiled));
  TargetGraphSnapshot snapshot;
  ASSERT_STATUS_OK(BuildTargetGraphSnapshot(graph, compiled, FunctionExtractorOptions{}, snapshot));
  std::vector<ReplacementPlan> plans;
  ASSERT_STATUS_OK(DiscoverReplacementPlans(compiled, snapshot, FunctionExtractorOptions{}, plans));
  ASSERT_EQ(plans.size(), 2u);
  EXPECT_LT(plans[0].primary_root_topological_position,
            plans[1].primary_root_topological_position);
}

TEST_F(FunctionExtractorTest, MatchesRepeatedAndAliasedFormalInputs) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
AliasedInputs (x, y) => (out) {
  sum = Add(x, x)
  out = Mul(sum, y)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] input) => (float[2] out) {
  sum = Add(input, input)
  out = Mul(sum, input)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* input = graph.GetNodeArg("input");
  NodeArg* output = graph.GetNodeArg("out");
  ASSERT_NE(input, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  AssertCallIO(call, std::vector<std::string>{input->Name(), input->Name()},
               std::vector<std::string>{output->Name()});
}

TEST_F(FunctionExtractorTest, RejectsReversedInternalOperands) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
Positional (x, y) => (out) {
  difference = Sub(x, y)
  out = Div(difference, y)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y) => (float[2] out) {
  difference = Sub(x, y)
  out = Div(y, difference)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, RootIndexAllowsExternalProducerAndOutputFanout) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] source, float[2] y) => (float[2] output_a, float[2] output_b) {
  x = Abs(source)
  sum = Add(x, y)
  matched_output = Relu(sum)
  output_a = Identity(matched_output)
  output_b = Neg(matched_output)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* matched_output = graph.GetNodeArg("matched_output");
  ASSERT_NE(matched_output, nullptr);
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(graph.GetConsumerNodes(matched_output->Name()).size(), 2u);
}

TEST_F(FunctionExtractorTest, RequiresExactOperatorIdentityAndArity) {
  const FunctionProto function_proto = MakeLinearFunction();
  for (const std::string& second_op : {"Sigmoid", "Neg"}) {
    SCOPED_TRACE(second_op);
    const std::string model_source =
        R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y) => (float[2] out) {
  sum = Add(x, y)
  out = )" +
        second_op + R"((sum)
})";
    auto model = MakeModelFromText(model_source, function_proto);
    Graph& graph = model->MainGraph();
    ASSERT_STATUS_OK(graph.Resolve());

    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(graph);
    ASSERT_STATUS_OK(result.status);
    EXPECT_EQ(result.replacements_applied, 0u);
  }
}

TEST_F(FunctionExtractorTest, RequiresExactEffectiveAttributes) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
Attributes (x) => (out) {
  transposed = Transpose <perm = [1, 0]> (x)
  out = Relu(transposed)
})");

  for (const std::vector<int64_t>& perm :
       {std::vector<int64_t>{1, 0}, std::vector<int64_t>{0, 1}}) {
    SCOPED_TRACE(::testing::PrintToString(perm));
    const std::string model_source =
        R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2, 2] x) => (float[2, 2] out) {
  transposed = Transpose <perm = [)" +
        std::to_string(perm[0]) + ", " + std::to_string(perm[1]) + R"(]> (x)
  out = Relu(transposed)
})";
    auto model = MakeModelFromText(model_source, function_proto);
    Graph& graph = model->MainGraph();
    ASSERT_STATUS_OK(graph.Resolve());

    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(graph);
    ASSERT_STATUS_OK(result.status);
    EXPECT_EQ(result.replacements_applied, perm == std::vector<int64_t>({1, 0}) ? 1u : 0u);
  }
}

TEST_F(FunctionExtractorTest, MatchesOptionalAndVariadicSlotsPositionally) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
OptionalVariadic (x, y) => (out) {
  clipped = Clip(x, "", "")
  out = Concat <axis = 0> (clipped, y)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[1] x, float[1] y) => (float[2] out) {
  clipped = Clip(x, "", "")
  out = Concat <axis = 0> (clipped, y)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
}

TEST_F(FunctionExtractorTest, MatchesOperatorWithOmittedOptionalOutput) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
OmittedOptionalOutput (x) => (out) {
  pooled, "" = MaxPool <kernel_shape = [2, 2]> (x)
  out = Identity(pooled)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[1, 1, 4, 4] x) => (float[1, 1, 3, 3] out) {
  pooled, "" = MaxPool <kernel_shape = [2, 2]> (x)
  out = Identity(pooled)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
}

TEST_F(FunctionExtractorTest, AppliesKnownTypeCompatibilityRules) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
KnownTypes (float[2] x) => (float[2] out) <float[2] intermediate> {
  intermediate = Identity(x)
  out = Identity(intermediate)
})");

  struct FloatShapeCase {
    const char* name;
    std::vector<int64_t> shape;
    size_t expected_replacements;
  };
  const std::vector<FloatShapeCase> float_shape_cases{
      {"compatible", {2}, 1},
      {"rank mismatch", {2, 1}, 0},
      {"dimension mismatch", {3}, 0},
  };
  for (const auto& test_case : float_shape_cases) {
    SCOPED_TRACE(test_case.name);
    const std::string type = TensorTypeText("float", test_case.shape);
    const std::string model_source =
        "<ir_version: 8, opset_import: [\"\" : 13, \"test.function\" : 1]>\n"
        "agraph (" +
        type + " x) => (" + type +
        " out) {\n"
        "  intermediate = Identity(x)\n"
        "  out = Identity(intermediate)\n"
        "}";
    auto model = MakeModelFromText(model_source, function_proto);
    Graph& graph = model->MainGraph();
    ASSERT_STATUS_OK(graph.Resolve());

    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(graph);
    ASSERT_STATUS_OK(result.status);
    EXPECT_EQ(result.replacements_applied, test_case.expected_replacements);
  }

  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (int32[2] x) => (int32[2] out) {
  intermediate = Identity(x)
  out = Identity(intermediate)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, RejectsNonTensorValueInfo) {
  FunctionProto function_proto = MakeLinearFunction("NonTensorValueInfo");
  // Inject the non-tensor TypeProto directly; the surrounding function remains parser-built.
  auto& value_info = *function_proto.add_value_info();
  value_info.set_name("sum");
  value_info.mutable_type()
      ->mutable_sequence_type()
      ->mutable_elem_type()
      ->mutable_tensor_type()
      ->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  ExpectConstructionRejected(std::move(function_proto));
}

// Literal matching and preservation.

TEST_F(FunctionExtractorTest, ComparesFloatingAndTensorBitsExactly) {
  // Raw payloads are required to distinguish signed zero and NaN bit patterns.
  auto make_float_tensor = [](uint32_t bits) {
    ONNX_NAMESPACE::TensorProto tensor;
    tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor.add_dims(1);
    tensor.set_raw_data(&bits, sizeof(bits));
    return tensor;
  };

  using function_extractor_internal::CompareTensorLiterals;
  bool equal = true;
  ASSERT_STATUS_OK(CompareTensorLiterals(
      make_float_tensor(0x00000000u), make_float_tensor(0x80000000u), 1024, equal));
  EXPECT_FALSE(equal);
  ASSERT_STATUS_OK(CompareTensorLiterals(
      make_float_tensor(0x7fc00001u), make_float_tensor(0x7fc00002u), 1024, equal));
  EXPECT_FALSE(equal);
  ASSERT_STATUS_OK(CompareTensorLiterals(
      make_float_tensor(0x7fc00001u), make_float_tensor(0x7fc00001u), 1024, equal));
  EXPECT_TRUE(equal);
}

TEST_F(FunctionExtractorTest, MatchesLiteralFromInitializer) {
  const FunctionProto function_proto = MakeLiteralFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] input) => (float[2] out) <float literal = {1.0}> {
  sum = Add(input, literal)
  out = Relu(sum)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* input = graph.GetNodeArg("input");
  NodeArg* literal = graph.GetNodeArg("literal");
  NodeArg* output = graph.GetNodeArg("out");
  ASSERT_NE(input, nullptr);
  ASSERT_NE(literal, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_STATUS_OK(graph.Resolve());
  const std::string literal_name = literal->Name();

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const ONNX_NAMESPACE::TensorProto* retained_literal = nullptr;
  EXPECT_TRUE(graph.GetInitializedTensor(literal_name, retained_literal));
  EXPECT_NE(retained_literal, nullptr);
  const Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  AssertCallIO(call, std::vector<std::string>{input->Name()},
               std::vector<std::string>{output->Name()});
}

TEST_F(FunctionExtractorTest, MatchesLiteralFromConstantNode) {
  const FunctionProto function_proto = MakeLiteralFunction();
  // Model::Load canonicalizes a parsed Constant node into an initializer. Build
  // this target directly so the test continues to cover a Constant-node witness.
  auto model = MakeModel(function_proto);
  Graph& graph = model->MainGraph();
  FunctionExtractorGraphBuilder builder(graph);
  NodeArg* input = builder.MakeInput<float>({2}, 0.0f, 1.0f);
  NodeArg* literal = builder.MakeIntermediate<float>({});
  NodeArg* sum = builder.MakeIntermediate<float>({2});
  NodeArg* output = builder.MakeOutput<float>({2});
  NodeAttributes constant_attributes{
      {"value", ONNX_NAMESPACE::MakeAttribute(
                    "value", ONNX_NAMESPACE::ToTensor<float>(1.0f))}};
  builder.AddNode("Constant", {}, {literal}, kOnnxDomain, &constant_attributes);
  builder.AddNode("Add", {input, literal}, {sum});
  builder.AddNode("Relu", {sum}, {output});
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Constant"), 1u);
}

TEST_F(FunctionExtractorTest, LeavesSharedAndDeadLiteralWitnesses) {
  const FunctionProto function_proto = MakeLiteralFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] input) => (float[2] graph_output) <float literal = {1.0}> {
  sum = Add(input, literal)
  matched_output = Relu(sum)
  graph_output = Add(matched_output, literal)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* literal = graph.GetNodeArg("literal");
  ASSERT_NE(literal, nullptr);
  ASSERT_STATUS_OK(graph.Resolve());
  const std::string literal_name = literal->Name();

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const ONNX_NAMESPACE::TensorProto* retained_literal = nullptr;
  EXPECT_TRUE(graph.GetInitializedTensor(literal_name, retained_literal));
  EXPECT_NE(retained_literal, nullptr);
}

TEST_F(FunctionExtractorTest, RejectsOverridableInitializerLiteral) {
  const FunctionProto function_proto = MakeLiteralFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] input, float literal = {1.0}) => (float[2] out) {
  sum = Add(input, literal)
  out = Relu(sum)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, RejectsLiteralFormalInputAlias) {
  const FunctionProto function_proto = MakeLiteralFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph () => (float[2] out) <float literal = {1.0}, float[2] sum> {
  sum = Add(literal, literal)
  out = Relu(sum)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, AllowsEqualLiteralsToShareWitness) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
SharedLiteral (x) => (out) {
  one_a = Constant <value = float {1.0}> ()
  one_b = Constant <value = float {1.0}> ()
  sum = Add(x, one_a)
  out = Mul(sum, one_b)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] input) => (float[2] out) <float literal = {1.0}> {
  sum = Add(input, literal)
  out = Mul(sum, literal)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 1u);
}

// Boundary closure, convexity, and graph scopes.

TEST_F(FunctionExtractorTest, RejectsPrivateIntermediateExternalConsumer) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y) => (float[2] matched_output, float[2] side_output) {
  sum = Add(x, y)
  matched_output = Relu(sum)
  side_output = Neg(sum)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, RejectsPrivateIntermediateGraphOutput) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y) => (float[2] sum, float[2] matched_output) {
  sum = Add(x, y)
  matched_output = Relu(sum)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, RejectsPrivateIntermediateImplicitCapture) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y, bool condition) =>
       (float[2] matched_output, float[2] if_output) {
  sum = Add(x, y)
  matched_output = Relu(sum)
  if_output = If(condition) <
    then_branch = then_body () => (float[2] then_output) {
      then_output = Identity(sum)
    },
    else_branch = else_body () => (float[2] else_output) {
      else_output = Identity(sum)
    }
  >
})",
      function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* sum = graph.GetNodeArg("sum");
  ASSERT_NE(sum, nullptr);
  Node& if_node = FindOnlyOp(graph, kOnnxDomain, "If");
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_EQ(if_node.ImplicitInputDefs().size(), 1u);
  ASSERT_EQ(if_node.ImplicitInputDefs()[0]->Name(), sum->Name());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, PreservesFormalOutputConsumersAndImplicitCaptures) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y, bool condition) =>
       (float[2] explicit_output, float[2] if_output) {
  sum = Add(x, y)
  matched_output = Relu(sum)
  explicit_output = Identity(matched_output)
  if_output = If(condition) <
    then_branch = then_body () => (float[2] then_output) {
      then_output = Identity(matched_output)
    },
    else_branch = else_body () => (float[2] else_output) {
      else_output = Identity(matched_output)
    }
  >
})",
      function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* matched_output = graph.GetNodeArg("matched_output");
  ASSERT_NE(matched_output, nullptr);
  Node& if_node = FindOnlyOp(graph, kOnnxDomain, "If");
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_EQ(if_node.ImplicitInputDefs().size(), 1u);

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  AssertResolved(graph);
  const Node& remaining_if = FindOnlyOp(graph, kOnnxDomain, "If");
  ASSERT_EQ(remaining_if.ImplicitInputDefs().size(), 1u);
  EXPECT_EQ(remaining_if.ImplicitInputDefs()[0]->Name(), matched_output->Name());
  EXPECT_EQ(graph.GetConsumerNodes(matched_output->Name()).size(), 2u);
}

TEST_F(FunctionExtractorTest, RejectsDownstreamFormalInputBinding) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
DownstreamInput (x, y) => (out) {
  activated = Relu(x)
  out = Add(activated, y)
})");
  for (const bool bind_formal_directly_to_matched_output : {false, true}) {
    SCOPED_TRACE(bind_formal_directly_to_matched_output);
    const std::string middle =
        bind_formal_directly_to_matched_output
            ? ""
            : "  formal_input_binding = Identity(activated)\n";
    const std::string binding =
        bind_formal_directly_to_matched_output ? "activated" : "formal_input_binding";
    const std::string model_source =
        R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x) => (float[2] out) {
  activated = Relu(x)
)" +
        middle + "  out = Add(activated, " + binding + ")\n}";
    auto model = MakeModelFromText(model_source, function_proto);
    Graph& graph = model->MainGraph();
    ASSERT_STATUS_OK(graph.Resolve());

    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(graph);
    ASSERT_STATUS_OK(result.status);
    EXPECT_EQ(result.replacements_applied, 0u);
  }
}

TEST_F(FunctionExtractorTest, RejectsNonConvexLeaveAndReenterPath) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
NonConvex (x, y) => (out) {
  activated = Relu(x)
  out = Add(activated, y)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x) => (float[2] out) {
  activated = Relu(x)
  outside = Identity(activated)
  out = Add(activated, outside)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
}

TEST_F(FunctionExtractorTest, RejectsProviderControlOrAnnotationMismatch) {
  const FunctionProto function_proto = MakeLinearFunction();
  enum class RejectionReason {
    ProviderAssignment,
    ControlEdge,
    LayeringAnnotation,
  };
  for (const RejectionReason reason :
       {RejectionReason::ProviderAssignment,
        RejectionReason::ControlEdge,
        RejectionReason::LayeringAnnotation}) {
    SCOPED_TRACE(static_cast<int>(reason));
    auto model = MakeLinearTargetModel(function_proto);
    Graph& graph = model->MainGraph();
    NodeArg* x = graph.GetNodeArg("x");
    ASSERT_NE(x, nullptr);
    // Provider, control-edge, and layering metadata are not representable in ONNX text.
    NodeIndex control_source_index = 0;
    NodeIndex control_target_index = 0;
    if (reason == RejectionReason::ControlEdge) {
      Node* add_node = nullptr;
      for (Node& node : graph.Nodes()) {
        if (node.OpType() == "Add") {
          add_node = &node;
          break;
        }
      }
      ASSERT_NE(add_node, nullptr);
      FunctionExtractorGraphBuilder builder(graph);
      NodeArg* control_output = builder.MakeIntermediate<float>({2});
      Node& control_source = builder.AddNode("Identity", {x}, {control_output});
      control_source_index = control_source.Index();
      control_target_index = add_node->Index();
    }
    ASSERT_STATUS_OK(graph.Resolve());
    if (reason == RejectionReason::ControlEdge) {
      ASSERT_TRUE(graph.AddControlEdge(control_source_index, control_target_index));
      using namespace function_extractor_internal;
      const NormalizedFunctionPattern normalized =
          NormalizeFunctionPattern(function_proto, FunctionExtractorOptions{});
      ASSERT_STATUS_OK(normalized.construction_status);
      CompiledFunctionPattern compiled;
      ASSERT_STATUS_OK(CompileFunctionPattern(normalized, graph, compiled));
      TargetGraphSnapshot snapshot;
      ASSERT_STATUS_OK(
          BuildTargetGraphSnapshot(graph, compiled, FunctionExtractorOptions{},
                                   snapshot));
      std::vector<ReplacementPlan> plans;
      ASSERT_STATUS_OK(DiscoverReplacementPlans(
          compiled, snapshot, FunctionExtractorOptions{}, plans));
      EXPECT_TRUE(plans.empty());
      continue;
    }
    auto nodes = graph.Nodes();
    auto node_it = nodes.begin();
    Node& first = *node_it;
    Node& second = *++node_it;
    if (reason == RejectionReason::ProviderAssignment) {
      first.SetExecutionProviderType(kCpuExecutionProvider);
    } else if (reason == RejectionReason::LayeringAnnotation) {
      first.SetLayeringAnnotation("layer-a");
      second.SetLayeringAnnotation("layer-b");
    }

    FunctionExtractor extractor(function_proto);
    const FunctionExtractionResult result = extractor.Extract(graph);
    ASSERT_STATUS_OK(result.status);
    EXPECT_EQ(result.replacements_applied, 0u);
  }
}

TEST_F(FunctionExtractorTest, DoesNotCrossGraphScopes) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y, bool condition) => (float[2] if_output) {
  sum = Add(x, y)
  if_output = If(condition) <
    then_branch = then_body () => (float[2] then_output) {
      then_output = Relu(sum)
    },
    else_branch = else_body () => (float[2] else_output) {
      else_output = Relu(sum)
    }
  >
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Add"), 1u);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 0u);
}

// Batching, application, fixpoint, and persistence.

TEST_F(FunctionExtractorTest, AppliesDisjointMatchesInOneBatch) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y) => (float[2] first, float[2] second) {
  first_sum = Add(x, y)
  first = Relu(first_sum)
  second_sum = Add(x, y)
  second = Relu(second_sum)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 2u);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 2u);
  EXPECT_EQ(graph.NumberOfNodes(), 2u);
}

TEST_F(FunctionExtractorTest, SelectsOverlappingMatchesDeterministically) {
  const FunctionProto function_proto = ParseFunction(
      R"(<opset_import: ["" : 13], domain: "test.function">
Overlapping (x, y) => (sum, out) {
  sum = Add(x, y)
  out = Relu(sum)
})");
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y) => (float[2] first, float[2] second) {
  sum = Add(x, y)
  first = Relu(sum)
  second = Relu(sum)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* sum = graph.GetNodeArg("sum");
  NodeArg* first = graph.GetNodeArg("first");
  ASSERT_NE(sum, nullptr);
  ASSERT_NE(first, nullptr);
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  ASSERT_EQ(call.OutputDefs().size(), 2u);
  EXPECT_EQ(call.OutputDefs()[0]->Name(), sum->Name());
  EXPECT_EQ(call.OutputDefs()[1]->Name(), first->Name());
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Relu"), 1u);
}

TEST_F(FunctionExtractorTest, DefersBoundaryAdjacentMatchToNextPass) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y, float[2] z) => (float[2] out) {
  first_sum = Add(x, y)
  boundary = Relu(first_sum)
  second_sum = Add(boundary, z)
  out = Relu(second_sum)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 2u);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 2u);
  AssertResolved(graph);
}

TEST_F(FunctionExtractorTest, AllowsSharedUnrelatedBoundaryValues) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] shared) => (float[2] first, float[2] second) {
  first_sum = Add(shared, shared)
  first = Relu(first_sum)
  second_sum = Add(shared, shared)
  second = Relu(second_sum)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  EXPECT_EQ(result.replacements_applied, 2u);
}

// Failure semantics and mutation invariants.

TEST_F(FunctionExtractorTest, RejectsStalePlanBeforeMutation) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeLinearTargetModel(function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  using namespace function_extractor_internal;
  const NormalizedFunctionPattern normalized =
      NormalizeFunctionPattern(function_proto, FunctionExtractorOptions{});
  CompiledFunctionPattern compiled;
  ASSERT_STATUS_OK(CompileFunctionPattern(normalized, graph, compiled));
  TargetGraphSnapshot snapshot;
  ASSERT_STATUS_OK(BuildTargetGraphSnapshot(graph, compiled, FunctionExtractorOptions{}, snapshot));
  std::vector<ReplacementPlan> plans;
  ASSERT_STATUS_OK(DiscoverReplacementPlans(compiled, snapshot, FunctionExtractorOptions{}, plans));
  ASSERT_EQ(plans.size(), 1u);
  snapshot.graph_viewer.reset();

  // Remove a planned node after discovery to create a stale plan.
  const NodeIndex stale_node_index = plans[0].removable_node_indices.back();
  ASSERT_TRUE(graph.RemoveNode(stale_node_index));
  EXPECT_FALSE(PrevalidatePlans(graph, compiled, plans).IsOK());
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 0u);
}

TEST_F(FunctionExtractorTest, ReturnsInvariantErrorAtPassCap) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeLinearTargetModel(function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());
  const size_t original_node_count = graph.NumberOfNodes();

  using namespace function_extractor_internal;
  const NormalizedFunctionPattern normalized =
      NormalizeFunctionPattern(function_proto, FunctionExtractorOptions{});
  ASSERT_STATUS_OK(normalized.construction_status);
  ExtractionControls controls;
  controls.maximum_passes = 0;

  const FunctionExtractionResult result =
      ExtractGraph(graph, normalized, FunctionExtractorOptions{}, controls);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_NE(result.status.ErrorMessage().find("defensive pass cap"), std::string::npos);
  EXPECT_EQ(result.replacements_applied, 0u);
  EXPECT_EQ(graph.NumberOfNodes(), original_node_count);
  AssertResolved(graph);
}

TEST_F(FunctionExtractorTest, ReportsAppliedCountOnResolveFailure) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeLinearTargetModel(function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  using namespace function_extractor_internal;
  const NormalizedFunctionPattern normalized =
      NormalizeFunctionPattern(function_proto, FunctionExtractorOptions{});
  ASSERT_STATUS_OK(normalized.construction_status);
  ExtractionControls controls;
  // The injected failure seam is intentionally unavailable in ONNX text.
  controls.resolve_graph = FailGraphResolve;

  const FunctionExtractionResult result =
      ExtractGraph(graph, normalized, FunctionExtractorOptions{}, controls);
  EXPECT_FALSE(result.status.IsOK());
  EXPECT_NE(result.status.ErrorMessage().find("injected resolve failure"), std::string::npos);
  EXPECT_EQ(result.replacements_applied, 1u);
  EXPECT_TRUE(graph.GraphResolveNeeded());
  EXPECT_EQ(graph.NumberOfNodes(), 1u);
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 1u);
}

TEST_F(FunctionExtractorTest, StrictlyDecreasesNodeCountPerReplacement) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeLinearTargetModel(function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());
  const size_t original_node_count = graph.NumberOfNodes();

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  EXPECT_LT(graph.NumberOfNodes(), original_node_count);
  EXPECT_EQ(graph.NumberOfNodes(), 1u);
}

TEST_F(FunctionExtractorTest, PreservesOutputIdentityAndFanout) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeModelFromText(
      R"(<ir_version: 8, opset_import: ["" : 13, "test.function" : 1]>
agraph (float[2] x, float[2] y) => (float[2] output_a, float[2] output_b) {
  sum = Add(x, y)
  pattern_output = Relu(sum)
  output_a = Identity(pattern_output)
  output_b = Neg(pattern_output)
})",
      function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* pattern_output = graph.GetNodeArg("pattern_output");
  NodeArg* output_a = graph.GetNodeArg("output_a");
  NodeArg* output_b = graph.GetNodeArg("output_b");
  ASSERT_NE(pattern_output, nullptr);
  ASSERT_NE(output_a, nullptr);
  ASSERT_NE(output_b, nullptr);
  ASSERT_STATUS_OK(graph.Resolve());
  const std::string pattern_output_name = pattern_output->Name();
  const ONNX_NAMESPACE::TypeProto pattern_output_type = *pattern_output->TypeAsProto();

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  const Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  ASSERT_EQ(call.OutputDefs()[0]->Name(), pattern_output_name);
  EXPECT_EQ(call.OutputDefs()[0]->TypeAsProto()->SerializeAsString(),
            pattern_output_type.SerializeAsString());
  EXPECT_EQ(graph.GetOutputs()[0]->Name(), output_a->Name());
  EXPECT_EQ(graph.GetOutputs()[1]->Name(), output_b->Name());
  EXPECT_EQ(graph.GetConsumerNodes(pattern_output_name).size(), 2u);
}

TEST_F(FunctionExtractorTest, ReturnsResolvedGraphAtFixpoint) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeLinearTargetModel(function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult first = extractor.Extract(graph);
  ASSERT_STATUS_OK(first.status);
  EXPECT_EQ(first.replacements_applied, 1u);
  AssertResolved(graph);

  const FunctionExtractionResult second = extractor.Extract(graph);
  ASSERT_STATUS_OK(second.status);
  EXPECT_EQ(second.replacements_applied, 0u);
  AssertResolved(graph);
}

TEST_F(FunctionExtractorTest, PersistsRegisteredCallAfterSerializeReload) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeLinearTargetModel(function_proto);
  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);

  std::shared_ptr<Model> reloaded_model = SerializeAndReload(*model);
  ASSERT_NE(reloaded_model, nullptr);
  AssertResolved(reloaded_model->MainGraph());
  EXPECT_EQ(CountOp(reloaded_model->MainGraph(), kFunctionDomain, function_proto.name()), 1u);
  EXPECT_EQ(reloaded_model->ToProto().functions_size(), 1);
}

TEST_F(FunctionExtractorTest, RoundTripsThroughInlineFunction) {
  const FunctionProto function_proto = MakeLinearFunction();
  auto model = MakeLinearTargetModel(function_proto);
  Graph& graph = model->MainGraph();
  NodeArg* output = graph.GetNodeArg("out");
  ASSERT_NE(output, nullptr);
  ASSERT_STATUS_OK(graph.Resolve());

  FunctionExtractor extractor(function_proto);
  const FunctionExtractionResult result = extractor.Extract(graph);
  ASSERT_STATUS_OK(result.status);
  ASSERT_EQ(result.replacements_applied, 1u);
  Node& call = FindOnlyOp(graph, kFunctionDomain, function_proto.name());
  ASSERT_STATUS_OK(graph.InlineFunction(call));
  ASSERT_STATUS_OK(graph.Resolve());
  EXPECT_EQ(CountOp(graph, kFunctionDomain, function_proto.name()), 0u);
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Add"), 1u);
  EXPECT_EQ(CountOp(graph, kOnnxDomain, "Relu"), 1u);
  EXPECT_EQ(graph.GetOutputs()[0]->Name(), output->Name());
  AssertResolved(graph);
}

}  // namespace test
}  // namespace onnxruntime
