// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnx/defs/function.h"
#include "core/framework/data_types.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/constants.h"
#include "core/framework/op_kernel.h"
#include "core/framework/sparse_tensor.h"
#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
// #include "test_utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {

static std::vector<ONNX_NAMESPACE::NodeProto> build_function_body() {
  // Output1 = Sub(Input1, Input2);
  return FunctionBodyHelper::BuildNodes({{{"Output1"}, "Sub", {"Input1", "Input3"}}});
}

static ONNX_NAMESPACE::OpSchema TestOpSchema() {
  ONNX_NAMESPACE::OpSchema schema;
  schema.SetName("TestOp")
      .SetDomain(onnxruntime::kMLDomain)
      .SetDoc(R"DOC(A test op.)DOC")
      .SinceVersion(10)
      .Input(
          0,
          "Input1",
          "First input",
          "T",
          OpSchema::Single)
      .Input(
          1,
          "Input2",
          "Second input",
          "T",
          OpSchema::Optional)
      .Input(
          2,
          "Input3",
          "Third input",
          "T",
          OpSchema::Single)
      .Output(
          0,
          "Output1",
          "First output",
          "T",
          OpSchema::Single)
      .TypeConstraint(
          "T",
          {"tensor(int64)"},
          "Type of all inputs and outputs.")
      .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
      .FunctionBody(build_function_body());
  return schema;
}

class FunctionTest : public testing::Test {
 protected:
  InferenceSession session_object;
  std::shared_ptr<CustomRegistry> registry;
  IOnnxRuntimeOpSchemaRegistryList custom_schema_registries_;
  std::unordered_map<std::string, int> domain_to_version;
  Model model;
  Graph& graph;

  std::vector<TypeProto> types;

 public:
  FunctionTest() : session_object(SessionOptions(), &DefaultLoggingManager()),
                   registry(std::make_shared<CustomRegistry>()),
                   custom_schema_registries_{registry->GetOpschemaRegistry()},
                   domain_to_version{{onnxruntime::kMLDomain, 10}},
                   model("FunctionTest", false, ModelMetaData(), custom_schema_registries_, domain_to_version, {}, DefaultLoggingManager().DefaultLogger()),
                   graph(model.MainGraph()) {
    EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());
    std::vector<OpSchema> schemas{TestOpSchema()};
    EXPECT_TRUE(registry->RegisterOpSet(schemas, onnxruntime::kMLDomain, 10, 11).IsOK());
  }

  NodeArg* Tensor(std::string name) {
    types.push_back(*DataTypeImpl::GetTensorType<int64_t>()->GetTypeProto());
    auto& arg = graph.GetOrCreateNodeArg(name, &types.back());
    return &arg;
  }

  NodeArg* NoValue() {
    auto& arg = graph.GetOrCreateNodeArg("", nullptr);
    return &arg;
  }

  void Node(std::string op, const std::vector<NodeArg*> inputs, const std::vector<NodeArg*> outputs) {
    auto& node = graph.AddNode("", op, "", inputs, outputs, nullptr, onnxruntime::kMLDomain);
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }
};

TEST_F(FunctionTest, Test1) {
  // Build model/graph

  // sparse1 <- SparseFromCOO(values, indices, shape)
  auto X = Tensor("X");
  auto Y = Tensor("Y");
  auto W = Tensor("W");
  auto Z = NoValue();

  Node("TestOp", {X, Z, W}, {Y});

  // Check graph
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

}  // namespace test
}  // namespace onnxruntime
