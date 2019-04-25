// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "onnx/defs/schema.h"
#include "core/graph/constants.h"
#include "core/framework/op_kernel.h"
#include "core/framework/sparse_tensor.h"

#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test_utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {

ONNX_NAMESPACE::OpSchema GetSparseFromCOOSchema() {
  ONNX_NAMESPACE::OpSchema schema("SparseFromCOO", __FILE__, __LINE__);
  schema.SetDoc(R"DOC(
This operator constructs a sparse tensor from three tensors that provide a COO
(coordinate) representation with linearized index values.
)DOC")
      .SetDomain(onnxruntime::kMLDomain)
      .Input(
          0,
          "values",
          "Single dimensional Tensor that holds all non-zero values",
          "T1",
          OpSchema::Single)
      .Input(
          1,
          "indices",
          "Single dimensional tensor that holds linearized indices of non-zero values",
          "T2",
          OpSchema::Single)
      .Input(
          2,
          "shape",
          "Single dimensional tensor that holds the shape of the underlying dense tensor",
          "T2",
          OpSchema::Single)
      .Output(
          0,
          "sparse_rep",
          "A sparse representation of the tensor",
          "T",
          OpSchema::Single)
      .TypeConstraint(
          "T1",
          {"tensor(int64)"},
          "Type of the values (input tensor)")
      .TypeConstraint(
          "T2",
          {"tensor(int64)"},
          "Type of index tensor and shape")
      .TypeConstraint(
          "T",
          {"sparse_tensor(int64)"},
          "Output type");
  schema.SinceVersion(10);
  return schema;
}

ONNX_NAMESPACE::OpSchema GetSparseToCOOSchema() {
  ONNX_NAMESPACE::OpSchema schema("SparseToCOO", __FILE__, __LINE__);
  schema.SetDoc("Unpack a sparse tensor.")
      .SetDomain(onnxruntime::kMLDomain)
      .Input(
          0,
          "sparse_rep",
          "A sparse tensor to be unpacked into COO format",
          "T1",
          OpSchema::Single)
      .Output(
          0,
          "values",
          "A single dimensional tensor that holds non-zero values in the input",
          "T2",
          OpSchema::Single)
      .TypeConstraint(
          "T1",
          {"sparse_tensor(int64)"},
          "Only int64 is allowed")
      .TypeConstraint(
          "T2",
          {"tensor(int64)"},
          "Type of the values component");
  schema.SinceVersion(10);
  return schema;
}

/**
 *  @brief An implementation of the SparseFromCOO op.
 */

class SparseFromCOOKernel final : public OpKernel {
 public:
  SparseFromCOOKernel(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* ctx) const override {
    ORT_ENFORCE(ctx->InputCount() == 3, "Expecting 3 inputs");

    const Tensor& values = *ctx->Input<Tensor>(0);
    const Tensor& indices = *ctx->Input<Tensor>(1);
    const Tensor& shape = *ctx->Input<Tensor>(2);

    // values and indices should be 1-dimensional tensors
    const TensorShape& val_shape = values.Shape();
    const TensorShape& ind_shape = indices.Shape();
    const TensorShape& shape_shape = shape.Shape();

    auto size = val_shape.Size();

    ORT_ENFORCE(val_shape.NumDimensions() == 1, "Values must be a 1-dimensional tensor.");
    ORT_ENFORCE(ind_shape.NumDimensions() == 1, "Indices must be a 1-dimensional tensor with linearized index values.");
    ORT_ENFORCE(ind_shape.Size() == size, "Values and Indices must have same size.");
    ORT_ENFORCE(shape_shape.NumDimensions() == 1, "Shape must be a 1-dimensional tensor.");

    SparseTensor* output_sparse_tensor = ctx->Output<SparseTensor>(0);
    ORT_ENFORCE(output_sparse_tensor != nullptr);

    output_sparse_tensor->Values().assign(values.Data<int64_t>(), values.Data<int64_t>() + size);
    output_sparse_tensor->Indices().assign(indices.Data<int64_t>(), indices.Data<int64_t>() + size);
    output_sparse_tensor->Shape() = TensorShape(shape.Data<int64_t>(), shape_shape.Size());

    return Status::OK();
  }
};

/**
 *  @brief An implementation of the SparseToCOO op.
 */
class SparseToCOOKernel final : public OpKernel {
 public:
  SparseToCOOKernel(const OpKernelInfo& info) : OpKernel{info} {}

  Status Compute(OpKernelContext* ctx) const override {
    ORT_ENFORCE(ctx->InputCount() == 1, "Expecting a single SparseTensorSample input");
    const SparseTensor* sparse_input = ctx->Input<SparseTensor>(0);
    const auto& values = sparse_input->Values();
    auto size = static_cast<int64_t>(values.size());

    // const int64_t dims[1] = {size};
    TensorShape output_shape{size};

    Tensor* output = ctx->Output(0, output_shape);
    int64_t* ptr = output->MutableData<int64_t>();
    ORT_ENFORCE(ptr != nullptr);
    for (auto i = size - 1; i >= 0; --i)
      *(ptr + i) = values[i];
    // *shape_data = sparse_input->Size();

    return Status::OK();
  }
};

KernelDefBuilder ConstructSparseFromCOO() {
  KernelDefBuilder def;
  def.SetName("SparseFromCOO")
      .SetDomain(onnxruntime::kMLDomain)
      .SinceVersion(10)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("values",
                      DataTypeImpl::GetTensorType<int64_t>())
      .TypeConstraint("indices",
                      DataTypeImpl::GetTensorType<int64_t>())
      .TypeConstraint("shape",
                      DataTypeImpl::GetTensorType<int64_t>())
      .TypeConstraint("sparse_rep",
                      DataTypeImpl::GetSparseTensorType<int64_t>());
  return def;
}

KernelDefBuilder ConstructSparseToCOO() {
  KernelDefBuilder def;
  def.SetName("SparseToCOO")
      .SetDomain(onnxruntime::kMLDomain)
      .SinceVersion(10)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("sparse_rep",
                      DataTypeImpl::GetSparseTensorType<int64_t>())
      .TypeConstraint("values",
                      DataTypeImpl::GetTensorType<int64_t>());
  return def;
}

class SparseTensorTests : public testing::Test {
 public:
  static void SetUpTestCase() {
    // MLDataType mltype = DataTypeImpl::GetType<SparseTensorSample>();
    // DataTypeImpl::RegisterDataType(mltype);
  }
};

TEST_F(SparseTensorTests, RunModel) {
  SessionOptions so;
  so.enable_sequential_execution = true;
  so.session_logid = "SparseTensorTest";
  so.session_log_verbosity_level = 1;

  // Both the session and the model need custom registries
  // so we construct it here before the model
  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();
  InferenceSession session_object{so, &DefaultLoggingManager()};
  EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());

  auto ops_schema = GetSparseFromCOOSchema();
  auto shape_schema = GetSparseToCOOSchema();
  std::vector<OpSchema> schemas = {ops_schema, shape_schema};
  EXPECT_TRUE(registry->RegisterOpSet(schemas, onnxruntime::kMLDomain, 10, 11).IsOK());
  // Register our kernels here
  auto ctor_def = ConstructSparseFromCOO();
  EXPECT_TRUE(registry->RegisterCustomKernel(ctor_def, [](const OpKernelInfo& info) { return new SparseFromCOOKernel(info); }).IsOK());
  auto shape_def = ConstructSparseToCOO();
  EXPECT_TRUE(registry->RegisterCustomKernel(shape_def, [](const OpKernelInfo& info) { return new SparseToCOOKernel(info); }).IsOK());

  IOnnxRuntimeOpSchemaRegistryList custom_schema_registries_ = {registry->GetOpschemaRegistry()};
  std::unordered_map<std::string, int> domain_to_version = {{onnxruntime::kMLDomain, 10}};

  Model model("SparseTensorTest", false, ModelMetaData(), custom_schema_registries_, domain_to_version);
  auto& graph = model.MainGraph();

  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  TypeProto input_tensor_proto(*DataTypeImpl::GetTensorType<int64_t>()->GetTypeProto());

  {
    // Sparse tensor will contain total 5 elements but only 2 of them a non-zero
    TypeProto input_values(input_tensor_proto);
    input_values.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
    auto& sparse_values_arg = graph.GetOrCreateNodeArg("sparse_values", &input_values);
    inputs.push_back(&sparse_values_arg);

    TypeProto input_indicies(input_tensor_proto);
    input_indicies.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
    auto& sparse_indicies_arg = graph.GetOrCreateNodeArg("sparse_indicies", &input_indicies);
    inputs.push_back(&sparse_indicies_arg);

    // Shape tensor will contain only one value
    TypeProto input_shape(input_tensor_proto);
    input_shape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    auto& sparse_shape_arg = graph.GetOrCreateNodeArg("sparse_shape", &input_shape);
    inputs.push_back(&sparse_shape_arg);

    //Output is our custom data type
    TypeProto output_sparse_tensor(*DataTypeImpl::GetSparseTensorType<int64_t>()->GetTypeProto());
    auto& output_sparse_tensor_arg = graph.GetOrCreateNodeArg("sparse_rep", &output_sparse_tensor);
    outputs.push_back(&output_sparse_tensor_arg);

    auto& node = graph.AddNode("ConstructSparseTensor", "SparseFromCOO", "Create a sparse tensor representation",
                               inputs, outputs, nullptr, onnxruntime::kMLDomain);
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }
  {
    // We start the input from previous node output
    inputs = std::move(outputs);
    outputs.clear();

    TypeProto output_shape(input_tensor_proto);
    output_shape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    auto& output_shape_arg = graph.GetOrCreateNodeArg("sparse_tensor_shape", &output_shape);
    outputs.push_back(&output_shape_arg);
    auto& node = graph.AddNode("FetchSparseTensorShape", "SparseToCOO", "Fetch shape from sparse tensor",
                               inputs, outputs, nullptr, onnxruntime::kMLDomain);
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }

  EXPECT_TRUE(graph.Resolve().IsOK());

  // Get a proto and load from it
  std::string serialized_model;
  auto model_proto = model.ToProto();
  EXPECT_TRUE(model_proto.SerializeToString(&serialized_model));
  std::stringstream sstr(serialized_model);
  EXPECT_TRUE(session_object.Load(sstr).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;

  // Prepare inputs/outputs
  std::vector<int64_t> val_dims = {2};
  std::vector<int64_t> values = {99, 2};
  // prepare inputs
  MLValue ml_values;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), val_dims, values, &ml_values);

  std::vector<int64_t> ind_dims = {2};
  std::vector<int64_t> indicies = {1, 4};
  MLValue ml_indicies;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), ind_dims, indicies, &ml_indicies);

  std::vector<int64_t> shape_dims = {1};
  std::vector<int64_t> shape = {5};
  MLValue ml_shape;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), shape_dims, shape, &ml_shape);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("sparse_values", ml_values));
  feeds.insert(std::make_pair("sparse_indicies", ml_indicies));
  feeds.insert(std::make_pair("sparse_shape", ml_shape));

  // Output
  std::vector<int64_t> output_shape_dims = {1};
  std::vector<int64_t> output_shape = {0};

  std::vector<std::string> output_names;
  output_names.push_back("sparse_tensor_shape");
  std::vector<MLValue> fetches;

  EXPECT_TRUE(session_object.Run(run_options, feeds, output_names, &fetches).IsOK());
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  // Should get the original shape back in the form of a tensor
  EXPECT_EQ(1, rtensor.Shape().NumDimensions());
  EXPECT_EQ(99, *rtensor.template Data<int64_t>());
}

}  // namespace test
}  // namespace onnxruntime
