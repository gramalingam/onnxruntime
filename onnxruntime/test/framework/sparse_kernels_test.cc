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

// op SparseFromCOO
struct SparseFromCOO {
  static ONNX_NAMESPACE::OpSchema OpSchema() {
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

  /**
 *  @brief An implementation of the SparseFromCOO op.
 */
  class OpKernelImpl final : public OpKernel {
   public:
    OpKernelImpl(const OpKernelInfo& info) : OpKernel{info} {}

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

  static KernelDefBuilder KernelDef() {
    KernelDefBuilder def;
    def.SetName("SparseFromCOO")
        .TypeConstraint("values", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("indices", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("sparse_rep", DataTypeImpl::GetSparseTensorType<int64_t>());
    return def;
  }
};

// op SparseAbs

// op SparseToCOO
struct SparseToCOO {
  static ONNX_NAMESPACE::OpSchema OpSchema() {
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
 *  @brief An implementation of the SparseToCOO op.
 */
  class OpKernelImpl final : public OpKernel {
   public:
    OpKernelImpl(const OpKernelInfo& info) : OpKernel{info} {}

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

  static KernelDefBuilder KernelDef() {
    KernelDefBuilder def;
    def.SetName("SparseToCOO")
        .TypeConstraint("sparse_rep", DataTypeImpl::GetSparseTensorType<int64_t>())
        .TypeConstraint("values", DataTypeImpl::GetTensorType<int64_t>());
    return def;
  }
};

using Action = std::function<void(CustomRegistry*)>;

class SparseTensorTests : public testing::Test {
 public:
  SparseTensorTests() : session_object(SessionOptions(), &DefaultLoggingManager()),
                        registry(std::make_shared<CustomRegistry>()),
                        custom_schema_registries_{registry->GetOpschemaRegistry()},
                        domain_to_version{{onnxruntime::kMLDomain, 10}},
                        model("SparseTensorTest", false, ModelMetaData(), custom_schema_registries_, domain_to_version),
                        graph(model.MainGraph()) {
    EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());
  }

  template <typename Op>
  void Add() {
    schemas.push_back(Op::OpSchema());

    Action register_kernel = [](CustomRegistry* registry) {
      auto kernel_def_builder = Op::KernelDef();
      kernel_def_builder
          .SetDomain(onnxruntime::kMLDomain)
          .SinceVersion(10)
          .Provider(onnxruntime::kCpuExecutionProvider);
      EXPECT_TRUE(registry->RegisterCustomKernel(kernel_def_builder, [](const OpKernelInfo& info) { return new Op::OpKernelImpl(info); }).IsOK());
    };
    register_actions.push_back(register_kernel);
  }

  void RegisterOps() {
    EXPECT_TRUE(registry->RegisterOpSet(schemas, onnxruntime::kMLDomain, 10, 11).IsOK());
    for (auto registerop : register_actions)
      registerop(registry.get());
  }

  void SerializeAndLoad() {
    // Serialize model and deserialize it back
    std::string serialized_model;
    auto model_proto = model.ToProto();
    EXPECT_TRUE(model_proto.SerializeToString(&serialized_model));
    std::stringstream sstr(serialized_model);
    EXPECT_TRUE(session_object.Load(sstr).IsOK());
    EXPECT_TRUE(session_object.Initialize().IsOK());
  }

  NodeArg* Sparse(std::string name) {
    types.push_back(*DataTypeImpl::GetSparseTensorType<int64_t>()->GetTypeProto());
    auto& arg = graph.GetOrCreateNodeArg(name, &types.back());
    return &arg;
  }

  NodeArg* Dense(std::string name) {
    types.push_back(*DataTypeImpl::GetTensorType<int64_t>()->GetTypeProto());
    auto& arg = graph.GetOrCreateNodeArg(name, &types.back());
    return &arg;
  }

  void Node(std::string op, const std::vector<NodeArg*> inputs, const std::vector<NodeArg*> outputs) {
    auto& node = graph.AddNode("", op, "", inputs, outputs, nullptr, onnxruntime::kMLDomain);
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }

  MLValue Constant(const std::vector<int64_t>& shape, const std::vector<int64_t>& elts) {
    // ml_values.push_back(MLValue());
    MLValue mlvalue;
    CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), shape, elts, &mlvalue);
    return mlvalue;
  }

 protected:
  std::vector<OpSchema> schemas;
  std::vector<Action> register_actions;
  std::shared_ptr<CustomRegistry> registry;
  InferenceSession session_object;

  IOnnxRuntimeOpSchemaRegistryList custom_schema_registries_;
  std::unordered_map<std::string, int> domain_to_version;
  Model model;
  Graph& graph;

  std::vector<TypeProto> types;
};

TEST_F(SparseTensorTests, RunModel) {
  // Register ops
  Add<SparseFromCOO>();
  Add<SparseToCOO>();
  RegisterOps();

  // Build model/graph
  auto* values_var = Dense("values");
  auto* indices = Dense("indices");
  auto* shape_var = Dense("shape");
  auto* output_sparse_tensor_arg = Sparse("sparse_rep");

  Node("SparseFromCOO", {values_var, indices, shape_var}, {output_sparse_tensor_arg});

  auto* output_shape_arg = Dense("sparse_tensor_shape");

  Node("SparseToCOO", {output_sparse_tensor_arg}, {output_shape_arg});

  EXPECT_TRUE(graph.Resolve().IsOK());

  // Serialize model and deserialize it back
  SerializeAndLoad();

  // Run the model
  RunOptions run_options;

  // Inputs for run:
  MLValue ml_values = Constant({2}, {99, 2});
  MLValue ml_indicies = Constant({2}, {1, 4});
  MLValue ml_shape = Constant({1}, {5});

  NameMLValMap feeds{
      {"values", ml_values},
      {"indices", ml_indicies},
      {"shape", ml_shape}};

  std::vector<std::string> output_names{"sparse_tensor_shape"};

  std::vector<MLValue> fetches;

  EXPECT_TRUE(session_object.Run(run_options, feeds, output_names, &fetches).IsOK());

  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();

  EXPECT_EQ(1, rtensor.Shape().NumDimensions());
  EXPECT_EQ(99, *rtensor.template Data<int64_t>());
}

}  // namespace test
}  // namespace onnxruntime
