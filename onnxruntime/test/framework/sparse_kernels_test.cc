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
  static std::string OpName() { return "SparseFromCOO"; };

  static ONNX_NAMESPACE::OpSchema OpSchema() {
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetDoc(R"DOC(
This operator constructs a sparse tensor from three tensors that provide a COO
(coordinate) representation.
)DOC")
        .Input(
            0,
            "values",
            "A 1-dimensional tensor of shape [NNZ] that holds all non-zero values",
            "T1",
            OpSchema::Single)
        .Input(
            1,
            "indices",
            "A 2-dimensional tensor of shape [NNZ,d] that holds the (d) indices of non-zero values",
            "T2",
            OpSchema::Single)
        .Input(
            2,
            "shape",
            "A 1-dimensional tensor of shape [d] that holds the shape of the underlying dense tensor",
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
      const Tensor& shape_tensor = *ctx->Input<Tensor>(2);

      // values and indices should be 1-dimensional tensors
      const TensorShape& val_shape = values.Shape();
      const TensorShape& ind_shape = indices.Shape();
      const TensorShape& shape_shape = shape_tensor.Shape();

      auto nnz = val_shape.Size();
      auto numdims = shape_shape.Size();

      ORT_ENFORCE(val_shape.NumDimensions() == 1, "Values must be a 1-dimensional tensor.");

      ORT_ENFORCE(ind_shape.NumDimensions() == 2, "Indices must be a 2-dimensional tensor.");
      ORT_ENFORCE(ind_shape[0] == nnz, "Indices must have [NNZ,d] shape.");
      ORT_ENFORCE(ind_shape[1] == numdims, "Indices must have [NNZ,d] shape.");

      ORT_ENFORCE(shape_shape.NumDimensions() == 1, "Shape must be a 1-dimensional tensor.");

      TensorShape shape(shape_tensor.Data<int64_t>(), shape_shape.Size());

      SparseTensor* output = ctx->Output(0, static_cast<size_t>(nnz), shape);
      ORT_ENFORCE(output != nullptr);

      memcpy(output->Values().MutableData<int64_t>(), values.Data<int64_t>(), nnz * sizeof(int64_t));
      memcpy(output->Indices().MutableData<int64_t>(), indices.Data<int64_t>(), nnz * numdims * sizeof(int64_t));
      // output->Shape() = TensorShape(shape_tensor.Data<int64_t>(), shape_shape.Size());

      return Status::OK();
    }
  };

  static KernelDefBuilder KernelDef() {
    KernelDefBuilder def;
    def.SetName(SparseFromCOO::OpName())
        .TypeConstraint("values", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("indices", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("sparse_rep", DataTypeImpl::GetSparseTensorType<int64_t>());
    return def;
  }
};

// op SparseAbs
struct SparseAbs {
  static const std::string OpName() { return "SparseAbs"; };

  // The ONNX schema for SparseAbs:
  static ONNX_NAMESPACE::OpSchema OpSchema() {
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetDoc(R"DOC(
This operator applies the Abs op element-wise to the input sparse-tensor.
)DOC")
        .Input(
            0,
            "input",
            "Input sparse tensor",
            "T",
            OpSchema::Single)
        .Output(
            0,
            "output",
            "Output sparse tensor",
            "T",
            OpSchema::Single)
        .TypeConstraint(
            "T",
            {"sparse_tensor(int64)"},
            "Input and Output type");
    return schema;
  }

  // A kernel implementation of SparseAbs:
  class OpKernelImpl final : public OpKernel {
   public:
    OpKernelImpl(const OpKernelInfo& info) : OpKernel{info} {}

    Status Compute(OpKernelContext* ctx) const override {
      ORT_ENFORCE(ctx->InputCount() == 1, "Expecting 1 input");

      const SparseTensor* input = ctx->Input<SparseTensor>(0);
      auto* input_values = input->Values().Data<int64_t>();
      auto nnz = input->NumValues();
      auto& shape = input->Shape();

      // Allocate/get output-tensor:
      SparseTensor* output = ctx->Output(0, nnz, shape);

      // compute output values:
      auto* output_values = output->Values().MutableData<int64_t>();
      for (int i = 0; i < nnz; ++i)
        output_values[i] = std::abs(input_values[i]);

      // Currently, there is no way to share the indices/shape between two sparse-tensors.
      // So, we copy indices/shape from input to output.
      // TODO: Extend allocation-planner to enable such sharing.
      const auto& input_indices = input->Indices();
      auto num_indices = input_indices.Shape().Size();
      memcpy(output->Indices().MutableData<int64_t>(), input_indices.Data<int64_t>(), num_indices * sizeof(int64_t));

      output->Shape() = shape;

      return Status::OK();
    }
  };

  // A KernelDefBuilder for SparseAbs:
  static KernelDefBuilder KernelDef() {
    KernelDefBuilder def;
    def.SetName(OpName())
        .TypeConstraint("T", DataTypeImpl::GetSparseTensorType<int64_t>());
    return def;
  }
};

// op SparseToValues
struct SparseToValues {
  static const std::string OpName() { return "SparseToValues"; };

  static ONNX_NAMESPACE::OpSchema OpSchema() {
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetDoc("Return the non-zero values in a sparse tensor (as a dense tensor).")
        .Input(
            0,
            "sparse_rep",
            "Input sparse tensor.",
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
    return schema;
  }

  //  A kernel implementation of SparseToValues
  class OpKernelImpl final : public OpKernel {
   public:
    OpKernelImpl(const OpKernelInfo& info) : OpKernel{info} {}

    Status Compute(OpKernelContext* ctx) const override {
      ORT_ENFORCE(ctx->InputCount() == 1, "Expecting a single SparseTensorSample input");
      const SparseTensor* sparse_input = ctx->Input<SparseTensor>(0);
      const auto* values = sparse_input->Values().Data<int64_t>();
      auto nnz = static_cast<int64_t>(sparse_input->NumValues());

      TensorShape output_shape{nnz};

      Tensor* output = ctx->Output(0, output_shape);
      int64_t* output_data = output->MutableData<int64_t>();
      ORT_ENFORCE(output_data != nullptr);

      memcpy(output_data, values, nnz * sizeof(int64_t));

      return Status::OK();
    }
  };

  // A KernelDefBuilder for SparseToValues
  static KernelDefBuilder KernelDef() {
    KernelDefBuilder def;
    def.SetName(OpName())
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
    auto schema = Op::OpSchema();
    schema.SetName(Op::OpName());
    schema.SetDomain(onnxruntime::kMLDomain);
    schema.SinceVersion(10);
    schemas.push_back(schema);

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

  MLValue Constant(const std::vector<int64_t>& elts) {
    const std::vector<int64_t> shape{static_cast<int64_t>(elts.size())};
    return Constant(elts, shape);
  }

  MLValue Constant(const std::vector<int64_t>& elts, const std::vector<int64_t>& shape) {
    MLValue mlvalue;
    CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), shape, elts, &mlvalue);
    return mlvalue;
  }

  void ExpectEq(MLValue val1, MLValue val2) {
    // Restricted to case where val1 and val2 are int64_t tensors
    auto& tensor1 = val1.Get<Tensor>();
    auto& tensor2 = val2.Get<Tensor>();
    EXPECT_EQ(tensor1.Shape().Size(), tensor2.Shape().Size());
    auto* data1 = tensor1.Data<int64_t>();
    auto* data2 = tensor2.Data<int64_t>();
    for (int64_t i = 0, limit = tensor1.Shape().Size(); i < limit; ++i) {
      EXPECT_EQ(data1[i], data2[i]);
    }
  }

  void ExpectEq(MLValue val1, const std::vector<int64_t>& data2) {
    // Restricted to case where val1 is an int64_t tensor
    auto& tensor1 = val1.Get<Tensor>();
    EXPECT_EQ(static_cast<uint64_t>(tensor1.Shape().Size()), data2.size());
    auto* data1 = tensor1.Data<int64_t>();
    for (int64_t i = 0, limit = tensor1.Shape().Size(); i < limit; ++i) {
      EXPECT_EQ(data1[i], data2[i]);
    }
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

TEST_F(SparseTensorTests, Test1) {
  // Register ops
  Add<SparseFromCOO>();
  Add<SparseAbs>();
  Add<SparseToValues>();
  RegisterOps();

  // Build model/graph
  auto NZV = Dense("values");   // Non-Zero-Values
  auto NZI = Dense("indices");  // Non-Zero-Indices
  auto shape = Dense("shape");
  auto sparse1 = Sparse("sparse1");

  Node(SparseFromCOO::OpName(), {NZV, NZI, shape}, {sparse1});

  auto sparse2 = Sparse("sparse2");
  Node(SparseAbs::OpName(), {sparse1}, {sparse2});

  auto NZV2 = Dense("output");
  Node(SparseToValues::OpName(), {sparse2}, {NZV2});

  EXPECT_TRUE(graph.Resolve().IsOK());

  // Serialize model and deserialize it back
  SerializeAndLoad();

  // Run the model
  RunOptions run_options;

  // Inputs for run:
  MLValue NZV_values = Constant({-99, 2});
  MLValue NZI_values = Constant({1, 4}, {2, 1});
  MLValue shape_value = Constant({5});

  NameMLValMap feeds{
      {NZV->Name(), NZV_values},
      {NZI->Name(), NZI_values},
      {shape->Name(), shape_value}};

  std::vector<std::string> output_names{NZV2->Name()};

  std::vector<MLValue> fetches;

  EXPECT_TRUE(session_object.Run(run_options, feeds, output_names, &fetches).IsOK());

  ASSERT_EQ(1, fetches.size());
  auto& output = fetches.front();

  ExpectEq(output, {99, 2});
}

}  // namespace test
}  // namespace onnxruntime
