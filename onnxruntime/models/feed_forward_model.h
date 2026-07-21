// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Toy "C++ model" that computes a simple feed-forward-network (FFN) layer
// directly in C++ by driving ONNX Runtime's own op kernels.
//
//     Y = MatMul( Relu( MatMul(X, W1) + b1 ), W2 ) + b2
//
// The goal of this first exercise is to explore the calling interface between
// a hand-written C++ model and the ORT op kernels, and how such a model can be
// structured.
//
// The model uses a 2-phase structure:
//   * Phase 1 (construction): FeedForwardModel's constructor captures the
//     weights and pre-creates the ORT op kernels it will need (two MatMuls and
//     one Relu). This is analogous to "compiling" an execution plan.
//   * Phase 2 (execution): FeedForwardModel::Run() accepts the model input and
//     produces the model output by invoking the pre-created kernels.
//
// ORT's public interface for invoking a single kernel standalone is
// Ort::Op::Create() / Ort::Op::Invoke() (backed by OrtApi::CreateOp /
// OrtApi::InvokeOp). Both need an OrtKernelInfo (to locate/instantiate the
// kernel) and, at invocation time, an OrtKernelContext (for the allocator,
// thread pool, compute stream, etc.). Those two handles are only produced by
// ORT when it instantiates and runs an operator kernel. Therefore the model is
// hosted inside a custom operator (FeedForwardOp): ORT hands the kernel info to
// the constructor and the kernel context to Run().

#pragma once

#include <cstdint>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace models {

// A single dense weight (and optional bias) captured at construction time.
struct DenseWeights {
  std::vector<float> weight;  // row-major [in_features, out_features]
  std::vector<int64_t> weight_shape;
  std::vector<float> bias;  // [out_features], may be empty
  std::vector<int64_t> bias_shape;
};

// Toy feed-forward model:  Y = MatMul(Relu(MatMul(X, W1) + b1), W2) + b2
//
// Phase 1 - the constructor captures the weights (as constant inputs of the
//           hosting node) and pre-creates the ORT kernels.
// Phase 2 - Run() takes the runtime input X and produces Y.
class FeedForwardModel {
 public:
  // `info` is the OrtKernelInfo handed to us by ORT when the hosting kernel is
  // constructed. The two weight matrices (and their optional biases) are read
  // from the hosting node's constant inputs, in this order:
  //   input 0: X   (runtime input, NOT read here)
  //   input 1: W1  (constant)
  //   input 2: b1  (constant, optional)
  //   input 3: W2  (constant)
  //   input 4: b2  (constant, optional)
  explicit FeedForwardModel(const OrtKernelInfo* info);

  // Phase 2: compute Y = FFN(X). `context` is the OrtKernelContext supplied by
  // ORT for this invocation; input 0 is X and output 0 is Y.
  void Run(OrtKernelContext* context);

  // ORT's custom-op contract calls Compute(); it simply forwards to Run().
  void Compute(OrtKernelContext* context) { Run(context); }

 private:
  // Reads a constant tensor input of the hosting node into `out`.
  // Returns false if the input is absent (optional bias) or not constant.
  bool ReadConstantInput(const Ort::ConstKernelInfo& info, size_t index, DenseWeights& out, bool is_bias);

  Ort::KernelInfo info_copy_{nullptr};  // owned copy kept alive for Op::Create

  DenseWeights layer1_;
  DenseWeights layer2_;

  Ort::Op matmul1_{nullptr};
  Ort::Op add1_{nullptr};  // only created when b1 is present
  Ort::Op relu_{nullptr};
  Ort::Op matmul2_{nullptr};
  Ort::Op add2_{nullptr};  // only created when b2 is present

  bool has_bias1_ = false;
  bool has_bias2_ = false;
};

// Custom operator that hosts FeedForwardModel. It exists so that ORT provides
// the OrtKernelInfo (at kernel construction) and OrtKernelContext (at Compute).
struct FeedForwardOp : Ort::CustomOpBase<FeedForwardOp, FeedForwardModel> {
  explicit FeedForwardOp(const char* provider = "CPUExecutionProvider") : provider_(provider) {}

  void* CreateKernel(const OrtApi& /*api*/, const OrtKernelInfo* info) const {
    return new FeedForwardModel(info);
  }

  const char* GetName() const { return "FeedForward"; }
  const char* GetExecutionProviderType() const { return provider_; }

  // Inputs: X, W1, [b1], W2, [b2]. Biases are optional.
  size_t GetInputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
    // b1 (index 2) and b2 (index 4) are optional.
    if (index == 2 || index == 4) {
      return INPUT_OUTPUT_OPTIONAL;
    }
    return INPUT_OUTPUT_REQUIRED;
  }

  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

 private:
  const char* provider_;
};

}  // namespace models
