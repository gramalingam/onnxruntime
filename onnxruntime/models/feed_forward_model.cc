// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "feed_forward_model.h"

#include <numeric>
#include <stdexcept>

namespace models {

namespace {

int64_t NumElements(const std::vector<int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

// Creates a standalone MatMul kernel (opset 13, float).
Ort::Op CreateMatMul(const Ort::KernelInfo& info) {
  const char* type_constraint_names[1] = {"T"};
  ONNXTensorElementDataType type_constraint_values[1] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  return Ort::Op::Create(info.GetConst(), "MatMul", "", 13,
                         type_constraint_names, type_constraint_values, 1,
                         /*attr_values*/ nullptr, /*attr_count*/ 0,
                         /*input_count*/ 2, /*output_count*/ 1);
}

// Creates a standalone Add kernel (opset 14, float).
Ort::Op CreateAdd(const Ort::KernelInfo& info) {
  const char* type_constraint_names[1] = {"T"};
  ONNXTensorElementDataType type_constraint_values[1] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  return Ort::Op::Create(info.GetConst(), "Add", "", 14,
                         type_constraint_names, type_constraint_values, 1,
                         nullptr, 0, 2, 1);
}

// Creates a standalone Relu kernel (opset 14, float).
Ort::Op CreateRelu(const Ort::KernelInfo& info) {
  const char* type_constraint_names[1] = {"T"};
  ONNXTensorElementDataType type_constraint_values[1] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  return Ort::Op::Create(info.GetConst(), "Relu", "", 14,
                         type_constraint_names, type_constraint_values, 1,
                         nullptr, 0, 1, 1);
}

}  // namespace

bool FeedForwardModel::ReadConstantInput(const Ort::ConstKernelInfo& info, size_t index,
                                         DenseWeights& out, bool is_bias) {
  // Skip inputs that were not supplied (e.g. an omitted optional bias).
  if (index >= info.GetInputCount()) {
    return false;
  }

  int is_constant = 0;
  Ort::ConstValue value = info.GetTensorConstantInput(index, &is_constant);
  if (!is_constant || value == nullptr) {
    // Present but not a compile-time constant: this toy model requires weights
    // to be constant so they can be captured in phase 1.
    if (!is_bias) {
      throw std::runtime_error("FeedForwardModel: weight input " + std::to_string(index) +
                               " must be a constant initializer");
    }
    return false;
  }

  auto shape = value.GetTensorTypeAndShapeInfo().GetShape();
  const int64_t count = NumElements(shape);
  const float* data = value.GetTensorData<float>();

  if (is_bias) {
    out.bias_shape = std::move(shape);
    out.bias.assign(data, data + count);
  } else {
    out.weight_shape = std::move(shape);
    out.weight.assign(data, data + count);
  }
  return true;
}

FeedForwardModel::FeedForwardModel(const OrtKernelInfo* info) {
  Ort::ConstKernelInfo const_info{info};
  info_copy_ = const_info.Copy();  // owns a copy that outlives `info`

  // Phase 1: capture the weights (constant node inputs 1..4).
  ReadConstantInput(const_info, 1, layer1_, /*is_bias*/ false);              // W1 (required)
  has_bias1_ = ReadConstantInput(const_info, 2, layer1_, /*is_bias*/ true);  // b1 (optional)
  ReadConstantInput(const_info, 3, layer2_, /*is_bias*/ false);              // W2 (required)
  has_bias2_ = ReadConstantInput(const_info, 4, layer2_, /*is_bias*/ true);  // b2 (optional)

  // Phase 1: pre-create the ORT op kernels that make up the execution plan.
  matmul1_ = CreateMatMul(info_copy_);
  if (has_bias1_) {
    add1_ = CreateAdd(info_copy_);
  }
  relu_ = CreateRelu(info_copy_);
  matmul2_ = CreateMatMul(info_copy_);
  if (has_bias2_) {
    add2_ = CreateAdd(info_copy_);
  }
}

void FeedForwardModel::Run(OrtKernelContext* context) {
  Ort::KernelContext ctx{context};

  // Phase 2: the runtime input X is node input 0.
  Ort::ConstValue x = ctx.GetInput(0);
  const auto x_shape = x.GetTensorTypeAndShapeInfo().GetShape();
  if (x_shape.size() != 2) {
    throw std::runtime_error("FeedForwardModel expects a 2-D input [M, K]");
  }
  const int64_t m = x_shape[0];
  const int64_t k = x_shape[1];
  const int64_t h = layer1_.weight_shape.at(1);  // W1: [K, H]
  const int64_t n = layer2_.weight_shape.at(1);  // W2: [H, N]

  auto cpu = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // Wrap the captured weights as OrtValues (no copy: they alias member storage).
  Ort::Value w1 = Ort::Value::CreateTensor<float>(
      cpu, layer1_.weight.data(), layer1_.weight.size(),
      layer1_.weight_shape.data(), layer1_.weight_shape.size());
  Ort::Value w2 = Ort::Value::CreateTensor<float>(
      cpu, layer2_.weight.data(), layer2_.weight.size(),
      layer2_.weight_shape.data(), layer2_.weight_shape.size());

  // Intermediate buffers for the hidden activations.
  std::vector<int64_t> hidden_shape = {m, h};
  std::vector<float> hidden(static_cast<size_t>(m * h));
  Ort::Value hidden_val = Ort::Value::CreateTensor<float>(
      cpu, hidden.data(), hidden.size(), hidden_shape.data(), hidden_shape.size());

  // hidden = X * W1
  {
    const OrtValue* inputs[2] = {x, w1};
    OrtValue* outputs[1] = {hidden_val};
    matmul1_.Invoke(context, inputs, 2, outputs, 1);
  }

  // hidden = hidden + b1
  if (has_bias1_) {
    Ort::Value b1 = Ort::Value::CreateTensor<float>(
        cpu, layer1_.bias.data(), layer1_.bias.size(),
        layer1_.bias_shape.data(), layer1_.bias_shape.size());
    const OrtValue* inputs[2] = {hidden_val, b1};
    OrtValue* outputs[1] = {hidden_val};  // in-place is fine: Add reads then writes
    add1_.Invoke(context, inputs, 2, outputs, 1);
  }

  // hidden = Relu(hidden)
  {
    const OrtValue* inputs[1] = {hidden_val};
    OrtValue* outputs[1] = {hidden_val};
    relu_.Invoke(context, inputs, 1, outputs, 1);
  }

  // Y = hidden * W2  (final output is node output 0)
  Ort::UnownedValue y = ctx.GetOutput(0, {m, n});
  {
    const OrtValue* inputs[2] = {hidden_val, w2};
    OrtValue* outputs[1] = {y};
    matmul2_.Invoke(context, inputs, 2, outputs, 1);
  }

  // Y = Y + b2
  if (has_bias2_) {
    Ort::Value b2 = Ort::Value::CreateTensor<float>(
        cpu, layer2_.bias.data(), layer2_.bias.size(),
        layer2_.bias_shape.data(), layer2_.bias_shape.size());
    const OrtValue* inputs[2] = {y, b2};
    OrtValue* outputs[1] = {y};
    add2_.Invoke(context, inputs, 2, outputs, 1);
  }

  (void)k;  // K is implied by the weight shapes; kept for readability.
}

}  // namespace models
