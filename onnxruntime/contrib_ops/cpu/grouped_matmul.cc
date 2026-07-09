// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/grouped_matmul.h"

#include <algorithm>
#include <vector>

#include "core/common/common.h"
#include "core/common/float16.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor_shape.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

namespace {

// Copy/convert a buffer of T into float. Only needed for the non-float (MLFloat16) path.
template <typename T>
void ToFloat(const T* src, float* dst, size_t count);

template <>
void ToFloat<MLFloat16>(const MLFloat16* src, float* dst, size_t count) {
  MlasConvertHalfToFloatBuffer(src, dst, count);
}

// Copy/convert a float buffer into T. Only needed for the non-float (MLFloat16) path.
template <typename T>
void FromFloat(const float* src, T* dst, size_t count);

template <>
void FromFloat<MLFloat16>(const float* src, MLFloat16* dst, size_t count) {
  MlasConvertFloatToHalfBuffer(src, dst, count);
}

}  // namespace

template <typename T>
Status GroupedMatMul<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* group_indices = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);

  const auto& input_shape = input->Shape();
  const auto& weights_shape = weights->Shape();

  ORT_RETURN_IF_NOT(input_shape.NumDimensions() >= 2,
                    "GroupedMatMul: input must have rank >= 2, got rank ", input_shape.NumDimensions());
  ORT_RETURN_IF_NOT(weights_shape.NumDimensions() == 3,
                    "GroupedMatMul: weights must have rank 3 (num_groups, K, N), got rank ",
                    weights_shape.NumDimensions());

  const int64_t K = input_shape[input_shape.NumDimensions() - 1];
  const int64_t num_tokens = input_shape.Size() / K;
  const int64_t num_groups = weights_shape[0];
  const int64_t weights_K = weights_shape[1];
  const int64_t N = weights_shape[2];

  ORT_RETURN_IF_NOT(weights_K == K,
                    "GroupedMatMul: weights dim 1 (", weights_K, ") must equal input last dim (", K, ").");

  const auto& indices_shape = group_indices->Shape();
  ORT_RETURN_IF_NOT(indices_shape.Size() == num_tokens,
                    "GroupedMatMul: group_indices must have one entry per token (", num_tokens,
                    "), got ", indices_shape.Size(), ".");

  if (bias != nullptr) {
    const auto& bias_shape = bias->Shape();
    ORT_RETURN_IF_NOT(bias_shape.NumDimensions() == 2 && bias_shape[0] == num_groups && bias_shape[1] == N,
                      "GroupedMatMul: bias must have shape (num_groups, N) = (", num_groups, ", ", N, ").");
  }

  // Output shape: input shape with last dim replaced by N.
  TensorShapeVector output_dims(input_shape.GetDims().begin(), input_shape.GetDims().end());
  output_dims[output_dims.size() - 1] = N;
  Tensor* output = context->Output(0, TensorShape(output_dims));

  if (num_tokens == 0 || N == 0) {
    return Status::OK();
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  // Convert (if needed) input, weights and bias to float, and prepare a float output buffer.
  const size_t input_count = static_cast<size_t>(num_tokens * K);
  const size_t weights_count = static_cast<size_t>(num_groups * K * N);
  const size_t output_count = static_cast<size_t>(num_tokens * N);

  const float* input_float;
  IAllocatorUniquePtr<float> input_float_buffer;
  if constexpr (std::is_same_v<T, float>) {
    input_float = input->Data<float>();
  } else {
    input_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, input_count);
    ToFloat<T>(input->Data<T>(), input_float_buffer.get(), input_count);
    input_float = input_float_buffer.get();
  }

  const float* weights_float;
  IAllocatorUniquePtr<float> weights_float_buffer;
  if constexpr (std::is_same_v<T, float>) {
    weights_float = weights->Data<float>();
  } else {
    weights_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, weights_count);
    ToFloat<T>(weights->Data<T>(), weights_float_buffer.get(), weights_count);
    weights_float = weights_float_buffer.get();
  }

  const float* bias_float = nullptr;
  IAllocatorUniquePtr<float> bias_float_buffer;
  if (bias != nullptr) {
    const size_t bias_count = static_cast<size_t>(num_groups * N);
    if constexpr (std::is_same_v<T, float>) {
      bias_float = bias->Data<float>();
    } else {
      bias_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, bias_count);
      ToFloat<T>(bias->Data<T>(), bias_float_buffer.get(), bias_count);
      bias_float = bias_float_buffer.get();
    }
  }

  float* output_float;
  IAllocatorUniquePtr<float> output_float_buffer;
  if constexpr (std::is_same_v<T, float>) {
    output_float = output->MutableData<float>();
  } else {
    output_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, output_count);
    output_float = output_float_buffer.get();
  }

  // Bucket tokens by group id (stable order preserves the input token order within a group).
  const int64_t* indices = group_indices->Data<int64_t>();
  std::vector<std::vector<int64_t>> group_tokens(static_cast<size_t>(num_groups));
  for (int64_t i = 0; i < num_tokens; ++i) {
    const int64_t g = indices[i];
    ORT_RETURN_IF_NOT(g >= 0 && g < num_groups,
                      "GroupedMatMul: group index ", g, " at token ", i, " is out of range [0, ", num_groups, ").");
    group_tokens[static_cast<size_t>(g)].push_back(i);
  }

  int64_t max_group_tokens = 0;
  for (const auto& tokens : group_tokens) {
    max_group_tokens = std::max(max_group_tokens, static_cast<int64_t>(tokens.size()));
  }

  // Reusable per-group gather/output buffers.
  auto gather_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(max_group_tokens * K));
  auto result_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(max_group_tokens * N));
  float* gathered = gather_buffer.get();
  float* result = result_buffer.get();

  for (int64_t g = 0; g < num_groups; ++g) {
    const auto& tokens = group_tokens[static_cast<size_t>(g)];
    if (tokens.empty()) {
      continue;  // Empty group: weights[g] is unused.
    }
    const int64_t count = static_cast<int64_t>(tokens.size());

    // Gather this group's token rows into a contiguous [count, K] block.
    for (int64_t r = 0; r < count; ++r) {
      const float* src = input_float + tokens[static_cast<size_t>(r)] * K;
      std::copy(src, src + K, gathered + r * K);
    }

    // One dense GEMM per group: [count, K] x [K, N] -> [count, N].
    // weights[g] is row-major [K, N] so ldb = N, no transpose.
    const float* weight_g = weights_float + g * K * N;
    MlasGemm(CblasNoTrans, CblasNoTrans,
             static_cast<size_t>(count), static_cast<size_t>(N), static_cast<size_t>(K),
             1.0f, gathered, static_cast<size_t>(K), weight_g, static_cast<size_t>(N),
             0.0f, result, static_cast<size_t>(N), tp, nullptr);

    // Add bias (if any) and scatter each row back to its original token position.
    const float* bias_g = bias_float ? (bias_float + g * N) : nullptr;
    for (int64_t r = 0; r < count; ++r) {
      const float* res_row = result + r * N;
      float* dst = output_float + tokens[static_cast<size_t>(r)] * N;
      if (bias_g) {
        for (int64_t j = 0; j < N; ++j) {
          dst[j] = res_row[j] + bias_g[j];
        }
      } else {
        std::copy(res_row, res_row + N, dst);
      }
    }
  }

  if constexpr (!std::is_same_v<T, float>) {
    FromFloat<T>(output_float, output->MutableData<T>(), output_count);
  }

  return Status::OK();
}

#define REGISTER_GROUPED_MATMUL_KERNEL(T)                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      GroupedMatMul, kMSDomain, 1, T, kCpuExecutionProvider,                 \
      KernelDefBuilder()                                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())              \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),       \
      GroupedMatMul<T>);

REGISTER_GROUPED_MATMUL_KERNEL(float)
REGISTER_GROUPED_MATMUL_KERNEL(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime
