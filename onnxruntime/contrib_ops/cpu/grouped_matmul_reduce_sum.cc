// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/grouped_matmul_reduce_sum.h"

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
Status GroupedMatMulReduceSum<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* group_indices = context->Input<Tensor>(2);
  const Tensor* combine_weights = context->Input<Tensor>(3);
  const Tensor* bias = context->Input<Tensor>(4);

  const auto& input_shape = input->Shape();
  const auto& weights_shape = weights->Shape();

  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 3,
                    "GroupedMatMulReduceSum: input must have rank 3 (M, k, K), got rank ",
                    input_shape.NumDimensions());
  ORT_RETURN_IF_NOT(weights_shape.NumDimensions() == 3,
                    "GroupedMatMulReduceSum: weights must have rank 3 (num_groups, K, N), got rank ",
                    weights_shape.NumDimensions());

  const int64_t M = input_shape[0];
  const int64_t k = input_shape[1];
  const int64_t K = input_shape[2];
  const int64_t num_groups = weights_shape[0];
  const int64_t weights_K = weights_shape[1];
  const int64_t N = weights_shape[2];
  const int64_t num_selections = M * k;

  ORT_RETURN_IF_NOT(weights_K == K,
                    "GroupedMatMulReduceSum: weights dim 1 (", weights_K, ") must equal input last dim (",
                    K, ").");

  const auto& indices_shape = group_indices->Shape();
  ORT_RETURN_IF_NOT(indices_shape.NumDimensions() == 2 && indices_shape[0] == M && indices_shape[1] == k,
                    "GroupedMatMulReduceSum: group_indices must have shape (M, k) = (", M, ", ", k,
                    "), got ", indices_shape.ToString(), ".");

  const auto& combine_shape = combine_weights->Shape();
  ORT_RETURN_IF_NOT(combine_shape.NumDimensions() == 2 && combine_shape[0] == M && combine_shape[1] == k,
                    "GroupedMatMulReduceSum: combine_weights must have shape (M, k) = (", M, ", ", k, ").");

  if (bias != nullptr) {
    const auto& bias_shape = bias->Shape();
    ORT_RETURN_IF_NOT(bias_shape.NumDimensions() == 2 && bias_shape[0] == num_groups && bias_shape[1] == N,
                      "GroupedMatMulReduceSum: bias must have shape (num_groups, N) = (", num_groups, ", ",
                      N, ").");
  }

  // Output shape is (M, N): the combined per-token result.
  TensorShapeVector output_dims;
  output_dims.push_back(M);
  output_dims.push_back(N);
  Tensor* output = context->Output(0, TensorShape(output_dims));

  if (M == 0 || N == 0) {
    return Status::OK();
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  const size_t input_count = static_cast<size_t>(num_selections * K);
  const size_t weights_count = static_cast<size_t>(num_groups * K * N);
  const size_t output_count = static_cast<size_t>(output->Shape().Size());

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

  const float* combine_float;
  IAllocatorUniquePtr<float> combine_float_buffer;
  if constexpr (std::is_same_v<T, float>) {
    combine_float = combine_weights->Data<float>();
  } else {
    combine_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(num_selections));
    ToFloat<T>(combine_weights->Data<T>(), combine_float_buffer.get(), static_cast<size_t>(num_selections));
    combine_float = combine_float_buffer.get();
  }

  float* output_float;
  IAllocatorUniquePtr<float> output_float_buffer;
  if constexpr (std::is_same_v<T, float>) {
    output_float = output->MutableData<float>();
  } else {
    output_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, output_count);
    output_float = output_float_buffer.get();
  }
  std::fill(output_float, output_float + output_count, 0.0f);

  // Bucket selections by group id. A "selection" is a (token, expert-slot) pair p in
  // [0, M*k); the source token row is p / k, the output row is also p / k (down-projection
  // combines all k slots of a token into a single output row).
  const int64_t* indices = group_indices->Data<int64_t>();
  std::vector<std::vector<int64_t>> group_selections(static_cast<size_t>(num_groups));
  for (int64_t p = 0; p < num_selections; ++p) {
    const int64_t g = indices[p];
    ORT_RETURN_IF_NOT(g >= 0 && g < num_groups,
                      "GroupedMatMulReduceSum: group index ", g, " at selection ", p,
                      " is out of range [0, ", num_groups, ").");
    group_selections[static_cast<size_t>(g)].push_back(p);
  }

  int64_t max_group_selections = 0;
  for (const auto& sels : group_selections) {
    max_group_selections = std::max(max_group_selections, static_cast<int64_t>(sels.size()));
  }

  // Reusable per-group gather/output buffers.
  auto gather_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(max_group_selections * K));
  auto result_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(max_group_selections * N));
  float* gathered = gather_buffer.get();
  float* result = result_buffer.get();

  for (int64_t g = 0; g < num_groups; ++g) {
    const auto& sels = group_selections[static_cast<size_t>(g)];
    if (sels.empty()) {
      continue;  // Empty group: weights[g] is unused.
    }
    const int64_t count = static_cast<int64_t>(sels.size());

    // Gather each selection's distinct (token, expert-slot) input row into a contiguous
    // [count, K] block. Unlike GroupedMatMul, each selection p has its own input row (input
    // is (M, k, K) with one row per selection) -- no row reuse across k slots.
    for (int64_t r = 0; r < count; ++r) {
      const int64_t p = sels[static_cast<size_t>(r)];
      const float* src = input_float + p * K;
      std::copy(src, src + K, gathered + r * K);
    }

    // One dense GEMM per group: [count, K] x [K, N] -> [count, N].
    const float* weight_g = weights_float + g * K * N;
    MlasGemm(CblasNoTrans, CblasNoTrans,
             static_cast<size_t>(count), static_cast<size_t>(N), static_cast<size_t>(K),
             1.0f, gathered, static_cast<size_t>(K), weight_g, static_cast<size_t>(N),
             0.0f, result, static_cast<size_t>(N), tp, nullptr);

    // Add bias (if any), scale by the selection's combine weight, and accumulate into the
    // token's single output row (reduce over the k expert slots).
    const float* bias_g = bias_float ? (bias_float + g * N) : nullptr;
    for (int64_t r = 0; r < count; ++r) {
      const int64_t p = sels[static_cast<size_t>(r)];
      const int64_t token = p / k;
      const float w = combine_float[p];
      const float* res_row = result + r * N;
      float* dst = output_float + token * N;
      if (bias_g) {
        for (int64_t j = 0; j < N; ++j) {
          dst[j] += w * (res_row[j] + bias_g[j]);
        }
      } else {
        for (int64_t j = 0; j < N; ++j) {
          dst[j] += w * res_row[j];
        }
      }
    }
  }

  if constexpr (!std::is_same_v<T, float>) {
    FromFloat<T>(output_float, output->MutableData<T>(), output_count);
  }

  return Status::OK();
}

#define REGISTER_GROUPED_MATMUL_REDUCE_SUM_KERNEL(T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                             \
      GroupedMatMulReduceSum, kMSDomain, 1, T, kCpuExecutionProvider,       \
      KernelDefBuilder()                                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())             \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),      \
      GroupedMatMulReduceSum<T>);

REGISTER_GROUPED_MATMUL_REDUCE_SUM_KERNEL(float)
REGISTER_GROUPED_MATMUL_REDUCE_SUM_KERNEL(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime
