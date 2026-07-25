// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/grouped_matmul_reduce_sum_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Gather: permuted[p * K + kk] = input[row_map[p] * K + kk]
//
// Unlike GroupedMatMul's gather (which divides by k to reuse a single shared input row across
// a token's k experts), GroupedMatMulReduceSum's `input` has shape (M, k, K) -- one distinct
// row per (token, expert-slot) selection -- so `row_map[p]` (a selection index in [0, M*k))
// indexes `input` directly, with no division.
template <typename T>
__global__ void GroupedMatMulReduceSumGatherKernel(const T* input, const int64_t* row_map,
                                                    T* permuted, int64_t num_selections, int64_t K) {
  const int64_t p = blockIdx.x;
  if (p >= num_selections) {
    return;
  }
  const int64_t selection = row_map[p];
  const T* src = input + selection * K;
  T* dst = permuted + p * K;
  for (int64_t kk = threadIdx.x; kk < K; kk += blockDim.x) {
    dst[kk] = src[kk];
  }
}

template <typename T>
void LaunchGroupedMatMulReduceSumGather(cudaStream_t stream, const T* input, const int64_t* row_map,
                                        T* permuted, int64_t num_selections, int64_t K) {
  if (num_selections == 0 || K == 0) {
    return;
  }
  constexpr int kThreadsPerBlock = 256;
  const int blocks = static_cast<int>(num_selections);
  GroupedMatMulReduceSumGatherKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      input, row_map, permuted, num_selections, K);
}

#define INSTANTIATE_GROUPED_MATMUL_REDUCE_SUM_GATHER(T)                    \
  template void LaunchGroupedMatMulReduceSumGather<T>(                    \
      cudaStream_t, const T*, const int64_t*, T*, int64_t, int64_t);

INSTANTIATE_GROUPED_MATMUL_REDUCE_SUM_GATHER(float)
INSTANTIATE_GROUPED_MATMUL_REDUCE_SUM_GATHER(half)
INSTANTIATE_GROUPED_MATMUL_REDUCE_SUM_GATHER(onnxruntime::BFloat16)

// One block per token i; each thread strides over N. For each output element, loops over the
// token's k expert slots, gathering from the group-contiguous `permuted` buffer (+ bias) and
// accumulating the combine-weighted sum in float for numerical stability regardless of T.
template <typename T>
__global__ void GroupedMatMulReduceSumFinalizeKernel(
    const T* permuted, const int64_t* pos_of_selection, const int64_t* group_ids,
    const T* combine_weights, const T* bias, T* output, int64_t M, int64_t k, int64_t N) {
  const int64_t i = blockIdx.x;  // token index
  if (i >= M) {
    return;
  }
  for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
    float acc = 0.0f;
    for (int64_t j = 0; j < k; ++j) {
      const int64_t sel = i * k + j;
      const int64_t pos = pos_of_selection[sel];
      float v = static_cast<float>(permuted[pos * N + n]);
      if (bias != nullptr) {
        const int64_t g = group_ids[pos];
        v += static_cast<float>(bias[g * N + n]);
      }
      const float w = static_cast<float>(combine_weights[sel]);
      acc += w * v;
    }
    output[i * N + n] = static_cast<T>(acc);
  }
}

template <typename T>
void LaunchGroupedMatMulReduceSumFinalize(cudaStream_t stream, const T* permuted,
                                          const int64_t* pos_of_selection, const int64_t* group_ids,
                                          const T* combine_weights, const T* bias, T* output,
                                          int64_t M, int64_t k, int64_t N) {
  if (M == 0 || N == 0) {
    return;
  }
  constexpr int kThreadsPerBlock = 256;
  const int blocks = static_cast<int>(M);
  GroupedMatMulReduceSumFinalizeKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      permuted, pos_of_selection, group_ids, combine_weights, bias, output, M, k, N);
}

#define INSTANTIATE_GROUPED_MATMUL_REDUCE_SUM_FINALIZE(T)                                        \
  template void LaunchGroupedMatMulReduceSumFinalize<T>(                                        \
      cudaStream_t, const T*, const int64_t*, const int64_t*, const T*, const T*, T*,           \
      int64_t, int64_t, int64_t);

INSTANTIATE_GROUPED_MATMUL_REDUCE_SUM_FINALIZE(float)
INSTANTIATE_GROUPED_MATMUL_REDUCE_SUM_FINALIZE(half)
INSTANTIATE_GROUPED_MATMUL_REDUCE_SUM_FINALIZE(onnxruntime::BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
