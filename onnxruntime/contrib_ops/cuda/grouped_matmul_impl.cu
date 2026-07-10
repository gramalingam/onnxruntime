// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/grouped_matmul_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Gather: permuted[p * K + kk] = input[(row_map[p] / k) * K + kk]
template <typename T>
__global__ void GroupedMatMulGatherKernel(const T* input, const int64_t* row_map, T* permuted,
                                          int64_t num_selections, int64_t K, int64_t k) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_selections * K;
  if (idx >= total) {
    return;
  }
  const int64_t p = idx / K;
  const int64_t kk = idx % K;
  const int64_t token = row_map[p] / k;
  permuted[idx] = input[token * K + kk];
}

// Scatter: result[row_map[p] * N + n] = permuted[p * N + n] + (bias ? bias[group_ids[p] * N + n] : 0)
template <typename T>
__global__ void GroupedMatMulScatterKernel(const T* permuted, const int64_t* row_map,
                                           const int64_t* group_ids, const T* bias, T* result,
                                           int64_t num_selections, int64_t N) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_selections * N;
  if (idx >= total) {
    return;
  }
  const int64_t p = idx / N;
  const int64_t n = idx % N;
  T value = permuted[idx];
  if (bias != nullptr) {
    value = static_cast<T>(static_cast<float>(value) +
                           static_cast<float>(bias[group_ids[p] * N + n]));
  }
  result[row_map[p] * N + n] = value;
}

// Combine: output[i * N + n] = sum_j combine_weights[i * k + j] * per_expert[(i * k + j) * N + n]
template <typename T>
__global__ void GroupedMatMulCombineKernel(const T* per_expert, const T* combine_weights,
                                           T* output, int64_t M, int64_t k, int64_t N) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = M * N;
  if (idx >= total) {
    return;
  }
  const int64_t i = idx / N;
  const int64_t n = idx % N;
  float acc = 0.0f;
  for (int64_t j = 0; j < k; ++j) {
    const int64_t sel = i * k + j;
    acc += static_cast<float>(combine_weights[sel]) *
           static_cast<float>(per_expert[sel * N + n]);
  }
  output[idx] = static_cast<T>(acc);
}

template <typename T>
void LaunchGroupedMatMulGather(cudaStream_t stream, const T* input, const int64_t* row_map,
                               T* permuted, int64_t num_selections, int64_t K, int64_t k) {
  const int64_t total = num_selections * K;
  if (total == 0) {
    return;
  }
  constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
  const int blocks = static_cast<int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
  GroupedMatMulGatherKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      input, row_map, permuted, num_selections, K, k);
}

template <typename T>
void LaunchGroupedMatMulScatter(cudaStream_t stream, const T* permuted, const int64_t* row_map,
                                const int64_t* group_ids, const T* bias, T* result,
                                int64_t num_selections, int64_t N) {
  const int64_t total = num_selections * N;
  if (total == 0) {
    return;
  }
  constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
  const int blocks = static_cast<int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
  GroupedMatMulScatterKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      permuted, row_map, group_ids, bias, result, num_selections, N);
}

template <typename T>
void LaunchGroupedMatMulCombine(cudaStream_t stream, const T* per_expert, const T* combine_weights,
                                T* output, int64_t M, int64_t k, int64_t N) {
  const int64_t total = M * N;
  if (total == 0) {
    return;
  }
  constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
  const int blocks = static_cast<int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
  GroupedMatMulCombineKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      per_expert, combine_weights, output, M, k, N);
}

#define INSTANTIATE_GROUPED_MATMUL_LAUNCHERS(T)                                                   \
  template void LaunchGroupedMatMulGather<T>(cudaStream_t, const T*, const int64_t*, T*,          \
                                             int64_t, int64_t, int64_t);                          \
  template void LaunchGroupedMatMulScatter<T>(cudaStream_t, const T*, const int64_t*,             \
                                              const int64_t*, const T*, T*, int64_t, int64_t);    \
  template void LaunchGroupedMatMulCombine<T>(cudaStream_t, const T*, const T*, T*,               \
                                              int64_t, int64_t, int64_t);

INSTANTIATE_GROUPED_MATMUL_LAUNCHERS(float)
INSTANTIATE_GROUPED_MATMUL_LAUNCHERS(half)
INSTANTIATE_GROUPED_MATMUL_LAUNCHERS(onnxruntime::BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
