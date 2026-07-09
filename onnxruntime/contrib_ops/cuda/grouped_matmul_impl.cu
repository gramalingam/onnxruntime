// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/grouped_matmul_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Gather: permuted[p * K + k] = input[row_map[p] * K + k]
template <typename T>
__global__ void GroupedMatMulGatherKernel(const T* input, const int64_t* row_map, T* permuted,
                                          int64_t num_tokens, int64_t K) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_tokens * K;
  if (idx >= total) {
    return;
  }
  const int64_t p = idx / K;
  const int64_t k = idx % K;
  permuted[idx] = input[row_map[p] * K + k];
}

// Scatter: output[row_map[p] * N + n] = permuted[p * N + n] + (bias ? bias[group_ids[p] * N + n] : 0)
template <typename T>
__global__ void GroupedMatMulScatterKernel(const T* permuted, const int64_t* row_map,
                                           const int64_t* group_ids, const T* bias, T* output,
                                           int64_t num_tokens, int64_t N) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_tokens * N;
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
  output[row_map[p] * N + n] = value;
}

template <typename T>
void LaunchGroupedMatMulGather(cudaStream_t stream, const T* input, const int64_t* row_map,
                               T* permuted, int64_t num_tokens, int64_t K) {
  const int64_t total = num_tokens * K;
  if (total == 0) {
    return;
  }
  constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
  const int blocks = static_cast<int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
  GroupedMatMulGatherKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      input, row_map, permuted, num_tokens, K);
}

template <typename T>
void LaunchGroupedMatMulScatter(cudaStream_t stream, const T* permuted, const int64_t* row_map,
                                const int64_t* group_ids, const T* bias, T* output,
                                int64_t num_tokens, int64_t N) {
  const int64_t total = num_tokens * N;
  if (total == 0) {
    return;
  }
  constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
  const int blocks = static_cast<int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
  GroupedMatMulScatterKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      permuted, row_map, group_ids, bias, output, num_tokens, N);
}

#define INSTANTIATE_GROUPED_MATMUL_LAUNCHERS(T)                                                   \
  template void LaunchGroupedMatMulGather<T>(cudaStream_t, const T*, const int64_t*, T*,          \
                                             int64_t, int64_t);                                   \
  template void LaunchGroupedMatMulScatter<T>(cudaStream_t, const T*, const int64_t*,             \
                                              const int64_t*, const T*, T*, int64_t, int64_t);

INSTANTIATE_GROUPED_MATMUL_LAUNCHERS(float)
INSTANTIATE_GROUPED_MATMUL_LAUNCHERS(half)
INSTANTIATE_GROUPED_MATMUL_LAUNCHERS(onnxruntime::BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
