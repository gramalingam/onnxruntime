// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/router_topk_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// One thread per row. num_experts (E) is small in MoE routers, so an O(k*E) iterative
// argmax selection (no per-thread dynamic storage) is both simple and fast.
template <typename T>
__global__ void RouterTopKKernel(const T* logits, T* weights, int64_t* indices,
                                 int64_t rows, int64_t E, int64_t k) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const T* row_logits = logits + row * E;
  T* row_weights = weights + row * k;
  int64_t* row_indices = indices + row * k;

  // Select the k largest logits in (value desc, index asc) order. `prev` tracks the last
  // selection so each iteration picks the next-smaller element in that total order.
  float prev_val = 0.0f;
  int64_t prev_idx = -1;
  bool have_prev = false;
  float max_logit = 0.0f;
  for (int64_t j = 0; j < k; ++j) {
    float best_val = 0.0f;
    int64_t best_idx = -1;
    for (int64_t e = 0; e < E; ++e) {
      const float v = static_cast<float>(row_logits[e]);
      // Candidate must come strictly after the previous selection in the total order.
      const bool after_prev = !have_prev || (v < prev_val) || (v == prev_val && e > prev_idx);
      const bool better = (best_idx < 0) || (v > best_val) || (v == best_val && e < best_idx);
      if (after_prev && better) {
        best_val = v;
        best_idx = e;
      }
    }
    row_indices[j] = best_idx;
    if (j == 0) {
      max_logit = best_val;
    }
    prev_val = best_val;
    prev_idx = best_idx;
    have_prev = true;
  }

  // Softmax over the k selected logits (max_logit is the largest, index 0). Read logit values
  // back through the stored indices to avoid a low-precision round-trip for half/bfloat16.
  float sum = 0.0f;
  for (int64_t j = 0; j < k; ++j) {
    const float e = expf(static_cast<float>(row_logits[row_indices[j]]) - max_logit);
    row_weights[j] = static_cast<T>(e);
    sum += e;
  }
  const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
  for (int64_t j = 0; j < k; ++j) {
    row_weights[j] = static_cast<T>(static_cast<float>(row_weights[j]) * inv_sum);
  }
}

template <typename T>
void LaunchRouterTopKKernel(cudaStream_t stream, const T* logits, T* weights, int64_t* indices,
                            int64_t rows, int64_t E, int64_t k) {
  if (rows == 0) {
    return;
  }
  constexpr int kThreadsPerBlock = 256;
  const int blocks = static_cast<int>((rows + kThreadsPerBlock - 1) / kThreadsPerBlock);
  RouterTopKKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(logits, weights, indices, rows, E, k);
}

#define INSTANTIATE_ROUTER_TOPK_LAUNCHER(T) \
  template void LaunchRouterTopKKernel<T>(cudaStream_t, const T*, T*, int64_t*, int64_t, int64_t, int64_t);

INSTANTIATE_ROUTER_TOPK_LAUNCHER(float)
INSTANTIATE_ROUTER_TOPK_LAUNCHER(half)
INSTANTIATE_ROUTER_TOPK_LAUNCHER(onnxruntime::BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
