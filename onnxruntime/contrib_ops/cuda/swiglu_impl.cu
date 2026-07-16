// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/swiglu_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void SwiGLUKernel(const T* gate, const T* linear, T* output, float alpha, int64_t count) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const float g = static_cast<float>(gate[idx]);
  const float l = static_cast<float>(linear[idx]);
  const float sig = 1.0f / (1.0f + expf(-alpha * g));
  output[idx] = static_cast<T>(g * sig * l);
}

template <typename T>
void LaunchSwiGLUKernel(cudaStream_t stream, const T* gate, const T* linear, T* output,
                        float alpha, int64_t count) {
  if (count == 0) {
    return;
  }
  constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
  const int blocks = static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
  SwiGLUKernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(gate, linear, output, alpha, count);
}

#define INSTANTIATE_SWIGLU_LAUNCHER(T) \
  template void LaunchSwiGLUKernel<T>(cudaStream_t, const T*, const T*, T*, float, int64_t);

INSTANTIATE_SWIGLU_LAUNCHER(float)
INSTANTIATE_SWIGLU_LAUNCHER(half)
INSTANTIATE_SWIGLU_LAUNCHER(onnxruntime::BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
