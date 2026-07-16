// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Elementwise fused SwiGLU: output[i] = gate[i] * sigmoid(alpha * gate[i]) * linear[i].
template <typename T>
void LaunchSwiGLUKernel(cudaStream_t stream, const T* gate, const T* linear, T* output,
                        float alpha, int64_t count);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
