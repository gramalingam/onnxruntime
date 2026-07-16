// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// For each of `rows` rows of `logits` (row length `E`), selects the top-k logits (largest,
// sorted descending with smaller index winning ties) and writes their renormalized softmax
// weights to `weights` and their expert indices to `indices` (both row length `k`).
template <typename T>
void LaunchRouterTopKKernel(cudaStream_t stream, const T* logits, T* weights, int64_t* indices,
                            int64_t rows, int64_t E, int64_t k);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
