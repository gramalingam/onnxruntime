// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Gathers token rows into group-contiguous order:
//   permuted[p, :] = input[row_map[p], :]   for p in [0, num_tokens), each row has K elements.
template <typename T>
void LaunchGroupedMatMulGather(cudaStream_t stream, const T* input, const int64_t* row_map,
                               T* permuted, int64_t num_tokens, int64_t K);

// Scatters the per-group GEMM results back to the original token order, adding the optional
// per-group bias:
//   output[row_map[p], :] = permuted[p, :] + (bias ? bias[group_ids[p], :] : 0)
// each row has N elements.
template <typename T>
void LaunchGroupedMatMulScatter(cudaStream_t stream, const T* permuted, const int64_t* row_map,
                                const int64_t* group_ids, const T* bias, T* output,
                                int64_t num_tokens, int64_t N);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
