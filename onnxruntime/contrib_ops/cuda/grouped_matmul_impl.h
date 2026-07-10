// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Gathers each selection's source token row into group-contiguous order. A selection index
// `sel` in [0, num_selections) maps to source token `sel / k`:
//   permuted[p, :] = input[row_map[p] / k, :]   for p in [0, num_selections), K elements/row.
template <typename T>
void LaunchGroupedMatMulGather(cudaStream_t stream, const T* input, const int64_t* row_map,
                               T* permuted, int64_t num_selections, int64_t K, int64_t k);

// Scatters the per-group GEMM results back to selection order, adding the optional per-group
// bias:
//   result[row_map[p], :] = permuted[p, :] + (bias ? bias[group_ids[p], :] : 0)
// each row has N elements. `result` is laid out as [num_selections, N] (i.e. [M, k, N]).
template <typename T>
void LaunchGroupedMatMulScatter(cudaStream_t stream, const T* permuted, const int64_t* row_map,
                                const int64_t* group_ids, const T* bias, T* result,
                                int64_t num_selections, int64_t N);

// Reduces the per-expert results [M, k, N] into the combined output [M, N] using the
// per-selection combine weights [M, k]:
//   output[i, n] = sum_j combine_weights[i, j] * per_expert[i, j, n]
template <typename T>
void LaunchGroupedMatMulCombine(cudaStream_t stream, const T* per_expert, const T* combine_weights,
                                T* output, int64_t M, int64_t k, int64_t N);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
