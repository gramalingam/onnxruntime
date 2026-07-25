// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Combines each token's k expert results (already computed into `permuted`, one row per
// selection in group-contiguous order, exactly as produced for GroupedMatMul) into a single
// weighted-sum output row, fusing GroupedMatMul's down-projection with a subsequent
// Mul(combine_weights) + ReduceSum(axis=1):
//
//   output[i, n] = sum_{j=0}^{k-1} combine_weights[i, j] *
//                  (permuted[pos_of_selection[i*k+j], n] + (bias ? bias[group_ids[pos], n] : 0))
//
// `pos_of_selection[s]` maps selection index s = i*k+j (in [0, M*k)) to its position in the
// group-contiguous `permuted` buffer -- the inverse of GroupedMatMul's `row_map`
// (row_map[pos_of_selection[s]] == s). `group_ids[pos]` is the group id at permuted position
// `pos` (same array GroupedMatMul's scatter step uses).
template <typename T>
void LaunchGroupedMatMulReduceSumFinalize(cudaStream_t stream, const T* permuted,
                                          const int64_t* pos_of_selection, const int64_t* group_ids,
                                          const T* combine_weights, const T* bias, T* output,
                                          int64_t M, int64_t k, int64_t N);

// Gather each (token, expert-slot) selection's own input row -- see the .cu file for why this
// differs from GroupedMatMul's row-reusing gather.
template <typename T>
void LaunchGroupedMatMulReduceSumGather(cudaStream_t stream, const T* input, const int64_t* row_map,
                                        T* permuted, int64_t num_selections, int64_t K);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
