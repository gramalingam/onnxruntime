// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Whether the CUTLASS single-launch grouped-GEMM path (see LaunchGroupedMatMulCutlassGemm below)
// is requested for GroupedMatMul's CUDA kernel, via ORT_GROUPED_MATMUL_CUDA_IMPL=cutlass
// (default/unset, or any other value, keeps the original per-group cuBLAS loop). This is a
// benchmarking-only switch: both implementations are kept side by side so their performance can
// be compared directly, see experiments/grouped_matmul_perf/README.md.
bool UseCutlassGroupedMatMulGemm();

// Runs the dense per-group GEMM step of GroupedMatMul as a *single* CUTLASS grouped-GEMM kernel
// launch covering all `num_groups` groups at once -- the same MoeGemmRunner machinery that backs
// com.microsoft.MoE's GEMM1/GEMM2 -- instead of one cuBLAS launch per non-empty group.
//
// Preconditions (mirroring the cuBLAS loop this replaces in grouped_matmul.cc):
//   - `permuted_input` ([num_selections, K]) and `permuted_output` ([num_selections, N]) are
//     already gathered/to-be-scattered in group-contiguous order: rows
//     [group_offsets_end[g - 1], group_offsets_end[g]) belong to group g (group_offsets_end[-1] ==
//     0), for g in [0, num_groups).
//   - `group_offsets_end` is a *device* array of cumulative row counts (i.e. the exclusive-end
//     offset of each group in the permuted order), size num_groups, dtype int64. This is exactly
//     CUTLASS's `total_tokens_including_expert` input.
//   - `weights` is laid out as [num_groups, K, N] row-major, matching GroupedMatMul's weight
//     layout directly. NOTE: CUTLASS's MoeGemmRunner expects the weight operand in *its* native
//     MoE layout, which is row-major [num_groups, N, K] (equivalently column-major [K, N]) --
//     e.g. com.microsoft.MoE's own fc2_experts_weights is documented as
//     (num_experts, hidden_size, inter_size) = (num_groups, N, K). Since that's the transpose of
//     GroupedMatMul's own [K, N] convention, this function transposes `weights` into
//     `weight_scratch` (caller-allocated, same total size as `weights`) before the GEMM launch.
//   - `weight_scratch` is a caller-allocated device buffer of size num_groups * K * N (same as
//     `weights`), used as scratch space for the transposed weights described above.
//
// Bias is intentionally NOT applied by this function: the caller's existing scatter kernel adds
// the per-group bias (matching the cuBLAS loop's behavior), so the two GEMM-execution strategies
// can be compared without any difference in bias handling.
//
// KNOWN LIMITATION (benchmarking-only path, not used by default): the underlying CUTLASS SIMT
// (fp32) grouped-GEMM kernel has been observed to produce incorrect results for pathological tiny
// problem sizes where *both* K and N are simultaneously non-tile-aligned (e.g. K=33, N=29 -- either
// one alone paired with a tile-aligned partner is fine). Verified correct at all realistic MoE
// scales tested (K/N in the hundreds to low thousands, as used by real models where hidden_size /
// inter_size are typically multiples of 64/128). See experiments/grouped_matmul_perf/README.md for
// details; not investigated further since this is an opt-in benchmarking switch, off by default.
template <typename T>
void LaunchGroupedMatMulCutlassGemm(cudaStream_t stream, const T* permuted_input, const T* weights,
                                    T* weight_scratch, const int64_t* group_offsets_end, T* permuted_output,
                                    int64_t num_selections, int64_t K, int64_t N, int64_t num_groups);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
