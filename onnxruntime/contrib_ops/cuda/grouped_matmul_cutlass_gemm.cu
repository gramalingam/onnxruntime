// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/grouped_matmul_cutlass_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <string>

#include "core/common/float16.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

bool UseCutlassGroupedMatMulGemm() {
  // Read once per call: this is a benchmarking-only flag, not a hot-path check, so there is no
  // need to cache the parsed value. Uses std::getenv (rather than
  // onnxruntime::ParseEnvironmentVariableWithDefault / Env::Default()) because this file is
  // compiled into the CUDA provider shared library (onnxruntime_providers_cuda), where the
  // internal onnxruntime::Env::Default() symbol is not exported -- calling it directly (without
  // routing through the SHARED_PROVIDER host bridge) causes an undefined-symbol failure when the
  // provider library is dlopen'd at runtime.
  const char* impl = std::getenv("ORT_GROUPED_MATMUL_CUDA_IMPL");
  return impl != nullptr && std::string(impl) == "cutlass";
}

// Transposes `weights` from GroupedMatMul's own row-major [num_groups, K, N] layout into
// row-major [num_groups, N, K] -- the layout CUTLASS's MoeGemmRunner expects for its B operand
// (equivalently column-major [K, N]; see the layout note in grouped_matmul_cutlass_gemm.h). One
// block per (group, output-feature-tile); straightforward (non-tiled-shared-memory) transpose
// since this is a benchmarking-only, opt-in path rather than a hot loop.
template <typename T>
__global__ void TransposeGroupedWeightsKernel(const T* weights, T* weights_t, int64_t K, int64_t N) {
  const int64_t g = blockIdx.y;
  const T* src = weights + g * K * N;
  T* dst = weights_t + g * K * N;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < K * N; idx += gridDim.x * blockDim.x) {
    const int64_t k = idx / N;
    const int64_t n = idx % N;
    dst[n * K + k] = src[idx];
  }
}

template <typename T>
void LaunchTransposeGroupedWeights(cudaStream_t stream, const T* weights, T* weights_t, int64_t num_groups,
                                   int64_t K, int64_t N) {
  if (num_groups == 0 || K == 0 || N == 0) {
    return;
  }
  constexpr int kThreadsPerBlock = 256;
  const int64_t total = K * N;
  const int blocks_x = static_cast<int>(std::min<int64_t>((total + kThreadsPerBlock - 1) / kThreadsPerBlock, 65535));
  const dim3 grid(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(num_groups));
  TransposeGroupedWeightsKernel<T><<<grid, kThreadsPerBlock, 0, stream>>>(weights, weights_t, K, N);
}

namespace {

// Shared implementation templated on the CUTLASS/CUDA-native element type (float, half or
// __nv_bfloat16 -- the types MoeGemmRunner is explicitly instantiated for). The public entry
// point below bridges onnxruntime::BFloat16 (the CudaT used by GroupedMatMul's CUDA kernel for
// the BFloat16 registration) to __nv_bfloat16 via reinterpret_cast, since the two types share the
// same 16-bit layout, exactly as other reinterpret_cast<const CudaT*> uses already do in
// grouped_matmul.cc for MLFloat16/half.
//
// `weights` must already be in CUTLASS's expected [num_groups, N, K] row-major layout (i.e. the
// caller has run LaunchTransposeGroupedWeights first) -- this function does not transpose.
template <typename CutlassT>
void RunGroupedMatMulCutlassGemm(cudaStream_t stream, const CutlassT* permuted_input, const CutlassT* weights,
                                 const int64_t* group_offsets_end, CutlassT* permuted_output,
                                 int64_t num_selections, int64_t K, int64_t N, int64_t num_groups) {
  using onnxruntime::llm::kernels::cutlass_kernels::ActivationType;
  using onnxruntime::llm::kernels::cutlass_kernels::GroupedGemmInput;
  using onnxruntime::llm::kernels::cutlass_kernels::MoeGemmRunner;
  using onnxruntime::llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;

  MoeGemmRunner<CutlassT, CutlassT, CutlassT> runner;
  const int sm = onnxruntime::llm::common::getSMVersion();

  // This benchmarking-only path always takes the first non-TMA ("Ampere") tactic candidate rather
  // than running com.microsoft.MoE's tactic profiler (see moe.cc's mGemmProfiler): on this
  // codebase's build target (SM80, no Hopper/TMA-WS kernels compiled), getConfigs(sm) only ever
  // returns non-TMA candidates anyway, so this is a deterministic, always-valid choice.
  const auto configs = MoeGemmRunner<CutlassT, CutlassT, CutlassT>::getConfigs(sm);
  ORT_ENFORCE(!configs.empty(), "GroupedMatMul (cutlass impl): no CUTLASS grouped-GEMM tactic available for SM ", sm);
  auto config = configs[0];
  for (const auto& c : configs) {
    if (!c.is_tma_warp_specialized) {
      config = c;
      break;
    }
  }
  ORT_ENFORCE(!config.is_tma_warp_specialized,
              "GroupedMatMul (cutlass impl): only non-TMA-warp-specialized tactics are supported.");

  GroupedGemmInput<CutlassT, CutlassT, CutlassT, CutlassT> inputs;
  inputs.A = permuted_input;
  inputs.total_tokens_including_expert = group_offsets_end;
  inputs.B = weights;
  inputs.scales = nullptr;
  inputs.zeros = nullptr;
  inputs.biases = nullptr;  // Bias is added by the caller's scatter kernel instead, see header comment.
  inputs.C = permuted_output;
  inputs.alpha_scales = nullptr;
  inputs.occupancy = nullptr;
  inputs.activation_type = ActivationType::Identity;
  inputs.num_rows = num_selections;
  inputs.n = N;
  inputs.k = K;
  inputs.num_experts = static_cast<int>(num_groups);
  inputs.bias_is_broadcast = true;
  inputs.use_fused_moe = false;
  inputs.stream = stream;
  inputs.gemm_config = config;

  runner.moeGemmBiasAct(inputs, TmaWarpSpecializedGroupedGemmInput{});
}

}  // namespace

template <typename T>
void LaunchGroupedMatMulCutlassGemm(cudaStream_t stream, const T* permuted_input, const T* weights,
                                    T* weight_scratch, const int64_t* group_offsets_end, T* permuted_output,
                                    int64_t num_selections, int64_t K, int64_t N, int64_t num_groups) {
  LaunchTransposeGroupedWeights<T>(stream, weights, weight_scratch, num_groups, K, N);
  RunGroupedMatMulCutlassGemm<T>(stream, permuted_input, weight_scratch, group_offsets_end, permuted_output,
                                 num_selections, K, N, num_groups);
}

template void LaunchGroupedMatMulCutlassGemm<float>(cudaStream_t, const float*, const float*, float*,
                                                     const int64_t*, float*, int64_t, int64_t, int64_t, int64_t);
template void LaunchGroupedMatMulCutlassGemm<half>(cudaStream_t, const half*, const half*, half*, const int64_t*,
                                                    half*, int64_t, int64_t, int64_t, int64_t);

// GroupedMatMul's BFloat16 CUDA kernel registration uses onnxruntime::cuda::ToCudaType<BFloat16>,
// whose MappedType is onnxruntime::BFloat16 itself (not __nv_bfloat16). Bridge the two here via
// reinterpret_cast: both are trivial 16-bit-wide wrappers with identical bit layout.
template <>
void LaunchGroupedMatMulCutlassGemm<onnxruntime::BFloat16>(
    cudaStream_t stream, const onnxruntime::BFloat16* permuted_input, const onnxruntime::BFloat16* weights,
    onnxruntime::BFloat16* weight_scratch, const int64_t* group_offsets_end,
    onnxruntime::BFloat16* permuted_output, int64_t num_selections, int64_t K, int64_t N, int64_t num_groups) {
  LaunchTransposeGroupedWeights<__nv_bfloat16>(stream, reinterpret_cast<const __nv_bfloat16*>(weights),
                                               reinterpret_cast<__nv_bfloat16*>(weight_scratch), num_groups, K, N);
  RunGroupedMatMulCutlassGemm<__nv_bfloat16>(
      stream, reinterpret_cast<const __nv_bfloat16*>(permuted_input),
      reinterpret_cast<const __nv_bfloat16*>(weight_scratch),
      group_offsets_end, reinterpret_cast<__nv_bfloat16*>(permuted_output), num_selections, K, N, num_groups);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
