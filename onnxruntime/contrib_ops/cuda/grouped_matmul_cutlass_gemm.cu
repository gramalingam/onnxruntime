// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/grouped_matmul_cutlass_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

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

// ---------------------------------------------------------------------------------------------
// Tactic/config profiler for the CUTLASS grouped-GEMM path, mirroring com.microsoft.MoE's
// GemmProfiler/MoeGemmProfiler (llm/moe_gemm/moe_gemm_profiler.{h,cc}): times every candidate
// CUTLASS tactic on-device and caches the fastest one, instead of always taking
// getConfigs(sm)[0]. Kept self-contained in this .cu file (a process-wide cache guarded by a
// mutex) rather than threaded through GroupedMatMul's OpKernel, since this is an opt-in
// benchmarking path with no per-session state elsewhere.
//
// Cache key: (N, K, num_groups, dtype, M-bucket). M is bucketed to the next power of two (capped
// at 8192, with M<=1 as its own bucket) for the same reason MoE's profiler does this: decode
// (small M) and prefill (large M) prefer different CUTLASS tile shapes, so each bucket gets its
// own tuned tactic rather than sharing one shape-only config across very different row counts.
using CutlassGemmConfig = onnxruntime::llm::cutlass_extensions::CutlassGemmConfig;

struct GroupedGemmProfileId {
  int64_t n{0};
  int64_t k{0};
  int64_t num_groups{0};
  int dtype_tag{0};  // Distinguishes float/half/bf16 instantiations sharing this cache.

  bool operator==(const GroupedGemmProfileId& other) const {
    return n == other.n && k == other.k && num_groups == other.num_groups && dtype_tag == other.dtype_tag;
  }
};

struct GroupedGemmProfileIdHash {
  std::size_t operator()(const GroupedGemmProfileId& id) const {
    std::size_t h = std::hash<int64_t>{}(id.n);
    h ^= std::hash<int64_t>{}(id.k) << 1;
    h ^= std::hash<int64_t>{}(id.num_groups) << 2;
    h ^= std::hash<int>{}(id.dtype_tag) << 3;
    return h;
  }
};

// Snap M to a representative profiling bucket -- same scheme as MoeGemmProfiler::bucketM.
int BucketM(int64_t m) {
  if (m <= 1) {
    return 1;
  }
  constexpr int64_t kMaxBucket = 8192;
  if (m >= kMaxBucket) {
    return static_cast<int>(kMaxBucket);
  }
  int64_t bucket = 1;
  while (bucket < m) {
    bucket <<= 1;
  }
  return static_cast<int>(bucket);
}

bool ProfilingEnabled() {
  // Opt-out escape hatch (e.g. for debugging, or environments where the extra warmup/timed
  // launches during the first call for each shape are undesirable): ORT_GROUPED_MATMUL_CUTLASS_PROFILE=0
  // reverts to always using the first available tactic, matching this path's original behavior.
  const char* val = std::getenv("ORT_GROUPED_MATMUL_CUTLASS_PROFILE");
  return val == nullptr || std::string(val) != "0";
}

// Process-wide cache: GroupedGemmProfileId -> (M bucket -> best config). Guarded by
// gProfileCacheMutex since GroupedMatMul instances (potentially across sessions/threads) share it.
std::unordered_map<GroupedGemmProfileId, std::unordered_map<int, CutlassGemmConfig>, GroupedGemmProfileIdHash>
    gProfileCache;
std::mutex gProfileCacheMutex;

template <typename T>
constexpr int DtypeTag();
template <>
constexpr int DtypeTag<float>() { return 0; }
template <>
constexpr int DtypeTag<half>() { return 1; }
template <>
constexpr int DtypeTag<__nv_bfloat16>() { return 2; }

// Times `config` by running the real GEMM call (via `run_once`) for a few warmup iterations
// followed by several timed iterations bracketed by CUDA events, on `stream`. Writes its
// (discarded) results into whatever output buffer `run_once` targets -- for this profiler that is
// always the caller's real `permuted_output` scratch buffer, which is safe to clobber here since
// the subsequent real GEMM call (with the chosen best config) overwrites it again anyway.
// Returns nullopt if the tactic fails (e.g. a tile-alignment incompatibility for this shape --
// see the KNOWN LIMITATION note in the header) rather than propagating the exception, so a single
// bad candidate does not abort profiling of the remaining ones.
template <typename RunOnceFn>
std::optional<float> TimeTactic(cudaStream_t stream, RunOnceFn&& run_once) {
  constexpr int kWarmupIters = 3;
  constexpr int kTimedIters = 10;
  try {
    for (int i = 0; i < kWarmupIters; ++i) {
      run_once();
    }
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CALL_THROW(cudaEventCreate(&start));
    CUDA_CALL_THROW(cudaEventCreate(&stop));
    CUDA_CALL_THROW(cudaEventRecord(start, stream));
    for (int i = 0; i < kTimedIters; ++i) {
      run_once();
    }
    CUDA_CALL_THROW(cudaEventRecord(stop, stream));
    CUDA_CALL_THROW(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CALL_THROW(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CALL_THROW(cudaEventDestroy(start));
    CUDA_CALL_THROW(cudaEventDestroy(stop));
    return elapsed_ms / kTimedIters;
  } catch (const std::exception& e) {
    // Deliberately std::cerr, not ORT's logging macros: this file is compiled into
    // onnxruntime_providers_cuda, which is dlopen'd at runtime without exporting ORT's internal
    // default-logger symbol -- using LOGS_DEFAULT here causes an undefined-symbol load failure
    // (same class of issue as the std::getenv note in UseCutlassGroupedMatMulGemm above).
    std::cerr << "GroupedMatMul (cutlass impl): tactic profiling attempt failed, skipping: " << e.what() << std::endl;
    cudaGetLastError();  // Clear any sticky CUDA error before trying the next tactic.
    return std::nullopt;
  }
}

// Picks (profiling and caching on first use, else returning the cached choice) the best CUTLASS
// tactic for this GEMM shape and M-bucket. `run_with_config(config)` runs the real GEMM call with
// that candidate's config -- the caller supplies this so this function stays independent of the
// GroupedGemmInput/element-type details.
template <typename CutlassT, typename RunWithConfigFn>
CutlassGemmConfig SelectGroupedMatMulGemmConfig(cudaStream_t stream, int64_t num_selections, int64_t K, int64_t N,
                                                int64_t num_groups, int sm,
                                                const std::vector<CutlassGemmConfig>& candidates,
                                                RunWithConfigFn&& run_with_config) {
  ORT_ENFORCE(!candidates.empty(),
              "GroupedMatMul (cutlass impl): no CUTLASS grouped-GEMM tactic available for SM ", sm);

  // Profiling launches kernels, records/synchronizes CUDA events on `stream`; both are illegal
  // while that stream is being captured into a CUDA graph (mirrors moe.cc's capture check).
  const bool stream_is_capturing = onnxruntime::llm::common::isCapturing(stream);
  const GroupedGemmProfileId id{N, K, num_groups, DtypeTag<CutlassT>()};
  const int bucket = BucketM(num_selections);

  {
    std::lock_guard<std::mutex> lock(gProfileCacheMutex);
    auto it = gProfileCache.find(id);
    if (it != gProfileCache.end()) {
      auto bucket_it = it->second.find(bucket);
      if (bucket_it != it->second.end()) {
        return bucket_it->second;
      }
    }
  }

  if (!ProfilingEnabled() || stream_is_capturing) {
    // Deterministic fallback: first candidate, matching this path's original behavior. Not
    // cached, so profiling can still happen normally once profiling becomes possible/enabled
    // (e.g. once graph capture finishes).
    return candidates[0];
  }

  float best_time = std::numeric_limits<float>::max();
  CutlassGemmConfig best_config = candidates[0];
  bool found_one = false;
  for (const auto& config : candidates) {
    auto elapsed = TimeTactic(stream, [&]() { run_with_config(config); });
    if (elapsed.has_value() && *elapsed < best_time) {
      best_time = *elapsed;
      best_config = config;
      found_one = true;
    }
  }
  if (!found_one) {
    std::cerr << "GroupedMatMul (cutlass impl): all tactics failed to profile for shape "
              << "(N=" << N << ", K=" << K << ", num_groups=" << num_groups << ", M=" << num_selections
              << "); falling back to the first candidate." << std::endl;
  }

  {
    std::lock_guard<std::mutex> lock(gProfileCacheMutex);
    gProfileCache[id][bucket] = best_config;
  }
  return best_config;
}

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

  // Only non-TMA ("Ampere") tactics are supported by this path (see the header comment for the
  // isTmaWarpSpecialized check below); this codebase's build target (SM80, no Hopper/TMA-WS
  // kernels compiled) only ever produces non-TMA candidates anyway, so this filter is a no-op in
  // practice but kept for correctness if that ever changes.
  const auto all_configs = MoeGemmRunner<CutlassT, CutlassT, CutlassT>::getConfigs(sm);
  std::vector<CutlassGemmConfig> candidates;
  for (const auto& c : all_configs) {
    if (!c.is_tma_warp_specialized) {
      candidates.push_back(c);
    }
  }

  auto build_inputs = [&](const CutlassGemmConfig& config) {
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
    return inputs;
  };

  // Tactic/config profiling: times every candidate on-device (first use of this
  // (N, K, num_groups, dtype, M-bucket) shape only; cached thereafter) and picks the fastest,
  // instead of always using candidates[0]. See SelectGroupedMatMulGemmConfig above.
  auto config = SelectGroupedMatMulGemmConfig<CutlassT>(
      stream, num_selections, K, N, num_groups, sm, candidates,
      [&](const CutlassGemmConfig& c) { runner.moeGemmBiasAct(build_inputs(c), TmaWarpSpecializedGroupedGemmInput{}); });

  runner.moeGemmBiasAct(build_inputs(config), TmaWarpSpecializedGroupedGemmInput{});
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
