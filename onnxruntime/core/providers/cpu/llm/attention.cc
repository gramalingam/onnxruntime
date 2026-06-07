// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/llm/attention.h"
#include "core/providers/cpu/llm/attention_helper.h"
#include "core/providers/cpu/llm/attention_softmax.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/env.h"
#include "core/platform/env_var_utils.h"
#include "core/platform/threadpool.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/gemm.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using onnxruntime::attention_helper::AttentionParameters;
using onnxruntime::attention_helper::QKMatMulOutputMode;
using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {

#define REGISTER_ONNX_KERNEL_TYPED(T)                                 \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                     \
      Attention,                                                      \
      24,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", BuildKernelDefConstraints<bool, T>()), \
      Attention<T>);

REGISTER_ONNX_KERNEL_TYPED(float)
REGISTER_ONNX_KERNEL_TYPED(MLFloat16)

#define REGISTER_ONNX_KERNEL_VERSIONED_TYPED(T)                       \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                           \
      Attention,                                                      \
      23,                                                             \
      23,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())     \
          .TypeConstraint("U", BuildKernelDefConstraints<bool, T>()), \
      Attention<T>);

REGISTER_ONNX_KERNEL_VERSIONED_TYPED(float)
REGISTER_ONNX_KERNEL_VERSIONED_TYPED(MLFloat16)

template <typename T, typename U>
void make_copy(T* mask_data, const U* mask_index, size_t size);

template <>
void make_copy<float, float>(float* mask_data, const float* mask_index, size_t size) {
  memcpy(mask_data, mask_index, size * sizeof(float));
}

template <>
void make_copy<MLFloat16, MLFloat16>(MLFloat16* mask_data, const MLFloat16* mask_index, size_t size) {
  memcpy(mask_data, mask_index, size * sizeof(MLFloat16));
}

template <>
void make_copy<float, bool>(float* mask_data, const bool* mask_index, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    mask_data[i] = mask_index[i] ? 0.0f : mask_filter_value<float>();
  }
}

template <>
void make_copy<MLFloat16, bool>(MLFloat16* mask_data, const bool* mask_index, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    mask_data[i] = mask_index[i] ? MLFloat16(0.f) : mask_filter_value<MLFloat16>();
  }
}

template <typename T>
inline void ComputeAttentionSoftcapInplace(T* scores, int sequence_length, T softcap) {
  MlasComputeSoftcap(scores, scores, sequence_length, softcap);
}

template <>
inline void ComputeAttentionSoftcapInplace(MLFloat16* scores, int sequence_length, MLFloat16 softcap) {
  // Mlas Lacks kernels for fp16 softcap. The code is similar to the softcap implementation in mlas.
  float x;
  float cap = softcap.ToFloat();
  for (size_t i = 0; i < static_cast<size_t>(sequence_length); i++) {
    x = std::tanh(scores[i].ToFloat() / cap) * cap;
    scores[i] = MLFloat16(x);
  }
}

// In-place elementwise add: scores[i] += addend[i].
//
// Used to apply attn_mask / attn_bias to the QK scores after softcap, per the
// ONNX Attention v23/24 spec ordering (onnx/onnx#7867 + #7913). For float,
// delegates to MLAS. For MLFloat16, uses a portable scalar fallback because
// MlasEltwiseAdd<MLAS_FP16> requires the per-platform EltwiseDispatch->Add_Fp16
// kernel slot to be populated, and only the ARM NEON build provides it
// (see onnxruntime/core/mlas/lib/eltwise.cpp:92-103); x86 and other targets
// would throw at runtime.
template <typename T>
inline void AddInPlace(T* scores, const T* addend, size_t count) {
  MlasEltwiseAdd<T>(addend, scores, scores, count);
}

template <>
inline void AddInPlace<MLFloat16>(MLFloat16* scores, const MLFloat16* addend, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    scores[i] = MLFloat16(scores[i].ToFloat() + addend[i].ToFloat());
  }
}

// Dispatches a GEMM operation across float and MLFloat16 types.
//   C = alpha * op(A) * op(B) + beta * C
//
// For float: delegates to math::GemmEx which calls MlasGemm (optimized SGEMM).
// For MLFloat16:
//   - If the hardware supports native fp16 GEMM for the given transpose combo
//     (checked via MlasHGemmSupported), uses MlasGemm directly.
//   - Otherwise, upcasts A/B/C to fp32, runs math::GemmEx (SGEMM), and downcasts
//     the result back to fp16.  This avoids Eigen's unoptimized fp16 codepath.
//
// The fp32 fallback handles strided C carefully: when ldc > N (e.g. 3D interleaved
// heads where multiple heads share a row), conversion is done row-by-row (N elements
// per row) to avoid overwriting adjacent heads' data.  When ldc == N (contiguous,
// the common 4D case), a single bulk conversion is used for efficiency.
//
// TODO(xadupre): Consider adding a MlasFlashAttention fast path for float32 when no masks, KV cache,
// softcap, or nonpad_kv_seqlen are active. This fuses Q*K, softmax, and QK*V into a single
// L2-cache-tiled pass. See MultiHeadAttention (contrib_ops/cpu/bert/multihead_attention.cc).
template <typename T>
inline void AttentionGemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                          int M, int N, int K,
                          float alpha,
                          const T* A, int lda,
                          const T* B, int ldb,
                          float beta,
                          T* C, int ldc,
                          const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* mlas_backend_kernel_selector_config) {
  if constexpr (std::is_same<T, float>::value) {
    math::GemmEx<T, ThreadPool>(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, nullptr,
                                mlas_backend_kernel_selector_config);
  } else if constexpr (std::is_same<T, MLFloat16>::value) {
    if (MlasHGemmSupported(transA, transB)) {
      MlasGemm(transA, transB, M, N, K, A, lda, B, ldb, C, ldc,
               MLFloat16(alpha).val, MLFloat16(beta).val, nullptr);
    } else {
      // fp16 fallback: upcast to fp32, run optimized SGEMM, downcast result.
      // Compute the exact contiguous span each matrix occupies: (rows-1)*stride + cols.
      // This is the distance from the first element to the last accessed element + 1.
      // Using rows*stride would overread when the pointer is offset into a larger
      // interleaved buffer (e.g., 3D layout where lda > K for a non-first head).
      size_t a_rows = (transA == CblasNoTrans) ? static_cast<size_t>(M) : static_cast<size_t>(K);
      size_t a_cols = (transA == CblasNoTrans) ? static_cast<size_t>(K) : static_cast<size_t>(M);
      size_t b_rows = (transB == CblasNoTrans) ? static_cast<size_t>(K) : static_cast<size_t>(N);
      size_t b_cols = (transB == CblasNoTrans) ? static_cast<size_t>(N) : static_cast<size_t>(K);
      size_t a_count = (a_rows > 0) ? (a_rows - 1) * static_cast<size_t>(lda) + a_cols : 0;
      size_t b_count = (b_rows > 0) ? (b_rows - 1) * static_cast<size_t>(ldb) + b_cols : 0;
      size_t c_count = (M > 0) ? static_cast<size_t>(M - 1) * static_cast<size_t>(ldc) + static_cast<size_t>(N) : 0;

      std::vector<float> a_fp32(a_count);
      std::vector<float> b_fp32(b_count);
      std::vector<float> c_fp32(c_count);

      // Upcast A and B in bulk (contiguous within each matrix's strided span).
      MlasConvertHalfToFloatBuffer(A, a_fp32.data(), a_count);
      MlasConvertHalfToFloatBuffer(B, b_fp32.data(), b_count);
      if (beta != 0.0f) {
        // C needs upcast only when beta != 0 (GEMM accumulates into C).
        // When ldc == N the buffer is contiguous — use a single bulk conversion.
        // When ldc > N (3D interleaved heads), convert only the N valid columns
        // per row to avoid reading into adjacent heads' memory.
        if (ldc == N) {
          MlasConvertHalfToFloatBuffer(C, c_fp32.data(), c_count);
        } else {
          for (int row = 0; row < M; ++row) {
            MlasConvertHalfToFloatBuffer(C + row * ldc, c_fp32.data() + row * ldc, static_cast<size_t>(N));
          }
        }
      }

      math::GemmEx<float, ThreadPool>(transA, transB, M, N, K,
                                      alpha, a_fp32.data(), lda,
                                      b_fp32.data(), ldb,
                                      beta, c_fp32.data(), ldc, nullptr,
                                      mlas_backend_kernel_selector_config);

      // Downcast result back to fp16.
      // Same ldc == N check: bulk conversion when contiguous, row-by-row when
      // strided to avoid overwriting adjacent heads' output data.
      if (ldc == N) {
        MlasConvertFloatToHalfBuffer(c_fp32.data(), C, c_count);
      } else {
        for (int row = 0; row < M; ++row) {
          MlasConvertFloatToHalfBuffer(c_fp32.data() + row * ldc, C + row * ldc, static_cast<size_t>(N));
        }
      }
    }
  } else {
    ORT_THROW("Unsupported data type for attention GEMM: ",
              DataTypeImpl::ToString(DataTypeImpl::GetType<T>()));
  }
}

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : AttentionBase<T>(info) {
  is_causal_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("is_causal", 0)) == 1;
  // kv_num_heads, q_num_head are mandatory for 3D inputs but not used for 4D inputs.
  // The dimension is not yet known. If not specified, the inputs is assumed to be 4D.
  kv_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_num_heads", 0));
  q_num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("q_num_heads", 0));
  int mode = static_cast<int>(info.GetAttrOrDefault<int64_t>("qk_matmul_output_mode", 0));
  qk_matmul_output_mode_ = info.node().OutputDefs().size() >= 4 && info.node().OutputDefs()[3]->Exists()
                               ? static_cast<attention_helper::QKMatMulOutputMode>(mode)
                               : attention_helper::QKMatMulOutputMode::kNone;
  ORT_ENFORCE(qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kNone ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kQK ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kPostSoftCap ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kPostMaskBias ||
                  qk_matmul_output_mode_ == attention_helper::QKMatMulOutputMode::kPostSoftMax,
              "qk_matmul_output_mode must be -1 (absent), 0, 1, 2, or 3.");
  // The default scale depends on the input dimensions. It is set to nan to indicate that it should be computed.
  scale_ = info.GetAttrOrDefault<float>("scale", std::numeric_limits<T>::quiet_NaN());
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  softmax_precision_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("softmax_precision", 0));
  ORT_ENFORCE(scale_ > 0 || std::isnan(scale_), "scale must be greater than 0 if specified");
}

template <typename T>
Status Attention<T>::Compute(OpKernelContext* context) const {
  const Tensor* Q = context->Input<Tensor>(0);
  const Tensor* K = context->Input<Tensor>(1);
  const Tensor* V = context->Input<Tensor>(2);
  const Tensor* attn_mask = context->Input<Tensor>(3);
  const Tensor* past_key = context->Input<Tensor>(4);
  const Tensor* past_value = context->Input<Tensor>(5);
  const Tensor* nonpad_kv_seqlen = context->Input<Tensor>(6);  // optional, Opset 24

  AttentionParameters parameters;
  TensorShape y_shape;
  TensorShape present_key_shape;
  TensorShape present_value_shape;
  TensorShape output_qk_shape;

  // ComputeOutputShapeForAttention also checks the validity of the inputs.
  ORT_ENFORCE(attention_helper::ComputeOutputShapeForAttention(
                  Q,
                  K,
                  V,
                  attn_mask,
                  past_key,
                  past_value,
                  nonpad_kv_seqlen,
                  is_causal_,
                  softcap_,
                  softmax_precision_,
                  qk_matmul_output_mode_,
                  kv_num_heads_,
                  q_num_heads_,
                  scale_,
                  parameters,
                  y_shape,
                  present_key_shape,
                  present_value_shape,
                  output_qk_shape)
                  .IsOK(),
              "Output shapes for Attention could not be computed.");

  Tensor* Y = context->Output(0, y_shape);
  Tensor* present_key = context->Output(1, present_key_shape);
  Tensor* present_value = context->Output(2, present_value_shape);
  Tensor* output_qk = parameters.qk_matmul_output_mode == QKMatMulOutputMode::kNone
                          ? nullptr
                          : context->Output(3, output_qk_shape);
  return this->ApplyAttention(context,
                              Q->Data<T>(),   // Q
                              K->Data<T>(),   // K
                              V->Data<T>(),   // V
                              attn_mask,      // const Tensor* mask_index,  // mask, nullptr if no mask
                              past_key,       // past K input tensor (if not using past state)
                              past_value,     // past V input tensor (if not using past state)
                              Y,              // first output
                              present_key,    // present K output tensor (if separating present KV)
                              present_value,  // present V output tensor (if separating present KV)
                              output_qk,      // Q*K output tensor (if returning Q*K value)
                              parameters      // attention parameters
  );
}

template <typename T>
void AttentionBase<T>::ComputeAttentionProbs(T* attention_probs,                     // output buffer with size BxNxSxT
                                             const T* Q,                             // Q data. Its size is BxNxSxH
                                             const T* K,                             // k data. Its size is BxNxLxH
                                             const Tensor* mask_index,               // mask
                                             const AttentionParameters& parameters,  // attention parameters
                                             const T* past_key,                      // past key only (if not using past state)
                                             T* present_key,                         // present key only (if not using present state)
                                             T* output_qk,                           // Q*K output
                                             ThreadPool* tp,
                                             AllocatorPtr allocator) const {
  // The case past_key != nullptr and present_key == nullptr is not supported.
  // We use the fact present_key is requested to avoid any extra allocation.
  // However, if present_key is not requested, we should avoid allocated more memory than needed but that mean
  // allocating one buffer per thread. That's why the implementation is not done.
  // The user should define a model with a present_key even if not used if past_key is not null.
  ORT_ENFORCE(!((past_key != nullptr) && (present_key == nullptr)),
              "The implementation does not support past_key provided and present_key being null.");
  const size_t past_chunk_length = static_cast<size_t>(parameters.past_sequence_length) * parameters.head_size;   // P x H
  const size_t q_input_chunk_length = static_cast<size_t>(parameters.q_sequence_length) * parameters.head_size;   // S x H
  const size_t k_input_chunk_length = static_cast<size_t>(parameters.kv_sequence_length) * parameters.head_size;  // L x H
  const size_t present_chunk_length = past_chunk_length + k_input_chunk_length;                                   // T x H

  TensorOpCost unit_cost;
  const ptrdiff_t probs_matrix_size = SafeInt<ptrdiff_t>(parameters.q_sequence_length) *
                                      parameters.total_sequence_length;
  const ptrdiff_t probs_matrix_bytes = probs_matrix_size * sizeof(T);
  unit_cost.compute_cycles =
      static_cast<double>(SafeInt<ptrdiff_t>(2) * parameters.head_size * probs_matrix_size);
  unit_cost.bytes_loaded = static_cast<double>((parameters.q_sequence_length +
                                                parameters.total_sequence_length) *
                                               parameters.head_size * sizeof(T));
  unit_cost.bytes_stored = static_cast<double>(probs_matrix_bytes);

  if (present_key) {
    double bytes_to_copy_key = present_chunk_length * static_cast<double>(sizeof(T));
    unit_cost.bytes_loaded += bytes_to_copy_key;
    unit_cost.bytes_stored += bytes_to_copy_key;
  }

  // Prepare mask
  // Merge causal mask with padding mask, and convert values from 0/1 to -inf/0.
  int mask_batch_size = static_cast<int>(mask_index == nullptr || mask_index->Shape().NumDimensions() < 4
                                             ? 1
                                             : mask_index->Shape().GetDims()[0]);
  int mask_num_heads = static_cast<int>(mask_index == nullptr || mask_index->Shape().NumDimensions() < 3
                                            ? 1
                                            : (mask_index->Shape().NumDimensions() < 4
                                                   ? mask_index->Shape().GetDims()[0]
                                                   : mask_index->Shape().GetDims()[1]));

  T* mask_data = nullptr;
  bool delete_mask_data = false;
  bool causal = parameters.is_causal && parameters.q_sequence_length > 1;
  if (mask_index == nullptr) {
    // No external mask: allocate only if causal behavior needed.
    if (causal) {
      size_t mask_bytes = SafeInt<size_t>(parameters.q_sequence_length) * parameters.total_sequence_length * sizeof(T);
      void* raw = allocator->Alloc(mask_bytes);
      memset(raw, 0, mask_bytes);  // start all allowed
      mask_data = static_cast<T*>(raw);
      for (int s = 0; s < parameters.q_sequence_length; ++s) {
        for (int t = parameters.past_sequence_length + s + 1; t < parameters.total_sequence_length; ++t) {
          mask_data[s * parameters.total_sequence_length + t] = mask_filter_value<T>();
        }
      }
      delete_mask_data = true;
    }
  } else {
    const bool is_bool_mask = mask_index->IsDataType<bool>();
    const bool need_copy = is_bool_mask || causal;  // copy if we must convert or overlay causal pattern
    if (need_copy) {
      size_t mask_bytes = SafeInt<size_t>(mask_index->Shape().Size()) * sizeof(T);
      mask_data = static_cast<T*>(allocator->Alloc(mask_bytes));
      delete_mask_data = true;
      if (is_bool_mask) {
        make_copy(mask_data, mask_index->Data<bool>(), SafeInt<size_t>(mask_index->Shape().Size()));
      } else {
        make_copy(mask_data, mask_index->Data<T>(), SafeInt<size_t>(mask_index->Shape().Size()));
      }
      if (causal) {
        // Overlay causal -inf above diagonal for every broadcast slice
        int slices = mask_batch_size * mask_num_heads;
        for (int slice = 0; slice < slices; ++slice) {
          T* base = mask_data + probs_matrix_size * slice;
          for (int s = 0; s < parameters.q_sequence_length; ++s) {
            for (int t = parameters.past_sequence_length + s + 1; t < parameters.total_sequence_length; ++t) {
              base[s * parameters.total_sequence_length + t] = mask_filter_value<T>();
            }
          }
        }
      }
    } else {
      // Reuse mask memory directly (numeric, non-causal)
      mask_data = const_cast<T*>(mask_index->Data<T>());
    }
  }

  if (nullptr != present_key && parameters.kv_num_heads != parameters.q_num_heads) {
    // This is not part of the main loop because it is not needed at every iteration and
    // we cannot ensure the inner body is executed first before getting used in another iteration.
    // parameters.batch_size * parameters.q_num_heads
    for (std::ptrdiff_t batch_i = 0; batch_i < parameters.batch_size; ++batch_i) {
      for (std::ptrdiff_t head_i = 0; head_i < parameters.kv_num_heads; ++head_i) {
        ConcatStateChunk(past_key, K, present_key,
                         past_chunk_length, k_input_chunk_length, present_chunk_length,
                         parameters.kv_num_heads, parameters.head_size, batch_i, head_i,
                         parameters.transpose_output);
      }
    }
  }

  // If present_key is not null, it is already initialized to zero.
  // Main loop
  // With 3D inputs, both Q and K are transposed with permutations (0, 2, 1, 3).
  // To avoid expressing the transposition, we use GemmEx with different values for lda, ldb.
  // If past_key is not null, then we need to concatenate it with K, the concatenation is not transposed.
  const int loop_len = parameters.batch_size * parameters.q_num_heads;
  const float alpha = parameters.scale;
  bool transposed_k = parameters.transpose_output && nullptr == present_key;

  ThreadPool::TryParallelFor(tp, loop_len, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    for (std::ptrdiff_t i = begin; i != end; ++i) {
      const ptrdiff_t output_offset = SafeInt<ptrdiff_t>(i) * probs_matrix_size;
      std::ptrdiff_t batch_i = i / parameters.q_num_heads;
      std::ptrdiff_t head_i = i % parameters.q_num_heads;
      const ptrdiff_t mask_data_offset = probs_matrix_size *
                                         (head_i % mask_num_heads + (batch_i % mask_batch_size) * mask_num_heads);

      T* output = attention_probs + output_offset;
      T* out_qk = output_qk == nullptr ? nullptr : output_qk + output_offset;

      // handling GQA
      std::ptrdiff_t head_ki = head_i * parameters.kv_num_heads / parameters.q_num_heads;
      std::ptrdiff_t ki = batch_i * parameters.kv_num_heads + head_ki;
      const T* k = K + k_input_chunk_length * ki;

      if (nullptr != present_key) {
        if (parameters.kv_num_heads != parameters.q_num_heads) {
          // Already done in a loop before this one.
          k = present_key + ki * present_chunk_length;
        } else {
          k = ConcatStateChunk(past_key, K, present_key,
                               past_chunk_length, k_input_chunk_length, present_chunk_length,
                               parameters.kv_num_heads, parameters.head_size, batch_i, head_i,
                               parameters.transpose_output);
        }
      }

      // Compute Q*K' + AttentionMask
      //
      // ONNX Attention v23/24 (per onnx/onnx#7867 + #7913) requires the mask/bias
      // to be applied AFTER softcap, otherwise -inf mask values get squashed by
      // tanh into -c, leaking probability through softmax onto masked positions.
      //
      // When softcap is disabled, mask add commutes with the (no-op) softcap, so we
      // follow the original code path verbatim:  fold the mask add into the GEMM as
      // `beta = 1` (preload mask, accumulate via FMA).  This preserves the FMA-fused
      // numerics that pre-spec-fix tests were calibrated against.
      // When softcap is active, we run GEMM with `beta = 0`, apply softcap inplace,
      // then add the mask explicitly via AddInPlace.
      //
      // The fold is also skipped when the caller wants a kQK / kPostSoftCap snapshot,
      // so the snapshot reflects the raw / post-softcap QK without the mask folded in.
      //
      //                     original                 transposed             each iteration
      // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
      // B: K'               (B x N x) T x H          (B x N x) H x T        H x T
      // C: attention_probs  (B x N x) S x T          (B x N x) S x T        S x T
      const bool softcap_active = (parameters.softcap > 0.0f);
      const bool snapshot_needs_pre_mask =
          out_qk != nullptr &&
          (parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQK ||
           parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kPostSoftCap);
      const bool fold_mask_into_gemm =
          (mask_data != nullptr) && !softcap_active && !snapshot_needs_pre_mask;
      float beta;
      if (fold_mask_into_gemm) {
        // Broadcast mask data: SxT -> SxT
        memcpy(output, mask_data + mask_data_offset, probs_matrix_bytes);
        beta = 1;
      } else {
        beta = 0;
      }

      const T* q_ptr = parameters.transpose_output
                           ? Q + q_input_chunk_length * parameters.q_num_heads * batch_i + head_i * parameters.head_size
                           : Q + q_input_chunk_length * i;
      int q_lda = parameters.transpose_output
                      ? parameters.head_size * parameters.q_num_heads
                      : parameters.head_size;
      const T* k_ptr = transposed_k
                           ? K + k_input_chunk_length * parameters.kv_num_heads * batch_i + head_ki * parameters.head_size
                           : k;
      int k_ldb = transposed_k
                      ? parameters.head_size * parameters.kv_num_heads
                      : parameters.head_size;

      AttentionGemm(CblasNoTrans, CblasTrans,
                    parameters.q_sequence_length, parameters.total_sequence_length, parameters.head_size,
                    alpha, q_ptr, q_lda, k_ptr, k_ldb, beta, output, parameters.total_sequence_length,
                    &mlas_backend_kernel_selector_config_);

      // Snapshot kQK (raw scale*Q*K^T): only reachable when fold path was skipped.
      if (out_qk != nullptr &&
          parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kQK) {
        memcpy(out_qk, output, SafeInt<size_t>(probs_matrix_size) * sizeof(T));
      }

      if (softcap_active) {
        // Softcap path (mask was NOT folded into GEMM since beta=0 above).
        if constexpr (std::is_same<T, float>::value) {
          ComputeAttentionSoftcapInplace(output, static_cast<int>(probs_matrix_size), parameters.softcap);
        } else if constexpr (std::is_same<T, MLFloat16>::value) {
          ComputeAttentionSoftcapInplace(output, static_cast<int>(probs_matrix_size), MLFloat16(parameters.softcap));
        } else {
          ORT_THROW("Unsupported data type for ComputeAttentionSoftcapInplace: ",
                    DataTypeImpl::ToString(DataTypeImpl::GetType<T>()));
        }
      }

      // Snapshot kPostSoftCap (post-softcap, pre-mask/bias).  When softcap is disabled
      // this equals raw scale*Q*K^T (kQK).  Reachable only when fold was skipped above.
      if (out_qk != nullptr &&
          parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kPostSoftCap) {
        memcpy(out_qk, output, SafeInt<size_t>(probs_matrix_size) * sizeof(T));
      }

      // Add mask explicitly when it wasn't folded into GEMM (single source of truth:
      // a non-zero `beta` is exactly the case where the mask was preloaded into C).
      if (mask_data != nullptr && !fold_mask_into_gemm) {
        AddInPlace(output, mask_data + mask_data_offset, probs_matrix_size);
      }

      // Apply nonpad_kv_seqlen masking (Opset 24+): mask out KV positions >= valid length per batch.
      // Done AFTER softcap+mask so the masked positions hold the `mask_filter_value<T>()` sentinel
      // (`std::numeric_limits<T>::lowest()` for floats, `MLFloat16::MinValue` for fp16 — see
      // `onnxruntime/core/providers/cpu/llm/attention.h`). The CPU softmax uses this finite sentinel
      // (not IEEE -inf) because MLAS' softmax kernel expects only finite inputs; the value is small
      // enough relative to any softcap-saturated score that the corresponding softmax weight is 0.
      if (parameters.has_nonpad_kv_seqlen) {
        int valid_kv_len = static_cast<int>(parameters.nonpad_kv_seqlen_data[batch_i]);
        for (int s = 0; s < parameters.q_sequence_length; ++s) {
          std::fill(output + s * parameters.total_sequence_length + valid_kv_len,
                    output + (s + 1) * parameters.total_sequence_length,
                    mask_filter_value<T>());
        }
      }

      // Snapshot kPostMaskBias (post-mask/bias, pre-softmax).
      if (out_qk != nullptr &&
          parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kPostMaskBias) {
        memcpy(out_qk, output, SafeInt<size_t>(probs_matrix_size) * sizeof(T));
      }

      ComputeAttentionSoftmaxInplace(output, parameters.q_sequence_length, parameters.total_sequence_length, nullptr, allocator);

      // Snapshot kPostSoftMax (post-softmax).
      if (out_qk != nullptr &&
          parameters.qk_matmul_output_mode == attention_helper::QKMatMulOutputMode::kPostSoftMax) {
        memcpy(out_qk, output, SafeInt<size_t>(probs_matrix_size) * sizeof(T));
      }
    }
  });
  if (delete_mask_data) {
    allocator->Free(mask_data);
  }
}

template <typename T>
T* AttentionBase<T>::ConcatStateChunk(const T* past,
                                      const T* base_chunk,  // chunk is K or V, it can be transposed or not
                                      T* present,
                                      size_t past_chunk_length,
                                      size_t input_chunk_length,  // chunk length of K or V
                                      size_t present_chunk_length,
                                      size_t num_heads,
                                      size_t head_size,
                                      std::ptrdiff_t batch_i,
                                      std::ptrdiff_t head_i,
                                      bool transposed) const {
  std::ptrdiff_t i = batch_i * num_heads + head_i % num_heads;

  T* start = present + i * present_chunk_length;

  T* p = start;
  if (nullptr != past) {
    const T* src_past = past + i * past_chunk_length;
    memcpy(p, src_past, past_chunk_length * sizeof(T));
    p += past_chunk_length;
  }

  if (transposed) {
    ORT_ENFORCE(head_size > 0 && num_heads > 0 && batch_i >= 0 && head_i >= 0,
                "Invalid parameters for ConcatStateChunk: head_size=", head_size, ", batch_i=", batch_i, ", head_i=", head_i);
    size_t sequence_length = SafeInt<size_t>(input_chunk_length / head_size);
    const T* chunk = base_chunk + head_i * head_size + input_chunk_length * num_heads * batch_i;
    for (size_t j = 0; j < sequence_length; ++j) {
      memcpy(p, chunk, head_size * sizeof(T));
      p += head_size;
      chunk += num_heads * head_size;
    }
  } else {
    const T* chunk = base_chunk + input_chunk_length * i;
    memcpy(p, chunk, (present_chunk_length - past_chunk_length) * sizeof(T));
  }
  return start;
}

template <typename T>
void AttentionBase<T>::ComputeVxAttentionScore(T* output,                  // buffer for the result with size BxSxNxH_v
                                               const T* attention_probs,   // Attention probs with size BxNxSxT
                                               const T* V,                 // V value with size BxNxLxH_v
                                               int batch_size,             // batch size
                                               int sequence_length,        // sequence length
                                               int kv_sequence_length,     // sequence length of K or V
                                               int past_sequence_length,   // sequence length in past state
                                               int total_sequence_length,  // total sequence length = past_sequence_length + kv_sequence_length
                                               int v_head_size,            // head size of V (H_v)
                                               int num_heads,              // number of attention heads
                                               int kv_num_heads,           // number of KV heads
                                               const T* past_value,        // past value only (if not using past state)
                                               T* present_value,           // present value only (if not using present state)
                                               bool transpose_output,      // whether to transpose the output (0, 2, 1, 3)
                                               ThreadPool* tp) const {
  ORT_ENFORCE(!((past_value != nullptr) && (present_value == nullptr)),
              "The implementation does not support past_value provided and present_value being null.");
  const ptrdiff_t past_chunk_length = SafeInt<ptrdiff_t>(past_sequence_length) * v_head_size;   // P x H_v
  const ptrdiff_t v_input_chunk_length = SafeInt<ptrdiff_t>(kv_sequence_length) * v_head_size;  // L x H_v
  const ptrdiff_t present_chunk_length = past_chunk_length + v_input_chunk_length;              // T x H_v

  // The cost of Gemm
  TensorOpCost unit_cost;
  unit_cost.compute_cycles =
      static_cast<double>(SafeInt<ptrdiff_t>(2) * sequence_length * v_head_size * total_sequence_length);
  unit_cost.bytes_loaded =
      static_cast<double>(SafeInt<ptrdiff_t>(sequence_length + v_head_size) * total_sequence_length * sizeof(T));
  unit_cost.bytes_stored = static_cast<double>(sequence_length * v_head_size * sizeof(T));

  const size_t bytes_to_copy_trans = SafeInt<size_t>(v_head_size) * sizeof(T);
  double bytes_to_copy_trans_all = static_cast<double>(sequence_length * bytes_to_copy_trans);
  unit_cost.bytes_loaded += bytes_to_copy_trans_all;
  unit_cost.bytes_stored += bytes_to_copy_trans_all;

  bool transposed_v = transpose_output && nullptr == present_value;
  if (nullptr != present_value && kv_num_heads != num_heads) {
    // This is not part of the main loop because it is not needed at every iteration and
    // we cannot ensure the inner body is executed first before getting used in another iteration.
    // parameters.batch_size * parameters.q_num_heads
    for (std::ptrdiff_t batch_i = 0; batch_i < batch_size; ++batch_i) {
      for (std::ptrdiff_t head_i = 0; head_i < kv_num_heads; ++head_i) {
        ConcatStateChunk(past_value, V, present_value,
                         past_chunk_length, v_input_chunk_length, present_chunk_length,
                         kv_num_heads, v_head_size, batch_i, head_i,
                         transpose_output);
      }
    }
  }

  ThreadPool::TryParallelFor(
      tp, SafeInt<ptrdiff_t>(batch_size) * num_heads, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t i = begin; i != end; ++i) {
          // handling GQA
          std::ptrdiff_t batch_i = i / num_heads;
          std::ptrdiff_t head_i = i % num_heads;
          std::ptrdiff_t head_vi = head_i * kv_num_heads / num_heads;
          std::ptrdiff_t vi = batch_i * kv_num_heads + head_vi;
          const T* v = V + v_input_chunk_length * vi;

          if (nullptr != present_value) {
            if (kv_num_heads != num_heads) {
              // Already done in a loop before this one.
              v = present_value + vi * present_chunk_length;
            } else {
              // transposed_v is false here.
              v = ConcatStateChunk(past_value, V, present_value,
                                   past_chunk_length, v_input_chunk_length, present_chunk_length,
                                   kv_num_heads, v_head_size, batch_i, head_i,
                                   transpose_output);
            }
          }

          // Compute QK * V
          ptrdiff_t attention_probs_offset = SafeInt<ptrdiff_t>(sequence_length) * total_sequence_length * i;
          const T* gemm_B;
          int gemm_ldb;
          T* gemm_C;
          int gemm_ldc;

          if (transpose_output) {
            // 3D inputs: V may be in strided layout, use appropriate strides.
            gemm_B = transposed_v ? V + head_vi * v_head_size + v_input_chunk_length * kv_num_heads * batch_i : v;
            gemm_ldb = transposed_v ? v_head_size * kv_num_heads : v_head_size;
            gemm_C = output + ((batch_i * sequence_length * num_heads + head_i) * v_head_size);
            gemm_ldc = v_head_size * num_heads;
          } else {
            // 4D inputs: V is already in head-contiguous layout.
            gemm_B = v;
            gemm_ldb = v_head_size;
            ptrdiff_t dest_offset = SafeInt<ptrdiff_t>(sequence_length) * v_head_size * i;
            gemm_C = output + dest_offset;
            gemm_ldc = v_head_size;
          }

          AttentionGemm(CblasNoTrans, CblasNoTrans,
                        sequence_length, v_head_size, total_sequence_length,
                        1.0f, attention_probs + attention_probs_offset, total_sequence_length,
                        gemm_B, gemm_ldb, 0.0f, gemm_C, gemm_ldc,
                        &mlas_backend_kernel_selector_config_);
        }
      });
}

// ---------------------------------------------------------------------------
// FlashAttention-style tiled CPU path
// ---------------------------------------------------------------------------
//
// Algorithm (per task = one (batch, q_head, q_block) triple):
//
//   Initialize m[q_block] = -inf, l[q_block] = 0, O[q_block, v_head] = 0
//   For kv_block in 0..total_seq by kv_block_size:
//     S = scale * Q_block @ K_block^T
//     S = softcap(S)           (optional)
//     S += attn_mask_tile      (optional)
//     S[nonpad positions] = -inf
//     S[causal-masked positions] = -inf
//     For each q row i:
//       m_new = max(m[i], rowmax(S[i,:]))
//       S[i,:] = exp(S[i,:] - m_new)
//       O[i,:] *= exp(m[i] - m_new)
//       l[i]   = exp(m[i] - m_new) * l[i] + rowsum(S[i,:])
//       m[i]   = m_new
//     O += S @ V_block
//   O /= l[:]
//
// All intermediate arithmetic is in float32 regardless of input type.
// For MLFloat16 inputs the Q/K/V tiles are upcasted per kv_block; the
// accumulated float32 output is downcasted at the end.
//
// Parallelism is over batch × q_heads × q_chunks.  Each worker uses a
// dedicated scratch slot (one per pool thread + one for the calling thread)
// to hold m, l, S_tile, O_tile, and (for fp16) float32 copies of Q/K/V tiles.
//
// Enabled at runtime by setting the environment variable:
//   ORT_ATTENTION_USE_FLASH=1
//
// qk_matmul_output_mode snapshots are NOT supported on this path; the caller
// falls back to the standard materialising path in that case.

template <typename T>
Status AttentionBase<T>::ApplyFlashAttention(
    OpKernelContext* context,
    const T* Q,
    const T* K,
    const T* V,
    const Tensor* mask_index,
    const T* past_key,
    const T* past_value,
    Tensor* output,
    Tensor* present_key,
    Tensor* present_value,
    const AttentionParameters& parameters,
    AllocatorPtr allocator,
    ThreadPool* tp) const {
  ORT_ENFORCE(!((past_key != nullptr) && (present_key == nullptr)),
              "ApplyFlashAttention: past_key without present_key is unsupported.");
  ORT_ENFORCE(!((past_value != nullptr) && (present_value == nullptr)),
              "ApplyFlashAttention: past_value without present_value is unsupported.");

  // ---- Constants / shorthand ----
  const int batch_size = parameters.batch_size;
  const int q_num_heads = parameters.q_num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;
  const int q_seq_len = parameters.q_sequence_length;
  const int kv_seq_len = parameters.kv_sequence_length;
  const int past_seq_len = parameters.past_sequence_length;
  const int total_seq_len = parameters.total_sequence_length;
  const float scale = parameters.scale;
  const bool causal = parameters.is_causal && q_seq_len > 1;
  const bool softcap_active = (parameters.softcap > 0.0f);
  const float softcap = parameters.softcap;
  constexpr bool is_fp16 = std::is_same<T, MLFloat16>::value;

  // ---- Tile-size computation (same formula as contrib MultiHeadAttention) ----
  int l2_cache_size = Env::Default().GetL2CacheSize();
  if (l2_cache_size <= 0) {
    l2_cache_size = 256 * 1024;  // conservative fallback: 256 KB
  }
  int kv_block_size = l2_cache_size / (static_cast<int>(sizeof(float)) * 4 * (head_size + v_head_size));
  kv_block_size = std::max(kv_block_size, 1);
  kv_block_size = std::min(kv_block_size, total_seq_len);
  int q_block_size = std::min(kv_block_size, head_size + v_head_size);
  q_block_size = std::max(q_block_size, 1);
  q_block_size = std::min(q_block_size, q_seq_len);

  // ---- Prepare attention mask (identical logic to ComputeAttentionProbs) ----
  const ptrdiff_t probs_matrix_size = SafeInt<ptrdiff_t>(q_seq_len) * total_seq_len;

  int mask_batch_size = static_cast<int>(mask_index == nullptr || mask_index->Shape().NumDimensions() < 4
                                             ? 1
                                             : mask_index->Shape().GetDims()[0]);
  int mask_num_heads = static_cast<int>(mask_index == nullptr || mask_index->Shape().NumDimensions() < 3
                                            ? 1
                                            : (mask_index->Shape().NumDimensions() < 4
                                                   ? mask_index->Shape().GetDims()[0]
                                                   : mask_index->Shape().GetDims()[1]));

  T* mask_data = nullptr;
  bool delete_mask_data = false;
  if (mask_index == nullptr) {
    if (causal) {
      size_t mask_bytes = SafeInt<size_t>(q_seq_len) * total_seq_len * sizeof(T);
      void* raw = allocator->Alloc(mask_bytes);
      memset(raw, 0, mask_bytes);
      mask_data = static_cast<T*>(raw);
      for (int s = 0; s < q_seq_len; ++s) {
        for (int t = past_seq_len + s + 1; t < total_seq_len; ++t) {
          mask_data[s * total_seq_len + t] = mask_filter_value<T>();
        }
      }
      delete_mask_data = true;
    }
  } else {
    const bool is_bool_mask = mask_index->IsDataType<bool>();
    const bool need_copy = is_bool_mask || causal;
    if (need_copy) {
      size_t mask_bytes = SafeInt<size_t>(mask_index->Shape().Size()) * sizeof(T);
      mask_data = static_cast<T*>(allocator->Alloc(mask_bytes));
      delete_mask_data = true;
      if (is_bool_mask) {
        make_copy(mask_data, mask_index->Data<bool>(), SafeInt<size_t>(mask_index->Shape().Size()));
      } else {
        make_copy(mask_data, mask_index->Data<T>(), SafeInt<size_t>(mask_index->Shape().Size()));
      }
      if (causal) {
        int slices = mask_batch_size * mask_num_heads;
        for (int slice = 0; slice < slices; ++slice) {
          T* base = mask_data + probs_matrix_size * slice;
          for (int s = 0; s < q_seq_len; ++s) {
            for (int t = past_seq_len + s + 1; t < total_seq_len; ++t) {
              base[s * total_seq_len + t] = mask_filter_value<T>();
            }
          }
        }
      }
    } else {
      mask_data = const_cast<T*>(mask_index->Data<T>());
    }
  }

  // ---- Materialise present_key / present_value (pre-build before parallel loop) ----
  // For the flash path we always materialise up-front when requested so the
  // inner tiled loop can address K/V as a simple contiguous array.
  const size_t past_k_chunk = static_cast<size_t>(past_seq_len) * head_size;
  const size_t k_input_chunk = static_cast<size_t>(kv_seq_len) * head_size;
  const size_t present_k_chunk = past_k_chunk + k_input_chunk;
  const size_t past_v_chunk = static_cast<size_t>(past_seq_len) * v_head_size;
  const size_t v_input_chunk = static_cast<size_t>(kv_seq_len) * v_head_size;
  const size_t present_v_chunk = past_v_chunk + v_input_chunk;

  T* present_key_data = present_key != nullptr ? present_key->MutableData<T>() : nullptr;
  T* present_value_data = present_value != nullptr ? present_value->MutableData<T>() : nullptr;

  if (present_key_data != nullptr) {
    for (std::ptrdiff_t bi = 0; bi < batch_size; ++bi) {
      for (std::ptrdiff_t hi = 0; hi < kv_num_heads; ++hi) {
        ConcatStateChunk(past_key, K, present_key_data,
                         past_k_chunk, k_input_chunk, present_k_chunk,
                         kv_num_heads, head_size, bi, hi, parameters.transpose_output);
      }
    }
  }
  if (present_value_data != nullptr) {
    for (std::ptrdiff_t bi = 0; bi < batch_size; ++bi) {
      for (std::ptrdiff_t hi = 0; hi < kv_num_heads; ++hi) {
        ConcatStateChunk(past_value, V, present_value_data,
                         past_v_chunk, v_input_chunk, present_v_chunk,
                         kv_num_heads, v_head_size, bi, hi, parameters.transpose_output);
      }
    }
  }

  // ---- Effective K / V pointers and their layout ----
  // After materialising present_key/value the storage is always in
  // non-transposed (BNSH-4D) order:  [batch, kv_head, total_seq, head_size].
  // When there is no present tensor we fall back to the original K/V pointer
  // and must respect the 3D-transposed layout if parameters.transpose_output.
  const T* K_eff = (present_key_data != nullptr) ? present_key_data : K;
  const T* V_eff = (present_value_data != nullptr) ? present_value_data : V;
  const int kv_seq_eff = (present_key_data != nullptr) ? total_seq_len : kv_seq_len;
  const bool kv_transposed = parameters.transpose_output && (present_key_data == nullptr);

  // ---- Scratch buffer size per parallel range ----
  // Scratch layout (all float32):
  //   m[q_block_size]                      running max per q-row
  //   l[q_block_size]                      running normalisation per q-row
  //   S[q_block_size * kv_block_size]      score tile
  //   O[q_block_size * v_head_size]        accumulated output tile
  //   q_fp32[q_block_size * head_size]     (fp16 only) upcast Q tile
  //   k_fp32[kv_block_size * head_size]    (fp16 only) upcast K tile
  //   v_fp32[kv_block_size * v_head_size]  (fp16 only) upcast V tile

  const size_t ml_floats = static_cast<size_t>(q_block_size) * 2;
  const size_t s_floats = static_cast<size_t>(q_block_size) * kv_block_size;
  const size_t o_floats = static_cast<size_t>(q_block_size) * v_head_size;
  const size_t fp16_extra = is_fp16
                                ? (static_cast<size_t>(q_block_size) * head_size +
                                   static_cast<size_t>(kv_block_size) * head_size +
                                   static_cast<size_t>(kv_block_size) * v_head_size)
                                : 0;
  const size_t floats_per_range = ml_floats + s_floats + o_floats + fp16_extra;

  // ---- Parallel loop over (batch, q_head, q_chunk) ----
  const int q_chunks = (q_seq_len + q_block_size - 1) / q_block_size;
  const int total_tasks = batch_size * q_num_heads * q_chunks;

  TensorOpCost unit_cost;
  unit_cost.compute_cycles = static_cast<double>(
      2 * head_size * q_block_size * total_seq_len +
      2 * total_seq_len * v_head_size * q_block_size);
  unit_cost.bytes_loaded =
      static_cast<double>((static_cast<size_t>(head_size) + v_head_size) * total_seq_len * sizeof(T));
  unit_cost.bytes_stored =
      static_cast<double>(static_cast<size_t>(q_block_size) * v_head_size * sizeof(T));

  T* output_data = output->MutableData<T>();

  ThreadPool::TryParallelFor(tp, total_tasks, unit_cost, [&](std::ptrdiff_t task_begin, std::ptrdiff_t task_end) {
    // Each parallel range allocates its own scratch on the stack.  floats_per_range
    // is at most a few tens of KB for typical head sizes so this is safe.
    std::vector<float> local_scratch(floats_per_range);
    float* slot_base = local_scratch.data();

    float* m_buf = slot_base;
    float* l_buf = m_buf + q_block_size;
    float* s_tile = l_buf + q_block_size;
    float* o_tile = s_tile + s_floats;
    // fp16-only upcast buffers (nullptr-equivalent for float; never dereferenced)
    float* q_fp32_buf = is_fp16 ? o_tile + o_floats : nullptr;
    float* k_fp32_buf = is_fp16 ? q_fp32_buf + static_cast<size_t>(q_block_size) * head_size : nullptr;
    float* v_fp32_buf = is_fp16 ? k_fp32_buf + static_cast<size_t>(kv_block_size) * head_size : nullptr;

    for (std::ptrdiff_t task = task_begin; task < task_end; ++task) {
      // Decode (batch_i, head_i, q_chunk_i) from flat task index.
      std::ptrdiff_t tmp = task;
      const std::ptrdiff_t q_chunk_i = tmp % q_chunks;
      tmp /= q_chunks;
      const std::ptrdiff_t head_i = tmp % q_num_heads;
      const std::ptrdiff_t batch_i = tmp / q_num_heads;

      const int q_block_start = static_cast<int>(q_chunk_i) * q_block_size;
      const int actual_q_rows = std::min(q_block_size, q_seq_len - q_block_start);

      // GQA: map q_head → kv_head
      const std::ptrdiff_t kv_head_i = head_i * kv_num_heads / q_num_heads;

      // ---- Q tile pointer and row-stride ----
      const T* q_tile_ptr;
      int q_lda;
      if (parameters.transpose_output) {
        // 3D layout [B, S, N, H]: Q[batch_i, q_block_start, head_i, :]
        q_tile_ptr = Q + batch_i * q_seq_len * q_num_heads * head_size +
                     q_block_start * q_num_heads * head_size +
                     head_i * head_size;
        q_lda = q_num_heads * head_size;
      } else {
        // 4D layout [B, N, S, H]: Q[batch_i, head_i, q_block_start, :]
        q_tile_ptr = Q + (batch_i * q_num_heads + head_i) * q_seq_len * head_size +
                     q_block_start * head_size;
        q_lda = head_size;
      }

      // For fp16, upcast the Q tile into a contiguous fp32 buffer once per q_block.
      const float* q_gemm_ptr;
      int q_gemm_lda;
      if constexpr (is_fp16) {
        for (int qr = 0; qr < actual_q_rows; ++qr) {
          MlasConvertHalfToFloatBuffer(q_tile_ptr + qr * q_lda,
                                       q_fp32_buf + qr * head_size,
                                       static_cast<size_t>(head_size));
        }
        q_gemm_ptr = q_fp32_buf;
        q_gemm_lda = head_size;
      } else {
        q_gemm_ptr = reinterpret_cast<const float*>(q_tile_ptr);
        q_gemm_lda = q_lda;
      }

      // ---- Mask slice offset for this (batch, head) ----
      const ptrdiff_t mask_data_offset =
          (mask_data != nullptr)
              ? probs_matrix_size * (head_i % mask_num_heads +
                                     (batch_i % mask_batch_size) * mask_num_heads)
              : 0;

      // Nonpad KV length for this batch item
      const int valid_kv_len =
          parameters.has_nonpad_kv_seqlen
              ? static_cast<int>(parameters.nonpad_kv_seqlen_data[batch_i])
              : total_seq_len;

      // ---- Initialise m, l, O ----
      for (int r = 0; r < actual_q_rows; ++r) {
        m_buf[r] = std::numeric_limits<float>::lowest();
        l_buf[r] = 0.0f;
      }
      // Zero O_tile (we always use beta=1 in the SGEMM below; first pass works
      // because O starts at 0 and exp(lowest - new_m) ≈ 0 zeroes the scale).
      std::fill(o_tile, o_tile + static_cast<size_t>(actual_q_rows) * v_head_size, 0.0f);

      // ---- Tiled KV loop ----
      for (int kv_start = 0; kv_start < total_seq_len; kv_start += kv_block_size) {
        const int actual_kv_rows = std::min(kv_block_size, total_seq_len - kv_start);

        // ---- K tile pointer and row-stride ----
        const T* k_tile_ptr;
        int k_ldb;
        if (kv_transposed) {
          // 3D layout [B, S, N, H]
          k_tile_ptr = K_eff + batch_i * kv_seq_eff * kv_num_heads * head_size +
                       kv_start * kv_num_heads * head_size +
                       kv_head_i * head_size;
          k_ldb = kv_num_heads * head_size;
        } else {
          // 4D layout [B, N, S, H]
          k_tile_ptr = K_eff + (batch_i * kv_num_heads + kv_head_i) * kv_seq_eff * head_size +
                       kv_start * head_size;
          k_ldb = head_size;
        }

        // ---- V tile pointer and row-stride ----
        const T* v_tile_ptr;
        int v_ldb;
        if (kv_transposed) {
          v_tile_ptr = V_eff + batch_i * kv_seq_eff * kv_num_heads * v_head_size +
                       kv_start * kv_num_heads * v_head_size +
                       kv_head_i * v_head_size;
          v_ldb = kv_num_heads * v_head_size;
        } else {
          v_tile_ptr = V_eff + (batch_i * kv_num_heads + kv_head_i) * kv_seq_eff * v_head_size +
                       kv_start * v_head_size;
          v_ldb = v_head_size;
        }

        // For fp16: upcast K and V tiles into contiguous fp32 buffers.
        const float* k_gemm_ptr;
        int k_gemm_ldb;
        const float* v_gemm_ptr;
        int v_gemm_ldb;
        if constexpr (is_fp16) {
          for (int kr = 0; kr < actual_kv_rows; ++kr) {
            MlasConvertHalfToFloatBuffer(k_tile_ptr + kr * k_ldb,
                                         k_fp32_buf + kr * head_size,
                                         static_cast<size_t>(head_size));
            MlasConvertHalfToFloatBuffer(v_tile_ptr + kr * v_ldb,
                                         v_fp32_buf + kr * v_head_size,
                                         static_cast<size_t>(v_head_size));
          }
          k_gemm_ptr = k_fp32_buf;
          k_gemm_ldb = head_size;
          v_gemm_ptr = v_fp32_buf;
          v_gemm_ldb = v_head_size;
        } else {
          k_gemm_ptr = reinterpret_cast<const float*>(k_tile_ptr);
          k_gemm_ldb = k_ldb;
          v_gemm_ptr = reinterpret_cast<const float*>(v_tile_ptr);
          v_gemm_ldb = v_ldb;
        }

        // ---- S_tile = scale * Q_block @ K_block^T ----
        // Shape: [actual_q_rows, actual_kv_rows], stored in s_tile with stride kv_block_size.
        MlasGemm(CblasNoTrans, CblasTrans,
                 static_cast<size_t>(actual_q_rows),
                 static_cast<size_t>(actual_kv_rows),
                 static_cast<size_t>(head_size),
                 scale, q_gemm_ptr, static_cast<size_t>(q_gemm_lda),
                 k_gemm_ptr, static_cast<size_t>(k_gemm_ldb),
                 0.0f, s_tile, static_cast<size_t>(kv_block_size),
                 nullptr, nullptr);

        // ---- Softcap (optional) ----
        if (softcap_active) {
          for (int r = 0; r < actual_q_rows; ++r) {
            float* row = s_tile + r * kv_block_size;
            for (int c = 0; c < actual_kv_rows; ++c) {
              row[c] = std::tanh(row[c] / softcap) * softcap;
            }
          }
        }

        // ---- External attention mask tile ----
        if (mask_data != nullptr) {
          for (int r = 0; r < actual_q_rows; ++r) {
            const T* mask_row = mask_data + mask_data_offset +
                                (q_block_start + r) * total_seq_len + kv_start;
            float* s_row = s_tile + r * kv_block_size;
            if constexpr (is_fp16) {
              for (int c = 0; c < actual_kv_rows; ++c) {
                s_row[c] += mask_row[c].ToFloat();
              }
            } else {
              for (int c = 0; c < actual_kv_rows; ++c) {
                s_row[c] += static_cast<float>(mask_row[c]);
              }
            }
          }
        }

        // ---- Nonpad masking ----
        if (valid_kv_len < total_seq_len) {
          for (int r = 0; r < actual_q_rows; ++r) {
            float* s_row = s_tile + r * kv_block_size;
            for (int c = 0; c < actual_kv_rows; ++c) {
              if (kv_start + c >= valid_kv_len) {
                s_row[c] = std::numeric_limits<float>::lowest();
              }
            }
          }
        }

        // ---- Causal masking (when causal==true, is_causal&&q_seq>1) ----
        // A position (q_block_start+r, kv_start+c) is causally masked when
        //   kv_start + c > past_seq_len + q_block_start + r
        if (causal) {
          for (int r = 0; r < actual_q_rows; ++r) {
            float* s_row = s_tile + r * kv_block_size;
            const int threshold = past_seq_len + q_block_start + r;
            for (int c = 0; c < actual_kv_rows; ++c) {
              if (kv_start + c > threshold) {
                s_row[c] = std::numeric_limits<float>::lowest();
              }
            }
          }
        }

        // ---- Online softmax update + O rescaling ----
        for (int r = 0; r < actual_q_rows; ++r) {
          float* s_row = s_tile + r * kv_block_size;
          float* o_row = o_tile + r * v_head_size;

          // Row max of this tile
          float row_max = s_row[0];
          for (int c = 1; c < actual_kv_rows; ++c) {
            if (s_row[c] > row_max) row_max = s_row[c];
          }

          const float new_m = std::max(m_buf[r], row_max);
          const float exp_diff = std::exp(m_buf[r] - new_m);

          // Rescale accumulated output for the change in running max
          if (exp_diff != 1.0f) {
            for (int c = 0; c < v_head_size; ++c) {
              o_row[c] *= exp_diff;
            }
          }

          // exp-normalise the score row and accumulate into l
          float rowsum = 0.0f;
          const float neg_new_m = -new_m;
          for (int c = 0; c < actual_kv_rows; ++c) {
            float e = std::exp(s_row[c] + neg_new_m);
            s_row[c] = e;
            rowsum += e;
          }

          l_buf[r] = exp_diff * l_buf[r] + rowsum;
          m_buf[r] = new_m;
        }

        // ---- O += S_norm @ V_block (SGEMM, single-threaded) ----
        MlasGemm(CblasNoTrans, CblasNoTrans,
                 static_cast<size_t>(actual_q_rows),
                 static_cast<size_t>(v_head_size),
                 static_cast<size_t>(actual_kv_rows),
                 1.0f, s_tile, static_cast<size_t>(kv_block_size),
                 v_gemm_ptr, static_cast<size_t>(v_gemm_ldb),
                 1.0f, o_tile, static_cast<size_t>(v_head_size),
                 nullptr, nullptr);
      }  // kv_block loop

      // ---- Normalise O and write to output ----
      for (int r = 0; r < actual_q_rows; ++r) {
        float* o_row = o_tile + r * v_head_size;
        const float inv_l = (l_buf[r] > 0.0f) ? 1.0f / l_buf[r] : 0.0f;
        for (int c = 0; c < v_head_size; ++c) {
          o_row[c] *= inv_l;
        }

        // Write to output tensor with the correct layout
        T* dst;
        if (parameters.transpose_output) {
          // 3D output [B, S, N, H_v]
          dst = output_data +
                batch_i * q_seq_len * q_num_heads * v_head_size +
                (q_block_start + r) * q_num_heads * v_head_size +
                head_i * v_head_size;
        } else {
          // 4D output [B, N, S, H_v]
          dst = output_data +
                (batch_i * q_num_heads + head_i) * q_seq_len * v_head_size +
                (q_block_start + r) * v_head_size;
        }

        if constexpr (is_fp16) {
          MlasConvertFloatToHalfBuffer(o_row, dst, static_cast<size_t>(v_head_size));
        } else {
          memcpy(dst, o_row, static_cast<size_t>(v_head_size) * sizeof(float));
        }
      }
    }  // task loop
  });  // TryParallelFor

  if (delete_mask_data) {
    allocator->Free(mask_data);
  }

  return Status::OK();
}

template <typename T>
Status AttentionBase<T>::ApplyAttention(OpKernelContext* context,
                                        const T* Q,                            // Q data with shape BxNxSxH
                                        const T* K,                            // K data with shape BxNxLxH
                                        const T* V,                            // V value with size BxNxLxH_v
                                        const Tensor* mask_index,              // mask index. nullptr if no mask or its size is B
                                        const Tensor* past_key,                // past K input tensor (if not using past state)
                                        const Tensor* past_value,              // past V input tensor (if not using past state)
                                        Tensor* output,                        // output tensor
                                        Tensor* present_key,                   // present K output tensor (if separating present KV)
                                        Tensor* present_value,                 // present V output tensor (if separating present KV)
                                        Tensor* output_qk,                     // Q*K output tensor (if returning Q*K value)
                                        const AttentionParameters& parameters  // attention parameters
) const {
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto* tp = context->GetOperatorThreadPool();

  const T* past_key_data = past_key != nullptr ? past_key->Data<T>() : nullptr;
  const T* past_value_data = past_value != nullptr ? past_value->Data<T>() : nullptr;

  // ORT_ATTENTION_USE_FLASH=1 selects the FlashAttention-style tiled path for
  // benchmarking.  The flash path avoids materialising the full [B,N,S,T]
  // probability tensor.  It falls back to the classic path when snapshot
  // outputs (qk_matmul_output_mode) are requested, because those require the
  // full matrix.
  static const bool use_flash =
      ParseEnvironmentVariableWithDefault<bool>("ORT_ATTENTION_USE_FLASH", false);

  if (use_flash && parameters.qk_matmul_output_mode == QKMatMulOutputMode::kNone) {
    return this->ApplyFlashAttention(context, Q, K, V,
                                     mask_index,
                                     past_key_data, past_value_data,
                                     output, present_key, present_value,
                                     parameters, allocator, tp);
  }

  T* present_key_data = present_key != nullptr ? present_key->MutableData<T>() : nullptr;
  T* present_value_data = present_value != nullptr ? present_value->MutableData<T>() : nullptr;
  T* output_qk_data = output_qk != nullptr ? output_qk->MutableData<T>() : nullptr;

  // Compute the attention score.
  size_t bytes = SafeInt<size_t>(parameters.batch_size) * parameters.q_num_heads *
                 parameters.q_sequence_length * parameters.total_sequence_length * sizeof(T);
  auto attention_probs = allocator->Alloc(bytes);
  BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));
  this->ComputeAttentionProbs(static_cast<T*>(attention_probs),
                              Q,
                              K,
                              mask_index,
                              parameters,
                              past_key_data,
                              present_key_data,
                              output_qk_data,
                              tp,
                              allocator);

  this->ComputeVxAttentionScore(output->MutableData<T>(),
                                static_cast<T*>(attention_probs),
                                V,
                                parameters.batch_size,
                                parameters.q_sequence_length,
                                parameters.kv_sequence_length,
                                parameters.past_sequence_length,
                                parameters.total_sequence_length,
                                parameters.v_head_size,
                                parameters.q_num_heads,
                                parameters.kv_num_heads,
                                past_value_data,
                                present_value_data,
                                parameters.transpose_output,
                                tp);

  return Status::OK();
}

}  // namespace onnxruntime
