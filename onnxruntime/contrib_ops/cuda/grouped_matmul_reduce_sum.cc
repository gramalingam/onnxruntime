// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/grouped_matmul_reduce_sum.h"

#include <vector>

#include "contrib_ops/cuda/grouped_matmul_cutlass_gemm.h"
#include "contrib_ops/cuda/grouped_matmul_reduce_sum_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      GroupedMatMulReduceSum, kMSDomain, 1, T, kCudaExecutionProvider,  \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
      GroupedMatMulReduceSum<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Status GroupedMatMulReduceSum<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename ToCudaType<T>::MappedType;

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* group_indices = context->Input<Tensor>(2);
  const Tensor* combine_weights = context->Input<Tensor>(3);
  const Tensor* bias = context->Input<Tensor>(4);

  const auto& input_shape = input->Shape();
  const auto& weights_shape = weights->Shape();

  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 3,
                    "GroupedMatMulReduceSum: input must have rank 3 (M, k, K), got rank ",
                    input_shape.NumDimensions());
  ORT_RETURN_IF_NOT(weights_shape.NumDimensions() == 3,
                    "GroupedMatMulReduceSum: weights must have rank 3 (num_groups, K, N), got rank ",
                    weights_shape.NumDimensions());

  const int64_t M = input_shape[0];
  const int64_t k = input_shape[1];
  const int64_t K = input_shape[2];
  const int64_t num_groups = weights_shape[0];
  const int64_t weights_K = weights_shape[1];
  const int64_t N = weights_shape[2];
  const int64_t num_selections = M * k;

  ORT_RETURN_IF_NOT(weights_K == K,
                    "GroupedMatMulReduceSum: weights dim 1 (", weights_K, ") must equal input last dim (",
                    K, ").");

  const auto& indices_shape = group_indices->Shape();
  ORT_RETURN_IF_NOT(indices_shape.NumDimensions() == 2 && indices_shape[0] == M && indices_shape[1] == k,
                    "GroupedMatMulReduceSum: group_indices must have shape (M, k) = (", M, ", ", k,
                    "), got ", indices_shape.ToString(), ".");

  const auto& combine_shape = combine_weights->Shape();
  ORT_RETURN_IF_NOT(combine_shape.NumDimensions() == 2 && combine_shape[0] == M && combine_shape[1] == k,
                    "GroupedMatMulReduceSum: combine_weights must have shape (M, k) = (", M, ", ", k, ").");

  if (bias != nullptr) {
    const auto& bias_shape = bias->Shape();
    ORT_RETURN_IF_NOT(bias_shape.NumDimensions() == 2 && bias_shape[0] == num_groups && bias_shape[1] == N,
                      "GroupedMatMulReduceSum: bias must have shape (num_groups, N) = (", num_groups, ", ",
                      N, ").");
  }

  // Output shape is (M, N): the combined per-token result.
  TensorShapeVector output_dims;
  output_dims.push_back(M);
  output_dims.push_back(N);
  Tensor* output = context->Output(0, TensorShape(output_dims));

  if (M == 0 || N == 0) {
    return Status::OK();
  }

  cudaStream_t stream = Stream(context);

  if (num_selections == 0) {
    // No selections at all (k == 0): output is all-zero.
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(output->MutableData<T>(), 0,
                                         static_cast<size_t>(M * N) * sizeof(T), stream));
    return Status::OK();
  }

  // Copy group indices to host to build the per-group permutation, exactly as GroupedMatMul
  // does. This is a small (num_selections) transfer and lets us sort/bucket without a device
  // radix sort.
  auto host_indices = AllocateBufferOnCPUPinned<int64_t>(static_cast<size_t>(num_selections));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_indices.get(), group_indices->Data<int64_t>(),
                                       static_cast<size_t>(num_selections) * sizeof(int64_t),
                                       cudaMemcpyDeviceToHost, stream));
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

  // Bucket selections by group id (stable within each group), producing:
  //   row_map[p]        = original selection index at group-contiguous position p
  //   group_ids[p]      = group id at position p
  //   group_offsets[g]  = start position of group g in the permuted order
  //   pos_of_selection[s] = position p such that row_map[p] == s (the inverse of row_map)
  std::vector<int64_t> group_counts(static_cast<size_t>(num_groups), 0);
  const int64_t* h_indices = host_indices.get();
  for (int64_t p = 0; p < num_selections; ++p) {
    const int64_t g = h_indices[p];
    ORT_RETURN_IF_NOT(g >= 0 && g < num_groups,
                      "GroupedMatMulReduceSum: group index ", g, " at selection ", p,
                      " is out of range [0, ", num_groups, ").");
    group_counts[static_cast<size_t>(g)]++;
  }

  std::vector<int64_t> group_offsets(static_cast<size_t>(num_groups) + 1, 0);
  for (int64_t g = 0; g < num_groups; ++g) {
    group_offsets[static_cast<size_t>(g) + 1] = group_offsets[static_cast<size_t>(g)] + group_counts[static_cast<size_t>(g)];
  }

  auto host_row_map = AllocateBufferOnCPUPinned<int64_t>(static_cast<size_t>(num_selections));
  auto host_group_ids = AllocateBufferOnCPUPinned<int64_t>(static_cast<size_t>(num_selections));
  auto host_pos_of_selection = AllocateBufferOnCPUPinned<int64_t>(static_cast<size_t>(num_selections));
  std::vector<int64_t> cursor(group_offsets.begin(), group_offsets.end() - 1);
  for (int64_t p = 0; p < num_selections; ++p) {
    const int64_t g = h_indices[p];
    const int64_t pos = cursor[static_cast<size_t>(g)]++;
    host_row_map.get()[pos] = p;
    host_group_ids.get()[pos] = g;
    host_pos_of_selection.get()[p] = pos;
  }

  // Upload permutation arrays to device.
  auto row_map = GetScratchBuffer<int64_t>(static_cast<size_t>(num_selections), context->GetComputeStream());
  auto group_ids = GetScratchBuffer<int64_t>(static_cast<size_t>(num_selections), context->GetComputeStream());
  auto pos_of_selection = GetScratchBuffer<int64_t>(static_cast<size_t>(num_selections), context->GetComputeStream());
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(row_map.get(), host_row_map.get(),
                                       static_cast<size_t>(num_selections) * sizeof(int64_t),
                                       cudaMemcpyHostToDevice, stream));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(group_ids.get(), host_group_ids.get(),
                                       static_cast<size_t>(num_selections) * sizeof(int64_t),
                                       cudaMemcpyHostToDevice, stream));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(pos_of_selection.get(), host_pos_of_selection.get(),
                                       static_cast<size_t>(num_selections) * sizeof(int64_t),
                                       cudaMemcpyHostToDevice, stream));

  // Gather each selection's own (token, expert-slot) input row into group-contiguous order
  // (no row reuse -- input has one distinct row per selection; see grouped_matmul_reduce_sum_impl.h).
  auto permuted_input = GetScratchBuffer<CudaT>(static_cast<size_t>(num_selections * K), context->GetComputeStream());
  auto permuted_output = GetScratchBuffer<CudaT>(static_cast<size_t>(num_selections * N), context->GetComputeStream());
  LaunchGroupedMatMulReduceSumGather<CudaT>(stream, reinterpret_cast<const CudaT*>(input->Data<T>()),
                                            row_map.get(), permuted_input.get(), num_selections, K);

  const CudaT* weights_data = reinterpret_cast<const CudaT*>(weights->Data<T>());

  if (UseCutlassGroupedMatMulGemm()) {
    // Single-launch CUTLASS grouped-GEMM path, same opt-in switch as GroupedMatMul (see
    // grouped_matmul_cutlass_gemm.h): ORT_GROUPED_MATMUL_CUDA_IMPL=cutlass.
    auto group_offsets_end = GetScratchBuffer<int64_t>(static_cast<size_t>(num_groups), context->GetComputeStream());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(group_offsets_end.get(), group_offsets.data() + 1,
                                         static_cast<size_t>(num_groups) * sizeof(int64_t),
                                         cudaMemcpyHostToDevice, stream));
    // Scratch for the [num_groups, N, K] transpose of `weights` that CUTLASS's MoeGemmRunner
    // expects (see the layout note in grouped_matmul_cutlass_gemm.h).
    auto weight_scratch = GetScratchBuffer<CudaT>(static_cast<size_t>(num_groups * K * N), context->GetComputeStream());
    LaunchGroupedMatMulCutlassGemm<CudaT>(stream, permuted_input.get(), weights_data, weight_scratch.get(),
                                          group_offsets_end.get(), permuted_output.get(), num_selections, K, N,
                                          num_groups);
  } else {
    // Default: one dense cuBLAS GEMM per non-empty group.
    const CudaT alpha = ToCudaType<T>::FromFloat(1.0f);
    const CudaT beta = ToCudaType<T>::FromFloat(0.0f);
    const auto& device_prop = GetDeviceProp();

    for (int64_t g = 0; g < num_groups; ++g) {
      const int64_t count = group_counts[static_cast<size_t>(g)];
      if (count == 0) {
        continue;  // Empty group: weights[g] is unused.
      }
      const int64_t offset = group_offsets[static_cast<size_t>(g)];
      const CudaT* a = permuted_input.get() + offset * K;  // [count, K] row-major
      const CudaT* w = weights_data + g * K * N;           // [K, N] row-major
      CudaT* c = permuted_output.get() + offset * N;       // [count, N] row-major

      // ORT tensors are row-major, cuBLAS is column-major, so swap the operands: compute
      // C^T = W^T * A^T which cuBLAS sees as an (N x count) = (N x K) * (K x count) product.
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          GetCublasHandle(context),
          CUBLAS_OP_N, CUBLAS_OP_N,
          static_cast<int>(N), static_cast<int>(count), static_cast<int>(K),
          &alpha,
          w, static_cast<int>(N),
          a, static_cast<int>(K),
          &beta,
          c, static_cast<int>(N),
          device_prop,
          UseTF32()));
    }
  }

  const CudaT* bias_data = bias ? reinterpret_cast<const CudaT*>(bias->Data<T>()) : nullptr;
  const CudaT* combine_data = reinterpret_cast<const CudaT*>(combine_weights->Data<T>());

  // Fused gather-combine-reduce: for each token, gather its k expert results (+ bias), scale by
  // the per-selection combine weight, and sum into a single output row.
  LaunchGroupedMatMulReduceSumFinalize<CudaT>(stream, permuted_output.get(), pos_of_selection.get(),
                                              group_ids.get(), combine_data, bias_data,
                                              reinterpret_cast<CudaT*>(output->MutableData<T>()),
                                              M, k, N);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
