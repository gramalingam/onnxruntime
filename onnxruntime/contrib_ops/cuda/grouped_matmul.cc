// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/grouped_matmul.h"

#include <vector>

#include "contrib_ops/cuda/grouped_matmul_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                              \
      GroupedMatMul, kMSDomain, 1, T, kCudaExecutionProvider, \
      (*KernelDefBuilder::Create())                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
      GroupedMatMul<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Status GroupedMatMul<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename ToCudaType<T>::MappedType;

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* group_indices = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);

  const auto& input_shape = input->Shape();
  const auto& weights_shape = weights->Shape();

  ORT_RETURN_IF_NOT(input_shape.NumDimensions() >= 2,
                    "GroupedMatMul: input must have rank >= 2, got rank ", input_shape.NumDimensions());
  ORT_RETURN_IF_NOT(weights_shape.NumDimensions() == 3,
                    "GroupedMatMul: weights must have rank 3 (num_groups, K, N), got rank ",
                    weights_shape.NumDimensions());

  const int64_t K = input_shape[input_shape.NumDimensions() - 1];
  const int64_t num_tokens = input_shape.Size() / K;
  const int64_t num_groups = weights_shape[0];
  const int64_t weights_K = weights_shape[1];
  const int64_t N = weights_shape[2];

  ORT_RETURN_IF_NOT(weights_K == K,
                    "GroupedMatMul: weights dim 1 (", weights_K, ") must equal input last dim (", K, ").");

  const auto& indices_shape = group_indices->Shape();
  ORT_RETURN_IF_NOT(indices_shape.Size() == num_tokens,
                    "GroupedMatMul: group_indices must have one entry per token (", num_tokens,
                    "), got ", indices_shape.Size(), ".");

  if (bias != nullptr) {
    const auto& bias_shape = bias->Shape();
    ORT_RETURN_IF_NOT(bias_shape.NumDimensions() == 2 && bias_shape[0] == num_groups && bias_shape[1] == N,
                      "GroupedMatMul: bias must have shape (num_groups, N) = (", num_groups, ", ", N, ").");
  }

  // Output shape: input shape with last dim replaced by N.
  TensorShapeVector output_dims(input_shape.GetDims().begin(), input_shape.GetDims().end());
  output_dims[output_dims.size() - 1] = N;
  Tensor* output = context->Output(0, TensorShape(output_dims));

  if (num_tokens == 0 || N == 0) {
    return Status::OK();
  }

  cudaStream_t stream = static_cast<cudaStream_t>(GetComputeStream(context));

  // Copy group indices to host to build the per-group permutation. This is a small (num_tokens)
  // transfer and lets us sort/bucket without a device radix sort.
  auto host_indices = AllocateBufferOnCPUPinned<int64_t>(static_cast<size_t>(num_tokens));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_indices.get(), group_indices->Data<int64_t>(),
                                       static_cast<size_t>(num_tokens) * sizeof(int64_t),
                                       cudaMemcpyDeviceToHost, stream));
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

  // Bucket tokens by group id (stable within each group), producing:
  //   row_map[p]   = original token index at group-contiguous position p
  //   group_ids[p] = group id at position p
  //   group_offsets[g] = start position of group g in the permuted order
  std::vector<int64_t> group_counts(static_cast<size_t>(num_groups), 0);
  const int64_t* h_indices = host_indices.get();
  for (int64_t i = 0; i < num_tokens; ++i) {
    const int64_t g = h_indices[i];
    ORT_RETURN_IF_NOT(g >= 0 && g < num_groups,
                      "GroupedMatMul: group index ", g, " at token ", i, " is out of range [0, ", num_groups, ").");
    group_counts[static_cast<size_t>(g)]++;
  }

  std::vector<int64_t> group_offsets(static_cast<size_t>(num_groups) + 1, 0);
  for (int64_t g = 0; g < num_groups; ++g) {
    group_offsets[static_cast<size_t>(g) + 1] = group_offsets[static_cast<size_t>(g)] + group_counts[static_cast<size_t>(g)];
  }

  auto host_row_map = AllocateBufferOnCPUPinned<int64_t>(static_cast<size_t>(num_tokens));
  auto host_group_ids = AllocateBufferOnCPUPinned<int64_t>(static_cast<size_t>(num_tokens));
  std::vector<int64_t> cursor(group_offsets.begin(), group_offsets.end() - 1);
  for (int64_t i = 0; i < num_tokens; ++i) {
    const int64_t g = h_indices[i];
    const int64_t pos = cursor[static_cast<size_t>(g)]++;
    host_row_map.get()[pos] = i;
    host_group_ids.get()[pos] = g;
  }

  // Upload permutation arrays to device.
  auto row_map = GetScratchBuffer<int64_t>(static_cast<size_t>(num_tokens), context->GetComputeStream());
  auto group_ids = GetScratchBuffer<int64_t>(static_cast<size_t>(num_tokens), context->GetComputeStream());
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(row_map.get(), host_row_map.get(),
                                       static_cast<size_t>(num_tokens) * sizeof(int64_t),
                                       cudaMemcpyHostToDevice, stream));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(group_ids.get(), host_group_ids.get(),
                                       static_cast<size_t>(num_tokens) * sizeof(int64_t),
                                       cudaMemcpyHostToDevice, stream));

  // Gather input rows into group-contiguous order.
  auto permuted_input = GetScratchBuffer<CudaT>(static_cast<size_t>(num_tokens * K), context->GetComputeStream());
  auto permuted_output = GetScratchBuffer<CudaT>(static_cast<size_t>(num_tokens * N), context->GetComputeStream());
  LaunchGroupedMatMulGather<CudaT>(stream, reinterpret_cast<const CudaT*>(input->Data<T>()),
                                   row_map.get(), permuted_input.get(), num_tokens, K);

  // One dense GEMM per non-empty group.
  const CudaT alpha = ToCudaType<T>::FromFloat(1.0f);
  const CudaT beta = ToCudaType<T>::FromFloat(0.0f);
  const auto& device_prop = GetDeviceProp();
  const CudaT* weights_data = reinterpret_cast<const CudaT*>(weights->Data<T>());

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

  // Scatter results back to original token order, adding the optional per-group bias.
  const CudaT* bias_data = bias ? reinterpret_cast<const CudaT*>(bias->Data<T>()) : nullptr;
  LaunchGroupedMatMulScatter<CudaT>(stream, permuted_output.get(), row_map.get(), group_ids.get(),
                                    bias_data, reinterpret_cast<CudaT*>(output->MutableData<T>()),
                                    num_tokens, N);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
