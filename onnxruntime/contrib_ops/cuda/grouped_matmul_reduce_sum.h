// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// GroupedMatMulReduceSum: fuses GroupedMatMul with a per-token weighted combine (Mul +
// ReduceSum) over the expert-selection axis -- the MoE down-projection pattern. See
// docs/GroupedMatMul.md for the full specification.
//
// Reuses the same gather + GEMM-execution steps as GroupedMatMul (including its
// ORT_GROUPED_MATMUL_CUDA_IMPL cublas/cutlass switch), replacing only the final scatter step
// with a fused gather-combine-reduce kernel (see grouped_matmul_reduce_sum_impl.h), so the two
// ops keep the exact same GEMM behavior/perf characteristics up to that point.
template <typename T>
class GroupedMatMulReduceSum final : public CudaKernel {
 public:
  explicit GroupedMatMulReduceSum(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
