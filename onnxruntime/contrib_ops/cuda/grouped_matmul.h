// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// GroupedMatMul: each token (row of `input`) is multiplied by k weight matrices selected from a
// stack of `num_groups` matrices via `group_indices` (shape [M, k]), producing the per-expert
// results (shape [M, k, N]).
// See docs/GroupedMatMul.md for the full specification.
//
// The kernel uses the standard grouped-GEMM strategy: selections are stable-sorted by group index,
// each selection's source token row is gathered into group-contiguous order, one dense cuBLAS GEMM
// is run per non-empty group, and the results (plus optional per-group bias) are scattered back to
// the [M, k, N] output in selection order.
template <typename T>
class GroupedMatMul final : public CudaKernel {
 public:
  explicit GroupedMatMul(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
