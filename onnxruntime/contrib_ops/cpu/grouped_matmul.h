// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// GroupedMatMul: each token (row of `input`) is multiplied by k weight matrices selected
// from a stack of `num_groups` matrices via `group_indices` (shape [M, k]). When optional
// `combine_weights` is provided, the k results are combined into a weighted sum. See
// docs/GroupedMatMul.md for the full specification.
template <typename T>
class GroupedMatMul final : public OpKernel {
 public:
  explicit GroupedMatMul(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
