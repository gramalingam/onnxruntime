// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// GroupedMatMul: for each token (row along the flattened leading dims of `input`),
// multiply by one weight matrix selected from a stack of `num_groups` matrices via
// `group_indices`. See docs/GroupedMatMul.md for the full specification.
template <typename T>
class GroupedMatMul final : public OpKernel {
 public:
  explicit GroupedMatMul(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
