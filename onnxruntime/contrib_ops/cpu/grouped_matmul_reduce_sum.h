// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// GroupedMatMulReduceSum: fuses GroupedMatMul with a per-token weighted combine (Mul +
// ReduceSum) over the expert-selection axis -- the MoE down-projection pattern. See
// docs/GroupedMatMul.md for the full specification.
template <typename T>
class GroupedMatMulReduceSum final : public OpKernel {
 public:
  explicit GroupedMatMulReduceSum(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
