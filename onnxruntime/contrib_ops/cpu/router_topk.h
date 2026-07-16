// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// RouterTopK: fused Mixture-of-Experts router top-k selection with renormalization.
//   values, indices = TopK(logits, k)   # k largest logits per row, sorted descending
//   weights = Softmax(values)           # softmax over only the k selected logits
// Equivalent to renormalizing the top-k of a full softmax, but softmax runs over k << E.
template <typename T>
class RouterTopK final : public OpKernel {
 public:
  explicit RouterTopK(const OpKernelInfo& info) : OpKernel(info) {
    k_ = info.GetAttrOrDefault<int64_t>("k", 1);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t k_;
};

}  // namespace contrib
}  // namespace onnxruntime
