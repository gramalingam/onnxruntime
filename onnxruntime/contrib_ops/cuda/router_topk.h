// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// RouterTopK: fused Mixture-of-Experts router top-k selection with renormalization.
//   values, indices = TopK(logits, k); weights = Softmax(values).
template <typename T>
class RouterTopK final : public CudaKernel {
 public:
  explicit RouterTopK(const OpKernelInfo& info) : CudaKernel(info) {
    k_ = info.GetAttrOrDefault<int64_t>("k", 1);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t k_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
