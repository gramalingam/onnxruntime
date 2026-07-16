// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

// SwiGLU: fused gated activation over two same-shaped tensors `gate` (G) and `linear` (L):
//   output = G * sigmoid(alpha * G) * L
template <typename T>
class SwiGLU final : public CudaKernel {
 public:
  explicit SwiGLU(const OpKernelInfo& info) : CudaKernel(info) {
    alpha_ = info.GetAttrOrDefault<float>("alpha", 1.0f);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float alpha_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
