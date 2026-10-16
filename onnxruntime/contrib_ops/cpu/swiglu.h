// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// SwiGLU: fused gated activation over two same-shaped tensors `gate` (G) and `linear` (L):
//   output = G * sigmoid(alpha * G) * L
// When alpha == 1 the gate is SiLU(G) = G * sigmoid(G). See docs/contrib op schema for details.
template <typename T>
class SwiGLU final : public OpKernel {
 public:
  explicit SwiGLU(const OpKernelInfo& info) : OpKernel(info) {
    alpha_ = info.GetAttrOrDefault<float>("alpha", 1.0f);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float alpha_;
};

}  // namespace contrib
}  // namespace onnxruntime
