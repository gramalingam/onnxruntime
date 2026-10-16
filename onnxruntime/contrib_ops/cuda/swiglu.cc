// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/swiglu.h"

#include "contrib_ops/cuda/swiglu_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                              \
      SwiGLU, kMSDomain, 1, T, kCudaExecutionProvider,        \
      (*KernelDefBuilder::Create())                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      SwiGLU<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Status SwiGLU<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename ToCudaType<T>::MappedType;

  const Tensor* gate = context->Input<Tensor>(0);
  const Tensor* linear = context->Input<Tensor>(1);

  const auto& gate_shape = gate->Shape();
  ORT_RETURN_IF_NOT(gate_shape == linear->Shape(),
                    "SwiGLU: gate and linear must have the same shape, got ",
                    gate_shape.ToString(), " and ", linear->Shape().ToString(), ".");

  Tensor* output = context->Output(0, gate_shape);
  const int64_t count = gate_shape.Size();
  if (count == 0) {
    return Status::OK();
  }

  cudaStream_t stream = Stream(context);
  LaunchSwiGLUKernel<CudaT>(stream,
                            reinterpret_cast<const CudaT*>(gate->Data<T>()),
                            reinterpret_cast<const CudaT*>(linear->Data<T>()),
                            reinterpret_cast<CudaT*>(output->MutableData<T>()),
                            alpha_, count);
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
