// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/router_topk.h"

#include "contrib_ops/cuda/router_topk_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                              \
      RouterTopK, kMSDomain, 1, T, kCudaExecutionProvider,    \
      (*KernelDefBuilder::Create())                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
      RouterTopK<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
Status RouterTopK<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename ToCudaType<T>::MappedType;

  const Tensor* logits = context->Input<Tensor>(0);
  const auto& logits_shape = logits->Shape();

  ORT_RETURN_IF_NOT(logits_shape.NumDimensions() >= 1,
                    "RouterTopK: logits must have rank >= 1.");
  const int64_t E = logits_shape[logits_shape.NumDimensions() - 1];
  const int64_t k = k_;
  ORT_RETURN_IF_NOT(k >= 1 && k <= E,
                    "RouterTopK: k (", k, ") must satisfy 1 <= k <= num_experts (", E, ").");
  const int64_t rows = (E == 0) ? 0 : logits_shape.Size() / E;

  TensorShapeVector out_dims(logits_shape.GetDims().begin(), logits_shape.GetDims().end());
  out_dims[out_dims.size() - 1] = k;
  const TensorShape out_shape(out_dims);
  Tensor* weights = context->Output(0, out_shape);
  Tensor* indices = context->Output(1, out_shape);
  if (rows == 0) {
    return Status::OK();
  }

  cudaStream_t stream = static_cast<cudaStream_t>(GetComputeStream(context));
  LaunchRouterTopKKernel<CudaT>(stream,
                                reinterpret_cast<const CudaT*>(logits->Data<T>()),
                                reinterpret_cast<CudaT*>(weights->MutableData<T>()),
                                indices->MutableData<int64_t>(),
                                rows, E, k);
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
