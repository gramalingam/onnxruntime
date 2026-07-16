// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/swiglu.h"

#include <algorithm>

#include "core/common/common.h"
#include "core/common/float16.h"
#include "core/common/narrow.h"
#include "core/framework/allocator.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

namespace {

// Computes out = g * sigmoid(alpha * g) * l over `count` float elements, using a `tmp`
// scratch buffer of at least `count` floats. Input/output buffers must not overlap `tmp`.
void SwiGLUFloat(const float* g, const float* l, float* out, float* tmp, float alpha, size_t count) {
  if (alpha == 1.0f) {
    // sigmoid(g) * g == SiLU(g); MLAS has a fused SiLU primitive.
    MlasComputeSilu(g, tmp, count);
  } else {
    for (size_t i = 0; i < count; ++i) {
      tmp[i] = g[i] * alpha;
    }
    MlasComputeLogistic(tmp, tmp, count);  // tmp = sigmoid(alpha * g)
    MlasEltwiseMul<float>(g, tmp, tmp, count);  // tmp = g * sigmoid(alpha * g)
  }
  MlasEltwiseMul<float>(tmp, l, out, count);  // out = (g * sigmoid(alpha*g)) * l
}

}  // namespace

template <typename T>
Status SwiGLU<T>::Compute(OpKernelContext* context) const {
  const Tensor* gate = context->Input<Tensor>(0);
  const Tensor* linear = context->Input<Tensor>(1);

  const auto& gate_shape = gate->Shape();
  ORT_RETURN_IF_NOT(gate_shape == linear->Shape(),
                    "SwiGLU: gate and linear must have the same shape, got ",
                    gate_shape.ToString(), " and ", linear->Shape().ToString(), ".");

  Tensor* output = context->Output(0, gate_shape);
  const int64_t elem_count = gate_shape.Size();
  if (elem_count == 0) {
    return Status::OK();
  }

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  const float alpha = alpha_;

  // Process in chunks so per-task scratch stays small and work parallelizes across the pool.
  constexpr int64_t length_per_task = 4096;
  const int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;

  if constexpr (std::is_same_v<T, float>) {
    const float* gate_data = gate->Data<float>();
    const float* linear_data = linear->Data<float>();
    float* output_data = output->MutableData<float>();
    concurrency::ThreadPool::TryBatchParallelFor(
        tp, static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          const int64_t start = task_idx * length_per_task;
          const size_t count = narrow<size_t>(std::min(length_per_task, elem_count - start));
          std::vector<float> tmp(count);
          SwiGLUFloat(gate_data + start, linear_data + start, output_data + start, tmp.data(), alpha, count);
        },
        0);
    return Status::OK();
  } else {
    // Half/bfloat: convert each chunk to float, compute, convert back.
    const T* gate_data = gate->Data<T>();
    const T* linear_data = linear->Data<T>();
    T* output_data = output->MutableData<T>();
    concurrency::ThreadPool::TryBatchParallelFor(
        tp, static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          const int64_t start = task_idx * length_per_task;
          const size_t count = narrow<size_t>(std::min(length_per_task, elem_count - start));
          std::vector<float> gf(count), lf(count), of(count), tmp(count);
          for (size_t i = 0; i < count; ++i) {
            gf[i] = static_cast<float>(gate_data[start + i]);
            lf[i] = static_cast<float>(linear_data[start + i]);
          }
          SwiGLUFloat(gf.data(), lf.data(), of.data(), tmp.data(), alpha, count);
          for (size_t i = 0; i < count; ++i) {
            output_data[start + i] = static_cast<T>(of[i]);
          }
        },
        0);
    return Status::OK();
  }
}

#define REGISTER_SWIGLU_KERNEL(T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      SwiGLU, kMSDomain, 1, T, kCpuExecutionProvider,               \
      KernelDefBuilder()                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),   \
      SwiGLU<T>);

REGISTER_SWIGLU_KERNEL(float)
REGISTER_SWIGLU_KERNEL(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime
