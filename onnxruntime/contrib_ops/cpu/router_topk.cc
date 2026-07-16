// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/router_topk.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "core/common/common.h"
#include "core/common/float16.h"
#include "core/framework/tensor_shape.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
Status RouterTopK<T>::Compute(OpKernelContext* context) const {
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

  const T* logits_data = logits->Data<T>();
  T* weights_data = weights->MutableData<T>();
  int64_t* indices_data = indices->MutableData<int64_t>();

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  concurrency::ThreadPool::TryBatchParallelFor(
      tp, static_cast<int32_t>(rows),
      [&](ptrdiff_t row) {
        const T* row_logits = logits_data + row * E;
        T* row_weights = weights_data + row * k;
        int64_t* row_indices = indices_data + row * k;

        // Select the top-k expert indices, sorted by (logit value desc, index asc), matching
        // ONNX TopK with largest=1, sorted=1.
        std::vector<int64_t> order(static_cast<size_t>(E));
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(
            order.begin(), order.begin() + static_cast<ptrdiff_t>(k), order.end(),
            [&](int64_t a, int64_t b) {
              const float va = static_cast<float>(row_logits[a]);
              const float vb = static_cast<float>(row_logits[b]);
              if (va != vb) {
                return va > vb;
              }
              return a < b;
            });

        // Softmax over the k selected logits (computed in float for numerical stability).
        float max_logit = static_cast<float>(row_logits[order[0]]);
        float sum = 0.0f;
        std::vector<float> exps(static_cast<size_t>(k));
        for (int64_t j = 0; j < k; ++j) {
          const int64_t idx = order[static_cast<size_t>(j)];
          const float e = std::exp(static_cast<float>(row_logits[idx]) - max_logit);
          exps[static_cast<size_t>(j)] = e;
          sum += e;
          row_indices[j] = idx;
        }
        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int64_t j = 0; j < k; ++j) {
          row_weights[j] = static_cast<T>(exps[static_cast<size_t>(j)] * inv_sum);
        }
      },
      0);

  return Status::OK();
}

#define REGISTER_ROUTER_TOPK_KERNEL(T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                    \
      RouterTopK, kMSDomain, 1, T, kCpuExecutionProvider,          \
      KernelDefBuilder()                                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())   \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
      RouterTopK<T>);

REGISTER_ROUTER_TOPK_KERNEL(float)
REGISTER_ROUTER_TOPK_KERNEL(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime
