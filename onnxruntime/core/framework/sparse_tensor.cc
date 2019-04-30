// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/framework/sparse_tensor.h"

using namespace onnxruntime::common;

namespace onnxruntime {

SparseTensor::SparseTensor(void* values, int64_t* indices, size_t nnz, const TensorShape& shape) : p_values_(values),
                                                                                                   p_indices_(indices),
                                                                                                   shape_(shape),
                                                                                                   nnz_(nnz) {}

}  // namespace onnxruntime
