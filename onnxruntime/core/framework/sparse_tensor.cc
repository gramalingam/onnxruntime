// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/framework/sparse_tensor.h"

using namespace onnxruntime::common;

namespace onnxruntime {

SparseTensor::SparseTensor(Tensor* values, Tensor* indices, const TensorShape& shape) : values_(values),
                                                                                        indices_(indices),
                                                                                        shape_(shape) {}

}  // namespace onnxruntime
