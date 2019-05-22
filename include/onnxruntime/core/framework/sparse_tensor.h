#pragma once

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"

using namespace onnxruntime::common;

namespace onnxruntime {

/**
 * @brief This class implements SparseTensor.
 */

class SparseTensor final {
 public:
  SparseTensor(Tensor* values, Tensor* indices, const TensorShape& shape);
  ~SparseTensor() = default;  // TODO

  SparseTensor(const SparseTensor&) = delete;
  SparseTensor& operator=(const SparseTensor&) = delete;
  SparseTensor(SparseTensor&&) = delete;
  SparseTensor& operator=(SparseTensor&&) = delete;

  size_t NumValues() const { return values_->Shape().Size(); }

  const Tensor& Values() const {
    return *values_;
  }

  const Tensor& Indices() const {
    return *indices_;
  }

  const TensorShape& Shape() const {
    return shape_;
  }

  Tensor& Values() {
    return *values_;
  }

  Tensor& Indices() {
    return *indices_;
  }

  TensorShape& Shape() {
    return shape_;
  }

 private:
  Tensor* values_;
  Tensor* indices_;
  TensorShape shape_;  // The value of a single dimension
};

}  // namespace onnxruntime
