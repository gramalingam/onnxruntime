#pragma once

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/framework/tensor_shape.h"

using namespace onnxruntime::common;

namespace onnxruntime {

/**
 * @brief This class implements SparseTensor.
 */

class SparseTensor final {
 public:
  SparseTensor(void* values, int64_t* indices, size_t nnz, const TensorShape& shape);
  ~SparseTensor() = default;  // TODO

  SparseTensor(const SparseTensor&) = default;
  SparseTensor& operator=(const SparseTensor&) = default;
  SparseTensor(SparseTensor&&) = default;
  SparseTensor& operator=(SparseTensor&&) = default;

  size_t NumValues() const { return nnz_; }

  const int64_t* Values() const {
    return static_cast<const int64_t*>(p_values_);  // TODO
  }

  const int64_t* Indices() const {
    return p_indices_;
  }

  const auto& Shape() const {
    return shape_;
  }

  int64_t* Values() {
    return static_cast<int64_t*>(p_values_);  // TODO
  }

  int64_t* Indices() {
    return p_indices_;
  }

  auto& Shape() {
    return shape_;
  }

 private:
  size_t nnz_;
  void* p_values_;
  int64_t* p_indices_;
  TensorShape shape_;  // The value of a single dimension
};

}  // namespace onnxruntime
