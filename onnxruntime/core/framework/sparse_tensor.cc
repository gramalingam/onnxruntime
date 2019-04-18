// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"

using namespace onnxruntime::common;

namespace onnxruntime {

// TODO: Other
// extend: RegisterAllProtos
// call: DataTypeImpl::RegisterDataType ?
// extend: DataTypeImpl::ToString(MLDataType type) in data_types.cc
// extend: DataTypeImpl::TypeFromProto

// An implementation of sparse tensor.

/**
 * @brief This class implements SparseTensor.
 *
 * @details The class captures the 3 necessary elements of a Sparse Tensor
 *          values - a vector of non-zero sparse tensor values
 *          indices - a vector of indices of non zero values
 *          shape   - a scalar tensor that indicates the size of a single dimension
 *                   It is assumed that all of the values for the tensors are int64
 *          we use tensor datatypes as effective memory managers.
 */

// This type is a result of the construct_sparse OpKernel.
class SparseTensor final {
 public:
  SparseTensor() = default;
  ~SparseTensor() = default;

  SparseTensor(const SparseTensor&) = default;
  SparseTensor& operator=(const SparseTensor&) = default;
  SparseTensor(SparseTensor&&) = default;
  SparseTensor& operator=(SparseTensor&&) = default;

  const auto& Values() const {
    return values_;
  }

  const auto& Indices() const {
    return indices_;
  }

  const auto& Size() const {
    return size_;
  }

  auto& Values() {
    return values_;
  }

  auto& Indices() {
    return indices_;
  }

  auto& Size() {
    return size_;
  }

 private:
  int values_;
  int indices_;
  int size_;  // The value of a single dimension
};

}  // namespace onnxruntime
