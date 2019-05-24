// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdio.h>
#include <onnx/onnx_pb.h>
#include "core/common/logging/logging.h"
#include "core/framework/data_types.h"
#include "core/framework/environment.h"
#include "core/framework/framework_common.h"
#include "core/framework/mem_buffer.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"

#include "onnx-ml.pb.h"
#include "predict.pb.h"

#include "converter.h"
#include "executor.h"
#include "util.h"

namespace onnxruntime {
namespace server {

namespace protobufutil = google::protobuf::util;

protobufutil::Status Executor::SetMLValue(const onnx::TensorProto& input_tensor,
                                          MemBufferArray& buffers,
                                          OrtAllocatorInfo* cpu_allocator_info,
                                          /* out */ MLValue& ml_value) {
  auto logger = env_->GetLogger(request_id_);

  size_t cpu_tensor_length = 0;
  auto status = onnxruntime::utils::GetSizeInBytesFromTensorProto<0>(input_tensor, &cpu_tensor_length);
  if (!status.IsOK()) {
    LOGS(*logger, ERROR) << "GetSizeInBytesFromTensorProto() failed. Error Message: " << status.ToString();
    return GenerateProtobufStatus(status, "GetSizeInBytesFromTensorProto() failed: " + status.ToString());
  }

  OrtCallback deleter;
  auto* buf = buffers.AllocNewBuffer(cpu_tensor_length);
  status = onnxruntime::utils::TensorProtoToMLValue(onnxruntime::Env::Default(), nullptr, input_tensor,
                                                    onnxruntime::MemBuffer(buf, cpu_tensor_length, *cpu_allocator_info),
                                                    ml_value, deleter);
  if (!status.IsOK()) {
    LOGS(*logger, ERROR) << "TensorProtoToMLValue() failed. Message: " << status.ToString();
    return GenerateProtobufStatus(status, "TensorProtoToMLValue() failed:" + status.ToString());
  }

  return protobufutil::Status::OK;
}

protobufutil::Status Executor::SetNameMLValueMap(onnxruntime::NameMLValMap& name_value_map,
                                                 const onnxruntime::server::PredictRequest& request,
                                                 MemBufferArray& buffers) {
  auto logger = env_->GetLogger(request_id_);

  OrtAllocatorInfo* allocator_info = nullptr;
  auto ort_status = OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &allocator_info);

  if (ort_status != nullptr || allocator_info == nullptr) {
    LOGS(*logger, ERROR) << "OrtCreateAllocatorInfo failed";
    return protobufutil::Status(protobufutil::error::Code::RESOURCE_EXHAUSTED, "OrtCreateAllocatorInfo() failed");
  }

  // Prepare the MLValue object
  for (const auto& input : request.inputs()) {
    using_raw_data_ = using_raw_data_ && input.second.has_raw_data();

    MLValue ml_value;
    auto status = SetMLValue(input.second, buffers, allocator_info, ml_value);
    if (status != protobufutil::Status::OK) {
      OrtReleaseAllocatorInfo(allocator_info);
      LOGS(*logger, ERROR) << "SetMLValue() failed! Input name: " << input.first;
      return status;
    }

    auto insertion_result = name_value_map.insert(std::make_pair(input.first, ml_value));
    if (!insertion_result.second) {
      OrtReleaseAllocatorInfo(allocator_info);
      LOGS(*logger, ERROR) << "SetNameMLValueMap() failed! Input name: " << input.first << " Trying to overwrite existing input value";
      return protobufutil::Status(protobufutil::error::Code::INVALID_ARGUMENT, "SetNameMLValueMap() failed: Cannot have two inputs with the same name");
    }
  }

  OrtReleaseAllocatorInfo(allocator_info);
  return protobufutil::Status::OK;
}

protobufutil::Status Executor::Predict(const std::string& model_name,
                                       const std::string& model_version,
                                       onnxruntime::server::PredictRequest& request,
                                       /* out */ onnxruntime::server::PredictResponse& response) {
  auto logger = env_->GetLogger(request_id_);

  // Convert PredictRequest to NameMLValMap
  MemBufferArray buffer_array;
  onnxruntime::NameMLValMap name_ml_value_map{};
  auto conversion_status = SetNameMLValueMap(name_ml_value_map, request, buffer_array);
  if (conversion_status != protobufutil::Status::OK) {
    return conversion_status;
  }

  // Prepare the output names and vector
  std::vector<std::string> output_names;

  if (!request.output_filter().empty()) {
    output_names.reserve(request.output_filter_size());
    for (const auto& name : request.output_filter()) {
      output_names.push_back(name);
    }
  } else {
    output_names = env_->GetModelOutputNames();
  }

  std::vector<OrtValue> outputs(output_names.size());

  // Run
  OrtRunOptions run_options{};
  run_options.run_log_verbosity_level = static_cast<unsigned int>(env_->GetLogSeverity());
  run_options.run_tag = request_id_;

  auto status = env_->GetSession()->Run(run_options, name_ml_value_map, output_names, &outputs);

  if (!status.IsOK()) {
    LOGS(*logger, ERROR) << "Run() failed."
                         << ". Error Message: " << status.ToString();
    return GenerateProtobufStatus(status, "Run() failed: " + status.ToString());
  }

  // Build the response
  for (size_t i = 0, sz = outputs.size(); i < sz; ++i) {
    onnx::TensorProto output_tensor{};
    status = MLValueToTensorProto(outputs[i], using_raw_data_, std::move(logger), output_tensor);
    logger = env_->GetLogger(request_id_);

    if (!status.IsOK()) {
      LOGS(*logger, ERROR) << "MLValueToTensorProto() failed. Output name: " << output_names[i] << ". Error Message: " << status.ToString();
      return GenerateProtobufStatus(status, "MLValueToTensorProto() failed: " + status.ToString());
    }

    auto insertion_result = response.mutable_outputs()->insert({output_names[i], output_tensor});

    if (!insertion_result.second) {
      LOGS(*logger, ERROR) << "SetNameMLValueMap() failed. Output name: " << output_names[i] << " Trying to overwrite existing output value";
      return protobufutil::Status(protobufutil::error::Code::INVALID_ARGUMENT, "SetNameMLValueMap() failed: Cannot have two outputs with the same name");
    }
  }

  return protobufutil::Status::OK;
}

}  // namespace server
}  // namespace onnxruntime