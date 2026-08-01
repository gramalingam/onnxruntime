#include "core/optimizer/function_extractor_pattern.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <cstring>
#include <deque>
#include <memory>
#include <type_traits>

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/function_template.h"
#include "core/graph/function_utils.h"
#include "core/graph/model.h"
#include "onnx/checker.h"

namespace onnxruntime {
namespace function_extractor_internal {
namespace {

using common::Status;

Status InvalidPattern(const std::string& message) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid function extraction pattern: ", message);
}

bool IsSupportedFormalAttributeType(ONNX_NAMESPACE::AttributeProto_AttributeType type) {
  switch (type) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT:
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRING:
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR:
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS:
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INTS:
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS:
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSORS:
      return true;
    default:
      return false;
  }
}

bool HasConcreteAttributeValue(const ONNX_NAMESPACE::AttributeProto& attribute) {
  switch (attribute.type()) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT:
      return attribute.has_f();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
      return attribute.has_i();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRING:
      return attribute.has_s();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR:
      return attribute.has_t();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS:
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INTS:
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS:
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSORS:
      return true;
    default:
      return false;
  }
}

bool AttributeReferenceContainsValue(const ONNX_NAMESPACE::AttributeProto& attribute) {
  switch (attribute.type()) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT:
      return attribute.has_f();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
      return attribute.has_i();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRING:
      return attribute.has_s();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR:
      return attribute.has_t();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS:
      return attribute.floats_size() != 0;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INTS:
      return attribute.ints_size() != 0;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS:
      return attribute.strings_size() != 0;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSORS:
      return attribute.tensors_size() != 0;
    default:
      return false;
  }
}

bool IsParameterizedConstantAttribute(std::string_view name,
                                      ONNX_NAMESPACE::AttributeProto_AttributeType type) {
  return (name == "value" && type == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) ||
         (name == "value_float" && type == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT) ||
         (name == "value_floats" && type == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS) ||
         (name == "value_int" && type == ONNX_NAMESPACE::AttributeProto_AttributeType_INT) ||
         (name == "value_ints" && type == ONNX_NAMESPACE::AttributeProto_AttributeType_INTS) ||
         (name == "value_string" && type == ONNX_NAMESPACE::AttributeProto_AttributeType_STRING) ||
         (name == "value_strings" && type == ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS);
}

std::string CanonicalDomain(std::string_view domain) {
  return domain.empty() || domain == kOnnxDomainAlias ? std::string{kOnnxDomain} : std::string{domain};
}

template <typename T>
Status AppendTensorData(const ONNX_NAMESPACE::TensorProto& tensor,
                        size_t element_count,
                        size_t max_bytes,
                        const std::filesystem::path* model_path,
                        std::string& data) {
  const size_t bytes = SafeInt<size_t>(element_count) * sizeof(T);
  ORT_RETURN_IF_NOT(bytes <= max_bytes, "Tensor literal byte budget exceeded.");
  InlinedVector<T> values(element_count);
  if (tensor.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
    ORT_RETURN_IF_NOT(model_path != nullptr, "External tensor data requires a model path.");
    ORT_RETURN_IF_ERROR(utils::UnpackTensor(tensor, *model_path, values.data(), element_count));
  } else {
    const void* raw_data = tensor.has_raw_data() ? tensor.raw_data().data() : nullptr;
    const size_t raw_data_size = tensor.has_raw_data() ? tensor.raw_data().size() : 0;
    ORT_RETURN_IF_ERROR(utils::UnpackTensor(tensor, raw_data, raw_data_size, values.data(), element_count));
  }
  data.append(reinterpret_cast<const char*>(values.data()), bytes);
  return Status::OK();
}

template <typename T>
Status AppendPackedTensorData(const ONNX_NAMESPACE::TensorProto& tensor,
                              size_t storage_count,
                              size_t element_count,
                              size_t max_bytes,
                              std::string& data) {
  const size_t bytes = SafeInt<size_t>(storage_count) * sizeof(T);
  ORT_RETURN_IF_NOT(bytes <= max_bytes, "Tensor literal byte budget exceeded.");
  InlinedVector<T> values(storage_count);
  const void* raw_data = tensor.has_raw_data() ? tensor.raw_data().data() : nullptr;
  const size_t raw_data_size = tensor.has_raw_data() ? tensor.raw_data().size() : 0;
  ORT_RETURN_IF_ERROR(
      utils::UnpackTensor(tensor, raw_data, raw_data_size, values.data(), element_count));
  data.append(reinterpret_cast<const char*>(values.data()), bytes);
  return Status::OK();
}

Status AppendBoolTensorData(const ONNX_NAMESPACE::TensorProto& tensor,
                            size_t element_count,
                            size_t max_bytes,
                            std::string& data) {
  ORT_RETURN_IF_NOT(element_count <= max_bytes, "Tensor literal byte budget exceeded.");
  std::unique_ptr<bool[]> values = std::make_unique<bool[]>(element_count);
  const void* raw_data = tensor.has_raw_data() ? tensor.raw_data().data() : nullptr;
  const size_t raw_data_size = tensor.has_raw_data() ? tensor.raw_data().size() : 0;
  ORT_RETURN_IF_ERROR(
      utils::UnpackTensor(tensor, raw_data, raw_data_size, values.get(), element_count));
  for (size_t i = 0; i < element_count; ++i) {
    data.push_back(values[i] ? '\1' : '\0');
  }
  return Status::OK();
}

template <typename T>
Status AppendComplexTensorData(const ONNX_NAMESPACE::TensorProto& tensor,
                               size_t element_count,
                               size_t max_bytes,
                               std::string& data) {
  const size_t scalar_count = SafeInt<size_t>(element_count) * 2;
  const size_t bytes = SafeInt<size_t>(scalar_count) * sizeof(T);
  ORT_RETURN_IF_NOT(bytes <= max_bytes, "Tensor literal byte budget exceeded.");
  if (tensor.has_raw_data()) {
    ORT_RETURN_IF_NOT(tensor.raw_data().size() == bytes,
                      "Complex tensor raw data size does not match its shape.");
    data.append(tensor.raw_data());
    return Status::OK();
  }
  if constexpr (std::is_same_v<T, float>) {
    ORT_RETURN_IF_NOT(static_cast<size_t>(tensor.float_data_size()) == scalar_count,
                      "Complex tensor data size does not match its shape.");
    for (const float value : tensor.float_data()) {
      data.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  } else {
    ORT_RETURN_IF_NOT(static_cast<size_t>(tensor.double_data_size()) == scalar_count,
                      "Complex tensor data size does not match its shape.");
    for (const double value : tensor.double_data()) {
      data.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
  return Status::OK();
}

Status TensorLogicalBytes(const ONNX_NAMESPACE::TensorProto& tensor,
                          size_t max_bytes,
                          const std::filesystem::path* model_path,
                          std::string& data) {
  size_t element_count = 1;
  for (const int64_t dim : tensor.dims()) {
    ORT_RETURN_IF_NOT(dim >= 0, "Tensor literal has a negative dimension.");
    element_count = SafeInt<size_t>(element_count) * static_cast<size_t>(dim);
  }

  data.clear();
  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      ORT_RETURN_IF_ERROR(AppendTensorData<float>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      ORT_RETURN_IF_ERROR(AppendTensorData<double>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      ORT_RETURN_IF_ERROR(AppendTensorData<uint16_t>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      ORT_RETURN_IF_ERROR(AppendTensorData<MLFloat16>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      ORT_RETURN_IF_ERROR(AppendTensorData<BFloat16>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      ORT_RETURN_IF_ERROR(AppendTensorData<int16_t>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      ORT_RETURN_IF_ERROR(AppendTensorData<int8_t>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      ORT_RETURN_IF_ERROR(AppendTensorData<uint8_t>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      ORT_RETURN_IF_ERROR(AppendBoolTensorData(tensor, element_count, max_bytes, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      ORT_RETURN_IF_ERROR(AppendTensorData<int32_t>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      ORT_RETURN_IF_ERROR(AppendTensorData<uint32_t>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      ORT_RETURN_IF_ERROR(AppendTensorData<int64_t>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      ORT_RETURN_IF_ERROR(AppendTensorData<uint64_t>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64:
      ORT_RETURN_IF_ERROR(AppendComplexTensorData<float>(tensor, element_count, max_bytes, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128:
      ORT_RETURN_IF_ERROR(AppendComplexTensorData<double>(tensor, element_count, max_bytes, data));
      break;
#if !defined(DISABLE_FLOAT8_TYPES)
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
      ORT_RETURN_IF_ERROR(AppendTensorData<Float8E4M3FN>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ:
      ORT_RETURN_IF_ERROR(AppendTensorData<Float8E4M3FNUZ>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:
      ORT_RETURN_IF_ERROR(AppendTensorData<Float8E5M2>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ:
      ORT_RETURN_IF_ERROR(AppendTensorData<Float8E5M2FNUZ>(tensor, element_count, max_bytes, model_path, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E8M0:
      ORT_RETURN_IF_ERROR(AppendTensorData<Float8E8M0>(tensor, element_count, max_bytes, model_path, data));
      break;
#endif
    case ONNX_NAMESPACE::TensorProto_DataType_INT4:
      ORT_RETURN_IF_ERROR(AppendPackedTensorData<Int4x2>(
          tensor, Int4x2::CalcNumInt4Pairs(element_count), element_count, max_bytes, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT4:
      ORT_RETURN_IF_ERROR(AppendPackedTensorData<UInt4x2>(
          tensor, UInt4x2::CalcNumInt4Pairs(element_count), element_count, max_bytes, data));
      break;
#if !defined(DISABLE_FLOAT4_TYPES)
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT4E2M1:
      ORT_RETURN_IF_ERROR(AppendPackedTensorData<Float4E2M1x2>(
          tensor, Float4E2M1x2::CalcNumFloat4Pairs(element_count), element_count, max_bytes, data));
      break;
#endif
    case ONNX_NAMESPACE::TensorProto_DataType_INT2:
      ORT_RETURN_IF_ERROR(AppendPackedTensorData<Int2x4>(
          tensor, Int2x4::CalcNumInt2Quads(element_count), element_count, max_bytes, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT2:
      ORT_RETURN_IF_ERROR(AppendPackedTensorData<UInt2x4>(
          tensor, UInt2x4::CalcNumInt2Quads(element_count), element_count, max_bytes, data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      for (const auto& value : tensor.string_data()) {
        const uint64_t length = value.size();
        ORT_RETURN_IF_NOT(SafeInt<size_t>(data.size()) + sizeof(length) + value.size() <= max_bytes,
                          "Tensor literal byte budget exceeded.");
        data.append(reinterpret_cast<const char*>(&length), sizeof(length));
        data.append(value);
      }
      break;
    default:
      return InvalidPattern("unsupported tensor literal data type " + std::to_string(tensor.data_type()));
  }

  ORT_RETURN_IF_NOT(data.size() <= max_bytes, "Tensor literal byte budget exceeded.");
  return Status::OK();
}

Status CanonicalizeTensorAttribute(const ONNX_NAMESPACE::TensorProto& source,
                                   size_t max_bytes,
                                   ONNX_NAMESPACE::TensorProto& canonical) {
  ORT_RETURN_IF(source.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL ||
                    source.external_data_size() != 0,
                "Formal attribute tensors may not use external data.");
  std::string bytes;
  ORT_RETURN_IF_ERROR(TensorLogicalBytes(source, max_bytes, nullptr, bytes));
  canonical.Clear();
  canonical.set_data_type(source.data_type());
  canonical.mutable_dims()->CopyFrom(source.dims());
  if (source.data_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
    canonical.mutable_string_data()->CopyFrom(source.string_data());
  } else {
    canonical.set_raw_data(std::move(bytes));
  }
  return Status::OK();
}

Status MakeLiteralDescriptor(const ONNX_NAMESPACE::NodeProto& node, size_t max_bytes, LiteralDescriptor& literal) {
  NodeAttributes attributes;
  for (const auto& attribute : node.attribute()) {
    if (!attribute.ref_attr_name().empty()) {
      return InvalidPattern("Constant attributes may not use ref_attr_name");
    }
    attributes.emplace(attribute.name(), attribute);
  }

  ORT_RETURN_IF_ERROR(NormalizeConstantAttributes(attributes, literal.tensor));
  ORT_RETURN_IF(literal.tensor.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL,
                "Function body literals may not use external tensor data.");
  std::string bytes;
  ORT_RETURN_IF_ERROR(TensorLogicalBytes(literal.tensor, max_bytes, nullptr, bytes));
  literal.byte_count = bytes.size();
  literal.fingerprint = std::to_string(literal.tensor.data_type()) + ":";
  for (const int64_t dim : literal.tensor.dims()) {
    literal.fingerprint += std::to_string(dim) + ",";
  }
  literal.fingerprint.push_back(':');
  literal.fingerprint.append(bytes);
  return Status::OK();
}

void AddSchemaDefaults(const ONNX_NAMESPACE::OpSchema& schema,
                       const InlinedHashSet<std::string>& variable_names,
                       NodeAttributes& attributes) {
  for (const auto& [name, definition] : schema.attributes()) {
    if (variable_names.find(name) == variable_names.end() &&
        attributes.find(name) == attributes.end() &&
        utils::HasName(definition.default_value)) {
      attributes.emplace(name, definition.default_value);
    }
  }
}

bool IsAllowedOnnxPureOp(std::string_view op_type) {
  static const InlinedHashSet<std::string> pure_ops{
      "Identity", "Add", "Sub", "Mul", "Div", "Relu", "Cast", "MatMul",
      "Transpose", "Reshape", "Clip", "Concat", "MaxPool", "LeakyRelu"};
  return pure_ops.find(op_type) != pure_ops.end();
}

Status ValidateTransitivePurity(const ONNX_NAMESPACE::FunctionProto& function,
                                const Graph& graph,
                                InlinedHashSet<std::string>& visiting) {
  const auto identity =
      function_utils::GetFunctionIdentifier(CanonicalDomain(function.domain()), function.name(), function.overload());
  ORT_RETURN_IF_NOT(visiting.insert(identity).second,
                    "Recursive model-local functions are not supported by FunctionExtractor.");

  for (const auto& node : function.node()) {
    for (const auto& attribute : node.attribute()) {
      ORT_RETURN_IF(attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH ||
                        attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS,
                    "Nested functions with graph attributes are not supported.");
    }
    const auto domain = CanonicalDomain(node.domain());
    if (domain == kOnnxDomain && IsAllowedOnnxPureOp(node.op_type())) continue;

    const auto function_id = function_utils::GetFunctionIdentifier(domain, node.op_type(), node.overload());
    const auto& local_functions = graph.GetModel().GetModelLocalFunctionTemplates();
    const auto local = local_functions.find(function_id);
    if (local != local_functions.end()) {
      ORT_RETURN_IF_ERROR(ValidateTransitivePurity(*local->second->onnx_func_proto_, graph, visiting));
      continue;
    }

    int version = -1;
    for (const auto& import : function.opset_import()) {
      if (CanonicalDomain(import.domain()) == domain) {
        version = static_cast<int>(import.version());
        break;
      }
    }
    ORT_RETURN_IF_NOT(version >= 0, "Nested function has no opset import for domain '", domain, "'.");
    const auto* schema = graph.GetSchemaRegistry()->GetSchema(node.op_type(), version, domain);
    ORT_RETURN_IF(schema == nullptr || schema->HasContextDependentFunction() || !schema->HasFunction(),
                  "Nested function contains an unknown or impure operation: ", domain, ":", node.op_type());
    const auto* nested = schema->GetFunction(version, domain == kOnnxDomain);
    ORT_RETURN_IF_NOT(nested != nullptr, "Nested schema function body is unavailable.");
    ORT_RETURN_IF_ERROR(ValidateTransitivePurity(*nested, graph, visiting));
  }

  visiting.erase(identity);
  return Status::OK();
}

Status ResolveNode(const PatternNode& pattern_node,
                   const NormalizedFunctionPattern& normalized_pattern,
                   const Graph& graph,
                   ResolvedPatternNode& resolved_node) {
  const auto& node_proto = normalized_pattern.function_proto.node(
      static_cast<int>(pattern_node.source_node_proto_index));
  resolved_node.canonical_domain = CanonicalDomain(node_proto.domain());
  resolved_node.op_type = node_proto.op_type();
  resolved_node.overload = node_proto.overload();
  resolved_node.input_arity = pattern_node.input_value_ids.size();
  resolved_node.output_arity = pattern_node.output_value_ids.size();
  resolved_node.attribute_variables = pattern_node.attribute_variables;
  resolved_node.is_parameterized_constant = pattern_node.is_parameterized_constant;
  InlinedHashSet<std::string> variable_names;
  for (const auto& occurrence : pattern_node.attribute_variables) {
    variable_names.insert(occurrence.operator_attribute_name);
  }
  for (const auto& attribute : node_proto.attribute()) {
    if (attribute.ref_attr_name().empty()) {
      resolved_node.effective_attributes.emplace(attribute.name(), attribute);
    }
  }

  const auto function_id =
      function_utils::GetFunctionIdentifier(resolved_node.canonical_domain, resolved_node.op_type,
                                            resolved_node.overload);
  const auto& local_functions = graph.GetModel().GetModelLocalFunctionTemplates();
  const auto local_function = local_functions.find(function_id);
  if (local_function != local_functions.end()) {
    resolved_node.schema = local_function->second->op_schema_.get();
    resolved_node.since_version = resolved_node.schema->since_version();
    resolved_node.function_fingerprint =
        CanonicalFunctionFingerprint(*local_function->second->onnx_func_proto_);
    InlinedHashSet<std::string> visiting;
    ORT_RETURN_IF_ERROR(
        ValidateTransitivePurity(*local_function->second->onnx_func_proto_, graph, visiting));
    resolved_node.transitively_pure = true;
    AddSchemaDefaults(*resolved_node.schema, variable_names, resolved_node.effective_attributes);
    return Status::OK();
  }

  int version = -1;
  for (const auto& import : normalized_pattern.function_proto.opset_import()) {
    if (CanonicalDomain(import.domain()) == resolved_node.canonical_domain) {
      version = static_cast<int>(import.version());
      break;
    }
  }
  if (version < 0) {
    const auto graph_import = graph.DomainToVersionMap().find(resolved_node.canonical_domain);
    if (graph_import != graph.DomainToVersionMap().end()) {
      version = graph_import->second;
    }
  }
  ORT_RETURN_IF_NOT(version >= 0, "No opset import for pattern node domain '",
                    resolved_node.canonical_domain, "'.");

  resolved_node.schema =
      graph.GetSchemaRegistry()->GetSchema(resolved_node.op_type, version, resolved_node.canonical_domain);
  ORT_RETURN_IF_NOT(resolved_node.schema != nullptr, "No schema for pattern node ",
                    resolved_node.canonical_domain, ":", resolved_node.op_type, " at opset ", version, ".");
  resolved_node.since_version = resolved_node.schema->since_version();
  const auto* standard_schema = ONNX_NAMESPACE::OpSchemaRegistry::Instance()->GetSchema(
      resolved_node.op_type, version,
      resolved_node.canonical_domain == kOnnxDomain ? "" : resolved_node.canonical_domain);
  resolved_node.is_standard_onnx_schema = resolved_node.schema == standard_schema;
  AddSchemaDefaults(*resolved_node.schema, variable_names, resolved_node.effective_attributes);
  return Status::OK();
}

bool AttributeEqual(const ONNX_NAMESPACE::AttributeProto& lhs, const ONNX_NAMESPACE::AttributeProto& rhs) {
  if (IsSupportedFormalAttributeType(lhs.type()) && lhs.type() == rhs.type()) {
    bool equal = false;
    const auto status =
        CompareFormalAttributes(lhs, rhs, std::numeric_limits<size_t>::max(), equal);
    return status.IsOK() && equal;
  }
  ONNX_NAMESPACE::AttributeProto normalized_lhs = lhs;
  ONNX_NAMESPACE::AttributeProto normalized_rhs = rhs;
  normalized_lhs.clear_doc_string();
  normalized_rhs.clear_doc_string();
  return normalized_lhs.SerializeAsString() == normalized_rhs.SerializeAsString();
}

}  // namespace

NormalizedFunctionPattern NormalizeFunctionPattern(
    const ONNX_NAMESPACE::FunctionProto& function_proto,
    const FunctionExtractorOptions& options) {
  NormalizedFunctionPattern result;
  auto fail = [&result](Status status) -> NormalizedFunctionPattern {
    result.construction_status = std::move(status);
    return std::move(result);
  };

  if (static_cast<size_t>(function_proto.node_size()) > options.max_pattern_nodes) {
    return fail(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Function pattern node budget exceeded."));
  }

  size_t pattern_work_units = 0;
  auto consume_pattern_work = [&](size_t units) {
    if (pattern_work_units > options.max_worklist_bindings ||
        units > options.max_worklist_bindings - pattern_work_units) {
      return false;
    }
    pattern_work_units += units;
    return true;
  };
  for (const auto& node : function_proto.node()) {
    const size_t input_slots = static_cast<size_t>(node.input_size());
    const size_t output_slots = static_cast<size_t>(node.output_size());
    // Count every input conservatively as both a body slot and a consumer incidence.
    if (!consume_pattern_work(input_slots) ||
        !consume_pattern_work(input_slots) ||
        !consume_pattern_work(output_slots)) {
      return fail(ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "Function extraction invariant/resource limit exceeded: pattern slot budget."));
    }
  }

  try {
    ONNX_NAMESPACE::checker::CheckerContext checker_context;
    checker_context.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    InlinedHashMap<std::string, int> imports;
    for (const auto& import : function_proto.opset_import()) {
      imports[CanonicalDomain(import.domain())] = static_cast<int>(import.version());
    }
    checker_context.set_opset_imports(
        std::unordered_map<std::string, int>(imports.begin(), imports.end()));
    ONNX_NAMESPACE::checker::LexicalScopeContext lexical_scope;
    ONNX_NAMESPACE::checker::check_function(function_proto, checker_context, lexical_scope);
  } catch (const ONNX_NAMESPACE::checker::ValidationError& error) {
    return fail(InvalidPattern(error.what()));
  }

  if (function_proto.name().empty()) {
    return fail(InvalidPattern("function name must not be empty"));
  }

  InlinedHashMap<std::string, PatternValueId> value_ids;
  InlinedHashSet<std::string> formal_names;
  auto add_formals = [&](const auto& names, bool input) -> Status {
    for (const auto& name : names) {
      ORT_RETURN_IF(name.empty(), "Formal ", input ? "input" : "output", " name must not be empty.");
      ORT_RETURN_IF(!formal_names.insert(name).second, "Formal names must be distinct and disjoint: ", name);
      PatternValueId value_id;
      const auto existing = value_ids.find(name);
      if (existing == value_ids.end()) {
        value_id = result.values.size();
        value_ids.emplace(name, value_id);
        result.values.push_back(PatternValue{});
        result.values.back().name = name;
      } else {
        value_id = existing->second;
      }
      if (input) {
        result.values[value_id].is_formal_input = true;
        result.formal_input_value_ids.push_back(value_id);
      } else {
        result.values[value_id].is_formal_output = true;
        result.formal_output_value_ids.push_back(value_id);
      }
    }
    return Status::OK();
  };
  Status status = add_formals(function_proto.input(), true);
  if (!status.IsOK()) return fail(InvalidPattern(status.ErrorMessage()));
  status = add_formals(function_proto.output(), false);
  if (!status.IsOK()) return fail(InvalidPattern(status.ErrorMessage()));

  size_t pattern_attribute_payload_bytes = 0;
  auto consume_attribute_bytes = [&](size_t bytes) {
    if (pattern_attribute_payload_bytes > options.max_attribute_bytes ||
        bytes > options.max_attribute_bytes - pattern_attribute_payload_bytes) {
      return false;
    }
    pattern_attribute_payload_bytes += bytes;
    return true;
  };
  InlinedHashMap<std::string, FormalAttributeId> formal_attribute_ids;
  for (const auto& name : function_proto.attribute()) {
    if (name.empty() || formal_attribute_ids.find(name) != formal_attribute_ids.end()) {
      return fail(InvalidPattern("required function attributes must be non-empty and non-duplicated"));
    }
    if (result.formal_attributes.size() >= options.max_formal_attributes ||
        !consume_pattern_work(1)) {
      return fail(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Function attribute pattern budget exceeded."));
    }
    const FormalAttributeId formal_id = result.formal_attributes.size();
    formal_attribute_ids.emplace(name, formal_id);
    result.formal_attributes.push_back(FormalAttributePattern{});
    result.formal_attributes.back().formal_name = name;
    result.formal_attributes.back().required = true;
  }
  for (const auto& attribute : function_proto.attribute_proto()) {
    if (attribute.name().empty() ||
        formal_attribute_ids.find(attribute.name()) != formal_attribute_ids.end()) {
      return fail(InvalidPattern(
          "defaulted function attributes must be non-empty, non-duplicated, and disjoint from required attributes"));
    }
    if (!attribute.ref_attr_name().empty() ||
        !IsSupportedFormalAttributeType(attribute.type()) ||
        !HasConcreteAttributeValue(attribute)) {
      return fail(InvalidPattern("function attribute default must contain one supported concrete value"));
    }
    if (result.formal_attributes.size() >= options.max_formal_attributes ||
        !consume_pattern_work(1) ||
        !consume_pattern_work(1)) {
      return fail(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Function attribute pattern budget exceeded."));
    }
    ONNX_NAMESPACE::AttributeProto canonical_default;
    status = CanonicalizeFormalAttribute(
        attribute.name(), attribute.type(), attribute, options.max_attribute_bytes, canonical_default);
    if (!status.IsOK()) return fail(std::move(status));
    if (!consume_attribute_bytes(AttributePayloadBytes(canonical_default))) {
      return fail(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Function attribute byte budget exceeded."));
    }
    const FormalAttributeId formal_id = result.formal_attributes.size();
    formal_attribute_ids.emplace(attribute.name(), formal_id);
    result.formal_attributes.push_back(FormalAttributePattern{});
    auto& formal = result.formal_attributes.back();
    formal.formal_name = attribute.name();
    formal.type = attribute.type();
    formal.canonical_default = std::move(canonical_default);
  }

  InlinedVector<InlinedVector<AttributeVariableOccurrence>> source_attribute_variables(
      function_proto.node_size());
  InlinedVector<bool> folded_constant_nodes(function_proto.node_size(), false);
  InlinedVector<bool> parameterized_constant_nodes(function_proto.node_size(), false);
  InlinedHashSet<std::string> produced_names;
  for (int node_index = 0; node_index < function_proto.node_size(); ++node_index) {
    const auto& node = function_proto.node(node_index);
    InlinedHashSet<std::string> attribute_names;
    for (const auto& attribute : node.attribute()) {
      if (!attribute_names.insert(attribute.name()).second) {
        return fail(InvalidPattern("duplicate node attribute '" + attribute.name() + "'"));
      }
      if (!attribute.ref_attr_name().empty()) {
        const auto formal_it = formal_attribute_ids.find(attribute.ref_attr_name());
        if (attribute.name().empty() ||
            formal_it == formal_attribute_ids.end() ||
            !IsSupportedFormalAttributeType(attribute.type()) ||
            AttributeReferenceContainsValue(attribute)) {
          return fail(InvalidPattern(
              "attribute reference must name a declared formal, declare a supported type, and contain no value"));
        }
        auto& formal = result.formal_attributes[formal_it->second];
        if (formal.type == ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED) {
          formal.type = attribute.type();
        } else if (formal.type != attribute.type()) {
          return fail(InvalidPattern("all occurrences of a formal attribute must have the same type"));
        }
        ONNX_NAMESPACE::AttributeProto occurrence_metadata = attribute;
        occurrence_metadata.clear_doc_string();
        if (!consume_pattern_work(1) ||
            !consume_attribute_bytes(occurrence_metadata.ByteSizeLong())) {
          return fail(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Function attribute pattern budget exceeded."));
        }
        source_attribute_variables[node_index].push_back(
            AttributeVariableOccurrence{kNoPatternNode, attribute.name(), formal_it->second});
        continue;
      }
      if (attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH ||
          attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS) {
        return fail(InvalidPattern("graph-valued attributes are not supported"));
      }
    }

    const bool is_constant = CanonicalDomain(node.domain()) == kOnnxDomain && node.op_type() == "Constant";
    if (is_constant && node.output_size() != 1) {
      return fail(InvalidPattern("Constant nodes must have exactly one output"));
    }
    if (is_constant && !source_attribute_variables[node_index].empty()) {
      if (node.input_size() != 0 || node.attribute_size() != 1 ||
          !IsParameterizedConstantAttribute(
              source_attribute_variables[node_index][0].operator_attribute_name,
              result.formal_attributes[source_attribute_variables[node_index][0].formal_attribute_id].type)) {
        return fail(InvalidPattern("parameterized Constant has an unsupported attribute form"));
      }
      parameterized_constant_nodes[node_index] = true;
    } else {
      folded_constant_nodes[node_index] = is_constant;
    }
    for (int output_index = 0; output_index < node.output_size(); ++output_index) {
      const std::string& output = node.output(output_index);
      if (output.empty()) {
        continue;
      }
      if (!produced_names.insert(output).second) {
        return fail(InvalidPattern("value '" + output + "' has multiple producers"));
      }
      PatternValueId value_id;
      const auto existing = value_ids.find(output);
      if (existing == value_ids.end()) {
        value_id = result.values.size();
        value_ids.emplace(output, value_id);
        result.values.push_back(PatternValue{});
        result.values.back().name = output;
      } else {
        value_id = existing->second;
      }
      if (result.values[value_id].is_formal_input) {
        return fail(InvalidPattern("formal input '" + output + "' is produced by a body node"));
      }
      if (folded_constant_nodes[node_index]) {
        result.values[value_id].is_literal = true;
        status = MakeLiteralDescriptor(node, options.max_literal_bytes, result.values[value_id].literal);
        if (!status.IsOK()) return fail(std::move(status));
      }
    }
  }

  InlinedVector<bool> formal_referenced(result.formal_attributes.size(), false);
  for (const auto& occurrences : source_attribute_variables) {
    for (const auto& occurrence : occurrences) {
      formal_referenced[occurrence.formal_attribute_id] = true;
    }
  }
  for (FormalAttributeId formal_id = 0;
       formal_id < result.formal_attributes.size();
       ++formal_id) {
    if (!formal_referenced[formal_id] ||
        result.formal_attributes[formal_id].type ==
            ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED) {
      return fail(InvalidPattern("every formal attribute must be referenced by the function body"));
    }
    ONNX_NAMESPACE::AttributeProto declaration_metadata;
    declaration_metadata.set_name(result.formal_attributes[formal_id].formal_name);
    declaration_metadata.set_type(result.formal_attributes[formal_id].type);
    if (!consume_attribute_bytes(declaration_metadata.ByteSizeLong())) {
      return fail(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Function attribute byte budget exceeded."));
    }
  }
  result.pattern_attribute_payload_bytes = pattern_attribute_payload_bytes;
  result.function_proto = function_proto;

  for (const auto& value_info : function_proto.value_info()) {
    if (!value_info.has_type() || !value_info.type().has_tensor_type()) {
      return fail(InvalidPattern("only tensor value_info entries are supported"));
    }
    const auto value = value_ids.find(value_info.name());
    if (value != value_ids.end()) {
      result.values[value->second].type = value_info.type();
      result.values[value->second].has_type = true;
    }
  }

  InlinedVector<PatternNodeId> source_to_pattern(function_proto.node_size(), kNoPatternNode);
  for (int source_index = 0; source_index < function_proto.node_size(); ++source_index) {
    if (folded_constant_nodes[source_index]) continue;
    if (function_proto.node(source_index).output_size() == 0) {
      return fail(InvalidPattern("function operation nodes must have outputs"));
    }
    source_to_pattern[source_index] = result.nodes.size();
    result.nodes.push_back(PatternNode{static_cast<size_t>(source_index)});
    auto& pattern_node = result.nodes.back();
    pattern_node.is_parameterized_constant = parameterized_constant_nodes[source_index];
    pattern_node.attribute_variables = source_attribute_variables[source_index];
    for (auto& occurrence : pattern_node.attribute_variables) {
      occurrence.pattern_node_id = source_to_pattern[source_index];
      result.formal_attributes[occurrence.formal_attribute_id].occurrences.push_back(occurrence);
    }
  }
  if (result.nodes.size() > options.max_pattern_nodes) {
    return fail(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Function pattern node budget exceeded."));
  }
  if (result.nodes.size() <= 1) {
    return fail(InvalidPattern("function pattern must contain more than one operation node"));
  }

  for (int source_index = 0; source_index < function_proto.node_size(); ++source_index) {
    const auto& source = function_proto.node(source_index);
    const PatternNodeId pattern_node_id = source_to_pattern[source_index];
    for (int output_index = 0; output_index < source.output_size(); ++output_index) {
      const auto& output = source.output(output_index);
      if (output.empty()) {
        if (!folded_constant_nodes[source_index]) {
          result.nodes[pattern_node_id].output_value_ids.push_back(kMissingPatternValue);
        }
        continue;
      }
      auto& value = result.values[value_ids.at(output)];
      if (!folded_constant_nodes[source_index]) {
        value.producer_node_id = pattern_node_id;
        value.producer_output_index = static_cast<size_t>(output_index);
        result.nodes[pattern_node_id].output_value_ids.push_back(value_ids.at(output));
      }
    }
    if (folded_constant_nodes[source_index]) continue;

    for (int input_index = 0; input_index < source.input_size(); ++input_index) {
      const auto& input = source.input(input_index);
      if (input.empty()) {
        result.nodes[pattern_node_id].input_value_ids.push_back(kMissingPatternValue);
        continue;
      }
      const auto value = value_ids.find(input);
      if (value == value_ids.end()) {
        return fail(InvalidPattern("input value '" + input + "' is undefined"));
      }
      result.nodes[pattern_node_id].input_value_ids.push_back(value->second);
      result.values[value->second].consumers.push_back(
          PatternValueConsumer{pattern_node_id, static_cast<size_t>(input_index)});
    }
  }

  size_t total_literal_bytes = 0;
  for (const auto& value : result.values) {
    if (value.is_literal) {
      total_literal_bytes = SafeInt<size_t>(total_literal_bytes) + value.literal.byte_count;
      if (value.is_formal_output) {
        return fail(InvalidPattern("formal outputs may not be produced by Constant nodes"));
      }
    }
    if ((value.is_formal_input || value.is_literal) && value.consumers.empty()) {
      return fail(InvalidPattern("formal inputs and literals must participate in the function body"));
    }
  }
  if (total_literal_bytes > options.max_literal_bytes) {
    return fail(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Function literal byte budget exceeded."));
  }

  InlinedVector<InlinedVector<PatternNodeId>> weak_neighbors(result.nodes.size());
  for (const auto& value : result.values) {
    InlinedVector<PatternNodeId> incident_nodes;
    if (value.producer_node_id != kNoPatternNode) {
      incident_nodes.push_back(value.producer_node_id);
    }
    for (const auto& consumer : value.consumers) {
      if (std::find(incident_nodes.begin(), incident_nodes.end(), consumer.node_id) ==
          incident_nodes.end()) {
        incident_nodes.push_back(consumer.node_id);
      }
    }
    for (size_t i = 1; i < incident_nodes.size(); ++i) {
      weak_neighbors[incident_nodes.front()].push_back(incident_nodes[i]);
      weak_neighbors[incident_nodes[i]].push_back(incident_nodes.front());
    }
  }
  InlinedHashSet<PatternNodeId> connected_nodes;
  InlinedVector<PatternNodeId> connectivity_pending{PatternNodeId{0}};
  while (!connectivity_pending.empty()) {
    const auto node_id = connectivity_pending.back();
    connectivity_pending.pop_back();
    if (!connected_nodes.insert(node_id).second) continue;
    connectivity_pending.insert(connectivity_pending.end(),
                                weak_neighbors[node_id].begin(),
                                weak_neighbors[node_id].end());
  }
  if (connected_nodes.size() != result.nodes.size()) {
    return fail(InvalidPattern("function operation and leaf data flow must form one connected component"));
  }

  InlinedVector<size_t> indegree(result.nodes.size(), 0);
  InlinedVector<InlinedVector<PatternNodeId>> successors(result.nodes.size());
  for (PatternNodeId node_id = 0; node_id < result.nodes.size(); ++node_id) {
    for (const auto input_id : result.nodes[node_id].input_value_ids) {
      if (input_id == kMissingPatternValue) continue;
      const auto producer = result.values[input_id].producer_node_id;
      if (producer != kNoPatternNode) {
        ++indegree[node_id];
        successors[producer].push_back(node_id);
      }
    }
  }
  std::deque<PatternNodeId> ready;
  for (PatternNodeId node_id = 0; node_id < indegree.size(); ++node_id) {
    if (indegree[node_id] == 0) ready.push_back(node_id);
  }
  InlinedVector<PatternNodeId> topological;
  while (!ready.empty()) {
    const auto node_id = ready.front();
    ready.pop_front();
    topological.push_back(node_id);
    for (const auto successor : successors[node_id]) {
      if (--indegree[successor] == 0) ready.push_back(successor);
    }
  }
  if (topological.size() != result.nodes.size()) {
    return fail(InvalidPattern("operation body must be acyclic"));
  }
  result.reverse_topological_node_ids.assign(topological.rbegin(), topological.rend());

  InlinedHashSet<PatternNodeId> reachable;
  InlinedVector<PatternNodeId> pending;
  for (const auto output_id : result.formal_output_value_ids) {
    const auto& value = result.values[output_id];
    if (value.producer_node_id == kNoPatternNode) {
      return fail(InvalidPattern("formal output '" + value.name + "' is not produced by an operation node"));
    }
    if (result.nodes[value.producer_node_id].is_parameterized_constant) {
      return fail(InvalidPattern("formal outputs may not be produced by Constant nodes"));
    }
    pending.push_back(value.producer_node_id);
  }
  while (!pending.empty()) {
    const auto node_id = pending.back();
    pending.pop_back();
    if (!reachable.insert(node_id).second) continue;
    for (const auto input_id : result.nodes[node_id].input_value_ids) {
      if (input_id == kMissingPatternValue) continue;
      const auto producer = result.values[input_id].producer_node_id;
      if (producer != kNoPatternNode) pending.push_back(producer);
    }
  }
  if (reachable.size() != result.nodes.size()) {
    return fail(InvalidPattern("every operation must be backward-reachable from a formal output"));
  }

  result.construction_status = Status::OK();
  return result;
}

Status CompileFunctionPattern(
    const NormalizedFunctionPattern& normalized_pattern,
    const Graph& graph,
    CompiledFunctionPattern& compiled_pattern) {
  ORT_RETURN_IF_ERROR(normalized_pattern.construction_status);
  compiled_pattern = CompiledFunctionPattern{};
  compiled_pattern.normalized_pattern = &normalized_pattern;
  compiled_pattern.resolved_nodes.resize(normalized_pattern.nodes.size());

  for (PatternNodeId node_id = 0; node_id < normalized_pattern.nodes.size(); ++node_id) {
    auto& resolved = compiled_pattern.resolved_nodes[node_id];
    resolved.pattern_node_id = node_id;
    ORT_RETURN_IF_ERROR(ResolveNode(normalized_pattern.nodes[node_id], normalized_pattern, graph, resolved));
    for (const auto& occurrence : resolved.attribute_variables) {
      const auto schema_attribute = resolved.schema->attributes().find(occurrence.operator_attribute_name);
      ORT_RETURN_IF(schema_attribute == resolved.schema->attributes().end() ||
                        schema_attribute->second.type !=
                            normalized_pattern.formal_attributes[occurrence.formal_attribute_id].type,
                    "Function attribute variable does not agree with the resolved operator schema: ",
                    occurrence.operator_attribute_name);
      if (utils::HasName(schema_attribute->second.default_value)) {
        ONNX_NAMESPACE::AttributeProto canonical_default;
        ORT_RETURN_IF_ERROR(CanonicalizeFormalAttribute(
            normalized_pattern.formal_attributes[occurrence.formal_attribute_id].formal_name,
            normalized_pattern.formal_attributes[occurrence.formal_attribute_id].type,
            schema_attribute->second.default_value,
            std::numeric_limits<size_t>::max(),
            canonical_default));
      }
    }
    ORT_RETURN_IF_NOT(IsV1PureOperator(resolved), "Function pattern contains an impure or unsupported operator: ",
                      resolved.canonical_domain, ":", resolved.op_type);
  }

  InlinedHashMap<PatternNodeId, size_t> group_by_producer;
  for (size_t formal_output_index = 0;
       formal_output_index < normalized_pattern.formal_output_value_ids.size();
       ++formal_output_index) {
    const auto& value =
        normalized_pattern.values[normalized_pattern.formal_output_value_ids[formal_output_index]];
    auto [group_it, inserted] =
        group_by_producer.emplace(value.producer_node_id, compiled_pattern.formal_output_producer_groups.size());
    if (inserted) {
      compiled_pattern.formal_output_producer_groups.push_back(
          FormalOutputProducerGroup{value.producer_node_id});
    }
    auto& group = compiled_pattern.formal_output_producer_groups[group_it->second];
    group.formal_output_indices.push_back(formal_output_index);
    group.producer_output_indices.push_back(value.producer_output_index);
  }
  return Status::OK();
}

Status ValidateRegisteredFunction(
    const NormalizedFunctionPattern& normalized_pattern,
    const Graph& graph) {
  const auto& function = normalized_pattern.function_proto;
  const auto function_id =
      function_utils::GetFunctionIdentifier(CanonicalDomain(function.domain()), function.name(), function.overload());
  const auto& local_functions = graph.GetModel().GetModelLocalFunctionTemplates();
  const auto local_function = local_functions.find(function_id);
  if (local_function != local_functions.end()) {
    ORT_RETURN_IF_NOT(
        CanonicalFunctionFingerprint(*local_function->second->onnx_func_proto_) ==
            CanonicalFunctionFingerprint(function),
        "The registered model-local function definition differs from the extraction pattern.");
    return Status::OK();
  }

  const auto domain = CanonicalDomain(function.domain());
  const auto import = graph.DomainToVersionMap().find(domain);
  ORT_RETURN_IF(import == graph.DomainToVersionMap().end(),
                "The extraction function is not registered in the target model.");
  const auto* schema = graph.GetSchemaRegistry()->GetSchema(function.name(), import->second, domain);
  ORT_RETURN_IF(schema == nullptr, "The extraction function is not registered in the target model.");
  ORT_RETURN_IF(schema->HasContextDependentFunction(),
                "Context-dependent schema functions are not supported by FunctionExtractor.");
  ORT_RETURN_IF_NOT(schema->HasFunction(),
                    "The registered schema does not provide a function body.");
  const auto* registered_function = schema->GetFunction(import->second, domain == kOnnxDomain);
  ORT_RETURN_IF_NOT(registered_function != nullptr, "The registered schema function body is unavailable.");
  ORT_RETURN_IF_NOT(CanonicalFunctionFingerprint(*registered_function) ==
                        CanonicalFunctionFingerprint(function),
                    "The registered schema function definition differs from the extraction pattern.");
  return Status::OK();
}

bool IsV1PureOperator(const ResolvedPatternNode& node) {
  if (node.is_parameterized_constant) {
    return node.is_standard_onnx_schema &&
           node.canonical_domain == kOnnxDomain && node.op_type == "Constant" &&
           node.input_arity == 0 && node.output_arity == 1 &&
           node.attribute_variables.size() == 1;
  }
  if (node.canonical_domain != kOnnxDomain) {
    return !node.function_fingerprint.empty() && node.transitively_pure;
  }
  return IsAllowedOnnxPureOp(node.op_type);
}

bool AreAttributesSemanticallyEqual(const NodeAttributes& lhs, const NodeAttributes& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (const auto& [name, attribute] : lhs) {
    const auto other = rhs.find(name);
    if (other == rhs.end() || !AttributeEqual(attribute, other->second)) return false;
  }
  return true;
}

size_t AttributePayloadBytes(const ONNX_NAMESPACE::AttributeProto& attribute) {
  ONNX_NAMESPACE::AttributeProto payload = attribute;
  payload.clear_name();
  payload.clear_ref_attr_name();
  payload.clear_doc_string();
  return payload.ByteSizeLong();
}

Status CanonicalizeFormalAttribute(
    std::string_view formal_name,
    ONNX_NAMESPACE::AttributeProto_AttributeType declared_type,
    const ONNX_NAMESPACE::AttributeProto& source,
    size_t max_attribute_bytes,
    ONNX_NAMESPACE::AttributeProto& canonical) {
  ORT_RETURN_IF_NOT(IsSupportedFormalAttributeType(declared_type) &&
                        source.type() == declared_type &&
                        HasConcreteAttributeValue(source),
                    "Formal attribute has no supported concrete value.");
  canonical.Clear();
  canonical.set_name(std::string{formal_name});
  canonical.set_type(declared_type);
  switch (declared_type) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT:
      canonical.set_f(source.f());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
      canonical.set_i(source.i());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRING:
      canonical.set_s(source.s());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR: {
      ORT_RETURN_IF_ERROR(
          CanonicalizeTensorAttribute(source.t(), max_attribute_bytes, *canonical.mutable_t()));
      break;
    }
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS:
      canonical.mutable_floats()->CopyFrom(source.floats());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INTS:
      canonical.mutable_ints()->CopyFrom(source.ints());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS:
      canonical.mutable_strings()->CopyFrom(source.strings());
      break;
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSORS:
      for (const auto& tensor : source.tensors()) {
        ORT_RETURN_IF_ERROR(
            CanonicalizeTensorAttribute(tensor, max_attribute_bytes, *canonical.add_tensors()));
      }
      break;
    default:
      return InvalidPattern("unsupported formal attribute type");
  }
  ORT_RETURN_IF(AttributePayloadBytes(canonical) > max_attribute_bytes,
                "Function attribute byte budget exceeded.");
  return Status::OK();
}

Status CompareFormalAttributes(
    const ONNX_NAMESPACE::AttributeProto& lhs,
    const ONNX_NAMESPACE::AttributeProto& rhs,
    size_t max_attribute_bytes,
    bool& equal) {
  equal = false;
  if (lhs.type() != rhs.type()) return Status::OK();
  switch (lhs.type()) {
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT: {
      uint32_t lhs_bits;
      uint32_t rhs_bits;
      const float lhs_value = lhs.f();
      const float rhs_value = rhs.f();
      std::memcpy(&lhs_bits, &lhs_value, sizeof(lhs_bits));
      std::memcpy(&rhs_bits, &rhs_value, sizeof(rhs_bits));
      equal = lhs_bits == rhs_bits;
      return Status::OK();
    }
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
      equal = lhs.i() == rhs.i();
      return Status::OK();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRING:
      equal = lhs.s() == rhs.s();
      return Status::OK();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR:
      return CompareTensorLiterals(lhs.t(), rhs.t(), max_attribute_bytes, equal);
    case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS:
      if (lhs.floats_size() != rhs.floats_size()) return Status::OK();
      equal = true;
      for (int i = 0; i < lhs.floats_size(); ++i) {
        uint32_t lhs_bits;
        uint32_t rhs_bits;
        const float lhs_value = lhs.floats(i);
        const float rhs_value = rhs.floats(i);
        std::memcpy(&lhs_bits, &lhs_value, sizeof(lhs_bits));
        std::memcpy(&rhs_bits, &rhs_value, sizeof(rhs_bits));
        if (lhs_bits != rhs_bits) {
          equal = false;
          break;
        }
      }
      return Status::OK();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_INTS:
      if (lhs.ints_size() != rhs.ints_size()) return Status::OK();
      equal = std::equal(lhs.ints().begin(), lhs.ints().end(), rhs.ints().begin());
      return Status::OK();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS:
      if (lhs.strings_size() != rhs.strings_size()) return Status::OK();
      equal = std::equal(lhs.strings().begin(), lhs.strings().end(), rhs.strings().begin());
      return Status::OK();
    case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSORS:
      if (lhs.tensors_size() != rhs.tensors_size()) return Status::OK();
      equal = true;
      for (int i = 0; i < lhs.tensors_size(); ++i) {
        bool tensor_equal = false;
        ORT_RETURN_IF_ERROR(
            CompareTensorLiterals(lhs.tensors(i), rhs.tensors(i), max_attribute_bytes, tensor_equal));
        if (!tensor_equal) {
          equal = false;
          break;
        }
      }
      return Status::OK();
    default:
      return InvalidPattern("unsupported formal attribute type");
  }
}

Status NormalizeConstantAttributes(
    const NodeAttributes& attributes,
    ONNX_NAMESPACE::TensorProto& tensor) {
  ORT_RETURN_IF_NOT(attributes.size() == 1,
                    "Constant nodes must specify exactly one supported value attribute.");
  const auto& [name, attribute] = *attributes.begin();
  tensor.Clear();
  if (name == "value" && attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
    tensor = attribute.t();
    return Status::OK();
  }
  if (name == "value_float" && attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT) {
    tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor.add_float_data(attribute.f());
    return Status::OK();
  }
  if (name == "value_floats" && attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS) {
    tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor.add_dims(attribute.floats_size());
    for (const auto value : attribute.floats()) tensor.add_float_data(value);
    return Status::OK();
  }
  if (name == "value_int" && attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INT) {
    tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    tensor.add_int64_data(attribute.i());
    return Status::OK();
  }
  if (name == "value_ints" && attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INTS) {
    tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    tensor.add_dims(attribute.ints_size());
    for (const auto value : attribute.ints()) tensor.add_int64_data(value);
    return Status::OK();
  }
  if (name == "value_string" && attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_STRING) {
    tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
    tensor.add_string_data(attribute.s());
    return Status::OK();
  }
  if (name == "value_strings" && attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS) {
    tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_STRING);
    tensor.add_dims(attribute.strings_size());
    for (const auto& value : attribute.strings()) tensor.add_string_data(value);
    return Status::OK();
  }
  return InvalidPattern("unsupported Constant value attribute");
}

Status CompareTensorLiterals(
    const ONNX_NAMESPACE::TensorProto& lhs,
    const ONNX_NAMESPACE::TensorProto& rhs,
    size_t max_literal_bytes,
    bool& equal,
    const std::filesystem::path* rhs_model_path) {
  equal = false;
  if (lhs.data_type() != rhs.data_type() || lhs.dims_size() != rhs.dims_size()) return Status::OK();
  for (int i = 0; i < lhs.dims_size(); ++i) {
    if (lhs.dims(i) != rhs.dims(i)) return Status::OK();
  }
  std::string lhs_bytes;
  std::string rhs_bytes;
  ORT_RETURN_IF_ERROR(TensorLogicalBytes(lhs, max_literal_bytes, nullptr, lhs_bytes));
  ORT_RETURN_IF_ERROR(TensorLogicalBytes(rhs, max_literal_bytes, rhs_model_path, rhs_bytes));
  equal = lhs_bytes == rhs_bytes;
  return Status::OK();
}

std::string CanonicalFunctionFingerprint(const ONNX_NAMESPACE::FunctionProto& function_proto) {
  ONNX_NAMESPACE::FunctionProto canonical = function_proto;
  canonical.clear_doc_string();
  for (auto& value_info : *canonical.mutable_value_info()) {
    value_info.clear_doc_string();
  }
  for (auto& node : *canonical.mutable_node()) {
    node.clear_name();
    node.clear_doc_string();
    std::vector<ONNX_NAMESPACE::AttributeProto> attributes(node.attribute().begin(), node.attribute().end());
    std::sort(attributes.begin(), attributes.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.name() < rhs.name(); });
    node.clear_attribute();
    for (auto& attribute : attributes) {
      attribute.clear_doc_string();
      *node.add_attribute() = std::move(attribute);
    }
  }
  std::vector<ONNX_NAMESPACE::OperatorSetIdProto> imports(
      canonical.opset_import().begin(), canonical.opset_import().end());
  std::sort(imports.begin(), imports.end(), [](const auto& lhs, const auto& rhs) {
    return CanonicalDomain(lhs.domain()) < CanonicalDomain(rhs.domain());
  });
  canonical.clear_opset_import();
  for (auto& import : imports) {
    import.set_domain(CanonicalDomain(import.domain()));
    *canonical.add_opset_import() = std::move(import);
  }
  return canonical.SerializeAsString();
}

}  // namespace function_extractor_internal
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
