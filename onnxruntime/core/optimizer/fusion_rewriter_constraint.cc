#include "core/optimizer/fusion_rewriter_constraint.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>

namespace onnxruntime {
namespace fusion_rewriter_internal {
namespace {

using common::Status;
using function_extractor_internal::CanonicalizeFormalAttribute;
using function_extractor_internal::CompareFormalAttributes;
using function_extractor_internal::CompareTensorLiterals;
using function_extractor_internal::kMissingPatternValue;
using function_extractor_internal::kNoPatternNode;
using function_extractor_internal::NormalizeConstantAttributes;

constexpr uint8_t kNodeIndexField = 1U << 0;
constexpr uint8_t kNodeDomainField = 1U << 1;
constexpr uint8_t kNodeOpTypeField = 1U << 2;
constexpr uint8_t kNodeOverloadField = 1U << 3;
constexpr uint8_t kNodeVersionField = 1U << 4;
constexpr uint8_t kValueNameField = 1U << 0;
constexpr uint8_t kValueExistsField = 1U << 1;
constexpr uint8_t kTypeKindField = 1U << 0;
constexpr uint8_t kTypeElementField = 1U << 1;
constexpr uint8_t kTypeFullField = 1U << 2;

Status InvalidConstraint(std::string_view message) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Invalid fusion constraint program: ", message);
}

bool IsKnownConstraintKind(ConstraintKind kind) {
  return kind >= ConstraintKind::kAllOf &&
         kind <= ConstraintKind::kSameAttributeValue;
}

bool IsKnownUnknownPolicy(FusionUnknownPolicy policy) {
  return policy == FusionUnknownPolicy::kReject ||
         policy == FusionUnknownPolicy::kNotContradicted;
}

bool IsCompatibilityConstraint(ConstraintKind kind) {
  switch (kind) {
    case ConstraintKind::kTypeEquals:
    case ConstraintKind::kSameElementType:
    case ConstraintKind::kSameRank:
    case ConstraintKind::kDimEquals:
    case ConstraintKind::kShapeEquals:
    case ConstraintKind::kSameAttributeValue:
      return true;
    default:
      return false;
  }
}

bool IsSupportedAttributeType(
    ONNX_NAMESPACE::AttributeProto_AttributeType type) {
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

Status ValidateRawExpr(const std::shared_ptr<const ConstraintExpr>& expr) {
  ORT_RETURN_IF_NOT(expr != nullptr, "Constraint expression is null.");
  ORT_RETURN_IF_NOT(IsKnownConstraintKind(expr->kind),
                    "Constraint expression has an unknown kind.");
  ORT_RETURN_IF_NOT(IsKnownUnknownPolicy(expr->unknown_policy),
                    "Constraint expression has an unknown policy.");
  ORT_RETURN_IF(expr->unknown_policy == FusionUnknownPolicy::kNotContradicted &&
                    !IsCompatibilityConstraint(expr->kind),
                "NotContradicted is legal only for compatibility predicates.");

  switch (expr->kind) {
    case ConstraintKind::kAllOf:
    case ConstraintKind::kAnyOf:
      for (const auto& operand : expr->operands) {
        ORT_RETURN_IF_ERROR(ValidateRawExpr(operand));
      }
      break;
    case ConstraintKind::kNot:
      ORT_RETURN_IF_NOT(expr->operands.size() == 1,
                        "Not requires exactly one operand.");
      ORT_RETURN_IF_ERROR(ValidateRawExpr(expr->operands.front()));
      break;
    case ConstraintKind::kRankIn:
      ORT_RETURN_IF(expr->minimum_rank > expr->maximum_rank,
                    "RankIn minimum exceeds maximum.");
      break;
    case ConstraintKind::kElementTypeIs:
      ORT_RETURN_IF(
          expr->element_type ==
                  ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED ||
              !ONNX_NAMESPACE::TensorProto_DataType_IsValid(
                  expr->element_type),
          "ElementTypeIs has an invalid tensor element type.");
      break;
    case ConstraintKind::kElementTypeIn:
      ORT_RETURN_IF(expr->element_types.empty(),
                    "ElementTypeIn requires at least one element type.");
      for (const int32_t element_type : expr->element_types) {
        ORT_RETURN_IF(
            element_type ==
                    ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED ||
                !ONNX_NAMESPACE::TensorProto_DataType_IsValid(
                    element_type),
            "ElementTypeIn has an invalid tensor element type.");
      }
      break;
    case ConstraintKind::kIntAttributeInRange:
      ORT_RETURN_IF(expr->minimum_integer > expr->maximum_integer,
                    "Integer attribute minimum exceeds maximum.");
      break;
    case ConstraintKind::kFloatAttributeInRange:
      ORT_RETURN_IF(std::isnan(expr->minimum_float) ||
                        std::isnan(expr->maximum_float) ||
                        expr->minimum_float > expr->maximum_float,
                    "Float attribute range is invalid.");
      break;
    case ConstraintKind::kAttributeTypeIs:
      ORT_RETURN_IF_NOT(IsSupportedAttributeType(expr->attribute_type),
                        "Unsupported attribute type.");
      break;
    case ConstraintKind::kAttributeEquals:
      ORT_RETURN_IF_NOT(expr->attribute_literals.size() == 1,
                        "AttributeEquals requires exactly one literal.");
      break;
    default:
      ORT_RETURN_IF_NOT(expr->operands.empty(),
                        "Leaf constraints cannot contain operands.");
      break;
  }
  return Status::OK();
}

Status ValidateRawDefinition(ConstraintProgramDefinition& definition) {
  ORT_RETURN_IF_ERROR(ValidateRawExpr(definition.predicate));
  for (const auto& dimension_class : definition.dimension_classes) {
    ORT_RETURN_IF_NOT(IsKnownUnknownPolicy(dimension_class.unknown_policy),
                      "Dimension equivalence class has an unknown policy.");
    ORT_RETURN_IF(dimension_class.dimensions.size() < 2,
                  "Dimension equivalence classes require at least two operands.");
  }
  return Status::OK();
}

std::shared_ptr<const ConstraintExpr> MakeExpr(ConstraintKind kind) {
  auto expr = std::make_shared<ConstraintExpr>();
  expr->kind = kind;
  return expr;
}

Status ResolveValueRef(
    const FusionValueRef& source,
    const function_extractor_internal::NormalizedFunctionPattern& pattern,
    ResolvedValueRef& resolved) {
  switch (source.kind) {
    case FusionValueRefKind::kFormalInput:
      ORT_RETURN_IF_NOT(source.index < pattern.formal_input_value_ids.size(),
                        "Formal input reference is out of range.");
      resolved.pattern_value = pattern.formal_input_value_ids[source.index];
      break;
    case FusionValueRefKind::kFormalOutput:
      ORT_RETURN_IF_NOT(source.index < pattern.formal_output_value_ids.size(),
                        "Formal output reference is out of range.");
      resolved.pattern_value = pattern.formal_output_value_ids[source.index];
      break;
    case FusionValueRefKind::kPatternValue:
      ORT_RETURN_IF_NOT(source.index < pattern.values.size(),
                        "Pattern value reference is out of range.");
      resolved.pattern_value = source.index;
      break;
    default:
      return InvalidConstraint("Value reference has an unknown kind.");
  }
  ORT_RETURN_IF(resolved.pattern_value == kMissingPatternValue,
                "Constraint references an unsupported missing pattern value.");
  return Status::OK();
}

Status ResolveDimRef(
    const FusionDimRef& source,
    const function_extractor_internal::NormalizedFunctionPattern& pattern,
    ResolvedDimRef& resolved) {
  ORT_RETURN_IF_ERROR(ResolveValueRef(source.value, pattern, resolved.value));
  resolved.axis = source.axis;
  const auto& pattern_value = pattern.values[resolved.value.pattern_value];
  if (pattern_value.has_type && pattern_value.type.has_tensor_type() &&
      pattern_value.type.tensor_type().has_shape()) {
    const int64_t rank =
        pattern_value.type.tensor_type().shape().dim_size();
    const int64_t normalized =
        source.axis < 0 ? source.axis + rank : source.axis;
    ORT_RETURN_IF(normalized < 0 || normalized >= rank,
                  "Dimension axis is outside the pattern-declared rank.");
    resolved.axis = normalized;
  }
  return Status::OK();
}

Status ResolveAttributeRef(
    const FusionAttributeRef& source,
    const function_extractor_internal::NormalizedFunctionPattern& pattern,
    ResolvedAttributeRef& resolved) {
  resolved.kind = source.kind;
  switch (source.kind) {
    case FusionAttributeRefKind::kFormalAttribute:
      ORT_RETURN_IF_NOT(
          source.formal_attribute_id < pattern.formal_attributes.size(),
          "Formal attribute reference is out of range.");
      resolved.formal_attribute = source.formal_attribute_id;
      break;
    case FusionAttributeRefKind::kEffectiveNodeAttribute:
      ORT_RETURN_IF_NOT(source.node.id < pattern.nodes.size(),
                        "Pattern node reference is out of range.");
      ORT_RETURN_IF(source.operator_attribute_name.empty(),
                    "Effective attribute name is empty.");
      resolved.pattern_node = source.node.id;
      resolved.operator_attribute_name = source.operator_attribute_name;
      break;
    default:
      return InvalidConstraint("Attribute reference has an unknown kind.");
  }
  return Status::OK();
}

Status CanonicalizeConstraintLiteral(
    const ONNX_NAMESPACE::AttributeProto& source, size_t max_attribute_bytes,
    ONNX_NAMESPACE::AttributeProto& canonical) {
  ORT_RETURN_IF_NOT(IsSupportedAttributeType(source.type()),
                    "Constraint attribute literal has an unsupported type.");
  return CanonicalizeFormalAttribute(
      "", source.type(), source, max_attribute_bytes, canonical);
}

Status CompileExpr(
    const std::shared_ptr<const ConstraintExpr>& source,
    const function_extractor_internal::NormalizedFunctionPattern& pattern,
    const ConstraintCompileOptions& options, size_t depth,
    size_t& node_count, FusionConstraintId& next_id,
    std::shared_ptr<const CompiledConstraintNode>& destination) {
  ORT_RETURN_IF(node_count >= options.max_constraint_nodes,
                "Fusion constraint node budget exceeded.");
  ORT_RETURN_IF(depth >= options.max_constraint_nodes,
                "Fusion constraint depth budget exceeded.");
  ++node_count;

  auto node = std::make_shared<CompiledConstraintNode>();
  node->kind = source->kind;
  node->id = next_id++;
  node->unknown_policy = source->unknown_policy;
  node->minimum_rank = source->minimum_rank;
  node->maximum_rank = source->maximum_rank;
  node->element_type = source->element_type;
  node->element_types = source->element_types;
  node->integer_value = source->integer_value;
  node->minimum_integer = source->minimum_integer;
  node->maximum_integer = source->maximum_integer;
  node->minimum_float = source->minimum_float;
  node->maximum_float = source->maximum_float;
  node->attribute_type = source->attribute_type;
  node->string_values = source->string_values;

  switch (source->kind) {
    case ConstraintKind::kAllOf:
    case ConstraintKind::kAnyOf:
    case ConstraintKind::kNot:
      for (const auto& operand : source->operands) {
        std::shared_ptr<const CompiledConstraintNode> compiled_operand;
        ORT_RETURN_IF_ERROR(CompileExpr(
            operand, pattern, options, depth + 1, node_count, next_id,
            compiled_operand));
        node->operands.push_back(std::move(compiled_operand));
      }
      break;
    case ConstraintKind::kIsPresent:
    case ConstraintKind::kIsMissing:
    case ConstraintKind::kIsTensor:
    case ConstraintKind::kElementTypeIs:
    case ConstraintKind::kElementTypeIn:
    case ConstraintKind::kRankIs:
    case ConstraintKind::kRankIn:
      ORT_RETURN_IF_ERROR(
          ResolveValueRef(source->lhs_value, pattern, node->lhs_value));
      break;
    case ConstraintKind::kTypeEquals:
    case ConstraintKind::kSameElementType:
    case ConstraintKind::kSameRank:
    case ConstraintKind::kShapeEquals:
      ORT_RETURN_IF_ERROR(
          ResolveValueRef(source->lhs_value, pattern, node->lhs_value));
      ORT_RETURN_IF_ERROR(
          ResolveValueRef(source->rhs_value, pattern, node->rhs_value));
      break;
    case ConstraintKind::kDimValueIs:
      ORT_RETURN_IF_ERROR(
          ResolveDimRef(source->lhs_dim, pattern, node->lhs_dim));
      break;
    case ConstraintKind::kDimEquals:
      ORT_RETURN_IF_ERROR(
          ResolveDimRef(source->lhs_dim, pattern, node->lhs_dim));
      ORT_RETURN_IF_ERROR(
          ResolveDimRef(source->rhs_dim, pattern, node->rhs_dim));
      break;
    case ConstraintKind::kAttributePresent:
    case ConstraintKind::kAttributeTypeIs:
    case ConstraintKind::kAttributeEquals:
    case ConstraintKind::kAttributeIn:
    case ConstraintKind::kIntAttributeInRange:
    case ConstraintKind::kFloatAttributeInRange:
    case ConstraintKind::kStringAttributeIn:
      ORT_RETURN_IF_ERROR(ResolveAttributeRef(
          source->lhs_attribute, pattern, node->lhs_attribute));
      break;
    case ConstraintKind::kSameAttributeValue:
      ORT_RETURN_IF_ERROR(ResolveAttributeRef(
          source->lhs_attribute, pattern, node->lhs_attribute));
      ORT_RETURN_IF_ERROR(ResolveAttributeRef(
          source->rhs_attribute, pattern, node->rhs_attribute));
      break;
  }

  for (const auto& literal : source->attribute_literals) {
    ONNX_NAMESPACE::AttributeProto canonical;
    ORT_RETURN_IF_ERROR(CanonicalizeConstraintLiteral(
        literal, options.max_attribute_bytes, canonical));
    node->attribute_literals.push_back(std::move(canonical));
  }

  std::sort(node->element_types.begin(), node->element_types.end());
  node->element_types.erase(
      std::unique(node->element_types.begin(), node->element_types.end()),
      node->element_types.end());
  std::sort(node->string_values.begin(), node->string_values.end());
  node->string_values.erase(
      std::unique(node->string_values.begin(), node->string_values.end()),
      node->string_values.end());
  destination = std::move(node);
  return Status::OK();
}

enum class TruthValue : uint8_t {
  kFalse,
  kTrue,
  kUnknown,
};

bool ApplyUnknownPolicy(TruthValue value, FusionUnknownPolicy policy) {
  if (value == TruthValue::kTrue) return true;
  if (value == TruthValue::kFalse) return false;
  return policy == FusionUnknownPolicy::kNotContradicted;
}

TruthValue MergeEquality(TruthValue current, TruthValue next) {
  if (current == TruthValue::kFalse || next == TruthValue::kFalse) {
    return TruthValue::kFalse;
  }
  if (current == TruthValue::kUnknown || next == TruthValue::kUnknown) {
    return TruthValue::kUnknown;
  }
  return TruthValue::kTrue;
}

TruthValue CompareDimensions(const DimensionFact& lhs,
                             const DimensionFact& rhs) {
  if (lhs.kind == DimensionFactKind::kValue &&
      rhs.kind == DimensionFactKind::kValue) {
    return lhs.value == rhs.value ? TruthValue::kTrue : TruthValue::kFalse;
  }
  if (lhs.kind == DimensionFactKind::kSymbol &&
      rhs.kind == DimensionFactKind::kSymbol &&
      lhs.symbol == rhs.symbol && !lhs.symbol.empty()) {
    return TruthValue::kTrue;
  }
  return TruthValue::kUnknown;
}

TruthValue CompareDimensionClass(gsl::span<const DimensionFact> facts) {
  if (facts.size() < 2) {
    return TruthValue::kTrue;
  }
  std::optional<int64_t> concrete_value;
  std::optional<std::string_view> symbol;
  bool has_unknown_comparison = false;
  for (const auto& fact : facts) {
    switch (fact.kind) {
      case DimensionFactKind::kUnknown:
        has_unknown_comparison = true;
        break;
      case DimensionFactKind::kValue:
        if (concrete_value.has_value() &&
            *concrete_value != fact.value) {
          return TruthValue::kFalse;
        }
        concrete_value = fact.value;
        if (symbol.has_value()) {
          has_unknown_comparison = true;
        }
        break;
      case DimensionFactKind::kSymbol:
        if (fact.symbol.empty()) {
          has_unknown_comparison = true;
          break;
        }
        if (symbol.has_value() && *symbol != fact.symbol) {
          has_unknown_comparison = true;
        } else if (!symbol.has_value()) {
          symbol = fact.symbol;
        }
        if (concrete_value.has_value()) {
          has_unknown_comparison = true;
        }
        break;
    }
  }
  return has_unknown_comparison ? TruthValue::kUnknown
                                : TruthValue::kTrue;
}

DimensionFact DimensionFromProto(
    const ONNX_NAMESPACE::TensorShapeProto_Dimension& dimension) {
  DimensionFact result;
  if (dimension.has_dim_value()) {
    result.kind = DimensionFactKind::kValue;
    result.value = dimension.dim_value();
  } else if (dimension.has_dim_param() && !dimension.dim_param().empty()) {
    result.kind = DimensionFactKind::kSymbol;
    result.symbol = dimension.dim_param();
  }
  return result;
}

TruthValue CompareShapes(const ONNX_NAMESPACE::TensorShapeProto* lhs,
                         const ONNX_NAMESPACE::TensorShapeProto* rhs) {
  if (lhs == nullptr || rhs == nullptr) return TruthValue::kUnknown;
  if (lhs->dim_size() != rhs->dim_size()) return TruthValue::kFalse;
  TruthValue result = TruthValue::kTrue;
  for (int i = 0; i < lhs->dim_size(); ++i) {
    result = MergeEquality(
        result,
        CompareDimensions(DimensionFromProto(lhs->dim(i)),
                          DimensionFromProto(rhs->dim(i))));
    if (result == TruthValue::kFalse) break;
  }
  return result;
}

template <typename TensorType>
TruthValue CompareTensorTypes(const TensorType& lhs, const TensorType& rhs) {
  if (lhs.elem_type() != 0 && rhs.elem_type() != 0 &&
      lhs.elem_type() != rhs.elem_type()) {
    return TruthValue::kFalse;
  }
  TruthValue result =
      lhs.elem_type() != 0 && rhs.elem_type() != 0
          ? TruthValue::kTrue
          : TruthValue::kUnknown;
  const auto shape_result = CompareShapes(
      lhs.has_shape() ? &lhs.shape() : nullptr,
      rhs.has_shape() ? &rhs.shape() : nullptr);
  return MergeEquality(result, shape_result);
}

TruthValue CompareTypes(const ONNX_NAMESPACE::TypeProto* lhs,
                        const ONNX_NAMESPACE::TypeProto* rhs) {
  if (lhs == nullptr || rhs == nullptr ||
      lhs->value_case() == ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET ||
      rhs->value_case() == ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET) {
    return TruthValue::kUnknown;
  }
  if (lhs->value_case() != rhs->value_case()) return TruthValue::kFalse;
  switch (lhs->value_case()) {
    case ONNX_NAMESPACE::TypeProto::kTensorType:
      return CompareTensorTypes(lhs->tensor_type(), rhs->tensor_type());
    case ONNX_NAMESPACE::TypeProto::kSparseTensorType:
      return CompareTensorTypes(lhs->sparse_tensor_type(),
                                rhs->sparse_tensor_type());
    case ONNX_NAMESPACE::TypeProto::kSequenceType:
      if (!lhs->sequence_type().has_elem_type() ||
          !rhs->sequence_type().has_elem_type()) {
        return TruthValue::kUnknown;
      }
      return CompareTypes(&lhs->sequence_type().elem_type(),
                          &rhs->sequence_type().elem_type());
    case ONNX_NAMESPACE::TypeProto::kOptionalType:
      if (!lhs->optional_type().has_elem_type() ||
          !rhs->optional_type().has_elem_type()) {
        return TruthValue::kUnknown;
      }
      return CompareTypes(&lhs->optional_type().elem_type(),
                          &rhs->optional_type().elem_type());
    case ONNX_NAMESPACE::TypeProto::kMapType: {
      const auto& lhs_map = lhs->map_type();
      const auto& rhs_map = rhs->map_type();
      if (lhs_map.key_type() != 0 && rhs_map.key_type() != 0 &&
          lhs_map.key_type() != rhs_map.key_type()) {
        return TruthValue::kFalse;
      }
      TruthValue result =
          lhs_map.key_type() != 0 && rhs_map.key_type() != 0
              ? TruthValue::kTrue
              : TruthValue::kUnknown;
      if (!lhs_map.has_value_type() || !rhs_map.has_value_type()) {
        return TruthValue::kUnknown;
      }
      return MergeEquality(
          result,
          CompareTypes(&lhs_map.value_type(), &rhs_map.value_type()));
    }
    default:
      return TruthValue::kUnknown;
  }
}

const ONNX_NAMESPACE::TensorShapeProto* TensorShape(const NodeArg* value) {
  if (value == nullptr) return nullptr;
  const auto* type = value->TypeAsProto();
  if (type == nullptr || !type->has_tensor_type() ||
      !type->tensor_type().has_shape()) {
    return nullptr;
  }
  return &type->tensor_type().shape();
}

const ONNX_NAMESPACE::AttributeProto* EffectiveTargetAttribute(
    const Node& node, std::string_view name) {
  const auto explicit_attribute =
      node.GetAttributes().find(std::string{name});
  if (explicit_attribute != node.GetAttributes().end()) {
    return &explicit_attribute->second;
  }
  if (node.Op() == nullptr) return nullptr;
  const auto schema_attribute =
      node.Op()->attributes().find(std::string{name});
  if (schema_attribute == node.Op()->attributes().end() ||
      schema_attribute->second.default_value.name().empty()) {
    return nullptr;
  }
  return &schema_attribute->second.default_value;
}

Status CanonicalizeObservedAttribute(
    std::string_view name, const ONNX_NAMESPACE::AttributeProto& source,
    size_t max_attribute_bytes,
    ONNX_NAMESPACE::AttributeProto& canonical) {
  return CanonicalizeFormalAttribute(name, source.type(), source,
                                     max_attribute_bytes, canonical);
}

Status AttributesEqual(const ONNX_NAMESPACE::AttributeProto& lhs,
                       const ONNX_NAMESPACE::AttributeProto& rhs,
                       size_t max_attribute_bytes, bool& equal) {
  return CompareFormalAttributes(lhs, rhs, max_attribute_bytes, equal);
}

const NodeArg* ValueAtBindingSites(
    const Graph& graph, gsl::span<const ValueBindingSite> binding_sites,
    bool& consistent) {
  consistent = true;
  const NodeArg* value = nullptr;
  bool initialized = false;
  for (const auto& site : binding_sites) {
    const auto* node = graph.GetNode(site.node_index);
    if (node == nullptr) {
      consistent = false;
      return nullptr;
    }
    const auto& definitions =
        site.is_output ? node->OutputDefs() : node->InputDefs();
    const auto* current =
        site.slot < definitions.size() ? definitions[site.slot] : nullptr;
    if (!initialized) {
      value = current;
      initialized = true;
    } else if (value != current) {
      const bool same_missing =
          (value == nullptr || !value->Exists()) &&
          (current == nullptr || !current->Exists());
      const bool same_name =
          value != nullptr && current != nullptr &&
          value->Exists() && current->Exists() &&
          value->Name() == current->Name();
      if (!same_missing && !same_name) {
        consistent = false;
        return nullptr;
      }
    }
  }
  return value;
}

bool DimensionFactsEqual(const DimensionFact& lhs,
                         const DimensionFact& rhs) {
  return lhs.kind == rhs.kind && lhs.value == rhs.value &&
         lhs.symbol == rhs.symbol;
}

bool DependencyLess(const DependencySnapshot& lhs,
                    const DependencySnapshot& rhs) {
  return std::tie(lhs.kind, lhs.pattern_node, lhs.pattern_value,
                  lhs.formal_attribute, lhs.name, lhs.axis) <
         std::tie(rhs.kind, rhs.pattern_node, rhs.pattern_value,
                  rhs.formal_attribute, rhs.name, rhs.axis);
}

struct AttributeValue {
  bool exists{};
  ONNX_NAMESPACE::AttributeProto canonical;
};

Status ReadAttribute(const ResolvedAttributeRef& reference,
                     DependencyRecorder& recorder, AttributeValue& value) {
  value = {};
  if (reference.kind == FusionAttributeRefKind::kFormalAttribute) {
    recorder.RecordFormalAttribute(reference.formal_attribute);
    const auto* attribute =
        recorder.FormalAttribute(reference.formal_attribute);
    if (attribute != nullptr) {
      value.exists = true;
      value.canonical = *attribute;
    }
  } else {
    recorder.RecordEffectiveAttribute(reference.pattern_node,
                                      reference.operator_attribute_name);
    ONNX_NAMESPACE::AttributeProto canonical;
    const auto* attribute = recorder.EffectiveAttribute(
        reference.pattern_node, reference.operator_attribute_name, canonical);
    if (attribute != nullptr) {
      value.exists = true;
      value.canonical = std::move(canonical);
    }
  }
  return recorder.Status();
}

TruthValue ElementTypesEqual(const NodeArg* lhs, const NodeArg* rhs) {
  if (lhs == nullptr || rhs == nullptr || !lhs->Exists() || !rhs->Exists()) {
    return TruthValue::kFalse;
  }
  const auto* lhs_type = lhs->TypeAsProto();
  const auto* rhs_type = rhs->TypeAsProto();
  if (lhs_type == nullptr || rhs_type == nullptr) return TruthValue::kUnknown;
  if (!lhs_type->has_tensor_type() || !rhs_type->has_tensor_type()) {
    return TruthValue::kFalse;
  }
  const int32_t lhs_element = lhs_type->tensor_type().elem_type();
  const int32_t rhs_element = rhs_type->tensor_type().elem_type();
  if (lhs_element == 0 || rhs_element == 0) return TruthValue::kUnknown;
  return lhs_element == rhs_element ? TruthValue::kTrue
                                    : TruthValue::kFalse;
}

TruthValue RanksEqual(const NodeArg* lhs, const NodeArg* rhs) {
  if (lhs == nullptr || rhs == nullptr || !lhs->Exists() || !rhs->Exists()) {
    return TruthValue::kFalse;
  }
  const auto* lhs_type = lhs->TypeAsProto();
  const auto* rhs_type = rhs->TypeAsProto();
  if ((lhs_type != nullptr && !lhs_type->has_tensor_type()) ||
      (rhs_type != nullptr && !rhs_type->has_tensor_type())) {
    return TruthValue::kFalse;
  }
  const auto* lhs_shape = TensorShape(lhs);
  const auto* rhs_shape = TensorShape(rhs);
  if (lhs_shape == nullptr || rhs_shape == nullptr) {
    return TruthValue::kUnknown;
  }
  return lhs_shape->dim_size() == rhs_shape->dim_size()
             ? TruthValue::kTrue
             : TruthValue::kFalse;
}

Status EvaluateNode(const CompiledConstraintNode& node,
                    DependencyRecorder& recorder, bool& value,
                    ConstraintEvaluationResult& result);

Status EvaluateLeaf(const CompiledConstraintNode& node,
                    DependencyRecorder& recorder, TruthValue& raw) {
  raw = TruthValue::kFalse;
  const auto read_value = [&](ResolvedValueRef reference,
                              uint8_t identity_fields) {
    recorder.RecordValueIdentity(reference.pattern_value, identity_fields);
    return recorder.TargetValue(reference.pattern_value);
  };

  switch (node.kind) {
    case ConstraintKind::kIsPresent: {
      const auto* target =
          read_value(node.lhs_value, kValueExistsField);
      raw = target != nullptr && target->Exists() ? TruthValue::kTrue
                                                  : TruthValue::kFalse;
      break;
    }
    case ConstraintKind::kIsMissing: {
      const auto* target =
          read_value(node.lhs_value, kValueExistsField);
      raw = target == nullptr || !target->Exists() ? TruthValue::kTrue
                                                   : TruthValue::kFalse;
      break;
    }
    case ConstraintKind::kIsTensor: {
      const auto* target =
          read_value(node.lhs_value, kValueExistsField);
      recorder.RecordValueType(node.lhs_value.pattern_value, kTypeKindField);
      if (target == nullptr || !target->Exists()) {
        raw = TruthValue::kFalse;
      } else if (target->TypeAsProto() == nullptr) {
        raw = TruthValue::kUnknown;
      } else {
        raw = target->TypeAsProto()->has_tensor_type()
                  ? TruthValue::kTrue
                  : TruthValue::kFalse;
      }
      break;
    }
    case ConstraintKind::kElementTypeIs:
    case ConstraintKind::kElementTypeIn: {
      const auto* target =
          read_value(node.lhs_value, kValueExistsField);
      recorder.RecordValueType(node.lhs_value.pattern_value,
                               kTypeKindField | kTypeElementField);
      if (target == nullptr || !target->Exists()) {
        raw = TruthValue::kFalse;
        break;
      }
      const auto* type = target->TypeAsProto();
      if (type == nullptr) {
        raw = TruthValue::kUnknown;
      } else if (!type->has_tensor_type()) {
        raw = TruthValue::kFalse;
      } else if (type->tensor_type().elem_type() == 0) {
        raw = TruthValue::kUnknown;
      } else if (node.kind == ConstraintKind::kElementTypeIs) {
        raw = type->tensor_type().elem_type() == node.element_type
                  ? TruthValue::kTrue
                  : TruthValue::kFalse;
      } else {
        raw = std::binary_search(
                  node.element_types.begin(), node.element_types.end(),
                  type->tensor_type().elem_type())
                  ? TruthValue::kTrue
                  : TruthValue::kFalse;
      }
      break;
    }
    case ConstraintKind::kTypeEquals: {
      const auto* lhs =
          read_value(node.lhs_value, kValueExistsField);
      const auto* rhs =
          read_value(node.rhs_value, kValueExistsField);
      recorder.RecordValueType(node.lhs_value.pattern_value,
                               kTypeKindField | kTypeElementField |
                                   kTypeFullField);
      recorder.RecordValueType(node.rhs_value.pattern_value,
                               kTypeKindField | kTypeElementField |
                                   kTypeFullField);
      raw = lhs == nullptr || rhs == nullptr || !lhs->Exists() || !rhs->Exists()
                ? TruthValue::kFalse
                : CompareTypes(lhs->TypeAsProto(), rhs->TypeAsProto());
      if (raw != TruthValue::kFalse) {
        recorder.RecordValueRank(node.lhs_value.pattern_value);
        recorder.RecordValueRank(node.rhs_value.pattern_value);
        const auto* lhs_shape = TensorShape(lhs);
        const auto* rhs_shape = TensorShape(rhs);
        if (lhs_shape != nullptr && rhs_shape != nullptr &&
            lhs_shape->dim_size() == rhs_shape->dim_size()) {
          for (int64_t axis = 0; axis < lhs_shape->dim_size(); ++axis) {
            recorder.RecordValueDimension(node.lhs_value.pattern_value, axis);
            recorder.RecordValueDimension(node.rhs_value.pattern_value, axis);
          }
        }
      }
      break;
    }
    case ConstraintKind::kSameElementType: {
      const auto* lhs =
          read_value(node.lhs_value, kValueExistsField);
      const auto* rhs =
          read_value(node.rhs_value, kValueExistsField);
      recorder.RecordValueType(node.lhs_value.pattern_value,
                               kTypeKindField | kTypeElementField);
      recorder.RecordValueType(node.rhs_value.pattern_value,
                               kTypeKindField | kTypeElementField);
      raw = ElementTypesEqual(lhs, rhs);
      break;
    }
    case ConstraintKind::kRankIs:
    case ConstraintKind::kRankIn: {
      const auto* target =
          read_value(node.lhs_value, kValueExistsField);
      recorder.RecordValueRank(node.lhs_value.pattern_value);
      if (target == nullptr || !target->Exists()) {
        raw = TruthValue::kFalse;
      } else if (target->TypeAsProto() != nullptr &&
                 !target->TypeAsProto()->has_tensor_type()) {
        raw = TruthValue::kFalse;
      } else if (TensorShape(target) == nullptr) {
        raw = TruthValue::kUnknown;
      } else {
        const size_t rank =
            static_cast<size_t>(TensorShape(target)->dim_size());
        raw = node.kind == ConstraintKind::kRankIs
                  ? (rank == node.minimum_rank ? TruthValue::kTrue
                                               : TruthValue::kFalse)
                  : (rank >= node.minimum_rank && rank <= node.maximum_rank
                         ? TruthValue::kTrue
                         : TruthValue::kFalse);
      }
      break;
    }
    case ConstraintKind::kSameRank: {
      const auto* lhs =
          read_value(node.lhs_value, kValueExistsField);
      const auto* rhs =
          read_value(node.rhs_value, kValueExistsField);
      recorder.RecordValueRank(node.lhs_value.pattern_value);
      recorder.RecordValueRank(node.rhs_value.pattern_value);
      raw = RanksEqual(lhs, rhs);
      break;
    }
    case ConstraintKind::kDimValueIs: {
      recorder.RecordValueIdentity(node.lhs_dim.value.pattern_value,
                                   kValueExistsField);
      bool rank_known = false;
      int64_t normalized_axis = -1;
      const DimensionFact fact = recorder.ReadDimension(
          node.lhs_dim.value.pattern_value, node.lhs_dim.axis, rank_known,
          normalized_axis);
      if (!rank_known) {
        raw = TruthValue::kUnknown;
      } else if (normalized_axis < 0) {
        raw = TruthValue::kFalse;
      } else if (fact.kind == DimensionFactKind::kValue) {
        raw = fact.value == node.integer_value ? TruthValue::kTrue
                                               : TruthValue::kFalse;
      } else {
        raw = TruthValue::kUnknown;
      }
      break;
    }
    case ConstraintKind::kDimEquals: {
      recorder.RecordValueIdentity(node.lhs_dim.value.pattern_value,
                                   kValueExistsField);
      recorder.RecordValueIdentity(node.rhs_dim.value.pattern_value,
                                   kValueExistsField);
      bool lhs_rank_known = false;
      bool rhs_rank_known = false;
      int64_t lhs_axis = -1;
      int64_t rhs_axis = -1;
      const auto lhs = recorder.ReadDimension(
          node.lhs_dim.value.pattern_value, node.lhs_dim.axis,
          lhs_rank_known, lhs_axis);
      const auto rhs = recorder.ReadDimension(
          node.rhs_dim.value.pattern_value, node.rhs_dim.axis,
          rhs_rank_known, rhs_axis);
      if (!lhs_rank_known || !rhs_rank_known) {
        raw = TruthValue::kUnknown;
      } else if (lhs_axis < 0 || rhs_axis < 0) {
        raw = TruthValue::kFalse;
      } else {
        raw = CompareDimensions(lhs, rhs);
      }
      break;
    }
    case ConstraintKind::kShapeEquals: {
      const auto* lhs =
          read_value(node.lhs_value, kValueExistsField);
      const auto* rhs =
          read_value(node.rhs_value, kValueExistsField);
      recorder.RecordValueRank(node.lhs_value.pattern_value);
      recorder.RecordValueRank(node.rhs_value.pattern_value);
      raw = lhs == nullptr || rhs == nullptr || !lhs->Exists() || !rhs->Exists()
                ? TruthValue::kFalse
                : CompareShapes(TensorShape(lhs), TensorShape(rhs));
      const auto* lhs_shape = TensorShape(lhs);
      const auto* rhs_shape = TensorShape(rhs);
      if (lhs_shape != nullptr && rhs_shape != nullptr &&
          lhs_shape->dim_size() == rhs_shape->dim_size()) {
        for (int64_t axis = 0; axis < lhs_shape->dim_size(); ++axis) {
          recorder.RecordValueDimension(node.lhs_value.pattern_value, axis);
          recorder.RecordValueDimension(node.rhs_value.pattern_value, axis);
        }
      }
      break;
    }
    case ConstraintKind::kAttributePresent:
    case ConstraintKind::kAttributeTypeIs:
    case ConstraintKind::kAttributeEquals:
    case ConstraintKind::kAttributeIn:
    case ConstraintKind::kIntAttributeInRange:
    case ConstraintKind::kFloatAttributeInRange:
    case ConstraintKind::kStringAttributeIn: {
      AttributeValue attribute;
      ORT_RETURN_IF_ERROR(
          ReadAttribute(node.lhs_attribute, recorder, attribute));
      if (node.kind == ConstraintKind::kAttributePresent) {
        raw = attribute.exists ? TruthValue::kTrue : TruthValue::kFalse;
      } else if (!attribute.exists) {
        raw = TruthValue::kFalse;
      } else if (node.kind == ConstraintKind::kAttributeTypeIs) {
        raw = attribute.canonical.type() == node.attribute_type
                  ? TruthValue::kTrue
                  : TruthValue::kFalse;
      } else if (node.kind == ConstraintKind::kAttributeEquals ||
                 node.kind == ConstraintKind::kAttributeIn) {
        raw = TruthValue::kFalse;
        for (const auto& literal : node.attribute_literals) {
          bool equal = false;
          ORT_RETURN_IF_ERROR(AttributesEqual(
              attribute.canonical, literal,
              std::numeric_limits<size_t>::max(), equal));
          if (equal) {
            raw = TruthValue::kTrue;
            break;
          }
        }
      } else if (node.kind == ConstraintKind::kIntAttributeInRange) {
        raw = attribute.canonical.type() ==
                          ONNX_NAMESPACE::AttributeProto_AttributeType_INT &&
                      attribute.canonical.i() >= node.minimum_integer &&
                      attribute.canonical.i() <= node.maximum_integer
                  ? TruthValue::kTrue
                  : TruthValue::kFalse;
      } else if (node.kind == ConstraintKind::kFloatAttributeInRange) {
        const float observed = attribute.canonical.f();
        raw = attribute.canonical.type() ==
                          ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT &&
                      !std::isnan(observed) &&
                      observed >= node.minimum_float &&
                      observed <= node.maximum_float
                  ? TruthValue::kTrue
                  : TruthValue::kFalse;
      } else {
        raw = attribute.canonical.type() ==
                          ONNX_NAMESPACE::AttributeProto_AttributeType_STRING &&
                      std::binary_search(
                          node.string_values.begin(), node.string_values.end(),
                          attribute.canonical.s())
                  ? TruthValue::kTrue
                  : TruthValue::kFalse;
      }
      break;
    }
    case ConstraintKind::kSameAttributeValue: {
      AttributeValue lhs;
      AttributeValue rhs;
      ORT_RETURN_IF_ERROR(ReadAttribute(node.lhs_attribute, recorder, lhs));
      ORT_RETURN_IF_ERROR(ReadAttribute(node.rhs_attribute, recorder, rhs));
      if (!lhs.exists && !rhs.exists) {
        raw = TruthValue::kUnknown;
      } else if (lhs.exists != rhs.exists) {
        raw = TruthValue::kFalse;
      } else {
        bool equal = false;
        ORT_RETURN_IF_ERROR(AttributesEqual(
            lhs.canonical, rhs.canonical,
            std::numeric_limits<size_t>::max(), equal));
        raw = equal ? TruthValue::kTrue : TruthValue::kFalse;
      }
      break;
    }
    default:
      return InvalidConstraint("Boolean node reached leaf evaluator.");
  }
  return recorder.Status();
}

Status EvaluateNode(const CompiledConstraintNode& node,
                    DependencyRecorder& recorder, bool& value,
                    ConstraintEvaluationResult& result) {
  switch (node.kind) {
    case ConstraintKind::kAllOf:
      value = true;
      for (const auto& operand : node.operands) {
        bool operand_value = false;
        ORT_RETURN_IF_ERROR(
            EvaluateNode(*operand, recorder, operand_value, result));
        if (!operand_value) {
          value = false;
          return Status::OK();
        }
      }
      return Status::OK();
    case ConstraintKind::kAnyOf:
      value = false;
      for (const auto& operand : node.operands) {
        bool operand_value = false;
        ORT_RETURN_IF_ERROR(
            EvaluateNode(*operand, recorder, operand_value, result));
        if (operand_value) {
          value = true;
          result.failed_constraint.reset();
          result.detail.clear();
          return Status::OK();
        }
      }
      return Status::OK();
    case ConstraintKind::kNot: {
      bool operand_value = false;
      ORT_RETURN_IF_ERROR(EvaluateNode(
          *node.operands.front(), recorder, operand_value, result));
      value = !operand_value;
      if (value) {
        result.failed_constraint.reset();
        result.detail.clear();
      } else {
        result.failed_constraint = node.id;
        result.detail = "negated constraint was satisfied";
      }
      return Status::OK();
    }
    default: {
      TruthValue raw = TruthValue::kFalse;
      ORT_RETURN_IF_ERROR(EvaluateLeaf(node, recorder, raw));
      value = ApplyUnknownPolicy(raw, node.unknown_policy);
      if (!value) {
        result.failed_constraint = node.id;
        result.detail =
            raw == TruthValue::kUnknown ? "required fact is unknown"
                                        : "constraint evaluated false";
      }
      return Status::OK();
    }
  }
}

}  // namespace

CompiledConstraintProgram CompileConstraintProgram(
    const ConstraintProgramDefinition& definition,
    const function_extractor_internal::NormalizedFunctionPattern& pattern,
    const ConstraintCompileOptions& options) {
  CompiledConstraintProgram compiled;
  if (!definition.construction_status.IsOK()) {
    compiled.construction_status = definition.construction_status;
    return compiled;
  }
  if (!pattern.construction_status.IsOK()) {
    compiled.construction_status = pattern.construction_status;
    return compiled;
  }

  size_t dimension_operand_count = 0;
  for (const auto& source_class : definition.dimension_classes) {
    if (dimension_operand_count >
            options.max_dimension_equivalence_operands ||
        source_class.dimensions.size() >
            options.max_dimension_equivalence_operands -
                dimension_operand_count) {
      compiled.construction_status =
          InvalidConstraint("Dimension equivalence operand budget exceeded.");
      return compiled;
    }
    dimension_operand_count += source_class.dimensions.size();
    CompiledDimensionEquivalenceClass destination_class;
    destination_class.label = source_class.label;
    destination_class.unknown_policy = source_class.unknown_policy;
    for (const auto& dimension : source_class.dimensions) {
      ResolvedDimRef resolved;
      auto status = ResolveDimRef(dimension, pattern, resolved);
      if (!status.IsOK()) {
        compiled.construction_status = std::move(status);
        return compiled;
      }
      destination_class.dimensions.push_back(std::move(resolved));
    }
    std::sort(
        destination_class.dimensions.begin(),
        destination_class.dimensions.end(),
        [](const ResolvedDimRef& lhs, const ResolvedDimRef& rhs) {
          return std::tie(lhs.value.pattern_value, lhs.axis) <
                 std::tie(rhs.value.pattern_value, rhs.axis);
        });
    const auto duplicate = std::adjacent_find(
        destination_class.dimensions.begin(),
        destination_class.dimensions.end(),
        [](const ResolvedDimRef& lhs, const ResolvedDimRef& rhs) {
          return lhs.value.pattern_value == rhs.value.pattern_value &&
                 lhs.axis == rhs.axis;
        });
    if (duplicate != destination_class.dimensions.end()) {
      compiled.construction_status =
          InvalidConstraint("Dimension equivalence class contains a duplicate operand.");
      return compiled;
    }
    compiled.dimension_classes.push_back(std::move(destination_class));
  }

  size_t node_count = 0;
  FusionConstraintId next_id = 0;
  auto status = CompileExpr(definition.predicate, pattern, options, 0,
                            node_count, next_id, compiled.predicate);
  if (!status.IsOK()) {
    compiled.construction_status = std::move(status);
    return compiled;
  }
  compiled.constraint_node_count = node_count;
  compiled.dimension_operand_count = dimension_operand_count;
  return compiled;
}

DependencyRecorder::DependencyRecorder(
    const function_extractor_internal::NormalizedFunctionPattern& pattern,
    const function_extractor_internal::MatchState& match,
    const function_extractor_internal::TargetGraphSnapshot& snapshot,
    size_t max_attribute_bytes, size_t max_literal_bytes)
    : pattern_(pattern),
      match_(match),
      snapshot_(snapshot),
      max_attribute_bytes_(max_attribute_bytes),
      max_literal_bytes_(max_literal_bytes) {}

void DependencyRecorder::SaveError(common::Status status) {
  if (status_.IsOK() && !status.IsOK()) status_ = std::move(status);
}

DependencySnapshot& DependencyRecorder::FindOrAdd(
    DependencyKind kind, FusionPatternNodeId pattern_node,
    FusionPatternValueId pattern_value,
    FusionFormalAttributeId formal_attribute, std::string_view name,
    int64_t axis) {
  for (auto& dependency : dependencies_) {
    if (dependency.kind == kind &&
        dependency.pattern_node == pattern_node &&
        dependency.pattern_value == pattern_value &&
        dependency.formal_attribute == formal_attribute &&
        dependency.name == name && dependency.axis == axis) {
      return dependency;
    }
  }
  DependencySnapshot dependency;
  dependency.kind = kind;
  dependency.pattern_node = pattern_node;
  dependency.pattern_value = pattern_value;
  dependency.formal_attribute = formal_attribute;
  dependency.name = std::string{name};
  dependency.axis = axis;
  dependencies_.push_back(std::move(dependency));
  return dependencies_.back();
}

const Node* DependencyRecorder::TargetNode(
    FusionPatternNodeId pattern_node) const {
  if (pattern_node >= match_.pattern_node_to_target.size()) return nullptr;
  return snapshot_.graph_viewer->GetNode(
      match_.pattern_node_to_target[pattern_node]);
}

const NodeArg* DependencyRecorder::TargetValue(
    FusionPatternValueId pattern_value) const {
  if (pattern_value >= match_.pattern_value_to_target.size()) return nullptr;
  return match_.pattern_value_to_target[pattern_value];
}

const ONNX_NAMESPACE::AttributeProto* DependencyRecorder::EffectiveAttribute(
    FusionPatternNodeId pattern_node, std::string_view name,
    ONNX_NAMESPACE::AttributeProto& canonical) {
  const auto* node = TargetNode(pattern_node);
  if (node == nullptr) return nullptr;
  const auto* attribute = EffectiveTargetAttribute(*node, name);
  if (attribute == nullptr) return nullptr;
  auto status = CanonicalizeObservedAttribute(
      name, *attribute, max_attribute_bytes_, canonical);
  if (!status.IsOK()) {
    SaveError(std::move(status));
    return nullptr;
  }
  return &canonical;
}

const ONNX_NAMESPACE::AttributeProto* DependencyRecorder::FormalAttribute(
    FusionFormalAttributeId formal_attribute) const {
  if (formal_attribute >= match_.formal_attribute_bindings.size() ||
      !match_.formal_attribute_bindings[formal_attribute].has_value()) {
    return nullptr;
  }
  return &*match_.formal_attribute_bindings[formal_attribute];
}

const function_extractor_internal::LiteralWitness*
DependencyRecorder::LiteralWitness(
    FusionPatternValueId pattern_value) const {
  const auto it = std::find_if(
      match_.literal_witnesses.begin(), match_.literal_witnesses.end(),
      [pattern_value](
          const function_extractor_internal::LiteralWitness& witness) {
        return witness.pattern_value_id == pattern_value;
      });
  return it == match_.literal_witnesses.end() ? nullptr : &*it;
}

std::vector<ValueBindingSite> DependencyRecorder::BindingSites(
    FusionPatternValueId pattern_value) const {
  std::vector<ValueBindingSite> sites;
  if (pattern_value >= pattern_.values.size()) return sites;
  const auto& value = pattern_.values[pattern_value];
  if (value.producer_node_id != kNoPatternNode &&
      value.producer_node_id < match_.pattern_node_to_target.size()) {
    sites.push_back(ValueBindingSite{
        match_.pattern_node_to_target[value.producer_node_id],
        value.producer_output_index, true});
  }
  for (const auto& consumer : value.consumers) {
    if (consumer.node_id < match_.pattern_node_to_target.size()) {
      sites.push_back(ValueBindingSite{
          match_.pattern_node_to_target[consumer.node_id],
          consumer.input_index, false});
    }
  }
  std::sort(sites.begin(), sites.end(),
            [](const ValueBindingSite& lhs, const ValueBindingSite& rhs) {
              return std::tie(lhs.node_index, lhs.is_output, lhs.slot) <
                     std::tie(rhs.node_index, rhs.is_output, rhs.slot);
            });
  sites.erase(
      std::unique(sites.begin(), sites.end(),
                  [](const ValueBindingSite& lhs,
                     const ValueBindingSite& rhs) {
                    return lhs.node_index == rhs.node_index &&
                           lhs.slot == rhs.slot &&
                           lhs.is_output == rhs.is_output;
                  }),
      sites.end());
  return sites;
}

void DependencyRecorder::RecordNodeIdentity(
    FusionPatternNodeId pattern_node, uint8_t fields) {
  auto& dependency =
      FindOrAdd(DependencyKind::kNodeIdentity, pattern_node, 0, 0, {}, -1);
  auto& identity = dependency.node_identity;
  identity.observed_fields |= fields;
  const auto* node = TargetNode(pattern_node);
  if (node == nullptr) return;
  identity.index = node->Index();
  identity.domain = node->Domain();
  identity.op_type = node->OpType();
  identity.overload = node->Overload();
  identity.since_version = node->SinceVersion();
}

void DependencyRecorder::RecordNodeSlots(
    FusionPatternNodeId pattern_node, bool inputs, bool outputs) {
  auto& dependency =
      FindOrAdd(DependencyKind::kNodeSlots, pattern_node, 0, 0, {}, -1);
  auto& slots = dependency.node_slots;
  const auto* node = TargetNode(pattern_node);
  if (node == nullptr) return;
  slots.target_node = node->Index();
  if (inputs && !slots.inputs_observed) {
    slots.inputs_observed = true;
    for (const auto* input : node->InputDefs()) {
      const bool exists = input != nullptr && input->Exists();
      slots.input_exists.push_back(exists);
      slots.input_names.push_back(exists ? input->Name() : std::string{});
    }
  }
  if (outputs && !slots.outputs_observed) {
    slots.outputs_observed = true;
    for (const auto* output : node->OutputDefs()) {
      const bool exists = output != nullptr && output->Exists();
      slots.output_exists.push_back(exists);
      slots.output_names.push_back(exists ? output->Name() : std::string{});
    }
  }
}

void DependencyRecorder::RecordValueIdentity(
    FusionPatternValueId pattern_value, uint8_t fields) {
  auto& dependency =
      FindOrAdd(DependencyKind::kValueIdentity, 0, pattern_value, 0, {}, -1);
  auto& identity = dependency.value_identity;
  identity.observed_fields |= fields;
  const auto* value = TargetValue(pattern_value);
  identity.exists = value != nullptr && value->Exists();
  identity.name = identity.exists ? value->Name() : std::string{};
  identity.binding_sites = BindingSites(pattern_value);
}

void DependencyRecorder::RecordValueType(
    FusionPatternValueId pattern_value, uint8_t fields) {
  auto& dependency =
      FindOrAdd(DependencyKind::kValueType, 0, pattern_value, 0, {}, -1);
  auto& type = dependency.value_type;
  type.observed_fields |= fields;
  const auto* value = TargetValue(pattern_value);
  type.value_name =
      value != nullptr && value->Exists() ? value->Name() : std::string{};
  type.binding_sites = BindingSites(pattern_value);
  const auto* proto = value == nullptr ? nullptr : value->TypeAsProto();
  type.has_type = proto != nullptr;
  type.is_tensor = proto != nullptr && proto->has_tensor_type();
  type.tensor_element_type =
      type.is_tensor ? proto->tensor_type().elem_type() : 0;
  if (proto != nullptr) type.canonical_type = *proto;
}

void DependencyRecorder::RecordValueRank(
    FusionPatternValueId pattern_value) {
  auto& dependency =
      FindOrAdd(DependencyKind::kValueRank, 0, pattern_value, 0, {}, -1);
  auto& rank = dependency.value_rank;
  const auto* value = TargetValue(pattern_value);
  rank.value_name =
      value != nullptr && value->Exists() ? value->Name() : std::string{};
  rank.binding_sites = BindingSites(pattern_value);
  const auto* shape = TensorShape(value);
  rank.has_rank = shape != nullptr;
  rank.rank =
      shape == nullptr ? 0 : static_cast<size_t>(shape->dim_size());
}

void DependencyRecorder::RecordValueDimension(
    FusionPatternValueId pattern_value, int64_t normalized_axis) {
  auto& dependency = FindOrAdd(
      DependencyKind::kValueDimension, 0, pattern_value, 0, {},
      normalized_axis);
  auto& dimension = dependency.value_dimension;
  const auto* value = TargetValue(pattern_value);
  dimension.value_name =
      value != nullptr && value->Exists() ? value->Name() : std::string{};
  dimension.binding_sites = BindingSites(pattern_value);
  const auto* shape = TensorShape(value);
  if (shape != nullptr && normalized_axis >= 0 &&
      normalized_axis < shape->dim_size()) {
    dimension.fact =
        DimensionFromProto(shape->dim(static_cast<int>(normalized_axis)));
  }
}

void DependencyRecorder::RecordMatchedProducer(
    FusionPatternValueId pattern_value) {
  auto& dependency =
      FindOrAdd(DependencyKind::kMatchedProducer, 0, pattern_value, 0, {}, -1);
  auto& producer = dependency.matched_producer;
  const auto* value = TargetValue(pattern_value);
  producer.value_name =
      value != nullptr && value->Exists() ? value->Name() : std::string{};
  producer.binding_sites = BindingSites(pattern_value);
  if (pattern_value >= pattern_.values.size()) return;
  const auto& pattern_value_info = pattern_.values[pattern_value];
  producer.has_producer =
      pattern_value_info.producer_node_id != kNoPatternNode;
  if (producer.has_producer &&
      pattern_value_info.producer_node_id <
          match_.pattern_node_to_target.size()) {
    producer.producer_pattern_node =
        pattern_value_info.producer_node_id;
    producer.producer_target_node =
        match_.pattern_node_to_target[pattern_value_info.producer_node_id];
    producer.output_slot = pattern_value_info.producer_output_index;
  }
}

void DependencyRecorder::RecordEffectiveAttribute(
    FusionPatternNodeId pattern_node, std::string_view name) {
  auto& dependency = FindOrAdd(
      DependencyKind::kEffectiveAttribute, pattern_node, 0, 0, name, -1);
  auto& attribute = dependency.attribute;
  const auto* node = TargetNode(pattern_node);
  if (node == nullptr) return;
  attribute.target_node = node->Index();
  attribute.operator_attribute_name = std::string{name};
  ONNX_NAMESPACE::AttributeProto canonical;
  const auto* observed =
      EffectiveAttribute(pattern_node, name, canonical);
  attribute.exists = observed != nullptr;
  if (observed != nullptr) attribute.canonical_value = std::move(canonical);
}

void DependencyRecorder::RecordFormalAttribute(
    FusionFormalAttributeId formal_attribute) {
  auto& dependency = FindOrAdd(
      DependencyKind::kFormalAttribute, 0, 0, formal_attribute, {}, -1);
  auto& snapshot = dependency.formal_attribute_value;
  const auto* value = FormalAttribute(formal_attribute);
  if (value != nullptr) snapshot.canonical_value = *value;
  snapshot.occurrences.clear();
  for (const auto& occurrence : match_.matched_attribute_occurrences) {
    if (occurrence.formal_attribute_id == formal_attribute) {
      snapshot.occurrences.emplace_back(
          occurrence.target_node_index,
          occurrence.operator_attribute_name);
    }
  }
  std::sort(snapshot.occurrences.begin(), snapshot.occurrences.end());
}

void DependencyRecorder::RecordLiteral(
    FusionPatternValueId pattern_value) {
  auto& dependency =
      FindOrAdd(DependencyKind::kLiteral, 0, pattern_value, 0, {}, -1);
  auto& literal = dependency.literal;
  const auto* value = TargetValue(pattern_value);
  literal.value_name =
      value != nullptr && value->Exists() ? value->Name() : std::string{};
  literal.binding_sites = BindingSites(pattern_value);
  const auto* witness = LiteralWitness(pattern_value);
  literal.is_initializer = witness != nullptr && witness->is_initializer;
  if (pattern_value >= pattern_.values.size() ||
      !pattern_.values[pattern_value].is_literal) {
    return;
  }
  ONNX_NAMESPACE::AttributeProto source;
  source.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  *source.mutable_t() = pattern_.values[pattern_value].literal.tensor;
  ONNX_NAMESPACE::AttributeProto canonical;
  auto status = CanonicalizeFormalAttribute(
      "", ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR, source,
      max_literal_bytes_, canonical);
  if (!status.IsOK()) {
    SaveError(std::move(status));
    return;
  }
  literal.canonical_tensor = std::move(*canonical.mutable_t());
}

DimensionFact DependencyRecorder::ReadDimension(
    FusionPatternValueId pattern_value, int64_t axis, bool& rank_known,
    int64_t& normalized_axis) {
  RecordValueRank(pattern_value);
  const auto* shape = TensorShape(TargetValue(pattern_value));
  rank_known = shape != nullptr;
  normalized_axis = -1;
  if (shape == nullptr) return {};
  const int64_t rank = shape->dim_size();
  normalized_axis = axis < 0 ? axis + rank : axis;
  if (normalized_axis < 0 || normalized_axis >= rank) return {};
  RecordValueDimension(pattern_value, normalized_axis);
  return DimensionFromProto(
      shape->dim(static_cast<int>(normalized_axis)));
}

std::vector<DependencySnapshot> DependencyRecorder::TakeSnapshot() {
  std::sort(dependencies_.begin(), dependencies_.end(), DependencyLess);
  return std::move(dependencies_);
}

Status EvaluateConstraintProgram(
    const CompiledConstraintProgram& program, DependencyRecorder& recorder,
    ConstraintEvaluationResult& result) {
  result = {};
  ORT_RETURN_IF_ERROR(program.construction_status);
  ORT_RETURN_IF_NOT(program.predicate != nullptr,
                    "Compiled fusion constraint predicate is missing.");

  for (const auto& dimension_class : program.dimension_classes) {
    std::vector<DimensionFact> facts;
    facts.reserve(dimension_class.dimensions.size());
    for (const auto& dimension : dimension_class.dimensions) {
      recorder.RecordValueIdentity(dimension.value.pattern_value,
                                   kValueExistsField);
      bool rank_known = false;
      int64_t normalized_axis = -1;
      const auto fact = recorder.ReadDimension(
          dimension.value.pattern_value, dimension.axis, rank_known,
          normalized_axis);
      if (!rank_known || normalized_axis < 0) {
        facts.push_back({});
      } else {
        facts.push_back(fact);
      }
    }
    const TruthValue class_value = CompareDimensionClass(facts);
    if (!ApplyUnknownPolicy(class_value, dimension_class.unknown_policy)) {
      result.satisfied = false;
      result.detail = dimension_class.label.empty()
                          ? "dimension equivalence class failed"
                          : "dimension equivalence class '" +
                                dimension_class.label + "' failed";
      return recorder.Status();
    }
  }

  bool predicate_value = false;
  ORT_RETURN_IF_ERROR(EvaluateNode(
      *program.predicate, recorder, predicate_value, result));
  result.satisfied = predicate_value;
  return recorder.Status();
}

Status PrevalidateDependencies(
    const Graph& graph, gsl::span<const DependencySnapshot> dependencies,
    size_t max_attribute_bytes, size_t max_literal_bytes) {
  for (const auto& dependency : dependencies) {
    switch (dependency.kind) {
      case DependencyKind::kNodeIdentity: {
        const auto& expected = dependency.node_identity;
        const auto* node = graph.GetNode(expected.index);
        ORT_RETURN_IF(node == nullptr, "observed node was removed");
        ORT_RETURN_IF(
            (expected.observed_fields & kNodeIndexField) != 0 &&
                node->Index() != expected.index,
            "observed node index changed");
        ORT_RETURN_IF(
            (expected.observed_fields & kNodeDomainField) != 0 &&
                node->Domain() != expected.domain,
            "observed node domain changed");
        ORT_RETURN_IF(
            (expected.observed_fields & kNodeOpTypeField) != 0 &&
                node->OpType() != expected.op_type,
            "observed node op type changed");
        ORT_RETURN_IF(
            (expected.observed_fields & kNodeOverloadField) != 0 &&
                node->Overload() != expected.overload,
            "observed node overload changed");
        ORT_RETURN_IF(
            (expected.observed_fields & kNodeVersionField) != 0 &&
                node->SinceVersion() != expected.since_version,
            "observed node version changed");
        break;
      }
      case DependencyKind::kNodeSlots: {
        const auto* node =
            graph.GetNode(dependency.node_slots.target_node);
        ORT_RETURN_IF(node == nullptr, "observed node slots were removed");
        const auto compare_slots =
            [](const auto& definitions,
               const std::vector<std::string>& names,
               const std::vector<bool>& exists) {
              if (definitions.size() != names.size() ||
                  names.size() != exists.size()) {
                return false;
              }
              for (size_t i = 0; i < names.size(); ++i) {
                const bool current_exists =
                    definitions[i] != nullptr && definitions[i]->Exists();
                if (current_exists != exists[i]) return false;
                if (current_exists &&
                    definitions[i]->Name() != names[i]) {
                  return false;
                }
              }
              return true;
            };
        ORT_RETURN_IF(
            dependency.node_slots.inputs_observed &&
                !compare_slots(node->InputDefs(),
                               dependency.node_slots.input_names,
                               dependency.node_slots.input_exists),
            "observed node inputs changed");
        ORT_RETURN_IF(
            dependency.node_slots.outputs_observed &&
                !compare_slots(node->OutputDefs(),
                               dependency.node_slots.output_names,
                               dependency.node_slots.output_exists),
            "observed node outputs changed");
        break;
      }
      case DependencyKind::kValueIdentity: {
        bool consistent = false;
        const auto* value = ValueAtBindingSites(
            graph, dependency.value_identity.binding_sites, consistent);
        ORT_RETURN_IF_NOT(consistent, "observed value binding changed");
        const bool exists = value != nullptr && value->Exists();
        ORT_RETURN_IF(
            (dependency.value_identity.observed_fields &
             kValueExistsField) != 0 &&
                exists != dependency.value_identity.exists,
            "observed value presence changed");
        ORT_RETURN_IF(
            (dependency.value_identity.observed_fields &
             kValueNameField) != 0 &&
                (exists ? value->Name() : std::string{}) !=
                    dependency.value_identity.name,
            "observed value name changed");
        break;
      }
      case DependencyKind::kValueType: {
        bool consistent = false;
        const auto* value = ValueAtBindingSites(
            graph, dependency.value_type.binding_sites, consistent);
        ORT_RETURN_IF_NOT(consistent, "observed value binding changed");
        const auto* type = value == nullptr ? nullptr : value->TypeAsProto();
        const bool has_type = type != nullptr;
        const bool is_tensor = has_type && type->has_tensor_type();
        const int32_t element_type =
            is_tensor ? type->tensor_type().elem_type() : 0;
        ORT_RETURN_IF(
            (dependency.value_type.observed_fields & kTypeKindField) != 0 &&
                (has_type != dependency.value_type.has_type ||
                 is_tensor != dependency.value_type.is_tensor),
            "observed value type kind changed");
        ORT_RETURN_IF(
            (dependency.value_type.observed_fields & kTypeElementField) != 0 &&
                element_type !=
                    dependency.value_type.tensor_element_type,
            "observed tensor element type changed");
        ORT_RETURN_IF(
            (dependency.value_type.observed_fields & kTypeFullField) != 0 &&
                (has_type != dependency.value_type.has_type ||
                 (has_type &&
                  type->SerializeAsString() !=
                      dependency.value_type.canonical_type.SerializeAsString())),
            "observed complete value type changed");
        break;
      }
      case DependencyKind::kValueRank: {
        bool consistent = false;
        const auto* value = ValueAtBindingSites(
            graph, dependency.value_rank.binding_sites, consistent);
        ORT_RETURN_IF_NOT(consistent, "observed value binding changed");
        const auto* shape = TensorShape(value);
        const bool has_rank = shape != nullptr;
        const size_t rank =
            has_rank ? static_cast<size_t>(shape->dim_size()) : 0;
        ORT_RETURN_IF(
            has_rank != dependency.value_rank.has_rank ||
                rank != dependency.value_rank.rank,
            "observed value rank changed");
        break;
      }
      case DependencyKind::kValueDimension: {
        bool consistent = false;
        const auto* value = ValueAtBindingSites(
            graph, dependency.value_dimension.binding_sites, consistent);
        ORT_RETURN_IF_NOT(consistent, "observed value binding changed");
        const auto* shape = TensorShape(value);
        DimensionFact fact;
        if (shape != nullptr && dependency.axis >= 0 &&
            dependency.axis < shape->dim_size()) {
          fact = DimensionFromProto(
              shape->dim(static_cast<int>(dependency.axis)));
        }
        ORT_RETURN_IF_NOT(
            DimensionFactsEqual(fact,
                                dependency.value_dimension.fact),
            "observed value dimension changed");
        break;
      }
      case DependencyKind::kMatchedProducer: {
        bool consistent = false;
        const auto* value = ValueAtBindingSites(
            graph, dependency.matched_producer.binding_sites, consistent);
        ORT_RETURN_IF_NOT(consistent, "observed value binding changed");
        const Node* producer =
            value == nullptr || !value->Exists()
                ? nullptr
                : graph.GetProducerNode(value->Name());
        if (!dependency.matched_producer.has_producer) {
          ORT_RETURN_IF(producer != nullptr,
                        "observed producer became present");
          break;
        }
        ORT_RETURN_IF(
            producer == nullptr ||
                producer->Index() !=
                    dependency.matched_producer.producer_target_node,
            "observed producer changed");
        ORT_RETURN_IF(
            dependency.matched_producer.output_slot >=
                    producer->OutputDefs().size() ||
                producer->OutputDefs()[dependency.matched_producer.output_slot] == nullptr ||
                value == nullptr ||
                producer->OutputDefs()[dependency.matched_producer.output_slot]
                        ->Name() != value->Name(),
            "observed producer output slot changed");
        break;
      }
      case DependencyKind::kEffectiveAttribute: {
        const auto* node =
            graph.GetNode(dependency.attribute.target_node);
        ORT_RETURN_IF(node == nullptr,
                      "observed attribute node was removed");
        const auto* attribute = EffectiveTargetAttribute(
            *node, dependency.attribute.operator_attribute_name);
        ORT_RETURN_IF((attribute != nullptr) !=
                          dependency.attribute.exists,
                      "observed attribute presence changed");
        if (attribute != nullptr) {
          ONNX_NAMESPACE::AttributeProto canonical;
          ORT_RETURN_IF_ERROR(CanonicalizeObservedAttribute(
              dependency.attribute.operator_attribute_name, *attribute,
              max_attribute_bytes, canonical));
          bool equal = false;
          ORT_RETURN_IF_ERROR(AttributesEqual(
              canonical, dependency.attribute.canonical_value,
              max_attribute_bytes, equal));
          ORT_RETURN_IF_NOT(equal, "observed attribute changed");
        }
        break;
      }
      case DependencyKind::kFormalAttribute: {
        for (const auto& [node_index, name] :
             dependency.formal_attribute_value.occurrences) {
          const auto* node = graph.GetNode(node_index);
          ORT_RETURN_IF(node == nullptr,
                        "formal attribute source node was removed");
          const auto* attribute =
              EffectiveTargetAttribute(*node, name);
          ORT_RETURN_IF(attribute == nullptr,
                        "formal attribute source became missing");
          ONNX_NAMESPACE::AttributeProto canonical;
          ORT_RETURN_IF_ERROR(CanonicalizeObservedAttribute(
              dependency.formal_attribute_value.canonical_value.name(),
              *attribute, max_attribute_bytes, canonical));
          bool equal = false;
          ORT_RETURN_IF_ERROR(AttributesEqual(
              canonical,
              dependency.formal_attribute_value.canonical_value,
              max_attribute_bytes, equal));
          ORT_RETURN_IF_NOT(equal, "formal attribute binding changed");
        }
        break;
      }
      case DependencyKind::kLiteral: {
        bool consistent = false;
        const auto* value = ValueAtBindingSites(
            graph, dependency.literal.binding_sites, consistent);
        ORT_RETURN_IF_NOT(consistent || value == nullptr,
                          "observed literal binding changed");
        ORT_RETURN_IF(value == nullptr || !value->Exists(),
                      "observed literal value became missing");
        const auto* initializer =
            graph.GetConstantInitializer(value->Name(), false);
        const ONNX_NAMESPACE::TensorProto* tensor = initializer;
        ONNX_NAMESPACE::TensorProto normalized;
        if (tensor == nullptr) {
          const auto* producer = graph.GetProducerNode(value->Name());
          ORT_RETURN_IF(
              producer == nullptr || producer->Domain() != kOnnxDomain ||
                  producer->OpType() != "Constant",
              "observed literal producer changed");
          ORT_RETURN_IF_ERROR(
              NormalizeConstantAttributes(producer->GetAttributes(),
                                          normalized));
          tensor = &normalized;
        }
        ORT_RETURN_IF(
            (initializer != nullptr) !=
                dependency.literal.is_initializer,
            "observed literal initializer classification changed");
        bool equal = false;
        const auto& model_path = graph.ModelPath();
        ORT_RETURN_IF_ERROR(CompareTensorLiterals(
            dependency.literal.canonical_tensor, *tensor,
            max_literal_bytes, equal, &model_path));
        ORT_RETURN_IF_NOT(equal, "observed literal changed");
        break;
      }
    }
  }
  return Status::OK();
}

}  // namespace fusion_rewriter_internal

namespace {

using fusion_rewriter_internal::ConstraintExpr;
using fusion_rewriter_internal::ConstraintKind;

template <typename ImplType>
std::shared_ptr<ImplType> MakeConstraintImpl(
    std::shared_ptr<const ConstraintExpr> expression) {
  auto impl = std::make_shared<ImplType>();
  impl->expression = std::move(expression);
  return impl;
}

std::vector<std::byte> TensorLogicalData(
    const ONNX_NAMESPACE::TensorProto& tensor) {
  std::vector<std::byte> data;
  if (tensor.data_type() ==
      ONNX_NAMESPACE::TensorProto_DataType_STRING) {
    size_t byte_count = 0;
    for (const auto& value : tensor.string_data()) {
      byte_count += sizeof(uint64_t) + value.size();
    }
    data.reserve(byte_count);
    for (const auto& value : tensor.string_data()) {
      const uint64_t length = value.size();
      const auto* length_bytes =
          reinterpret_cast<const std::byte*>(&length);
      data.insert(data.end(), length_bytes,
                  length_bytes + sizeof(length));
      const auto* value_bytes =
          reinterpret_cast<const std::byte*>(value.data());
      data.insert(data.end(), value_bytes,
                  value_bytes + value.size());
    }
  } else {
    const auto* begin =
        reinterpret_cast<const std::byte*>(tensor.raw_data().data());
    data.assign(begin, begin + tensor.raw_data().size());
  }
  return data;
}

}  // namespace

FusionValueRef FusionValueRef::FormalInput(size_t index) {
  return {FusionValueRefKind::kFormalInput, index};
}

FusionValueRef FusionValueRef::FormalOutput(size_t index) {
  return {FusionValueRefKind::kFormalOutput, index};
}

FusionValueRef FusionValueRef::PatternValue(FusionPatternValueId id) {
  return {FusionValueRefKind::kPatternValue, id};
}

FusionAttributeRef FusionAttributeRef::Formal(
    FusionFormalAttributeId id) {
  return {FusionAttributeRefKind::kFormalAttribute, id, {}, {}};
}

FusionAttributeRef FusionAttributeRef::Effective(
    FusionNodeRef node, std::string operator_attribute_name) {
  return {FusionAttributeRefKind::kEffectiveNodeAttribute, 0, node,
          std::move(operator_attribute_name)};
}

FusionConstraint::FusionConstraint()
    : impl_(MakeConstraintImpl<Impl>(
          fusion_rewriter_internal::MakeExpr(ConstraintKind::kAllOf))) {}

FusionConstraint::~FusionConstraint() = default;
FusionConstraint::FusionConstraint(const FusionConstraint&) = default;
FusionConstraint& FusionConstraint::operator=(const FusionConstraint&) =
    default;
FusionConstraint::FusionConstraint(FusionConstraint&&) noexcept = default;
FusionConstraint& FusionConstraint::operator=(
    FusionConstraint&&) noexcept = default;

FusionConstraint::FusionConstraint(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}

FusionConstraint FusionConstraint::AllOf(
    std::vector<FusionConstraint> operands) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kAllOf;
  for (auto& operand : operands) {
    expression->operands.push_back(
        operand.impl_ == nullptr ? nullptr : operand.impl_->expression);
  }
  return FusionConstraint(
      MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::AnyOf(
    std::vector<FusionConstraint> operands) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kAnyOf;
  for (auto& operand : operands) {
    expression->operands.push_back(
        operand.impl_ == nullptr ? nullptr : operand.impl_->expression);
  }
  return FusionConstraint(
      MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::Not(FusionConstraint operand) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kNot;
  expression->operands.push_back(
      operand.impl_ == nullptr ? nullptr : operand.impl_->expression);
  return FusionConstraint(
      MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::IsPresent(FusionValueRef value) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kIsPresent;
  expression->lhs_value = value;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::IsMissing(FusionValueRef value) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kIsMissing;
  expression->lhs_value = value;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::IsTensor(FusionValueRef value) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kIsTensor;
  expression->lhs_value = value;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::ElementTypeIs(
    FusionValueRef value, int32_t elem_type) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kElementTypeIs;
  expression->lhs_value = value;
  expression->element_type = elem_type;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::ElementTypeIn(
    FusionValueRef value, std::vector<int32_t> elem_types) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kElementTypeIn;
  expression->lhs_value = value;
  expression->element_types = std::move(elem_types);
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::TypeEquals(
    FusionValueRef lhs, FusionValueRef rhs, FusionUnknownPolicy policy) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kTypeEquals;
  expression->lhs_value = lhs;
  expression->rhs_value = rhs;
  expression->unknown_policy = policy;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::SameElementType(
    FusionValueRef lhs, FusionValueRef rhs, FusionUnknownPolicy policy) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kSameElementType;
  expression->lhs_value = lhs;
  expression->rhs_value = rhs;
  expression->unknown_policy = policy;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::RankIs(
    FusionValueRef value, size_t rank) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kRankIs;
  expression->lhs_value = value;
  expression->minimum_rank = rank;
  expression->maximum_rank = rank;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::RankIn(
    FusionValueRef value, size_t minimum, size_t maximum) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kRankIn;
  expression->lhs_value = value;
  expression->minimum_rank = minimum;
  expression->maximum_rank = maximum;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::SameRank(
    FusionValueRef lhs, FusionValueRef rhs, FusionUnknownPolicy policy) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kSameRank;
  expression->lhs_value = lhs;
  expression->rhs_value = rhs;
  expression->unknown_policy = policy;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::DimValueIs(
    FusionDimRef dim, int64_t value) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kDimValueIs;
  expression->lhs_dim = dim;
  expression->integer_value = value;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::DimEquals(
    FusionDimRef lhs, FusionDimRef rhs, FusionUnknownPolicy policy) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kDimEquals;
  expression->lhs_dim = lhs;
  expression->rhs_dim = rhs;
  expression->unknown_policy = policy;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::ShapeEquals(
    FusionValueRef lhs, FusionValueRef rhs, FusionUnknownPolicy policy) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kShapeEquals;
  expression->lhs_value = lhs;
  expression->rhs_value = rhs;
  expression->unknown_policy = policy;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::AttributePresent(
    FusionAttributeRef attribute) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kAttributePresent;
  expression->lhs_attribute = std::move(attribute);
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::AttributeTypeIs(
    FusionAttributeRef attribute,
    ONNX_NAMESPACE::AttributeProto_AttributeType type) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kAttributeTypeIs;
  expression->lhs_attribute = std::move(attribute);
  expression->attribute_type = type;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::AttributeEquals(
    FusionAttributeRef attribute,
    ONNX_NAMESPACE::AttributeProto canonical_literal) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kAttributeEquals;
  expression->lhs_attribute = std::move(attribute);
  expression->attribute_literals.push_back(std::move(canonical_literal));
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::AttributeIn(
    FusionAttributeRef attribute,
    std::vector<ONNX_NAMESPACE::AttributeProto> canonical_literals) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kAttributeIn;
  expression->lhs_attribute = std::move(attribute);
  expression->attribute_literals = std::move(canonical_literals);
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::IntAttributeInRange(
    FusionAttributeRef attribute, int64_t minimum, int64_t maximum) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kIntAttributeInRange;
  expression->lhs_attribute = std::move(attribute);
  expression->minimum_integer = minimum;
  expression->maximum_integer = maximum;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::FloatAttributeInRange(
    FusionAttributeRef attribute, float minimum, float maximum) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kFloatAttributeInRange;
  expression->lhs_attribute = std::move(attribute);
  expression->minimum_float = minimum;
  expression->maximum_float = maximum;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::StringAttributeIn(
    FusionAttributeRef attribute, std::vector<std::string> values) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kStringAttributeIn;
  expression->lhs_attribute = std::move(attribute);
  expression->string_values = std::move(values);
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraint FusionConstraint::SameAttributeValue(
    FusionAttributeRef lhs, FusionAttributeRef rhs,
    FusionUnknownPolicy policy) {
  auto expression =
      std::make_shared<fusion_rewriter_internal::ConstraintExpr>();
  expression->kind = ConstraintKind::kSameAttributeValue;
  expression->lhs_attribute = std::move(lhs);
  expression->rhs_attribute = std::move(rhs);
  expression->unknown_policy = policy;
  return FusionConstraint(MakeConstraintImpl<Impl>(std::move(expression)));
}

FusionConstraintProgram::FusionConstraintProgram(
    std::vector<FusionDimensionEquivalenceClass> dimension_classes,
    FusionConstraint predicate) {
  auto definition =
      std::make_shared<fusion_rewriter_internal::ConstraintProgramDefinition>();
  definition->dimension_classes = std::move(dimension_classes);
  definition->predicate =
      predicate.impl_ == nullptr ? nullptr : predicate.impl_->expression;
  auto status =
      fusion_rewriter_internal::ValidateRawDefinition(*definition);
  if (!status.IsOK()) definition->construction_status = std::move(status);
  auto impl = std::make_shared<Impl>();
  impl->definition = std::move(definition);
  impl_ = std::move(impl);
}

FusionConstraintProgram::~FusionConstraintProgram() = default;
FusionConstraintProgram::FusionConstraintProgram(
    const FusionConstraintProgram&) = default;
FusionConstraintProgram& FusionConstraintProgram::operator=(
    const FusionConstraintProgram&) = default;
FusionConstraintProgram::FusionConstraintProgram(
    FusionConstraintProgram&&) noexcept = default;
FusionConstraintProgram& FusionConstraintProgram::operator=(
    FusionConstraintProgram&&) noexcept = default;

struct FusionDimensionView::Impl {
  fusion_rewriter_internal::DimensionFact fact;
  fusion_rewriter_internal::DependencyRecorder* recorder{};
  FusionPatternValueId pattern_value{};
  int64_t axis{-1};
};

struct FusionShapeView::Impl {
  bool has_rank{};
  std::vector<FusionDimensionView::Impl> dimensions;
  fusion_rewriter_internal::DependencyRecorder* recorder{};
  FusionPatternValueId pattern_value{};
};

struct FusionTypeView::Impl {
  bool has_type{};
  bool is_tensor{};
  int32_t tensor_element_type{};
  fusion_rewriter_internal::DependencyRecorder* recorder{};
  FusionPatternValueId pattern_value{};
};

struct FusionTensorView::Impl {
  int32_t element_type{};
  std::vector<int64_t> dimensions;
  std::vector<std::byte> logical_data;
};

struct FusionAttributeView::Impl {
  bool exists{};
  ONNX_NAMESPACE::AttributeProto canonical;
  mutable std::deque<FusionTensorView::Impl> tensor_views;
};

struct FusionLiteralView::Impl {
  bool is_initializer{};
  ONNX_NAMESPACE::TensorProto tensor;
  mutable std::optional<FusionTensorView::Impl> tensor_view;
};

struct FusionValueView::Impl {
  std::string name;
  bool exists{};
  FusionTypeView::Impl type;
  FusionShapeView::Impl shape;
  std::optional<FusionPatternNodeId> matched_producer;
  std::optional<size_t> producer_output_index;
  fusion_rewriter_internal::DependencyRecorder* recorder{};
  FusionPatternValueId pattern_value{};
};

struct FusionNodeView::Impl {
  NodeIndex index{};
  std::string domain;
  std::string op_type;
  std::string overload;
  int since_version{-1};
  std::vector<FusionPatternValueId> inputs;
  std::vector<FusionPatternValueId> outputs;
  fusion_rewriter_internal::DependencyRecorder* recorder{};
  FusionPatternNodeId pattern_node{};
  mutable std::deque<FusionAttributeView::Impl> attributes;
};

struct FusionMatchContext::Impl {
  const function_extractor_internal::NormalizedFunctionPattern* pattern{};
  const function_extractor_internal::CompiledFunctionPattern*
      compiled_pattern{};
  fusion_rewriter_internal::DependencyRecorder* recorder{};
  mutable std::deque<FusionNodeView::Impl> nodes;
  mutable std::deque<FusionValueView::Impl> values;
  mutable std::deque<FusionLiteralView::Impl> literals;
  mutable std::deque<FusionAttributeView::Impl> attributes;

  FusionValueView::Impl MakeValue(FusionPatternValueId id) const {
    FusionValueView::Impl result;
    result.recorder = recorder;
    result.pattern_value = id;
    if (id >= pattern->values.size()) return result;
    const auto* value = recorder->TargetValue(id);
    result.exists = value != nullptr && value->Exists();
    if (result.exists) result.name = value->Name();
    result.type.recorder = recorder;
    result.type.pattern_value = id;
    result.shape.recorder = recorder;
    result.shape.pattern_value = id;
    const auto* type = value == nullptr ? nullptr : value->TypeAsProto();
    result.type.has_type = type != nullptr;
    result.type.is_tensor = type != nullptr && type->has_tensor_type();
    result.type.tensor_element_type =
        result.type.is_tensor ? type->tensor_type().elem_type() : 0;
    const auto* shape =
        value == nullptr ? nullptr : value->Shape();
    result.shape.has_rank =
        result.type.is_tensor && shape != nullptr;
    if (result.shape.has_rank) {
      result.shape.dimensions.reserve(shape->dim_size());
      for (int64_t axis = 0; axis < shape->dim_size(); ++axis) {
        result.shape.dimensions.emplace_back();
        auto& dimension = result.shape.dimensions.back();
        dimension.fact =
            fusion_rewriter_internal::DimensionFromProto(
                shape->dim(static_cast<int>(axis)));
        dimension.recorder = recorder;
        dimension.pattern_value = id;
        dimension.axis = axis;
      }
    }
    const auto& pattern_value = pattern->values[id];
    if (pattern_value.producer_node_id !=
        function_extractor_internal::kNoPatternNode) {
      result.matched_producer = pattern_value.producer_node_id;
      result.producer_output_index =
          pattern_value.producer_output_index;
    }
    return result;
  }
};

FusionDimensionView::FusionDimensionView(const Impl* impl) : impl_(impl) {}

FusionDimensionKind FusionDimensionView::Kind() const {
  if (impl_ == nullptr) return FusionDimensionKind::kUnknown;
  impl_->recorder->RecordValueIdentity(
      impl_->pattern_value, fusion_rewriter_internal::kValueNameField |
                                fusion_rewriter_internal::kValueExistsField);
  impl_->recorder->RecordValueDimension(impl_->pattern_value, impl_->axis);
  switch (impl_->fact.kind) {
    case fusion_rewriter_internal::DimensionFactKind::kValue:
      return FusionDimensionKind::kValue;
    case fusion_rewriter_internal::DimensionFactKind::kSymbol:
      return FusionDimensionKind::kSymbol;
    default:
      return FusionDimensionKind::kUnknown;
  }
}

std::optional<int64_t> FusionDimensionView::Value() const {
  if (impl_ == nullptr) return std::nullopt;
  impl_->recorder->RecordValueIdentity(
      impl_->pattern_value, fusion_rewriter_internal::kValueNameField |
                                fusion_rewriter_internal::kValueExistsField);
  impl_->recorder->RecordValueDimension(impl_->pattern_value, impl_->axis);
  return impl_->fact.kind ==
                 fusion_rewriter_internal::DimensionFactKind::kValue
             ? std::optional<int64_t>{impl_->fact.value}
             : std::nullopt;
}

std::optional<std::string_view> FusionDimensionView::Symbol() const {
  if (impl_ == nullptr) return std::nullopt;
  impl_->recorder->RecordValueIdentity(
      impl_->pattern_value, fusion_rewriter_internal::kValueNameField |
                                fusion_rewriter_internal::kValueExistsField);
  impl_->recorder->RecordValueDimension(impl_->pattern_value, impl_->axis);
  return impl_->fact.kind ==
                 fusion_rewriter_internal::DimensionFactKind::kSymbol
             ? std::optional<std::string_view>{impl_->fact.symbol}
             : std::nullopt;
}

FusionShapeView::FusionShapeView(const Impl* impl) : impl_(impl) {}

bool FusionShapeView::HasRank() const {
  if (impl_ == nullptr) return false;
  impl_->recorder->RecordValueIdentity(
      impl_->pattern_value, fusion_rewriter_internal::kValueNameField |
                                fusion_rewriter_internal::kValueExistsField);
  impl_->recorder->RecordValueRank(impl_->pattern_value);
  return impl_->has_rank;
}

size_t FusionShapeView::Rank() const {
  if (impl_ == nullptr) return 0;
  impl_->recorder->RecordValueIdentity(
      impl_->pattern_value, fusion_rewriter_internal::kValueNameField |
                                fusion_rewriter_internal::kValueExistsField);
  impl_->recorder->RecordValueRank(impl_->pattern_value);
  return impl_->has_rank ? impl_->dimensions.size() : 0;
}

std::optional<FusionDimensionView> FusionShapeView::Dimension(
    size_t axis) const {
  if (impl_ == nullptr) return std::nullopt;
  impl_->recorder->RecordValueIdentity(
      impl_->pattern_value, fusion_rewriter_internal::kValueNameField |
                                fusion_rewriter_internal::kValueExistsField);
  impl_->recorder->RecordValueRank(impl_->pattern_value);
  if (!impl_->has_rank || axis >= impl_->dimensions.size()) {
    return std::nullopt;
  }
  impl_->recorder->RecordValueDimension(
      impl_->pattern_value, static_cast<int64_t>(axis));
  return FusionDimensionView(&impl_->dimensions[axis]);
}

FusionTypeView::FusionTypeView(const Impl* impl) : impl_(impl) {}

bool FusionTypeView::IsTensor() const {
  if (impl_ == nullptr) return false;
  impl_->recorder->RecordValueIdentity(
      impl_->pattern_value, fusion_rewriter_internal::kValueNameField |
                                fusion_rewriter_internal::kValueExistsField);
  impl_->recorder->RecordValueType(
      impl_->pattern_value,
      fusion_rewriter_internal::kTypeKindField);
  return impl_->has_type && impl_->is_tensor;
}

std::optional<int32_t> FusionTypeView::TensorElementType() const {
  if (impl_ == nullptr) return std::nullopt;
  impl_->recorder->RecordValueIdentity(
      impl_->pattern_value, fusion_rewriter_internal::kValueNameField |
                                fusion_rewriter_internal::kValueExistsField);
  impl_->recorder->RecordValueType(
      impl_->pattern_value,
      fusion_rewriter_internal::kTypeKindField |
          fusion_rewriter_internal::kTypeElementField);
  return impl_->has_type && impl_->is_tensor &&
                 impl_->tensor_element_type != 0
             ? std::optional<int32_t>{impl_->tensor_element_type}
             : std::nullopt;
}

FusionTensorView::FusionTensorView(const Impl* impl) : impl_(impl) {}

int32_t FusionTensorView::ElementType() const {
  return impl_ == nullptr ? 0 : impl_->element_type;
}

gsl::span<const int64_t> FusionTensorView::Dimensions() const {
  return impl_ == nullptr ? gsl::span<const int64_t>{}
                          : gsl::span<const int64_t>{impl_->dimensions};
}

gsl::span<const std::byte> FusionTensorView::LogicalData() const {
  return impl_ == nullptr ? gsl::span<const std::byte>{}
                          : gsl::span<const std::byte>{
                                impl_->logical_data};
}

FusionAttributeView::FusionAttributeView(const Impl* impl) : impl_(impl) {}

bool FusionAttributeView::Exists() const {
  return impl_ != nullptr && impl_->exists;
}

ONNX_NAMESPACE::AttributeProto_AttributeType
FusionAttributeView::Type() const {
  return impl_ == nullptr || !impl_->exists
             ? ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED
             : impl_->canonical.type();
}

std::optional<float> FusionAttributeView::Float() const {
  return impl_ != nullptr && impl_->exists &&
                 impl_->canonical.type() ==
                     ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT
             ? std::optional<float>{impl_->canonical.f()}
             : std::nullopt;
}

std::optional<int64_t> FusionAttributeView::Int() const {
  return impl_ != nullptr && impl_->exists &&
                 impl_->canonical.type() ==
                     ONNX_NAMESPACE::AttributeProto_AttributeType_INT
             ? std::optional<int64_t>{impl_->canonical.i()}
             : std::nullopt;
}

std::optional<std::string_view> FusionAttributeView::String() const {
  return impl_ != nullptr && impl_->exists &&
                 impl_->canonical.type() ==
                     ONNX_NAMESPACE::AttributeProto_AttributeType_STRING
             ? std::optional<std::string_view>{impl_->canonical.s()}
             : std::nullopt;
}

std::optional<gsl::span<const float>> FusionAttributeView::Floats() const {
  if (impl_ == nullptr || !impl_->exists ||
      impl_->canonical.type() !=
          ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS) {
    return std::nullopt;
  }
  return gsl::span<const float>{impl_->canonical.floats().data(),
                                static_cast<size_t>(
                                    impl_->canonical.floats_size())};
}

std::optional<gsl::span<const int64_t>> FusionAttributeView::Ints() const {
  if (impl_ == nullptr || !impl_->exists ||
      impl_->canonical.type() !=
          ONNX_NAMESPACE::AttributeProto_AttributeType_INTS) {
    return std::nullopt;
  }
  return gsl::span<const int64_t>{
      impl_->canonical.ints().data(),
      static_cast<size_t>(impl_->canonical.ints_size())};
}

std::optional<std::vector<std::string_view>>
FusionAttributeView::Strings() const {
  if (impl_ == nullptr || !impl_->exists ||
      impl_->canonical.type() !=
          ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS) {
    return std::nullopt;
  }
  std::vector<std::string_view> result;
  result.reserve(impl_->canonical.strings_size());
  for (const auto& value : impl_->canonical.strings()) {
    result.push_back(value);
  }
  return result;
}

std::optional<FusionTensorView> FusionAttributeView::Tensor() const {
  if (impl_ == nullptr || !impl_->exists ||
      impl_->canonical.type() !=
          ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
    return std::nullopt;
  }
  if (impl_->tensor_views.empty()) {
    FusionTensorView::Impl tensor;
    tensor.element_type = impl_->canonical.t().data_type();
    tensor.dimensions.assign(impl_->canonical.t().dims().begin(),
                             impl_->canonical.t().dims().end());
    tensor.logical_data = TensorLogicalData(impl_->canonical.t());
    impl_->tensor_views.push_back(std::move(tensor));
  }
  return FusionTensorView(&impl_->tensor_views.front());
}

std::optional<std::vector<FusionTensorView>>
FusionAttributeView::Tensors() const {
  if (impl_ == nullptr || !impl_->exists ||
      impl_->canonical.type() !=
          ONNX_NAMESPACE::AttributeProto_AttributeType_TENSORS) {
    return std::nullopt;
  }
  if (impl_->tensor_views.empty()) {
    for (const auto& source : impl_->canonical.tensors()) {
      FusionTensorView::Impl tensor;
      tensor.element_type = source.data_type();
      tensor.dimensions.assign(source.dims().begin(), source.dims().end());
      tensor.logical_data = TensorLogicalData(source);
      impl_->tensor_views.push_back(std::move(tensor));
    }
  }
  std::vector<FusionTensorView> result;
  result.reserve(impl_->tensor_views.size());
  for (const auto& tensor : impl_->tensor_views) {
    result.push_back(FusionTensorView(&tensor));
  }
  return result;
}

FusionLiteralView::FusionLiteralView(const Impl* impl) : impl_(impl) {}

bool FusionLiteralView::IsInitializer() const {
  return impl_ != nullptr && impl_->is_initializer;
}

FusionTensorView FusionLiteralView::Tensor() const {
  if (impl_ == nullptr) return FusionTensorView(nullptr);
  if (!impl_->tensor_view.has_value()) {
    FusionTensorView::Impl tensor;
    tensor.element_type = impl_->tensor.data_type();
    tensor.dimensions.assign(impl_->tensor.dims().begin(),
                             impl_->tensor.dims().end());
    tensor.logical_data = TensorLogicalData(impl_->tensor);
    impl_->tensor_view = std::move(tensor);
  }
  return FusionTensorView(&*impl_->tensor_view);
}

FusionValueView::FusionValueView(const Impl* impl) : impl_(impl) {}

std::string_view FusionValueView::Name() const {
  if (impl_ == nullptr) return {};
  impl_->recorder->RecordValueIdentity(
      impl_->pattern_value, fusion_rewriter_internal::kValueNameField);
  return impl_->name;
}

bool FusionValueView::Exists() const {
  if (impl_ == nullptr) return false;
  impl_->recorder->RecordValueIdentity(
      impl_->pattern_value, fusion_rewriter_internal::kValueExistsField);
  return impl_->exists;
}

FusionTypeView FusionValueView::Type() const {
  return FusionTypeView(impl_ == nullptr ? nullptr : &impl_->type);
}

FusionShapeView FusionValueView::Shape() const {
  return FusionShapeView(impl_ == nullptr ? nullptr : &impl_->shape);
}

std::optional<FusionPatternNodeId>
FusionValueView::MatchedProducer() const {
  if (impl_ == nullptr) return std::nullopt;
  impl_->recorder->RecordMatchedProducer(impl_->pattern_value);
  return impl_->matched_producer;
}

std::optional<size_t> FusionValueView::ProducerOutputIndex() const {
  if (impl_ == nullptr) return std::nullopt;
  impl_->recorder->RecordMatchedProducer(impl_->pattern_value);
  return impl_->producer_output_index;
}

FusionNodeView::FusionNodeView(const Impl* impl) : impl_(impl) {}

NodeIndex FusionNodeView::Index() const {
  if (impl_ == nullptr) return 0;
  impl_->recorder->RecordNodeIdentity(
      impl_->pattern_node, fusion_rewriter_internal::kNodeIndexField);
  return impl_->index;
}

std::string_view FusionNodeView::Domain() const {
  if (impl_ == nullptr) return {};
  impl_->recorder->RecordNodeIdentity(
      impl_->pattern_node, fusion_rewriter_internal::kNodeDomainField);
  return impl_->domain;
}

std::string_view FusionNodeView::OpType() const {
  if (impl_ == nullptr) return {};
  impl_->recorder->RecordNodeIdentity(
      impl_->pattern_node, fusion_rewriter_internal::kNodeOpTypeField);
  return impl_->op_type;
}

std::string_view FusionNodeView::Overload() const {
  if (impl_ == nullptr) return {};
  impl_->recorder->RecordNodeIdentity(
      impl_->pattern_node, fusion_rewriter_internal::kNodeOverloadField);
  return impl_->overload;
}

int FusionNodeView::SinceVersion() const {
  if (impl_ == nullptr) return -1;
  impl_->recorder->RecordNodeIdentity(
      impl_->pattern_node, fusion_rewriter_internal::kNodeVersionField);
  return impl_->since_version;
}

gsl::span<const FusionPatternValueId> FusionNodeView::Inputs() const {
  if (impl_ == nullptr) return {};
  impl_->recorder->RecordNodeSlots(impl_->pattern_node, true, false);
  return impl_->inputs;
}

gsl::span<const FusionPatternValueId> FusionNodeView::Outputs() const {
  if (impl_ == nullptr) return {};
  impl_->recorder->RecordNodeSlots(impl_->pattern_node, false, true);
  return impl_->outputs;
}

FusionAttributeView FusionNodeView::EffectiveAttribute(
    std::string_view name) const {
  if (impl_ == nullptr) return FusionAttributeView(nullptr);
  impl_->recorder->RecordEffectiveAttribute(impl_->pattern_node, name);
  FusionAttributeView::Impl attribute;
  ONNX_NAMESPACE::AttributeProto canonical;
  const auto* observed = impl_->recorder->EffectiveAttribute(
      impl_->pattern_node, name, canonical);
  attribute.exists = observed != nullptr;
  if (observed != nullptr) attribute.canonical = std::move(canonical);
  impl_->attributes.push_back(std::move(attribute));
  return FusionAttributeView(&impl_->attributes.back());
}

FusionMatchContext::FusionMatchContext(const Impl* impl) : impl_(impl) {}

FusionNodeView FusionMatchContext::MatchedNode(
    FusionPatternNodeId id) const {
  if (impl_ == nullptr || id >= impl_->pattern->nodes.size()) {
    return FusionNodeView(nullptr);
  }
  FusionNodeView::Impl node_view;
  node_view.recorder = impl_->recorder;
  node_view.pattern_node = id;
  const auto* node = impl_->recorder->TargetNode(id);
  if (node != nullptr) {
    node_view.index = node->Index();
    node_view.domain = node->Domain();
    node_view.op_type = node->OpType();
    node_view.overload = node->Overload();
    node_view.since_version = node->SinceVersion();
  }
  for (const auto value : impl_->pattern->nodes[id].input_value_ids) {
    node_view.inputs.push_back(value);
  }
  for (const auto value : impl_->pattern->nodes[id].output_value_ids) {
    node_view.outputs.push_back(value);
  }
  impl_->nodes.push_back(std::move(node_view));
  return FusionNodeView(&impl_->nodes.back());
}

FusionValueView FusionMatchContext::BoundValue(
    FusionPatternValueId id) const {
  if (impl_ == nullptr || id >= impl_->pattern->values.size()) {
    return FusionValueView(nullptr);
  }
  impl_->values.push_back(impl_->MakeValue(id));
  return FusionValueView(&impl_->values.back());
}

FusionValueView FusionMatchContext::BoundInput(size_t index) const {
  if (impl_ == nullptr ||
      index >= impl_->pattern->formal_input_value_ids.size()) {
    return FusionValueView(nullptr);
  }
  return BoundValue(impl_->pattern->formal_input_value_ids[index]);
}

FusionValueView FusionMatchContext::BoundOutput(size_t index) const {
  if (impl_ == nullptr ||
      index >= impl_->pattern->formal_output_value_ids.size()) {
    return FusionValueView(nullptr);
  }
  return BoundValue(impl_->pattern->formal_output_value_ids[index]);
}

FusionLiteralView FusionMatchContext::Literal(
    FusionPatternValueId id) const {
  if (impl_ == nullptr || id >= impl_->pattern->values.size() ||
      !impl_->pattern->values[id].is_literal) {
    return FusionLiteralView(nullptr);
  }
  impl_->recorder->RecordLiteral(id);
  FusionLiteralView::Impl literal;
  const auto dependencies = impl_->recorder->Dependencies();
  const auto dependency = std::find_if(
      dependencies.begin(), dependencies.end(),
      [id](const fusion_rewriter_internal::DependencySnapshot& item) {
        return item.kind ==
                   fusion_rewriter_internal::DependencyKind::kLiteral &&
               item.pattern_value == id;
      });
  if (dependency != dependencies.end()) {
    literal.is_initializer = dependency->literal.is_initializer;
    literal.tensor = dependency->literal.canonical_tensor;
  }
  impl_->literals.push_back(std::move(literal));
  return FusionLiteralView(&impl_->literals.back());
}

FusionAttributeView FusionMatchContext::BoundAttribute(
    FusionFormalAttributeId id) const {
  if (impl_ == nullptr ||
      id >= impl_->pattern->formal_attributes.size()) {
    return FusionAttributeView(nullptr);
  }
  impl_->recorder->RecordFormalAttribute(id);
  FusionAttributeView::Impl attribute;
  const auto* observed = impl_->recorder->FormalAttribute(id);
  attribute.exists = observed != nullptr;
  if (observed != nullptr) attribute.canonical = *observed;
  impl_->attributes.push_back(std::move(attribute));
  return FusionAttributeView(&impl_->attributes.back());
}

FusionAttributeView FusionMatchContext::EffectiveAttribute(
    FusionPatternNodeId node, std::string_view name) const {
  if (impl_ == nullptr || node >= impl_->pattern->nodes.size()) {
    return FusionAttributeView(nullptr);
  }
  impl_->recorder->RecordEffectiveAttribute(node, name);
  FusionAttributeView::Impl attribute;
  ONNX_NAMESPACE::AttributeProto canonical;
  const auto* observed =
      impl_->recorder->EffectiveAttribute(node, name, canonical);
  attribute.exists = observed != nullptr;
  if (observed != nullptr) attribute.canonical = std::move(canonical);
  impl_->attributes.push_back(std::move(attribute));
  return FusionAttributeView(&impl_->attributes.back());
}

common::Status FusionPredicateInvoker::InvokePredicate(
    const FusionMatchPredicate& predicate,
    const function_extractor_internal::NormalizedFunctionPattern& pattern,
    const function_extractor_internal::CompiledFunctionPattern&
        compiled_pattern,
    const function_extractor_internal::MatchState& match,
    const function_extractor_internal::TargetGraphSnapshot& snapshot,
    fusion_rewriter_internal::DependencyRecorder& recorder,
    FusionConditionResult& result) {
  ORT_UNUSED_PARAMETER(match);
  ORT_UNUSED_PARAMETER(snapshot);
  result = {};
  if (!predicate) return common::Status::OK();
  FusionMatchContext::Impl impl;
  impl.pattern = &pattern;
  impl.compiled_pattern = &compiled_pattern;
  impl.recorder = &recorder;
  const FusionMatchContext context(&impl);
  ORT_RETURN_IF_ERROR(predicate(context, result));
  ORT_RETURN_IF_ERROR(recorder.Status());
  return common::Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
