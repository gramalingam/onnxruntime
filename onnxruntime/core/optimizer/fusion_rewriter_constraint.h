#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/common/status.h"
#include "core/optimizer/function_extractor_matcher.h"
#include "core/optimizer/fusion_rewriter.h"
#include "gsl/gsl"

namespace onnxruntime::fusion_rewriter_internal {

enum class ConstraintKind : uint8_t {
  kAllOf,
  kAnyOf,
  kNot,
  kIsPresent,
  kIsMissing,
  kIsTensor,
  kElementTypeIs,
  kElementTypeIn,
  kTypeEquals,
  kSameElementType,
  kRankIs,
  kRankIn,
  kSameRank,
  kDimValueIs,
  kDimEquals,
  kShapeEquals,
  kAttributePresent,
  kAttributeTypeIs,
  kAttributeEquals,
  kAttributeIn,
  kIntAttributeInRange,
  kFloatAttributeInRange,
  kStringAttributeIn,
  kSameAttributeValue,
};

struct ConstraintExpr {
  ConstraintKind kind{ConstraintKind::kAllOf};
  std::vector<std::shared_ptr<const ConstraintExpr>> operands;
  FusionValueRef lhs_value{FusionValueRefKind::kPatternValue, 0};
  FusionValueRef rhs_value{FusionValueRefKind::kPatternValue, 0};
  FusionDimRef lhs_dim{{FusionValueRefKind::kPatternValue, 0}, 0};
  FusionDimRef rhs_dim{{FusionValueRefKind::kPatternValue, 0}, 0};
  FusionAttributeRef lhs_attribute{
      FusionAttributeRefKind::kFormalAttribute, 0, {}, {}};
  FusionAttributeRef rhs_attribute{
      FusionAttributeRefKind::kFormalAttribute, 0, {}, {}};
  FusionUnknownPolicy unknown_policy{FusionUnknownPolicy::kReject};
  size_t minimum_rank{};
  size_t maximum_rank{};
  int32_t element_type{};
  std::vector<int32_t> element_types;
  int64_t integer_value{};
  int64_t minimum_integer{};
  int64_t maximum_integer{};
  float minimum_float{};
  float maximum_float{};
  ONNX_NAMESPACE::AttributeProto_AttributeType attribute_type{
      ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED};
  std::vector<ONNX_NAMESPACE::AttributeProto> attribute_literals;
  std::vector<std::string> string_values;
};

struct ConstraintProgramDefinition {
  std::vector<FusionDimensionEquivalenceClass> dimension_classes;
  std::shared_ptr<const ConstraintExpr> predicate;
  common::Status construction_status{common::Status::OK()};
};

struct ResolvedValueRef {
  FusionPatternValueId pattern_value{};
};

struct ResolvedDimRef {
  ResolvedValueRef value;
  int64_t axis{};
};

struct ResolvedAttributeRef {
  FusionAttributeRefKind kind{FusionAttributeRefKind::kFormalAttribute};
  FusionFormalAttributeId formal_attribute{};
  FusionPatternNodeId pattern_node{};
  std::string operator_attribute_name;
};

struct CompiledConstraintNode {
  ConstraintKind kind{ConstraintKind::kAllOf};
  FusionConstraintId id{};
  std::vector<std::shared_ptr<const CompiledConstraintNode>> operands;
  ResolvedValueRef lhs_value;
  ResolvedValueRef rhs_value;
  ResolvedDimRef lhs_dim;
  ResolvedDimRef rhs_dim;
  ResolvedAttributeRef lhs_attribute;
  ResolvedAttributeRef rhs_attribute;
  FusionUnknownPolicy unknown_policy{FusionUnknownPolicy::kReject};
  size_t minimum_rank{};
  size_t maximum_rank{};
  int32_t element_type{};
  std::vector<int32_t> element_types;
  int64_t integer_value{};
  int64_t minimum_integer{};
  int64_t maximum_integer{};
  float minimum_float{};
  float maximum_float{};
  ONNX_NAMESPACE::AttributeProto_AttributeType attribute_type{
      ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED};
  std::vector<ONNX_NAMESPACE::AttributeProto> attribute_literals;
  std::vector<std::string> string_values;
};

struct CompiledDimensionEquivalenceClass {
  std::string label;
  std::vector<ResolvedDimRef> dimensions;
  FusionUnknownPolicy unknown_policy{FusionUnknownPolicy::kReject};
};

struct CompiledConstraintProgram {
  std::vector<CompiledDimensionEquivalenceClass> dimension_classes;
  std::shared_ptr<const CompiledConstraintNode> predicate;
  common::Status construction_status{common::Status::OK()};
  size_t constraint_node_count{};
  size_t dimension_operand_count{};
};

struct ConstraintCompileOptions {
  size_t max_constraint_nodes{4096};
  size_t max_dimension_equivalence_operands{4096};
  size_t max_attribute_bytes{64U * 1024U * 1024U};
};

CompiledConstraintProgram CompileConstraintProgram(
    const ConstraintProgramDefinition& definition,
    const function_extractor_internal::NormalizedFunctionPattern& pattern,
    const ConstraintCompileOptions& options);

enum class DependencyKind : uint8_t {
  kNodeIdentity,
  kNodeSlots,
  kValueIdentity,
  kValueType,
  kValueRank,
  kValueDimension,
  kMatchedProducer,
  kEffectiveAttribute,
  kFormalAttribute,
  kLiteral,
};

struct ValueBindingSite {
  NodeIndex node_index{};
  size_t slot{};
  bool is_output{};
};

struct NodeIdentitySnapshot {
  uint8_t observed_fields{};
  NodeIndex index{};
  std::string domain;
  std::string op_type;
  std::string overload;
  int since_version{-1};
};

struct NodeSlotsSnapshot {
  NodeIndex target_node{};
  bool inputs_observed{};
  bool outputs_observed{};
  std::vector<std::string> input_names;
  std::vector<bool> input_exists;
  std::vector<std::string> output_names;
  std::vector<bool> output_exists;
};

struct ValueIdentitySnapshot {
  uint8_t observed_fields{};
  bool exists{};
  std::string name;
  std::vector<ValueBindingSite> binding_sites;
};

struct ValueTypeSnapshot {
  uint8_t observed_fields{};
  bool has_type{};
  bool is_tensor{};
  int32_t tensor_element_type{};
  ONNX_NAMESPACE::TypeProto canonical_type;
  std::string value_name;
  std::vector<ValueBindingSite> binding_sites;
};

struct ValueRankSnapshot {
  bool has_rank{};
  size_t rank{};
  std::string value_name;
  std::vector<ValueBindingSite> binding_sites;
};

enum class DimensionFactKind : uint8_t {
  kUnknown,
  kValue,
  kSymbol,
};

struct DimensionFact {
  DimensionFactKind kind{DimensionFactKind::kUnknown};
  int64_t value{};
  std::string symbol;
};

struct ValueDimensionSnapshot {
  DimensionFact fact;
  std::string value_name;
  std::vector<ValueBindingSite> binding_sites;
};

struct MatchedProducerSnapshot {
  bool has_producer{};
  FusionPatternNodeId producer_pattern_node{};
  NodeIndex producer_target_node{};
  size_t output_slot{};
  std::string value_name;
  std::vector<ValueBindingSite> binding_sites;
};

struct AttributeSnapshot {
  bool exists{};
  ONNX_NAMESPACE::AttributeProto canonical_value;
  NodeIndex target_node{};
  std::string operator_attribute_name;
};

struct FormalAttributeSnapshot {
  ONNX_NAMESPACE::AttributeProto canonical_value;
  std::vector<std::pair<NodeIndex, std::string>> occurrences;
};

struct LiteralSnapshot {
  bool is_initializer{};
  std::string value_name;
  ONNX_NAMESPACE::TensorProto canonical_tensor;
  std::vector<ValueBindingSite> binding_sites;
};

struct DependencySnapshot {
  DependencyKind kind{DependencyKind::kValueIdentity};
  FusionPatternNodeId pattern_node{};
  FusionPatternValueId pattern_value{};
  FusionFormalAttributeId formal_attribute{};
  std::string name;
  int64_t axis{-1};
  NodeIdentitySnapshot node_identity;
  NodeSlotsSnapshot node_slots;
  ValueIdentitySnapshot value_identity;
  ValueTypeSnapshot value_type;
  ValueRankSnapshot value_rank;
  ValueDimensionSnapshot value_dimension;
  MatchedProducerSnapshot matched_producer;
  AttributeSnapshot attribute;
  FormalAttributeSnapshot formal_attribute_value;
  LiteralSnapshot literal;
};

class DependencyRecorder final {
 public:
  DependencyRecorder(
      const function_extractor_internal::NormalizedFunctionPattern& pattern,
      const function_extractor_internal::MatchState& match,
      const function_extractor_internal::TargetGraphSnapshot& snapshot,
      size_t max_attribute_bytes,
      size_t max_literal_bytes);

  const common::Status& Status() const noexcept { return status_; }
  std::vector<DependencySnapshot> TakeSnapshot();
  gsl::span<const DependencySnapshot> Dependencies() const noexcept {
    return dependencies_;
  }

  const Node* TargetNode(FusionPatternNodeId pattern_node) const;
  const NodeArg* TargetValue(FusionPatternValueId pattern_value) const;
  const ONNX_NAMESPACE::AttributeProto* EffectiveAttribute(
      FusionPatternNodeId pattern_node, std::string_view name,
      ONNX_NAMESPACE::AttributeProto& canonical);
  const ONNX_NAMESPACE::AttributeProto* FormalAttribute(
      FusionFormalAttributeId formal_attribute) const;
  const function_extractor_internal::LiteralWitness* LiteralWitness(
      FusionPatternValueId pattern_value) const;

  void RecordNodeIdentity(FusionPatternNodeId pattern_node, uint8_t fields);
  void RecordNodeSlots(FusionPatternNodeId pattern_node, bool inputs, bool outputs);
  void RecordValueIdentity(FusionPatternValueId pattern_value, uint8_t fields);
  void RecordValueType(FusionPatternValueId pattern_value, uint8_t fields);
  void RecordValueRank(FusionPatternValueId pattern_value);
  void RecordValueDimension(FusionPatternValueId pattern_value, int64_t normalized_axis);
  void RecordMatchedProducer(FusionPatternValueId pattern_value);
  void RecordEffectiveAttribute(FusionPatternNodeId pattern_node, std::string_view name);
  void RecordFormalAttribute(FusionFormalAttributeId formal_attribute);
  void RecordLiteral(FusionPatternValueId pattern_value);

  DimensionFact ReadDimension(
      FusionPatternValueId pattern_value, int64_t axis, bool& rank_known,
      int64_t& normalized_axis);

 private:
  DependencySnapshot& FindOrAdd(
      DependencyKind kind, FusionPatternNodeId pattern_node,
      FusionPatternValueId pattern_value,
      FusionFormalAttributeId formal_attribute,
      std::string_view name, int64_t axis);
  std::vector<ValueBindingSite> BindingSites(
      FusionPatternValueId pattern_value) const;
  void SaveError(common::Status status);

  const function_extractor_internal::NormalizedFunctionPattern& pattern_;
  const function_extractor_internal::MatchState& match_;
  const function_extractor_internal::TargetGraphSnapshot& snapshot_;
  size_t max_attribute_bytes_;
  size_t max_literal_bytes_;
  std::vector<DependencySnapshot> dependencies_;
  common::Status status_{common::Status::OK()};
};

struct ConstraintEvaluationResult {
  bool satisfied{true};
  std::optional<FusionConstraintId> failed_constraint;
  std::string detail;
};

common::Status EvaluateConstraintProgram(
    const CompiledConstraintProgram& program,
    DependencyRecorder& recorder,
    ConstraintEvaluationResult& result);

common::Status PrevalidateDependencies(
    const Graph& graph, gsl::span<const DependencySnapshot> dependencies,
    size_t max_attribute_bytes, size_t max_literal_bytes);

}  // namespace onnxruntime::fusion_rewriter_internal

namespace onnxruntime {

struct FusionConstraint::Impl {
  std::shared_ptr<const fusion_rewriter_internal::ConstraintExpr> expression;
};

struct FusionConstraintProgram::Impl {
  std::shared_ptr<const fusion_rewriter_internal::ConstraintProgramDefinition>
      definition;
};

// This friend of FusionMatchContext is the sole internal constructor path for
// the ephemeral opaque callback view.
struct FusionPredicateInvoker {
  static common::Status InvokePredicate(
      const FusionMatchPredicate& predicate,
      const function_extractor_internal::NormalizedFunctionPattern& pattern,
      const function_extractor_internal::CompiledFunctionPattern& compiled_pattern,
      const function_extractor_internal::MatchState& match,
      const function_extractor_internal::TargetGraphSnapshot& snapshot,
      fusion_rewriter_internal::DependencyRecorder& recorder,
      FusionConditionResult& result);
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
