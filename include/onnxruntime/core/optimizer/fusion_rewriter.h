#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include "core/graph/onnx_protobuf.h"
#include "gsl/gsl"

namespace onnxruntime {

class Graph;
class Model;
namespace fusion_rewriter_internal {
class FusionRuleSetTestAccess;
class FusionDiagnosticsTestAccess;
class FusionRuleSetExecution;
class FusionDiagnosticsAccess;
}  // namespace fusion_rewriter_internal

using PatternFunctionProto = ONNX_NAMESPACE::FunctionProto;
using FusionRuleId = uint32_t;
using FusionConstraintId = uint32_t;
using FusionPatternNodeId = size_t;
using FusionPatternValueId = size_t;
using FusionFormalAttributeId = size_t;

/** Controls whether an unknown fact rejects a positive constraint. */
enum class FusionUnknownPolicy : uint8_t {
  kReject,
  kNotContradicted,
};

enum class FusionValueRefKind : uint8_t {
  kFormalInput,
  kFormalOutput,
  kPatternValue,
};

struct FusionValueRef {
  /** References a formal or internal pattern value, never a target NodeArg. */
  FusionValueRefKind kind;
  size_t index;

  static FusionValueRef FormalInput(size_t index);
  static FusionValueRef FormalOutput(size_t index);
  static FusionValueRef PatternValue(FusionPatternValueId id);
};

struct FusionNodeRef {
  /** Identifies a pattern operation node. */
  FusionPatternNodeId id;
};

struct FusionDimRef {
  FusionValueRef value;
  int64_t axis;
};

enum class FusionAttributeRefKind : uint8_t {
  kFormalAttribute,
  kEffectiveNodeAttribute,
};

struct FusionAttributeRef {
  FusionAttributeRefKind kind;
  FusionFormalAttributeId formal_attribute_id;
  FusionNodeRef node;
  std::string operator_attribute_name;

  static FusionAttributeRef Formal(FusionFormalAttributeId id);
  static FusionAttributeRef Effective(FusionNodeRef node,
                                      std::string operator_attribute_name);
};

struct FusionDimensionEquivalenceClass {
  /** All listed dimensions must be equal under `unknown_policy`. */
  std::string label;
  std::vector<FusionDimRef> dimensions;
  FusionUnknownPolicy unknown_policy{FusionUnknownPolicy::kReject};
};

class FusionConstraint final {
 public:
  /** A composable declarative condition evaluated after complete binding. */
  FusionConstraint();
  ~FusionConstraint();
  FusionConstraint(const FusionConstraint&);
  FusionConstraint& operator=(const FusionConstraint&);
  FusionConstraint(FusionConstraint&&) noexcept;
  FusionConstraint& operator=(FusionConstraint&&) noexcept;

  static FusionConstraint AllOf(std::vector<FusionConstraint> operands);
  static FusionConstraint AnyOf(std::vector<FusionConstraint> operands);
  static FusionConstraint Not(FusionConstraint operand);
  static FusionConstraint IsPresent(FusionValueRef value);
  static FusionConstraint IsMissing(FusionValueRef value);
  static FusionConstraint IsTensor(FusionValueRef value);
  static FusionConstraint ElementTypeIs(FusionValueRef value, int32_t elem_type);
  static FusionConstraint ElementTypeIn(FusionValueRef value,
                                        std::vector<int32_t> elem_types);
  static FusionConstraint TypeEquals(
      FusionValueRef lhs, FusionValueRef rhs,
      FusionUnknownPolicy policy = FusionUnknownPolicy::kReject);
  static FusionConstraint SameElementType(
      FusionValueRef lhs, FusionValueRef rhs,
      FusionUnknownPolicy policy = FusionUnknownPolicy::kReject);
  static FusionConstraint RankIs(FusionValueRef value, size_t rank);
  static FusionConstraint RankIn(FusionValueRef value,
                                 size_t minimum, size_t maximum);
  static FusionConstraint SameRank(
      FusionValueRef lhs, FusionValueRef rhs,
      FusionUnknownPolicy policy = FusionUnknownPolicy::kReject);
  static FusionConstraint DimValueIs(FusionDimRef dim, int64_t value);
  static FusionConstraint DimEquals(
      FusionDimRef lhs, FusionDimRef rhs,
      FusionUnknownPolicy policy = FusionUnknownPolicy::kReject);
  static FusionConstraint ShapeEquals(
      FusionValueRef lhs, FusionValueRef rhs,
      FusionUnknownPolicy policy = FusionUnknownPolicy::kReject);
  static FusionConstraint AttributePresent(FusionAttributeRef attribute);
  static FusionConstraint AttributeTypeIs(
      FusionAttributeRef attribute,
      ONNX_NAMESPACE::AttributeProto_AttributeType type);
  static FusionConstraint AttributeEquals(
      FusionAttributeRef attribute,
      ONNX_NAMESPACE::AttributeProto canonical_literal);
  static FusionConstraint AttributeIn(
      FusionAttributeRef attribute,
      std::vector<ONNX_NAMESPACE::AttributeProto> canonical_literals);
  static FusionConstraint IntAttributeInRange(
      FusionAttributeRef attribute, int64_t minimum, int64_t maximum);
  static FusionConstraint FloatAttributeInRange(
      FusionAttributeRef attribute, float minimum, float maximum);
  static FusionConstraint StringAttributeIn(
      FusionAttributeRef attribute, std::vector<std::string> values);
  static FusionConstraint SameAttributeValue(
      FusionAttributeRef lhs, FusionAttributeRef rhs,
      FusionUnknownPolicy policy = FusionUnknownPolicy::kReject);

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
  explicit FusionConstraint(std::shared_ptr<const Impl> impl);
  friend class FusionConstraintProgram;
};

class FusionConstraintProgram final {
 public:
  /** Combines dimension-unification classes with a constraint predicate. */
  FusionConstraintProgram(
      std::vector<FusionDimensionEquivalenceClass> dimension_classes,
      FusionConstraint predicate);
  ~FusionConstraintProgram();
  FusionConstraintProgram(const FusionConstraintProgram&);
  FusionConstraintProgram& operator=(const FusionConstraintProgram&);
  FusionConstraintProgram(FusionConstraintProgram&&) noexcept;
  FusionConstraintProgram& operator=(FusionConstraintProgram&&) noexcept;

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
  friend class FusionRule;
};

struct FusionReplacementInput {
  /** Nullopt emits an omitted optional input position. */
  std::optional<size_t> formal_input_index;
};

struct FusionReplacementOutput {
  size_t formal_output_index;
};

enum class FusionReplacementAttributeSource : uint8_t {
  kFormalAttribute,
  kLiteral,
};

struct FusionReplacementAttribute {
  std::string emitted_name;
  FusionReplacementAttributeSource source;
  FusionFormalAttributeId formal_attribute_id{};
  ONNX_NAMESPACE::AttributeProto literal;
};

struct FusionReplacementCall {
  /** Replacement identity is independent of the pattern function identity. */
  std::string domain;
  std::string op_type;
  int since_version{-1};
  std::string overload;
  std::vector<FusionReplacementInput> inputs;
  std::vector<FusionReplacementOutput> outputs;
  std::vector<FusionReplacementAttribute> attributes;
};

enum class FusionDimensionKind : uint8_t {
  kUnknown,
  kValue,
  kSymbol,
};

/**
 * Read-only dimension observed from the matched target graph.
 *
 * Instances are callback-only views. They and all data returned from them are
 * valid only for the current FusionMatchPredicate invocation.
 */
class FusionDimensionView final {
 public:
  FusionDimensionKind Kind() const;
  std::optional<int64_t> Value() const;
  std::optional<std::string_view> Symbol() const;

 private:
  struct Impl;
  const Impl* impl_{};
  explicit FusionDimensionView(const Impl*);
  friend class FusionShapeView;
};

/**
 * Callback-only target shape view.
 *
 * An unknown shape has `HasRank() == false`; `Rank()` then returns 0, which is
 * also the valid rank of a scalar. Call `HasRank()` before interpreting 0.
 */
class FusionShapeView final {
 public:
  bool HasRank() const;
  size_t Rank() const;
  std::optional<FusionDimensionView> Dimension(size_t axis) const;

 private:
  struct Impl;
  const Impl* impl_{};
  explicit FusionShapeView(const Impl*);
  friend class FusionValueView;
  friend class FusionMatchContext;
};

/** Callback-only target type view. Unknown and non-tensor types are explicit. */
class FusionTypeView final {
 public:
  bool IsTensor() const;
  std::optional<int32_t> TensorElementType() const;

 private:
  struct Impl;
  const Impl* impl_{};
  explicit FusionTypeView(const Impl*);
  friend class FusionValueView;
  friend class FusionMatchContext;
};

/** Callback-only canonical tensor-literal view. */
class FusionTensorView final {
 public:
  int32_t ElementType() const;
  gsl::span<const int64_t> Dimensions() const;
  gsl::span<const std::byte> LogicalData() const;

 private:
  struct Impl;
  const Impl* impl_{};
  explicit FusionTensorView(const Impl*);
  friend class FusionAttributeView;
  friend class FusionLiteralView;
};

/**
 * Callback-only target attribute view.
 *
 * `Exists() == false` is the missing-attribute sentinel; typed accessors then
 * return std::nullopt.
 */
class FusionAttributeView final {
 public:
  bool Exists() const;
  ONNX_NAMESPACE::AttributeProto_AttributeType Type() const;
  std::optional<float> Float() const;
  std::optional<int64_t> Int() const;
  std::optional<std::string_view> String() const;
  std::optional<gsl::span<const float>> Floats() const;
  std::optional<gsl::span<const int64_t>> Ints() const;
  std::optional<std::vector<std::string_view>> Strings() const;
  std::optional<FusionTensorView> Tensor() const;
  std::optional<std::vector<FusionTensorView>> Tensors() const;

 private:
  struct Impl;
  const Impl* impl_{};
  explicit FusionAttributeView(const Impl*);
  friend class FusionNodeView;
  friend class FusionMatchContext;
};

/** Callback-only normalized literal witness from the matched target graph. */
class FusionLiteralView final {
 public:
  bool IsInitializer() const;
  FusionTensorView Tensor() const;

 private:
  struct Impl;
  const Impl* impl_{};
  explicit FusionLiteralView(const Impl*);
  friend class FusionMatchContext;
};

/**
 * Callback-only target value view selected by a pattern value reference.
 *
 * `Exists() == false` represents a stably omitted optional binding. Its name is
 * empty, type/shape are unknown, and producer accessors return std::nullopt.
 */
class FusionValueView final {
 public:
  std::string_view Name() const;
  bool Exists() const;
  FusionTypeView Type() const;
  FusionShapeView Shape() const;
  std::optional<FusionPatternNodeId> MatchedProducer() const;
  std::optional<size_t> ProducerOutputIndex() const;

 private:
  struct Impl;
  const Impl* impl_{};
  explicit FusionValueView(const Impl*);
  friend class FusionMatchContext;
};

/**
 * Callback-only target operation-node view selected by a pattern operation
 * node ID. Inputs()/Outputs() return pattern value IDs for further lookup.
 */
class FusionNodeView final {
 public:
  NodeIndex Index() const;
  std::string_view Domain() const;
  std::string_view OpType() const;
  std::string_view Overload() const;
  int SinceVersion() const;
  gsl::span<const FusionPatternValueId> Inputs() const;
  gsl::span<const FusionPatternValueId> Outputs() const;
  FusionAttributeView EffectiveAttribute(std::string_view name) const;

 private:
  struct Impl;
  const Impl* impl_{};
  explicit FusionNodeView(const Impl*);
  friend class FusionMatchContext;
};

/**
 * Complete match exposed to a FusionMatchPredicate.
 *
 * Accessor arguments identify pattern entities; returned views describe their
 * bound target entities. This object and every view obtained from it must not
 * be retained after the callback returns.
 */
class FusionMatchContext final {
 public:
  /** Returns the target operation node bound to pattern operation node `id`. */
  FusionNodeView MatchedNode(FusionPatternNodeId id) const;
  /** Returns the target value bound to an internal pattern value. */
  FusionValueView BoundValue(FusionPatternValueId id) const;
  /** Returns the target value bound to a pattern formal input. */
  FusionValueView BoundInput(size_t index) const;
  /** Returns the target value bound to a pattern formal output. */
  FusionValueView BoundOutput(size_t index) const;
  FusionLiteralView Literal(FusionPatternValueId id) const;
  FusionAttributeView BoundAttribute(FusionFormalAttributeId id) const;
  FusionAttributeView EffectiveAttribute(FusionPatternNodeId node,
                                         std::string_view name) const;

 private:
  struct Impl;
  const Impl* impl_{};
  explicit FusionMatchContext(const Impl*);
  friend struct FusionPredicateInvoker;
};

enum class FusionConditionDecision : uint8_t {
  kSatisfied,
  kNotSatisfied,
};

struct FusionConditionFailure {
  std::string reason;
  std::optional<FusionPatternNodeId> node;
  std::optional<FusionPatternValueId> value;
  std::optional<FusionFormalAttributeId> attribute;
};

struct FusionConditionResult {
  FusionConditionDecision decision{FusionConditionDecision::kSatisfied};
  std::optional<FusionConditionFailure> failure;
};

using FusionMatchPredicate = std::function<common::Status(
    const FusionMatchContext&, FusionConditionResult&)>;

struct FusionRuleOptions {
  FusionRuleId id{};
  std::string name;
  /**
   * Tie-breaker among rules matching the same anchor. An anchor is the target
   * operation node that produces the rule's primary pattern output group.
   */
  int32_t anchor_local_priority{};
};

/** Immutable pattern, condition, and replacement-call definition. */
class FusionRule final {
 public:
  FusionRule(const PatternFunctionProto& pattern,
             FusionReplacementCall replacement,
             FusionConstraintProgram constraints,
             FusionMatchPredicate predicate,
             FusionRuleOptions options);
  ~FusionRule();
  FusionRule(FusionRule&&) noexcept;
  FusionRule& operator=(FusionRule&&) noexcept;
  FusionRule(const FusionRule&) = delete;
  FusionRule& operator=(const FusionRule&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class FusionRuleSet;
};

/**
 * Controls diagnostics and dry-run behavior.
 *
 * `kOff` allocates and records nothing. `kBestFailure` retains the best failure
 * for each unsuccessful rule. `kAllFailures` retains bounded failure and
 * success records. `kDryRun` records as `kAllFailures` but suppresses graph
 * mutation after discovery and prevalidation.
 */
enum class FusionDiagnosticMode : uint8_t {
  kOff,
  kBestFailure,
  kAllFailures,
  kDryRun,
};

enum class FusionMatchStage : uint8_t {
  kRootSignature,
  kStructuralNode,
  kStructuralEdge,
  kValueBinding,
  kAttributeBinding,
  kLiteral,
  kCondition,
  kClosure,
  kConvexity,
  kFinalValidation,
  kPrevalidation,
  kSuccess,
};

enum class FusionFailureCode : uint16_t {
  kOpMismatch,
  kOutputSlotMismatch,
  kRepeatedBindingMismatch,
  kMissingEffectiveAttribute,
  kAttributeValueMismatch,
  kLiteralMismatch,
  kUnknownRank,
  kDimensionMismatch,
  kConstraintFalse,
  kCallbackRejected,
  kExternalPrivateUse,
  kNonConvex,
  kStalePlan,
};

/** Bounded diagnostic record for one unsuccessful or successful match. */
struct FusionFailureRecord {
  FusionRuleId rule_id{};
  FusionMatchStage stage{};
  FusionFailureCode code{};
  NodeIndex anchor_node{};
  size_t anchor_output_slot{};
  std::optional<FusionPatternNodeId> pattern_node;
  std::optional<FusionPatternValueId> pattern_value;
  std::optional<FusionConstraintId> constraint;
  std::optional<NodeIndex> target_node;
  std::optional<size_t> target_slot;
  std::string target_value_name;
  size_t pattern_nodes_matched{};
  std::string detail;
};

/**
 * Caller-owned bounded diagnostic collector.
 *
 * Apply resets a supplied collector before each invocation. The collector must
 * be non-null whenever diagnostics are enabled and may be null only for
 * `kOff`.
 */
class FusionTraceCollector final {
 public:
  FusionTraceCollector();
  ~FusionTraceCollector();
  gsl::span<const FusionFailureRecord> BestFailures() const;
  gsl::span<const FusionFailureRecord> Records() const;
  size_t SuccessCount(FusionRuleId rule_id) const;
  bool Truncated() const;
  std::string Format() const;
  void Clear();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class FusionRuleSet;
  friend class fusion_rewriter_internal::FusionDiagnosticsAccess;
  friend class fusion_rewriter_internal::FusionDiagnosticsTestAccess;
};

/** Semantic, resource, fixpoint, and diagnostic limits for a FusionRuleSet. */
struct FusionRuleSetOptions {
  size_t max_pattern_nodes{1024};
  size_t max_target_nodes{1'000'000};
  size_t max_output_root_tuples{100'000};
  size_t max_work_units{1'000'000};
  size_t max_literal_bytes{64U * 1024U * 1024U};
  size_t max_formal_attributes{256};
  size_t max_attribute_bytes{64U * 1024U * 1024U};
  size_t max_rules{1024};
  size_t max_constraint_nodes{4096};
  size_t max_dimension_equivalence_operands{4096};
  size_t max_rule_attempts{10'000'000};
  size_t max_condition_evaluations{1'000'000};
  size_t max_epochs{1'000'000};
  size_t max_replacements{1'000'000};
  size_t max_diagnostic_records{1024};
  size_t max_diagnostic_bytes{1U * 1024U * 1024U};
  FusionDiagnosticMode diagnostic_mode{FusionDiagnosticMode::kOff};
};

/** Exact mutation progress and terminal status from one Apply invocation. */
struct FusionRewriteResult {
  common::Status status{common::Status::OK()};
  size_t replacements_applied{};
  size_t epochs_completed{};
};

/**
 * Applies rules over immutable graph epochs until no replacement remains.
 *
 * Each epoch discovers all rules against one graph snapshot, selects
 * non-conflicting plans consumer-anchor first, prevalidates the whole batch,
 * applies it, resolves the graph, and requires a strict node-count decrease.
 */
class FusionRuleSet final {
 public:
  explicit FusionRuleSet(std::vector<FusionRule> rules,
                         FusionRuleSetOptions options = {});
  ~FusionRuleSet();
  FusionRewriteResult Apply(Model& model,
                            FusionTraceCollector* trace = nullptr) const;
  FusionRewriteResult Apply(Graph& graph,
                            FusionTraceCollector* trace = nullptr) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class fusion_rewriter_internal::FusionRuleSetExecution;
  friend class fusion_rewriter_internal::FusionRuleSetTestAccess;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(FusionRuleSet);
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
