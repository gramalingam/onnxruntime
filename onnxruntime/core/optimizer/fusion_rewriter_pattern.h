#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <memory>
#include <string>

#include "core/optimizer/function_extractor_matcher.h"
#include "core/optimizer/function_extractor_pattern.h"
#include "core/optimizer/fusion_rewriter_constraint.h"
#include "core/optimizer/fusion_rewriter.h"
#include "core/optimizer/fusion_rewriter_matcher.h"

namespace onnxruntime::fusion_rewriter_internal {

struct FusionRuleInternal {
  FusionRuleInternal(
      const PatternFunctionProto& pattern,
      FusionReplacementCall replacement,
      const ConstraintProgramDefinition& constraints,
      FusionMatchPredicate predicate,
      FusionRuleOptions options,
      const FusionRuleSetOptions& rule_set_options);

  PatternFunctionProto pattern;
  FusionReplacementCall replacement;
  CompiledConstraintProgram constraints;
  FusionMatchPredicate predicate;
  FusionRuleOptions options;
  FunctionExtractorOptions matcher_options;
  function_extractor_internal::NormalizedFunctionPattern normalized_pattern;
  common::Status construction_status{common::Status::OK()};
};

struct CompiledFusionRule {
  const FusionRuleInternal* rule{};
  function_extractor_internal::CompiledFunctionPattern compiled_pattern;
  const ONNX_NAMESPACE::OpSchema* replacement_schema{};
  std::string canonical_replacement_domain;
};

struct FusionRuleSetState {
  FusionRuleSetOptions options;
  std::vector<std::unique_ptr<FusionRuleInternal>> normalized_rules;
  common::Status construction_status{common::Status::OK()};
};

struct FusionReplacementPlan {
  function_extractor_internal::ReplacementPlan base;
  FusionRuleId rule_id{};
  int32_t anchor_local_priority{};
  size_t registration_order{};
  size_t tuple_ordinal{};
  std::string replacement_domain;
  std::string replacement_op_type;
  std::string replacement_overload;
  int replacement_since_version{-1};
  InlinedVector<NodeArg*> call_inputs;
  InlinedVector<NodeArg*> call_outputs;
  NodeAttributes call_attributes;
  std::vector<DependencySnapshot> dependencies;
  InlinedVector<ObservedDependencySummary> observed_dependencies;
};

struct FusionConditionEvidence {
  std::vector<DependencySnapshot> dependencies;
};

FunctionExtractorOptions MakeMatcherOptions(
    const FusionRuleSetOptions& options);

common::Status CompileFusionRule(
    const FusionRuleInternal& rule,
    const Graph& graph,
    CompiledFusionRule& compiled_rule);

common::Status ValidateReplacementMappings(
    const FusionRuleInternal& rule);

common::Status MaterializeFusionReplacementPlan(
    const CompiledFusionRule& compiled_rule,
    function_extractor_internal::ReplacementPlan base_plan,
    size_t registration_order,
    size_t tuple_ordinal,
    FusionReplacementPlan& fusion_plan);

}  // namespace onnxruntime::fusion_rewriter_internal

#endif  // !defined(ORT_MINIMAL_BUILD)
