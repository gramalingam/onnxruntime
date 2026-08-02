#include "core/optimizer/fusion_rewriter.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <limits>
#include <unordered_set>
#include <utility>

#include "core/graph/model.h"
#include "core/optimizer/function_extractor_matcher.h"
#include "core/optimizer/fusion_rewriter_constraint.h"
#include "core/optimizer/fusion_rewriter_diagnostics.h"
#include "core/optimizer/fusion_rewriter_matcher.h"
#include "core/optimizer/fusion_rewriter_pattern.h"

namespace onnxruntime {

struct FusionRule::Impl {
  Impl(const PatternFunctionProto& pattern_proto,
       FusionReplacementCall replacement_call,
       FusionConstraintProgram constraint_program,
       FusionMatchPredicate match_predicate,
       FusionRuleOptions rule_options)
      : pattern(pattern_proto),
        replacement(std::move(replacement_call)),
        constraints(std::move(constraint_program)),
        predicate(std::move(match_predicate)),
        options(std::move(rule_options)) {
    constraint_definition = constraints.impl_->definition;
  }

  PatternFunctionProto pattern;
  FusionReplacementCall replacement;
  FusionConstraintProgram constraints;
  std::shared_ptr<const fusion_rewriter_internal::ConstraintProgramDefinition>
      constraint_definition;
  FusionMatchPredicate predicate;
  FusionRuleOptions options;
};
struct FusionRuleSet::Impl
    : fusion_rewriter_internal::FusionRuleSetState {
  Impl(std::vector<FusionRule> rules, FusionRuleSetOptions rule_set_options) {
    options = std::move(rule_set_options);
    if (rules.size() > options.max_rules) {
      construction_status = ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "FusionRuleSet rule budget exceeded.");
      return;
    }
    std::unordered_set<FusionRuleId> rule_ids;
    normalized_rules.reserve(rules.size());
    for (auto& rule : rules) {
      if (rule.impl_ == nullptr) {
        construction_status = ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "FusionRuleSet contains a moved-from rule.");
        return;
      }
      if (!rule_ids.insert(rule.impl_->options.id).second) {
        construction_status = ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "FusionRuleSet contains duplicate RuleId ",
            rule.impl_->options.id, ".");
        return;
      }
      normalized_rules.push_back(
          std::make_unique<fusion_rewriter_internal::NormalizedFusionRule>(
              rule.impl_->pattern, std::move(rule.impl_->replacement),
              *rule.impl_->constraint_definition,
              std::move(rule.impl_->predicate),
              std::move(rule.impl_->options), options));
      if (!normalized_rules.back()->construction_status.IsOK()) {
        construction_status =
            normalized_rules.back()->construction_status;
        return;
      }
    }
  }
};

FusionRule::FusionRule(
    const PatternFunctionProto& pattern, FusionReplacementCall replacement,
    FusionConstraintProgram constraints, FusionMatchPredicate predicate,
    FusionRuleOptions options)
    : impl_(std::make_unique<Impl>(
          pattern, std::move(replacement), std::move(constraints),
          std::move(predicate), std::move(options))) {}
FusionRule::~FusionRule() = default;
FusionRule::FusionRule(FusionRule&&) noexcept = default;
FusionRule& FusionRule::operator=(FusionRule&&) noexcept = default;

namespace fusion_rewriter_internal {
namespace {

bool PlansConflict(
    const FusionReplacementPlan& lhs,
    const FusionReplacementPlan& rhs) {
  return function_extractor_internal::ReplacementPlansConflict(
      lhs.base, rhs.base);
}

common::Status ValidateGraphForDiscovery(const Graph& graph) {
  ORT_RETURN_IF(graph.GraphResolveNeeded(),
                "FusionRuleSet requires a resolved graph; ",
                "GraphResolveNeeded() is true.");
  for (const auto& node : graph.Nodes()) {
    ORT_RETURN_IF(node.Op() == nullptr,
                  "FusionRuleSet requires every target node to have a ",
                  "resolved schema.");
  }
  return common::Status::OK();
}

FusionMatchStage ToFusionStage(
    function_extractor_internal::MatcherFailureStage stage) {
  using MatcherStage =
      function_extractor_internal::MatcherFailureStage;
  switch (stage) {
    case MatcherStage::kStructuralNode:
      return FusionMatchStage::kStructuralNode;
    case MatcherStage::kStructuralEdge:
      return FusionMatchStage::kStructuralEdge;
    case MatcherStage::kValueBinding:
      return FusionMatchStage::kValueBinding;
    case MatcherStage::kAttributeBinding:
      return FusionMatchStage::kAttributeBinding;
    case MatcherStage::kLiteral:
      return FusionMatchStage::kLiteral;
    case MatcherStage::kClosure:
      return FusionMatchStage::kClosure;
    case MatcherStage::kConvexity:
      return FusionMatchStage::kConvexity;
    case MatcherStage::kFinalValidation:
      return FusionMatchStage::kFinalValidation;
  }
  ORT_THROW("Unknown matcher failure stage.");
}

FusionFailureCode ToFusionCode(
    function_extractor_internal::MatcherFailureCode code) {
  using MatcherCode = function_extractor_internal::MatcherFailureCode;
  switch (code) {
    case MatcherCode::kOpMismatch:
      return FusionFailureCode::kOpMismatch;
    case MatcherCode::kOutputSlotMismatch:
      return FusionFailureCode::kOutputSlotMismatch;
    case MatcherCode::kRepeatedBindingMismatch:
      return FusionFailureCode::kRepeatedBindingMismatch;
    case MatcherCode::kMissingEffectiveAttribute:
      return FusionFailureCode::kMissingEffectiveAttribute;
    case MatcherCode::kAttributeValueMismatch:
      return FusionFailureCode::kAttributeValueMismatch;
    case MatcherCode::kLiteralMismatch:
      return FusionFailureCode::kLiteralMismatch;
    case MatcherCode::kExternalPrivateUse:
      return FusionFailureCode::kExternalPrivateUse;
    case MatcherCode::kNonConvex:
      return FusionFailureCode::kNonConvex;
  }
  ORT_THROW("Unknown matcher failure code.");
}

std::string_view MatcherFailureDetail(
    function_extractor_internal::MatcherFailureCode code) {
  using MatcherCode = function_extractor_internal::MatcherFailureCode;
  switch (code) {
    case MatcherCode::kOpMismatch:
      return "pattern and target operation identities differ";
    case MatcherCode::kOutputSlotMismatch:
      return "pattern and target operation slots differ";
    case MatcherCode::kRepeatedBindingMismatch:
      return "repeated pattern value bindings differ";
    case MatcherCode::kMissingEffectiveAttribute:
      return "required effective attribute is missing";
    case MatcherCode::kAttributeValueMismatch:
      return "effective attribute values differ";
    case MatcherCode::kLiteralMismatch:
      return "tensor literal differs";
    case MatcherCode::kExternalPrivateUse:
      return "private pattern value has an external use";
    case MatcherCode::kNonConvex:
      return "matched pattern operation nodes are non-convex";
  }
  ORT_THROW("Unknown matcher failure code.");
}

common::Status ValidateReplacementCall(
    const Graph& graph, const CompiledFusionRule& compiled,
    const FusionReplacementPlan& plan);

common::Status DiscoverSelectedPlans(
    const FusionRuleSetState& rule_set,
    Graph& graph,
    std::vector<CompiledFusionRule>& compiled_rules,
    std::vector<FusionReplacementPlan>& selected_plans,
    size_t& condition_evaluations,
    size_t& rule_attempts,
    size_t& work_units,
    size_t epoch,
    FailureSink* failure_sink) {
  ORT_RETURN_IF_ERROR(ValidateGraphForDiscovery(graph));
  compiled_rules.clear();
  selected_plans.clear();
  compiled_rules.reserve(rule_set.normalized_rules.size());

  std::vector<FusionReplacementPlan> discovered_plans;
  function_extractor_internal::TargetGraphSnapshot snapshot;
  ORT_RETURN_IF_ERROR(function_extractor_internal::BuildTargetGraphSnapshot(
      graph, MakeMatcherOptions(rule_set.options), snapshot));
  ORT_RETURN_IF(
      work_units > rule_set.options.max_work_units ||
          snapshot.aggregate_work_units >
              rule_set.options.max_work_units - work_units,
      "FusionRuleSet aggregate matcher work budget exceeded.");
  work_units += snapshot.aggregate_work_units;
  for (size_t registration_order = 0;
       registration_order < rule_set.normalized_rules.size();
       ++registration_order) {
    compiled_rules.emplace_back();
    auto& compiled = compiled_rules.back();
    ORT_RETURN_IF_ERROR(CompileFusionRule(
        *rule_set.normalized_rules[registration_order], graph, compiled));

    std::vector<function_extractor_internal::ReplacementPlan> base_plans;
    function_extractor_internal::CompleteBindingHook condition_hook =
        [&](const function_extractor_internal::MatchState& match,
            const function_extractor_internal::TargetGraphSnapshot&
                match_snapshot,
            bool& accepted, std::shared_ptr<void>& extension_data) {
          ORT_RETURN_IF(
              condition_evaluations >=
                  rule_set.options.max_condition_evaluations,
              "FusionRuleSet condition-evaluation budget exceeded.");
          ++condition_evaluations;
          DependencyRecorder recorder(
              *compiled.compiled_pattern.normalized_pattern, match,
              match_snapshot,
              compiled.rule->matcher_options.max_attribute_bytes,
              compiled.rule->matcher_options.max_literal_bytes);
          ConstraintEvaluationResult constraint_result;
          ORT_RETURN_IF_ERROR(EvaluateConstraintProgram(
              compiled.rule->constraints, recorder, constraint_result));
          ORT_RETURN_IF_ERROR(recorder.Status());
          if (!constraint_result.satisfied) {
            if (failure_sink != nullptr) {
              FusionFailureRecord failure;
              failure.rule_id = compiled.rule->options.id;
              failure.stage = FusionMatchStage::kCondition;
              failure.code = FusionFailureCode::kConstraintFalse;
              failure.constraint =
                  constraint_result.failed_constraint;
              failure.detail = constraint_result.detail;
              failure.anchor_node = match.anchor_node;
              failure.anchor_output_slot =
                  match.anchor_output_slot;
              failure.pattern_nodes_matched =
                  match.target_node_to_pattern.size();
              failure_sink->RecordFailure(
                  failure, epoch, match.anchor_rank,
                  match.tuple_ordinal);
            }
            accepted = false;
            return common::Status::OK();
          }
          if (compiled.rule->predicate) {
            FusionConditionResult predicate_result;
            ORT_RETURN_IF_ERROR(
                onnxruntime::FusionPredicateInvoker::InvokePredicate(
                    compiled.rule->predicate,
                    *compiled.compiled_pattern.normalized_pattern,
                    compiled.compiled_pattern, match, match_snapshot,
                    recorder, predicate_result));
            ORT_RETURN_IF_ERROR(recorder.Status());
            if (predicate_result.decision ==
                FusionConditionDecision::kNotSatisfied) {
              if (failure_sink != nullptr) {
                FusionFailureRecord failure;
                failure.rule_id = compiled.rule->options.id;
                failure.stage = FusionMatchStage::kCondition;
                failure.code = FusionFailureCode::kCallbackRejected;
                failure.anchor_node = match.anchor_node;
                failure.anchor_output_slot =
                    match.anchor_output_slot;
                failure.pattern_nodes_matched =
                    match.target_node_to_pattern.size();
                if (predicate_result.failure.has_value()) {
                  failure.pattern_node =
                      predicate_result.failure->node;
                  failure.pattern_value =
                      predicate_result.failure->value;
                  failure.detail =
                      predicate_result.failure->reason;
                }
                failure_sink->RecordFailure(
                    failure, epoch, match.anchor_rank,
                    match.tuple_ordinal);
              }
              accepted = false;
              return common::Status::OK();
            }
          }
          auto evidence = std::make_shared<FusionConditionEvidence>();
          evidence->dependencies = recorder.TakeSnapshot();
          extension_data = std::move(evidence);
          accepted = true;
          return common::Status::OK();
        };
    function_extractor_internal::MatchFailureHook matcher_failure_hook;
    if (failure_sink != nullptr) {
      matcher_failure_hook =
          [&](const function_extractor_internal::MatcherFailure& rejection,
              NodeIndex anchor_node, size_t anchor_output_slot,
              size_t anchor_rank,
              size_t rejection_tuple_ordinal) {
            FusionFailureRecord failure;
            failure.rule_id = compiled.rule->options.id;
            failure.stage = ToFusionStage(rejection.stage);
            failure.code = ToFusionCode(rejection.code);
            failure.anchor_node = anchor_node;
            failure.anchor_output_slot = anchor_output_slot;
            failure.pattern_node = rejection.pattern_node;
            failure.pattern_value = rejection.pattern_value;
            failure.target_node = rejection.target_node;
            failure.target_slot = rejection.target_slot;
            failure.target_value_name =
                rejection.target_value_name;
            failure.pattern_nodes_matched =
                rejection.pattern_nodes_matched;
            failure.detail =
                rejection.detail.empty()
                    ? std::string{MatcherFailureDetail(rejection.code)}
                    : rejection.detail;
            failure_sink->RecordFailure(
                failure, epoch, anchor_rank,
                rejection_tuple_ordinal);
          };
    }
    function_extractor_internal::MatcherExecutionOptions
        matcher_execution_options;
    matcher_execution_options.allow_omitted_optional_formal_inputs = true;
    matcher_execution_options.total_attempts = &rule_attempts;
    matcher_execution_options.max_attempts =
        rule_set.options.max_rule_attempts;
    matcher_execution_options.total_work_units = &work_units;
    matcher_execution_options.failure_hook =
        failure_sink == nullptr ? nullptr : &matcher_failure_hook;
    ORT_RETURN_IF_ERROR(function_extractor_internal::DiscoverReplacementPlans(
        compiled.compiled_pattern, snapshot,
        compiled.rule->matcher_options, base_plans, nullptr,
        &condition_hook, matcher_execution_options));
    if (base_plans.empty() && failure_sink != nullptr) {
      for (FusionPatternValueId value_id = 0;
           value_id < compiled.rule->normalized_pattern.values.size();
           ++value_id) {
        const auto& value =
            compiled.rule->normalized_pattern.values[value_id];
        if (!value.is_literal) continue;
        const auto* target_value =
            graph.GetNodeArg(value.name);
        if (target_value == nullptr) continue;
        const ONNX_NAMESPACE::TensorProto* target_tensor =
            graph.GetConstantInitializer(value.name, false);
        ONNX_NAMESPACE::TensorProto normalized_constant;
        if (target_tensor == nullptr) {
          const auto producer = snapshot.producers.find(target_value);
          if (producer == snapshot.producers.end()) continue;
          const auto* node =
              snapshot.graph_viewer->GetNode(producer->second.node_index);
          if (node == nullptr ||
              !function_extractor_internal::NormalizeConstantAttributes(
                   node->GetAttributes(), normalized_constant)
                   .IsOK()) {
            continue;
          }
          target_tensor = &normalized_constant;
        }
        bool equal = false;
        ORT_RETURN_IF_ERROR(
            function_extractor_internal::CompareTensorLiterals(
                value.literal.tensor, *target_tensor,
                compiled.rule->matcher_options.max_literal_bytes,
                equal, &graph.ModelPath()));
        if (equal) continue;
        FusionFailureRecord failure;
        failure.rule_id = compiled.rule->options.id;
        failure.stage = FusionMatchStage::kLiteral;
        failure.code = FusionFailureCode::kLiteralMismatch;
        failure.pattern_value = value_id;
        failure.target_value_name = value.name;
        failure_sink->RecordFailure(
            failure, epoch, 0, rule_attempts);
        break;
      }
    }
    for (auto& base_plan : base_plans) {
      const size_t plan_tuple_ordinal = base_plan.tuple_ordinal;
      FusionReplacementPlan plan;
      ORT_RETURN_IF_ERROR(MaterializeFusionReplacementPlan(
          compiled, std::move(base_plan), registration_order,
          plan_tuple_ordinal, plan));
      const auto validation_status =
          ValidateReplacementCall(graph, compiled, plan);
      if (!validation_status.IsOK()) {
        if (failure_sink != nullptr) {
          FusionFailureRecord failure;
          failure.rule_id = plan.rule_id;
          failure.stage = FusionMatchStage::kFinalValidation;
          failure.code = FusionFailureCode::kOpMismatch;
          failure.anchor_node = plan.base.anchor_node;
          failure.anchor_output_slot =
              plan.base.anchor_output_slot;
          failure.pattern_nodes_matched =
              plan.base.pattern_node_to_target.size();
          failure.detail = validation_status.ErrorMessage();
          failure_sink->RecordFailure(
              failure, epoch, plan.base.anchor_rank,
              plan.base.tuple_ordinal);
        }
        continue;
      }
      discovered_plans.push_back(std::move(plan));
    }
  }

  std::sort(
      discovered_plans.begin(), discovered_plans.end(),
      [](const FusionReplacementPlan& lhs,
         const FusionReplacementPlan& rhs) {
        if (lhs.base.primary_root_topological_position !=
            rhs.base.primary_root_topological_position) {
          return lhs.base.primary_root_topological_position >
                 rhs.base.primary_root_topological_position;
        }
        return std::tie(lhs.anchor_local_priority,
                        lhs.registration_order,
                        lhs.tuple_ordinal,
                        lhs.base.removable_node_indices) <
               std::tie(rhs.anchor_local_priority,
                        rhs.registration_order,
                        rhs.tuple_ordinal,
                        rhs.base.removable_node_indices);
      });

  for (auto& candidate : discovered_plans) {
    bool conflicts = false;
    for (const auto& selected : selected_plans) {
      if (PlansConflict(candidate, selected)) {
        conflicts = true;
        break;
      }
    }
    if (!conflicts) {
      selected_plans.push_back(std::move(candidate));
    }
  }
  return common::Status::OK();
}

common::Status ComputeReplacementInputArgCounts(
    const ONNX_NAMESPACE::OpSchema& schema,
    gsl::span<const NodeArg* const> inputs,
    std::vector<int>& input_arg_count) {
  input_arg_count.assign(schema.inputs().size(), 0);
  size_t input_index = 0;
  for (size_t formal_index = 0;
       formal_index < schema.inputs().size(); ++formal_index) {
    const auto& formal = schema.inputs()[formal_index];
    if (formal.GetOption() == ONNX_NAMESPACE::OpSchema::Variadic) {
      ORT_RETURN_IF(formal_index + 1 != schema.inputs().size(),
                    "Fusion replacement variadic input must be last.");
      const size_t variadic_count = inputs.size() - input_index;
      ORT_RETURN_IF(variadic_count <
                        static_cast<size_t>(formal.GetMinArity()),
                    "Fusion replacement variadic input does not meet ",
                    "its minimum arity.");
      ORT_RETURN_IF(
          variadic_count >
              static_cast<size_t>(std::numeric_limits<int>::max()),
          "Fusion replacement variadic input count is too large.");
      input_arg_count[formal_index] =
          static_cast<int>(variadic_count);
      for (; input_index < inputs.size(); ++input_index) {
        ORT_RETURN_IF(inputs[input_index] == nullptr ||
                          !inputs[input_index]->Exists(),
                      "Fusion replacement variadic input contains a ",
                      "missing value.");
      }
      continue;
    }

    if (input_index == inputs.size()) {
      ORT_RETURN_IF(formal.GetOption() ==
                        ONNX_NAMESPACE::OpSchema::Single,
                    "Fusion replacement is missing a required input.");
      continue;
    }

    input_arg_count[formal_index] = 1;
    const auto* input = inputs[input_index++];
    ORT_RETURN_IF(
        formal.GetOption() == ONNX_NAMESPACE::OpSchema::Single &&
            (input == nullptr || !input->Exists()),
        "Fusion replacement required input is missing.");
  }
  ORT_RETURN_IF(input_index != inputs.size(),
                "Fusion replacement has excess inputs.");
  return common::Status::OK();
}

common::Status ValidateReplacementCall(
    const Graph& graph, const CompiledFusionRule& compiled,
    const FusionReplacementPlan& plan) {
  std::vector<const NodeArg*> inputs(
      plan.call_inputs.begin(), plan.call_inputs.end());
  std::vector<const NodeArg*> outputs(
      plan.call_outputs.begin(), plan.call_outputs.end());
  std::vector<int> input_arg_count;
  ORT_RETURN_IF_ERROR(ComputeReplacementInputArgCounts(
      *compiled.replacement_schema, inputs, input_arg_count));
  std::vector<std::optional<ONNX_NAMESPACE::TypeProto>>
      inferred_output_types;
  return graph.ValidateAndInferNodeTypeAndShape(
      plan.replacement_op_type, *compiled.replacement_schema,
      inputs, input_arg_count, outputs, plan.call_attributes,
      inferred_output_types);
}

common::Status PrevalidateSelectedPlans(
    const FusionRuleSetState&,
    const Graph& graph,
    gsl::span<const CompiledFusionRule> compiled_rules,
    gsl::span<const FusionReplacementPlan> plans) {
  for (const auto& plan : plans) {
    ORT_RETURN_IF(plan.registration_order >= compiled_rules.size(),
                  "Fusion replacement plan has an invalid rule index.");
    ORT_RETURN_IF_ERROR(function_extractor_internal::PrevalidatePlans(
        graph,
        compiled_rules[plan.registration_order].compiled_pattern,
        gsl::span<const function_extractor_internal::ReplacementPlan>{
            &plan.base, 1},
        false));
    ORT_RETURN_IF_ERROR(PrevalidateDependencies(
        graph, plan.dependencies,
        compiled_rules[plan.registration_order]
            .rule->matcher_options.max_attribute_bytes,
        compiled_rules[plan.registration_order]
            .rule->matcher_options.max_literal_bytes));
    const auto& compiled =
        compiled_rules[plan.registration_order];
    ORT_RETURN_IF_ERROR(
        ValidateReplacementCall(graph, compiled, plan));
  }
  return common::Status::OK();
}

FusionRewriteResult ApplyRuleSet(
    const FusionRuleSetState& rule_set,
    Graph& graph,
    const FusionExecutionControls& controls,
    FusionTraceCollector* trace) {
  FusionRewriteResult result;
  result.status = rule_set.construction_status;
  if (!result.status.IsOK()) return result;
  std::unique_ptr<FailureSink> failure_sink;
  result.status = CreateFailureSink(
      trace, rule_set.options.diagnostic_mode,
      rule_set.options.max_diagnostic_records,
      rule_set.options.max_diagnostic_bytes, failure_sink);
  if (!result.status.IsOK()) return result;
  if (failure_sink != nullptr) {
    for (size_t index = 0;
         index < rule_set.normalized_rules.size(); ++index) {
      const auto& rule = *rule_set.normalized_rules[index];
      failure_sink->RegisterRule(
          rule.options.id, rule.options.anchor_local_priority, index);
    }
  }

  const size_t initial_node_count =
      static_cast<size_t>(graph.NumberOfNodes());
  const size_t epoch_cap = controls.maximum_epochs.value_or(
      std::min(initial_node_count, rule_set.options.max_epochs));
  std::unordered_set<std::string> literal_initializers_to_preserve;
  size_t condition_evaluations = 0;
  size_t rule_attempts = 0;
  size_t work_units = 0;
  for (size_t epoch = 0; epoch <= epoch_cap; ++epoch) {
    std::vector<CompiledFusionRule> compiled_rules;
    std::vector<FusionReplacementPlan> selected_plans;
    result.status = DiscoverSelectedPlans(
        rule_set, graph, compiled_rules, selected_plans,
        condition_evaluations, rule_attempts, work_units, epoch,
        failure_sink.get());
    if (!result.status.IsOK()) return result;
    if (selected_plans.empty()) {
      result.status = common::Status::OK();
      return result;
    }
    result.status = PrevalidateSelectedPlans(
        rule_set, graph, compiled_rules, selected_plans);
    if (!result.status.IsOK()) return result;
    if (failure_sink != nullptr) {
      for (const auto& plan : selected_plans) {
        failure_sink->RecordSuccess(plan.rule_id);
      }
    }
    if (rule_set.options.diagnostic_mode ==
        FusionDiagnosticMode::kDryRun) {
      result.status = common::Status::OK();
      return result;
    }
    if (epoch >= epoch_cap) {
      result.status = ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "FusionRuleSet reached its defensive epoch cap despite strict ",
          "node-count decrease.");
      return result;
    }
    if (selected_plans.size() >
        rule_set.options.max_replacements -
            result.replacements_applied) {
      result.status = ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "FusionRuleSet replacement budget exceeded.");
      return result;
    }

    const size_t nodes_before =
        static_cast<size_t>(graph.NumberOfNodes());
    for (auto& plan : selected_plans) {
      if (result.replacements_applied >=
          rule_set.options.max_replacements) {
        result.status = ORT_MAKE_STATUS(
            ONNXRUNTIME, FAIL,
            "FusionRuleSet replacement budget exceeded.");
        return result;
      }
      for (const auto& witness : plan.base.literal_witnesses) {
        if (witness.is_initializer) {
          literal_initializers_to_preserve.insert(
              witness.target_value->Name());
        }
      }
      auto apply_plan = plan.base;
      apply_plan.call_inputs = plan.call_inputs;
      apply_plan.call_outputs = plan.call_outputs;
      apply_plan.call_attributes = plan.call_attributes;
      apply_plan.generated_call_name =
          graph.GenerateNodeName(plan.replacement_op_type);
      bool call_added = false;
      result.status =
          function_extractor_internal::ApplyReplacementPlan(
              graph, apply_plan, plan.replacement_op_type,
              plan.replacement_domain, plan.replacement_overload,
              "Call created by FusionRuleSet", call_added);
      if (call_added) ++result.replacements_applied;
      if (!result.status.IsOK()) return result;
    }

    graph.SetGraphResolveNeeded().SetGraphProtoSyncNeeded();
    Graph::ResolveOptions resolve_options;
    resolve_options.initializer_names_to_preserve =
        &literal_initializers_to_preserve;
    result.status = controls.resolve_graph != nullptr
                        ? controls.resolve_graph(graph, resolve_options)
                        : graph.Resolve(resolve_options);
    if (!result.status.IsOK()) return result;
    const size_t nodes_after =
        static_cast<size_t>(graph.NumberOfNodes());
    if (nodes_after >= nodes_before) {
      result.status = ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "FusionRuleSet epoch did not strictly reduce node count.");
      return result;
    }
    ++result.epochs_completed;
    if (controls.epoch_observer != nullptr) {
      controls.epoch_observer(
          controls.epoch_observer_state, epoch,
          nodes_before, nodes_after);
    }
  }

  result.status = ORT_MAKE_STATUS(
      ONNXRUNTIME, FAIL, "FusionRuleSet epoch invariant failure.");
  return result;
}

}  // namespace

struct FusionTestPlan::Impl {
  FusionReplacementPlan plan;
};

FusionRewriteResult FusionRuleSetExecution::Apply(
    const FusionRuleSet& rule_set, Graph& graph,
    const FusionExecutionControls& controls,
    FusionTraceCollector* trace) {
  return ApplyRuleSet(*rule_set.impl_, graph, controls, trace);
}

FusionTestPlan::FusionTestPlan() = default;
FusionTestPlan::~FusionTestPlan() = default;
FusionTestPlan::FusionTestPlan(FusionTestPlan&&) noexcept = default;
FusionTestPlan& FusionTestPlan::operator=(FusionTestPlan&&) noexcept =
    default;

common::Status FusionRuleSetTestAccess::DiscoverPlans(
    const FusionRuleSet& rule_set, Graph& graph,
    std::vector<FusionTestPlan>& plans) {
  std::vector<CompiledFusionRule> compiled_rules;
  std::vector<FusionReplacementPlan> selected_plans;
  size_t condition_evaluations = 0;
  size_t rule_attempts = 0;
  size_t work_units = 0;
  ORT_RETURN_IF_ERROR(DiscoverSelectedPlans(
      *rule_set.impl_, graph, compiled_rules, selected_plans,
      condition_evaluations, rule_attempts, work_units, 0, nullptr));
  plans.clear();
  plans.reserve(selected_plans.size());
  for (auto& selected : selected_plans) {
    FusionTestPlan plan;
    plan.impl_ = std::make_unique<FusionTestPlan::Impl>();
    plan.impl_->plan = std::move(selected);
    plans.push_back(std::move(plan));
  }
  return common::Status::OK();
}

common::Status FusionRuleSetTestAccess::PrevalidatePlans(
    const FusionRuleSet& rule_set, Graph& graph,
    gsl::span<const FusionTestPlan> plans) {
  std::vector<CompiledFusionRule> compiled_rules;
  compiled_rules.reserve(rule_set.impl_->normalized_rules.size());
  for (const auto& rule : rule_set.impl_->normalized_rules) {
    compiled_rules.emplace_back();
    ORT_RETURN_IF_ERROR(
        CompileFusionRule(*rule, graph, compiled_rules.back()));
  }
  for (const auto& plan : plans) {
    ORT_RETURN_IF(plan.impl_ == nullptr,
                  "FusionTestPlan is empty.");
    ORT_RETURN_IF_ERROR(PrevalidateSelectedPlans(
        *rule_set.impl_, graph, compiled_rules,
        gsl::span<const FusionReplacementPlan>{
            &plan.impl_->plan, 1}));
  }
  return common::Status::OK();
}

gsl::span<const ObservedDependencySummary>
FusionRuleSetTestAccess::ObservedDependencies(
    const FusionTestPlan& plan) {
  return plan.impl_ == nullptr
             ? gsl::span<const ObservedDependencySummary>{}
             : gsl::span<const ObservedDependencySummary>{
                   plan.impl_->plan.observed_dependencies};
}

FusionRewriteResult FusionRuleSetTestAccess::Apply(
    const FusionRuleSet& rule_set, Graph& graph,
    const FusionExecutionControls& controls,
    FusionTraceCollector* trace) {
  return FusionRuleSetExecution::Apply(
      rule_set, graph, controls, trace);
}

}  // namespace fusion_rewriter_internal

FusionRuleSet::FusionRuleSet(
    std::vector<FusionRule> rules, FusionRuleSetOptions options)
    : impl_(std::make_unique<Impl>(
          std::move(rules), std::move(options))) {}
FusionRuleSet::~FusionRuleSet() = default;
FusionRewriteResult FusionRuleSet::Apply(Model& model, FusionTraceCollector* trace) const {
  return Apply(model.MainGraph(), trace);
}
FusionRewriteResult FusionRuleSet::Apply(
    Graph& graph, FusionTraceCollector* trace) const {
  return fusion_rewriter_internal::FusionRuleSetExecution::Apply(
      *this, graph, {}, trace);
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
