#include "core/optimizer/fusion_rewriter.h"

#if !defined(ORT_MINIMAL_BUILD)

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
          std::make_unique<fusion_rewriter_internal::FusionRuleInternal>(
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

common::Status DiscoverSelectedPlans(
    const FusionRuleSetState& rule_set,
    Graph& graph,
    std::vector<CompiledFusionRule>& compiled_rules,
    std::vector<FusionReplacementPlan>& selected_plans,
    size_t& condition_evaluations,
    size_t epoch,
    FailureSink* failure_sink) {
  ORT_RETURN_IF_ERROR(ValidateGraphForDiscovery(graph));
  compiled_rules.clear();
  selected_plans.clear();
  compiled_rules.reserve(rule_set.normalized_rules.size());

  std::vector<FusionReplacementPlan> discovered_plans;
  size_t attempts = 0;
  size_t tuple_ordinal = 0;
  for (size_t registration_order = 0;
       registration_order < rule_set.normalized_rules.size();
       ++registration_order) {
    compiled_rules.emplace_back();
    auto& compiled = compiled_rules.back();
    ORT_RETURN_IF_ERROR(CompileFusionRule(
        *rule_set.normalized_rules[registration_order], graph, compiled));

    function_extractor_internal::TargetGraphSnapshot snapshot;
    ORT_RETURN_IF_ERROR(function_extractor_internal::BuildTargetGraphSnapshot(
        graph, compiled.compiled_pattern,
        compiled.rule->matcher_options, snapshot));
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
              failure.pattern_nodes_matched =
                  match.pattern_node_to_target.size();
              failure_sink->RecordFailure(
                  failure, epoch, 0, tuple_ordinal);
            }
            accepted = false;
            return common::Status::OK();
          }
          if (compiled.rule->predicate) {
            FusionConditionResult predicate_result;
            ORT_RETURN_IF_ERROR(
                onnxruntime::FusionRuleInternal::InvokePredicate(
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
                failure.pattern_nodes_matched =
                    match.pattern_node_to_target.size();
                if (predicate_result.failure.has_value()) {
                  failure.pattern_node =
                      predicate_result.failure->node;
                  failure.pattern_value =
                      predicate_result.failure->value;
                  failure.detail =
                      predicate_result.failure->reason;
                }
                failure_sink->RecordFailure(
                    failure, epoch, 0, tuple_ordinal);
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
    ORT_RETURN_IF_ERROR(function_extractor_internal::DiscoverReplacementPlans(
        compiled.compiled_pattern, snapshot,
        compiled.rule->matcher_options, base_plans, nullptr,
        &condition_hook));
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
            failure, epoch, 0, tuple_ordinal);
        break;
      }
    }
    for (auto& base_plan : base_plans) {
      ORT_RETURN_IF(attempts >= rule_set.options.max_rule_attempts,
                    "FusionRuleSet rule-attempt budget exceeded.");
      ++attempts;
      FusionReplacementPlan plan;
      ORT_RETURN_IF_ERROR(MaterializeFusionReplacementPlan(
          compiled, std::move(base_plan), registration_order,
          tuple_ordinal++, plan));
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
    std::vector<const NodeArg*> inputs(
        plan.call_inputs.begin(), plan.call_inputs.end());
    std::vector<const NodeArg*> outputs(
        plan.call_outputs.begin(), plan.call_outputs.end());
    std::vector<int> input_arg_count(
        compiled.replacement_schema->inputs().size(), 0);
    for (size_t index = 0;
         index < inputs.size() &&
         index < input_arg_count.size();
         ++index) {
      input_arg_count[index] = 1;
    }
    std::vector<std::optional<ONNX_NAMESPACE::TypeProto>>
        inferred_output_types;
    ORT_RETURN_IF_ERROR(graph.ValidateAndInferNodeTypeAndShape(
        plan.replacement_op_type, *compiled.replacement_schema,
        inputs, input_arg_count, outputs, plan.call_attributes,
        inferred_output_types));
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
  for (size_t epoch = 0; epoch <= epoch_cap; ++epoch) {
    std::vector<CompiledFusionRule> compiled_rules;
    std::vector<FusionReplacementPlan> selected_plans;
    result.status = DiscoverSelectedPlans(
        rule_set, graph, compiled_rules, selected_plans,
        condition_evaluations, epoch, failure_sink.get());
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
    std::vector<std::pair<std::string, InlinedHashSet<std::string>>>
        emitted_attribute_names;
    emitted_attribute_names.reserve(selected_plans.size());
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
      InlinedHashSet<std::string> attribute_names;
      for (const auto& [name, unused] : plan.call_attributes) {
        ORT_UNUSED_PARAMETER(unused);
        attribute_names.insert(name);
      }
      emitted_attribute_names.emplace_back(
          apply_plan.generated_call_name, std::move(attribute_names));
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
    for (const auto& [node_name, attribute_names] :
         emitted_attribute_names) {
      for (auto& node : graph.Nodes()) {
        if (node.Name() != node_name) continue;
        auto& attributes = node.GetMutableAttributes();
        std::erase_if(attributes, [&](const auto& entry) {
          return attribute_names.find(entry.first) ==
                 attribute_names.end();
        });
        break;
      }
    }
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
  ORT_RETURN_IF_ERROR(DiscoverSelectedPlans(
      *rule_set.impl_, graph, compiled_rules, selected_plans,
      condition_evaluations, 0, nullptr));
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
  return ApplyRuleSet(*rule_set.impl_, graph, controls, trace);
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
  return fusion_rewriter_internal::FusionRuleSetTestAccess::Apply(
      *this, graph, {}, trace);
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
