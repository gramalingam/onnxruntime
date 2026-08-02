#include "core/optimizer/fusion_rewriter_pattern.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <utility>

#include "core/graph/constants.h"
#include "core/graph/function_utils.h"
#include "core/graph/model.h"
#include "core/graph/schema_registry.h"

namespace onnxruntime::fusion_rewriter_internal {
namespace {

std::string CanonicalDomain(std::string_view domain) {
  return domain.empty() || domain == kOnnxDomainAlias
             ? std::string{kOnnxDomain}
             : std::string{domain};
}

common::Status InvalidRule(std::string_view message) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Invalid FusionRule: ", message);
}

}  // namespace

FunctionExtractorOptions MakeMatcherOptions(
    const FusionRuleSetOptions& options) {
  FunctionExtractorOptions matcher_options;
  matcher_options.max_pattern_nodes = options.max_pattern_nodes;
  matcher_options.max_target_nodes = options.max_target_nodes;
  matcher_options.max_output_root_tuples = options.max_output_root_tuples;
  matcher_options.max_worklist_bindings = options.max_work_units;
  matcher_options.max_literal_bytes = options.max_literal_bytes;
  matcher_options.max_formal_attributes = options.max_formal_attributes;
  matcher_options.max_attribute_bytes = options.max_attribute_bytes;
  return matcher_options;
}

FusionRuleInternal::FusionRuleInternal(
    const PatternFunctionProto& pattern_proto,
    FusionReplacementCall replacement_call,
    const ConstraintProgramDefinition& constraint_program,
    FusionMatchPredicate match_predicate,
    FusionRuleOptions rule_options,
    const FusionRuleSetOptions& rule_set_options)
    : pattern(pattern_proto),
      replacement(std::move(replacement_call)),
      predicate(std::move(match_predicate)),
      options(std::move(rule_options)),
      matcher_options(MakeMatcherOptions(rule_set_options)),
      normalized_pattern(
          function_extractor_internal::NormalizeFunctionPattern(
              pattern, matcher_options)) {
  if (!normalized_pattern.construction_status.IsOK()) {
    construction_status = normalized_pattern.construction_status;
    return;
  }
  ConstraintCompileOptions constraint_options;
  constraint_options.max_constraint_nodes =
      rule_set_options.max_constraint_nodes;
  constraint_options.max_dimension_equivalence_operands =
      rule_set_options.max_dimension_equivalence_operands;
  constraint_options.max_attribute_bytes =
      rule_set_options.max_attribute_bytes;
  constraints = CompileConstraintProgram(
      constraint_program, normalized_pattern, constraint_options);
  if (!constraints.construction_status.IsOK()) {
    construction_status = constraints.construction_status;
    return;
  }
  construction_status = ValidateReplacementMappings(*this);
}

common::Status ValidateReplacementMappings(
    const FusionRuleInternal& rule) {
  const auto& replacement = rule.replacement;
  const auto& normalized = rule.normalized_pattern;
  ORT_RETURN_IF(replacement.op_type.empty(),
                "Fusion replacement op_type must not be empty.");
  ORT_RETURN_IF(replacement.since_version < 0,
                "Fusion replacement since_version must be non-negative.");
  ORT_RETURN_IF(normalized.nodes.size() <= 1,
                "Fusion pattern must remove more than one operation node.");

  for (const auto& input : replacement.inputs) {
    ORT_RETURN_IF(input.formal_input_index.has_value() &&
                      *input.formal_input_index >=
                          normalized.formal_input_value_ids.size(),
                  "Fusion replacement input mapping is out of range.");
  }

  InlinedHashSet<size_t> mapped_outputs;
  for (const auto& output : replacement.outputs) {
    ORT_RETURN_IF(output.formal_output_index >=
                      normalized.formal_output_value_ids.size(),
                  "Fusion replacement output mapping is out of range.");
    ORT_RETURN_IF(!mapped_outputs.insert(output.formal_output_index).second,
                  "Fusion replacement output mapping is duplicated.");
  }
  ORT_RETURN_IF(mapped_outputs.size() !=
                    normalized.formal_output_value_ids.size(),
                "Fusion replacement must map every formal output exactly once.");

  InlinedHashSet<std::string> emitted_attribute_names;
  for (const auto& attribute : replacement.attributes) {
    ORT_RETURN_IF(attribute.emitted_name.empty() ||
                      !emitted_attribute_names.insert(
                                                  attribute.emitted_name)
                           .second,
                  "Fusion replacement attribute names must be non-empty and unique.");
    if (attribute.source ==
        FusionReplacementAttributeSource::kFormalAttribute) {
      ORT_RETURN_IF(attribute.formal_attribute_id >=
                        normalized.formal_attributes.size(),
                    "Fusion replacement formal attribute mapping is out of range.");
      continue;
    }
    ORT_RETURN_IF(attribute.source !=
                      FusionReplacementAttributeSource::kLiteral,
                  "Fusion replacement attribute source is invalid.");
    ONNX_NAMESPACE::AttributeProto canonical;
    ORT_RETURN_IF_ERROR(
        function_extractor_internal::CanonicalizeFormalAttribute(
            attribute.emitted_name, attribute.literal.type(),
            attribute.literal, rule.matcher_options.max_attribute_bytes,
            canonical));
  }
  return common::Status::OK();
}

common::Status CompileFusionRule(
    const FusionRuleInternal& rule,
    const Graph& graph,
    CompiledFusionRule& compiled_rule) {
  ORT_RETURN_IF_ERROR(rule.construction_status);
  compiled_rule = CompiledFusionRule{};
  compiled_rule.rule = &rule;
  ORT_RETURN_IF_ERROR(function_extractor_internal::CompileFunctionPattern(
      rule.normalized_pattern, graph, compiled_rule.compiled_pattern));

  compiled_rule.canonical_replacement_domain =
      CanonicalDomain(rule.replacement.domain);
  const auto import = graph.DomainToVersionMap().find(
      compiled_rule.canonical_replacement_domain);
  ORT_RETURN_IF(import == graph.DomainToVersionMap().end(),
                "No target-model opset import for fusion replacement domain '",
                compiled_rule.canonical_replacement_domain, "'.");
  ORT_RETURN_IF(
      rule.replacement.since_version > import->second,
      "Fusion replacement since_version exceeds the target-model import.");
  compiled_rule.replacement_schema =
      graph.GetSchemaRegistry()->GetSchema(
          rule.replacement.op_type, rule.replacement.since_version,
          compiled_rule.canonical_replacement_domain);
  if (compiled_rule.replacement_schema == nullptr) {
    const auto function_id = function_utils::GetFunctionIdentifier(
        compiled_rule.canonical_replacement_domain,
        rule.replacement.op_type, rule.replacement.overload);
    const auto& local_functions =
        graph.GetModel().GetModelLocalFunctionTemplates();
    const auto local_function = local_functions.find(function_id);
    if (local_function != local_functions.end()) {
      compiled_rule.replacement_schema =
          local_function->second->op_schema_.get();
    }
  }
  ORT_RETURN_IF(compiled_rule.replacement_schema == nullptr,
                "No registered schema/function for fusion replacement ",
                compiled_rule.canonical_replacement_domain, ":",
                rule.replacement.op_type, ".");
  ORT_RETURN_IF(compiled_rule.replacement_schema->since_version() !=
                    rule.replacement.since_version,
                "Fusion replacement exact since_version does not match the ",
                "schema selected by the target-model import.");
  return common::Status::OK();
}

common::Status MaterializeFusionReplacementPlan(
    const CompiledFusionRule& compiled_rule,
    function_extractor_internal::ReplacementPlan base_plan,
    size_t registration_order,
    size_t tuple_ordinal,
    FusionReplacementPlan& fusion_plan) {
  const auto& rule = *compiled_rule.rule;
  fusion_plan = FusionReplacementPlan{};
  fusion_plan.base = std::move(base_plan);
  fusion_plan.rule_id = rule.options.id;
  fusion_plan.anchor_local_priority = rule.options.anchor_local_priority;
  fusion_plan.registration_order = registration_order;
  fusion_plan.tuple_ordinal = tuple_ordinal;
  fusion_plan.replacement_domain = compiled_rule.canonical_replacement_domain;
  fusion_plan.replacement_op_type = rule.replacement.op_type;
  fusion_plan.replacement_overload = rule.replacement.overload;
  fusion_plan.replacement_since_version = rule.replacement.since_version;
  if (fusion_plan.base.extension_data != nullptr) {
    auto evidence = std::static_pointer_cast<FusionConditionEvidence>(
        fusion_plan.base.extension_data);
    fusion_plan.dependencies = std::move(evidence->dependencies);
    fusion_plan.observed_dependencies.reserve(
        fusion_plan.dependencies.size());
    for (const auto& dependency : fusion_plan.dependencies) {
      fusion_plan.observed_dependencies.push_back(
          ObservedDependencySummary{
              static_cast<ObservedDependencyKind>(dependency.kind),
              dependency.pattern_node,
              dependency.pattern_value,
              dependency.formal_attribute,
              dependency.name,
              dependency.axis});
    }
  }

  fusion_plan.call_inputs.reserve(rule.replacement.inputs.size());
  for (const auto& input : rule.replacement.inputs) {
    fusion_plan.call_inputs.push_back(
        input.formal_input_index.has_value()
            ? fusion_plan.base.call_inputs[*input.formal_input_index]
            : nullptr);
  }

  fusion_plan.call_outputs.reserve(rule.replacement.outputs.size());
  for (const auto& output : rule.replacement.outputs) {
    fusion_plan.call_outputs.push_back(
        fusion_plan.base.call_outputs[output.formal_output_index]);
  }

  for (const auto& attribute : rule.replacement.attributes) {
    ONNX_NAMESPACE::AttributeProto emitted;
    if (attribute.source ==
        FusionReplacementAttributeSource::kFormalAttribute) {
      const auto& formal =
          rule.normalized_pattern
              .formal_attributes[attribute.formal_attribute_id];
      const auto bound =
          fusion_plan.base.call_attributes.find(formal.formal_name);
      ORT_RETURN_IF(bound == fusion_plan.base.call_attributes.end(),
                    "Missing bound formal fusion attribute '",
                    formal.formal_name,
                    "'.");
      emitted = bound->second;
      emitted.set_name(attribute.emitted_name);
    } else {
      ORT_RETURN_IF_ERROR(
          function_extractor_internal::CanonicalizeFormalAttribute(
              attribute.emitted_name, attribute.literal.type(),
              attribute.literal, rule.matcher_options.max_attribute_bytes,
              emitted));
    }
    const auto [_, inserted] = fusion_plan.call_attributes.emplace(
        attribute.emitted_name, std::move(emitted));
    ORT_RETURN_IF_NOT(inserted,
                      "Duplicate emitted fusion attribute '",
                      attribute.emitted_name, "'.");
  }
  return common::Status::OK();
}

}  // namespace onnxruntime::fusion_rewriter_internal

#endif  // !defined(ORT_MINIMAL_BUILD)
