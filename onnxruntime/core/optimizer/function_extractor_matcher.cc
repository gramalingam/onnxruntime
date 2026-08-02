#include "core/optimizer/function_extractor_matcher.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <deque>
#include <functional>
#include <tuple>

#include "core/common/safeint.h"

namespace onnxruntime {
namespace function_extractor_internal {
namespace {

using common::Status;

Status MatcherError(const std::string& message) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "FunctionExtractor matcher: ", message);
}

std::string RootSignatureKey(std::string_view domain,
                             std::string_view op_type,
                             std::string_view overload,
                             int since_version,
                             size_t output_arity) {
  return MakeString(domain.size(), ':', domain,
                    op_type.size(), ':', op_type,
                    overload.size(), ':', overload,
                    since_version, ':', output_arity);
}

std::string RootSignatureKey(const ResolvedPatternNode& node) {
  return RootSignatureKey(node.canonical_domain, node.op_type, node.overload,
                          node.since_version, node.output_arity);
}

std::string RootSignatureKey(const Node& node) {
  return RootSignatureKey(node.Domain(), node.OpType(), node.Overload(),
                          node.SinceVersion(), node.OutputDefs().size());
}

bool PatternTypeCompatible(const PatternValue& pattern_value, const NodeArg& target_value) {
  if (!pattern_value.has_type || target_value.TypeAsProto() == nullptr) return true;
  const auto& lhs = pattern_value.type;
  const auto& rhs = *target_value.TypeAsProto();
  if (lhs.value_case() != rhs.value_case()) return false;
  if (!lhs.has_tensor_type() || !rhs.has_tensor_type()) return false;
  const auto& lhs_tensor = lhs.tensor_type();
  const auto& rhs_tensor = rhs.tensor_type();
  if (lhs_tensor.elem_type() != 0 && rhs_tensor.elem_type() != 0 &&
      lhs_tensor.elem_type() != rhs_tensor.elem_type()) {
    return false;
  }
  if (!lhs_tensor.has_shape() || !rhs_tensor.has_shape()) return true;
  if (lhs_tensor.shape().dim_size() != rhs_tensor.shape().dim_size()) return false;
  for (int i = 0; i < lhs_tensor.shape().dim_size(); ++i) {
    const auto& lhs_dim = lhs_tensor.shape().dim(i);
    const auto& rhs_dim = rhs_tensor.shape().dim(i);
    if (lhs_dim.has_dim_value() && rhs_dim.has_dim_value() &&
        lhs_dim.dim_value() != rhs_dim.dim_value()) {
      return false;
    }
  }
  return true;
}

const ONNX_NAMESPACE::AttributeProto* EffectiveTargetAttribute(
    const Node& target_node, std::string_view name) {
  const auto explicit_attribute = target_node.GetAttributes().find(std::string{name});
  if (explicit_attribute != target_node.GetAttributes().end()) return &explicit_attribute->second;
  if (target_node.Op() == nullptr) return nullptr;
  const auto schema_attribute = target_node.Op()->attributes().find(std::string{name});
  if (schema_attribute == target_node.Op()->attributes().end() ||
      schema_attribute->second.default_value.name().empty()) {
    return nullptr;
  }
  return &schema_attribute->second.default_value;
}

bool IsVariableAttributeName(const ResolvedPatternNode& pattern_node, std::string_view name) {
  return std::any_of(
      pattern_node.attribute_variables.begin(), pattern_node.attribute_variables.end(),
      [&](const AttributeVariableOccurrence& occurrence) {
        return occurrence.operator_attribute_name == name;
      });
}

bool NodeSignatureMatches(const ResolvedPatternNode& pattern_node, const Node& target_node) {
  if (target_node.Op() == nullptr ||
      pattern_node.canonical_domain != target_node.Domain()) {
    return false;
  }
  if (pattern_node.op_type != target_node.OpType() ||
      pattern_node.overload != target_node.Overload() ||
      pattern_node.since_version != target_node.SinceVersion() ||
      pattern_node.input_arity < target_node.InputDefs().size() ||
      pattern_node.output_arity != target_node.OutputDefs().size()) {
    return false;
  }
  for (size_t input_index = target_node.InputDefs().size();
       input_index < pattern_node.input_arity; ++input_index) {
    if (input_index >= pattern_node.schema->inputs().size() ||
        pattern_node.schema->inputs()[input_index].GetOption() !=
            ONNX_NAMESPACE::OpSchema::Optional) {
      return false;
    }
  }
  for (const auto& [name, pattern_attribute] : pattern_node.effective_attributes) {
    const auto* target_attribute = EffectiveTargetAttribute(target_node, name);
    if (target_attribute == nullptr) return false;
    NodeAttributes lhs{{name, pattern_attribute}};
    NodeAttributes rhs{{name, *target_attribute}};
    if (!AreAttributesSemanticallyEqual(lhs, rhs)) return false;
  }
  for (const auto& [name, unused] : target_node.GetAttributes()) {
    ORT_UNUSED_PARAMETER(unused);
    if (pattern_node.effective_attributes.find(name) == pattern_node.effective_attributes.end() &&
        !IsVariableAttributeName(pattern_node, name)) {
      return false;
    }
  }
  return true;
}

Status MatchLiteral(const PatternValue& pattern_value,
                    const NodeArg& target_value,
                    const TargetGraphSnapshot& snapshot,
                    const FunctionExtractorOptions& options,
                    size_t& literal_bytes_compared,
                    LiteralWitness& witness,
                    bool& matched) {
  matched = false;
  witness.pattern_value_id = kMissingPatternValue;
  witness.target_value = &target_value;

  const auto initializer = snapshot.constant_initializers.find(target_value.Name());
  const ONNX_NAMESPACE::TensorProto* target_tensor = nullptr;
  ONNX_NAMESPACE::TensorProto normalized_constant;
  if (initializer != snapshot.constant_initializers.end()) {
    target_tensor = initializer->second;
    witness.is_initializer = true;
  } else {
    const auto producer = snapshot.producers.find(&target_value);
    if (producer == snapshot.producers.end()) return Status::OK();
    const auto* node = snapshot.graph_viewer->GetNode(producer->second.node_index);
    if (node == nullptr || node->Domain() != kOnnxDomain || node->OpType() != "Constant") {
      return Status::OK();
    }
    const auto status = NormalizeConstantAttributes(node->GetAttributes(), normalized_constant);
    if (!status.IsOK()) return Status::OK();
    target_tensor = &normalized_constant;
    witness.constant_node_index = node->Index();
  }

  literal_bytes_compared =
      SafeInt<size_t>(literal_bytes_compared) + pattern_value.literal.byte_count;
  ORT_RETURN_IF(literal_bytes_compared > options.max_literal_bytes,
                "FunctionExtractor literal comparison byte budget exceeded.");
  const auto& model_path = snapshot.graph->ModelPath();
  ORT_RETURN_IF_ERROR(CompareTensorLiterals(pattern_value.literal.tensor, *target_tensor,
                                            options.max_literal_bytes, matched, &model_path));
  return Status::OK();
}

struct CandidateMatcher {
  const CompiledFunctionPattern& compiled;
  const TargetGraphSnapshot& snapshot;
  const FunctionExtractorOptions& options;
  MatcherDiagnostics* diagnostics;
  size_t& literal_bytes_compared;
  size_t& attribute_payload_bytes_inspected;
  size_t& aggregate_work_units;
  size_t primary_output_group;
  const CompleteBindingHook* complete_binding_hook;
  bool allow_omitted_optional_formal_inputs;
  bool capture_failure;
  MatchState state;
  MatcherFailure failure;

  void Reject(MatcherFailureStage stage, MatcherFailureCode code,
              std::optional<PatternNodeId> pattern_node = std::nullopt,
              std::optional<PatternValueId> pattern_value = std::nullopt,
              std::optional<NodeIndex> target_node = std::nullopt,
              std::optional<size_t> target_slot = std::nullopt,
              std::string_view detail = {}) {
    if (!capture_failure) {
      return;
    }
    failure.valid = true;
    failure.stage = stage;
    failure.code = code;
    failure.pattern_node = pattern_node;
    failure.pattern_value = pattern_value;
    failure.target_node = target_node;
    failure.target_slot = target_slot;
    failure.pattern_nodes_matched =
        state.target_node_to_pattern.size();
    failure.detail = detail;
  }

  Status ConsumeWork() {
    ORT_RETURN_IF(aggregate_work_units >= options.max_worklist_bindings,
                  "FunctionExtractor aggregate matcher work budget exceeded.");
    ++aggregate_work_units;
    return Status::OK();
  }

  Status ConsumeAttributeBytes(size_t bytes) {
    ORT_RETURN_IF(attribute_payload_bytes_inspected > options.max_attribute_bytes ||
                      bytes > options.max_attribute_bytes - attribute_payload_bytes_inspected,
                  "FunctionExtractor attribute byte budget exceeded.");
    attribute_payload_bytes_inspected += bytes;
    return Status::OK();
  }

  Status Schedule(PatternValueId pattern_value_id, const NodeArg* target_value, bool& matched) {
    if (target_value == nullptr || !target_value->Exists()) {
      Reject(MatcherFailureStage::kValueBinding,
             MatcherFailureCode::kRepeatedBindingMismatch,
             std::nullopt, pattern_value_id);
      matched = false;
      return Status::OK();
    }
    if (!PatternTypeCompatible(compiled.normalized_pattern->values[pattern_value_id], *target_value)) {
      Reject(MatcherFailureStage::kValueBinding,
             MatcherFailureCode::kRepeatedBindingMismatch,
             std::nullopt, pattern_value_id);
      matched = false;
      return Status::OK();
    }
    auto& visit_state = state.value_visit_states[pattern_value_id];
    if (visit_state != ValueVisitState::Unseen) {
      matched = state.pattern_value_to_target[pattern_value_id] == target_value;
      if (!matched) {
        Reject(MatcherFailureStage::kValueBinding,
               MatcherFailureCode::kRepeatedBindingMismatch,
               std::nullopt, pattern_value_id);
      }
      return Status::OK();
    }
    ORT_RETURN_IF_ERROR(ConsumeWork());
    state.pattern_value_to_target[pattern_value_id] = target_value;
    visit_state = ValueVisitState::Scheduled;
    ++state.scheduled_binding_count;
    if (diagnostics != nullptr) ++diagnostics->worklist_bindings_scheduled;
    return Status::OK();
  }

  Status NextScheduledValue(PatternValueId& scheduled_value_id) {
    const auto& normalized = *compiled.normalized_pattern;
    for (const auto node_id : normalized.reverse_topological_node_ids) {
      for (const auto output_id : normalized.nodes[node_id].output_value_ids) {
        ORT_RETURN_IF_ERROR(ConsumeWork());
        if (output_id == kMissingPatternValue) continue;
        if (state.value_visit_states[output_id] == ValueVisitState::Scheduled) {
          scheduled_value_id = output_id;
          return Status::OK();
        }
      }
      for (const auto input_id : normalized.nodes[node_id].input_value_ids) {
        ORT_RETURN_IF_ERROR(ConsumeWork());
        if (input_id != kMissingPatternValue &&
            state.value_visit_states[input_id] == ValueVisitState::Scheduled) {
          scheduled_value_id = input_id;
          return Status::OK();
        }
      }
    }
    for (PatternValueId value_id = 0; value_id < state.value_visit_states.size(); ++value_id) {
      ORT_RETURN_IF_ERROR(ConsumeWork());
      if (state.value_visit_states[value_id] == ValueVisitState::Scheduled) {
        scheduled_value_id = value_id;
        return Status::OK();
      }
    }
    scheduled_value_id = kMissingPatternValue;
    return Status::OK();
  }

  Status CheckOrBindAttribute(const AttributeVariableOccurrence& occurrence,
                              const Node& target_node,
                              bool& matched) {
    ORT_RETURN_IF_ERROR(ConsumeWork());
    const auto* target_attribute =
        EffectiveTargetAttribute(target_node, occurrence.operator_attribute_name);
    if (target_attribute == nullptr) {
      Reject(MatcherFailureStage::kAttributeBinding,
             MatcherFailureCode::kMissingEffectiveAttribute,
             std::nullopt, std::nullopt, target_node.Index(),
             std::nullopt, occurrence.operator_attribute_name);
      matched = false;
      return Status::OK();
    }
    const auto& formal =
        compiled.normalized_pattern->formal_attributes[occurrence.formal_attribute_id];
    ONNX_NAMESPACE::AttributeProto canonical;
    ORT_RETURN_IF(attribute_payload_bytes_inspected > options.max_attribute_bytes,
                  "FunctionExtractor attribute byte budget exceeded.");
    const size_t remaining_attribute_bytes =
        options.max_attribute_bytes - attribute_payload_bytes_inspected;
    ORT_RETURN_IF_ERROR(CanonicalizeFormalAttribute(
        formal.formal_name, formal.type, *target_attribute,
        remaining_attribute_bytes, canonical));
    ORT_RETURN_IF_ERROR(ConsumeAttributeBytes(AttributePayloadBytes(canonical)));
    auto& binding = state.formal_attribute_bindings[occurrence.formal_attribute_id];
    if (!binding.has_value()) {
      binding = canonical;
    } else {
      bool equal = false;
      ORT_RETURN_IF_ERROR(
          CompareFormalAttributes(*binding, canonical, options.max_attribute_bytes, equal));
      if (!equal) {
        Reject(MatcherFailureStage::kAttributeBinding,
               MatcherFailureCode::kAttributeValueMismatch,
               std::nullopt, std::nullopt, target_node.Index(),
               std::nullopt, occurrence.operator_attribute_name);
        matched = false;
        return Status::OK();
      }
    }
    state.matched_attribute_occurrences.push_back(
        MatchedAttributeOccurrence{target_node.Index(),
                                   occurrence.operator_attribute_name,
                                   occurrence.formal_attribute_id,
                                   std::move(canonical)});
    return Status::OK();
  }

  Status BindProducer(PatternValueId value_id, const NodeArg& target_value, bool& matched) {
    const auto& normalized = *compiled.normalized_pattern;
    const auto& pattern_value = normalized.values[value_id];
    const auto target_producer = snapshot.producers.find(&target_value);
    if (target_producer == snapshot.producers.end() ||
        target_producer->second.output_index != pattern_value.producer_output_index) {
      Reject(MatcherFailureStage::kStructuralEdge,
             MatcherFailureCode::kOutputSlotMismatch,
             pattern_value.producer_node_id, value_id);
      matched = false;
      return Status::OK();
    }

    const auto pattern_node_id = pattern_value.producer_node_id;
    const auto target_node_index = target_producer->second.node_index;
    const auto* target_node = snapshot.graph_viewer->GetNode(target_node_index);
    if (target_node == nullptr ||
        !NodeSignatureMatches(compiled.resolved_nodes[pattern_node_id], *target_node) ||
        !target_node->GetExecutionProviderType().empty() ||
        snapshot.control_edge_nodes.find(target_node_index) != snapshot.control_edge_nodes.end()) {
      Reject(MatcherFailureStage::kStructuralNode,
             MatcherFailureCode::kOpMismatch, pattern_node_id, value_id,
             target_node_index);
      matched = false;
      return Status::OK();
    }

    auto& mapped_target = state.pattern_node_to_target[pattern_node_id];
    bool newly_mapped = false;
    if (mapped_target != std::numeric_limits<NodeIndex>::max()) {
      matched = mapped_target == target_node_index;
      if (!matched) {
        Reject(MatcherFailureStage::kValueBinding,
               MatcherFailureCode::kRepeatedBindingMismatch,
               pattern_node_id, value_id, target_node_index);
      }
      if (!matched) return Status::OK();
    } else {
      if (state.target_node_to_pattern.find(target_node_index) != state.target_node_to_pattern.end()) {
        Reject(MatcherFailureStage::kValueBinding,
               MatcherFailureCode::kRepeatedBindingMismatch,
               pattern_node_id, value_id, target_node_index);
        matched = false;
        return Status::OK();
      }
      mapped_target = target_node_index;
      state.target_node_to_pattern.emplace(target_node_index, pattern_node_id);
      newly_mapped = true;
    }

    if (newly_mapped) {
      for (const auto& occurrence : compiled.resolved_nodes[pattern_node_id].attribute_variables) {
        ORT_RETURN_IF_ERROR(CheckOrBindAttribute(occurrence, *target_node, matched));
        if (!matched) return Status::OK();
      }
    }

    const auto& pattern_node = normalized.nodes[pattern_node_id];
    for (size_t output_index = 0; output_index < pattern_node.output_value_ids.size(); ++output_index) {
      ORT_RETURN_IF_ERROR(ConsumeWork());
      const auto pattern_output = pattern_node.output_value_ids[output_index];
      const auto* target_output = target_node->OutputDefs()[output_index];
      if (pattern_output == kMissingPatternValue) {
        if (target_output != nullptr && target_output->Exists()) {
          Reject(MatcherFailureStage::kStructuralEdge,
                 MatcherFailureCode::kOutputSlotMismatch,
                 pattern_node_id, std::nullopt, target_node_index,
                 output_index);
          matched = false;
          return Status::OK();
        }
        continue;
      }
      if (target_output == nullptr || !target_output->Exists()) {
        Reject(MatcherFailureStage::kStructuralEdge,
               MatcherFailureCode::kOutputSlotMismatch,
               pattern_node_id, pattern_output, target_node_index,
               output_index);
        matched = false;
        return Status::OK();
      }
      ORT_RETURN_IF_ERROR(Schedule(pattern_output, target_output, matched));
      if (!matched) return Status::OK();
    }
    for (size_t input_index = 0; input_index < pattern_node.input_value_ids.size(); ++input_index) {
      ORT_RETURN_IF_ERROR(ConsumeWork());
      const auto pattern_input = pattern_node.input_value_ids[input_index];
      const auto* target_input =
          input_index < target_node->InputDefs().size()
              ? target_node->InputDefs()[input_index]
              : nullptr;
      if (pattern_input == kMissingPatternValue) {
        if (target_input != nullptr && target_input->Exists()) {
          Reject(MatcherFailureStage::kStructuralEdge,
                 MatcherFailureCode::kOutputSlotMismatch,
                 pattern_node_id, std::nullopt, target_node_index,
                 input_index);
          matched = false;
          return Status::OK();
        }
        continue;
      }
      if (target_input == nullptr || !target_input->Exists()) {
        const auto& resolved_node = compiled.resolved_nodes[pattern_node_id];
        const bool is_optional_formal_input =
            normalized.values[pattern_input].is_formal_input &&
            input_index < resolved_node.schema->inputs().size() &&
            resolved_node.schema->inputs()[input_index].GetOption() ==
                ONNX_NAMESPACE::OpSchema::Optional;
        if (allow_omitted_optional_formal_inputs &&
            is_optional_formal_input) {
          auto& visit_state = state.value_visit_states[pattern_input];
          if (visit_state == ValueVisitState::Unseen) {
            visit_state = ValueVisitState::Processed;
          } else if (state.pattern_value_to_target[pattern_input] != nullptr) {
            Reject(MatcherFailureStage::kValueBinding,
                   MatcherFailureCode::kRepeatedBindingMismatch,
                   pattern_node_id, pattern_input, target_node_index,
                   input_index);
            matched = false;
            return Status::OK();
          }
          continue;
        }
        Reject(MatcherFailureStage::kStructuralEdge,
               MatcherFailureCode::kOutputSlotMismatch,
               pattern_node_id, pattern_input, target_node_index,
               input_index);
        matched = false;
        return Status::OK();
      }
      ORT_RETURN_IF_ERROR(Schedule(pattern_input, target_input, matched));
      if (!matched) return Status::OK();
    }
    return Status::OK();
  }

  Status Run(gsl::span<const NodeIndex> output_root_nodes, ReplacementPlan& plan, bool& matched) {
    const auto& normalized = *compiled.normalized_pattern;
    state.pattern_node_to_target.assign(normalized.nodes.size(), std::numeric_limits<NodeIndex>::max());
    state.pattern_value_to_target.assign(normalized.values.size(), nullptr);
    state.value_visit_states.assign(normalized.values.size(), ValueVisitState::Unseen);
    state.formal_input_bindings.assign(normalized.formal_input_value_ids.size(), nullptr);
    state.formal_attribute_bindings.resize(normalized.formal_attributes.size());
    matched = true;

    for (size_t group_index = 0; group_index < compiled.formal_output_producer_groups.size(); ++group_index) {
      const auto& group = compiled.formal_output_producer_groups[group_index];
      const auto* target_node = snapshot.graph_viewer->GetNode(output_root_nodes[group_index]);
      if (target_node == nullptr) {
        Reject(MatcherFailureStage::kStructuralNode,
               MatcherFailureCode::kOpMismatch,
               group.producer_node_id);
        matched = false;
        return Status::OK();
      }
      for (size_t i = 0; i < group.formal_output_indices.size(); ++i) {
        ORT_RETURN_IF_ERROR(ConsumeWork());
        const auto formal_output_index = group.formal_output_indices[i];
        const auto target_output_index = group.producer_output_indices[i];
        if (target_output_index >= target_node->OutputDefs().size()) {
          Reject(MatcherFailureStage::kStructuralEdge,
                 MatcherFailureCode::kOutputSlotMismatch,
                 group.producer_node_id, std::nullopt,
                 target_node->Index(), target_output_index);
          matched = false;
          return Status::OK();
        }
        bool binding_matches = true;
        ORT_RETURN_IF_ERROR(Schedule(normalized.formal_output_value_ids[formal_output_index],
                                     target_node->OutputDefs()[target_output_index], binding_matches));
        if (!binding_matches) {
          matched = false;
          return Status::OK();
        }
      }
    }

    while (true) {
      PatternValueId value_id;
      ORT_RETURN_IF_ERROR(NextScheduledValue(value_id));
      if (value_id == kMissingPatternValue) break;
      state.value_visit_states[value_id] = ValueVisitState::Processed;
      if (diagnostics != nullptr) ++diagnostics->worklist_bindings_processed;
      const auto& pattern_value = normalized.values[value_id];
      const auto* target_value = state.pattern_value_to_target[value_id];

      if (pattern_value.is_formal_input) {
        const auto formal_it =
            std::find(normalized.formal_input_value_ids.begin(), normalized.formal_input_value_ids.end(), value_id);
        state.formal_input_bindings[static_cast<size_t>(
            std::distance(normalized.formal_input_value_ids.begin(), formal_it))] = target_value;
        continue;
      }
      if (pattern_value.is_literal) {
        LiteralWitness witness;
        ORT_RETURN_IF_ERROR(MatchLiteral(pattern_value, *target_value, snapshot, options,
                                         literal_bytes_compared, witness, matched));
        if (!matched) {
          Reject(MatcherFailureStage::kLiteral,
                 MatcherFailureCode::kLiteralMismatch,
                 std::nullopt, value_id, std::nullopt,
                 std::nullopt, "tensor literal differs");
          if (capture_failure) {
            failure.target_value_name = target_value->Name();
          }
          return Status::OK();
        }
        witness.pattern_value_id = value_id;
        state.literal_witnesses.push_back(std::move(witness));
        continue;
      }
      ORT_RETURN_IF_ERROR(BindProducer(value_id, *target_value, matched));
      if (!matched) return Status::OK();
    }

    for (const auto mapped_node : state.pattern_node_to_target) {
      if (mapped_node == std::numeric_limits<NodeIndex>::max()) {
        Reject(MatcherFailureStage::kFinalValidation,
               MatcherFailureCode::kOpMismatch);
        matched = false;
        return Status::OK();
      }
    }
    for (const auto& binding : state.formal_attribute_bindings) {
      if (!binding.has_value()) {
        Reject(MatcherFailureStage::kFinalValidation,
               MatcherFailureCode::kMissingEffectiveAttribute);
        matched = false;
        return Status::OK();
      }
    }
    InlinedHashSet<const NodeArg*> formal_inputs(state.formal_input_bindings.begin(),
                                                 state.formal_input_bindings.end());
    for (const auto& witness : state.literal_witnesses) {
      if (formal_inputs.find(witness.target_value) != formal_inputs.end()) {
        Reject(MatcherFailureStage::kFinalValidation,
               MatcherFailureCode::kRepeatedBindingMismatch,
               std::nullopt, witness.pattern_value_id);
        matched = false;
        return Status::OK();
      }
    }

    std::shared_ptr<void> extension_data;
    if (complete_binding_hook != nullptr) {
      bool accepted = true;
      ORT_RETURN_IF_ERROR((*complete_binding_hook)(
          state, snapshot, accepted, extension_data));
      if (!accepted) {
        matched = false;
        return Status::OK();
      }
    }

    ORT_RETURN_IF_ERROR(ValidateCandidateAndBuildPlan(plan, matched));
    if (matched) plan.extension_data = std::move(extension_data);
    return Status::OK();
  }

  Status ValidateCandidateAndBuildPlan(ReplacementPlan& plan, bool& matched) {
    const auto& normalized = *compiled.normalized_pattern;
    plan = ReplacementPlan{};
    plan.removable_node_indices.assign(state.pattern_node_to_target.begin(), state.pattern_node_to_target.end());
    std::sort(plan.removable_node_indices.begin(), plan.removable_node_indices.end(),
              [&](NodeIndex lhs, NodeIndex rhs) {
                return snapshot.topological_positions.at(lhs) < snapshot.topological_positions.at(rhs);
              });
    if (plan.removable_node_indices.size() <= 1) {
      Reject(MatcherFailureStage::kFinalValidation,
             MatcherFailureCode::kOpMismatch);
      matched = false;
      return Status::OK();
    }
    const InlinedHashSet<NodeIndex> removable(plan.removable_node_indices.begin(),
                                              plan.removable_node_indices.end());

    // Boundary closure and scheduling constraints.
    for (const auto* formal_input : state.formal_input_bindings) {
      const auto producer = snapshot.producers.find(formal_input);
      if (producer != snapshot.producers.end() &&
          removable.find(producer->second.node_index) != removable.end()) {
        Reject(MatcherFailureStage::kClosure,
               MatcherFailureCode::kExternalPrivateUse);
        matched = false;
        return Status::OK();
      }
    }

    std::string annotation;
    bool first_node = true;
    for (const auto node_index : plan.removable_node_indices) {
      const auto* node = snapshot.graph_viewer->GetNode(node_index);
      if (first_node) {
        annotation = node->GetLayeringAnnotation();
        first_node = false;
      } else if (node->GetLayeringAnnotation() != annotation) {
        Reject(MatcherFailureStage::kFinalValidation,
               MatcherFailureCode::kOpMismatch);
        matched = false;
        return Status::OK();
      }
    }
    plan.layering_annotation = std::move(annotation);

    for (PatternValueId value_id = 0; value_id < normalized.values.size(); ++value_id) {
      ORT_RETURN_IF_ERROR(ConsumeWork());
      const auto& pattern_value = normalized.values[value_id];
      if (pattern_value.producer_node_id == kNoPatternNode || pattern_value.is_formal_output) continue;
      const auto* target_value = state.pattern_value_to_target[value_id];
      if (target_value == nullptr) continue;
      if (snapshot.graph_outputs.find(target_value) != snapshot.graph_outputs.end()) {
        Reject(MatcherFailureStage::kClosure,
               MatcherFailureCode::kExternalPrivateUse,
               pattern_value.producer_node_id, value_id);
        matched = false;
        return Status::OK();
      }
      const auto explicit_consumers = snapshot.explicit_consumers.find(target_value);
      if (explicit_consumers != snapshot.explicit_consumers.end()) {
        for (const auto& consumer : explicit_consumers->second) {
          ORT_RETURN_IF_ERROR(ConsumeWork());
          if (removable.find(consumer.node_index) == removable.end()) {
            Reject(MatcherFailureStage::kClosure,
                   MatcherFailureCode::kExternalPrivateUse,
                   pattern_value.producer_node_id, value_id,
                   consumer.node_index, consumer.input_index);
            matched = false;
            return Status::OK();
          }
        }
      }
      const auto implicit_consumers = snapshot.implicit_consumers.find(target_value);
      if (implicit_consumers != snapshot.implicit_consumers.end()) {
        for (const auto consumer : implicit_consumers->second) {
          ORT_RETURN_IF_ERROR(ConsumeWork());
          if (removable.find(consumer) == removable.end()) {
            Reject(MatcherFailureStage::kClosure,
                   MatcherFailureCode::kExternalPrivateUse,
                   pattern_value.producer_node_id, value_id, consumer);
            matched = false;
            return Status::OK();
          }
        }
      }
    }

    // Convexity: no path may leave R and later re-enter it.
    InlinedHashSet<NodeIndex> visited_outside;
    std::deque<NodeIndex> pending;
    auto enqueue_consumers = [&](const NodeArg* value) -> Status {
      const auto explicit_consumers = snapshot.explicit_consumers.find(value);
      if (explicit_consumers != snapshot.explicit_consumers.end()) {
        for (const auto& consumer : explicit_consumers->second) {
          ORT_RETURN_IF_ERROR(ConsumeWork());
          if (removable.find(consumer.node_index) == removable.end()) {
            pending.push_back(consumer.node_index);
          }
        }
      }
      const auto implicit_consumers = snapshot.implicit_consumers.find(value);
      if (implicit_consumers != snapshot.implicit_consumers.end()) {
        for (const auto consumer : implicit_consumers->second) {
          ORT_RETURN_IF_ERROR(ConsumeWork());
          if (removable.find(consumer) == removable.end()) pending.push_back(consumer);
        }
      }
      return Status::OK();
    };
    for (const auto node_index : plan.removable_node_indices) {
      const auto* node = snapshot.graph_viewer->GetNode(node_index);
      for (const auto* output : node->OutputDefs()) {
        if (output == nullptr || !output->Exists()) continue;
        ORT_RETURN_IF_ERROR(enqueue_consumers(output));
      }
    }
    while (!pending.empty()) {
      ORT_RETURN_IF_ERROR(ConsumeWork());
      const auto outside = pending.front();
      pending.pop_front();
      if (!visited_outside.insert(outside).second) continue;
      const auto* node = snapshot.graph_viewer->GetNode(outside);
      if (node == nullptr) continue;
      for (const auto* output : node->OutputDefs()) {
        if (output == nullptr || !output->Exists()) continue;
        const auto explicit_consumers = snapshot.explicit_consumers.find(output);
        if (explicit_consumers != snapshot.explicit_consumers.end()) {
          for (const auto& consumer : explicit_consumers->second) {
            ORT_RETURN_IF_ERROR(ConsumeWork());
            if (removable.find(consumer.node_index) != removable.end()) {
              Reject(MatcherFailureStage::kConvexity,
                     MatcherFailureCode::kNonConvex,
                     std::nullopt, std::nullopt, consumer.node_index);
              matched = false;
              return Status::OK();
            }
          }
        }
        const auto implicit_consumers = snapshot.implicit_consumers.find(output);
        if (implicit_consumers != snapshot.implicit_consumers.end()) {
          for (const auto consumer : implicit_consumers->second) {
            ORT_RETURN_IF_ERROR(ConsumeWork());
            if (removable.find(consumer) != removable.end()) {
              Reject(MatcherFailureStage::kConvexity,
                     MatcherFailureCode::kNonConvex,
                     std::nullopt, std::nullopt, consumer);
              matched = false;
              return Status::OK();
            }
          }
        }
        ORT_RETURN_IF_ERROR(enqueue_consumers(output));
      }
    }

    // Materialize the immutable replacement recipe only after validation.
    for (const auto* input : state.formal_input_bindings) {
      plan.call_inputs.push_back(const_cast<NodeArg*>(input));
    }
    for (const auto output_id : normalized.formal_output_value_ids) {
      plan.call_outputs.push_back(const_cast<NodeArg*>(state.pattern_value_to_target[output_id]));
    }
    plan.literal_witnesses = state.literal_witnesses;
    plan.matched_attribute_occurrences = state.matched_attribute_occurrences;
    for (FormalAttributeId formal_id = 0;
         formal_id < normalized.formal_attributes.size();
         ++formal_id) {
      plan.call_attributes.emplace(
          normalized.formal_attributes[formal_id].formal_name,
          *state.formal_attribute_bindings[formal_id]);
    }
    plan.pattern_node_to_target = state.pattern_node_to_target;
    plan.primary_root_topological_position =
        snapshot.topological_positions.at(
            state.pattern_node_to_target[compiled.formal_output_producer_groups[primary_output_group].producer_node_id]);

    for (const auto node_index : plan.removable_node_indices) {
      const auto* node = snapshot.graph_viewer->GetNode(node_index);
      const auto input_edges = graph_utils::GraphEdge::GetNodeInputEdges(*node);
      plan.matched_input_edges.insert(plan.matched_input_edges.end(), input_edges.begin(), input_edges.end());
      for (const auto& edge : input_edges) {
        if (removable.find(edge.src_node) == removable.end()) plan.explicit_input_edges.push_back(edge);
      }
      for (const auto& edge : graph_utils::GraphEdge::GetNodeOutputEdges(*node)) {
        if (removable.find(edge.dst_node) == removable.end()) plan.explicit_output_edges.push_back(edge);
      }
      for (const auto* output : node->OutputDefs()) {
        if (output == nullptr || !output->Exists()) continue;
        const auto implicit = snapshot.implicit_consumers.find(output);
        if (implicit != snapshot.implicit_consumers.end()) {
          plan.implicit_consumers.emplace(output, implicit->second);
        } else {
          plan.implicit_consumers.emplace(output, InlinedVector<NodeIndex>{});
        }
        if (snapshot.graph_outputs.find(output) != snapshot.graph_outputs.end()) {
          plan.graph_outputs.insert(output);
        }
      }
    }
    return Status::OK();
  }
};

bool PlansConflict(const ReplacementPlan& lhs, const ReplacementPlan& rhs) {
  InlinedHashSet<NodeIndex> lhs_nodes(lhs.removable_node_indices.begin(), lhs.removable_node_indices.end());
  for (const auto node : rhs.removable_node_indices) {
    if (lhs_nodes.find(node) != lhs_nodes.end()) return true;
  }
  InlinedHashSet<const NodeArg*> lhs_inputs(lhs.call_inputs.begin(), lhs.call_inputs.end());
  InlinedHashSet<const NodeArg*> rhs_inputs(rhs.call_inputs.begin(), rhs.call_inputs.end());
  for (const auto* output : lhs.call_outputs) {
    if (rhs_inputs.find(output) != rhs_inputs.end()) return true;
  }
  for (const auto* output : rhs.call_outputs) {
    if (lhs_inputs.find(output) != lhs_inputs.end()) return true;
  }
  return false;
}

}  // namespace

Status BuildTargetGraphSnapshot(
    const Graph& graph,
    const FunctionExtractorOptions& options,
    TargetGraphSnapshot& snapshot) {
  snapshot = TargetGraphSnapshot{};
  snapshot.graph = &graph;
  snapshot.graph_viewer = std::make_unique<GraphViewer>(graph);
  ORT_RETURN_IF(static_cast<size_t>(snapshot.graph_viewer->NumberOfNodes()) > options.max_target_nodes,
                "FunctionExtractor target node budget exceeded.");
  const auto& topological_order = snapshot.graph_viewer->GetNodesInTopologicalOrder();
  snapshot.topological_node_indices.assign(topological_order.begin(), topological_order.end());
  InlinedHashMap<std::string, NodeIndex> node_indices_by_name;
  for (const auto node_index : snapshot.topological_node_indices) {
    const auto* node = snapshot.graph_viewer->GetNode(node_index);
    ORT_RETURN_IF(node == nullptr || node->Op() == nullptr,
                  "FunctionExtractor requires a resolved target graph.");
    node_indices_by_name.emplace(node->Name(), node_index);
    snapshot.nodes_by_root_signature[RootSignatureKey(*node)].push_back(
        node_index);
  }
  for (size_t position = 0; position < snapshot.topological_node_indices.size(); ++position) {
    const auto node_index = snapshot.topological_node_indices[position];
    snapshot.topological_positions.emplace(node_index, position);
    const auto* node = snapshot.graph_viewer->GetNode(node_index);
    if (!node->ControlInputs().empty()) {
      snapshot.control_edge_nodes.insert(node_index);
      for (const auto& control_input : node->ControlInputs()) {
        const auto source = node_indices_by_name.find(control_input);
        if (source != node_indices_by_name.end()) {
          snapshot.control_edge_nodes.insert(source->second);
        }
      }
    }
    for (size_t output_index = 0; output_index < node->OutputDefs().size(); ++output_index) {
      const auto* output = node->OutputDefs()[output_index];
      if (output != nullptr && output->Exists()) {
        snapshot.producers.emplace(output, ProducerSlot{node_index, output_index});
      }
    }
    for (size_t input_index = 0; input_index < node->InputDefs().size(); ++input_index) {
      const auto* input = node->InputDefs()[input_index];
      if (input != nullptr && input->Exists()) {
        snapshot.explicit_consumers[input].push_back(ConsumerSlot{node_index, input_index});
      }
    }
    for (const auto* implicit_input : node->ImplicitInputDefs()) {
      if (implicit_input != nullptr && implicit_input->Exists()) {
        snapshot.implicit_consumers[implicit_input].push_back(node_index);
      }
    }
  }
  for (const auto* output : snapshot.graph_viewer->GetOutputs()) {
    snapshot.graph_outputs.insert(output);
  }
  for (const auto& [name, tensor] : snapshot.graph_viewer->GetAllInitializedTensors()) {
    if (graph.GetConstantInitializer(name, false) != nullptr) {
      snapshot.constant_initializers.emplace(name, tensor);
    }
  }
  return Status::OK();
}

Status BuildTargetGraphSnapshot(
    const Graph& graph,
    const CompiledFunctionPattern&,
    const FunctionExtractorOptions& options,
    TargetGraphSnapshot& snapshot) {
  return BuildTargetGraphSnapshot(graph, options, snapshot);
}

Status DiscoverReplacementPlans(
    const CompiledFunctionPattern& compiled_pattern,
    const TargetGraphSnapshot& snapshot,
    const FunctionExtractorOptions& options,
    std::vector<ReplacementPlan>& plans,
    MatcherDiagnostics* diagnostics,
    const CompleteBindingHook* complete_binding_hook,
    const MatcherExecutionOptions& execution_options) {
  plans.clear();
  const auto& groups = compiled_pattern.formal_output_producer_groups;
  ORT_RETURN_IF(groups.empty(), "FunctionExtractor pattern has no formal output producer.");
  InlinedVector<InlinedVector<NodeIndex>> candidates(groups.size());
  size_t candidate_entry_count = 0;

  size_t tuple_count = 0;
  size_t literal_bytes_compared = 0;
  size_t attribute_payload_bytes_inspected =
      compiled_pattern.normalized_pattern->pattern_attribute_payload_bytes;
  ORT_RETURN_IF(attribute_payload_bytes_inspected > options.max_attribute_bytes,
                "FunctionExtractor attribute byte budget exceeded.");
  size_t local_work_units = snapshot.aggregate_work_units;
  size_t& aggregate_work_units =
      execution_options.total_work_units == nullptr
          ? local_work_units
          : *execution_options.total_work_units;
  for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
    const auto pattern_node_id = groups[group_index].producer_node_id;
    const auto signature_candidates = snapshot.nodes_by_root_signature.find(
        RootSignatureKey(compiled_pattern.resolved_nodes[pattern_node_id]));
    if (signature_candidates == snapshot.nodes_by_root_signature.end()) {
      continue;
    }
    for (const auto node_index : signature_candidates->second) {
      ORT_RETURN_IF(aggregate_work_units >= options.max_worklist_bindings,
                    "FunctionExtractor aggregate root-index work budget exceeded.");
      ++aggregate_work_units;
      const auto* node = snapshot.graph_viewer->GetNode(node_index);
      if (!NodeSignatureMatches(
              compiled_pattern.resolved_nodes[pattern_node_id], *node)) {
        continue;
      }
      ORT_RETURN_IF(candidate_entry_count >= options.max_output_root_tuples,
                    "FunctionExtractor root candidate-entry budget exceeded.");
      candidates[group_index].push_back(node_index);
      ++candidate_entry_count;
    }
  }
  InlinedVector<NodeIndex> tuple(groups.size(), std::numeric_limits<NodeIndex>::max());
  InlinedVector<size_t> group_order(groups.size());
  for (size_t i = 0; i < group_order.size(); ++i) group_order[i] = i;
  std::sort(group_order.begin(), group_order.end(), [&](size_t lhs, size_t rhs) {
    return std::pair{candidates[lhs].size(), lhs} < std::pair{candidates[rhs].size(), rhs};
  });
  const size_t primary_output_group = group_order.front();
  InlinedHashSet<NodeIndex> chosen_nodes;
  Status enumeration_status = Status::OK();
  std::function<void(size_t)> enumerate = [&](size_t order_index) {
    if (!enumeration_status.IsOK()) return;
    if (order_index == groups.size()) {
      if (aggregate_work_units >= options.max_worklist_bindings) {
        enumeration_status =
            MatcherError("aggregate output-root tuple construction budget exceeded");
        return;
      }
      ++aggregate_work_units;
      ++tuple_count;
      if (tuple_count > options.max_output_root_tuples) {
        enumeration_status =
            MatcherError("output-root tuple budget exceeded");
        return;
      }
      if (execution_options.total_attempts != nullptr) {
        if (*execution_options.total_attempts >=
            execution_options.max_attempts) {
          enumeration_status =
              MatcherError("rule-attempt budget exceeded");
          return;
        }
        ++*execution_options.total_attempts;
      }
      if (diagnostics != nullptr) ++diagnostics->output_root_tuples_considered;
      CandidateMatcher matcher{compiled_pattern, snapshot, options, diagnostics,
                               literal_bytes_compared, attribute_payload_bytes_inspected,
                               aggregate_work_units,
                               primary_output_group, complete_binding_hook,
                               execution_options
                                   .allow_omitted_optional_formal_inputs,
                               execution_options.failure_hook != nullptr};
      matcher.state.anchor_node = tuple[primary_output_group];
      matcher.state.anchor_output_slot =
          groups[primary_output_group].producer_output_indices.front();
      matcher.state.anchor_rank =
          snapshot.topological_node_indices.size() - 1 -
          snapshot.topological_positions.at(matcher.state.anchor_node);
      matcher.state.tuple_ordinal = tuple_count - 1;
      ReplacementPlan plan;
      bool matched = false;
      enumeration_status = matcher.Run(tuple, plan, matched);
      if (!enumeration_status.IsOK()) return;
      if (!matched) {
        if (execution_options.failure_hook != nullptr &&
            matcher.failure.valid) {
          (*execution_options.failure_hook)(
              matcher.failure, matcher.state.anchor_node,
              matcher.state.anchor_output_slot,
              matcher.state.anchor_rank, matcher.state.tuple_ordinal);
        }
        return;
      }
      if (diagnostics != nullptr) {
        ++diagnostics->structurally_matched_candidates;
        ++diagnostics->accepted_candidates;
      }
      plan.anchor_rank =
          snapshot.topological_node_indices.size() - 1 -
          plan.primary_root_topological_position;
      plan.anchor_node = matcher.state.anchor_node;
      plan.anchor_output_slot =
          matcher.state.anchor_output_slot;
      plan.tuple_ordinal = tuple_count - 1;
      plans.push_back(std::move(plan));
      return;
    }
    const size_t group_index = group_order[order_index];
    for (const auto candidate : candidates[group_index]) {
      if (aggregate_work_units >= options.max_worklist_bindings) {
        enumeration_status =
            MatcherError("aggregate output-root enumeration budget exceeded");
        return;
      }
      ++aggregate_work_units;
      if (!chosen_nodes.insert(candidate).second) continue;
      tuple[group_index] = candidate;
      enumerate(order_index + 1);
      chosen_nodes.erase(candidate);
    }
  };
  enumerate(0);
  ORT_RETURN_IF_ERROR(enumeration_status);

  std::sort(plans.begin(), plans.end(), [](const ReplacementPlan& lhs, const ReplacementPlan& rhs) {
    return std::tie(lhs.primary_root_topological_position, lhs.removable_node_indices) <
           std::tie(rhs.primary_root_topological_position, rhs.removable_node_indices);
  });
  return Status::OK();
}

Status SelectNonConflictingPlans(
    gsl::span<const ReplacementPlan> plans,
    std::vector<size_t>& selected_plan_indices) {
  selected_plan_indices.clear();
  for (size_t candidate_index = 0; candidate_index < plans.size(); ++candidate_index) {
    bool conflict = false;
    for (const auto selected_index : selected_plan_indices) {
      if (PlansConflict(plans[candidate_index], plans[selected_index])) {
        conflict = true;
        break;
      }
    }
    if (!conflict) selected_plan_indices.push_back(candidate_index);
  }
  return Status::OK();
}

bool ReplacementPlansConflict(
    const ReplacementPlan& lhs,
    const ReplacementPlan& rhs) {
  return PlansConflict(lhs, rhs);
}

Status PrevalidatePlans(
    const Graph& graph,
    const CompiledFunctionPattern& compiled_pattern,
    gsl::span<const ReplacementPlan> plans,
    bool require_registered_pattern) {
  if (require_registered_pattern) {
    ORT_RETURN_IF_ERROR(ValidateRegisteredFunction(*compiled_pattern.normalized_pattern, graph));
  }
  FunctionExtractorOptions snapshot_options;
  snapshot_options.max_target_nodes = std::numeric_limits<size_t>::max();
  TargetGraphSnapshot snapshot;
  CompiledFunctionPattern snapshot_pattern;
  snapshot_pattern.normalized_pattern = compiled_pattern.normalized_pattern;
  ORT_RETURN_IF_ERROR(BuildTargetGraphSnapshot(graph, snapshot_pattern, snapshot_options, snapshot));

  auto edge_less = [](const graph_utils::GraphEdge& lhs, const graph_utils::GraphEdge& rhs) {
    return std::tie(lhs.src_node, lhs.dst_node, lhs.src_arg_index, lhs.dst_arg_index, lhs.arg_name) <
           std::tie(rhs.src_node, rhs.dst_node, rhs.src_arg_index, rhs.dst_arg_index, rhs.arg_name);
  };
  auto edges_equal = [&](std::vector<graph_utils::GraphEdge> lhs,
                         std::vector<graph_utils::GraphEdge> rhs) {
    std::sort(lhs.begin(), lhs.end(), edge_less);
    std::sort(rhs.begin(), rhs.end(), edge_less);
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (edge_less(lhs[i], rhs[i]) || edge_less(rhs[i], lhs[i])) return false;
    }
    return true;
  };

  for (size_t plan_index = 0; plan_index < plans.size(); ++plan_index) {
    const auto& plan = plans[plan_index];
    ORT_RETURN_IF(plan.removable_node_indices.size() <= 1,
                  "Replacement plan does not strictly reduce the graph node count.");
    for (const auto node_index : plan.removable_node_indices) {
      const auto* node = graph.GetNode(node_index);
      ORT_RETURN_IF(node == nullptr, "Replacement plan references a removed node.");
      ORT_RETURN_IF(node->GetLayeringAnnotation() != plan.layering_annotation,
                    "Replacement plan layering annotation changed.");
    }
    ORT_RETURN_IF(plan.pattern_node_to_target.size() != compiled_pattern.resolved_nodes.size(),
                  "Replacement plan pattern mapping changed.");
    for (PatternNodeId pattern_node_id = 0;
         pattern_node_id < plan.pattern_node_to_target.size();
         ++pattern_node_id) {
      const auto target_node_index = plan.pattern_node_to_target[pattern_node_id];
      const auto* node = graph.GetNode(target_node_index);
      ORT_RETURN_IF(node == nullptr ||
                        !NodeSignatureMatches(compiled_pattern.resolved_nodes[pattern_node_id], *node) ||
                        !node->GetExecutionProviderType().empty() ||
                        snapshot.control_edge_nodes.find(target_node_index) !=
                            snapshot.control_edge_nodes.end(),
                    "Replacement plan node semantics changed.");
    }
    ORT_RETURN_IF(plan.call_attributes.size() !=
                      compiled_pattern.normalized_pattern->formal_attributes.size(),
                  "Replacement plan call attributes changed.");
    for (const auto& occurrence : plan.matched_attribute_occurrences) {
      const auto* node = graph.GetNode(occurrence.target_node_index);
      ORT_RETURN_IF(node == nullptr, "Replacement plan attribute node was removed.");
      const auto* target_attribute =
          EffectiveTargetAttribute(*node, occurrence.operator_attribute_name);
      ORT_RETURN_IF(target_attribute == nullptr,
                    "Replacement plan target attribute became missing.");
      const auto& formal =
          compiled_pattern.normalized_pattern->formal_attributes[occurrence.formal_attribute_id];
      const size_t max_attribute_bytes =
          compiled_pattern.normalized_pattern->max_attribute_bytes;
      ONNX_NAMESPACE::AttributeProto canonical;
      ORT_RETURN_IF_ERROR(CanonicalizeFormalAttribute(
          formal.formal_name, formal.type, *target_attribute,
          max_attribute_bytes, canonical));
      bool equal = false;
      ORT_RETURN_IF_ERROR(CompareFormalAttributes(
          canonical, occurrence.canonical_value,
          max_attribute_bytes, equal));
      ORT_RETURN_IF_NOT(equal, "Replacement plan target attribute changed.");
      const auto call_attribute = plan.call_attributes.find(formal.formal_name);
      ORT_RETURN_IF(call_attribute == plan.call_attributes.end(),
                    "Replacement plan formal attribute became missing.");
      ORT_RETURN_IF_ERROR(CompareFormalAttributes(
          canonical, call_attribute->second,
          max_attribute_bytes, equal));
      ORT_RETURN_IF_NOT(equal, "Replacement plan formal binding changed.");
    }
    std::vector<graph_utils::GraphEdge> current_input_edges;
    std::vector<graph_utils::GraphEdge> current_output_edges;
    InlinedHashSet<NodeIndex> removable(plan.removable_node_indices.begin(),
                                        plan.removable_node_indices.end());
    for (const auto node_index : plan.removable_node_indices) {
      const auto node_edges = graph_utils::GraphEdge::GetNodeInputEdges(*graph.GetNode(node_index));
      current_input_edges.insert(current_input_edges.end(), node_edges.begin(), node_edges.end());
      for (const auto& edge : graph_utils::GraphEdge::GetNodeOutputEdges(*graph.GetNode(node_index))) {
        if (removable.find(edge.dst_node) == removable.end()) current_output_edges.push_back(edge);
      }
    }
    ORT_RETURN_IF_NOT(edges_equal(current_input_edges, plan.matched_input_edges),
                      "Replacement plan input edges changed.");
    ORT_RETURN_IF_NOT(edges_equal(current_output_edges, plan.explicit_output_edges),
                      "Replacement plan output edges changed.");
    for (const auto* input : plan.call_inputs) {
      ORT_RETURN_IF(input != nullptr && !input->Exists(),
                    "Replacement plan has a stale call input.");
    }
    for (const auto* output : plan.call_outputs) {
      ORT_RETURN_IF(output == nullptr || !output->Exists(), "Replacement plan has a stale call output.");
    }
    for (size_t other_index = 0; other_index < plan_index; ++other_index) {
      ORT_RETURN_IF(PlansConflict(plan, plans[other_index]),
                    "Selected replacement plans conflict.");
    }
    for (const auto& [value, expected_consumers] : plan.implicit_consumers) {
      const auto current = snapshot.implicit_consumers.find(value);
      InlinedVector<NodeIndex> current_consumers;
      if (current != snapshot.implicit_consumers.end()) current_consumers = current->second;
      auto expected = expected_consumers;
      std::sort(current_consumers.begin(), current_consumers.end());
      std::sort(expected.begin(), expected.end());
      ORT_RETURN_IF(current_consumers != expected,
                    "Replacement plan implicit consumers changed.");
      const bool currently_graph_output =
          snapshot.graph_outputs.find(value) != snapshot.graph_outputs.end();
      const bool expected_graph_output =
          plan.graph_outputs.find(value) != plan.graph_outputs.end();
      ORT_RETURN_IF(currently_graph_output != expected_graph_output,
                    "Replacement plan graph-output membership changed.");
    }

    InlinedHashSet<NodeIndex> visited_outside;
    std::deque<NodeIndex> pending;
    for (const auto node_index : plan.removable_node_indices) {
      const auto* node = graph.GetNode(node_index);
      for (const auto* output : node->OutputDefs()) {
        if (output == nullptr || !output->Exists()) continue;
        const auto explicit_consumers = snapshot.explicit_consumers.find(output);
        if (explicit_consumers != snapshot.explicit_consumers.end()) {
          for (const auto& consumer : explicit_consumers->second) {
            if (removable.find(consumer.node_index) == removable.end()) {
              pending.push_back(consumer.node_index);
            }
          }
        }
        const auto implicit_consumers = snapshot.implicit_consumers.find(output);
        if (implicit_consumers != snapshot.implicit_consumers.end()) {
          for (const auto consumer : implicit_consumers->second) {
            if (removable.find(consumer) == removable.end()) pending.push_back(consumer);
          }
        }
      }
    }
    while (!pending.empty()) {
      const auto outside = pending.front();
      pending.pop_front();
      if (!visited_outside.insert(outside).second) continue;
      const auto* node = graph.GetNode(outside);
      ORT_RETURN_IF(node == nullptr, "Replacement plan outside dependency changed.");
      for (const auto* output : node->OutputDefs()) {
        if (output == nullptr || !output->Exists()) continue;
        const auto explicit_consumers = snapshot.explicit_consumers.find(output);
        if (explicit_consumers != snapshot.explicit_consumers.end()) {
          for (const auto& consumer : explicit_consumers->second) {
            ORT_RETURN_IF(removable.find(consumer.node_index) != removable.end(),
                          "Replacement plan is no longer convex.");
            pending.push_back(consumer.node_index);
          }
        }
        const auto implicit_consumers = snapshot.implicit_consumers.find(output);
        if (implicit_consumers != snapshot.implicit_consumers.end()) {
          for (const auto consumer : implicit_consumers->second) {
            ORT_RETURN_IF(removable.find(consumer) != removable.end(),
                          "Replacement plan is no longer convex.");
            pending.push_back(consumer);
          }
        }
      }
    }
    if (!plan.generated_call_name.empty()) {
      for (const auto& node : graph.Nodes()) {
        ORT_RETURN_IF(node.Name() == plan.generated_call_name,
                      "Replacement plan call node name is no longer unique.");
      }
    }
    for (const auto& witness : plan.literal_witnesses) {
      ORT_RETURN_IF(witness.target_value == nullptr || !witness.target_value->Exists(),
                    "Replacement plan has a stale literal witness.");
      if (witness.is_initializer) {
        const auto* tensor = graph.GetConstantInitializer(witness.target_value->Name(), false);
        ORT_RETURN_IF(tensor == nullptr,
                      "Replacement plan initializer witness is no longer constant.");
        bool equal = false;
        const auto& pattern_value =
            compiled_pattern.normalized_pattern->values[witness.pattern_value_id];
        const auto& model_path = graph.ModelPath();
        ORT_RETURN_IF_ERROR(CompareTensorLiterals(pattern_value.literal.tensor, *tensor,
                                                  pattern_value.literal.byte_count, equal, &model_path));
        ORT_RETURN_IF_NOT(equal, "Replacement plan initializer literal changed.");
      } else {
        const auto* constant = graph.GetNode(witness.constant_node_index);
        ORT_RETURN_IF(constant == nullptr || constant->Domain() != kOnnxDomain ||
                          constant->OpType() != "Constant",
                      "Replacement plan Constant witness was removed.");
        ONNX_NAMESPACE::TensorProto normalized_constant;
        ORT_RETURN_IF_ERROR(
            NormalizeConstantAttributes(constant->GetAttributes(), normalized_constant));
        bool equal = false;
        const auto& pattern_value =
            compiled_pattern.normalized_pattern->values[witness.pattern_value_id];
        ORT_RETURN_IF_ERROR(CompareTensorLiterals(pattern_value.literal.tensor, normalized_constant,
                                                  pattern_value.literal.byte_count, equal));
        ORT_RETURN_IF_NOT(equal, "Replacement plan Constant literal changed.");
      }
    }
  }
  return Status::OK();
}

}  // namespace function_extractor_internal
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
