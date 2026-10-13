// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/pattern_rewriter.h"

#include <algorithm>
#include <limits>
#include <sstream>

#include "core/graph/graph_utils.h"
#include "onnx/checker.h"

namespace onnxruntime {
namespace rewrite {
namespace {

std::string Key(const std::string& op, const std::string& domain, int version) {
  return domain + '\0' + op + '\0' + std::to_string(version);
}

std::string Signature(const Pattern& pattern) {
  std::ostringstream stream;
  stream << pattern.ValueCount() << ':';
  for (const auto& node : pattern.Nodes()) {
    stream << node.op.domain.size() << ':' << node.op.domain << node.op.name.size() << ':' << node.op.name;
    for (int version : node.op.versions) stream << ':' << version;
    stream << '/' << node.output;
    for (Value input : node.inputs) stream << ',' << input;
    stream << ';';
  }
  return stream.str();
}

using Producers = InlinedHashMap<const NodeArg*, NodeIndex>;

bool MatchPattern(const Pattern& pattern, const Graph& graph, const Node& root, const Producers& producers,
                  Match& match) {
  const auto& root_op = pattern.Nodes().back().op;
  // Cheap root rejection precedes binding allocation in the reference implementation too.
  if (root.OpType() != root_op.name || root.Domain() != root_op.domain ||
      std::find(root_op.versions.begin(), root_op.versions.end(), root.SinceVersion()) == root_op.versions.end()) {
    return false;
  }
  match.values.assign(pattern.ValueCount(), nullptr);
  match.nodes.assign(pattern.Nodes().size(), nullptr);
  match.nodes.back() = &root;
  InlinedHashSet<NodeIndex> distinct;
  for (size_t i = pattern.Nodes().size(); i-- > 0;) {
    const auto& expected = pattern.Nodes()[i];
    const Node* node = match.nodes[i];
    if (!node && match.values[expected.output]) {
      auto it = producers.find(match.values[expected.output]);
      if (it != producers.end()) node = graph.GetNode(it->second);
    }
    if (!node || !distinct.insert(node->Index()).second || node->OpType() != expected.op.name ||
        node->Domain() != expected.op.domain ||
        std::find(expected.op.versions.begin(), expected.op.versions.end(), node->SinceVersion()) ==
            expected.op.versions.end() ||
        node->InputDefs().size() != expected.inputs.size() || node->OutputDefs().size() != 1 ||
        !node->GetAttributeNameToSubgraphMap().empty() || !node->ImplicitInputDefs().empty() ||
        node->GetLayeringAnnotation() != root.GetLayeringAnnotation() ||
        node->GetExecutionProviderType() != root.GetExecutionProviderType()) {
      return false;
    }
    match.nodes[i] = node;
    const NodeArg* output = node->OutputDefs()[0];
    if (!output->Exists() || (match.values[expected.output] && match.values[expected.output] != output)) return false;
    match.values[expected.output] = output;
    for (size_t j = 0; j < expected.inputs.size(); ++j) {
      const NodeArg* actual = node->InputDefs()[j];
      const NodeArg*& binding = match.values[expected.inputs[j]];
      if (!actual->Exists() || (binding && binding != actual)) return false;
      binding = actual;
    }
  }
  // Input variables must be outside the deletion set, even when aliasing another binding.
  for (Value value = 0; value < pattern.ValueCount(); ++value) {
    if (!pattern.IsInput(value)) continue;
    auto it = producers.find(match.values[value]);
    if (it != producers.end() && distinct.count(it->second)) return false;
  }
  for (size_t i = 0; i + 1 < match.nodes.size(); ++i) {
    const Node& node = *match.nodes[i];
    if (graph.NodeProducesGraphOutput(node)) return false;
    for (auto edge = node.OutputEdgesBegin(); edge != node.OutputEdgesEnd(); ++edge) {
      if (!distinct.count(edge->GetNode().Index())) return false;
    }
  }
  return true;
}

Status Replace(Graph& graph, const Pattern& pattern, const Match& match, const Replacement& recipe,
               Producers& producers, NodeIndex& added) {
  auto imported = graph.DomainToVersionMap().find(recipe.domain);
  ORT_RETURN_IF(imported == graph.DomainToVersionMap().end(), "Replacement domain is not imported: ", recipe.domain);
  InlinedVector<NodeArg*> inputs;
  for (Value value : recipe.inputs) {
    ORT_RETURN_IF(value >= match.values.size() || !pattern.IsInput(value),
                  "Replacement inputs must be pattern input variables.");
    inputs.push_back(graph.GetNodeArg(match.values[value]->Name()));
  }
  const Node& root = *match.nodes.back();
  NodeArg* output = graph.GetNodeArg(root.OutputDefs()[0]->Name());
  Node& replacement = graph.AddNode(graph.GenerateNodeName("rewrite"), recipe.op, "PatternRewriter", inputs, {output},
                                    root, &recipe.attributes, recipe.domain);
  // Validate the imported schema before deleting anything.
  if (!graph.SetOpSchemaFromRegistryForNode(replacement) || replacement.SinceVersion() != recipe.version) {
    graph.RemoveNode(replacement.Index());
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Replacement schema/version is unavailable: ", recipe.op);
  }
  ORT_TRY {
    ONNX_NAMESPACE::NodeProto proto;
    replacement.ToProto(proto);
    replacement.Op()->Verify(proto);
  }
  ORT_CATCH(const ONNX_NAMESPACE::checker::ValidationError& error) {
    graph.RemoveNode(replacement.Index());
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid replacement: ", error.what());
  }
  replacement.SetExecutionProviderType(root.GetExecutionProviderType());
  added = replacement.Index();
  struct Edge {
    NodeIndex destination;
    int source_arg;
    int destination_arg;
  };
  InlinedVector<Edge> outgoing;
  for (auto it = root.OutputEdgesBegin(); it != root.OutputEdgesEnd(); ++it) {
    outgoing.push_back({it->GetNode().Index(), it->GetSrcArgIndex(), it->GetDstArgIndex()});
  }
  InlinedVector<NodeIndex> removed;
  for (const Node* node : match.nodes) {
    removed.push_back(node->Index());
    producers.erase(node->OutputDefs()[0]);
    for (const NodeArg* input : node->InputDefs()) {
      graph.RemoveConsumerNode(input->Name(), graph.GetNode(node->Index()));
    }
    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(node->Index()));
  }
  // Remove in reverse order so producers cannot be left with dangling edges.
  for (auto it = removed.rbegin(); it != removed.rend(); ++it) {
    ORT_ENFORCE(graph.RemoveNode(*it), "Failed to remove matched node.");
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    graph.AddConsumerNode(inputs[i]->Name(), &replacement);
    auto it = producers.find(inputs[i]);
    if (it == producers.end()) continue;
    const Node& producer = *graph.GetNode(it->second);
    auto output_it = std::find(producer.OutputDefs().begin(), producer.OutputDefs().end(), inputs[i]);
    graph.AddEdge(producer.Index(), added, static_cast<int>(std::distance(producer.OutputDefs().begin(), output_it)),
                  static_cast<int>(i));
  }
  for (const auto& edge : outgoing) graph.AddEdge(added, edge.destination, edge.source_arg, edge.destination_arg);
  producers[output] = added;
  graph.UpdateProducerNode(output->Name(), added);
  graph.NotifyNodeReplacement(removed, added);
  return Status::OK();
}
}  // namespace

Value Pattern::Input() {
  ORT_ENFORCE(value_count_ < 256, "Patterns are limited to 256 values.");
  return value_count_++;
}

Value Pattern::Op(Operator op, std::initializer_list<Value> inputs) {
  ORT_ENFORCE(!op.name.empty() && !op.versions.empty(), "Pattern operators need a name and explicit schema versions.");
  ORT_ENFORCE(nodes_.size() < 64, "Patterns are limited to 64 nodes.");
  ORT_ENFORCE(value_count_ < 256 && inputs.size() <= 64,
              "Patterns are limited to 256 values and 64 inputs per operator.");
  if (op.domain == kOnnxDomainAlias) op.domain = kOnnxDomain;
  for (int version : op.versions) ORT_ENFORCE(version > 0, "Schema versions must be positive.");
  std::sort(op.versions.begin(), op.versions.end());
  op.versions.erase(std::unique(op.versions.begin(), op.versions.end()), op.versions.end());
  for (Value input : inputs) ORT_ENFORCE(input < value_count_, "Pattern inputs must already be defined.");
  Value output = value_count_++;
  nodes_.push_back({std::move(op), InlinedVector<Value>(inputs), output});
  return output;
}

bool Pattern::IsInput(Value value) const {
  return std::none_of(nodes_.begin(), nodes_.end(), [value](const auto& node) { return node.output == value; });
}

PatternRewriter::PatternRewriter(Rules rules, Options options)
    : GraphTransformer("PatternRewriter"), rules_(std::move(rules)), options_(options) {
  InlinedHashMap<std::string, size_t> signatures;
  for (size_t i = 0; i < rules_.size(); ++i) {
    const Rule& rule = rules_[i];
    ORT_ENFORCE(!rule.name.empty() && rule.pattern.Nodes().size() >= 2 && rule.guard && rule.replacement,
                "Rules need a name, at least two nodes, a guard and a recipe.");
    // Every node must reach the last node. Disconnected definitions are not patterns.
    InlinedHashSet<Value> reachable{rule.pattern.Nodes().back().output};
    for (auto it = rule.pattern.Nodes().rbegin(); it != rule.pattern.Nodes().rend(); ++it) {
      ORT_ENFORCE(reachable.count(it->output), "Disconnected pattern node.");
      reachable.insert(it->inputs.begin(), it->inputs.end());
    }
    ORT_ENFORCE(reachable.size() == rule.pattern.ValueCount(), "Unused pattern input.");
    invalidation_depth_ = std::max(invalidation_depth_, rule.pattern.Nodes().size());
    if (!options_.indexed) continue;
    std::string signature = Signature(rule.pattern);
    auto [it, inserted] = signatures.emplace(signature, groups_.size());
    if (inserted) {
      const auto& root = rule.pattern.Nodes().back().op;
      for (int version : root.versions) {
        roots_[Key(root.name, root.domain, version)].push_back(groups_.size());
      }
      groups_.push_back({rule.pattern, {}});
    }
    groups_[it->second].rules.push_back(i);
  }
}

Status PatternRewriter::Run(Graph& graph, Statistics& statistics) const {
  statistics = {};
  ORT_RETURN_IF(graph.GraphResolveNeeded(), "PatternRewriter requires an already resolved graph.");
  Producers producers;
  InlinedVector<NodeIndex> worklist;
  InlinedHashSet<NodeIndex> queued;
  GraphViewer viewer(graph);
  for (NodeIndex index : viewer.GetNodesInTopologicalOrder()) {
    const Node& node = *graph.GetNode(index);
    worklist.push_back(index);
    queued.insert(index);
    for (const NodeArg* output : node.OutputDefs()) {
      if (output->Exists()) producers[output] = index;
    }
  }
  // Nodes are revisited only after a strictly node-count-decreasing rewrite.
  for (size_t cursor = 0; cursor < worklist.size(); ++cursor) {
    queued.erase(worklist[cursor]);
    const Node* root = graph.GetNode(worklist[cursor]);
    if (!root) continue;
    ++statistics.visited;
    size_t winner = std::numeric_limits<size_t>::max();
    Match winning_match;
    const Pattern* winning_pattern = nullptr;
    auto consider = [&](const Pattern& pattern, gsl::span<const size_t> rule_ids) {
      statistics.candidates += rule_ids.size();
      ++statistics.structural_checks;
      Match match;
      if (!MatchPattern(pattern, graph, *root, producers, match)) return;
      for (size_t id : rule_ids) {
        if (!options_.match_only && id >= winner) break;
        ++statistics.guards;
        if (!rules_[id].guard(graph, match)) continue;
        if (options_.match_only) ++statistics.matches;
        if (id < winner) {
          winner = id;
          winning_match = match;
          winning_pattern = &pattern;
        }
        if (!options_.match_only) break;
      }
    };
    if (options_.indexed) {
      auto it = roots_.find(Key(root->OpType(), root->Domain(), root->SinceVersion()));
      if (it != roots_.end()) {
        for (size_t id : it->second) {
          const auto& group = groups_[id];
          if (!options_.match_only && group.rules.front() >= winner) continue;
          consider(group.pattern, group.rules);
        }
      }
    } else {
      for (size_t id = 0; id < rules_.size(); ++id) {
        consider(rules_[id].pattern, gsl::span<const size_t>(&id, 1));
        if (!options_.match_only && winner != std::numeric_limits<size_t>::max()) break;
      }
    }
    if (options_.match_only || !winning_pattern) continue;
    ++statistics.matches;
    InlinedVector<NodeIndex> affected;
    for (Value value = 0; value < winning_pattern->ValueCount(); ++value) {
      if (!winning_pattern->IsInput(value)) continue;
      auto it = producers.find(winning_match.values[value]);
      if (it != producers.end()) affected.push_back(it->second);
    }
    NodeIndex added;
    ORT_RETURN_IF_ERROR(Replace(graph, *winning_pattern, winning_match,
                                rules_[winner].replacement(graph, winning_match), producers, added));
    ++statistics.rewrites;
    affected.push_back(added);
    // A changed producer or use count can affect a deeper pattern root, not just a direct consumer.
    InlinedHashSet<NodeIndex> seen;
    for (size_t depth = 0; depth <= invalidation_depth_ && !affected.empty(); ++depth) {
      std::sort(affected.begin(), affected.end());
      InlinedVector<NodeIndex> next;
      for (NodeIndex index : affected) {
        if (!seen.insert(index).second) continue;
        const Node* node = graph.GetNode(index);
        if (!node) continue;
        if (queued.insert(index).second) worklist.push_back(index);
        for (auto it = node->OutputEdgesBegin(); it != node->OutputEdgesEnd(); ++it) {
          next.push_back(it->GetNode().Index());
        }
      }
      affected = std::move(next);
    }
  }
  return Status::OK();
}

Status PatternRewriter::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  for (auto& node : graph.Nodes()) ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  Statistics statistics;
  ORT_RETURN_IF_ERROR(Run(graph, statistics));
  modified |= statistics.rewrites != 0;
  return Status::OK();
}

}  // namespace rewrite
}  // namespace onnxruntime
