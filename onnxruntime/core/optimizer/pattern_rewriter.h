// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <functional>
#include <initializer_list>
#include <string>

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {
namespace rewrite {

using Value = size_t;

struct Operator {
  std::string name;
  std::string domain;
};

struct OpsetRequirement {
  std::string domain;
  int min_version;
  int max_version;  // Inclusive model/function import range, not schema since-versions.
};

// A straight-line definition describes a DAG, not necessarily a chain.
// Reusing a Value requires exactly the same NodeArg, including for input variables.
class Pattern {
 public:
  Value Input();
  Value Op(Operator op, std::initializer_list<Value> inputs);

  struct NodePattern {
    Operator op;
    InlinedVector<Value> inputs;
    Value output;
  };
  const auto& Nodes() const { return nodes_; }
  size_t ValueCount() const { return value_count_; }
  bool IsInput(Value value) const;

 private:
  size_t value_count_ = 0;
  InlinedVector<NodePattern, 1> nodes_;
};

struct Match {
  InlinedVector<const Node*> nodes;
  InlinedVector<const NodeArg*> values;
};

struct Replacement {
  std::string op;
  std::string domain;
  InlinedVector<Value> inputs;
  NodeAttributes attributes;
};

struct Rule {
  std::string name;
  Pattern pattern;
  InlinedVector<OpsetRequirement, 1> opsets;
  // Guards and recipes must not mutate the graph or retain pointers into a Match.
  std::function<bool(const Graph&, const Match&)> guard;
  std::function<Replacement(const Graph&, const Match&)> replacement;
};

using Rules = InlinedVector<Rule, 1>;

struct Statistics {
  size_t visited = 0;
  size_t candidates = 0;
  size_t structural_checks = 0;
  size_t guards = 0;
  size_t matches = 0;
  size_t rewrites = 0;
};

struct Options {
  bool match_only = false;
  bool indexed = true;  // false is the rule-by-rule reference implementation.
};

// Immutable compiled rules; statistics belong to a call, not this transformer.
// Not registered among ORT's default optimizers. Call Apply or Run explicitly.
class PatternRewriter final : public GraphTransformer {
 public:
  explicit PatternRewriter(Rules rules, Options options = {});
  Status Run(Graph& graph, Statistics& statistics) const;

 private:
  struct Group {
    Pattern pattern;
    InlinedVector<size_t> rules;
  };
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  Rules rules_;
  InlinedVector<Group, 1> groups_;
  InlinedHashMap<std::string, InlinedVector<size_t>> roots_;
  size_t invalidation_depth_ = 0;
  Options options_;
};

// Conservative CPU/unassigned, float32 examples adapted from genuine ONNXScript rules.
Rules MakeExampleRules();

}  // namespace rewrite
}  // namespace onnxruntime
