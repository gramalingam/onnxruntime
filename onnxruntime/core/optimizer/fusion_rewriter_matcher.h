#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/graph/graph.h"
#include "core/optimizer/fusion_rewriter.h"
#include "gsl/gsl"

namespace onnxruntime::fusion_rewriter_internal {

enum class ObservedDependencyKind : uint8_t {
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

struct ObservedDependencySummary {
  ObservedDependencyKind kind;
  FusionPatternNodeId pattern_node{};
  FusionPatternValueId pattern_value{};
  FusionFormalAttributeId formal_attribute{};
  std::string name;
  int64_t axis{-1};
};

class FusionTestPlan final {
 public:
  FusionTestPlan();
  ~FusionTestPlan();
  FusionTestPlan(FusionTestPlan&&) noexcept;
  FusionTestPlan& operator=(FusionTestPlan&&) noexcept;
  FusionTestPlan(const FusionTestPlan&) = delete;
  FusionTestPlan& operator=(const FusionTestPlan&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class FusionRuleSetTestAccess;
};

using GraphResolveFunction =
    common::Status (*)(Graph&, const Graph::ResolveOptions&);
using EpochObserverFunction =
    void (*)(void* state, size_t epoch,
             size_t nodes_before, size_t nodes_after);

struct FusionExecutionControls {
  std::optional<size_t> maximum_epochs;
  GraphResolveFunction resolve_graph{};
  EpochObserverFunction epoch_observer{};
  void* epoch_observer_state{};
};

class FusionRuleSetTestAccess final {
 public:
  static common::Status DiscoverPlans(
      const FusionRuleSet& rule_set, Graph& graph,
      std::vector<FusionTestPlan>& plans);
  static common::Status PrevalidatePlans(
      const FusionRuleSet& rule_set, Graph& graph,
      gsl::span<const FusionTestPlan> plans);
  static gsl::span<const ObservedDependencySummary> ObservedDependencies(
      const FusionTestPlan& plan);
  static FusionRewriteResult Apply(
      const FusionRuleSet& rule_set, Graph& graph,
      const FusionExecutionControls& controls,
      FusionTraceCollector* trace = nullptr);
};

}  // namespace onnxruntime::fusion_rewriter_internal

#endif  // !defined(ORT_MINIMAL_BUILD)
