#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <memory>

#include "core/optimizer/fusion_rewriter.h"

namespace onnxruntime::fusion_rewriter_internal {

// Invocation-local diagnostics sink. The driver owns it and passes a nullable
// pointer to matcher rejection sites. kOff is represented by a null pointer.
class FailureSink final {
 public:
  explicit FailureSink(FusionTraceCollector& collector,
                       FusionDiagnosticMode mode);

  FusionDiagnosticMode Mode() const noexcept;

  void RegisterRule(FusionRuleId rule_id,
                    int32_t anchor_local_priority,
                    size_t registration_order);
  void RecordFailure(const FusionFailureRecord& record,
                     size_t epoch, size_t anchor_rank,
                     size_t tuple_ordinal);
  void RecordSuccess(FusionRuleId rule_id);
  void RecordSuccess(const FusionFailureRecord& record);

 private:
  FusionTraceCollector& collector_;
  FusionDiagnosticMode mode_;
};

// Holds the invariant metadata shared by every rejection from one output-root
// tuple. Construct this only when failure_sink is non-null.
class MatchAttempt final {
 public:
  MatchAttempt(FailureSink& failure_sink,
               FusionRuleId rule_id,
               NodeIndex anchor_node,
               size_t anchor_output_slot,
               size_t epoch,
               size_t anchor_rank,
               size_t tuple_ordinal);

  void SetPatternNodesMatched(size_t pattern_nodes_matched) noexcept;
  FusionFailureRecord MakeFailure(FusionMatchStage stage,
                                  FusionFailureCode code) const;
  void RecordFailure(FusionFailureRecord record) const;
  FusionFailureRecord MakeSuccess() const;

 private:
  FailureSink& failure_sink_;
  FusionRuleId rule_id_;
  NodeIndex anchor_node_;
  size_t anchor_output_slot_;
  size_t epoch_;
  size_t anchor_rank_;
  size_t tuple_ordinal_;
  size_t pattern_nodes_matched_{};
};

// Configures collector for one Apply invocation. In kOff mode this clears a
// supplied collector and returns a null sink without allocating one.
common::Status CreateFailureSink(
    FusionTraceCollector* collector,
    FusionDiagnosticMode mode,
    size_t max_records,
    size_t max_bytes,
    std::unique_ptr<FailureSink>& failure_sink);

class FusionDiagnosticsTestAccess final {
 public:
  static void Configure(FusionTraceCollector& collector,
                        FusionDiagnosticMode mode,
                        size_t max_records, size_t max_bytes);
  static void RecordFailure(FusionTraceCollector& collector,
                            const FusionFailureRecord& record,
                            size_t epoch, size_t anchor_rank,
                            size_t tuple_ordinal);
  static void RecordSuccess(FusionTraceCollector& collector,
                            FusionRuleId rule_id);

 private:
  static void RegisterRule(FusionTraceCollector& collector,
                           FusionRuleId rule_id,
                           int32_t anchor_local_priority,
                           size_t registration_order);
  static void RecordSuccessEvent(FusionTraceCollector& collector,
                                 const FusionFailureRecord& record);

  friend class FailureSink;
};

}  // namespace onnxruntime::fusion_rewriter_internal

#endif  // !defined(ORT_MINIMAL_BUILD)
