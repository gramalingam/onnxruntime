#include "core/optimizer/fusion_rewriter_diagnostics.h"

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <limits>
#include <map>
#include <sstream>
#include <tuple>
#include <utility>

namespace onnxruntime {
namespace {

using fusion_rewriter_internal::FusionDiagnosticsTestAccess;

struct FailureScore {
  FusionMatchStage stage{};
  size_t pattern_nodes_matched{};
  size_t epoch{};
  size_t anchor_rank{};
  size_t tuple_ordinal{};
};

struct RulePresentationOrder {
  int32_t anchor_local_priority{};
  size_t registration_order{std::numeric_limits<size_t>::max()};
};

struct AllFailureBest {
  FailureScore score;
  size_t record_index{};
};

size_t RecordBytes(const FusionFailureRecord& record) {
  if (record.detail.size() >
      std::numeric_limits<size_t>::max() - record.target_value_name.size()) {
    return std::numeric_limits<size_t>::max();
  }

  return record.target_value_name.size() + record.detail.size();
}

bool IsBetterScore(const FailureScore& candidate, const FailureScore& current) {
  if (candidate.stage != current.stage) {
    return candidate.stage > current.stage;
  }

  if (candidate.pattern_nodes_matched != current.pattern_nodes_matched) {
    return candidate.pattern_nodes_matched > current.pattern_nodes_matched;
  }

  return std::tie(candidate.epoch, candidate.anchor_rank, candidate.tuple_ordinal) <
         std::tie(current.epoch, current.anchor_rank, current.tuple_ordinal);
}

const char* StageName(FusionMatchStage stage) {
  switch (stage) {
    case FusionMatchStage::kRootSignature:
      return "RootSignature";
    case FusionMatchStage::kStructuralNode:
      return "StructuralNode";
    case FusionMatchStage::kStructuralEdge:
      return "StructuralEdge";
    case FusionMatchStage::kValueBinding:
      return "ValueBinding";
    case FusionMatchStage::kAttributeBinding:
      return "AttributeBinding";
    case FusionMatchStage::kLiteral:
      return "Literal";
    case FusionMatchStage::kCondition:
      return "Condition";
    case FusionMatchStage::kClosure:
      return "Closure";
    case FusionMatchStage::kConvexity:
      return "Convexity";
    case FusionMatchStage::kFinalValidation:
      return "FinalValidation";
    case FusionMatchStage::kPrevalidation:
      return "Prevalidation";
    case FusionMatchStage::kSuccess:
      return "Success";
  }

  return "UnknownStage";
}

const char* FailureCodeName(FusionFailureCode code) {
  switch (code) {
    case FusionFailureCode::kOpMismatch:
      return "OpMismatch";
    case FusionFailureCode::kOutputSlotMismatch:
      return "OutputSlotMismatch";
    case FusionFailureCode::kRepeatedBindingMismatch:
      return "RepeatedBindingMismatch";
    case FusionFailureCode::kMissingEffectiveAttribute:
      return "MissingEffectiveAttribute";
    case FusionFailureCode::kAttributeValueMismatch:
      return "AttributeValueMismatch";
    case FusionFailureCode::kLiteralMismatch:
      return "LiteralMismatch";
    case FusionFailureCode::kUnknownRank:
      return "UnknownRank";
    case FusionFailureCode::kDimensionMismatch:
      return "DimensionMismatch";
    case FusionFailureCode::kConstraintFalse:
      return "ConstraintFalse";
    case FusionFailureCode::kCallbackRejected:
      return "CallbackRejected";
    case FusionFailureCode::kExternalPrivateUse:
      return "ExternalPrivateUse";
    case FusionFailureCode::kNonConvex:
      return "NonConvex";
    case FusionFailureCode::kStalePlan:
      return "StalePlan";
  }

  return "UnknownFailure";
}

void FormatRecord(std::ostringstream& output, const FusionFailureRecord& record) {
  output << "rule=" << record.rule_id
         << " stage=" << StageName(record.stage);
  if (record.stage != FusionMatchStage::kSuccess) {
    output << " code=" << FailureCodeName(record.code);
  }

  output << " anchor=" << record.anchor_node << ':' << record.anchor_output_slot
         << " matched_nodes=" << record.pattern_nodes_matched;
  if (record.pattern_node) {
    output << " pattern_node=" << *record.pattern_node;
  }
  if (record.pattern_value) {
    output << " pattern_value=" << *record.pattern_value;
  }
  if (record.constraint) {
    output << " constraint=" << *record.constraint;
  }
  if (record.target_node) {
    output << " target_node=" << *record.target_node;
  }
  if (record.target_slot) {
    output << " target_slot=" << *record.target_slot;
  }
  if (!record.target_value_name.empty()) {
    output << " target_value=\"" << record.target_value_name << '"';
  }
  if (!record.detail.empty()) {
    output << " detail=\"" << record.detail << '"';
  }
}

}  // namespace

struct FusionTraceCollector::Impl {
  FusionDiagnosticMode mode{FusionDiagnosticMode::kOff};
  size_t max_records{};
  size_t max_bytes{};
  size_t retained_records{};
  size_t retained_bytes{};
  bool truncated{};

  std::vector<FusionFailureRecord> best_failures;
  std::map<FusionRuleId, FailureScore> best_scores;
  std::vector<FusionFailureRecord> records;
  std::map<FusionRuleId, AllFailureBest> all_failure_best;
  std::map<FusionRuleId, size_t> success_counts;
  std::map<FusionRuleId, RulePresentationOrder> presentation_order;
  mutable std::vector<FusionFailureRecord> all_failure_best_cache;
  mutable bool all_failure_best_cache_dirty{true};

  RulePresentationOrder PresentationOrder(FusionRuleId rule_id) const {
    const auto it = presentation_order.find(rule_id);
    return it == presentation_order.end()
               ? RulePresentationOrder{0, std::numeric_limits<size_t>::max()}
               : it->second;
  }

  bool RuleLess(FusionRuleId lhs, FusionRuleId rhs) const {
    const auto lhs_order = PresentationOrder(lhs);
    const auto rhs_order = PresentationOrder(rhs);
    return std::tie(lhs_order.anchor_local_priority,
                    lhs_order.registration_order, lhs) <
           std::tie(rhs_order.anchor_local_priority,
                    rhs_order.registration_order, rhs);
  }

  bool CanRetain(size_t bytes) const {
    if (retained_records >= max_records) {
      return false;
    }

    return bytes <= max_bytes && retained_bytes <= max_bytes - bytes;
  }

  bool CanReplace(size_t old_bytes, size_t new_bytes) const {
    const size_t bytes_without_old =
        old_bytes <= retained_bytes ? retained_bytes - old_bytes : 0;
    return new_bytes <= max_bytes && bytes_without_old <= max_bytes - new_bytes;
  }

  void SortBestFailures() {
    std::sort(best_failures.begin(), best_failures.end(),
              [this](const FusionFailureRecord& lhs,
                     const FusionFailureRecord& rhs) {
                return RuleLess(lhs.rule_id, rhs.rule_id);
              });
  }

  void RemoveBestFailure(FusionRuleId rule_id) {
    const auto record_it = std::find_if(
        best_failures.begin(), best_failures.end(),
        [rule_id](const FusionFailureRecord& record) {
          return record.rule_id == rule_id;
        });
    if (record_it == best_failures.end()) {
      return;
    }

    retained_bytes -= RecordBytes(*record_it);
    --retained_records;
    best_failures.erase(record_it);
    best_scores.erase(rule_id);
  }

  void RecordBestFailure(const FusionFailureRecord& record,
                         const FailureScore& score) {
    if (success_counts.find(record.rule_id) != success_counts.end()) {
      return;
    }

    const auto score_it = best_scores.find(record.rule_id);
    if (score_it == best_scores.end()) {
      const size_t bytes = RecordBytes(record);
      if (!CanRetain(bytes)) {
        truncated = true;
        return;
      }

      best_failures.push_back(record);
      best_scores.emplace(record.rule_id, score);
      ++retained_records;
      retained_bytes += bytes;
      SortBestFailures();
      return;
    }

    if (!IsBetterScore(score, score_it->second)) {
      return;
    }

    const auto record_it = std::find_if(
        best_failures.begin(), best_failures.end(),
        [&record](const FusionFailureRecord& current) {
          return current.rule_id == record.rule_id;
        });
    ORT_ENFORCE(record_it != best_failures.end());

    const size_t old_bytes = RecordBytes(*record_it);
    const size_t new_bytes = RecordBytes(record);
    if (!CanReplace(old_bytes, new_bytes)) {
      truncated = true;
      return;
    }

    retained_bytes = retained_bytes - old_bytes + new_bytes;
    *record_it = record;
    score_it->second = score;
  }

  std::optional<size_t> AppendRecord(const FusionFailureRecord& record) {
    const size_t bytes = RecordBytes(record);
    if (!CanRetain(bytes)) {
      truncated = true;
      return std::nullopt;
    }

    records.push_back(record);
    ++retained_records;
    retained_bytes += bytes;
    return records.size() - 1;
  }

  gsl::span<const FusionFailureRecord> BestFailures() const {
    if (mode != FusionDiagnosticMode::kAllFailures) {
      return best_failures;
    }

    if (all_failure_best_cache_dirty) {
      all_failure_best_cache.clear();
      all_failure_best_cache.reserve(all_failure_best.size());
      for (const auto& [rule_id, best] : all_failure_best) {
        ORT_ENFORCE(best.record_index < records.size());
        all_failure_best_cache.push_back(records[best.record_index]);
      }
      std::sort(all_failure_best_cache.begin(), all_failure_best_cache.end(),
                [this](const FusionFailureRecord& lhs,
                       const FusionFailureRecord& rhs) {
                  return RuleLess(lhs.rule_id, rhs.rule_id);
                });
      all_failure_best_cache_dirty = false;
    }

    return all_failure_best_cache;
  }
};

FusionTraceCollector::FusionTraceCollector() : impl_(std::make_unique<Impl>()) {}
FusionTraceCollector::~FusionTraceCollector() = default;

gsl::span<const FusionFailureRecord> FusionTraceCollector::BestFailures() const {
  return impl_->BestFailures();
}

gsl::span<const FusionFailureRecord> FusionTraceCollector::Records() const {
  return impl_->records;
}

size_t FusionTraceCollector::SuccessCount(FusionRuleId rule_id) const {
  const auto it = impl_->success_counts.find(rule_id);
  return it == impl_->success_counts.end() ? 0 : it->second;
}

bool FusionTraceCollector::Truncated() const {
  return impl_->truncated;
}

std::string FusionTraceCollector::Format() const {
  std::ostringstream output;
  bool needs_separator = false;

  const auto best_failures = BestFailures();
  if (!best_failures.empty()) {
    output << "Best failures:";
    for (const auto& record : best_failures) {
      output << '\n';
      FormatRecord(output, record);
    }
    needs_separator = true;
  }

  if (!impl_->records.empty()) {
    if (needs_separator) {
      output << '\n';
    }
    output << "Records:";
    for (const auto& record : impl_->records) {
      output << '\n';
      FormatRecord(output, record);
    }
    needs_separator = true;
  }

  if (!impl_->success_counts.empty()) {
    if (needs_separator) {
      output << '\n';
    }
    output << "Success counts:";
    for (const auto& [rule_id, count] : impl_->success_counts) {
      output << "\nrule=" << rule_id << " count=" << count;
    }
    needs_separator = true;
  }

  if (impl_->truncated) {
    if (needs_separator) {
      output << '\n';
    }
    output << "truncated=true";
  }

  return output.str();
}

void FusionTraceCollector::Clear() {
  *impl_ = Impl{};
}

namespace fusion_rewriter_internal {

FailureSink::FailureSink(FusionTraceCollector& collector,
                         FusionDiagnosticMode mode)
    : collector_(collector), mode_(mode) {
  ORT_ENFORCE(mode != FusionDiagnosticMode::kOff);
}

FusionDiagnosticMode FailureSink::Mode() const noexcept {
  return mode_;
}

void FailureSink::RegisterRule(FusionRuleId rule_id,
                               int32_t anchor_local_priority,
                               size_t registration_order) {
  FusionDiagnosticsAccess::RegisterRule(
      collector_, rule_id, anchor_local_priority, registration_order);
}

void FailureSink::RecordFailure(const FusionFailureRecord& record,
                                size_t epoch, size_t anchor_rank,
                                size_t tuple_ordinal) {
  FusionDiagnosticsAccess::RecordFailure(
      collector_, record, epoch, anchor_rank, tuple_ordinal);
}

void FailureSink::RecordSuccess(FusionRuleId rule_id) {
  FusionDiagnosticsAccess::RecordSuccess(collector_, rule_id);
}

void FailureSink::RecordSuccess(const FusionFailureRecord& record) {
  FusionDiagnosticsAccess::RecordSuccessEvent(collector_, record);
}

MatchAttempt::MatchAttempt(FailureSink& failure_sink,
                           FusionRuleId rule_id,
                           NodeIndex anchor_node,
                           size_t anchor_output_slot,
                           size_t epoch,
                           size_t anchor_rank,
                           size_t tuple_ordinal)
    : failure_sink_(failure_sink),
      rule_id_(rule_id),
      anchor_node_(anchor_node),
      anchor_output_slot_(anchor_output_slot),
      epoch_(epoch),
      anchor_rank_(anchor_rank),
      tuple_ordinal_(tuple_ordinal) {}

void MatchAttempt::SetPatternNodesMatched(
    size_t pattern_nodes_matched) noexcept {
  pattern_nodes_matched_ = pattern_nodes_matched;
}

FusionFailureRecord MatchAttempt::MakeFailure(
    FusionMatchStage stage, FusionFailureCode code) const {
  FusionFailureRecord record;
  record.rule_id = rule_id_;
  record.stage = stage;
  record.code = code;
  record.anchor_node = anchor_node_;
  record.anchor_output_slot = anchor_output_slot_;
  record.pattern_nodes_matched = pattern_nodes_matched_;
  return record;
}

void MatchAttempt::RecordFailure(FusionFailureRecord record) const {
  record.rule_id = rule_id_;
  record.anchor_node = anchor_node_;
  record.anchor_output_slot = anchor_output_slot_;
  record.pattern_nodes_matched = pattern_nodes_matched_;
  failure_sink_.RecordFailure(record, epoch_, anchor_rank_, tuple_ordinal_);
}

FusionFailureRecord MatchAttempt::MakeSuccess() const {
  FusionFailureRecord record;
  record.rule_id = rule_id_;
  record.stage = FusionMatchStage::kSuccess;
  record.anchor_node = anchor_node_;
  record.anchor_output_slot = anchor_output_slot_;
  record.pattern_nodes_matched = pattern_nodes_matched_;
  return record;
}

common::Status CreateFailureSink(
    FusionTraceCollector* collector,
    FusionDiagnosticMode mode,
    size_t max_records,
    size_t max_bytes,
    std::unique_ptr<FailureSink>& failure_sink) {
  failure_sink.reset();

  if (mode == FusionDiagnosticMode::kOff) {
    if (collector != nullptr) {
      FusionDiagnosticsAccess::Configure(
          *collector, mode, max_records, max_bytes);
    }
    return common::Status::OK();
  }

  if (collector == nullptr) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "FusionTraceCollector must be non-null when fusion diagnostics are enabled.");
  }

  FusionDiagnosticsAccess::Configure(
      *collector, mode, max_records, max_bytes);
  failure_sink = std::make_unique<FailureSink>(*collector, mode);
  return common::Status::OK();
}

void FusionDiagnosticsAccess::Configure(
    FusionTraceCollector& collector,
    FusionDiagnosticMode mode,
    size_t max_records,
    size_t max_bytes) {
  *collector.impl_ = FusionTraceCollector::Impl{};
  collector.impl_->mode = mode;
  collector.impl_->max_records = max_records;
  collector.impl_->max_bytes = max_bytes;
}

void FusionDiagnosticsAccess::RecordFailure(
    FusionTraceCollector& collector,
    const FusionFailureRecord& record,
    size_t epoch,
    size_t anchor_rank,
    size_t tuple_ordinal) {
  auto& impl = *collector.impl_;
  if (impl.mode == FusionDiagnosticMode::kOff ||
      record.stage == FusionMatchStage::kSuccess) {
    return;
  }

  const FailureScore score{
      record.stage, record.pattern_nodes_matched,
      epoch, anchor_rank, tuple_ordinal};

  if (impl.mode == FusionDiagnosticMode::kAllFailures) {
    const auto record_index = impl.AppendRecord(record);
    if (!record_index) {
      return;
    }

    const auto best_it = impl.all_failure_best.find(record.rule_id);
    if (best_it == impl.all_failure_best.end() ||
        IsBetterScore(score, best_it->second.score)) {
      impl.all_failure_best[record.rule_id] =
          AllFailureBest{score, *record_index};
      impl.all_failure_best_cache_dirty = true;
    }
    return;
  }

  impl.RecordBestFailure(record, score);
}

void FusionDiagnosticsAccess::RecordSuccess(
    FusionTraceCollector& collector,
    FusionRuleId rule_id) {
  FusionFailureRecord record;
  record.rule_id = rule_id;
  record.stage = FusionMatchStage::kSuccess;
  RecordSuccessEvent(collector, record);
}

void FusionDiagnosticsAccess::RegisterRule(
    FusionTraceCollector& collector,
    FusionRuleId rule_id,
    int32_t anchor_local_priority,
    size_t registration_order) {
  auto& impl = *collector.impl_;
  if (impl.mode == FusionDiagnosticMode::kOff) {
    return;
  }

  impl.presentation_order[rule_id] =
      RulePresentationOrder{anchor_local_priority, registration_order};
  impl.SortBestFailures();
  impl.all_failure_best_cache_dirty = true;
}

void FusionDiagnosticsAccess::RecordSuccessEvent(
    FusionTraceCollector& collector,
    const FusionFailureRecord& input_record) {
  auto& impl = *collector.impl_;
  if (impl.mode == FusionDiagnosticMode::kOff) {
    return;
  }

  FusionFailureRecord record = input_record;
  record.stage = FusionMatchStage::kSuccess;
  ++impl.success_counts[record.rule_id];

  if (impl.mode == FusionDiagnosticMode::kBestFailure ||
      impl.mode == FusionDiagnosticMode::kDryRun) {
    impl.RemoveBestFailure(record.rule_id);
  }

  if (impl.mode == FusionDiagnosticMode::kAllFailures ||
      impl.mode == FusionDiagnosticMode::kDryRun) {
    static_cast<void>(impl.AppendRecord(record));
  }
}

void FusionDiagnosticsTestAccess::Configure(
    FusionTraceCollector& collector,
    FusionDiagnosticMode mode,
    size_t max_records,
    size_t max_bytes) {
  FusionDiagnosticsAccess::Configure(
      collector, mode, max_records, max_bytes);
}

void FusionDiagnosticsTestAccess::RecordFailure(
    FusionTraceCollector& collector,
    const FusionFailureRecord& record,
    size_t epoch,
    size_t anchor_rank,
    size_t tuple_ordinal) {
  FusionDiagnosticsAccess::RecordFailure(
      collector, record, epoch, anchor_rank, tuple_ordinal);
}

void FusionDiagnosticsTestAccess::RecordSuccess(
    FusionTraceCollector& collector,
    FusionRuleId rule_id) {
  FusionDiagnosticsAccess::RecordSuccess(collector, rule_id);
}

}  // namespace fusion_rewriter_internal
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
