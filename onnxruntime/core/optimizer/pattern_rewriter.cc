// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/optimizer/pattern_rewriter.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <iterator>
#include <limits>
#include <sstream>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "onnx/checker.h"

namespace onnxruntime {
namespace rewrite {
namespace {

std::string Key(const std::string& op, const std::string& domain) { return domain + '\0' + op; }

std::string Signature(const Pattern& pattern) {
  std::ostringstream stream;
  stream << pattern.ValueCount() << ':';
  stream.precision(std::numeric_limits<float>::max_digits10);
  for (int id = 0; id < static_cast<int>(pattern.ValueCount()); ++id) {
    auto it = pattern.Constants().find(id);
    if (it == pattern.Constants().end()) continue;
    stream << 'c' << id << ':' << it->second.index() << ':';
    std::visit([&](auto value) { stream << value; }, it->second);
    stream << ';';
  }
  for (const auto& node : pattern.Nodes()) {
    stream << node.op.domain.size() << ':' << node.op.domain << node.op.name.size() << ':' << node.op.name;
    stream << '/';
    for (Value output : node.outputs) stream << output << ',';
    stream << '/';
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
  if (root.OpType() != root_op.name || root.Domain() != root_op.domain) {
    return false;
  }
  match.values.assign(pattern.ValueCount(), nullptr);
  match.nodes.assign(pattern.Nodes().size(), nullptr);
  match.nodes.back() = &root;
  InlinedHashSet<NodeIndex> distinct;
  for (size_t i = pattern.Nodes().size(); i-- > 0;) {
    const auto& expected = pattern.Nodes()[i];
    const Node* node = match.nodes[i];
    if (!node) {
      for (Value output : expected.outputs) {
        if (!match.values[output]) continue;
        auto it = producers.find(match.values[output]);
        if (it != producers.end()) node = graph.GetNode(it->second);
        break;
      }
    }
    if (!node || !distinct.insert(node->Index()).second || node->OpType() != expected.op.name ||
        node->Domain() != expected.op.domain || node->InputDefs().size() != expected.inputs.size() ||
        node->OutputDefs().size() != expected.outputs.size() || !node->GetAttributeNameToSubgraphMap().empty() ||
        !node->ImplicitInputDefs().empty() || node->GetLayeringAnnotation() != root.GetLayeringAnnotation()) {
      return false;
    }
    match.nodes[i] = node;
    for (const auto& attribute : expected.attributes) {
      auto found = node->GetAttributes().find(attribute.name);
      if (found == node->GetAttributes().end() || !attribute.Matches(found->second)) return false;
    }
    for (size_t j = 0; j < expected.outputs.size(); ++j) {
      const NodeArg* output = node->OutputDefs()[j];
      const NodeArg*& binding = match.values[expected.outputs[j]];
      if (!output->Exists() || (binding && binding != output)) return false;
      binding = output;
    }
    for (size_t j = 0; j < expected.inputs.size(); ++j) {
      const NodeArg* actual = node->InputDefs()[j];
      const NodeArg*& binding = match.values[expected.inputs[j]];
      if (!actual->Exists() || (binding && binding != actual)) return false;
      binding = actual;
    }
  }
  // Input variables must be outside the deletion set, even when aliasing another binding.
  for (int value = 0; value < static_cast<int>(pattern.ValueCount()); ++value) {
    if (!pattern.IsInput(value)) continue;
    auto it = producers.find(match.values[value]);
    if (it != producers.end() && distinct.count(it->second)) return false;
  }
  for (const auto& [handle, expected] : pattern.Constants()) {
    const auto* tensor = graph_utils::GetConstantInitializer(graph, match.values[handle]->Name());
    if (!tensor || tensor->dims_size() != 0 ||
        tensor->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL)
      return false;
    const bool floating = std::holds_alternative<float>(expected);
    if (tensor->data_type() !=
        (floating ? ONNX_NAMESPACE::TensorProto_DataType_FLOAT : ONNX_NAMESPACE::TensorProto_DataType_INT64))
      return false;
    Initializer initializer(graph, *tensor, graph.ModelPath());
    if (initializer.size() != 1) return false;
    if (floating ? *initializer.data<float>() != std::get<float>(expected)
                 : *initializer.data<int64_t>() != std::get<int64_t>(expected))
      return false;
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

bool CanRewritePartition(const Match& match, const Options& options) {
  const auto& provider = match.nodes.back()->GetExecutionProviderType();
  if (!provider.empty() && !options.compatible_execution_providers.count(provider)) return false;
  return std::all_of(match.nodes.begin(), match.nodes.end(),
                     [&](const Node* node) { return node->GetExecutionProviderType() == provider; });
}

Status Replace(Graph& graph, const Pattern& pattern, const Match& match, const Replacement& recipe,
               Producers& producers, NodeIndex& added) {
  const auto& domain = recipe.domain == kOnnxDomainAlias ? kOnnxDomain : recipe.domain;
  auto imported = graph.DomainToVersionMap().find(domain);
  ORT_RETURN_IF(imported == graph.DomainToVersionMap().end(), "Replacement domain is not imported: ", domain);
  InlinedVector<NodeArg*> inputs;
  for (Value value : recipe.inputs) {
    ORT_RETURN_IF(
        value.handle < 0 || static_cast<size_t>(value.handle) >= match.values.size() || !pattern.IsInput(value),
        "Replacement inputs must be pattern input variables.");
    inputs.push_back(graph.GetNodeArg(match.values[value]->Name()));
  }
  const Node& root = *match.nodes.back();
  InlinedVector<NodeArg*> outputs;
  for (const NodeArg* output : root.OutputDefs()) outputs.push_back(graph.GetNodeArg(output->Name()));
  Node& replacement = graph.AddNode(graph.GenerateNodeName("rewrite"), recipe.op, "PatternRewriter", inputs, outputs,
                                    root, &recipe.attributes, domain);
  // Validate the imported schema before deleting anything.
  if (!graph.SetOpSchemaFromRegistryForNode(replacement)) {
    graph.RemoveNode(replacement.Index());
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Replacement schema is unavailable: ", recipe.op);
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
    for (const NodeArg* output : node->OutputDefs()) producers.erase(output);
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
  for (NodeArg* output : outputs) {
    producers[output] = added;
    graph.UpdateProducerNode(output->Name(), added);
  }
  graph.NotifyNodeReplacement(removed, added);
  return Status::OK();
}
}  // namespace

AttributeMatcher AttributeMatcher::Equal(std::string name, int64_t value) {
  return {std::move(name), [value](const auto& attribute) {
            return attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INT && attribute.i() == value;
          }};
}

AttributeMatcher AttributeMatcher::Equal(std::string name, float value) {
  ORT_ENFORCE(std::isfinite(value), "Attribute float constants must be finite.");
  return {std::move(name), [value](const auto& attribute) {
            return attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT && attribute.f() == value;
          }};
}

AttributeMatcher AttributeMatcher::Equal(std::string name, std::string value) {
  return {std::move(name), [value = std::move(value)](const auto& attribute) {
            return attribute.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_STRING && attribute.s() == value;
          }};
}

Value Pattern::AddValue(std::string_view name) {
  ORT_ENFORCE(value_count_ < 256, "Patterns are limited to 256 values.");
  if (!name.empty()) {
    ORT_ENFORCE(std::none_of(values_.begin(), values_.end(), [&](const auto& value) { return value.name == name; }),
                "Duplicate pattern value name: ", name);
  }
  Value value{std::string(name), static_cast<int>(value_count_++)};
  values_.push_back(value);
  return value;
}

Value Pattern::Input(std::string_view name) { return AddValue(name); }

Value Pattern::Constant(float constant) {
  ORT_ENFORCE(std::isfinite(constant), "Pattern float constants must be finite.");
  auto value = AddValue("");
  constants_.emplace(value.handle, constant);
  return value;
}

Value Pattern::Constant(int64_t constant) {
  auto value = AddValue("");
  constants_.emplace(value.handle, constant);
  return value;
}

Value Pattern::NamedValue(std::string_view name) const {
  auto it = std::find_if(values_.begin(), values_.end(), [&](const auto& value) { return value.name == name; });
  ORT_ENFORCE(!name.empty() && it != values_.end(), "Unknown pattern value: ", name);
  return *it;
}

void Pattern::Name(Value value, std::string_view name) {
  ORT_ENFORCE(value.handle >= 0 && static_cast<size_t>(value.handle) < value_count_ && !name.empty(),
              "Invalid pattern value or empty name.");
  ORT_ENFORCE(std::none_of(values_.begin(), values_.end(),
                           [&](const auto& other) { return other.handle != value.handle && other.name == name; }),
              "Duplicate pattern value name: ", name);
  values_[value.handle].name = name;
  for (auto& node : nodes_) {
    for (auto& output : node.outputs)
      if (output.handle == value.handle) output.name = name;
    for (auto& input : node.inputs)
      if (input.handle == value.handle) input.name = name;
  }
}

void Pattern::Attributes(std::initializer_list<AttributeMatcher> matchers) {
  Attributes(InlinedVector<AttributeMatcher, 1>(matchers));
}

void Pattern::Attributes(InlinedVector<AttributeMatcher, 1> matchers) {
  ORT_ENFORCE(!nodes_.empty(), "Attribute matchers require a pattern node.");
  InlinedHashSet<std::string> names;
  for (const auto& attr : nodes_.back().attributes) names.insert(attr.name);
  for (const auto& attr : matchers) {
    ORT_ENFORCE(!attr.name.empty() && attr.predicate && names.insert(attr.name).second,
                "Invalid or duplicate attribute matcher: ", attr.name);
  }
  auto& target = nodes_.back().attributes;
  target.insert(target.end(), std::make_move_iterator(matchers.begin()), std::make_move_iterator(matchers.end()));
}

Value Pattern::Op(Operator op, std::initializer_list<Value> inputs) { return MultiOp(std::move(op), inputs, 1)[0]; }

InlinedVector<Value> Pattern::MultiOp(Operator op, std::initializer_list<Value> inputs, size_t output_count) {
  return MultiOp(std::move(op), gsl::span<const Value>(inputs.begin(), inputs.size()), output_count);
}

InlinedVector<Value> Pattern::MultiOp(Operator op, gsl::span<const Value> inputs, size_t output_count) {
  ORT_ENFORCE(!op.name.empty(), "Pattern operators need a name.");
  ORT_ENFORCE(nodes_.size() < 64, "Patterns are limited to 64 nodes.");
  ORT_ENFORCE(output_count > 0 && output_count <= 64, "Pattern operators need between 1 and 64 outputs.");
  ORT_ENFORCE(value_count_ <= 256 - output_count && inputs.size() <= 64,
              "Patterns are limited to 256 values and 64 inputs per operator.");
  if (op.domain == kOnnxDomainAlias) op.domain = kOnnxDomain;
  for (Value input : inputs)
    ORT_ENFORCE(input.handle >= 0 && static_cast<size_t>(input.handle) < value_count_,
                "Pattern inputs must already be defined.");
  InlinedVector<Value> outputs;
  for (size_t i = 0; i < output_count; ++i) outputs.push_back(AddValue(""));
  InlinedVector<Value> canonical_inputs;
  for (Value input : inputs) canonical_inputs.push_back(values_[input.handle]);
  nodes_.push_back({std::move(op), std::move(canonical_inputs), outputs, {}});
  return outputs;
}

bool Pattern::IsInput(Value value) const {
  if (value.handle < 0 || static_cast<size_t>(value.handle) >= value_count_) return false;
  return std::none_of(nodes_.begin(), nodes_.end(), [value](const auto& node) {
    return std::find(node.outputs.begin(), node.outputs.end(), value) != node.outputs.end();
  });
}

namespace {
class PatternParser {
 public:
  explicit PatternParser(std::string_view text) : text_(text) {}

  Status Parse(Pattern& destination) {
    ORT_RETURN_IF_ERROR(Next());
    ORT_RETURN_IF_ERROR(Expect("("));
    if (token_ != ")") {
      do {
        std::string name;
        ORT_RETURN_IF_ERROR(Identifier(name));
        ORT_RETURN_IF(symbols_.count(name), Error("duplicate input name"));
        ORT_RETURN_IF(pattern_.ValueCount() >= 256, Error("too many values"));
        symbols_.emplace(name, pattern_.Input(name));
        if (token_ != ",") break;
        ORT_RETURN_IF_ERROR(Next());
      } while (true);
    }
    ORT_RETURN_IF_ERROR(Expect(")"));
    ORT_RETURN_IF_ERROR(Expect("=>"));
    bool returned = false;
    while (kind_ != Kind::End) {
      if (kind_ == Kind::Symbol && token_ == ";") {
        ORT_RETURN_IF_ERROR(Next());
        continue;
      }
      InlinedVector<std::string> names;
      if (kind_ == Kind::Identifier && token_ == "return") {
        returned = true;
        ORT_RETURN_IF_ERROR(Next());
      } else {
        do {
          std::string name;
          ORT_RETURN_IF_ERROR(Identifier(name));
          ORT_RETURN_IF(symbols_.count(name) || std::find(names.begin(), names.end(), name) != names.end(),
                        Error("duplicate assignment name"));
          names.push_back(std::move(name));
          if (token_ != ",") break;
          ORT_RETURN_IF_ERROR(Next());
        } while (true);
        ORT_RETURN_IF_ERROR(Expect("="));
      }
      ORT_RETURN_IF_ERROR(Call(names));
      if (returned) {
        if (kind_ == Kind::Symbol && token_ == ";") ORT_RETURN_IF_ERROR(Next());
        ORT_RETURN_IF(kind_ != Kind::End, Error("unexpected text after return"));
        break;
      }
    }
    ORT_RETURN_IF(!returned, Error("missing return"));
    InlinedHashSet<int> reachable;
    for (Value output : pattern_.Nodes().back().outputs) reachable.insert(output.handle);
    for (auto it = pattern_.Nodes().rbegin(); it != pattern_.Nodes().rend(); ++it) {
      ORT_RETURN_IF(std::none_of(it->outputs.begin(), it->outputs.end(),
                                 [&](Value output) { return reachable.count(output.handle); }),
                    Error("disconnected pattern node"));
      for (Value output : it->outputs) reachable.insert(output.handle);
      for (Value input : it->inputs) reachable.insert(input.handle);
    }
    ORT_RETURN_IF(reachable.size() != pattern_.ValueCount(), Error("unused input"));
    destination = std::move(pattern_);
    return Status::OK();
  }

 private:
  Status Error(const char* message) const {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Pattern parse error at byte ", token_start_, ": ", message);
  }

  static bool Alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
  static bool Digit(char c) { return c >= '0' && c <= '9'; }

  Status Next() {
    while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_;
    token_start_ = position_;
    token_.clear();
    kind_ = Kind::End;
    if (position_ == text_.size()) return Status::OK();
    kind_ = Kind::Symbol;
    char c = text_[position_++];
    if (Alpha(c)) {
      kind_ = Kind::Identifier;
      while (position_ < text_.size() &&
             (Alpha(text_[position_]) || Digit(text_[position_]) || text_[position_] == '.'))
        ++position_;
    } else if (Digit(c) || c == '-' || c == '+' || c == '.') {
      kind_ = Kind::Number;
      while (position_ < text_.size()) {
        char d = text_[position_];
        if (!Digit(d) && d != '.' && d != 'e' && d != 'E' && d != '+' && d != '-') break;
        ++position_;
      }
    } else if (c == '\'' || c == '"') {
      kind_ = Kind::String;
      bool closed = false;
      while (position_ < text_.size()) {
        char d = text_[position_++];
        if (d == c) {
          closed = true;
          break;
        }
        if (d == '\\') {
          ORT_RETURN_IF(position_ == text_.size(), Error("unterminated string escape"));
          d = text_[position_++];
          ORT_RETURN_IF(d != '\\' && d != '\'' && d != '"', Error("unsupported string escape"));
        }
        token_.push_back(d);
      }
      ORT_RETURN_IF(!closed, Error("unterminated string"));
      return Status::OK();
    } else if ((c == '=' && position_ < text_.size() && text_[position_] == '>') ||
               (c == ':' && position_ < text_.size() && text_[position_] == ':')) {
      ++position_;
    } else {
      ORT_RETURN_IF(std::string_view("(),=;").find(c) == std::string_view::npos, Error("unexpected character"));
    }
    token_ = text_.substr(token_start_, position_ - token_start_);
    return Status::OK();
  }

  Status Expect(std::string_view token) {
    ORT_RETURN_IF(kind_ != Kind::Symbol || token_ != token, Error("unexpected token"));
    return Next();
  }
  Status Identifier(std::string& name) {
    ORT_RETURN_IF(kind_ != Kind::Identifier || token_ == "return" || token_.find('.') != std::string::npos,
                  Error("expected value identifier"));
    name = token_;
    return Next();
  }

  Status Number(std::variant<float, int64_t>& result) {
    ORT_RETURN_IF(kind_ != Kind::Number, Error("expected numeric literal"));
    std::string_view number = token_;
    if (!number.empty() && number.front() == '+') number.remove_prefix(1);
    if (number.find_first_of(".eE") != std::string_view::npos) {
      float value;
      auto parsed = std::from_chars(number.data(), number.data() + number.size(), value);
      ORT_RETURN_IF(parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() || !std::isfinite(value),
                    Error("invalid or nonfinite float literal"));
      result = value;
    } else {
      int64_t value;
      auto parsed = std::from_chars(number.data(), number.data() + number.size(), value);
      ORT_RETURN_IF(parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size(),
                    Error("invalid integer literal"));
      result = value;
    }
    return Next();
  }

  Status Call(const InlinedVector<std::string>& names) {
    ORT_RETURN_IF(kind_ != Kind::Identifier || token_ == "return", Error("expected operator name"));
    Operator op{token_, ""};
    ORT_RETURN_IF_ERROR(Next());
    if (token_ == "::") {
      op.domain = std::move(op.name);
      ORT_RETURN_IF_ERROR(Next());
      ORT_RETURN_IF(kind_ != Kind::Identifier, Error("expected qualified operator name"));
      op.name = token_;
      ORT_RETURN_IF_ERROR(Next());
    }
    ORT_RETURN_IF_ERROR(Expect("("));
    InlinedVector<Value> inputs;
    InlinedVector<AttributeMatcher, 1> attributes;
    InlinedHashSet<std::string> attribute_names;
    bool keyword = false;
    if (token_ != ")") {
      do {
        if (kind_ == Kind::Identifier) {
          std::string name = token_;
          ORT_RETURN_IF_ERROR(Next());
          if (token_ == "=") {
            keyword = true;
            ORT_RETURN_IF(!attribute_names.insert(name).second, Error("duplicate attribute"));
            ORT_RETURN_IF_ERROR(Next());
            if (kind_ == Kind::String) {
              attributes.push_back(AttributeMatcher::Equal(name, token_));
              ORT_RETURN_IF_ERROR(Next());
            } else {
              std::variant<float, int64_t> number;
              ORT_RETURN_IF_ERROR(Number(number));
              attributes.push_back(
                  std::visit([&](auto value) { return AttributeMatcher::Equal(name, value); }, number));
            }
          } else {
            ORT_RETURN_IF(keyword, Error("positional input after attribute"));
            auto found = symbols_.find(name);
            ORT_RETURN_IF(found == symbols_.end(), Error("undefined input value"));
            inputs.push_back(found->second);
          }
        } else {
          ORT_RETURN_IF(keyword, Error("positional input after attribute"));
          std::variant<float, int64_t> number;
          ORT_RETURN_IF_ERROR(Number(number));
          ORT_RETURN_IF(pattern_.ValueCount() >= 256, Error("too many values"));
          inputs.push_back(std::visit([&](auto value) { return pattern_.Constant(value); }, number));
        }
        ORT_RETURN_IF(inputs.size() > 64, Error("too many operator inputs"));
        if (token_ != ",") break;
        ORT_RETURN_IF_ERROR(Next());
      } while (true);
    }
    ORT_RETURN_IF_ERROR(Expect(")"));
    size_t outputs = names.empty() ? 1 : names.size();
    ORT_RETURN_IF(outputs > 64 || pattern_.Nodes().size() >= 64 || pattern_.ValueCount() + outputs > 256,
                  Error("pattern limits exceeded"));
    auto values = pattern_.MultiOp(std::move(op), gsl::span<const Value>(inputs), outputs);
    pattern_.Attributes(std::move(attributes));
    for (size_t i = 0; i < names.size(); ++i) {
      pattern_.Name(values[i], names[i]);
      symbols_.emplace(names[i], pattern_.NamedValue(names[i]));
    }
    return Status::OK();
  }

  enum class Kind {
    End,
    Symbol,
    Identifier,
    Number,
    String
  };
  std::string_view text_;
  size_t position_ = 0;
  size_t token_start_ = 0;
  Kind kind_ = Kind::Symbol;
  std::string token_;
  Pattern pattern_;
  InlinedHashMap<std::string, Value> symbols_;
};
}  // namespace

Status Pattern::Parse(std::string_view text, Pattern& destination) { return PatternParser(text).Parse(destination); }

PatternRewriter::PatternRewriter(Rules rules, Options options)
    : GraphTransformer("PatternRewriter"), rules_(std::move(rules)), options_(options) {
  InlinedHashMap<std::string, size_t> signatures;
  for (size_t i = 0; i < rules_.size(); ++i) {
    Rule& rule = rules_[i];
    ORT_ENFORCE(!rule.name.empty() && rule.pattern.Nodes().size() >= 2 && rule.guard && rule.replacement,
                "Rules need a name, at least two nodes, a guard and a recipe.");
    InlinedHashSet<std::string> domains;
    for (auto& requirement : rule.opsets) {
      if (requirement.domain == kOnnxDomainAlias) requirement.domain = kOnnxDomain;
      ORT_ENFORCE(requirement.min_version > 0 && requirement.max_version >= requirement.min_version,
                  "Rule opset requirements need a positive, nonempty inclusive range.");
      ORT_ENFORCE(domains.insert(requirement.domain).second, "Duplicate rule opset domain: ", requirement.domain);
    }
    for (const auto& node : rule.pattern.Nodes()) {
      ORT_ENFORCE(domains.count(node.op.domain), "Missing rule opset requirement for domain: ", node.op.domain);
    }
    // Every node must reach the last node. Disconnected definitions are not patterns.
    const auto& root_outputs = rule.pattern.Nodes().back().outputs;
    InlinedHashSet<int> reachable;
    for (Value output : root_outputs) reachable.insert(output.handle);
    for (auto it = rule.pattern.Nodes().rbegin(); it != rule.pattern.Nodes().rend(); ++it) {
      ORT_ENFORCE(
          std::any_of(it->outputs.begin(), it->outputs.end(), [&](Value output) { return reachable.count(output); }),
          "Disconnected pattern node.");
      for (Value output : it->outputs) reachable.insert(output.handle);
      for (Value input : it->inputs) reachable.insert(input.handle);
    }
    ORT_ENFORCE(reachable.size() == rule.pattern.ValueCount(), "Unused pattern input.");
    invalidation_depth_ = std::max(invalidation_depth_, rule.pattern.Nodes().size());
    if (!options_.indexed) continue;
    std::string signature = Signature(rule.pattern);
    // Opaque predicates cannot be compared safely; keep their plans distinct.
    if (std::any_of(rule.pattern.Nodes().begin(), rule.pattern.Nodes().end(),
                    [](const auto& node) { return !node.attributes.empty(); }))
      signature += "#" + std::to_string(i);
    auto [it, inserted] = signatures.emplace(signature, groups_.size());
    if (inserted) {
      const auto& root = rule.pattern.Nodes().back().op;
      roots_[Key(root.name, root.domain)].push_back(groups_.size());
      groups_.push_back({rule.pattern, {}});
    }
    groups_[it->second].rules.push_back(i);
  }
}

Status PatternRewriter::Run(Graph& graph, Statistics& statistics) const {
  statistics = {};
  ORT_RETURN_IF(graph.GraphResolveNeeded(), "PatternRewriter requires an already resolved graph.");
  // Import applicability is evaluated once per rule and graph, never per matched node.
  InlinedVector<char> applicable;
  InlinedVector<size_t> applicable_rules;
  for (size_t id = 0; id < rules_.size(); ++id) {
    bool supported = std::all_of(rules_[id].opsets.begin(), rules_[id].opsets.end(), [&](const auto& requirement) {
      auto imported = graph.DomainToVersionMap().find(requirement.domain);
      return imported != graph.DomainToVersionMap().end() && imported->second >= requirement.min_version &&
             imported->second <= requirement.max_version;
    });
    applicable.push_back(supported);
    if (supported) applicable_rules.push_back(id);
  }
  if (applicable_rules.empty()) return Status::OK();
  InlinedVector<InlinedVector<size_t>, 1> applicable_groups;
  for (const auto& group : groups_) {
    InlinedVector<size_t> ids;
    for (size_t id : group.rules) {
      if (applicable[id]) ids.push_back(id);
    }
    applicable_groups.push_back(std::move(ids));
  }
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
      if (!MatchPattern(pattern, graph, *root, producers, match) || !CanRewritePartition(match, options_)) return;
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
      auto it = roots_.find(Key(root->OpType(), root->Domain()));
      if (it != roots_.end()) {
        for (size_t id : it->second) {
          const auto& group = groups_[id];
          const auto& ids = applicable_groups[id];
          if (ids.empty() || (!options_.match_only && ids.front() >= winner)) continue;
          consider(group.pattern, ids);
        }
      }
    } else {
      for (size_t id : applicable_rules) {
        consider(rules_[id].pattern, gsl::span<const size_t>(&id, 1));
        if (!options_.match_only && winner != std::numeric_limits<size_t>::max()) break;
      }
    }
    if (options_.match_only || !winning_pattern) continue;
    ++statistics.matches;
    InlinedVector<NodeIndex> affected;
    for (int value = 0; value < static_cast<int>(winning_pattern->ValueCount()); ++value) {
      if (!winning_pattern->IsInput(value)) continue;
      auto it = producers.find(winning_match.values[value]);
      if (it != producers.end()) affected.push_back(it->second);
    }
    NodeIndex added;
    auto recipe = rules_[winner].replacement(graph, winning_match);
    const auto& domain = recipe.domain == kOnnxDomainAlias ? kOnnxDomain : recipe.domain;
    ORT_RETURN_IF(std::none_of(rules_[winner].opsets.begin(), rules_[winner].opsets.end(),
                               [&](const auto& requirement) { return requirement.domain == domain; }),
                  "Replacement domain has no rule opset requirement: ", domain);
    ORT_RETURN_IF_ERROR(Replace(graph, *winning_pattern, winning_match, recipe, producers, added));
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
