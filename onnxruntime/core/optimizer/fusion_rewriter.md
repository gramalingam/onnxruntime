# Conditional fusion-rewrite engine design

## 1. Decision summary

Generalize `FunctionExtractor` into a **conditional, whole-graph fusion engine**
without replacing its proven matcher.

An **anchor** is the target output-producing node/output slot at which a rule
attempt is seeded.

A `FusionRule` consists of:

1. a `FunctionProto`, exposed through the `PatternFunctionProto` C++ alias, whose
   body is the expansion to recognize;
2. a narrow `ReplacementCall` descriptor naming the primitive or function call
   to emit and mapping pattern boundaries/attributes to that call;
3. an optional declarative `ConstraintProgram`;
4. an optional read-only C++ match-condition callback; and
5. stable rule identity and anchor-local priority.

Pattern and replacement identities are deliberately independent. A fusion such as
an ONNX arithmetic GELU expansion to `com.microsoft::FastGelu` must not require
registering the expansion body under the already existing primitive's identity.

The engine retains FunctionExtractor's output-seeded, reverse-topological,
backtrack-free matching, exact attribute binding, closure, convexity, immutable
replacement plans, non-atomic apply contract, and resolve-between-batches
discipline.

The architectural shift is that graph traversal, mutation safety, and fixpoint
iteration belong to one `FusionRuleSet`, not to each rule. All rules share one
immutable snapshot per epoch. Matches are discovered without mutation, a
deterministic nonconflicting batch is selected, the batch is applied, the graph is
resolved, and a new epoch starts. The implementation never depends on a mutable
container iterator accidentally visiting newly inserted nodes.

This is intentionally a **fusion** engine, not a general term-rewriting system:
every successful rule removes more than one matched pattern operation node and
adds exactly one call, primitive or function. Total node count therefore strictly
decreases, giving a simple global termination proof even when rules enable one
another.

## 2. Goals and non-goals

### Goals

- Express common shape, type, symbolic-dimension, and attribute preconditions
  declaratively.
- Permit uncommon semantic conditions through a typed C++ callback.
- Evaluate all conditions against the complete concrete match, not against the
  abstract function body.
- Explain near misses with bounded, structured diagnostics that are off by
  default.
- Apply an ordered set of rules safely across a mutating ORT graph.
- Allow a newly created fused call to participate in later fusions.
- Preserve FunctionExtractor's false-negative-over-false-positive stance.
- Keep rule definitions explicit and easy for humans and AI agents to inspect:
  portable pattern, portable common constraints, isolated custom callback.

### Non-goals

- Expanding one node into several, same-size canonicalization, or oscillating
  rewrites.
- Commutative/permutation matching or general backtracking.
- Matching across graph scopes or recursively rewriting nested subgraphs in the
  first version.
- Evaluating data-dependent runtime values. Conditions use model metadata,
  constant initializers/literals, attributes, and inferred static facts only.
- Kernel selection or execution-provider partitioning inside the matcher.
- Transactional rollback after graph mutation.

## 3. Consolidated contract

### 3.1 Rule contract

Each rule has:

```text
FusionRule {
  RuleId id;                         // stable, unique, diagnostic identity
  int32_t anchor_local_priority;     // lower wins at the same consumer anchor
  owned PatternFunctionProto pattern;
  ReplacementCall replacement;
  ConstraintProgram constraints;     // optional, immutable, declarative
  MatchConditionFn condition;        // optional, read-only C++ escape hatch
}
```

`PatternFunctionProto` is a C++ alias, not a new protobuf or wrapper type:

```cpp
using PatternFunctionProto = ONNX_NAMESPACE::FunctionProto;
```

The alias creates no distinct runtime or type identity. Public signatures may
spell the alias to communicate intent, while callers pass an ordinary
`FunctionProto`; internal normalization produces the existing ID-based compiled
pattern representation.

An **anchor** is the target output-producing node/output slot at which a rule
attempt is seeded. For a pattern with multiple formal-output producer groups, it
is the target node for the primary group selected by FunctionExtractor's existing
candidate-rarity ordering.

At one anchor, rules are ordered by
`(anchor_local_priority, registration_order, RuleId)`. Duplicate `RuleId`s are
rejected during `FusionRuleSet` construction. This is deliberately not a global
priority: Section 8 chooses consumer-anchor-first conflict resolution.

The pattern-role `FunctionProto` is owned and interpreted only as a graph pattern.
It obeys the structural portion of the current FunctionExtractor v2 contract:

- its body is a connected, acyclic, pure tensor DAG;
- its formal inputs/outputs define named pattern boundaries;
- fixed and parameterized attributes retain current exact semantics;
- more than one pattern operation node must be removed.

Its own `(domain, name, overload)` is diagnostic metadata, not the emitted callee
and need not be registered. Its `opset_import` still resolves body nodes exactly
against the target model's registries.

The replacement is a separate, narrow descriptor:

```text
ReplacementCall {
  domain;
  op_type;
  since_version;         // exact registered schema/function version
  overload;
  input_bindings[];       // ordered FormalInput references or Missing
  output_bindings[];      // ordered FormalOutput references
  attributes[];           // literal or bound-formal-attribute sources
}
```

The target model must resolve this exact call identity to an operator schema or
model-local function, and the model's domain import must select a compatible
version. The descriptor cannot contain a replacement graph or invoke an arbitrary
builder callback. It may only connect existing pattern boundary values and
canonical attributes to one call. Thus pattern/replacement identity is decoupled
without introducing a second graph language.

### 3.2 Target graph contract

- The graph is resolved on entry and at the start of every discovery epoch.
- `GraphResolveNeeded()` is false and each target node has a non-null `Node::Op()`.
- The caller has exclusive mutation access.
- One invocation searches one graph scope.
- Provider assignment, control edges, purity, layering annotations, literal
  witnesses, closure, and convexity retain FunctionExtractor's current rules.
- Successful return leaves the graph resolved at a fusion fixpoint.
- A pre-mutation failure leaves the current batch untouched. A failure after the
  first mutation is non-atomic and reports the exact number of calls added.

### 3.3 Condition contract

- The declarative program and callback are predicates. They cannot alter the
  removable set, promote private values to outputs, or change bindings.
- Missing type, rank, dimension, or attribute information is **not proof**.
  Declarative leaves first compute `Unknown`, then map it according to that
  leaf's explicit, review-visible unknown policy. The default maps it to
  rejection.
- The first version supports only `UnknownPolicy::Reject` and
  `UnknownPolicy::NotContradicted`. The latter is legal only for compatibility
  checks such as "known dimensions must not disagree"; it must not be used for a
  kernel precondition such as "rank is 4".
- The callback is invoked once per complete structural candidate. It receives a
  read-only, ephemeral view. It must be deterministic, side-effect free, and must
  not retain graph pointers after return.
- A callback error is an engine error. A normal unsatisfied condition is a
  candidate rejection, not an error.

## 4. Rule and constraint API

The following is an API shape, not a required spelling:

```cpp
using RuleId = uint32_t;
using ConstraintId = uint32_t;

struct FusionRuleOptions {
  RuleId id;
  std::string name;
  int32_t anchor_local_priority{};
};

class FusionRule final {
 public:
  FusionRule(const ONNX_NAMESPACE::FunctionProto& pattern_function,
             ReplacementCall replacement,
             ConstraintProgram constraints,
             MatchConditionFn condition,
             FusionRuleOptions options);
};

struct FusionRuleSetOptions {
  size_t max_epochs{1'000'000};       // additionally capped by entry node count
  size_t max_replacements{1'000'000};
  size_t max_rule_attempts{10'000'000};
  size_t max_condition_evaluations{1'000'000};
  DiagnosticOptions diagnostics{};
};

struct FusionRewriteResult {
  common::Status status;
  size_t replacements_applied{};
  size_t epochs_completed{};
};

class FusionRuleSet final {
 public:
  FusionRuleSet(std::vector<FusionRule> rules,
                FusionRuleSetOptions options = {});
  FusionRewriteResult Apply(Model& model,
                            MatchTraceCollector* trace = nullptr) const;
  FusionRewriteResult Apply(Graph& graph,
                            MatchTraceCollector* trace = nullptr) const;
};
```

The rule set owns copied pattern protos, replacement descriptors, compiled
constraint programs, and callbacks. Per-graph compiled patterns, replacement
schemas, snapshots, bindings, plans, and diagnostic attempts are invocation-local.

### 4.1 Stable references

Constraint authors refer to pattern entities, never target names:

```text
ValueRef =
  FormalInput(index or name)
  FormalOutput(index or name)
  PatternValue(PatternValueId)

NodeRef = PatternNode(PatternNodeId)

AttributeRef =
  FormalAttribute(FormalAttributeId or name)
  EffectiveNodeAttribute(NodeRef, operator_local_name)

DimRef = (ValueRef, axis)
```

Names are resolved to numeric IDs when the rule is constructed. Unknown names,
invalid axes in exact-rank declarations, and references to missing/unsupported
entities reject the rule before any target graph is inspected.

### 4.2 Declarative sublanguage

`ConstraintProgram` has two immutable, bounded parts:

```text
ConstraintProgram {
  dimension_equivalence_classes[];
  predicate;                         // AllOf/AnyOf/Not over pure typed leaves
}
```

Dimension equivalence classes are evaluated first as one order-independent
unification phase. They do not bind a mutable symbol environment. The subsequent
boolean AST is pure, short-circuiting, and deterministic. Equivalence-class count,
operand count, AST depth, and leaf count have explicit construction budgets.

Every leaf is constructed with:

```text
UnknownPolicy::Reject             // default
UnknownPolicy::NotContradicted    // compatibility checks only
```

Leaf evaluation first produces `True`, `False`, or `Unknown`. The policy is
applied only at that leaf. Dimension equivalence classes use the same mapping:

| Raw leaf result | `Reject` | `NotContradicted` |
|---|---|---|
| `True` | `True` | `True` |
| `False` | `False` | `False` |
| `Unknown` | `False` | `True` |

Boolean operators therefore receive only booleans. `Not` negates the
policy-resolved leaf/subexpression; it never applies an implicit unknown policy.
This makes permissive unknown handling local and visible in code review, for
example:

```text
DimEquals(Dim(x, 0), Dim(y, 0), UnknownPolicy::NotContradicted)
RankIs(x, 4, UnknownPolicy::Reject)
```

`NotContradicted` is valid only for a predicate whose documented meaning is
"accept unless known facts conflict." Rule construction rejects it for positive
requirements such as `RankIs`, `ElementTypeIn`, `AttributePresent`, and bounded
attribute ranges.

#### Type and dtype leaves

```text
IsTensor(value)
IsSparseTensor(value)                 // reserved; rejected by v1 rules
IsPresent(value)
IsMissing(value)
ElementTypeIs(value, elem_type)
ElementTypeIn(value, {elem_types...})
SameElementType(lhs, rhs)
TypeProtoEquals(lhs, rhs)
```

`ElementType*` requires an available tensor element type.
`TypeProtoEquals` compares the complete supported type kind and recursively
compares known structure; it is not string comparison of `NodeArg::Type()`.
`IsPresent` and `IsMissing` inspect only whether an optional pattern slot bound an
existing target value; they never treat unknown type/shape as a missing value.

#### Rank and shape leaves

```text
RankIs(value, rank)
RankIn(value, min_rank, max_rank)
SameRank(lhs, rhs)
DimValueIs(DimRef, int64_value)
DimEquals(lhs_dim, rhs_dim)
ShapeEquals(lhs, rhs)
```

Dimension facts are:

```text
Concrete(int64_t)
Symbol(std::string)
Unknown
```

`DimEquals` succeeds only when equality is provable:

- equal concrete values;
- the same non-empty `dim_param`.

Different symbols, concrete-versus-symbol, and unknown facts are not provably
equal. They produce `Unknown` when no contradiction is known and `False` for a
known contradiction; the leaf's explicit policy then applies. `ShapeEquals`
first compares rank, then applies this rule to each dimension. Negative axes are
normalized only after rank is known.

Each separate equivalence-class declaration is:

```text
DimensionEquivalenceClass {
  label;                    // diagnostic only
  dimensions[];             // complete atomic operand set
  unknown_policy;           // Reject by default
}
```

The evaluator extracts every dimension fact, computes equality over the complete
set without modifying shared state, and applies the class's unknown policy. All
classes must succeed before the boolean AST runs. A class cannot occur inside
`AllOf`, `AnyOf`, or `Not`; rule construction rejects such a representation.
There is therefore nothing to roll back or merge across branches.

SDPA-style declarations are:

```text
DimensionEquivalenceClass("B",
  {Dim(query, 0), Dim(key, 0), Dim(value, 0)})
DimensionEquivalenceClass("H",
  {Dim(query, 1), Dim(key, 1), Dim(value, 1)})
DimensionEquivalenceClass("Dh",
  {Dim(query, 3), Dim(key, 3)})
```

#### Attribute leaves

```text
AttributePresent(attribute)
AttributeTypeIs(attribute, type)
AttributeEquals(attribute, canonical_literal)
AttributeIn(attribute, {canonical_literals...})
IntAttributeInRange(attribute, min, max)
FloatAttributeInRange(attribute, min, max)
StringAttributeIn(attribute, {byte_strings...})
SameAttributeValue(lhs, rhs)
```

Attribute lookup uses the same effective view and canonical equality as
FunctionExtractor:

1. explicit node attribute;
2. resolved operator-schema default; or
3. missing.

Formal attribute references use the match-local canonical binding. Node attribute
references use the concrete target node mapped from `NodeRef`. Range checks reject
NaN and use explicitly documented inclusive/exclusive bounds.

#### Boolean composition

Common programs should be flat conjunctions. `AnyOf` exists for bounded structural
alternatives that share one body, such as an attribute accepting two enum values.
`Not` receives the boolean produced after each leaf's explicit unknown-policy
mapping; it never changes or supplies that policy.

### 4.3 C++ escape hatch

```cpp
enum class ConditionDecision : uint8_t {
  kSatisfied,
  kNotSatisfied,
};

struct ConditionFailure {
  std::string_view reason;              // copied only when tracing is enabled
  std::optional<PatternNodeId> node;
  std::optional<PatternValueId> value;
  std::optional<FormalAttributeId> attribute;
};

struct ConditionResult {
  ConditionDecision decision;
  std::optional<ConditionFailure> failure;
};

class MatchedNodeView final {
 public:
  NodeIndex Index() const;
  std::string_view Domain() const;
  std::string_view OpType() const;
  std::string_view Overload() const;
  int SinceVersion() const;
  gsl::span<const PatternValueId> InputValues() const;
  gsl::span<const PatternValueId> OutputValues() const;
  AttributeView EffectiveAttribute(std::string_view name) const;
};

class BoundValueView final {
 public:
  std::string_view Name() const;
  TypeView Type() const;
  ShapeView Shape() const;
  std::optional<PatternNodeId> MatchedProducer() const;
  std::optional<size_t> ProducerOutputIndex() const;
};

class FusionMatchContext final {
 public:
  PatternView Pattern() const;
  MatchedNodeView MatchedNode(PatternNodeId) const;
  BoundValueView BoundValue(PatternValueId) const;
  LiteralView Literal(PatternValueId) const;
  BoundValueView BoundInput(size_t) const;
  BoundValueView BoundOutput(size_t) const;
  AttributeView BoundAttribute(FormalAttributeId) const;
  TypeView Type(PatternValueId) const;
  ShapeView Shape(PatternValueId) const;
  AttributeView EffectiveAttribute(PatternNodeId,
                                   std::string_view name) const;
};

using MatchConditionFn =
    std::function<common::Status(const FusionMatchContext&, ConditionResult&)>;
```

The implementation should prefer a small-function wrapper or function pointer plus
opaque state if `std::function` allocation becomes material; that is an
implementation choice, not a semantic difference.

The callback sees the full injective pattern-node mapping, every bound
input/output/internal value, inferred `TypeProto`/shape facts, effective
attributes, literal witnesses, and canonical formal-attribute bindings through
opaque views. It receives no `Graph`, `Node`, or `NodeArg`, including indirectly:
`MatchedNodeView` exposes no containing graph, raw edges, mutable definitions, or
subgraph map, and `BoundValueView` exposes no producer/consumer pointers.
Restricting observation to the match is required for batch soundness: a callback
cannot base acceptance on a distant node that a co-selected plan may replace.

`TypeView`, `ShapeView`, `AttributeView`, and `LiteralView` are
capability-limited value facades. They expose only type kind, tensor element type,
rank/dimension facts, attribute type, and typed scalar/repeated/tensor contents.
They never return protobuf pointers, schema pointers, snapshot references,
backing containers, or generic "unwrap" handles. `PatternView` exposes only stable
pattern IDs, names, and formal-role metadata. These whitelists are closed APIs;
adding an accessor requires updating dependency capture and prevalidation.

Every view accessor records the observed fact in the candidate's dependency
recorder. The immutable replacement plan owns canonical snapshots of all facts
actually observed:

- mapped node existence, resolved domain/op/overload/version, and requested slot
  mappings;
- bound value identity, name, requested type/shape/dimension facts, and requested
  matched-producer/slot facts;
- every requested effective or formal attribute's presence, type, and canonical
  value;
- requested literal-witness facts; and
- the declarative program's statically known operands.

Whole-batch prevalidation re-reads and compares every recorded dependency before
the first mutation. A callback cannot observe an unrecorded match fact because
the opaque accessors are the only observation path. Returning or retaining a view
past callback completion is invalid, and views are non-copyable outside that
invocation.

Reusable helper functions should return `ConditionResult` fragments so callbacks
can preserve a precise source without throwing. Exceptions never cross this API.

## 5. Condition evaluation pipeline

The normative candidate pipeline is:

```text
root enumeration
  -> complete structural match and attribute binding
  -> dimension-equivalence phase
  -> pure declarative predicate AST
  -> C++ condition callback
  -> closure
  -> convexity
  -> final semantic validation
  -> global conflict selection
  -> whole-batch prevalidation
  -> apply
  -> Graph::Resolve
```

### Why conditions run here

Conditions require the complete concrete binding. Before structural completion,
not all formal inputs, outputs, repeated values, attributes, or pattern nodes are
known. Running the callback incrementally would expose partial-state ordering and
make callbacks difficult to reason about.

Conditions run before closure and convexity because they are local, read-only, and
usually cheaper than consumer-index scans and convexity BFS. A dtype or rank
failure should reject immediately. Closure and convexity remain authoritative
safety checks and cannot be overridden by a condition.

At condition time, internal bindings are only structurally internal; closure has
not yet proved them private. Constraint authors and callbacks must not assume
absence of outside explicit/implicit consumers. A future privacy-dependent
condition class would need a separate post-closure gate. In diagnostics, the
first executed failing stage is reported; the engine does not continue through
closure merely to discover a later failure.

The compiled declarative evaluator may opportunistically evaluate a leaf as soon
as all of its operands are bound, solely as an optimization. Its observable result,
diagnostic identity, and three-valued semantics must equal evaluation at the
normative point. The C++ callback is never invoked early or more than once.

Whole-batch prevalidation rechecks facts that can become stale through another
selected plan: node/value existence, attribute occurrences, literals, consumers,
closure, convexity, replacement-call compatibility, and every fact recorded by
the declarative evaluator or callback's opaque views. Conditions need not rerun:
prevalidation proves that their complete observable dependency set is unchanged.
Selected plans are nonoverlapping, boundary-adjacent plans conflict, and no
unrecorded callback observation is possible.

### 5.1 Shape/type source and unknowns

At the start of an epoch:

- `NodeArg::TypeAsProto()` is the full inferred type source;
- `NodeArg::Shape()` supplies tensor/scalar shape when available;
- tensor element type is read from `TypeProto::tensor_type().elem_type()`;
- dimensions use `TensorShapeProto::Dimension::dim_value` or `dim_param`; and
- `Node::Op()` supplies the resolved schema and attribute defaults.

These facts are available only on a resolved graph. `GraphViewer` topological
order and cached schema/type facts are treated as invalid after mutation.

The engine does not run ad hoc inference while matching. If required information
is absent, the condition rejects conservatively. After a batch, `Graph::Resolve()`
runs normal ORT type/shape inference before the next epoch.

`Graph::UpdateShapeInference(Node&)` is not the general solution: it is intended
for local constant-folding use, excludes control-flow nodes, mutates output shape
facts, and does not restore a structurally modified graph's global invariants.

### 5.2 Replacement-call validation

The replacement is the primitive or function call identified by
`ReplacementCall`, independently of the pattern proto's identity. Its formal
input/output type constraints and type/shape inference come from its resolved
operator or model-local-function schema.

Before mutation, the plan validates:

1. the replacement domain/op/overload and exact `since_version` resolve
   unambiguously under the target model's domain imports and registries;
2. descriptor input/output mappings are complete, in range, and obey the
   replacement schema's arity and optional slots;
3. literal and pattern-formal attribute mappings produce exactly the required
   canonical replacement attribute set;
4. each existing boundary `NodeArg::TypeAsProto()` satisfies the call schema's
   type constraints; and
5. inferred replacement outputs against the existing matched formal-output
   `NodeArg` types/shapes, rejecting known contradictions.

This should use a reusable, read-only "virtual node validation" helper extracted
from the logic behind `Graph::InferAndVerifyTypeMatch`, rather than temporarily
adding a node to the live graph. The helper supplies boundary type protos,
canonical call attributes, and an inference context, then returns inferred output
protos without mutating `NodeArg`s. Until that helper exists, a rule must duplicate
all nontrivial replacement preconditions in its `ConstraintProgram`, and
`Graph::Resolve()` remains the post-mutation safety net. Shipping broad rule
coverage should be gated on the virtual validator to avoid avoidable non-atomic
resolve failures.

Rule-authored constraints express additional semantic or kernel preconditions not
already guaranteed by the replacement schema. They do not replace schema
validation. Pattern compilation separately resolves pattern-body nodes; it never
requires the pattern proto itself to be registered.

### 5.3 Mapping ONNXScript's three levels

| ONNXScript mechanism | C++ model |
|---|---|
| `RewriteRule(..., condition_function=...)` | `FusionRule::condition` over `FusionMatchContext`. |
| `RewriteRuleClassBase.check(...)` | Same callback, packaged by a rule-builder class if desired. There is one semantic callback level. |
| `Var(check=...)` and node `_check=...` | Prefer declarative leaves scoped by `ValueRef` or `NodeRef`; unusual checks use the same whole-match callback and identify their source in `ConditionFailure`. |

Unlike ONNXScript, executable checks are not embedded in the portable graph
pattern. This separation keeps the pattern `FunctionProto` serializable, makes
common preconditions inspectable, and avoids three subtly different callback
lifecycles.

## 6. Worked example: Tanh GELU to FastGelu

### 6.1 Pattern and replacement

Create a pattern-only `FunctionProto`, for example with diagnostic identity
`ort.pattern::TanhGeluFloat`, whose body expresses:

```text
x3 = Pow(x, FLOAT(3.0))
cubic_coefficient = FLOAT(0.044715)
t0 = Mul(cubic_coefficient, x3)
t1 = Add(x, t0)
t2 = Mul(FLOAT(0.7978845608028654), t1)  // sqrt(2 / pi)
t3 = Tanh(t2)
t4 = Add(t3, FLOAT(1.0))
t5 = Mul(FLOAT(0.5), t4)
y  = Mul(x, t5)
```

The pattern's own identity is not registered and is never emitted. Its scalar
constants are explicitly typed FLOAT fixed body `Constant`s and therefore use
FunctionExtractor's exact literal-witness rules. The eight arithmetic operations
are the removable set.

The separate descriptor is:

```text
ReplacementCall {
  domain = "com.microsoft";
  op_type = "FastGelu";
  since_version = 1;
  overload = "";
  input_bindings = [FormalInput("x")];
  output_bindings = [FormalOutput("y")];
  attributes = [];
}
```

A successful match emits `y = com.microsoft::FastGelu(x)` using the exact
existing target `NodeArg` for `y`. The registered FastGelu schema, not the pattern
proto identity, validates this call.

Exact typed literals mean one pattern cannot honestly advertise FLOAT16,
BFLOAT16, FLOAT, and DOUBLE matches. Supporting all four requires four rule
variants with constants encoded in the target dtype (and the exact target
expansion for that dtype), sharing the same replacement descriptor. A rule-builder
may generate those variants, but each remains an independently compiled,
diagnosable exact pattern.

### 6.2 Constraints

An illustrative conservative program is:

```text
AllOf(
  IsTensor(FormalInput("x")),
  ElementTypeIs(FormalInput("x"), FLOAT),
  SameElementType(FormalInput("x"), FormalOutput("y")),
  ShapeEquals(FormalInput("x"), FormalOutput("y"))
)
```

`ShapeEquals` requires statically provable equality in this example. A production
rule may omit it when the exact registered FastGelu schema already proves
shape-preserving output and the virtual-call validator verifies compatibility.
Omitting redundant constraints improves coverage without weakening safety.

No C++ callback is required. If a particular execution provider later requires,
for example, a bounded last dimension or disallows a denormal mode, that
provider-independent semantic requirement may be expressed declaratively when
possible. Provider kernel availability itself remains outside this graph rewrite.

### 6.3 Pipeline

1. The final `Mul` is considered as an output producer.
2. Reverse matching forces every predecessor by output slot and positional input.
3. Exact constants, fixed attributes, schema identities, and repeated binding of
   `x` are checked.
4. The complete `x` and `y` bindings are evaluated by the constraint program.
5. Closure proves private intermediates have no external or implicit consumers.
6. Convexity proves no path leaves and re-enters the eight-node region.
7. Replacement-call schema compatibility is validated.
8. A plan removes the eight nodes and adds one `FastGelu` call.
9. The batch is resolved; later epochs may fuse that new call with surrounding
   operations.

### 6.4 Near miss

If the FLOAT structure nearly matches but the `0.044715` coefficient differs by
one bit, tracing reports:

```text
rule: TanhGeluFloatToFastGelu [id=17]
stage: literal
anchor: node 842 "gelu_output_mul" (Mul, output 0)
matched: 6/8 pattern operation nodes
pattern: value "cubic_coefficient", FLOAT scalar 0x3d372713
target: value "gelu_cubic_coefficient" initializer, FLOAT scalar 0x3d372714
result: rejected -- tensor literal differs at element 0
```

The condition is not reported because structural matching never completed. A
shape/type near miss that reaches the condition stage uses the condition-format
record from Section 7 and identifies the decisive `ConstraintId`.

## 7. Match-failure diagnostics

### 7.1 Modes and cost

```text
DiagnosticMode::Off          // default
DiagnosticMode::BestFailure  // bounded best attempt per rule
DiagnosticMode::AllFailures  // bounded event log for tests/debugging
DiagnosticMode::DryRun       // BestFailure + successes, no mutation
```

With `Off`, the hot path has only a nullable sink check at existing rejection
sites. It does not allocate diagnostic strings, copy bindings, retain target
objects, or format protos.

`BestFailure` stores at most one compact failure **per rule for the complete
`Apply` invocation**, and only when that rule produced no accepted candidate. If
a rule succeeds in any epoch, its ordinary failures are discarded; success
counters remain available, and `DryRun` records the deterministically selected
successes. Human-readable strings are formatted after matching. `AllFailures`
retains bounded successes and failures regardless of whether the rule succeeds
and requires explicit
`max_diagnostic_records` and `max_diagnostic_bytes`; reaching either cap sets a
`truncated` flag but does not fail rewriting.

`DryRun` performs discovery, conditions, closure, convexity, selection, and
prevalidation against the immutable graph, reports matches, and never calls apply
or `Graph::Resolve()`.

### 7.2 Structured record

```text
MatchFailureRecord {
  RuleId rule_id;
  MatchStage stage;
  MatchFailureCode code;
  NodeIndex anchor_node;
  size_t anchor_output_slot;
  optional<PatternNodeId> pattern_node;
  optional<PatternValueId> pattern_value;
  optional<ConstraintId> constraint;
  optional<NodeIndex> target_node;
  optional<size_t> target_slot;
  string target_value_name;            // copied only when enabled
  size_t pattern_nodes_matched;
  SmallVector<FailureArgument> args;    // enum/int/type/dim/attr summaries
}
```

Stages, in increasing progress order:

```text
RootSignature
StructuralNode
StructuralEdge
ValueBinding
AttributeBinding
Literal
Condition
Closure
Convexity
FinalValidation
Prevalidation
Success
```

Failure codes are stable enums such as `OpMismatch`, `OutputSlotMismatch`,
`RepeatedBindingMismatch`, `MissingEffectiveAttribute`,
`AttributeValueMismatch`, `UnknownRank`, `DimensionMismatch`,
`ConstraintFalse`, `ExternalPrivateUse`, `NonConvex`, and `StalePlan`.
User callback failures use `CallbackRejected` plus a copied reason when tracing.

### 7.3 Accumulation during reverse matching

Each output-root tuple owns a lightweight `MatchAttempt`:

1. Record anchor and rule identity.
2. As the reverse-topological worklist advances, update
   `pattern_nodes_matched` and the current pattern/target source IDs.
3. At the first rejecting check, create one terminal failure record. Matching
   remains fail-fast; it does not continue solely to collect more errors.
4. Submit the record to the collector.

`MatchTraceCollector` is one invocation-level object containing an independent
`RuleTrace` slot for each `RuleId`; it never compares failures from different
rules. Each unsuccessful rule's slot keeps its best failed attempt by:

```text
(stage reached, pattern_nodes_matched,
 earliest epoch, anchor reverse-topological rank, deterministic_tuple_ordinal)
```

Later stages dominate larger partial structural matches, following ONNXScript's
useful "best failure" principle. Exact deterministic tie-breaking avoids
thread/order-dependent reports. There is no global best failure across rules.
Reports are presented by `(anchor_local_priority, registration_order, RuleId)`;
presentation order does not affect intra-rule scoring or rewrite selection.
`AllFailures` additionally appends terminal records in attempt order.

Constraint evaluation records the exact `ConstraintId` and operand references.
Nested `AllOf`/`AnyOf` failures retain a bounded path from the root expression to
the decisive leaf. Callback helpers set a source value/node/attribute; a callback
that returns only `NotSatisfied` receives the generic reason
`CallbackRejectedWithoutReason`.

Diagnostics are observational. Enabling them cannot change enumeration, budgets,
selection, or mutation. Diagnostic storage has separate caps and is not charged
to semantic matching budgets.

## 8. Whole-graph `FusionRuleSet`

### 8.1 Why reverse topological order

FunctionExtractor patterns are rooted at formal outputs and matched backward.
The natural anchor is therefore a target output-producing node.

Within each resolved epoch, anchors are visited in reverse topological order:

- an anchor's downstream context is already stable and available for closure and
  convexity;
- consumer-side fusions win deterministically over upstream overlapping fusions;
- plans can be applied safely in reverse topological order, matching the existing
  edge-removal discipline; and
- the traversal aligns with the matcher instead of converting between a forward
  driver and backward candidate order.

Forward traversal is not incorrect. ONNXScript uses it effectively because its IR
has a mutation-aware linked-list iterator and inserts replacements after the
current anchor. ORT's `GraphViewer` explicitly invalidates cached topological order
after mutation, `Graph::RemoveNode` frees the node, and `Graph::AddNode` may grow
the backing vector. The C++ design must not infer safety from `GraphNodes`
iterators merely skipping null slots.

### 8.2 One immutable discovery epoch

For each epoch:

1. Require a resolved graph.
2. Compile every pattern and resolve every replacement against the current model's
   schema/function registries.
3. Build one shared target snapshot: topological order/positions, producers,
   explicit and implicit consumers, graph outputs, control-edge nodes,
   initializers, and root-signature buckets.
4. Visit anchors in reverse topological order.
5. Use the anchor's coarse signature to obtain applicable rules. For every
   applicable rule in anchor-local order, enumerate compatible formal-output root
   tuples anchored there and run the complete candidate pipeline.
6. Retain every accepted immutable plan. Rules are not hidden merely because an
   earlier rule accepted at that anchor; this makes diagnostics complete and lets
   global conflict selection apply one consistent consumer-anchor-first policy.
7. Sort plans by:

   ```text
   (anchor reverse-topological rank,
    anchor-local rule priority,
    rule registration order,
    root-tuple ordinal,
    removable NodeIndex vector)
   ```

8. Greedily select a deterministic maximal nonconflicting set.
9. Prevalidate the complete selected batch.
10. Destroy the snapshot before mutation.
11. Apply plans in reverse topological order, resolve once, and begin the next
    epoch.

Two plans conflict under FunctionExtractor's existing rules:

- their removable sets overlap; or
- either plan's boundary output is the other's boundary input.

The second rule intentionally defers node-disjoint adjacent rewrites. It prevents
one selected plan from invalidating another's boundary type, producer, consumer,
or condition facts inside the batch.

Rules may share compiled root-signature indexes and the target snapshot, but each
candidate retains rule-specific pattern state and budgets. A rule cannot consume
another rule's matcher budget silently; the rule set also has aggregate attempt
and condition-evaluation caps.

### 8.3 What happens after replacement

There is no live iterator to continue.

After a batch, every `Node*`, `GraphViewer` order, consumer index, schema-derived
view, and pointer-bearing plan is discarded. `Graph::Resolve()` restores graph
invariants and type/shape facts. The next epoch obtains a fresh topological order.

The newly created call is an ordinary node in that order and is eligible as an
anchor or predecessor for every rule. A downstream node visited earlier in the
old epoch is also reconsidered, so a new producer can enable a surrounding fusion.
This explicit outer revisit replaces ONNXScript's incidental same-sweep visit of
nodes inserted after the current linked-list position.

No "resume NodeIndex" is persisted. Node indexes are stable identifiers only while
their nodes exist; removed-node indexes and all `Node*` handles are invalid.

### 8.4 Fixpoint and convergence

Let `N_e` be the number of graph nodes at the start of epoch `e`. Every selected
plan removes `r > 1` mapped pattern operation nodes and adds one call.
Nonconflicting plans have disjoint removable sets, so:

```text
N_(e+1) = N_e - sum(r_i - 1) < N_e
```

Therefore:

- every successful epoch strictly decreases node count;
- at most `N_0 - 1` replacements can be applied;
- at most `N_0` successful epochs are possible; and
- cycles between rules are impossible even if a new call enables another rule.

The driver stops successfully when an epoch selects no plan. That is the rule-set
fixpoint for the current graph scope, consumer-anchor policy, and registered
rules.

Defensive budgets are checked with the existing pre-increment `counter >= limit`
discipline:

- epochs, additionally capped by `N_0`;
- replacements, additionally capped by `N_0 - 1`;
- rule/anchor attempts;
- output-root tuples;
- aggregate matcher work;
- condition evaluations and constraint leaves;
- closure/convexity traversal;
- literal and attribute payload bytes; and
- diagnostic records/bytes, separately.

Reaching a semantic work budget returns `FAIL` before the current batch mutates.
Reaching the node-count-derived epoch bound despite strict decrease is an internal
invariant failure.

### 8.5 Consumer-anchor precedence and anchor-local priority

The normative global policy is **consumer-anchor-first**. When plans at different
anchors overlap, the plan whose anchor appears first in reverse topological order
wins, regardless of either rule's numeric priority. This preserves the traversal
decision in Section 8.1 and favors the larger downstream context before an
upstream rewrite can hide it.

`anchor_local_priority` resolves competing plans only when they have the same
anchor rank. Registration order, tuple ordinal, and removable-node indexes then
break remaining ties. It is deterministic policy, not semantic correctness:
every accepted plan is individually semantics preserving. A losing plan may be
rediscovered in the next epoch.

This avoids two failure modes:

- applying the first rule immediately at an anchor and invalidating traversal; and
- permanently suppressing another rule that becomes applicable after an
  overlapping rewrite changes the graph.

The engine does not promise a globally maximum number of fusions. It promises a
deterministic maximal batch and a deterministic fixpoint.

## 9. Edge cases

- **Unknown rank/dtype/shape:** declarative proof-required constraints reject.
  Callbacks may explicitly accept unknowns, but that decision is visible in code.
- **Symbolic dimensions:** equality uses actual `dim_param` identity or equal
  concrete values. Different symbol names are not assumed equal.
- **Negative axes:** reject when rank is unknown or normalization is out of range.
- **Optional missing values:** a `ValueRef` to an omitted slot evaluates as
  missing; only an explicit `IsMissing`/`IsPresent` leaf may inspect it.
- **Schema defaults:** attribute predicates see the same effective explicit-or-
  schema-default view as structural matching.
- **Function defaults:** do not synthesize target evidence. Formal bindings retain
  current FunctionExtractor semantics.
- **Repeated formal values/attributes:** constraints observe the single consistent
  binding already proved by the matcher.
- **Initializers/constants:** fixed literals remain witnesses outside the removable
  set; parameterized standard ONNX `Constant` remains structural.
- **Graph outputs and fan-out:** formal outputs may retain both; private values may
  not.
- **Implicit captures:** remain external consumers for closure and convexity.
- **Provider-assigned/control-edge nodes:** reject only when mapped/prevalidated.
- **Subgraphs:** not searched in v1. A future recursive driver must define scope,
  ordering, budgets, and function-body policy separately.
- **Callback nondeterminism or mutation:** contract violation. Debug builds may
  fingerprint the graph before/after callback execution.
- **Callback exceptions:** catch at the API boundary and convert to failure status;
  never treat them as an ordinary non-match.
- **Rule duplicates:** multiple rules may share one `ReplacementCall` identity,
  including dtype-specific variants, while owning different exact patterns.
  Duplicate `RuleId`s are rejected; consumer-anchor precedence and then
  anchor-local priority resolve overlap.
- **Replacement already a single node:** the pattern still must remove more than
  one node. A matched existing call is not rewritten to itself.
- **No-op/expanding rules:** rejected at rule construction; otherwise the
  convergence proof fails.
- **Post-mutation resolve failure:** report non-atomic failure and exact calls
  added. Clone-and-swap remains the caller's atomicity option.

## 10. Delta from FunctionExtractor v1/v2

| Area | Current FunctionExtractor | Proposed engine |
|---|---|---|
| Rule count | One function per extractor invocation. | Ordered, owned set of function-backed rules. |
| Structural matcher | Output-seeded, reverse-topological, positional, exact. | Reused without semantic change. |
| Attributes | Typed formal binding and exact fixed/effective values. | Reused; declarative predicates may inspect bindings/effective attributes. |
| Conditions | Built-in structural/type compatibility only. | Declarative typed constraint AST plus one C++ callback. |
| Shape semantics | Known facts are conservative compatibility checks. | Explicit rank/shape/dim predicates over concrete bindings. |
| Diagnostics | Aggregate counters only. | Optional structured best-failure/all-failure/dry-run tracing. |
| Discovery | One pattern builds its own snapshot/indexes. | Rules share one snapshot and anchor/signature dispatch per epoch. |
| Selection | Nonconflicting plans for one function. | Global deterministic selection across rules and anchors. |
| Iteration | Per-extractor strict-decrease fixpoint. | Rule-set strict-decrease fixpoint with cross-rule enabling. |
| Mutation safety | Snapshot discarded, batch applied, resolve, repeat. | Same discipline; explicitly no mutable traversal iterator. |
| Replacement | One call to the matched FunctionProto identity, which requires exact registration. | Independent narrow `ReplacementCall`; primitive and function calls are supported without registering the pattern body under the callee identity. |

The proposal is additive at the architecture level. `FunctionExtractor` can become
a compatibility facade that constructs a one-rule `FusionRuleSet` whose
`ReplacementCall` names its source function identity, preserving its exact
registration contract. General fusion rules instead provide an independent
replacement descriptor. Keeping two independent match/apply implementations would
create semantic drift and should be avoided.

## 11. Incremental implementation roadmap

### Phase 1: Extract a reusable one-rule core

Files:

- `function_extractor_pattern.{h,cc}`
- `function_extractor_matcher.{h,cc}`
- `function_extractor.cc`

Changes:

1. Rename internal extractor-specific concepts only where needed to make the core
   rule-neutral; do not churn public names gratuitously.
2. Separate:
   - context-free normalized pattern;
   - per-graph compiled pattern;
   - resolved replacement-call schema;
   - shared target snapshot; and
   - rule-specific candidate discovery.
3. Parameterize candidate discovery by an anchor/root group so a RuleSet can
   dispatch rules from one graph traversal.
4. Preserve all existing FunctionExtractor tests through a one-rule adapter.

Exit criterion: no behavioral delta for the current public API.

### Phase 2: Declarative constraints

New modules:

- `fusion_constraint.h`
- `fusion_constraint.cc`
- `fusion_match_context.h`

Changes:

1. Add stable `ValueRef`, `NodeRef`, `AttributeRef`, and `ConstraintId`.
2. Compile names to numeric pattern IDs during rule construction.
3. Implement the separate, order-independent dimension-equivalence phase.
4. Implement bounded pure-AST validation and deterministic three-valued
   evaluation.
5. Add a condition gate after complete structural matching and before closure.
6. Store exact declarative operands and their canonical observed facts for
   prevalidation.

Tests:

- every leaf and unknown policy;
- concrete and symbolic dimension equality;
- order-independent dimension equivalence and rejection of equivalence
  declarations nested under `AnyOf`/`Not`;
- negative axes;
- explicit versus schema-default attributes;
- formal attribute bindings;
- boolean short-circuiting;
- condition budget exhaustion; and
- no graph mutation on rejection.

### Phase 3: C++ callback

New/changed modules:

- `fusion_rule.h`
- `fusion_match_context.{h,cc}`

Changes:

1. Expose the immutable full binding view.
2. Expose only opaque node/value/attribute facts; never return `Node`, `NodeArg`,
   `Graph`, raw edges, or containing-graph access.
3. Record every opaque-view accessor dependency in the immutable plan and
   prevalidate it before mutation.
4. Define `Status + ConditionResult`, with sourced ordinary failures.
5. Enforce one invocation per complete candidate.

Tests:

- full node/value/attribute visibility;
- inability to escape from opaque views to graph or edge APIs;
- stale-plan rejection for every whitelisted observable dependency;
- sourced and unsourced rejection;
- callback error propagation;
- deterministic invocation count;
- callback not run after structural/declarative failure; and
- pointers cannot survive plan construction.

### Phase 4: Replacement-call virtual validation

ORT infrastructure touch-point:

- reusable code near `Graph::InferAndVerifyTypeMatch` in
  `onnxruntime/core/graph/graph.cc`;
- a narrow internal header under `core/graph`, not an optimizer copy of ONNX
  inference logic.

Changes:

1. Resolve the independent `ReplacementCall` identity and validate its boundary
   and attribute mappings.
2. Build a read-only inference context from schema, canonical attributes, and
   boundary `TypeProto`s.
3. Validate schema type constraints and infer output types/shapes without adding a
   live node or mutating `NodeArg`s.
4. Compare inferred outputs with existing formal-output values.
5. Reuse the helper in fusion prevalidation; consider later reuse by other
   optimizers.

Tests:

- schema type mismatch;
- inferred known-dimension contradiction;
- unknown-compatible output;
- model-local function schema;
- required/defaulted attributes; and
- inference failure before mutation.

### Phase 5: Structured diagnostics

New modules:

- `fusion_diagnostics.h`
- `fusion_diagnostics.cc`

Changes:

1. Add stable stages, reason codes, compact failure records, and collector.
2. Thread a nullable sink through existing matcher rejection sites.
3. Add constraint/callback source reporting.
4. Implement deterministic best-failure scoring, bounded all-failure mode, text
   formatting, and dry-run.

Tests:

- exact source IDs for node/edge/literal/attribute/condition failures;
- later stage outranks larger earlier partial match;
- one best failure per unsuccessful rule, successful-rule failure suppression,
  and presentation ordering independent of scoring;
- deterministic ties;
- truncation;
- diagnostics off produces no retained records; and
- dry-run leaves serialized graph bytes unchanged.

### Phase 6: Whole-graph RuleSet

New modules:

- public `fusion_rule.h`
- public `fusion_rule_set.h`
- `fusion_rule_set.cc`

Changes:

1. Compile all rules per resolved epoch.
2. Share target snapshot and signature-to-rule dispatch.
3. Discover all accepted plans in reverse topological anchor order.
4. Select and prevalidate one global nonconflicting batch.
5. Reuse existing apply mechanics, resolve once per batch, and repeat.
6. Make `FunctionExtractor` a one-rule facade.

Tests:

- anchor-local rule priority at one anchor;
- cross-anchor overlap proving a downstream consumer anchor wins even when the
  upstream rule has a numerically lower priority;
- overlapping rules;
- disjoint cross-rule batch;
- boundary-adjacent deferral;
- new fused call enables another rule in the next epoch;
- downstream node reconsidered after upstream replacement;
- removed-node/iterator invalidation regression;
- deterministic output across registration and allocation noise;
- strict node-count decrease and defensive pass cap;
- aggregate RuleSet budgets; and
- non-atomic apply/resolve accounting.

### Phase 7: Rule migration and performance

1. Implement dtype-specific GELU variants, MatMul+Add, FP16 Softmax, BiasGelu,
   LayerNorm, and a symbolic-shape SDPA rule as representative coverage.
2. Compare each with its existing hand-written ORT fusion for correctness,
   provider behavior, and graph output.
3. Measure snapshot sharing, signature dispatch, condition cost, and resolve
   frequency on large models.
4. Migrate only rules whose semantics and performance are equal or better.
5. Keep specialized transformers where the single-call/function-pattern model is
   not a natural fit.

## 12. Required review decisions

The proposal recommends these choices as normative:

1. **One pattern language:** retain a pattern-only `FunctionProto`; decouple it
   from a narrow one-call replacement descriptor rather than introducing a second
   replacement-graph DSL.
2. **One callback level:** node/value-local common checks compile to declarative
   constraints; unusual logic uses one full-match callback.
3. **Conditions before closure:** complete binding first, then cheap semantic
   rejection, then graph-boundary proofs.
4. **Immutable epochs:** discover and select from a resolved snapshot; never mutate
   through an active graph iterator.
5. **Outer fixpoint:** newly created nodes are reconsidered after resolve, not
   through container-specific same-sweep behavior.
6. **Fusion-only convergence:** require `removed_nodes > 1` and one replacement
   call, preserving strict node-count decrease.
7. **Virtual call validation:** factor a read-only schema/type/shape validator
   before scaling rule coverage.
8. **Diagnostics off by default:** best-failure tracing is explicit, bounded, and
   observational.
9. **Consumer-anchor-first selection:** reverse-topological anchor precedence is
   global; numeric rule priority is explicitly anchor-local.
10. **Pure symbolic-shape semantics:** dimension equivalence classes run in a
    separate order-independent phase; the boolean predicate AST has no bindings or
    rollback semantics.

These choices preserve FunctionExtractor's correctness foundation while adding the
three capabilities needed for a practical fusion framework: semantic conditions,
explainable failure, and safe cross-rule graph rewriting.
