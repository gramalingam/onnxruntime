# FunctionExtractor design

## 1. Problem and design stance

The utility is a **reverse function inliner**. Given an ONNX
`FunctionProto F`, it finds subgraphs equivalent to `F`'s body and replaces each
with one call to `(F.domain, F.name, F.overload)`.

**Requirement (v2).** Prefer false negatives to false positives. Matching is exact,
schema-resolved, positional, scope-local, and conservative about observable values.

**Rationale.** A missed fusion affects performance; an incorrect match silently
changes model semantics.

The implementation has six phases:

1. Context-free `FunctionProto` validation and normalization.
2. Per-model schema/function resolution and pattern compilation.
3. Read-only target snapshot and output-rooted match discovery.
4. Per-candidate closure and convexity validation.
5. Global conflict selection and whole-batch prevalidation.
6. In-place mutation, resolution, and fixpoint iteration.

Discovery never mutates the graph. Application is a **prevalidated mutation
sequence**, not a transaction: ORT has no general graph rollback.

## 2. Terminology

| Term | Definition |
|---|---|
| Formal input/output | An ordered name in `FunctionProto.input` or `.output`. |
| Pattern operation node | A body node retained for structural matching and mapped injectively to a target `Node`. This includes only a parameterized `Constant` admitted by Section 4.3. |
| Pattern value | A named body tensor. It has independent flags such as `is_formal_input`, `is_formal_output`, and `is_literal`; these are not mutually exclusive enum alternatives. |
| Missing slot | An empty optional input/output entry (`""`). It is a slot property, not a value. |
| Literal | A tensor value owned by a fixed-attribute body `Constant` node after normalization. |
| Literal witness | A target constant initializer or `Constant` node output whose value proves a fixed literal match. It is never removed. |
| Formal attribute parameter | A name declared in `FunctionProto.attribute` (required) or a concrete default declared in `FunctionProto.attribute_proto` (defaulted). |
| Attribute variable occurrence | A body-node attribute whose `ref_attr_name` names a formal attribute parameter. Its `name` is the operator-local attribute name and its `type` declares the parameter type. |
| Attribute binding | One concrete canonical value associated with a formal attribute parameter for one candidate match. Its `name` is normalized to the formal parameter name. |
| Aggregate work unit | One bounded normalization, root-enumeration, worklist, or attribute-occurrence operation charged before execution against `max_worklist_bindings`. |
| Output-root tuple | One proposed target value for each formal output, grouped when several outputs come from different slots of the same producer. |
| Candidate | The complete structural mapping forced by one output-root tuple, not yet proven safe at its boundary. |
| Accepted match | A candidate that passes closure, convexity, and all semantic gates. |
| `R` / removable set | Exactly the target nodes mapped from pattern operation nodes. Literal witnesses are never in `R`. |
| Boundary input | A target value consumed by `R` but not produced by `R`; it binds a formal input or witnesses a literal. |
| Boundary output | A target value produced by `R` and declared as a formal output. |
| Private intermediate | A target value produced and consumed inside `R` that is not a formal output. |
| Explicit consumer | A node that lists the value in `InputDefs()`. |
| Implicit consumer/capture | A node that lists the value in `ImplicitInputDefs()`, normally because a nested subgraph captures it. |
| Closed match | A match whose private intermediates have no explicit or implicit use outside `R` and are not graph outputs. |
| Convex match | No data-flow path leaves `R` and later re-enters `R`. |
| Replacement plan | Immutable node indexes, `NodeArg` identities, attribute bindings, boundary edges, and call metadata sufficient to prevalidate and apply one rewrite. |

## 3. Shipped contract: v1 core plus v2 attributes

v2 retains the v1 positional data-flow matcher and adds typed function
attribute-parameter variables. The rules below replace v1's rejection of
`FunctionProto.attribute`, `FunctionProto.attribute_proto`, and `ref_attr_name`.

**v2 delta checklist.**

1. Compile required/defaulted declarations and body references as typed variables.
2. Retain a narrowly admitted parameterized ONNX `Constant` as a structural node.
3. Register required model-local formals as required and `attribute_proto`
   formals as optional-with-default in the generated `OpSchema`.
4. Bind target-effective values deterministically and emit every formal explicitly.
5. Add formal-count and attribute-byte budgets, and charge every variable
   occurrence to the existing aggregate-work budget.

### 3.1 Supported and unsupported features

| Feature | v2 behavior |
|---|---|
| Connected acyclic tensor DAG | Supported and required. |
| Multiple formal inputs/outputs and branching | Supported. |
| Repeated use of a formal input | Supported; all uses bind the same target value. |
| Two distinct formals bound to one target value | Supported. |
| Fixed body attributes | Supported with exact normalized equality. |
| Required/defaulted function attributes | Supported as typed variables. Defaults are declaration metadata: they neither seed nor constrain matching, and extractor-generated calls still emit every target-derived binding explicitly. |
| Body attribute references (`ref_attr_name`) | Supported when they name a declared formal and have a supported type. |
| Body `Constant` used as an internal input | Supported through a non-removable literal witness. |
| Parameterized body `Constant` | Narrowly admitted as specified in Section 4.3; retained structurally and matched only to a target `Constant` node. |
| `Constant`-derived formal output | **Rejected.** |
| Positional optional/variadic arguments | Supported; omitted slots must agree exactly and variadic order is preserved. |
| Input permutations/commutativity | Not supported; identity permutation only. |
| Pattern graph/control-flow attributes | Rejected. |
| Recursive matching of nested target scopes | Not supported; only the supplied graph scope is searched. |
| Target nested-subgraph captures | Indexed as implicit consumers and enforced by closure. |
| Sparse/external-data body constants | Rejected. |
| Overridable initializer as a literal | Rejected. |
| Sequences, maps, optionals, or non-tensor body values | Rejected. |
| Provider-assigned nodes | Rejected only when mapped into a candidate; unrelated target nodes are allowed. |
| Control edges on matched nodes | Rejected. |
| Side-effecting or nondeterministic operations | Rejected by an explicit purity allowlist. |
| Adjacent or overlapping rewrites in one batch | Conflicting; one is deferred to a later pass. |
| Single-operation extraction | Rejected because `|R|` must exceed one. |

### 3.2 Function pattern contract

All context-free checks run when the extractor is constructed, before any target
graph is inspected.

**Requirement.** Run the applicable ONNX `FunctionProto` checker validation and
then enforce all v2-specific checks below. If the checker requires model context,
run its context-free subset in the constructor and the remaining checks before
target snapshot construction. Invalid input returns an error without touching the
target. Because the required constructor itself has no `Status` return, it stores
any context-free validation failure; the first `Extract()` returns that failure
before accessing the supplied model or graph.

The function must satisfy:

1. `name` is non-empty; identity is `(domain, name, overload)`.
2. Formal inputs are non-empty and distinct.
3. Formal outputs are non-empty and distinct.
4. Formal input and formal output name sets are disjoint; passthrough outputs are
   retained as a v1 restriction and remain rejected in v2.
5. No formal input name is produced by a body node.
6. Every non-empty, non-formal input value has exactly one body producer.
7. Every body-produced non-empty value has exactly one producer (SSA).
8. Every formal output is produced by a non-`Constant` pattern operation node.
   A fixed or parameterized `Constant`-derived formal output is rejected.
9. Every body value is reachable in one connected data-flow component after formal
   inputs and literals are treated as leaves, and the operation-node graph is a DAG.
10. Every pattern operation node is transitively backward-reachable from at least
    one formal output. A node on a dead branch, including a sibling branch connected
    only through a formal input or a node downstream of an already-produced formal
    output whose result reaches no formal output, is rejected at compilation.
11. Body nodes contain no graph-valued attributes, control-flow bodies, unsupported
    value kinds, or unsupported constants.
12. Required/defaulted function attribute declarations have non-empty, unique
    names and do not overlap. A formal appears in exactly one of
    `FunctionProto.attribute` or `FunctionProto.attribute_proto`.
13. Each `attribute_proto` default is concrete: it has a supported type and the
    corresponding value encoding (an empty repeated value is valid), has no
    `ref_attr_name`, and its `name` is the formal parameter name.
14. Each body `ref_attr_name` names a declared formal, has a non-empty
    operator-local `name`, has a supported non-`UNDEFINED` `type`, and contains no
    concrete value. All occurrences of one formal declare the same type; a
    defaulted formal's occurrences also match its default type.
15. Every formal attribute parameter is referenced at least once. An unreferenced
    required formal cannot be inferred; v2 also rejects unreferenced defaulted
    formals as unnecessary API surface rather than inventing call attributes.
16. Supported formal-attribute types are `FLOAT`, `INT`, `STRING`, `TENSOR`,
    `FLOATS`, `INTS`, `STRINGS`, and `TENSORS`. Dense tensors count against the
    attribute-byte budget and must carry inline data; external-data tensors are
    rejected. `GRAPH(S)`, `SPARSE_TENSOR(S)`, `TYPE_PROTO(S)`, and `UNDEFINED`
    remain rejected.
17. Node signatures, arities, optional slots, imports, and call signatures are legal
    once resolved against the target model context.
18. A parameterized `Constant` satisfies all Section 4.3 admission criteria.

**Alias rule.**

- Distinct formal inputs may bind the same target `NodeArg`.
- A literal witness may satisfy multiple equal literal leaves.
- A target value may not simultaneously bind a formal input and witness a literal.
- A literal can never be a formal output.
- Internal pattern values map consistently to one target `NodeArg`; pattern
  operation nodes map injectively to target nodes.

### 3.3 Target graph contract

These are the complete assumptions on `Model`/`Graph`.

1. The graph is valid, acyclic, SSA, and already resolved. Every target node examined
   by the matcher must have non-null `Node::Op()`. An unresolved graph is rejected
   before candidate enumeration.
2. Extraction searches only the supplied graph scope. It neither descends into
   nested subgraphs nor spans lexical scopes.
3. The caller holds exclusive mutation access to the graph. The owning model's
   schema registries, model-local function table, opset imports, and initializers
   remain immutable for the entire call.
4. `FunctionExtractor` has no internal synchronization or concurrent-reuse guard.
   Callers must serialize access to an extractor and graph; concurrent callers use
   separate extractor instances and distinct graphs.
5. Every node mapped into an accepted match has an empty execution-provider
   assignment. Provider-assigned nodes elsewhere in the target graph do not by
   themselves reject snapshot construction.
6. Matched nodes have no control edges. Layering annotations across `R` must be
   identical; after `Graph::AddNode`, the call inherits that annotation through
   `Node::SetLayeringAnnotation`.
7. The target snapshot indexes both explicit consumers and implicit captures from
   every node's `ImplicitInputDefs()`. Either is an observable use for closure and
   revalidation.
8. A target initializer can witness a body literal only if it is constant and
   non-overridable. Literal witnesses remain in the target after extraction.
9. Before any mutation, the owning model must already resolve the exact function
   identity used by the extractor:
   - for a model-local function, an identical definition must already exist;
   - for a schema-defined function, the registry definition must match.
   Same identity with a different body/definition is an error. Neither
   `Extract(Model&)` nor `Extract(Graph&)` registers or replaces functions.
10. `Extract()` resolves the graph after every applied batch. Successful return
    guarantees a resolved graph at fixpoint.

**Public result.**

```text
FunctionExtractionResult {
  Status status;
  size_t replacements_applied;
}
```

Both `Extract(Model&)` and `Extract(Graph&)` return this result. The count includes
all calls added successfully, including earlier batches if a later batch fails.

**Error-state guarantee.**

- Validation, discovery, resource-limit, conflict-selection, or whole-batch
  prevalidation errors occur before that batch's first mutation.
- Once mutation begins, allocation failure, an implementation invariant failure,
  or `Graph::Resolve()` failure can leave the graph modified and possibly unresolved.
  The result reports the error and exact number of calls already added. No atomic
  rollback is promised.
- A caller requiring atomicity must invoke extraction on a cloned model and publish
  the clone only after success. Clone-and-swap is outside v2.

## 4. Pattern preparation and resolution

The constructor stores an owned copy of `F` and builds a context-free
`NormalizedFunctionPattern`, extending the existing v1 representation rather than
introducing a parallel `NormalizedFunctionProto` type:

1. Perform the checks in Section 3.2 that do not require a model.
2. Preserve ordered formal input/output slots.
3. Build a formal-attribute declaration table keyed by formal name. Each entry
   stores the formal name and resolved attribute type. Required types are inferred
   from occurrences; defaulted types originate in `attribute_proto` and must agree
   with all occurrences. The owned `FunctionProto` retains declaration kind and
   default values. Compile each valid `ref_attr_name` into a per-node
   `(operator-local name, formal ID)` occurrence; it is not part of the node's
   fixed-attribute fingerprint.
4. Normalize a body `Constant` with only fixed attributes into a literal descriptor.
   If a `Constant` has a formal-parameter variable, validate the context-free
   Section 4.3 criteria, set `is_parameterized_constant`, and retain it as a
   pattern operation node; reject any other variable/fixed mixture. It matches
   structurally only to a target `Constant`, never to an initializer.
5. Build a value table with independent flags (`formal input`, `formal output`,
   `literal`, `internally produced`) and a separate slot table for missing optional
   arguments.
6. Build producer/consumer tables and topologically sort retained pattern operation
   nodes.
7. Starting from every formal output, walk backward through pattern producers and
   mark reachable operation nodes. Reject unless the marked set equals the complete
   retained pattern operation-node set; this enforces Section 3.2's
   output-reachability restriction before any target enumeration.
8. Normalize syntactic data independent of schemas without guessing defaults.

The compiled representation is logically:

```text
FormalAttributePattern {
  formal_name;
  type;
}

PatternNode {
  attribute_variables[];  // (operator-local name, formal-attribute ID)
  is_parameterized_constant;
}

NormalizedFunctionPattern {
  owned FunctionProto;    // retains required/defaulted declarations and defaults
  formal_attributes[];
  values[];
  nodes[];
  reverse_topological_node_ids[];
  pattern_attribute_payload_bytes;
  max_attribute_bytes;
  construction_status;
}
```

Every formal appears exactly once in either `FunctionProto.attribute` or
`attribute_proto` and is referenced at least once. Numeric IDs connect occurrences
to formals after validation. The normalized pattern owns the source proto;
candidate states and replacement plans own canonical bound `AttributeProto`s.

At `Extract()` time, compile a `CompiledFunctionPattern` against the owning model's
schema registries, opset imports, and complete model-local function table:

1. Resolve every body call exactly as the target model does.
2. For a primitive/custom operator, record a stable resolved-schema identity:
   canonical domain, `op_type`, effective version, and registry/schema identity.
3. For a body node that resolves to another function, record the resolved function
   identity **and a canonical definition fingerprint**, not merely the op tuple.
   The fingerprint includes required/defaulted formal declarations, defaults, and
   body references. The target node must resolve to the same definition.
4. Normalize schema defaults and validate arity/type constraints.
5. Validate every attribute-variable occurrence against the resolved operator
   schema: its operator-local name exists, its declared type agrees with the schema,
   and any schema default is representable by the v2 equality rules.
6. Reject any operation outside the shipped pure-operator policy in Section 4.3.
7. Compute each output-producer's conservative root-index key using only invariants
   available before mapping:
   `(canonical domain, op type, overload, resolved version, input/output arity)`.
   The snapshot applies exact fixed-attribute matching after this coarse lookup.
   Attribute-variable values, formal-output types, and graph degrees are deferred
   to tuple matching; requiring them in the coarse index could miss valid boundary
   configurations.
8. Group formal outputs by producing pattern node and record each producing output
   slot. Candidate rarity is measured later from the target snapshot.
9. Store pattern operation nodes in reverse topological order. This order is
   used as the deterministic priority for the output-seeded worklist in Section 6.

The shipped implementation does not cache `CompiledFunctionPattern`: it recompiles
against the graph before discovery and again after each successful batch/resolve.
This preserves the context-free constructor without retaining stale schema or
model-local-function identities.

### 4.1 Attribute and literal equality

**ONNX binding model.** `FunctionProto.attribute` contains required formal names;
`FunctionProto.attribute_proto` contains concrete defaults for defaulted formals,
which therefore do not also appear in `attribute`. A body occurrence uses its
operator-local `name` and sets `ref_attr_name` to the formal. During function
specialization, a concrete call attribute overrides the formal default; otherwise
the default is used. The selected value replaces `ref_attr_name` while retaining
the occurrence's operator-local name. The compilation, matching, and call-emission
rules below are the inverse of that operation.

**Fixed attributes.** After operator-schema defaults are materialized, every
non-variable pattern attribute must equal the target's effective attribute exactly
by operator-local name, type, and value. After variable occurrences are consumed,
unexpected target effective attributes reject the match. Output-producer signatures
include normalized fixed attributes only.

**Effective target attributes.** For an operator-local attribute name, matching
deterministically uses:

1. the target node's explicit concrete attribute, if present;
2. otherwise the resolved target operator schema's concrete default; or
3. `Missing` if neither exists.

Resolve/materialize this view without changing the graph. A variable occurrence
requires a concrete effective target value. `Missing` rejects the candidate for
both required and defaulted formals. In particular, a function default by itself
does **not** prove that an omitted target attribute has that value: specialization
would materialize the default in the body. If target omission is backed by an
operator-schema default, bind that effective value. It may equal the function
default, or it may differ and become an explicit call-site override.

**CanonicalizeFormalAttribute(formal_name, declared_type, source).** This is the
single canonicalization operation used for defaults, match bindings, plan storage,
and call emission:

1. require `source.type == declared_type`;
2. copy only the active value field(s);
3. set `name = formal_name` and `type = declared_type`;
4. omit `ref_attr_name`, `doc_string`, and other non-semantic metadata.

Empty repeated values are concrete. Formal-binding equality compares only canonical
type and value. Scalar, repeated, string, and tensor equality use the same
representation-exact rules below; strings compare byte-for-byte.

| Supported enum | Active value field | Canonical equality |
|---|---|---|
| `FLOAT` | `f` | Exact normalized IEEE-754 bits. |
| `INT` | `i` | Exact signed integer. |
| `STRING` | `s` | Exact bytes. |
| `FLOATS` | `floats` | Same length and exact normalized bits elementwise. |
| `INTS` | `ints` | Same length and exact integers elementwise. |
| `STRINGS` | `strings` | Same length and exact bytes elementwise. |
| `TENSOR` | `t` | Exact logical tensor type, shape, and element bits. |
| `TENSORS` | `tensors` | Same length and exact logical tensor equality elementwise. |

For byte metering, use the canonical encoded size of `type` plus the active value
field(s), excluding `name`, `ref_attr_name`, and documentation. Encoded repeated-
element framing counts, so long lists of empty strings or empty tensors still
consume budget.

**Floating values.** Equality is representation-exact after storage normalization:
signed zero remains distinct, and NaNs compare equal only when their normalized bit
patterns (including payload/sign) match. No numeric tolerance is used.

**Tensor literals.** Compare data type, rank, concrete dimensions, and logical
element bit patterns exactly after normalizing typed-field versus `raw_data`
encoding. External-data body literals are rejected. Target external initializers
may be read only within the configured byte budget and must yield the same logical
bits.

### 4.2 Type compatibility

Types are a rejection predicate, not a substitute for structural equivalence:

- known tensor element types must be identical;
- if both ranks are known, ranks must be identical;
- corresponding concrete dimension values must be identical;
- symbolic dimension names carry no matching significance;
- an unknown rank/dimension does not reject an otherwise valid match;
- conflicting known container/value kinds reject.

Resolved schema identity remains the semantic source of truth.

### 4.3 Purity source of truth

ONNX schemas do not provide a complete stable side-effect/nondeterminism flag.
The shipped policy allowlists these standard ONNX operations:
`Identity`, `Add`, `Sub`, `Mul`, `Div`, `Relu`, `Cast`, `MatMul`, `Transpose`,
`Reshape`, `Clip`, `Concat`, `MaxPool`, and `LeakyRelu`. Unknown primitive/custom
operators are denied. Resolved model-local or schema functions are allowed only
when their complete transitive bodies resolve to this policy (or other transitively
pure functions), are non-recursive, and contain no graph-valued attributes.

**Parameterized `Constant` admission.** v2 adds one explicit, narrow purity case;
it does not add `"Constant"` unconditionally to the general string allowlist. A
body `Constant` is retained and admitted as a structural pattern operation only
when all of the following hold:

1. it resolves to the standard ONNX-domain `Constant` schema selected by the
   function's opset import, has zero inputs and one output, and is deterministic;
2. it has exactly one attribute, that attribute is a variable occurrence, and its
   operator-local name/type is one of:
   `value`/`TENSOR`, `value_float`/`FLOAT`, `value_floats`/`FLOATS`,
   `value_int`/`INT`, `value_ints`/`INTS`, `value_string`/`STRING`, or
   `value_strings`/`STRINGS`;
3. a `value` tensor is dense, inline, and within the attribute payload budget;
   sparse, external-data, graph, type-proto, and other value forms are rejected;
4. its output is internal, backward-reachable from a non-`Constant` formal-output
   producer, and is not itself a formal output.

Normatively, the v2 allowed set is the v1 set plus
`(standard ONNX Constant schema identity, is_parameterized_constant == true)`.
There is no plain op-type-only `"Constant"` entry.

The normalizer marks this classification before fixed-`Constant` folding and does
not create a literal descriptor. Per-model compilation admits it through an
`is_parameterized_constant` special case after the resolved-schema and criteria
checks; all other structurally retained `Constant`s fail purity validation. The
matcher then treats it like any other forced node pair: the target must be a
`Constant` with the same resolved schema, arity, and output slot, and
`CheckOrBindAttribute` must consume the same operator-local value attribute.
Unexpected target attributes reject. It never matches an initializer. This makes
the admission explicit without broadening the v1 literal or purity rules.

### 4.4 Model-local function schema registration

`IOTypeConstraintHelper` in `onnxruntime/core/graph/function_utils.cc` uses two
separate registration loops:

1. A referenced name in `FunctionProto.attribute` has its type inferred from body
   occurrences and is registered with
   `OpSchema::Attr(name, description, type, /*required=*/true)`. Calls that omit it
   fail schema validation.
2. Each `FunctionProto.attribute_proto` entry is validated as a unique concrete
   default whose type agrees with its occurrences, then registered as an optional
   `OpSchema::Attribute` carrying that default. Omission uses the default; an
   explicit call value overrides it.

The helper also rejects required/default overlap and duplicate defaults. For
compatibility with existing model-local functions, an **unreferenced** required
declaration has no inferable operator-attribute type and is skipped by schema
registration. This differs from `FunctionExtractor` pattern validation, which
rejects every unreferenced formal attribute.

This shared registration behavior lets `Graph::Resolve()` validate emitted
FunctionExtractor calls without adding any dynamic function-registration API.

## 5. Target snapshot

Construct one read-only snapshot from a `GraphViewer`:

1. Capture topological `NodeIndex` order and position.
2. Validate that every target node has a resolved schema. Provider assignment and
   control-edge constraints are checked only when a node is mapped or prevalidated.
3. Index producers by `NodeArg`.
4. Index explicit consumers from normal input edges/lookups.
5. Independently scan **every target node's `ImplicitInputDefs()`** and build
   `NodeArg -> implicit consumer NodeIndex` entries.
6. Index graph outputs, constant/non-overridable initializers, control-edge nodes,
   topological positions, and compatible root candidates. Layering annotations are
   read from mapped nodes during candidate validation.

The snapshot is immutable and invalid after any graph mutation.

## 6. Output-rooted reverse-topological matching

### 6.1 Enumerate output-root tuples

Matching begins at the function's formal outputs, not at an arbitrary internal
node.

1. For each distinct formal-output producer group, build a candidate list in target
   topological order. A coarse key uses canonical domain, op type, overload,
   resolved version, and input/output arity; `NodeSignatureMatches` then checks
   effective fixed attributes and rejects unexpected non-variable target
   attributes. Formal-output slots and known value types are checked when seeding
   and scheduling the tuple. Variable values and target predecessor/consumer counts
   are excluded from the index: a formal input may have an external producer and a
   formal output may have arbitrary external fan-out.
2. If several formal outputs are produced by different slots of that same pattern
   node, binding the node fixes all of those target output slots; do not enumerate
   them independently.
3. Sort producer groups by `(candidate-list size, group index)`. The smallest list
   becomes the primary group, and the deterministic recursive product follows this
   order. Reject tuples that reuse one target node for distinct producer groups.
4. Each product element is one **output-root tuple** containing a proposed target
   value for every formal output. Seed all formal-output bindings together.

This outer tuple enumeration is necessary for branched functions whose separate
output-producing nodes share an upstream value: walking backward from one output
cannot discover a sibling consumer. It is bounded by the output-root-tuple resource
limit. It is candidate enumeration, not backtracking inside the matcher.

### 6.2 Deterministic worklist

For each output-root tuple, create a fresh `MatchState`:

- pattern operation node to target `NodeIndex`;
- reverse target-node map for injectivity;
- pattern value to target `NodeArg*`;
- per-pattern-value state: `Unseen`, `Scheduled`, or `Processed`;
- ordered formal-input bindings;
- formal attribute parameter to `Unbound` or canonical concrete binding;
- literal witnesses; and
- a worklist of required `(pattern value, target value)` bindings.

Use one `Schedule(pattern_value, target_value)` operation:

1. If the value is `Unseen`, bind it to `target_value`, mark it `Scheduled`, and
   enqueue it.
2. If it is already `Scheduled` or `Processed`, require the identical target
   `NodeArg` and return without enqueueing or expanding it again. Disagreement fails
   the whole output-root tuple.

Seed the worklist by calling `Schedule` for all ordered
formal-output bindings implied by the producer-group tuple. To select the next
value, scan pattern nodes in reverse topological order, checking their output slots
then input slots; fall back to pattern value ID order.

For each popped `(pattern_value, target_value)` requirement:

1. Require state `Scheduled`, mark the value `Processed`, and retain the target
   binding established by `Schedule` **before** expanding its producer.
2. If it is a formal-input leaf, record/check that formal-input boundary binding and
   terminate this branch. The leaf matches whatever target value the forced walk
   reaches; its producer is not absorbed.
3. If it is a literal, validate and record the target initializer or target
   `Constant`-output witness, enforce the alias rule, and terminate this branch.
4. Otherwise find the unique pattern producer and its output-slot index. Require
   `target_value` to have a unique target producer at that same output slot.
5. Bind/check the producer-node pair. Compare resolved operator/function identity,
   fixed normalized attributes, arity, output-slot index, type compatibility,
   provider, annotation, and control-edge constraints. For each attribute-variable
   occurrence on the pattern node, execute `CheckOrBindAttribute` below. Enforce
   reverse node-map injectivity.
6. For each producer input slot in increasing positional order:
   - pattern `""` requires target `""`;
   - otherwise call `Schedule(pattern_input_value, target_input_value)`.
   Variadic inputs retain their declared order.
7. Schedule every non-missing output slot of a mapped producer. This enforces
   consistency when multiple formal outputs or branches converge on one node and
   validates secondary outputs positionally.

Every required pattern value in the formal-output backward closure is expanded at
most once, so diamonds do not re-expand a shared producer cone. Repeated scans for
the next scheduled value are charged to the aggregate work budget. Secondary
non-missing outputs of a mapped node are scheduled even when they do not lead to a
formal output. When the worklist
becomes empty, require every pattern operation node to be mapped and every declared
formal attribute parameter to be `Bound`. The structural candidate succeeds;
Section 7 then checks additional target consumers, closure, and convexity. Any
mismatch above fails the **entire output-root tuple** immediately, and the outer
loop tries the next tuple.

### 6.3 Deterministic attribute-variable binding (v2)

`CheckOrBindAttribute(formal, pattern_occurrence, target_node)` is normative:

1. Consume one shared `aggregate_work_units` unit **before** attribute lookup,
   canonicalization, or comparison, using Section 13's pre-increment limit rule.
2. Look up the one effective target attribute named by
   `pattern_occurrence.name` using Section 4.1.
3. If it is `Missing`, fail the whole output-root tuple. There is no branch that
   guesses the function default.
4. Compute the remaining attribute-byte budget and apply
   `CanonicalizeFormalAttribute` from Section 4.1 under that limit.
5. After canonicalization succeeds, cumulatively charge the canonical payload size.
6. If the formal is `Unbound`, store that canonical value and mark it `Bound`.
7. If it is already `Bound`, require exact type-and-value equality with the stored
   binding. Agreement returns without changing state; disagreement fails the tuple.

The same formal may occur on multiple attributes of one node or on multiple body
nodes. All occurrences share this one match-local binding. Different candidate
matches have independent maps and may bind the same formal to different values.
A declared function default is retained as declaration metadata and validated at
compile time, but it neither pre-binds nor constrains the variable: an explicit
call-site value may override it.

### 6.4 Why v2 still has no backtracking

**Invariant.** ONNX SSA gives every internal pattern value exactly one producer.
The target graph is also SSA, and v2 maps inputs positionally with no commutative
permutations. Therefore, after formal-output target values are seeded:

- each non-leaf value forces one pattern producer/output slot;
- its bound target value forces one target producer/output slot; and
- each producer input slot forces exactly one next value binding.

Attribute variables do not change this property. The already-forced node pair
provides exactly one effective target attribute: first occurrence binds and later
occurrences either agree or reject. Function defaults do not create alternatives.
Therefore there are no alternative bindings and hence no choice points, undo log,
checkpoint, or backtracking in the v2 core loop. A consistency, identity,
attribute, arity, slot, type, or injectivity failure rejects the tuple rather than
exploring another path. Alternative target output roots are tried only by the
deterministic outer enumeration in Section 6.1.

Target tensor names are irrelevant.

## 7. Candidate validation

### 7.1 Closure

For each input of `R`, its target value must be produced by:

- another node in `R`;
- a bound formal input source;
- a recorded literal witness; or
- an omitted optional slot.

For each output of `R`:

- a boundary output may have arbitrary explicit/implicit outside consumers and may
  be a graph output because the replacement call reuses that `NodeArg`;
- every private intermediate must have all explicit consumers in `R`, have **no
  implicit consumers outside `R`**, and not be a graph output.

An unmatched `If`, `Loop`, `Scan`, or other node that captures a private
intermediate through `ImplicitInputDefs()` therefore rejects the match even when
`GetConsumerNodes()` reports no ordinary consumer.

Do not promote an externally used private intermediate to a new call output. That
would change the supplied function signature; reject the candidate.

### 7.2 Convexity

Closure and convexity are separate requirements. Reject if any data-flow path leaves
`R` and later re-enters `R`. In particular, a formal-input binding may not be
transitively downstream of an output of `R`.

Run a visited-set BFS from explicit and implicit consumers outside `R`, following
their outputs to further consumers. Encountering any node in `R` proves
non-convexity. The aggregate work budget bounds this traversal.

### 7.3 Final semantic gates

Accept only if:

- ordered call inputs equal formal-input bindings;
- ordered call outputs are the exact target `NodeArg`s bound to formal outputs;
- every declared formal attribute has one canonical concrete binding and the
  binding equals every recorded target occurrence;
- registration and resolved callee identity remain exact;
- every literal witness still has the required value;
- `|R| > 1`;
- all resource budgets remain within limits.

Reusing formal-output `NodeArg`s preserves names, types, graph-output identity, and
downstream uses. Because literal-derived formal outputs are forbidden, no target
initializer or `Constant` is already a competing producer for a call output.
Attribute binding neither changes `R` nor creates graph edges, so connectedness,
output reachability, closure, convexity, and boundary-conflict rules are unchanged
from v1.

## 8. Literal-witness lifecycle

This inherited v1 lifecycle applies only to fixed-attribute body `Constant` nodes.
A parameterized `Constant` remains a structural node in `R` and does not use a
literal witness.

The fixed-literal lifecycle is deliberately simple and complete:

1. **Compile:** each admitted fixed-attribute body `Constant` becomes a literal
   descriptor; it is not a pattern operation node.
2. **Match:** a consuming pattern slot binds to a target constant initializer or
   target `Constant` output with identical logical bits.
3. **Plan:** record the pattern value ID, target `NodeArg*`, whether the witness is
   an initializer, and the target `Constant` node index when applicable. No value
   fingerprint is stored, and the witness is not added to `R`.
4. **Conflict selection:** shared witnesses never conflict by themselves.
5. **Prevalidation:** confirm the witness still exists, is non-overridable when an
   initializer, and still has identical value.
6. **Apply:** the call has no input for the literal; the function body owns it.
   Edges from the witness to nodes in `R` disappear when those consumers are
   removed.
7. **Afterward:** leave both initializer and `Constant` node witnesses in place,
   whether shared or newly dead. General dead-code/dead-initializer cleanup is a
   separate pass. Batch resolution passes initializer witness names through
   `Graph::ResolveOptions::initializer_names_to_preserve`.

This policy can leave dead constants but cannot create duplicate producers or delete
a value still used elsewhere. A body literal may never produce a formal output or
alias a formal input.

## 9. Global selection, prevalidation, and fixpoint

Sort accepted plans by primary output-root topological position and then target node
indexes.
Select a deterministic maximal set.

Two plans conflict when:

1. their `R` sets intersect; or
2. a boundary output of either plan is a boundary input of the other.

The second rule treats node-disjoint but data-flow-adjacent occurrences as conflicts.
The earlier plan in deterministic order is selected; the other is rediscovered in
the next pass. Sharing an unrelated formal input or literal witness is not a
conflict.

Before the first mutation of a selected batch, prevalidate **all** selected plans:

- referenced nodes and `NodeArg`s still exist;
- each recorded target attribute occurrence still canonicalizes to its plan-owned
  canonical value and still equals the emitted formal binding;
- explicit and implicit consumer indexes still match;
- closure/convexity still holds and each literal witness re-compares equal to the
  normalized pattern literal;
- selected plans are pairwise nonconflicting;
- exact call registration and resolved identity are available;
- unique node names are chosen and verified unused;
- all ordered inputs/outputs, edges to remove/add, annotations, and removal order are
  materialized;
- the exact formal-name-keyed call-attribute set is materialized with no
  `ref_attr_name`;
- every plan has `|R| > 1`.

A stale plan aborts the batch before mutation; it is never silently skipped.

Each immutable `ReplacementPlan` stores:

```text
call_attributes: formal parameter name -> canonical concrete AttributeProto
attribute_occurrences:
  (target NodeIndex, operator-local attribute name,
   FormalAttributeId, owned canonical AttributeProto)
```

This data is copied into the plan; it must not point into mutable target-node
attribute storage.

Apply plans in reverse topological order. Each adds one node and removes `|R| > 1`
nodes, so total graph node count strictly decreases. Literal witnesses do not affect
the measure because they are not removed.

Resolve after the batch, rebuild the snapshot, and repeat until no match is accepted.
The defensive pass cap is the node count at method entry. Reaching it is an internal
invariant error, not partial success, because strict node-count decrease should make
it unreachable.

## 10. Applying one replacement

For each fully prevalidated plan:

1. Remove outgoing edges from nodes in `R` in reverse topological order, then
   remove the nodes themselves in reverse topological order.
2. Add a uniquely named node with:
   - `op_type = F.name`;
   - `domain = F.domain`;
   - `overload = F.overload`;
   - ordered bound formal inputs;
   - exact existing formal-output `NodeArg`s;
   - exactly one concrete call attribute for **every** declared formal attribute
     parameter, keyed by the formal parameter name;
   - no fixed body attributes and no unrelated anchor attributes;
   - the common layering annotation.
3. Re-add explicit input edges from producers outside `R` to the corresponding call
   input slots, and explicit output edges from call output slots to outside
   consumers, using the precomputed general-DAG edge plan. Do not assume one
   “first” and “last” node as chain-oriented fusion helpers do.
4. Do not remove or rename literal witnesses or initializers.

The emitted attribute set is deliberately explicit and canonical: required and
defaulted formals are both emitted, including bindings equal to declared defaults.
Each entry is already the output of `CanonicalizeFormalAttribute` in Section 4.1
and is copied without recanonicalizing. The operator-local body attribute name is
never used on the call. Thus the call alone records the exact specialization that
was matched and does not depend on a default remaining unchanged. `NodeAttributes`
is a map, so declaration order is not semantically significant.

Attribute bindings have no graph identity and add no mutation dependency between
plans. Node-disjoint matches may bind different values and each emitted call keeps
its own immutable set. Existing node/boundary conflict rules remain unchanged.

After the entire selected batch, call `Graph::Resolve()`. Continue only if it
succeeds. Successful method return always means the final graph is resolved.

All expected semantic failures occur before step 1. From the first edge removal
onward there is no rollback. Node removal, allocation, edge reconstruction, or
resolution can fail after mutation; Section 3.3 defines the resulting non-atomic
error state.

## 11. Worked example

Function body with two inputs, one private intermediate, one literal, and two
outputs:

```text
Function Pair(A, B) -> (Y, Z)

                 A ----\
                        Add ---- t ---- Mul ---- Y
                 B ----/          \      /
                                    \   c=Constant(2)
                                     \
                                      Relu ---- Z
```

`t` is the one private intermediate. Pattern operation nodes are `Add`, `Mul`, and
`Relu`; the body `Constant(2)` is a literal descriptor.

Target before:

```text
x1 ----\
        Add[n10] -- v -- Mul[n11] -- y ----> outside consumers
x2 ----/             /       \
            initializer two   (none)
                  |
                  +-------- value 2

v ------------------- Relu[n12] -- z ----> graph output
```

Bindings and plan:

```text
A -> x1                 Y -> y
B -> x2                 Z -> z
t -> v                  literal 2 -> initializer "two"
R = {n10, n11, n12}
call inputs  = [x1, x2]
call outputs = [y, z]
```

Traversal is deterministic:

1. Output-root enumeration proposes the tuple `(Y -> y, Z -> z)`, whose producers
   are `n11` and `n12`.
2. The reverse-topological worklist verifies `Mul[n11] -> Y` and
   `Relu[n12] -> Z`, pushing `(t -> v)` from both branches plus the literal
   requirement `(2 -> initializer "two")`.
3. The duplicate `(t -> v)` requirements agree and coalesce. Its unique producers
   force `Add -> n10`, which positionally pushes `(A -> x1)` and `(B -> x2)`.
4. `A` and `B` are formal-input leaves, so both branches terminate. The empty
   worklist yields the candidate; closure and convexity then validate it.

No alternative edge or input binding is explored. If, for example, the `Z` root
were produced by a different `Relu` whose input was not `v`, the duplicate
requirement would disagree and that entire output-root tuple would fail.

Target after:

```text
x1 ----\
        Pair(x1, x2)[n13] ---- y ----> same outside consumers
x2 ----/                 \
                          z ----> same graph output

initializer "two" remains and may later be removed by generic cleanup.
```

The call has no input for `two`; `Constant(2)` is owned by `Pair`'s body.

**Rejection variant.** If an unmatched node also consumes private value `v`, or an
unmatched control-flow node lists `v` in `ImplicitInputDefs()`, `v` is externally
observable. Closure rejects the candidate. Declaring only `Y` and `Z` does not
authorize the extractor to add `v` as a third function output.

### 11.1 v2 attribute-parameter example

```text
Function TwiceLeaky<required slope>(X) -> Y
  t = LeakyRelu<alpha = @slope>(X)
  Y = LeakyRelu<alpha = @slope>(t)
```

For a target pair whose effective `alpha` is `0.2` on both nodes, the forced
reverse walk first binds `slope -> FLOAT(0.2)` and the second occurrence agrees.
The replacement is `TwiceLeaky<slope = 0.2>(X) -> Y`. If the second effective
value is `0.3`, the tuple fails; the matcher does not split the one function
parameter into two values.

If `slope` instead has a function default of `0.01` and a target node omits
`alpha`, the matcher inspects `LeakyRelu`'s resolved operator-schema default. A
concrete effective `0.01` binds successfully and the call still emits
`slope = 0.01`. If the operator-schema default is `0.02`, the binding is `0.02`
and the emitted call explicitly overrides the function default. If the target
operator has no concrete default, omission is `Missing` and the candidate fails.

## 12. Correctness conditions

Replacement is semantics-preserving only when:

1. **Resolved-node equivalence:** each mapped operation resolves to the same schema
   or exact nested-function definition and has equal fixed effective attributes.
2. **Output reachability:** compilation proves every pattern operation is an
   ancestor of at least one formal output, so the output-rooted traversal covers the
   entire admitted body.
3. **Edge equivalence:** ordered slots, repeated uses, and omitted optionals agree.
   Reverse-topological worklist consistency proves that all formal-output roots
   induce one mutually consistent mapping.
4. **Input equivalence:** every formal receives exactly the target value used by the
   matched body.
5. **Literal equivalence:** every function-owned literal has an exact target witness.
6. **Attribute instantiation equivalence:** every formal attribute is bound from a
   concrete effective target value, all its occurrences agree exactly, and the
   call emits that binding under the formal name. Specializing the call therefore
   reconstructs each operator-local attribute value.
7. **Output equivalence:** the call reuses every matched formal-output `NodeArg`.
8. **Producer uniqueness:** no formal output is literal-derived, so adding the call
   cannot coexist with an initializer/`Constant` producer for that output.
9. **Closure:** private intermediates have no explicit or implicit external observer.
10. **Convexity:** no outside computation lies on a path between nodes in `R`.
11. **Scope equivalence:** no match crosses lexical graph scopes.
12. **Registration equivalence:** the new call resolves to the exact `FunctionProto`
    used for matching.
13. **Scheduling equivalence:** `R` excludes provider assignments and control edges
    and preserves common layering annotation.

Under these conditions, inlining the new call reconstructs a graph isomorphic to
the removed region, modulo local names and constant materialization. Extract-then-
inline round-trip equivalence is the primary correctness oracle.

## 13. Resource limits

Adversarial or accidental large patterns must not cause unbounded work.

**Requirement.** Expose configurable budgets with conservative defaults:

| Budget | Default |
|---|---:|
| Total `FunctionProto` body nodes, including `Constant` | 1,024 |
| Target nodes in one scope | 1,000,000 |
| Root-candidate entries and output-root tuples | 100,000 |
| Aggregate normalization/enumeration/matcher work units (`max_worklist_bindings`) | 1,000,000 |
| Total literal bytes compared per extraction pass | 64 MiB |
| Formal attribute parameters | 256 |
| Pattern declaration/default plus target attribute bytes inspected/compared per discovery pass | 64 MiB |
| Nested target recursion depth | 0 (unsupported) |

Exceeding a budget returns a `FAIL` status before the current batch mutates.
Identity-only input ordering means each tuple is verified without search or
permutation explosion.

`max_worklist_bindings` is the public field name, but its scope is aggregate work,
not only successful bindings. Matcher single-unit counters use:

```text
ConsumeWorkUnit(counter, limit):
  if counter >= limit: return FAIL
  ++counter
```

The check occurs before the work; zero permits none. Construction-time pattern
normalization uses equivalent overflow-safe checked addition and charges each
formal declaration, each default canonicalization, each body attribute-variable
occurrence, and existing node/slot work. During each discovery pass, the cumulative
`aggregate_work_units` counter is shared by root enumeration, tuple construction,
worklist scheduling/scanning, and matching. Every invocation of
`CheckOrBindAttribute` consumes one unit before lookup, including first bindings,
repeated consistency checks, missing attributes, and ultimately rejected tuples.
Thus many tiny occurrences cannot bypass the aggregate cap.

Attribute byte accounting uses the canonical encoded size defined in Section 4.1.
During construction, accumulate `pattern_attribute_payload_bytes` **before**
copying/materializing each item. It includes:

- each required/defaulted formal declaration's encoded name and type;
- every `attribute_proto` canonical default value;
- every occurrence's encoded operator-local name, `ref_attr_name`, and type.

Occurrence count is independently charged as work units, so zero-length payloads
still consume budget. Use overflow-safe checked addition for every component:

```text
if counter > max_attribute_bytes or
   payload_bytes > max_attribute_bytes - counter:
  return FAIL
counter += payload_bytes
```

At the start of every discovery pass, initialize
`attribute_payload_bytes_inspected = pattern_attribute_payload_bytes`; this charges
all declaration, default, and occurrence metadata against every pass and guarantees
an oversized declaration/default fails before target enumeration. For each concrete
effective target attribute, `CheckOrBindAttribute` first canonicalizes under the
remaining byte budget, then cumulatively adds the resulting canonical payload size,
then performs first-binding or consistency comparison. The counter is cumulative
across first bindings, repeated comparisons, and accepted/rejected tuples.
Parameterized-`Constant` values follow the same rule.
`max_formal_attributes` is checked during declaration-table construction.

Per-model compilation separately starts from `pattern_attribute_payload_bytes` and
charges concrete resolved operator-schema defaults for variable occurrences before
accepting the compiled pattern.

## 14. Implementation map

| File | Shipped responsibility |
|---|---|
| `include/onnxruntime/core/optimizer/function_extractor.h` | Public options, result, constructor, and `Extract(Model&)` / `Extract(Graph&)`. |
| `onnxruntime/core/optimizer/function_extractor_pattern.{h,cc}` | Context-free normalization, resolved pattern compilation, purity, registration validation, literal/attribute canonicalization, and pattern-side budgets. |
| `onnxruntime/core/optimizer/function_extractor_matcher.{h,cc}` | Immutable target snapshot, root enumeration, deterministic matcher state, closure/convexity, conflict selection, replacement plans, and whole-batch prevalidation. |
| `onnxruntime/core/optimizer/function_extractor.cc` | Fixpoint orchestration, mutation, boundary-edge reconstruction, graph resolve, and result accounting. |
| `onnxruntime/core/graph/function_utils.cc` | Generated model-local function schema registration for required and defaulted attributes. |
| `onnxruntime/test/optimizer/function_extractor_test.cc` | Unit and end-to-end coverage. |

The utility is excluded from minimal builds. It is independent of
`GraphTransformer::ApplyImpl`; callers invoke it directly.

## 15. Verification coverage

`function_extractor_test.cc` covers the shipped contract with focused tests for:

- invalid formals, attributes, data flow, disconnected/output-unreachable bodies,
  constant outputs, one-operation patterns, unsupported features, registration
  mismatches, unresolved graphs, purity, and all resource budgets;
- linear, branched, multi-output, diamond, aliased/repeated-input, optional,
  variadic, omitted-output, type-compatible, and deterministic root matching;
- exact operator/effective-attribute/literal-bit equality, including signed zero
  and NaN payloads;
- required/defaulted attribute binding, empty repeated values, repeated-occurrence
  consistency, missing effective attributes, independent per-match bindings, and
  exact emitted call attributes;
- parameterized `Constant` structural matching, initializer non-matching, and
  rejection of a custom schema masquerading as standard ONNX `Constant`;
- generated model-local schema behavior for missing required attributes, omitted
  defaulted attributes, explicit overrides, and compatible unreferenced required
  declarations;
- literal witnesses from initializers and `Constant` nodes, alias rules, retained
  shared/dead witnesses, and overridable-initializer rejection;
- closure, graph outputs, implicit captures, convexity, provider/control/annotation
  restrictions, and graph-scope boundaries;
- disjoint batches, deterministic overlap selection, boundary-adjacent deferral,
  shared unrelated boundaries, stale node/attribute plans, pass-cap failure,
  non-atomic resolve failure, strict node-count decrease, output identity/fan-out,
  fixpoint resolution, serialize/reload persistence, and extract-inline round trips.

## Appendix A. ORT and ONNX implementation references

| Concern | Stable symbol/file reference |
|---|---|
| Node identity, inputs/outputs, attributes, explicit edges, implicit inputs | `Node` in `include/onnxruntime/core/graph/graph.h` |
| Graph iteration, initializers, outputs, mutation | `Graph` in `include/onnxruntime/core/graph/graph.h` |
| Stable read-only traversal | `GraphViewer` in `include/onnxruntime/core/graph/graph_viewer.h` |
| Producer/consumer lookup | `Graph::GetProducerNode`, `Graph::GetConsumerNodes` |
| Mutation | `Graph::AddNode`, `Graph::RemoveNode`, `Graph::AddEdge`, `Graph::RemoveEdge` |
| Existing fusion helpers | `graph_utils::FinalizeNodeFusion` in `onnxruntime/core/graph/graph_utils.*` |
| Selector/action precedent | `NodeSelector`, `NodesToOptimize`, `ReplaceWithNew` under `onnxruntime/core/optimizer/selectors_actions/` |
| Function specialization/inlining | `function_utils::Specialize`, `Graph::InlineFunctionProto` |
| Function structure and identity | `FunctionProto` in `cmake/external/onnx/onnx/onnx-ml.proto` |
| Model-local function ownership | `Model::model_local_functions_`, `ModelProto.functions` |

`FunctionProto` contains ordered `input`/`output`, body `node`, required/defaulted
attributes, `opset_import`, `domain`, `name`, `overload`, and optional `value_info`.
A function call is identified by `(domain, name, overload)`.

## Appendix B. Future extensions

The following are intentionally non-normative:

- schema-reviewed commutative input permutations. Such a matcher introduces the
  only input-binding choice points: it must use bounded search with undo
  checkpoints/backtracking scoped to permutation alternatives. The v2 positional
  worklist remains backtrack-free;
- recursive bottom-up extraction in nested graph scopes;
- post-partition extraction with provider capability proof;
- removal of exclusive literal witnesses, including initializer-definition removal;
- constant-derived or passthrough formal outputs with explicit producer transfer;
- disconnected function components;
- sequences, maps, optionals, sparse/external literals, and control-flow bodies;
- clone-and-swap atomic extraction;
- runtime-optimization serialization through selector/actions.
