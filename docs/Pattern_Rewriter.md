# Writing C++ graph rewrite rules

The experimental `onnxruntime::rewrite::PatternRewriter` replaces connected
operator patterns with semantically equivalent graphs. Include
`core/optimizer/pattern_rewriter.h`. Construct `Pattern` and `Replacement` from
text; programmatic builders and mutable replacement recipes are not public APIs.
Check-and-bind callbacks and replacement callbacks remain ordinary C++ functions.

A `Rule` takes `(name, pattern, replacement, check_and_bind, opset_requirements)`.
The pattern may be text or a parsed `Pattern`; the replacement may be text,
a parsed `Replacement`, or a callback returning one. Text accepts string literals,
`std::string`, and `std::string_view` and is parsed once during rule construction;
syntax errors are reported immediately. Parsed definitions own their data, so
the original text need not outlive the rule. Rules remain copyable and movable.
The matcher establishes structural and deletion safety. **The rule author must establish
semantic equivalence, supported types, and kernel compatibility.**

## Composable Mobius MoE experiments

`MakeMoEFusionRules()` reimplements the
[GroupedMatMul MoE experiment](https://github.com/gramalingam/onnxruntime/pull/11)
using `PatternLibrary`, and adds a bounded dense-source rule described below.
These are **opt-in experiments**, not a generally equivalent or production-ready
MoE optimizer. They are not registered as default graph transformers.

```cpp
rewrite::PatternRewriter rewriter(rewrite::MakeMoEFusionRules());
rewrite::Statistics statistics;
ORT_THROW_IF_ERROR(rewriter.Run(graph, statistics));
ORT_THROW_IF_ERROR(graph.Resolve());
```

Apply it to a resolved graph before function inlining or generic
MatMul/activation fusions. The source uses `local.moe.GroupedMatMul` version 1,
a model-local ONNX function implementing the bias-free
[GroupedMatMul RFC](https://github.com/onnx/onnx/pull/8193): inputs `[M,H]`,
weights `[E,H,I]`, indices `[M,k]`, output `[M,k,I]`. The rule assumes this
trusted local function's semantics; the function name is not a standard ONNX
schema or a validation of arbitrary model-local function definitions.

One rule composes the following named patterns rather than constructing
different pattern strings for each routing choice:

- `Routing` is the common logits/top-k/weights/indices tensor contract.
  `RoutingForms = TopKSoftmax | SoftmaxTopK | NormalizedSoftmaxTopK` recognizes
  `Softmax(TopK(logits))`, `TopK(Softmax(logits))`, and the latter divided by the
  sum of selected probabilities. `NormalizedSoftmaxTopK` calls `SoftmaxTopK`
  directly, passing the same immutable `top_k` interface input explicitly.
- `Experts` is implemented by `GroupedExperts`, with explicit inputs for
  gate/up/down weights and reshape constants. It calls the ordinary `SwiGLU` body for the
  gated `Swish(alpha=1)` activation. `MoEPattern` calls the routing and expert bodies,
  weights the expert outputs, and reduces over selected slots.

This library uses PascalCase for all interface and pattern definitions
(bodies and alternatives alike), and snake_case for tensor and instance names.
Capitalization is a local naming convention, not a distinction in DSL semantics.
`weights, indices = RoutingForms(logits, top_k)` binds the two output tensors
directly to local names, in interface output order. The checker reads them with
`match.valueOf("weights")` and `match.valueOf("indices")`, without a routing
instance view.
`GroupedExperts` receives every interface input explicitly at its call site.

Handwritten `RoutingView` reads directly bound tensors from `Match`, while
`ExpertsView` uses scoped `BindingView` lookups for the named expert instance.
The matched weights producer distinguishes unnormalized
TopK weights from the two normalized forms. Shared C++ checks decode immutable
int64 tensors with `ReadConstantTensor`, propagating malformed-payload errors.
They require opset 24 float32, rank-2 token inputs, positive static expert,
hidden and intermediate dimensions, matching token dimensions proven by equal
positive integers or the same nonempty symbolic name, and static `1 <= k <= E`.
Anonymous unknown dimensions are not equality evidence. Flatten/regroup shapes
must be `[-1,I]` and `[-1,k,H]`; down-projection indices must flatten the same
selected indices to `[-1,1]`.

The checker binds `k` and the normalization attribute with `Match::update`.
One pre-parsed replacement references scoped expert leaves, transposes and
interleaves gate/up rows to `[E,2*I,H]`, and emits
`com.microsoft.MoE(activation_type="swiglu", swiglu_fusion=1)`. It passes raw
logits and sets `normalize_routing_weights=1` for the first and third routing
forms, otherwise zero. Constant folding may evaluate weight packing.
The layout targets CPU MoE; it is not a portable CUDA/other-EP lowering.
The default rewriter provider and deletion-boundary checks reject incompatible
assigned providers, external consumers and exposed intermediate graph outputs.
Quantized experts, biases, clipped activations and other routing/layout choices
are deliberately out of scope.

**Grouped-rule numerical limitations:** CPU MoE's expert selection does not implement ONNX
TopK's lower-index tie preference. Full-softmax versus selected-softmax
evaluation and normalization can also differ through rounding, underflow,
overflow and nonfinite values. These rules do not prove absence of routing
ties or numerical equivalence; graph resolution is not an execution comparison.
Use only as an explicit experiment with trusted inputs/functions and separate
target-specific numerical validation, not as a general-purpose optimization.

The three `test/testdata/transform/fusion/mobius_moe_*.onnx` fixtures and adjacent
`generate_mobius_moe.py` are preserved from the original experiment. The
generator records Mobius revision `b1e9c8e9dec1dfa206183f957e77d07cfb734df2`,
uses real Mobius routing with deterministic synthetic weights and a grouped
expert body, embeds the local function, and downloads no checkpoints.
`MoEFusionRulesTests.*` checks all routing forms and different expert counts,
raw-logit and attribute preservation, resolve/reload/idempotence, explicit routing
output bindings, scoped expert views, and rejection/error cases in indexed and reference modes.
The grouped-fixture cases remain structural tests; their end-to-end numerical comparison is deferred.
The CPU packing regression separately constant-folds the generated weight
subgraph with distinct gate/up/down entries and checks every interleaved FC1
and transposed FC2 element in both matcher modes.

### Bounded dense two-expert Mobius lowering

`DenseTwoExpertMoE` recognizes the standard-op skeleton of Mobius
`TopKGate` + bias-free SiLU `MLP` + `MoELayer.forward`: `TopK` followed by
selected-logit `Softmax`, two complete expert bodies, per-expert
`Equal`/`CastLike`/weighted `ReduceSum` masks, then an ordered sum of the
contributions. `DenseMoEPattern` calls the shared `TopKSoftmax` body and two
`DenseExpert` bodies, each of which calls the shared `SwiGLU` body.
No local GroupedMatMul function or matcher extension is needed.

This initial rule requires exactly **two experts, top-k=1, float32, opset 24**,
rank-2 token inputs with proven matching token dimensions, positive static
hidden/intermediate sizes, unscaled routing, and immutable bias-free weights. Source gate/up
weights are `[I,H]` and down weights `[H,I]`, each used through
`Transpose(perm=[1,0])` + `MatMul`. The checker verifies the two scalar expert
IDs are 0 and 1 and all six matrices have uniform geometry. Apply before
constant folding removes those Transposes or generic expert fusions hide
the source operators. Constants must already be represented as initializers.

The pre-parsed replacement reproduces ONNX `TopK` on the original logits,
then uses `Equal(indices, [0,1])` and a float `Cast` as CPU MoE routing scores.
With `k=1` and normalized routing, the sole selected weight is one.
The scores have a unique winner, so CPU MoE cannot override ONNX's lower-index
tie choice (or lose close unequal logits through its full-softmax rounding).
This adapter is specific to `k=1`; extending it to larger k requires preserving
the actual selected routing weights, not just one-hot membership.
Gate/up rows are interleaved, not concatenated, into `[2,2*I,H]`, and down
matrices are stacked into `[2,H,I]` for CPU `swiglu_fusion=1`.

**Admitted numerical contract:** opting into this dense rule assumes finite
inputs, weights, logits and all source/target intermediates, and accepts
tolerance-based float32 equivalence rather than bitwise equality or signed-zero
preservation. This assumption is supplied by the caller, not proven by the
static checker or enforced with runtime guards. In particular, finite
parameters alone do not guarantee finite intermediates for arbitrary runtime
activations. A TODO at eligibility records that dense inactive-expert NaN/Inf
propagation versus target skipping still needs investigation.
The overflow counterexample remains a regression demonstrating behavior
**outside** this contract; the rule is not IEEE-equivalent on such inputs.

`DenseTwoExpertMobiusNumericalParityAndPacking` runs both genuine standard
source and rewritten CPU graphs with automatic optimization disabled. Distinct
non-square expert weights, multiple tokens, negative/zero activations, equal
and unequal logits (including differences lost by full-softmax rounding),
both experts selected, static/symbolic token metadata,
and both matcher modes are covered. The output is checked against both the
source and an independent double-precision reference using absolute-plus-relative
tolerance `1e-5 + 1e-5 * |reference|`. The test asserts that CPU executes
the MoE node and separately verifies every packed gate/up/down element after
constant folding, serialization/reload, and rewrite idempotence.
Negative cases reject top-k=2, wrong activation/transpose/mask/reduction,
overridable weights, exposed intermediate outputs, incompatible providers,
and unproven expert dimensions or unknown-rank token metadata.

### Coverage relative to current Mobius MoE sources

The inventory baseline is Mobius
[`d69efe03a9f4c1ada744bc31dd01077a710b3c39`](https://github.com/onnxruntime/mobius/tree/d69efe03a9f4c1ada744bc31dd01077a710b3c39).
The grouped fixtures above use real routing components but substitute a local
GroupedMatMul expert body. They are **not evidence of matching** standard or
mixed-op sources; the separate dense tests above establish only the bounded
coverage listed here.

| Source | Current coverage | Required before extending |
|---|---|---|
| A10 Qwen4 `_standard_dispatch` | Deferred | Verify `Scan` accumulator recurrence, nested nonempty `If`, captured weight banks, token gathering/scattering, FP32 SiLU opmath and casts. Pattern nodes with subgraphs are currently rejected; matching an isolated body is not proof of the enclosing recurrence. |
| A10 `MoELayer.forward` dense mask-and-sum | Two bias-free float32 SiLU experts, top-k=1, under the finite-intermediate tolerance contract | Other counts/k, activations, biases and routing forms remain deferred. The explicit ONNX selection adapter preserves ties; inactive nonfinite propagation is outside the admitted contract. |
| A10 Gemma4 vectorized fallback | Deferred | Separate Einsum/routing-layout checks and exact activation qualification. QuickGELU with alpha 1.702 is not exact erf/tanh GELU. |
| B1 dense `MatMulNBits` experts to QMoE | Deferred | Uniform INT4 geometry, interleaved packed weights **and** scales/zero points, CPU ABI qualification, routing and dense inactive-expert semantics. Not covered by float grouped rules or by projection fusion alone. |

Source anchors: [`MoELayer.forward`](https://github.com/onnxruntime/mobius/blob/d69efe03a9f4c1ada744bc31dd01077a710b3c39/src/mobius/components/_moe.py#L671-L691),
[`Qwen4._standard_dispatch`](https://github.com/onnxruntime/mobius/blob/d69efe03a9f4c1ada744bc31dd01077a710b3c39/src/mobius/models/qwen4_exp.py#L1108-L1184),
and [`_qmoe_fusion._fuse_layer`](https://github.com/onnxruntime/mobius/blob/d69efe03a9f4c1ada744bc31dd01077a710b3c39/src/mobius/rewrite_rules/_qmoe_fusion.py#L350-L449).
The last source emits concatenated gate/up rows with `swiglu_fusion=2`;
this tree's CPU QMoE requires interleaved `swiglu_fusion=1`. Copying that
upstream replacement is therefore not a runnable CPU implementation.

`DenseMobiusRoutingAndInactiveExpertCounterexamples` faithfully reproduces
the pinned `TopKGate`, bias-free `MLP` and dense `MoELayer.forward` graph for
two experts and `k=1`, without downloading checkpoints or introducing a local
GroupedMatMul function. It runs the source, two hand-built candidate targets,
and the new dense rule's replacement on CPU:

- With equal logits and distinct expert outputs, ONNX TopK selects expert 0,
  whereas CPU MoE selects expert 1. Building selection scores from the original
  TopK indices with `Equal`/`Cast` restores the selected expert and the finite
  output in this test; raw-logit substitution does not. The registered dense
  rule uses this adapter, and its actual rewritten target also passes.
- With untied logits, finite activations and finite weights, an inactive
  expert's `FLT_MAX` down-projection overflows. Dense evaluation multiplies
  that infinity by its zero mask and returns NaN. Both naive and
  selection-adapted MoE skip the inactive expert and return finite results.
  Thus a tie adapter alone is insufficient, and checking finite constant
  weights does not prove finite runtime intermediates. The registered dense
  replacement exhibits the same divergence outside its admitted contract.

No full A10 or B1 coverage is claimed. Arbitrary nonfinite/overflow behavior
would require runtime guards with a faithful fallback or a different target
execution policy, and the recurrent source independently requires region
support. A static match callback cannot infer the needed runtime activation
bounds.

## Example: HardSigmoid followed by Mul

This complete rule recognizes the HardSwish decomposition with the exact
HardSigmoid attributes shown:

```cpp
#include "core/optimizer/pattern_rewriter.h"

using namespace onnxruntime;
using namespace onnxruntime::rewrite;

Rule MakeHardSwishRule() {
  const auto* pattern = R"(
    (x:float[]) => {
      sigmoid = HardSigmoid(x, alpha=0.16666667, beta=0.5)
      return Mul(sigmoid, x)
    }
  )";
  const auto* replacement = "(x) => HardSwish(x)";
  return Rule{"HardSigmoidMul", pattern, replacement, {}, {{kOnnxDomain, 14, 21}}};
}
```

Both occurrences of `x` must bind the **same `NodeArg`**, not just tensors with
equal shapes or contents. `float[]` checks that `x` is a float32 tensor of any
rank, including unknown rank (`float` alone would require a scalar);
resolved schemas establish the corresponding intermediate and output types.
It does not require every matched value to have the same element type.

The fourth `Rule` argument, `check_and_bind`, is an optional C++ callback that
checks semantic conditions not expressed in the pattern and prepares new
match-local bindings for the replacement. It returns `false` to reject a match,
or `true` to accept it with any prepared bindings. Here `{}` means no additional
checks or bindings: accept after the built-in matching and safety checks.
All constructor overloads default it to `{}`. Use the explicit
`{}` placeholder above when supplying the fifth argument, opset requirements.
The opset list defaults to empty, not an inferred version range: if omitted,
populate `rule.opsets` before constructing a `PatternRewriter`. Missing required
domain ranges remain an error. `check_and_bind_calls` counts only callbacks
actually invoked, so an empty callback contributes zero.

These opset bounds are a conservative supported range for this example, not an
automatic statement about every schema version. Floating-point rewrites can
change evaluation order; validate equivalence within the intended tolerance.

## Pattern and replacement syntax

Definitions start with named inputs and `=>`. Braces around the body are optional;
use them to make multi-statement rules easier to read. Assignments create named
operator outputs; `return` either calls the final operator (the root) or names an
ordered list of previously defined operator outputs. Whitespace separates calls, and semicolons are
optional separators. A trailing semicolon before the closing brace is allowed.
Only whitespace and comments may follow the closing brace. In both patterns and
replacements, `#` starts a comment through the end of the line (or end of text).
Comments act as whitespace between tokens, including in headers and bodies;
they cannot split an identifier, number, or multi-character token.
Hashes inside single- or double-quoted attribute strings remain literal:

```text
# A complete line comment
(x) => {
  a = Identity(x) # An inline comment
  return Identity(a)
} # A trailing comment
```

```cpp
Pattern pattern(R"(
  (x) => {
    add = Add(x, 3.0)
    clip = Clip(add, 0.0, 6.0)
    product = Mul(clip, x)
    result = Div(product, 6.0)
    return result;
  }
)");
Replacement replacement("(x) => HardSwish(x)");
```

The example assigns one root output, named `result`, then returns it. Assignment
and return are separate statements: `return result = Div(...)` is rejected.
A final call without an assignment, such as `return Div(product, 6.0)`, also describes one output,
but does not give that output a lookup name. Replacements additionally accept
a final call without the `return` keyword, as above.

Use `domain.Operator(...)` (ONNX text-format notation) or
`domain::Operator(...)` for qualified operators in patterns and replacements:

```cpp
Pattern pattern("(x,y) => { mm = com.microsoft.FusedMatMul(x,y); return Identity(mm); }");
Replacement replacement("(x,y) => com.microsoft.FusedMatMul(x,y)");
```

An unqualified operator uses the standard ONNX domain. `ai.onnx::Identity`
and `ai.onnx.Identity` are equivalent to `Identity`. Dotted qualification
splits at the last dot, so `com.microsoft.FusedMatMul` has domain
`com.microsoft` and operator name `FusedMatMul`.

Names must be unique and defined before use. Every declared pattern input must
participate, and every operator must reach the root. Inputs to different named
variables may alias; repeating one variable requires identical binding.
`match.valueOf("x")` and `match.valueOf("clip")` return bound `const NodeArg*`
values. Unknown names and unbound values raise explicit errors. Read-only
`Pattern::NamedValue`, `Nodes`, `Constants`, `Outputs`, `ValueCount`, `IsInput`,
`IsOptional`, and `AttributeNames`
are available for introspection.

### Optional type and shape annotations

Pattern header variables and assigned outputs may carry `: type` annotations.
Unannotated variables retain their existing unconstrained behavior:

```cpp
Pattern pattern(R"(
  (x:float[M,N], y:float[N,K]) => {
    product:float[M,K] = MatMul(x,y);
    return Identity(product);
  }
)");
```

Dimension names such as `M`, `N`, and `K` are **declared implicitly** by their
first appearance in an annotation. Do not list them as separate header inputs.
They cannot collide with tensor, attribute, or element-type variable names.

| Annotation | Constraint |
|---|---|
| `float` | Rank-zero float32 tensor (known scalar shape) |
| `float[]` or `float[...]` | Float32 tensor of any rank, including unknown rank |
| `float[M,N]` | Known rank 2, with named dimension bindings |
| `float[2,3]` | Known rank 2 and exact static dimensions 2 and 3 |
| `float[M,...]` | Known rank at least 1, binding its first dimension |
| `float[_,N]` | Known rank 2; first dimension unconstrained and unnamed |
| `tensor[M,N]` | Known rank 2, without an element-type constraint |
| `tensor<T>[M,N]` | Known rank 2, with shared element-type variable `T` |
| `tensor<T>[]` | Any rank, with shared element-type variable `T` |

Tensor element names follow ONNX, for example `float`, `double`, `float16`,
`bfloat16`, `int64`, `int32`, `uint8`, `bool`, and `string`. Only tensor types
are supported; sequences, maps, and optional ONNX value types are not annotations.
`None` in a type union describes an absent operator slot, not an ONNX optional
value type: `(x:float[], low:float | None, high:float | None)` allows absent
scalar Clip bounds. An absent slot matches only its `None` alternative, without
binding any dimensions or element type.

The matcher uses the model's existing metadata; it does not infer shapes or
solve equations. Rank-specific constraints reject unknown rank, and fixed
dimensions require known equal integers. A named dimension binds to an integer
for a static dimension or a nonempty string for a symbolic dimension. Repeated
known bindings must agree exactly: integer 2 differs from symbolic `"batch"`,
and `"batch"` differs from `"other"`. An anonymous unknown dimension leaves
the variable unbound and does not conflict with a later known binding.
Consequently, repeated names do **not** prove runtime equality when occurrences
are unknown; check-and-bind callbacks requiring that proof must reject unknown metadata.
Likewise, a missing element type leaves `T` unbound, while a concrete annotation
such as `float` requires a known matching element type.

In a check-and-bind or replacement callback, `match.dimensionOf("M")` returns a
`Dimension` (`std::variant<std::monostate, int64_t, std::string>`), and
`match.elementTypeOf("T")` returns an optional ONNX element-type enum value
represented as `int32_t`. Unknown declared bindings are empty; undeclared names
raise an error. These bindings are local to each match, cannot be overwritten
with computed attributes, and survive clearing computed attributes.
`Pattern::TypeConstraints()` and `IsAttribute()` expose read-only introspection.

Attribute captures can be declared and annotated in the header:

```cpp
Pattern pattern("(x:float[M,N], axes:ints) => { t=Transpose(x,perm=axes); return Identity(t); }");
Replacement replacement("(x,axes) => Transpose(x,perm=axes)");
```

A header name used in keyword position denotes an attribute, not a tensor.
Attribute annotations are `int`, `float`, `string`, `ints`, `floats`, `strings`,
or `tensor`, matching the corresponding concrete `AttributeProto` type.
For example, `float` means a FLOAT attribute in keyword position but a scalar
float32 tensor in tensor position; use `int64` for an integer tensor and `int`
for an INT attribute. Attribute variables cannot also be tensor inputs, carry
shapes, or use type unions (including `None`). Undeclared captures remain supported and untyped.

Annotations are constraints on **patterns only**; replacements continue to use
unannotated inputs and derive types/shapes from preserved outputs and schema
inference. Annotations on returns or replacements are rejected rather than
silently ignored. Each annotation allows at most 64 dimensions, with at most
256 distinct implicit dimension/element-type variables per pattern.

### Type alternatives

Use `|` for a disjunction of tensor constraints:

```cpp
Pattern pattern("(x:float[M] | float[M,N]) => return Identity(x)");
Pattern optional("(x,low:float | None) => return Clip(x,low)");
```

Alternatives are tried left to right, with annotated variables processed in
definition order. A call inserts its callee's ordered constraint occurrences
at the call site, preserving callee order even when actual arguments are
reordered or aliased to the same value. Matching finds the first **globally consistent** combination:
if a later constraint conflicts with a tentative dimension or element-type
binding, earlier annotation alternatives are reconsidered. Failed alternatives
leave no bindings behind. Variables mentioned only in unchosen alternatives
remain unbound. Check-and-bind callbacks run once after a consistent combination is selected;
Callback rejection does not trigger another combination.

This bounded search applies only to type/shape annotations, not graph structure:
there is still no search over producers or operand permutations. The product
of alternative counts across annotations is limited to 64 at parse time.
Empty branches, duplicate constraints, and `None` alone are rejected.
`None | float[M]` is also accepted, but `None` is permitted only for header tensor
inputs, not assigned outputs or attribute captures. Replacements remain untyped.
The former experimental `x?` syntax is removed; use `x:tensor[...] | None` for
an unconstrained optional tensor slot.

### Optional input slots

Use `None` in an operator input position to require an absent input. It matches
both an empty ONNX input name and an omitted trailing slot. Mark a header
variable with a tensor type union including `None` to allow either a present value or an absent slot:

```cpp
Pattern pattern("(x,low:tensor[...] | None,high:tensor[...] | None) => { a=Identity(x); return Clip(a,low,high); }");
Replacement replacement("(x,low,high) => Clip(x,low,high)");
Pattern absent_min("(x,high) => { a=Identity(x); return Clip(a,None,high); }");
```

An ordinary header variable requires a present input. Repeated optional variables
must bind identically, including absence: `(x,v:tensor[...] | None) => Clip(x,v,v)` does not match
one present and one absent slot. `match.valueOf("low")` returns `nullptr` when
the optional binding is absent; check-and-bind callbacks must check it before dereferencing.
Replacements preserve an absent binding as an empty input name. Explicit `None`
is also supported in replacements. Schema validation still rejects missing
required inputs. This is slot matching, not variadic matching: extra actual
input slots are not accepted. `None` is reserved and cannot be a variable name.

### Constants and attributes

Positional numeric literals in patterns match **rank-zero constant initializers**
of the exact value and element type: `3.0`, `3e0`, and `0.5` are float32;
`3`, `-1`, and `0` are int64. Thus `2` and `2.0` are distinct constraints.
A singleton vector is not a scalar. External tensor data and overridable
initializers do not satisfy these constraints. Constant operators require
an earlier constant-folding pass.

A bracketed positional literal is a **rank-one tensor**, not an attribute:

```cpp
Pattern pattern("(x) => { reshaped = Reshape(x,[2,3]); return Identity(reshaped); }");
Replacement replacement("(x) => Reshape(x,[2,3])");
```

`[1,2]` is an int64 tensor of shape `[2]`; `[1.0,2.0]` is float32 of the same
shape. `[1]` has shape `[1]` and does not match scalar `1` or a matrix `[[1]]`.
Matching requires exact rank, length, element type, order, and values. Nested
lists are not supported. All elements must have the same numeric type spelling:
mixing integer and floating-point literals is rejected instead of silently
converting them. Float elements must be finite. `[]` is an empty int64 tensor
of shape `[0]`. Replacement tensor lists synthesize immutable initializers.

Keyword arguments constrain attributes, not tensor inputs:

```cpp
Pattern pattern("(x) => { h = HardSigmoid(x, alpha=0.25, beta=0.5); return Mul(h,x); }");
```

Attribute equality compares the protobuf attribute type as well as its value.
For example, `axis=-1` is an integer attribute, while `epsilon=1e-5` is a float.
Single- or double-quoted string attributes are supported; strings may escape
quotes and backslashes. Other attributes are unconstrained. A listed attribute
must be present in the resolved node's attribute map. Staged replacement nodes
receive schema default attributes before verification and requeueing, so later
rules and callbacks see defaults without an intervening `Graph::Resolve()`.

In keyword positions the same numeric list syntax instead denotes an attribute:

```cpp
Pattern pattern("(x) => { t=Transpose(x,perm=[1,0]); return Identity(t); }");
Replacement replacement("(x) => Transpose(x,perm=[1,0])");
```

Integer lists become `INTS` attributes; float lists become `FLOATS`.
Matching compares the exact attribute type, length, order, and each value.
`[]` denotes an empty `INTS` attribute. Scalar and list attributes are distinct.
Pattern syntax supports scalar integer, float, and string attribute equality,
and numeric list equality, but not custom predicate syntax or string lists.
Inspect `match.nodes` in the C++ check-and-bind callback for ranges, shapes, or other semantic conditions.
Positional inputs must precede keyword attributes.

Instead of a literal, a pattern keyword may capture an attribute by name:

```cpp
Pattern pattern("(x) => { t=Transpose(x,perm=axes); return Identity(t); }");
Replacement replacement("(x,axes) => Transpose(x,perm=axes)");
```

The attribute must be present and concrete. `match.attributeOf("axes")` exposes
the captured `AttributeProto` to check-and-bind callbacks and text replacements without extra preparation.
Capture names are separate from tensor values and cannot collide with them;
declared header captures are classified as attributes rather than tensor inputs.
Repeating a capture requires identical protobuf attribute content, ignoring the
original attribute name and documentation string; type and list order matter.
Captures can hold any concrete attribute type, but replacement schemas decide
whether that type is valid. Captures are per match, survive clearing computed
attributes, and cannot be overwritten with `match.update`. Missing attributes
are not captured from schema defaults: only the resolved node's map is used.

Scalar numeric literals in replacements synthesize rank-zero float32 or int64
initializers; list literals synthesize rank-one tensors. Replacement keyword
literals set scalar or list attributes on the new node.

### Branched patterns and multiple outputs

Assignments can describe a DAG rather than only a chain:

```cpp
Pattern pattern(R"(
  (x,y) => {
    sum = Add(x,y)
    product = Mul(sum,x)
    return Add(sum,product)
  }
)");
```

Here both uses of `sum` must have the same producer. The matcher does not search
for equivalent computations or reorder commutative inputs.

Comma-separated assignment names declare the exact ordered output count:

```cpp
Pattern pattern("(x) => { first,second = Split(x,axis=0); return Concat(first,second,axis=0); }");
Replacement replacement("(x) => Identity(x)");
```

For a multi-output root, assign its outputs first, then return their names:

```cpp
Pattern pattern(R"(
  (x) => {
    copied = Identity(x);
    left,right = Split(copied,axis=0);
    return left,right;
  }
)");
Replacement replacement("(x) => { left,right = Split(x,axis=0); return left,right; }");
```

This replacement preserves the two explicitly returned original `NodeArg`s in order,
including their names, types, shapes, graph-output roles, and consumers.
The replacement names are local labels, not new externally visible graph names.
`return Split(x)` describes an operator with one declared output; use assignment
names to describe its exact arity. Named returns may select and reorder outputs,
including outputs from different nodes:

```cpp
Pattern pattern("(x) => { a=Identity(x); y=Add(a,a); z=Mul(y,y); return y,z; }");
Replacement replacement("(x) => { y=Add(x,x); z=Mul(y,y); return y,z; }");
```

The final operator remains the unique sink used to anchor backward matching.
Every defined operator must reach it, independently of the returned-value order.
Multiple independent sinks are rejected. Return position defines correspondence:
the first replacement return inherits the first pattern return's `NodeArg`,
and so on. The returned counts must match. Returning a leaf input, duplicate
names, or undefined values is rejected.

Only **explicitly returned** values may have graph-output roles, external
consumers, or implicit subgraph captures. All other outputs, including unreturned
sink outputs, must remain within the deletion set or be unused. Internally
consumed returned values are correctly wired into the replacement graph.
If an external consumer of a returned intermediate feeds a pattern input,
the staged producer dependencies are checked for cycles before deletion.
A replacement that introduces a cycle is rejected with a status and rolled
back; a mapping that preserves an acyclic dependency graph is allowed.

## Checking applicability and preparing bindings

`Rule::CheckAndBind` is `std::function<bool(const Graph&, Match&)>`.
It checks semantic applicability and can prepare match-local bindings before
returning `true`. Return `false` (or `match.Reject(reason)`) to reject the rule.
Declare the attribute binding in the replacement's input list, set it with
`match.update`, and reference it as a keyword argument in a text replacement.
No replacement lambda is needed just to prepare attributes.

The following rank-two specialization of the `TransposeMatMul` example rule
checks a node attribute rather than inspecting types in C++. The pattern's
annotations already require float32 matrices with compatible dimensions.
The callback accepts only `Transpose` with an explicit `perm=[1,0]`, then
binds a new integer attribute variable, `rhs_transpose`, for the text replacement.
The resulting `FusedMatMul` uses `transB=1` to perform the same transpose internally.
The Microsoft-domain opset must already be imported.

```cpp
#include "core/optimizer/pattern_rewriter.h"

using namespace onnxruntime;
using namespace onnxruntime::rewrite;

Rule MakeTransposeMatMulRule() {
  const auto* pattern = R"(
    (x:float[M,N], y:float[K,N]) => {
      transposed = Transpose(y)
      return MatMul(x,transposed)
    }
  )";
  auto check_and_bind = [](const Graph&, Match& match) {
    const auto& attributes = match.nodes[0]->GetAttributes();
    auto perm = attributes.find("perm");
    if (perm == attributes.end() ||
        perm->second.type() != ONNX_NAMESPACE::AttributeProto_AttributeType_INTS ||
        perm->second.ints_size() != 2 ||
        perm->second.ints(0) != 1 || perm->second.ints(1) != 0)
      return false;
    ONNX_NAMESPACE::AttributeProto trans_b;
    trans_b.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    trans_b.set_i(1);
    match.update("rhs_transpose", std::move(trans_b));
    return true;
  };
  const auto* replacement = "(x,y,rhs_transpose) => com.microsoft.FusedMatMul(x,y,transB=rhs_transpose)";
  return Rule{"TransposeMatMul", pattern, replacement, check_and_bind, {{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}}};
}
```

`match.nodes[0]` is the first pattern operator, `Transpose`, not the root
`MatMul`. `rhs_transpose` is a new match-local name, not a graph value or an
attribute captured by the pattern. The replacement declares it in its header
and uses it in keyword position to set `transB`. No replacement lambda is needed.
This specialization deliberately rejects an omitted `perm`, even though the
rank-two default would also transpose the matrix.

Only the selected rule's replacement runs, with the bindings prepared by its successful check-and-bind
callback. Replacement callbacks remain supported for choosing a parsed recipe;
they receive `const Match&` and cannot update it. Parse any such recipes outside
the callback to avoid per-match parsing.

Each rule gets a separate copy of the structural match. Rejected rules and
other candidates cannot leak prepared attributes, captured bindings, or rejection
reasons. Match-only mode also runs check-and-bind callbacks, but never replacements
or graph mutations. Neither callback may mutate the graph or retain match
pointers; callbacks should have no externally visible side effects.

`update` is non-const and accepts a concrete `AttributeProto`, including
list-valued and tensor-valued attributes. A tensor binding can be used as a
`ConstantOfShape` node's `value` attribute; it is not a new arbitrary tensor-input binding API.
It cannot overwrite a named pattern value or store a referenced/undefined
attribute. Prepared and captured attributes remain available to the replacement.
`attributeOf` retrieves either kind through a read-only lookup and errors if it is missing. A replacement header
name cannot be used as both a tensor and an attribute. Attribute-only names need
not appear in the pattern, but must be supplied by the callback. Reusing a parsed
plan does not reuse a previous match's attributes. `ClearUpdates` and `Reject`
are also non-const; there is no mutable storage behind a const match.

Replacement tensor inputs may reference only pattern input variables, literal
constants, or earlier outputs in the replacement itself. A matched intermediate
is about to be deleted and cannot be a replacement input. Replacement sequences
must be connected to their final operator.

## When a match succeeds or fails

Before calling a rule's check-and-bind callback, the matcher checks:

- Exact operator name/domain, ordered output counts and input slots, and repeated
  binding identity. Omitted trailing inputs match only optional variables or
  explicit `None`; variadic patterns are not supported.
- Literal and listed attribute constraints. It does not otherwise infer semantic
  equivalence or check arbitrary shapes/types for the author.
- Distinct matched nodes with the same layer annotation. Matched operators cannot
  own subgraphs or have implicit inputs.
- Pattern input variables are outside the matched deletion set.
- Every unreturned output stays internal. A graph output, an external consumer, or
  an implicit capture by an external subgraph of an unreturned value prevents the rewrite.
- Execution-provider eligibility as described below.

Returned values may have graph outputs or external consumers because their
identities are preserved. Even match-only mode applies deletion-boundary checks:
it reports safe rewrite eligibility, not arbitrary structural occurrences.

### Types in the supplied examples

`MakeExampleRules()` supplies six rules when contrib operators are enabled:

| Rule | Explicit float32 checks |
|---|---|
| HardSwish and HardSigmoidMul | `x` |
| MatMulDiv and FusedMatMulDiv | `x`, `y`; separate finite scalar initializer check on `divisor` |
| TransposeMatMul | `x`, `y` |
| RmsNormalization | `x`, `scale`; separate positive finite scalar initializer check on `epsilon` |

There is no match-wide float32 restriction. Independently typed operands such
as shape/index tensors or a Pow exponent require their own semantic constraints.
The supplied RMS pattern specifically matches `Pow(x,2.0)`; the float32 exponent
is a literal constraint, not a blanket callback check.

The RMS example also requires last-axis reduction, keepdims, known positive
width and a matching one-dimensional scale, with pre-opset-18 attribute axes.
TransposeMatMul requires a known rank of at least two and exactly the last-two-axis
swap. Division examples require a finite scaling result. Consult the rule
definitions and correctness tests before extending their supported ranges.
Contrib-disabled builds supply only the two HardSwish rules.

## Opset and execution-provider applicability

Each `OpsetRequirement` is `{domain, min_version, max_version}`, with positive,
inclusive bounds. Declare every domain used by a pattern or replacement;
additional requirements are allowed. `""` and `"ai.onnx"` denote the same domain.
Duplicate domains, reversed ranges, and undeclared pattern domains are errors.

All requirements are evaluated against each graph's `DomainToVersionMap()`,
including independently for subgraphs. Missing imports or out-of-range versions
make the rule inapplicable **before its check-and-bind callback runs**. The engine does not add
imports, compare individual node schema since-versions, or rewrite model-local
function bodies. Authors must establish correctness across their declared range.

All matched nodes must have the same execution-provider assignment. Unassigned
nodes are eligible; assigned nodes must belong to
`Options::compatible_execution_providers`, which defaults to CPU only.
Mixed assignments are rejected. This policy applies before the rule callback and
also in match-only mode. To opt into another provider:

```cpp
Options options;
options.compatible_execution_providers.insert(kCudaExecutionProvider);
PatternRewriter rewriter(Rules{MakeHardSwishRule()}, options);
```

Allowing a provider is the caller's assertion of kernel and precision
compatibility, not automatic kernel discovery. Clearing the allowed set permits
only unassigned nodes. Replacements inherit the root's provider and layer
annotation.

## Applying rules, priority, and errors

Register the transformer through the normal session pipeline:

```cpp
auto transformer = std::make_unique<PatternRewriter>(Rules{MakeHardSwishRule()});
ORT_RETURN_IF_ERROR(session.RegisterGraphTransformer(
    std::move(transformer), TransformerLevel::Level1));
```

For one already-resolved graph, use `Run` and explicitly resolve afterwards:

```cpp
PatternRewriter rewriter(Rules{MakeHardSwishRule()});
Statistics statistics;
ORT_RETURN_IF_ERROR(rewriter.Run(graph, statistics));
ORT_RETURN_IF_ERROR(graph.Resolve());
```

`Run` handles one graph. The transformer's `Apply` path recurses into subgraphs
and performs normal graph resolution. Callers must not concurrently mutate a
graph. Options and compiled rules are immutable; match state and statistics are
per call.

Rules have deterministic **registration-order priority**, including across
different structural groups. At each eligible root, rewriting chooses the first
passing check-and-bind callback. With `Options{.match_only=true}`, all passing callbacks are counted
and replacements are not called. `indexed=false` selects the reference matching
mode with the same eligibility and priority semantics.

Every pattern and replacement must contain at least one node, and their
returned-value counts must agree. Single-node patterns, same-size replacements,
and expansion are supported. Zero-node replacements and returning leaf inputs
remain unsupported; identity elimination is a separate transformation.

`Options::max_iterations` is a positive upper bound, defaulting to 10.
Each iteration processes a fixed candidate batch, initially all live nodes in
topological order. Replacements and affected existing nodes are queued for the
next batch, never appended to the current batch. An existing later candidate
can still match an earlier replacement in the same batch. Processing stops
when a batch performs no rewrites, there are no pending candidates, or the bound
is reached. Match-only mode processes exactly one batch without mutation.

```cpp
Options options;
options.max_iterations = 3;
PatternRewriter rewriter(rules, options);
```

The iteration cap bounds repeated self-rewrites and rule cycles, even when
node count grows. It does not prove convergence or semantic equivalence.
`Statistics::iterations` counts processed batches;
`iteration_limit_reached` is true if the final allowed batch rewrote nodes and
left candidates pending. Reaching the limit returns success with the current
valid rewrite result, not an error; the flag does not assert another match exists.
Calling `Run` again starts a fresh budget. External transformer-manager passes
also have their own budgets. Avoid expansion rules that multiply graph size
unnecessarily: the iteration limit is not a memory bound or a bound on callbacks.

After a rewrite, the engine updates edges and producer/consumer indexes,
notifies node replacement, and requeues replacement nodes and affected
descendants through a bounded invalidation walk. This permits newly enabled
matches in subsequent batches without retaining pointers to deleted nodes.

Malformed definitions throw `OnnxRuntimeException` with parse-location
diagnostics; exception-disabled builds follow ORT's abort behavior. Invalid rule
configuration raises an explicit error when the rewriter is constructed.
Direct parsed replacements are checked against their rule during construction;
callback-generated replacements also undergo runtime validation. Returning false from check-and-bind
is an ordinary nonmatch.

Replacement schemas and attributes are validated before matched nodes are
deleted. Failed staging rolls back the staged nodes and literal initializers;
there is no whole-call transaction undoing earlier successful rewrites.
Errors propagate as statuses or ORT exceptions rather than being treated as
nonmatches. Final graph resolution checks types and inferred shapes; a rule
author must not rely on schema staging alone to prove type correctness.

`Statistics` exposes live roots visited (including revisits), enumerated
candidates, structural checks, `check_and_bind_calls`, successful matches, and completed
rewrites. Match-only counts all passing check-and-bind callbacks; rewrite mode counts selected
winners. These counters do not alter rule semantics.

### Opt-in failure diagnostics

Set `Options::on_failure` to receive structured `MatchFailure` records containing
the rule name, candidate root index, `FailureStage`, and a reason string:

```cpp
Options options;
options.on_failure = [](const MatchFailure& failure) {
  std::cerr << failure.rule << ": " << failure.reason << '\n';
};
```

Stages distinguish opset filtering, structure, type/shape constraints, constants, attributes, deletion
boundaries, execution providers, check-and-bind callbacks, and replacement validation. An opset
rejection is reported once per rule/graph with root index
`std::numeric_limits<NodeIndex>::max()` because no candidate was inspected.
Other failures are reported for attempted candidates; indexed mode skips
unrelated root operators, so diagnostic counts need not match reference mode.
Shared structural failures are attributed to each affected rule.

A check-and-bind callback can use `return match.Reject("unsupported element type");` instead of
bare `false` to supply a reason. This changes only diagnostics, not eligibility.
Records own their strings and contain no graph pointers. Callbacks run
synchronously, must not mutate the graph or reenter the rewriter, and may copy
records for later inspection. No diagnostic records are constructed when
diagnostics are disabled; `Reject` still stores an explicitly supplied callback
message. Replacement validation status errors are reported and
still propagated; exceptions from user callbacks or named lookups still propagate
normally and are not converted to nonmatches.

## Limits and unsupported forms

Concrete entry plans are limited to 64 nodes, 256 values, and 64 inputs/outputs per
operator. They are connected acyclic graphs rooted at the final operator.
Named pattern alternatives are expanded as described below. There is no inline
operator/subgraph alternation, implicit commutation, variadic matching,
general tensor synthesis, domain-import editing, or model-local function
rewriting. Control-flow operators are not matched, although their subgraphs can
be rewritten through `Apply`.

Tests in `onnxruntime/test/optimizer/pattern_rewriter_test.cc` cover parsing,
check-and-bind callbacks, boundaries, multi-output rewrites, rule priority, and numerical CPU
equivalence, including small decoder prefill/decode graphs. Add structural and
executed equivalence tests when writing or extending a rule.

## Matching algorithm outline

A single match is a **backward traversal from a candidate sink node**, without
structural backtracking. Starting with the pattern's final operator bound to that model
node, the matcher visits pattern operators in reverse definition order and
follows bound values to their producers through a value-to-producer dictionary.
Each input/output slot binds to one exact model value; repeated bindings must
agree. Independent DAG branches need not have the same node ordering in the
model, but operand order and data dependencies must match. A mismatch rejects
the candidate rather than trying other producers or operand permutations.
Constants, attributes, deletion boundaries, and EP eligibility are checked
before the rule's check-and-bind callback. Type unions use the separate bounded
annotation-choice search described above.

Multiple rules are handled **node-first**, inside an outer bounded retry loop:

```text
filter rules by this graph's opset imports
for each iteration, up to max_iterations (default 10):
    for each still-live node in the current candidate batch:
        find candidate rule groups by the node's (domain, operator name)
        match their patterns and evaluate eligible check-and-bind callbacks
        apply the passing rule with earliest registration priority
        queue replacement nodes and affected nodes for the next batch
    stop if no rewrite occurred or no candidates remain
```

The first batch contains all model nodes in topological order; later batches
contain only queued candidates. New nodes and revisits wait for the next batch,
although a later node already in the current batch sees earlier graph changes.
The retry loop is therefore **outside** the node loop, not a separate ten-retry
loop at each node.

By default, a hash map indexed by `(domain, operator name)` selects candidate
groups using the model node as the prospective pattern root. Structurally
identical patterns can share a traversal, followed by their rule-specific check-and-bind callbacks;
patterns with attribute predicates or type/shape annotations are kept separate. The earliest passing
rule wins even across groups, and lower-priority work is skipped when possible.
Structural results are cached per group and candidate node, but callbacks
run in global registration order, not group order. Rewrite mode stops at
the first accepted rule without invoking lower-priority callbacks; match-only
mode visits all matching rules in that same order. Group signatures use
locale-independent serialization.
With `indexed=false`, the reference mode instead tries all opset-applicable
rules in registration order at each node. Match-only mode processes one batch,
counts all passing rules, and never applies replacements.

## Composable pattern libraries

`PatternLibrary` parses an owning library of interfaces and named patterns.
Every concrete named pattern is callable; graph bodies and alternatives are
two definition forms in one namespace, not separate declaration kinds.
Definitions may refer forward to other definitions.
`library.GetPattern("Fuse")` returns a copyable `Pattern` usable by every
existing `Rule` constructor. The library and its source text need not outlive
the returned pattern.

```cpp
PatternLibrary library(R"(
  interface GatedActivation(g:float[M,N], u:float[M,N]) -> (product:float[M,N])

  pattern plain implements GatedActivation {
    s = Sigmoid(g)
    a = Mul(g,s)
    p = Mul(a,u)
    return (product=p)
  }

  pattern scaled implements GatedActivation: (alpha:float[...]) -> {
    scaled = Mul(g,alpha)
    s = Sigmoid(scaled)
    a = Mul(g,s)
    p = Mul(a,u)
    return (product=p)
  }

  pattern Forms = plain | scaled

  pattern wrapped implements GatedActivation {
    act = Forms(g,u)
    return (product=act.product)
  }

  pattern Fuse(g,u) {
    outer = wrapped(g,u)
    return Identity(outer.product)
  }
)");
```

An **interface** is an abstract tensor binding contract, not a callable graph.
A **pattern body** can implement that contract, inheriting interface inputs
and returning its interface outputs, either positionally or
with explicit `(port=local_value, ...)` mappings. Named mappings may reorder
ports. Output annotations constrain the mapped values as well as input
annotations; they do not manufacture shape evidence. Identity mappings such
as `return (y=x)` are allowed inside patterns and emit no operator.
A body without `implements` declares its inputs directly, as in `pattern Fuse(g,u)`.
Its output port names come from explicit `(port=value)` mappings or returned
value names (`return act.product` exposes `product`). A returned unnamed
operator call exposes `output`, or `output0`, `output1`, etc. for multiple
outputs. All named patterns, including patterns suitable for entry use, may
be called directly or through alternatives.

### Calls versus standalone entries

Reusable bodies may return identity aliases or independent output branches.
Every declared input must be used, and every operator must reach at least
one returned value, but those returned values need not have a single root.
`GetPattern(name)` validates **every fully expanded plan** as a standalone
entry: it must have an operator root, all operators must feed that final
operator, and returned values cannot be leaf/identity aliases or duplicate
the same value through different output names. Callable bodies may still
export multiple aliases of the same value. An entry violation raises a
use-time diagnostic naming the requested entry. Inherited interface inputs
and mapped outputs must remain tensor bindings, checked during library
construction; extra body-only inputs may capture attributes. Mapped interface
outputs are always required to be present, even when unannotated and aliased
to an optional input. Interface tensor-kind requirements survive calls and
aliases; callers cannot reclassify these mapped values as attribute captures
or pass them to attribute formals. Conflicts fail during library construction.
Thus a library
may contain useful callable patterns that cannot independently serve as rules:

```text
pattern identity(x) { return (y=x) }
pattern branches(x,z) {
  left = Identity(x)
  right = Neg(z)
  return (a=left,b=right)
}
pattern join(x,z) {
  p = branches(x,z)
  q = identity(p.a)
  return Add(q.y,p.b)
}
```

`GetPattern("join")` succeeds; `GetPattern("identity")` and
`GetPattern("branches")` fail. These checks apply equally to body patterns,
aliases and alternative definitions. Legacy standalone `Pattern("(x)=>...")`
syntax and its existing entry validation are unchanged.

The colon signature extends a pattern with external inputs discovered by
matching. Here callers supply only `g,u`, while `alpha` binds to the input of
the matched Mul. It remains a tensor, not an automatically decoded float.
Additional attribute captures can also be declared in that signature using
the existing attribute-annotation syntax. There is no separate capture
language. Both `float[]` and `float[...]` continue to mean **any rank**, not
scalar.

An **alternative pattern** supplies ordered named members. Members can be
body patterns, other alternative patterns, or single aliases; interfaces
cannot be members. They must have the same named interface (or all have no
interface) and identical ordered input/output port names and annotations.
This is a strict structural annotation check (including symbol spellings),
not semantic type equivalence; equivalent any-rank `[]`/`[...]` are normalized.
Extra discovered inputs of an `implements` body are not common call ports.
A named call `act=Forms(g,u)` and the anonymous call
`act=(plain | scaled)(g,u)` have the same matching behavior. Each occurrence
has an independent selected immediate member, available through `alternative()`;
no user-defined tag name is required. Nesting and single aliases preserve
their selection identities:

```text
pattern Inner = plain | scaled
pattern Outer = Inner | plain
pattern Alias = Outer
pattern nested(g,u) {
  act = Alias(g,u)
  return Identity(act.product)
}
```

For a selection of `scaled`, `act` reports definition `Alias`, member `Outer`;
successive `alternativeView()` calls report `Outer`/`Inner`, `Inner`/`scaled`,
then body `scaled` with no member. Anonymous alternatives use the same
representation and retain their immediate named members too. Patterns may
call other patterns within the expansion and dependency-nesting bounds.
Inline operator or
subgraph alternative spellings are not supported; existing `|` type
annotations are unchanged.

Multi-output calls can bind output tensors explicitly:

```text
weights, indices = RoutingForms(logits, top_k)
```

Names bind directly to the corresponding output `NodeArg`s in declared output
order, just as for multi-output ONNX operators. The number of names must match
the number of outputs; duplicate or already-defined names are rejected.
Annotations on these names constrain the bound output tensors conjunctively,
including when multiple outputs alias each other or a previously annotated input.
Nested calls and
alternatives retain occurrence-local internal bindings without exposing an
instance under any of the output names. Checkers use `match.valueOf("weights")`,
not `match.instance("weights")`. Internal generated scopes are private.

The existing named-instance form remains supported for inspecting concrete
internals: use `act=Forms(g,u)`, `act.product` for common outputs, and C++ views.
A single-output instance additionally
allows `act` as shorthand for its output. Tensor handles, attribute captures,
dimensions and element-type variables are remapped into the call occurrence.
Two instances do not share a capture just because its local spelling is the
same. Formal inputs and returned aliases bind to the caller's actual
`NodeArg`s; no integer handle from another definition is reused.
Caller and callee annotations are conjunctive, while alternatives within
one annotation remain disjunctive.
An annotation classified as a tensor in a compiled callee stays a tensor
after composition. A caller cannot turn that mapped value into an attribute
through a later keyword capture or attribute-formal call; this unsupported
conversion fails during construction, including for ordinary identity aliases.
Typed attribute formals already classified as attributes in the callee remain
supported.

Graph alternatives expand to at most **64 complete concrete plans per
definition**, in source order (the rightmost call varies fastest).
Each plan retains the existing limits of 64 nodes, 256 values and 64
type-annotation combinations. Each plan also permits at most **256 expanded
call occurrences**, counted cumulatively through nested bodies, including
zero-node identity calls. This bounds scoped metadata even when composition
adds no nodes or values; overflow is rejected before copying callee metadata.
Libraries contain at most 256 definitions. A dependency path may contain at
most **64 concrete pattern definitions**, including its entry and any named
alternative/alias wrappers (interfaces do not count). This bound is independent
of declaration order and is enforced before deeper recursive compilation to
protect native stack usage. Forward-reference chains at 64 succeed; 65 or
more fail with an explicit nesting-limit diagnostic.
Recursive calls, duplicate definitions/ports, missing members, incompatible
interfaces and invalid mappings fail during construction. An unresolved
unqualified call must name a registered standard ONNX operator; use an
explicit domain for other operators. This disambiguates misspelled pattern
calls from ONNX calls without changing standalone `Pattern` parsing.

Each complete candidate passes the existing structural, type/constant,
deletion-boundary, layer, provider and opset checks before C++ acceptance.
If the checker rejects it, the next concrete candidate is tried with fresh
captures and computed attributes. Earlier rules retain priority over every
alternative of a later rule. Search is bounded enumeration, not a recursive
producer/permutation matcher. Replacement or callback errors propagate;
they never cause a retry. Match-only statistics count passing concrete plans,
so multiple alternatives of one logical rule can contribute matches.

### Instance binding views and handwritten C++ structs

`match.instance("outer").instance("act")` returns a read-only `BindingView`.
Its `valueOf`, `attributeOf`, `dimensionOf`, and `elementTypeOf` use names
relative to that occurrence. `definition()` identifies the called definition,
`interfaceName()` its contract, and `alternative()` the selected immediate member
(empty for a body call). `alternativeView()` enters that member's metadata;
calling it on a body view raises an error. Repeated calls traverse nested
alternatives/aliases without losing intermediate tags. This navigation does
not add tensor namespaces: all selected member locals use the same occurrence
path, and `instance("child")` addresses a nested call inside the selected body.
Entry metadata is available through `match.bindings()`.
Fully qualified lookups such as
`match.valueOf("outer.act.alpha")` are equivalent. Missing instances and
unavailable member-local names raise errors; they are not nullable optional
fields. Only the selected member's concrete bindings exist.

Views are non-owning and valid only during the candidate callback. They must
not be retained across graph mutation. The following manual structs show what
a future generator could emit; **this API does not generate C++**:

```cpp
struct GatedView {
  const NodeArg* g;
  const NodeArg* u;
  const NodeArg* product;
  explicit GatedView(BindingView v)
      : g(v.valueOf("g")), u(v.valueOf("u")), product(v.valueOf("product")) {
    ORT_ENFORCE(v.interfaceName() == "GatedActivation");
  }
};
struct PlainView {
  GatedView ports;
  const NodeArg* sigmoid;
  explicit PlainView(BindingView v) : ports(v), sigmoid(v.valueOf("s")) {}
};
struct ScaledView {
  GatedView ports;
  const NodeArg* alpha;
  explicit ScaledView(BindingView v) : ports(v), alpha(v.valueOf("alpha")) {}
};
struct ActivationView {
  GatedView ports;
  std::variant<PlainView, ScaledView> alternative;
  static std::variant<PlainView, ScaledView> Select(BindingView v) {
    if (v.alternative() == "plain") return PlainView(v.alternativeView());
    ORT_ENFORCE(v.alternative() == "scaled", "Unexpected activation member.");
    return ScaledView(v.alternativeView());
  }
  explicit ActivationView(BindingView v) : ports(v), alternative(Select(v)) {}
};
struct FusionView {
  GatedView outer;
  ActivationView act;
  explicit FusionView(const Match& m)
      : outer(m.instance("outer")), act(m.instance("outer").instance("act")) {}
};

bool CheckGatingShapes(const GatedView& v) {
  if (!v.g->Shape() || !v.u->Shape() ||
      v.g->Shape()->dim_size() != 2 || v.u->Shape()->dim_size() != 2)
    return false;
  for (int i = 0; i < 2; ++i) {
    const auto& g = v.g->Shape()->dim(i);
    const auto& u = v.u->Shape()->dim(i);
    if (!g.has_dim_value() || !u.has_dim_value() ||
        g.dim_value() <= 0 || g.dim_value() != u.dim_value())
      return false;
  }
  return true;
}

Rule MakePatternFusion(const PatternLibrary& library) {
  auto check = [](const Graph& graph, Match& match) {
    FusionView view(match);
    if (!CheckGatingShapes(view.outer) || !CheckGatingShapes(view.act.ports))
      return match.Reject("Unsupported gating shapes.");
    float alpha = 1.0f;
    if (const auto* scaled = std::get_if<ScaledView>(&view.act.alternative)) {
      const auto* shape = scaled->alpha->Shape();
      if (!shape || !(shape->dim_size() == 0 ||
                      (shape->dim_size() == 1 && shape->dim(0).has_dim_value() &&
                       shape->dim(0).dim_value() == 1)))
        return match.Reject("Alpha must have scalar or [1] shape.");
      InlinedVector<float> data;
      bool available;
      ORT_THROW_IF_ERROR(ReadConstantTensor(graph, scaled->alpha, data, available));
      if (!available || data.size() != 1 || !std::isfinite(data[0]) || data[0] <= 0)
        return match.Reject("Unsupported alpha.");
      alpha = data[0];
    }
    ONNX_NAMESPACE::AttributeProto attribute;
    attribute.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
    attribute.set_f(alpha);
    match.update("activation_alpha", std::move(attribute));
    return true;
  };
  Replacement recipe(R"(
    (g,u,activation_alpha) => {
      a = com.microsoft.QuickGelu(g,alpha=activation_alpha)
      return Mul(a,u)
    }
  )");
  return Rule{"PatternActivation", library.GetPattern("Fuse"), recipe, check,
              {{kOnnxDomain, 17, 17}, {kMSDomain, 1, 1}}};
}
```

Include `<cmath>` for `std::isfinite`, in addition to the rewriter header.
This synthetic fusion recognizes `g * sigmoid(alpha*g) * u`, removes its
trailing Identity, and selects the QuickGelu alpha in C++. The tests execute
both source and replacement with CPU EP for plain, scalar-alpha and
one-element-alpha cases. This is a small floating-point example, not a
general MoE/QMoE lowering or a bitwise-equivalence guarantee.

Replacement callbacks may likewise select among pre-parsed recipes in C++;
that choice is host construction, **not** an ONNX If. Pre-parsed recipes can
reference scoped tensor leaves, for example `(x,p.alpha)=>Mul(x,p.alpha)`.
Such names must exist in every concrete plan when the recipe is attached
directly to a rule; use a callback to choose a member-specific recipe.
Matched internal outputs remain forbidden replacement inputs, including
when accessed through an interface alias.

### Optional immutable tensor constraints and constant readers

The prefix qualifier `const`, as in `top_k:const int64[1]`, requires an
immutable constant initializer in addition to the tensor annotation.
It does not decode contents, bind a scalar, or require local payload storage.
Runtime inputs and overridable initializers do not satisfy it.
It can qualify individual tensor alternatives, including
`const float[1] | float[] | None`. It cannot qualify an attribute type.

`ReadConstantTensor(graph, value, data, available)` has typed overloads for
`InlinedVector<float>` and `InlinedVector<int64_t>`. A missing value,
nonconstant/overridable initializer, wrong element type, or external payload
returns OK with `available=false`, leaving `data` unchanged. A malformed
available payload returns an error, leaving both outputs unchanged.
Success decodes raw or typed protobuf storage and sets `available=true`.
C++ checks rank, supported values and all dependent relationships; decoding
does not implicitly enforce scalar shape. Propagate reader errors, for
example with `ORT_THROW_IF_ERROR` in a bool checker, rather than turning
corruption into an ordinary alternative rejection.

There are no pattern-valued parameters, construction-time scalar parameters,
`where` clauses, tensor-content capture declarations, or C++ code generation.
Semantic acceptance, tensor decoding, arithmetic and kernel support checks
remain C++ responsibilities.
