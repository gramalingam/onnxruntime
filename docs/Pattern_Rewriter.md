# Experimental indexed C++ pattern rewriter

`onnxruntime/core/optimizer/pattern_rewriter.h` defines an opt-in `GraphTransformer`.
It is **not registered in the default optimization pipeline**, and introduces no
session configuration switch or public C ABI. C++ callers may register it using
`InferenceSession::RegisterGraphTransformer`, just like existing custom transformers.
The example predicates are provider-independent and restrict types to float32.
The engine's generic partition policy defaults to CPU or unassigned graphs;
allowing other EPs requires the caller to verify replacement kernel compatibility.

## Source investigation and supported examples

The ONNXScript reference revision is
[`9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36`](https://github.com/microsoft/onnxscript/tree/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36).
The following files were examined, rather than assuming that Python syntax defines
the right architecture for ORT:

| ONNXScript source at that revision | C++ adaptation |
|---|---|
| [`rewriter/_matcher.py`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/_matcher.py) | Backward DAG binding, repeated-value equality, distinct matched nodes, and removal boundary checks |
| [`rewriter/_rewrite_rule.py`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/_rewrite_rule.py) | Separate structural patterns, semantic guards, replacement recipes, ordered priority, recursive subgraph handling |
| [`rules/common/_fuse_hardswish.py`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/rules/common/_fuse_hardswish.py) | `Div(Mul(Clip(Add(x,3),0,6),x),6)` and `Mul(HardSigmoid(x),x)` to HardSwish |
| [`ort_fusions/fused_matmul_rule_sets.py`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/ort_fusions/fused_matmul_rule_sets.py) | MatMul/scalar Div, FusedMatMul/scalar Div (preserving attributes), and last-two-dimension Transpose/MatMul |
| [`ort_fusions/rms_normalization.py`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/ort_fusions/rms_normalization.py) | Float32 Pow/ReduceMean/Add/Sqrt/Reciprocal/Mul/Mul to SimplifiedLayerNormalization |
| [`ort_fusions/softmax.py`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/ort_fusions/softmax.py) | Studied but deliberately not enabled: fp16 cast removal needs a separately verified precision/EP policy |
| [`rules/fusion/_layer_norm.py`, `LayerNormBiasFusion`](https://github.com/microsoft/onnxscript/blob/9ff5a7c7718ab99d2ee0fa9b5effee2c543f3e36/onnxscript/rewriter/rules/fusion/_layer_norm.py) | Test-only adaptation matches all three LayerNormalization outputs, fuses a trailing bias Add, and rejects observable statistics |

Existing ORT `SelectorActionRegistry` already dispatches selectors by root op/domain
and version. `QuickGeluFusion` illustrates repeated inputs and provider restrictions,
while `graph_utils::FinalizeNodeFusion` illustrates edge maintenance. The new
transformer uses ORT's graph/node/edge APIs directly; it does not copy the graph to
a parallel IR. Generic backward bindings and sharing identical structural plans
are the additional capabilities over hand-written selectors.

## Defining a rule

The small DSL uses integer value handles. An input introduces a variable;
`Op` introduces one output, while `MultiOp` introduces an ordered output list.
Reusing a handle requires the identical
`NodeArg`, not merely the same shape or equivalent contents. Pattern operators
specify only name and domain. **Model-imported opset ranges belong to the rule**,
not to individual nodes or their schema since-versions.

```cpp
using namespace onnxruntime::rewrite;
Pattern p;
auto x = p.Input();
auto h = p.Op("HardSigmoid", {x});
p.Op("Mul", {h, x});
Rule rule{
    "HardSigmoidMul", p,
    {{kOnnxDomain, 14, 21}},  // inclusive imported opset range
    guard,  // checks float32, alpha=1/6, beta=1/2
    [x](const Graph&, const Match&) {
      return Replacement{"HardSwish", kOnnxDomain, {x}, {}};
    }};
auto transformer = std::make_unique<PatternRewriter>(Rules{rule});
ORT_RETURN_IF_ERROR(session.RegisterGraphTransformer(std::move(transformer),
                                                    TransformerLevel::Level1));
```

`pattern_rewriter_rules.cc` contains complete readable guard/recipe examples.
`Op(name, inputs)` defaults to the standard ONNX domain; the optional third
argument selects another domain, e.g. `p.Op("FusedMatMul", {x, y}, kMSDomain)`.
The original `Op(Operator{name, domain}, inputs)` overload remains available.
Name overloads accept literals, `std::string` and `std::string_view`; template
deduction keeps the original braced operator form unambiguous.

### Multiple outputs

`MultiOp(name, inputs, output_count, domain)` returns an ordered vector of distinct
value handles. `domain` is optional, and an `Operator` overload is also available.
For example:

```cpp
Pattern p;
auto x = p.Input();
auto pieces = p.MultiOp("Split", {x}, 2);
p.Op("Concat", {pieces[0], pieces[1]});
```

The output count must match the node exactly. Matching can find a producer
through any output, including its second or later output, and checks every
bound output against its exact output slot. Unused outputs of a connected node
are allowed in the definition, but cannot be graph outputs or have external
consumers if the node is deleted. The tests include Split/Concat and a genuine
ONNXScript LayerNorm-bias adaptation with unused mean/inverse-standard-deviation
outputs; exposing either statistic prevents its deletion.

A multi-output root is also supported. The single replacement node inherits
**all root outputs in the original order**, including their names, types, shapes,
graph-output status, and consumers. The recipe does not need an output list.
Its schema must permit that output count. Identity/Split to Split tests execute
both root outputs and downstream fanout, including repeated uses of output 1.
This does not yet support mapping observable outputs from multiple matched nodes
to arbitrary replacement slots, or matching omitted optional output placeholders.

### Guards, applicability and integration

Guards and recipes are trusted C++ code: they must be pure, guards must depend on
local bindings/attributes/constants rather than mutable global graph conditions,
they must not retain match
pointers, and must establish semantic equivalence and replacement type
compatibility. The caller must establish kernel compatibility for its EP policy.
The engine provides structural and deletion safety; it cannot prove
an arbitrary callback's mathematical correctness. Guard false means an ordinary
nonmatch; invalid patterns or recipes report explicit errors.

Each `OpsetRequirement` is `{domain, min_version, max_version}`, with inclusive,
positive bounds. A rule must declare every domain used by its pattern or
replacement; additional domain requirements are allowed. All requirements must
be satisfied by the graph's import map. Missing imports or out-of-range versions
make the whole rule inapplicable, before matching or guards. Duplicate domains
(including `""`/`"ai.onnx"` aliases), reversed ranges and undeclared pattern
domains are rejected when compiling rules.

For example, MatMul/Div to FusedMatMul declares
`{{kOnnxDomain, 9, 21}, {kMSDomain, 1, 1}}`. The standard-domain nodes use the
model's ONNX import; the replacement uses the existing Microsoft-domain import.
Rule authors are responsible for equivalence throughout the declared ranges.
The engine neither derives nor compares node schema since-versions. Replacement
schemas are resolved normally from imports and validated before deletion.

`Apply` recurses into subgraphs and resolves the graph through the normal ORT
transformer wrapper. For diagnostics and separate engine/Resolve timing:

```cpp
PatternRewriter rewriter(MakeExampleRules());
Statistics stats;
ORT_RETURN_IF_ERROR(rewriter.Run(graph, stats));  // already-resolved graph, one graph only
ORT_RETURN_IF_ERROR(graph.Resolve());            // required after rewriting
```

Callers must not concurrently mutate a graph. Compiled rules and options are
immutable and per-call state/statistics are local.

### Generic execution-provider policy

Rules do not inspect EP assignments. After structural matching and before rule
guards, the engine applies `CanRewritePartition`: all matched nodes must have
the same assignment, and an assigned EP must belong to
`Options::compatible_execution_providers`. Unassigned nodes are always eligible.
The default allowed set contains only CPU, preserving the verified execution scope.
Callers can opt into an EP (or clear the set for unassigned-only rewrites):

```cpp
Options options;
options.compatible_execution_providers.insert(kCudaExecutionProvider);
PatternRewriter rewriter(MakeExampleRules(), options);
```

The replacement inherits the root's assignment. Allowing an EP is a caller assertion
of kernel/precision compatibility, not automatic kernel discovery or evidence of
GPU execution. Mixed assignments are rejected even if both EPs are allowed. This
generic policy also applies in match-only mode, which measures safe rewrite
eligibility rather than partition-independent structural matches.

## Compilation, matching, mutation, and termination

Construction validates connected acyclic definitions, used inputs, rule-level
domain/opset requirements, and limits of 64 nodes, 256 values and 64 inputs/outputs per operator.
It buckets groups by root op/domain and interns identical structural signatures,
including ordered output handles and binding identities but not rule-level import ranges.
Rules with identical topology but different attribute/constant conditions share
one structural match, even when their applicability ranges differ.
There is **no arbitrary-prefix trie**: different structural
plans still match independently. This is intentionally a simpler, measurable
starting point than a decision DAG compiler.

Each call filters rules once against `Graph::DomainToVersionMap()`, retaining
registration order within each structural group. No applicability state is cached
across graphs; recursive subgraph calls use their own graph import context.
Filtering costs the sum of rule requirement counts, outside the node traversal.
If no rules apply, the call returns without traversing nodes. Model-local function
rewriting remains unsupported; future function rewriting must use that function's
imports, not the enclosing model's.

The call then builds a producer index and topological worklist once. Matching walks
the compiled node definitions backward without recursive search, alternation,
commutative backtracking, or unbounded neighborhood exploration. Each node checks
op/domain, exact input/output counts, repeated bindings and layer annotation
agreement. Provider eligibility is a separate generic check. All non-root outputs
must be internal: graph outputs,
external consumers (including implicit subgraph captures), and pattern inputs
produced inside the deletion set prevent replacement.

Registration order defines priority even across different structural groups.
Match-only mode reports all passing rule guards; rewrite mode selects only the
first passing rule. Replacements may consume **only input variables**, retain the
root's output `NodeArg`s (name/type/shape and external users), and inherit its
execution provider and layer annotation. The replacement domain must be declared
in the rule requirements and already imported. Replacement schema/attribute checks
precede deletion. ORT edges and producer/consumer indexes are updated; removed
nodes are reported via `NotifyNodeReplacement`. Final Resolve validates types and
inferred shapes, rather than silently ignoring graph-validation errors.

After mutation, the replacement and surviving input producers seed a bounded
forward invalidation walk. Descendant roots up to the maximum pattern size are
requeued, since producer changes or reduced use counts can unlock deeper patterns.
Queue deduplication avoids redundant pending entries; removed IDs are skipped.
Traversal and rule priority are deterministic.

Every rule replaces at least two nodes with exactly one node. Thus every
successful rewrite strictly decreases live node count, and a graph with N nodes
admits at most N-1 rewrites. This is a termination policy, **not a runtime bound**:
guards, fanout, structural checks, and worklist invalidation still cost time.
Expansion rules and one-node rewrites are deliberately rejected.

For match-only workloads, let U be the number of distinct structural plans in a
root bucket and P the maximum pattern size. Structure work is bounded by the
sum over visited roots of U times pattern/boundary checking cost, plus passing
plan guard costs. Boundary checks include matched-node fanout; they are not
constant-time. The reference mode performs the same checks per rule instead,
with cheap root op/domain rejection before allocating binding buffers.
Rewrite workloads additionally pay edge updates and bounded descendant walks;
large fanout can make these expensive. Arbitrary trusted guards have no imposed
time bound. Consequently timings below are empirical observations, never
mathematical worst-case guarantees for arbitrary rules or models.

## Intentional limitations and precision

Matched nodes may have multiple outputs, but replacements remain single-node
and must preserve the root's complete ordered output list. Optional/missing
inputs or outputs, variadic patterns,
model-local function rewriting, tensor synthesis, domain-import edits, alternative
patterns and implicit commutation are unsupported. Control-flow nodes are not
matched, but `Apply` rewrites inside their subgraphs. External tensor data and
overridable initializers do not satisfy example constant guards; scalar constants
must be finite float32 rank-zero initializers. Singleton vectors are deliberately
not treated as scalars, since broadcasting can otherwise change the output rank.
Constant nodes require an earlier constant-folding pass.

The examples support a conservative subset, not every ONNXScript variant:
RMSNorm requires the last axis, keepdims, positive finite epsilon, float32,
known positive-width matching one-dimensional scale, and pre-opset-18 attribute
axes. Transpose requires known rank >=2 and exactly the last-two-axis swap.
RMSNorm declares ONNX opsets 7 through 17. MatMul Div requires a finite nonzero
scalar and finite reciprocal. The FusedMatMul examples declare ONNX opsets 9
through 21 and require an existing `com.microsoft` opset-1 import; they do not add
one. The upper bounds are conservative supported ranges, not inferred schema
version boundaries; extending them requires reviewing and testing semantics.
HardSwish requires exact float32 constants/attributes and model opset 14 through
21. The tested CPU build executes HardSwish by expanding its ONNX function body
to HardSigmoid/Mul; RMSNorm and FusedMatMul execute native CPU kernels.
Floating-point fusions change evaluation order; equivalence means tested
tolerance, not bitwise identity. NaNs/infinities and every extreme finite input
are not claimed to have bitwise-equivalent behavior.

## Correctness and performance reproduction

The normal tests live in `onnxruntime/test/optimizer/pattern_rewriter_test.cc`.
They include executed CPU equivalence and structural assertions, not only match
counts. Use an isolated build directory and virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe tools\ci_build\build.py --build_dir build\rewriter --config Release --update --parallel 4 --skip_tests --cmake_generator "Visual Studio 17 2022" --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
.\.venv\Scripts\python.exe tools\ci_build\build.py --build_dir build\rewriter --config Release --build --parallel 4 --target onnxruntime_test_all --skip_tests
Set-Location build\rewriter\Release\Release
.\onnxruntime_test_all.exe --gtest_filter="PatternRewriterTests.*"
$env:ORT_REWRITER_BENCHMARK_CSV = "C:\absolute\path\rewriter.csv"
.\onnxruntime_test_all.exe --gtest_also_run_disabled_tests --gtest_filter="PatternRewriterBenchmark.DISABLED_Decoder"
```

Then, from the repository root:

```powershell
.\.venv\Scripts\python.exe tools\ci_build\summarize_rewriter_benchmark.py C:\absolute\path\rewriter.csv
```

Ensure CMake is on PATH (the measurement host used Visual Studio's bundled CMake).
`FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER` avoids accidentally importing an
incompatible installed dependency; this host initially found a Debug-only protobuf
installation while building Release. Dependencies are built inside this isolated
build tree instead of modifying the shared installation.

The benchmark is a disabled Google Test so it uses the exact same linked ORT graph,
schema, optimizer and CPU components as correctness tests; it is not a mock matcher
microbenchmark. The script verifies the full matrix, repeat counts, deterministic
counters, and indexed/reference agreement on matches, rewrites and output counts.

## Decoder topology and measurement methodology

`BuildDecoder` is a **hand-constructed, model-derived eager Llama-style topology**,
not an exported pretrained model or a claim of identical exporter node order.
Sources at ORT base revision
[`42518d89d7c5393baf97d69f8c2670668c40decf`](https://github.com/microsoft/onnxruntime/tree/42518d89d7c5393baf97d69f8c2670668c40decf)
are `onnxruntime/python/tools/transformers/models/llama/convert_to_onnx.py`
(eager decoder, dynamic past/present KV exports),
`models/llama/llama_inputs.py` (cache layout),
`onnxruntime/test/optimizer/group_query_attention_pre_norm_fusion_test.cc`
(small Q/K/V/head/cache geometry), and ONNXScript's RMSNorm source above.

Each block has two decomposed RMSNorms, Q/K/V projections, reshape/transposes,
split-half RoPE via Slice/Neg/Concat and gathered sine/cosine position caches,
past/current KV concatenation with per-layer graph outputs, grouped-head expansion,
QK transpose/MatMul/scaling/causal-mask/Softmax/AV attention, output projection,
two residual adds, and a SiLU-gated up/gate/down MLP. B=1, query heads=4,
KV heads=2, head size=4, hidden width=16, MLP width=64, past length=3.
Sine/cosine caches use standard paired frequencies; weights are tiny random
float32 tensors. No tokenizer, embedding lookup, LM head, dynamic position/cache
control flow, mixed precision, quantization or production tensor sizes are modeled.
Both sequence=1 decode and sequence=4 causal prefill have executed equivalence tests.

Benchmarks use decode sequence=1 and 1/8/32 layers, with 1/10/100/300/500 rules:

| Workload number | Meaning |
|---|---|
| 0 | Match-only, active examples last, realistic semantic guards |
| 1 | Match-only adversarial same-root/same-structure misses after evaluating genuine guards |
| 2 | Successful rewriting, active examples last |
| 3 | Synthetic distinct same-Mul-root nonmatches: 9 unary nodes with 500 different Sigmoid/Tanh combinations |

Workload 3 is explicitly **synthetic stress**, not a Llama operator distribution.
Hundreds of realistic rules are simulated by topology-equivalent guarded variants
of six genuine patterns, not claimed to be hundreds of independently authored
fusion algorithms. Workload 3 exposes the lack of shared-prefix compilation.
For one rule, RMSNorm is selected; active examples rotate consistently thereafter.

Rule creation/compilation, engine execution, and final Resolve are separately
timed with `steady_clock`. Generation, model copying/loading, initial Resolve,
and random feeds are excluded. Engine time includes per-graph rule import filtering,
producer-index/worklist creation, structural matching, guards, boundary checks, edge mutation and
invalidation; it excludes final Resolve. The timed graphs use an error-severity
logger to exclude unused-initializer warning I/O, without suppressing failed
statuses. Read-only Resolve is usually a no-op.
Two warmups and nine measured runs use a fresh graph each time; the CSV records
all repeats, counters and output node counts. Medians and **maximum of nine**,
not an invented statistically significant p99, are reported.

`visited` counts live worklist roots, including revisits. `candidates` counts
enumerated rule IDs (all rules in the reference mode, root-indexed rules in the
indexed mode); `structural_checks` counts plan attempts, including cheap root
rejections in the reference. `guards` counts semantic callback evaluations after
structural and deletion-boundary checks. `matches` counts all passing guards in
match-only mode, or selected winners in rewrite mode; `rewrites` counts completed
mutations. Match-only therefore measures safe rewrite eligibility, not a search
that ignores external consumers. Reference and indexed outputs, matches, rewrites
and visited counts are cross-checked by the summary script.

Measured results and empirical extrapolation are recorded in
[Pattern_Rewriter_Performance.md](Pattern_Rewriter_Performance.md).
