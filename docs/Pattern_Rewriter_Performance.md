# Pattern rewriter: measured correctness and performance

These are measurements of the real ORT graph rewrite implementation described in
[Pattern_Rewriter.md](Pattern_Rewriter.md), not a standalone matcher simulation.
The implementation is experimental and opt-in; the default optimizer is unchanged.
Raw measurements are in [Pattern_Rewriter_Benchmark.csv](Pattern_Rewriter_Benchmark.csv).
The original tables below record revision
`8a2b73c7e7818925c161d1fd7e4c7ce93f0acba6`; they are retained as historical measurements.
The named-value/parser/attribute-matcher follow-up is measured separately in
[Pattern_Rewriter_Language_Benchmark.csv](Pattern_Rewriter_Language_Benchmark.csv)
and the next section, rather than silently replacing the original measurements.

## Named-value and pattern-language follow-up

The same CPU Release build configuration, decoder generator, full 1080-row
matrix, two warmups and nine repeats were rerun after adding named handles,
scalar literal constraints, attribute predicates and the text parser.
HardSigmoid/Mul now uses node attribute matchers instead of equivalent guard
checks. At the measured revision, the benchmark constructs its patterns through the C++ builder;
preparation includes building/compiling those rules, **not text parsing latency**.
The subsequent string-constructor migration converts all six example rules to
text definitions. Those historical preparation timings therefore do not include
the current parser cost; graph matching and mutation semantics are unchanged.
The parser is exercised by correctness tests, including an executed HardSwish fusion.
No decoder operator distribution or benchmark workload definition changed.

At 32 layers / 2241 nodes / 500 rules, all times are milliseconds:

| Workload | Reference engine median | Indexed engine median | Indexed engine max | Indexed preparation median | Indexed Resolve median | Indexed total median |
|---|---:|---:|---:|---:|---:|---:|
| Active match-only | 64.5107 | 20.7664 | 25.0146 | 4.0833 | 0.0283 | 24.8781 |
| Adversarial semantic misses | 64.2870 | 19.6209 | 21.4712 | 3.8433 | 0.0302 | 23.4582 |
| Successful rewriting | 73.3457 | 25.0688 | 45.5535 | 3.9737 | 22.3129 | 51.9711 |
| Synthetic distinct structural misses | 67.3232 | 55.9487 | 79.9540 | 11.0489 | 0.0287 | 66.8546 |

The largest indexed engine observation across the complete matrix was 79.9540 ms.
This remains empirical evidence for subsecond cost at the tested scale, not a
guarantee for arbitrary patterns, callbacks or graphs. Names increase pattern
storage and copying cost. Attribute-predicate plans deliberately do not share
structural matches, so a large set of attribute-heavy rules can cost more than
the measured decoder rule variants. For example, the 500-rule active match-only
indexed run now performs 27,264 structural checks rather than the historical
1,024, while guards (10,688), matches (128), visits (2,241) and output nodes
(2,241) are unchanged. Successful rewriting still removes 128 patterns and
leaves 1,793 nodes. The earlier extrapolation budgets below
belong to the historical revision and should not be treated as updated budgets.

Validation: CPU Release incremental build passed; **46 correctness/regression
tests passed** (35 rewriter tests and 11 existing optimizer tests); the full
benchmark passed. The summarizer checked nine repeats in every group,
deterministic counters and indexed/reference agreement. Numerical decoder
results remained max absolute error 0 for decode and 2.98023e-08 for prefill;
prefill max relative error with denominator floor 1e-6 was 2.73599e-07.

Reproduction from the worktree, with CMake available on PATH:

```powershell
.\.venv\Scripts\python.exe tools\ci_build\build.py --build_dir build\rewriter --config Release --build --parallel 4 --target onnxruntime_test_all --skip_tests
$env:ORT_REWRITER_BENCHMARK_CSV = Join-Path (Get-Location) 'docs\Pattern_Rewriter_Language_Benchmark.csv'
Push-Location build\rewriter\Release\Release
.\onnxruntime_test_all.exe --gtest_filter="PatternRewriterTests.*:GraphTransformationTests.MatMulScaleFusion*:GraphTransformationTests.QuickGelu*"
.\onnxruntime_test_all.exe --gtest_also_run_disabled_tests --gtest_filter="PatternRewriterBenchmark.DISABLED_Decoder"
Pop-Location
.\.venv\Scripts\python.exe tools\ci_build\summarize_rewriter_benchmark.py docs\Pattern_Rewriter_Language_Benchmark.csv
```

All four commands passed for the recorded follow-up measurements.

## Configuration and scope

Measured October 13, 2026, after adding multi-output patterns, concise constructors
and generic EP policy, on Windows 11 Enterprise 10.0.26100, Dell Precision
5820 Tower X-Series, Intel Core i9-10920X (nominal 3.50 GHz, 24 logical processors),
approximately 64 GiB RAM. ORT base revision:
`42518d89d7c5393baf97d69f8c2670668c40decf`, version 1.32.0.
CPU/static Release build, contrib enabled, MSVC 19.44.35228.0, Visual Studio 2022,
CMake 3.31.6-msvc6, Python 3.13. Four build workers were used; compilation finished
before measurement. No GPU, CPU affinity, or dedicated-host isolation was used.
Matcher execution and graph mutation are single-threaded.

The graphs are hand-constructed, model-derived Llama-style decoders, **not
pretrained model exports**. Each layer contains RoPE, GQA, past/present KV caches,
two RMSNorms, attention, residuals and a SiLU-gated MLP. Small float32 weights
preserve the topology, not production tensor sizes. The design document provides
source revisions, geometry, construction details and omitted exporter features.

| Layers | Input nodes | Graph inputs | Graph outputs | Nodes after active 500-rule rewrite | Rewrites |
|---:|---:|---:|---:|---:|---:|
| 1 | 71 | 3 | 3 | 57 | 4 |
| 8 | 561 | 17 | 17 | 449 | 32 |
| 32 | 2241 | 65 | 65 | 1793 | 128 |

There are 70 nodes per layer plus one final Identity. The per-layer distribution
is Add 7, Concat 4, Div 1, Expand 2, Gather 4, MatMul 9, Mul 10, Neg 2, Pow 2,
Reciprocal 2, ReduceMean 2, Reshape 12, Sigmoid 1, Slice 4, Softmax 1, Sqrt 2,
Transpose 5. Present K/V are graph outputs, not disposable intermediates.

## Methodology

The complete matrix has 1/8/32 layers, 1/10/100/300/500 rules, four workloads and
indexed/reference implementations: 120 groups, nine measured repetitions each,
**1080 CSV rows**. Each group has two additional warmups. Every run uses a fresh
copied/resolved graph. Generation, loading, copying, initial Resolve and tensor
execution are excluded from core timings.

Preparation includes rule construction and compilation. Engine time includes
per-graph rule import filtering, producer indexing, worklist construction, structural
matching, guards, boundary checks, mutation and invalidation. Final Resolve is separate; error-level logging
excludes warning output for removed initializers. All failed statuses remain
fatal test failures. Total times below are medians of the per-repeat sum, not sums
of independently selected medians.

The rule-by-rule reference uses the same guards, matcher, mutation and worklist,
with cheap root rejection before binding allocation. It does not compile root
indexes or shared structural groups. This isolates dispatch/structural sharing,
not differences in mathematical rules or an artificially allocation-heavy baseline.

Hundreds of realistic rules are guarded variants of six genuine ONNXScript-derived
patterns; active rules are last. They are **not hundreds of independent fusion
algorithms**. The adversarial workload evaluates real semantic guards and then
rejects otherwise eligible matches by a deliberately absent node name. A separate
synthetic workload has distinct same-Mul-root patterns with nine Sigmoid/Tanh nodes;
it exposes the lack of arbitrary-prefix sharing. Neither nonmatch workload consists
only of inert different-root rules.

## Results

At 32 layers / 2241 nodes / 500 rules:

| Workload | Reference engine median ms | Indexed engine median ms | Indexed engine max ms | Engine speedup | Indexed preparation median ms | Indexed Resolve median ms | Reference total median ms | Indexed total median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Active match-only | 70.6460 | 13.3980 | 22.2259 | 5.27x | 4.3209 | 0.0208 | 71.4841 | 18.8780 |
| Same-structure semantic misses | 72.5865 | 11.7040 | 28.8747 | 6.20x | 3.2723 | 0.0162 | 73.1829 | 15.0976 |
| Successful rewriting | 76.4116 | 16.3844 | 33.4006 | 4.66x | 3.4737 | 19.8251 | 97.6291 | 42.6921 |
| Synthetic distinct structural misses | 72.6219 | 74.4585 | 86.9363 | 0.98x | 9.0458 | 0.0267 | 74.5306 | 83.8081 |

The synthetic case is **slower end-to-end** when preparation is included.
Distinct plans do not share prefixes, and compilation is more expensive than the
reference. These results are sensitive to host noise; an earlier import-filtered
implementation was modestly faster on the same workload. Root indexing
alone does not solve every large-rule-set workload.
Conversely, an immutable compiled transformer can amortize preparation across
graphs; the benchmark deliberately charges preparation to every run.

Rule-count scaling for successful rewriting on the same 32-layer graph:

| Rules | Reference engine median ms | Indexed engine median ms | Indexed preparation median ms | Indexed Resolve median ms | Rewrites |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.9963 | 5.4241 | 0.0575 | 24.8174 | 64 |
| 10 | 6.0513 | 6.3679 | 0.1280 | 22.1893 | 128 |
| 100 | 17.2254 | 6.8823 | 0.7320 | 19.0990 | 128 |
| 300 | 39.1072 | 12.5730 | 2.1582 | 20.0874 | 128 |
| 500 | 76.4116 | 16.3844 | 3.4737 | 19.8251 | 128 |

One rule includes only RMSNorm and therefore performs fewer rewrites. Indexing has
overhead: at one rule, active match-only on 32 layers measured 2.4979 ms indexed
versus 2.4580 ms reference. There is no claim that indexing always wins.

Layer scaling with 500 rules, indexed implementation (median / maximum of nine):

| Layers | Workload | Engine ms | Final Resolve ms | Preparation ms | Total ms |
|---:|---|---:|---:|---:|---:|
| 1 | Active match-only | 0.3494 / 0.3716 | 0.0004 / 0.0008 | 3.3583 / 3.5701 | 3.7042 / 3.9421 |
| 8 | Active match-only | 3.3238 / 4.6376 | 0.0040 / 0.0070 | 3.5224 / 4.1014 | 6.7686 / 8.2005 |
| 32 | Active match-only | 13.3980 / 22.2259 | 0.0208 / 0.0237 | 4.3209 / 11.9689 | 18.8780 / 34.2185 |
| 1 | Semantic misses | 0.3672 / 0.6532 | 0.0005 / 0.0007 | 3.3979 / 4.9755 | 3.8006 / 5.3490 |
| 8 | Semantic misses | 2.8641 / 8.2150 | 0.0057 / 0.0109 | 3.5428 / 13.3055 | 6.8985 / 21.5277 |
| 32 | Semantic misses | 11.7040 / 28.8747 | 0.0162 / 0.0293 | 3.2723 / 3.4929 | 15.0976 / 32.3371 |
| 1 | Rewriting | 0.4197 / 0.6654 | 0.4468 / 1.9302 | 3.3783 / 5.2108 | 4.2666 / 6.3475 |
| 8 | Rewriting | 3.4031 / 4.7259 | 4.0870 / 5.0392 | 3.8477 / 4.9182 | 11.5174 / 13.1849 |
| 32 | Rewriting | 16.3844 / 33.4006 | 19.8251 / 33.2741 | 3.4737 / 5.3419 | 42.6921 / 67.2441 |
| 1 | Synthetic misses | 1.8062 / 2.1824 | 0.0009 / 0.0014 | 9.5869 / 13.4720 | 11.3684 / 15.2904 |
| 8 | Synthetic misses | 15.3799 / 18.3429 | 0.0068 / 0.0076 | 10.7740 / 27.7695 | 26.1607 / 46.1200 |
| 32 | Synthetic misses | 74.4585 / 86.9363 | 0.0267 / 0.0297 | 9.0458 / 10.3327 | 83.8081 / 96.4596 |

These maxima are observed tails, not p99 estimates. With nine repetitions and an
unisolated host, occasional scheduling outliers are expected; raw rows are retained.
This refresh has visibly larger outliers in both implementations than the earlier
measurements. Differences between separately measured commits do not isolate the
cost of the multi-output support or generic EP policy.
Final Resolve is a significant part of successful rewriting and cannot be ignored
when estimating initialization cost. Tensor inference speed is not benchmarked.

### Counters and agreement

Counters were deterministic across all repeats, and the summarizer verified that
indexed/reference live visits, matches, rewrites and output nodes agree in every
group. Representative 32-layer / 500-rule counters:

| Workload | Implementation | Live visits | Candidates | Structural checks | Guards | Matches | Rewrites | Output nodes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Active match-only | Indexed | 2241 | 85344 | 1024 | 10688 | 128 | 0 | 2241 |
| Semantic misses | Indexed | 2241 | 85344 | 1024 | 10688 | 0 | 0 | 2241 |
| Rewriting | Indexed | 2464 | 85344 | 1024 | 10688 | 128 | 128 | 1793 |
| Synthetic misses | Indexed | 2241 | 160000 | 160000 | 0 | 0 | 0 | 2241 |

Reference structural checks are 1,120,500 for the three match-only workloads and
1,231,776 for rewriting. The identical-plan compilation keeps realistic structural
checks at 1024 despite 500 guarded rules. Guards remain proportional to eligible
variants: sharing structures does not eliminate arbitrary predicate costs.

## Conservative empirical extrapolation

An 80-layer graph with this exact topology has 5601 input nodes and, after the
active multi-rule rewrite, 4481 nodes and 320 rewrites. This graph was **not measured**.
The following deliberately padded estimates assume the same operator distribution,
bounded fanout, local guard costs, schema imports, CPU and build configuration.

For each engine/Resolve component, take the larger observed maximum-per-layer
from 8 and 32 layers, multiply by 80, then multiply by two. Preparation is charged
once at twice the largest observed maximum across 1/8/32 layers. Sum the independently
derived component budgets (which is more conservative than an observed joint sum):

| 500-rule workload | Preparation budget ms | Engine budget ms | Resolve budget ms | Combined budget ms |
|---|---:|---:|---:|---:|
| Active match-only | 23.9378 | 111.1295 | 0.1400 | 135.2073 |
| Semantic misses | 26.6110 | 164.3000 | 0.2180 | 191.1290 |
| Rewriting | 10.6838 | 167.0030 | 166.3705 | 344.0573 |
| Synthetic distinct structural misses | 55.5390 | 434.6815 | 0.1520 | 490.3725 |

These are empirical planning budgets, **not mathematical worst-case guarantees,
confidence intervals or measured 80-layer results**. A two-times margin is a
heuristic, not a proof. Different exports, large fanout, deep common-prefix
nonmatches, expensive guards, mixed precision, quantization, large initializer
payloads and memory/scheduler pressure can exceed them. Production export benchmarks
are necessary before deploying broadly.

## Correctness evidence and reproduction

Run from the isolated worktree root with Visual Studio's CMake on PATH:

```powershell
.\.venv\Scripts\python.exe tools\ci_build\build.py --build_dir build\rewriter --config Release --update --skip_submodule_sync --parallel 4 --skip_tests --cmake_generator "Visual Studio 17 2022" --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
.\.venv\Scripts\python.exe tools\ci_build\build.py --build_dir build\rewriter --config Release --build --parallel 4 --target onnxruntime_test_all --skip_tests
Set-Location build\rewriter\Release\Release
.\onnxruntime_test_all.exe --gtest_filter="PatternRewriterTests.*:GraphTransformationTests.MatMulScaleFusion*:GraphTransformationTests.QuickGelu*"
$env:ORT_REWRITER_BENCHMARK_CSV = "C:\repos\copilot-worktrees\onnxruntime\gramalingam-bookish-goggles\docs\Pattern_Rewriter_Benchmark.csv"
.\onnxruntime_test_all.exe --gtest_also_run_disabled_tests --gtest_filter="PatternRewriterBenchmark.DISABLED_Decoder"
```

Results: configuration and CPU Release build **passed**; **41 correctness/regression
tests passed** (30 rewriter tests, 11 existing MatMulScaleFusion/QuickGelu tests);
the disabled benchmark **passed** in 48.321 seconds, including untimed work.
No full ORT suite or GPU build was run.

Tests cover positive/negative structural matches, repeated bindings and a shared-node
diamond, graph outputs, fanout, implicit subgraph captures, leaf multi-output
producers, layer annotations, opsets, invalid patterns/recipes/attributes, provider/type
restrictions, immutable scalar constants, broadcasting rank, priority, overlap,
mutation, repeated application and termination. Import-specific tests cover
inclusive range boundaries across schema changes, rejected graphs sharing a
schema with accepted graphs, missing/additional domains, alias normalization,
invalid/duplicate requirements, filtering before guards, priority within shared
plans, and reuse of a compiled transformer across import contexts.
Provider tests cover unassigned/CPU nodes, rejected assigned EPs, explicit generic
EP opt-in, mixed assignments, and policy-before-guard ordering. These are graph
tests; no GPU execution is claimed. Multi-output tests cover exact output counts,
bindings reached through a nonzero output slot, ordered/repeated/shared bindings,
unused outputs, secondary graph outputs/external consumers/implicit captures,
all-root-output preservation, slot-specific fanout, invalid replacement arity,
repeated application, and executed Split/Concat and Identity/Split rewrites.
An executed test-only adaptation of ONNXScript's `LayerNormBiasFusion` matches
three LayerNormalization outputs and fuses Add only when statistics are unobserved.
Executed tests cover HardSwish,
MatMul/Div, Transpose/MatMul, combined Transpose/Div, a retained Dropout output,
and the complete two-layer decoder including all present KV outputs. CPU HardSwish
executes through its ONNX HardSigmoid/Mul function body; the structural test verifies
the HardSwish replacement before session initialization, and the numerical test
checks the expanded body rather than mistaking expansion for a failed fusion.

| Two-layer numerical decoder | Outputs compared | Rewrites | Maximum absolute error | Maximum relative error, denominator floor 1e-6 |
|---|---:|---:|---:|---:|
| Decode, sequence 1 | 5 | 8 | 0 | 0 |
| Causal prefill, sequence 4 | 5 | 8 | 2.98023e-08 | 2.73599e-07 |

Every output element also passes the tolerance `2e-5 + 2e-4 * abs(reference)`.
These random small-weight examples establish executed equivalence for representative
cases, not every finite/extreme input or bitwise identity.

From the repository root, these commands also **passed**:

```powershell
.\.venv\Scripts\python.exe tools\ci_build\summarize_rewriter_benchmark.py docs\Pattern_Rewriter_Benchmark.csv
& "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\Llvm\bin\clang-format.exe" --dry-run --Werror onnxruntime\core\optimizer\pattern_rewriter.h onnxruntime\core\optimizer\pattern_rewriter.cc onnxruntime\core\optimizer\pattern_rewriter_rules.cc onnxruntime\test\optimizer\pattern_rewriter_test.cc
.\.venv\Scripts\python.exe -m ruff check tools\ci_build\summarize_rewriter_benchmark.py
git --no-pager diff --check
```

The summarizer validates the full measured matrix and prints every group, including
the smaller rule counts and baseline rows omitted from the selected tables above.
