# Pattern rewriter: measured correctness and performance

These are measurements of the real ORT graph rewrite implementation described in
[Pattern_Rewriter.md](Pattern_Rewriter.md), not a standalone matcher simulation.
The implementation is experimental and opt-in; the default optimizer is unchanged.
Raw measurements are in [Pattern_Rewriter_Benchmark.csv](Pattern_Rewriter_Benchmark.csv).

## Configuration and scope

Measured October 13, 2026, after migrating applicability to rule-level import
ranges, on Windows 11 Enterprise 10.0.26100, Dell Precision
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
| Active match-only | 56.1170 | 11.1454 | 12.1992 | 5.03x | 3.0375 | 0.0109 | 56.5110 | 14.1701 |
| Same-structure semantic misses | 52.7693 | 11.2454 | 12.9956 | 4.69x | 2.9337 | 0.0133 | 53.1632 | 14.6545 |
| Successful rewriting | 59.0984 | 13.8497 | 17.9727 | 4.27x | 2.8381 | 15.9662 | 79.3526 | 33.2214 |
| Synthetic distinct structural misses | 61.1454 | 50.4680 | 54.7899 | 1.21x | 8.4203 | 0.0219 | 62.0567 | 59.2537 |

The synthetic case improves only about **5% end-to-end** when preparation is included.
Distinct plans do not share prefixes, and compilation is more expensive than the
reference. This modest gain is sensitive to host noise; the earlier node-version
implementation was slightly slower end-to-end on the same workload. Root indexing
alone does not solve every large-rule-set workload.
Conversely, an immutable compiled transformer can amortize preparation across
graphs; the benchmark deliberately charges preparation to every run.

Rule-count scaling for successful rewriting on the same 32-layer graph:

| Rules | Reference engine median ms | Indexed engine median ms | Indexed preparation median ms | Indexed Resolve median ms | Rewrites |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.2304 | 4.2336 | 0.0487 | 21.3210 | 64 |
| 10 | 5.9163 | 5.8939 | 0.1142 | 19.0531 | 128 |
| 100 | 16.1076 | 7.6142 | 0.6775 | 21.1160 | 128 |
| 300 | 39.5080 | 12.7172 | 2.0887 | 19.4651 | 128 |
| 500 | 59.0984 | 13.8497 | 2.8381 | 15.9662 | 128 |

One rule includes only RMSNorm and therefore performs fewer rewrites. Indexing has
overhead: at one rule, active match-only on 32 layers measured 2.1630 ms indexed
versus 1.7440 ms reference. There is no claim that indexing always wins.

Layer scaling with 500 rules, indexed implementation (median / maximum of nine):

| Layers | Workload | Engine ms | Final Resolve ms | Preparation ms | Total ms |
|---:|---|---:|---:|---:|---:|
| 1 | Active match-only | 0.3044 / 0.3558 | 0.0003 / 0.0003 | 2.7565 / 3.0317 | 3.0562 / 3.3878 |
| 8 | Active match-only | 2.5450 / 2.8323 | 0.0021 / 0.0038 | 3.0551 / 3.7659 | 5.6184 / 6.4384 |
| 32 | Active match-only | 11.1454 / 12.1992 | 0.0109 / 0.0224 | 3.0375 / 3.8524 | 14.1701 / 16.0740 |
| 1 | Semantic misses | 0.3058 / 0.3420 | 0.0003 / 0.0004 | 2.7639 / 3.0397 | 3.0661 / 3.3771 |
| 8 | Semantic misses | 2.7329 / 3.6636 | 0.0027 / 0.0108 | 3.1306 / 4.0923 | 5.8990 / 7.7667 |
| 32 | Semantic misses | 11.2454 / 12.9956 | 0.0133 / 0.0450 | 2.9337 / 3.6696 | 14.6545 / 16.3258 |
| 1 | Rewriting | 0.3655 / 0.4051 | 0.3925 / 0.4417 | 2.7517 / 3.0277 | 3.5067 / 3.8745 |
| 8 | Rewriting | 3.1418 / 3.4517 | 3.7279 / 5.3076 | 3.0703 / 3.6653 | 10.2405 / 11.4026 |
| 32 | Rewriting | 13.8497 / 17.9727 | 15.9662 / 19.7425 | 2.8381 / 3.9390 | 33.2214 / 37.1322 |
| 1 | Synthetic misses | 1.6808 / 1.7417 | 0.0006 / 0.0011 | 8.2579 / 8.8746 | 9.8978 / 10.5562 |
| 8 | Synthetic misses | 14.0484 / 18.8847 | 0.0057 / 0.0099 | 8.1347 / 8.7927 | 22.1533 / 27.6270 |
| 32 | Synthetic misses | 50.4680 / 54.7899 | 0.0219 / 0.0277 | 8.4203 / 10.6919 | 59.2537 / 62.7946 |

These maxima are observed tails, not p99 estimates. With nine repetitions and an
unisolated host, occasional scheduling outliers are expected; raw rows are retained.
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
| Active match-only | 7.7048 | 60.9960 | 0.1120 | 68.8128 |
| Semantic misses | 8.1846 | 73.2720 | 0.2250 | 81.6816 |
| Rewriting | 7.8780 | 89.8635 | 106.1520 | 203.8935 |
| Synthetic distinct structural misses | 21.3838 | 377.6940 | 0.1980 | 399.2758 |

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

Results: configuration and CPU Release build **passed**; **32 correctness/regression
tests passed** (21 rewriter tests, 11 existing MatMulScaleFusion/QuickGelu tests);
the disabled benchmark **passed** in 39.511 seconds, including untimed work.
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
