# Pattern rewriter: measured correctness and performance

These are measurements of the real ORT graph rewrite implementation described in
[Pattern_Rewriter.md](Pattern_Rewriter.md), not a standalone matcher simulation.
The implementation is experimental and opt-in; the default optimizer is unchanged.
Raw measurements are in [Pattern_Rewriter_Benchmark.csv](Pattern_Rewriter_Benchmark.csv).

## Configuration and scope

Measured October 12, 2026, on Windows 11 Enterprise 10.0.26100, Dell Precision
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
producer indexing, worklist construction, structural matching, guards, boundary
checks, mutation and invalidation. Final Resolve is separate; error-level logging
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
| Active match-only | 52.6198 | 11.7152 | 13.6583 | 4.49x | 3.9925 | 0.0120 | 52.9667 | 16.0010 |
| Same-structure semantic misses | 52.0305 | 12.1930 | 13.5606 | 4.27x | 4.1632 | 0.0140 | 52.3255 | 16.2942 |
| Successful rewriting | 56.6447 | 15.2342 | 15.5768 | 3.72x | 3.6603 | 15.7775 | 73.8887 | 35.1392 |
| Synthetic distinct structural misses | 60.1767 | 51.4131 | 56.3220 | 1.17x | 10.7039 | 0.0198 | 61.2582 | 62.5633 |

The synthetic case is slightly **slower end-to-end** when preparation is included.
Distinct plans do not share prefixes, and compilation is more expensive than the
reference. Root indexing alone does not solve every large-rule-set workload.
Conversely, an immutable compiled transformer can amortize preparation across
graphs; the benchmark deliberately charges preparation to every run.

Rule-count scaling for successful rewriting on the same 32-layer graph:

| Rules | Reference engine median ms | Indexed engine median ms | Indexed preparation median ms | Indexed Resolve median ms | Rewrites |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.6076 | 3.4208 | 0.0526 | 15.8041 | 64 |
| 10 | 5.3811 | 4.9361 | 0.1267 | 15.5876 | 128 |
| 100 | 15.0763 | 6.4233 | 0.7810 | 15.4729 | 128 |
| 300 | 35.7440 | 10.6578 | 2.3393 | 15.9817 | 128 |
| 500 | 56.6447 | 15.2342 | 3.6603 | 15.7775 | 128 |

One rule includes only RMSNorm and therefore performs fewer rewrites. Indexing has
overhead: at one rule, active match-only on 32 layers measured 1.6214 ms indexed
versus 1.4410 ms reference. There is no claim that indexing always wins.

Layer scaling with 500 rules, indexed implementation (median / maximum of nine):

| Layers | Workload | Engine ms | Final Resolve ms | Preparation ms | Total ms |
|---:|---|---:|---:|---:|---:|
| 1 | Active match-only | 0.3036 / 0.3411 | 0.0003 / 0.0004 | 3.8058 / 4.0849 | 4.1085 / 4.3888 |
| 8 | Active match-only | 2.4784 / 2.9074 | 0.0014 / 0.0018 | 3.7039 / 4.1180 | 6.3529 / 6.6130 |
| 32 | Active match-only | 11.7152 / 13.6583 | 0.0120 / 0.0265 | 3.9925 / 4.6410 | 16.0010 / 17.5769 |
| 1 | Semantic misses | 0.3036 / 0.3330 | 0.0003 / 0.0003 | 3.7988 / 3.9937 | 4.1027 / 4.3151 |
| 8 | Semantic misses | 2.4457 / 2.9943 | 0.0016 / 0.0019 | 3.8615 / 4.2955 | 6.3502 / 6.8849 |
| 32 | Semantic misses | 12.1930 / 13.5606 | 0.0140 / 0.0202 | 4.1632 / 4.5481 | 16.2942 / 17.7423 |
| 1 | Rewriting | 0.3607 / 0.3921 | 0.3916 / 0.4191 | 3.7945 / 3.8828 | 4.5527 / 4.6940 |
| 8 | Rewriting | 3.0267 / 3.5526 | 3.3116 / 3.8850 | 3.8548 / 4.1095 | 10.4697 / 10.8026 |
| 32 | Rewriting | 15.2342 / 15.5768 | 15.7775 / 17.8379 | 3.6603 / 4.1761 | 35.1392 / 36.4129 |
| 1 | Synthetic misses | 1.5513 / 1.6795 | 0.0004 / 0.0162 | 10.4264 / 10.8349 | 12.1063 / 12.3867 |
| 8 | Synthetic misses | 12.4773 / 13.0209 | 0.0018 / 0.0026 | 10.3402 / 11.3380 | 22.8945 / 23.9891 |
| 32 | Synthetic misses | 51.4131 / 56.3220 | 0.0198 / 0.0312 | 10.7039 / 11.3868 | 62.5633 / 67.5525 |

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
| Active match-only | 9.2820 | 68.2915 | 0.1325 | 77.7060 |
| Semantic misses | 9.0962 | 67.8030 | 0.1010 | 77.0002 |
| Rewriting | 8.3522 | 77.8840 | 89.1895 | 175.4257 |
| Synthetic distinct structural misses | 22.7736 | 281.6100 | 0.1560 | 304.5396 |

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

Results: configuration and CPU Release build **passed**; **27 correctness/regression
tests passed** (16 new tests, 11 existing MatMulScaleFusion/QuickGelu tests);
the disabled benchmark **passed** in 34.583 seconds, including untimed work.
No full ORT suite or GPU build was run.

Tests cover positive/negative structural matches, repeated bindings and a shared-node
diamond, graph outputs, fanout, implicit subgraph captures, leaf multi-output
producers, layer annotations, opsets, invalid patterns/recipes/attributes, provider/type
restrictions, immutable scalar constants, broadcasting rank, priority, overlap,
mutation, repeated application and termination. Executed tests cover HardSwish,
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
