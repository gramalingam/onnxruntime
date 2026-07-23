# GroupedMatMul: fused op vs. function-definition expansion — performance

This experiment compares the fused contrib op **`com.microsoft.GroupedMatMul`** against its
documented **function-definition expansion** (the "reference decomposition" in
[`docs/GroupedMatMul.md`](../../docs/GroupedMatMul.md)), which expresses the same
Mixture-of-Experts grouped matmul with standard ONNX ops:

```
idx_flat = Reshape(group_indices, [M*k])
W_sel    = Gather(weights, idx_flat, axis=0)          # [M*k, K, N]  (duplicates weights!)
X        = Reshape(Expand(Unsqueeze(input, 1), [M,k,K]), [M*k, 1, K])
r        = MatMul(X, W_sel)                           # [M*k, 1, N]
r        = r + Unsqueeze(Gather(bias, idx_flat), 1)   # optional
output   = Reshape(r, [M, k, N])
```

## How to run

```bash
# Requires an ONNX Runtime build that contains the GroupedMatMul contrib op.
# CPU:
python benchmark_grouped_matmul.py --providers cpu --csv results_cpu_fp32.csv
# CUDA (on a GPU host with a --use_cuda ORT build):
python benchmark_grouped_matmul.py --providers cuda --csv results_cuda_fp32.csv
# Variants:
python benchmark_grouped_matmul.py --dtype float16
python benchmark_grouped_matmul.py --bias
```

Run from **outside** the repository root, or the local `onnxruntime/` source directory
shadows the installed `onnxruntime` package.

The harness builds one ONNX model per variant (fused vs. decomposition), warms up, then times
`iters` runs and reports mean latency, `decomp/fused` speedup, and the max relative error
between the two outputs (a correctness cross-check). The decomposition's `W_sel` tensor is
`O(M·k·K·N)`; cases whose `W_sel` would exceed `--mem-budget-gb` (default 3 GiB) run the fused
op only and report `OOM-skip` for the decomposition.

## Sizes

Realistic MoE feed-forward shapes: `M` tokens, top-`k` experts per token, hidden `K`, ffn/out
`N`, `num_groups` (G) experts. Chosen so the decomposition's duplicated-weight tensor stays
within the sandbox's memory budget.

## Results

Environment: 4-core x86-64 CPU, 15 GiB RAM, ONNX Runtime 1.29.0 built from this branch
(Release, CPU EP), `ORT_ENABLE_ALL` graph optimization, 3 warmup + 10 timed iterations.
Latency is mean wall-clock milliseconds per `Run`.

### CPU, float32 (no bias)

| case         |    M | k |    K |    N |  G | fused ms | decomp ms | speedup | max rel err |
|--------------|-----:|--:|-----:|-----:|---:|---------:|----------:|--------:|------------:|
| tiny         |  256 | 1 |  512 |  512 |  8 |    1.013 |    16.807 |  16.58x |    4.5e-07  |
| small-dense  |  512 | 1 |  768 |  768 |  8 |    3.759 |    78.156 |  20.79x |    6.5e-07  |
| small-top2   |  512 | 2 |  512 |  512 | 16 |    3.559 |    70.970 |  19.94x |    6.1e-07  |
| medium-dense | 1024 | 1 |  768 |  768 | 16 |    9.475 |   161.765 |  17.07x |    5.8e-07  |
| medium-top2  |  512 | 2 |  768 |  768 | 32 |    9.839 |   178.145 |  18.11x |    7.0e-07  |
| large-tokens | 2048 | 1 |  512 |  512 | 32 |    8.652 |   152.460 |  17.62x |    5.3e-07  |
| wide-hidden  |  256 | 2 | 1024 | 1024 |  8 |    7.917 |   146.166 |  18.46x |    6.1e-07  |
| many-experts |  512 | 1 |  512 |  512 | 64 |    4.532 |    42.236 |   9.32x |    5.0e-07  |

### CPU, float32 (with bias)

| case         | fused ms | decomp ms | speedup |
|--------------|---------:|----------:|--------:|
| tiny         |    1.029 |    16.930 |  16.45x |
| small-dense  |    3.789 |    78.884 |  20.82x |
| small-top2   |    3.655 |    71.043 |  19.44x |
| medium-dense |    8.297 |   161.592 |  19.48x |
| medium-top2  |   10.255 |   177.944 |  17.35x |
| large-tokens |    9.340 |   152.366 |  16.31x |
| wide-hidden  |    8.267 |   149.062 |  18.03x |
| many-experts |    5.065 |    41.696 |   8.23x |

### CPU, float16 (no bias)

| case         | fused ms | decomp ms | speedup | max rel err |
|--------------|---------:|----------:|--------:|------------:|
| tiny         |    1.306 |    16.945 |  12.97x |    2.9e-04  |
| small-dense  |    4.586 |    81.234 |  17.71x |    4.8e-04  |
| small-top2   |    4.271 |    72.485 |  16.97x |    2.8e-04  |
| medium-dense |    9.788 |   169.004 |  17.27x |    4.4e-04  |
| medium-top2  |   15.022 |   187.968 |  12.51x |    4.8e-04  |
| large-tokens |   10.814 |   151.038 |  13.97x |    5.4e-04  |
| wide-hidden  |    9.220 |   153.267 |  16.62x |    3.6e-04  |
| many-experts |    9.199 |    44.095 |   4.79x |    2.9e-04  |

Raw CSVs: `results_cpu_fp32.csv`, `results_cpu_fp32_bias.csv`, `results_cpu_fp16.csv`.

### CUDA

CUDA numbers were **not** collected: this sandbox has no GPU (`nvidia-smi` absent) and the
installed build is CPU-only. The harness fully supports CUDA — on a GPU host, build ORT with
`--use_cuda` and run `--providers cuda`. The relative ranking is expected to be even more
lopsided on GPU: the decomposition's `Gather` materialises a fresh `[M·k, K, N]` weight tensor
(hundreds of MB–GB of extra HBM traffic per run), whereas the fused kernel reads each weight
matrix once.

## Conclusions

- **The fused `GroupedMatMul` is ~8–21× faster than its standard-op decomposition on CPU**
  across all tested shapes and dtypes, confirming the motivation for a dedicated op.
- **Correctness matches**: max relative error is ~5e-07 (float32) / ~5e-04 (float16),
  i.e. the fused op and the decomposition compute the same result within numerical tolerance.
- **The decomposition's cost is dominated by the `Gather` that duplicates weights**
  (`W_sel` is `O(M·k·K·N)`). This is why its latency scales with `M·k·K·N` rather than with
  the actual GEMM work, and why the `many-experts` case (small `M·k·K·N`, many groups) shows
  the smallest gap — there the fused op's per-group sort/dispatch overhead is relatively larger
  while the decomposition duplicates comparatively little weight data.
- **Memory blow-up is the harder limit**: at larger MoE shapes the decomposition's `W_sel`
  tensor exceeds available memory (many GB), so the decomposition cannot run at all while the
  fused op runs comfortably. The `--mem-budget-gb` guard makes this explicit (`OOM-skip`).

---

# MoE fusion microbenchmarks (candidate fusions beyond GroupedMatMul)

`benchmark_moe_fusions.py` measures the two non-GEMM op clusters that remain in a standard
top-k MoE feed-forward layer after the grouped GEMMs are handled by `GroupedMatMul`. These are
the fusion candidates identified in the pre-implementation analysis. Each cluster is timed *in
isolation* to quantify the headroom a dedicated fused kernel could recover. The harness uses
only standard ONNX ops plus `com.microsoft.QuickGelu` (which equals SiLU at `alpha=1`), so it
runs against a **stock** ONNX Runtime build — it does **not** need the GroupedMatMul op.

## How to run

```bash
python benchmark_moe_fusions.py --providers cpu                 # both candidates
python benchmark_moe_fusions.py --bench swiglu --dtype float16
python benchmark_moe_fusions.py --bench router --csv router.csv
python benchmark_moe_fusions.py --providers cuda               # on a GPU host
```

## Candidate A — SwiGLU gated activation `Mul(SiLU(g), u)`

This touches the layer's largest intermediates: two `[T, F]` tensors (`T` = token·expert-slot
rows, `F` = FFN inner dim). Three numerically-equivalent-in-math variants bracket the fusion
win:

- `unfused`   : `Sigmoid(g) → Mul(g,·) → Mul(·,u)` — 3 elementwise passes (naive graph).
- `quickgelu` : `QuickGelu(g,alpha=1) → Mul(·,u)` — 2 passes (SiLU as one contrib op).
- `fused_lb`  : `Mul(g,u)` — 1 pass; a **lower bound** for a single fused SwiGLU kernel (same
  memory traffic, minus the cheap sigmoid arithmetic).

`unfused/lb` is the bandwidth headroom a fused gated-activation kernel (or a GroupedMatMul
activation epilogue) targets.

## Candidate B — Router `Softmax → TopK (→ renormalize)`

Softmax is monotonic, so `TopK(Softmax(logits))` selects the same experts as `TopK(logits)`,
and *renormalized* top-k of a full softmax equals a softmax over just the top-k logits
(`p_i/Σ_{j∈topk}p_j = e^{l_i}/Σ_{j∈topk}e^{l_j}`). Two equivalent strategies are timed:

- `naive` : `Softmax(logits)[M,E] → TopK(k) → ReduceSum → Div` — full `E`-wide softmax.
- `fused` : `TopK(logits,k) → Softmax([M,k])` — softmax on `k ≪ E` only.

`E` (#experts) is small, so this is a launch-latency / intermediate-allocation win, not an
arithmetic one. The harness confirms the two strategies agree (max rel err ≈ 1e-7 fp32).

## Results

Environment: 4-core x86-64 CPU, 15 GiB RAM, stock ONNX Runtime 1.27.0 (CPU EP),
`ORT_ENABLE_ALL`, 5 warmup + 20 timed iterations. Latency = mean ms/`Run`.

### Candidate A — SwiGLU, CPU

**float32**

| case         |    T |     F | unfused ms | quickgelu ms | fused_lb ms | unfused/lb | qg/lb |
|--------------|-----:|------:|-----------:|-------------:|------------:|-----------:|------:|
| small        |  512 |  2048 |     0.4642 |       0.4673 |      0.2351 |      1.97x | 1.99x |
| mixtral-ffn  | 1024 | 14336 |    10.8825 |      10.9013 |      6.1769 |      1.76x | 1.76x |
| deepseek-ffn | 2048 |  1408 |     1.8996 |       1.9792 |      0.8079 |      2.35x | 2.45x |
| wide         | 1024 |  8192 |     5.9711 |       6.0188 |      3.4434 |      1.73x | 1.75x |
| many-tokens  | 8192 |  4096 |    25.4093 |      25.3144 |     14.5151 |      1.75x | 1.74x |

**float16**

| case         | unfused ms | quickgelu ms | fused_lb ms | unfused/lb | qg/lb |
|--------------|-----------:|-------------:|------------:|-----------:|------:|
| small        |     1.0048 |       0.7839 |      0.5676 |      1.77x | 1.38x |
| mixtral-ffn  |    28.1380 |      21.6397 |     16.9777 |      1.66x | 1.27x |
| deepseek-ffn |     4.3555 |       2.8420 |      2.2594 |      1.93x | 1.26x |
| wide         |    15.1434 |      11.5234 |      9.5571 |      1.58x | 1.21x |
| many-tokens  |    63.7327 |      47.9868 |     38.1889 |      1.67x | 1.26x |

### Candidate B — Router, CPU

**float32**

| case          |    M |   E | k | naive ms | fused ms | speedup | max rel err |
|---------------|-----:|----:|--:|---------:|---------:|--------:|------------:|
| mixtral       | 4096 |   8 | 2 |   0.3869 |   0.3280 |   1.18x |   1.2e-07   |
| deepseek      | 4096 |  64 | 6 |   1.2939 |   1.1491 |   1.13x |   1.1e-07   |
| switch-many   | 4096 | 128 | 1 |   0.5844 |   0.3433 |   1.70x |   0.0       |
| large-experts | 8192 | 256 | 8 |   5.6638 |   4.7160 |   1.20x |   1.4e-07   |
| small-batch   |  512 |  32 | 4 |   0.1277 |   0.1195 |   1.07x |   7.4e-08   |

**float16**

| case          | naive ms | fused ms | speedup |
|---------------|---------:|---------:|--------:|
| mixtral       |   0.5716 |   0.4939 |   1.16x |
| deepseek      |   1.3857 |   1.2248 |   1.13x |
| switch-many   |   0.7896 |   0.4152 |   1.90x |
| large-experts |   6.3966 |   5.0442 |   1.27x |
| small-batch   |   0.1820 |   0.1624 |   1.12x |

Raw CSVs: `results_moe_fusions_cpu_fp32.csv`, `results_moe_fusions_cpu_fp16.csv`.

## Conclusions

- **SwiGLU fusion is the bigger lever.** Collapsing `Sigmoid+Mul+Mul` to a single fused pass
  is worth **~1.6–2.4×** on the gated-activation cluster on CPU, and that cluster operates on
  the layer's largest `[T, F]` tensors — so the absolute time saved is substantial
  (e.g. mixtral-ffn: ~10.9 → ~6.2 ms). Folding the activation into the GroupedMatMul epilogue
  (so the `[T, F]` activations never round-trip to memory) would capture at least this much and
  likely more. The `fused_lb` column is a conservative lower bound (single `Mul`), so a real
  fused SiLU-gate kernel lands between `quickgelu` and `fused_lb`.
- **`QuickGelu` already recovers part of the win in fp16** (SiLU in one op: ~1.2–1.4× over the
  naive 3-op form), but in fp32 the ORT CPU EP already runs the 3-op form about as fast as the
  2-op form — the remaining gap to `fused_lb` is the memory-traffic headroom, which only a true
  gate-fusion (one pass) closes.
- **Router fusion is a smaller, cheaper win: ~1.1–1.3×** in the typical few-expert case, rising
  to **~1.7–1.9×** for the single-expert / many-expert `switch-many` case where skipping the
  full `[M, E]` softmax matters most. It is numerically identical to the standard renormalized
  top-k router (max rel err ≈ 1e-7 fp32). Because the tensors are tiny, most of the benefit is
  from removing kernel launches and the `[M, E]` intermediate — expected to matter more on GPU
  (launch-latency-bound) than the modest CPU numbers here suggest.
- **Priority order confirmed:** SwiGLU/gated-activation fusion first (largest absolute saving,
  operates on the biggest tensors), router `Softmax+TopK` fusion second (cheap to implement,
  best on GPU / high-expert-count configs).
- **CUDA** was not measured (no GPU in this environment); the harness supports `--providers
  cuda`. The relative wins should grow on GPU, where both clusters are bandwidth-/launch-bound.

---

# MoE layer: fused `com.microsoft.MoE` vs. `GroupedMatMul`-based expansion

`benchmark_moe_vs_grouped.py` implements a full standard top-k MoE feed-forward layer **two
ways** and times them against each other, to quantify the perf gap between the complex
all-in-one MoE fusion and the simpler `GroupedMatMul`-based decomposition:

- **FUSED** — a single `com.microsoft.MoE` node (routing + both grouped GEMMs + activation +
  router-weighted combine in one kernel).
- **EXPANDED** — the `docs/GroupedMatMul.md` "Typical MoE usage" recipe:
  `router (Softmax+TopK | RouterTopK)` → `GroupedMatMul` (FC1) → activation →
  `GroupedMatMul` (FC2) → `Mul` + `ReduceSum` router-weighted combine.

## How to run

```bash
# Requires an ORT build from this branch (has GroupedMatMul / SwiGLU / RouterTopK).
# Run from OUTSIDE the repo root so the local onnxruntime/ source doesn't shadow the package.
python benchmark_moe_vs_grouped.py --providers cpu --csv results_moe_vs_grouped_cpu_fp32.csv
python benchmark_moe_vs_grouped.py --swiglu-impl fused-op        # expanded side uses com.microsoft.SwiGLU
python benchmark_moe_vs_grouped.py --router-impl routertopk      # expanded router uses com.microsoft.RouterTopK
python benchmark_moe_vs_grouped.py --profile                     # per-op kernel-time breakdown
python benchmark_moe_vs_grouped.py --intra-op-threads 1          # single-thread comparison
python benchmark_moe_vs_grouped.py --mem-budget-gb 8             # allow larger weight footprints
python benchmark_moe_vs_grouped.py --providers cuda --dtype float16   # on a GPU host
```

The harness builds one FUSED and one EXPANDED model per case, warms up (`--warmup`, default 5),
times `--iters` runs (default 30), and reports **median** latency for each side, the
`expanded/fused` speedup (`fused_x`), and the **max relative error** between the two outputs (a
hard correctness gate — see below). The CSV additionally records mean/median/min/std for both
sides, the pinned thread count, and the per-case weight footprint.

## Reading the report

- **`regime`** labels each case: `decode/launch-bound` (M=1 — dominated by dispatch/launch
  overhead, *not* compute), `decode-batch` (small M), or `prefill/compute`. The M=4096/8192
  `prefill-*` cases are compute-bound anchors: if the gap shrinks there it is overhead-bound; if
  it persists the fused GEMM path is fundamentally faster. Anchors grow *activation* memory, not
  weight memory.
- **`--profile`** enables ORT profiling and prints a per-op kernel-time breakdown for each side,
  so the gap can be attributed to a mechanism (op-dispatch overhead vs. intermediate
  reshape/round-trips vs. GEMM efficiency) instead of a single black-box ratio.
- **Threads**: the intra-op thread count is pinned with `--intra-op-threads` (default: ORT's
  automatic count) and printed in the header and CSV, so numbers are reproducible.

## Memory

Expert weights are baked as **raw-bytes ONNX initializers** (production-representative; neither
the MoE nor the GroupedMatMul CPU kernel implements `PrePack`, so there is no prepack asymmetry).
The two weight layouts are large and identical in size regardless of M — a Mixtral-swiglu layer
is ~5.25 GiB per layout in fp32. To fit modest boxes the harness (1) builds+times the fused
model to completion, frees it, then builds+times the expanded model, so **only one weight layout
is resident at a time**, and (2) applies `--mem-budget-gb` (default 4): any case whose
single-layout weight footprint exceeds the budget is reported as `OOM-skip`. Peak RSS is roughly
2× the budget per model. With the default budget the `mixtral-swiglu` / `decode-swiglu` cases
(~5.25 GiB) OOM-skip; raise `--mem-budget-gb` on a larger box to include them.

These multi-GiB weight layouts exceed protobuf's hard ~2 GiB single-message limit, so each model
is serialized with **ONNX external data** (weights in a side file) and the session is created
from that file path — `model.SerializeToString()` would otherwise raise on the ≥2 GiB cases (all
decode cases, both mixtral, `switch-top1`). With external data the real limiter is host RAM, not
serialization, so `--mem-budget-gb` remains the single knob to tune per box; the default of 4 GiB
is a conservative fit for a ~15 GiB box (peak ≈ 2× budget) and lets the previously-crashing
`decode-silu` / `mixtral-silu` (3.5 GiB) cases run.

The external-data side files are written per case into a fresh `tempfile.mkdtemp()` directory
(the model plus one `.bin` per weight tensor) and the whole directory is removed in a
`try/finally` after that model is timed, so no multi-GiB scratch files are left behind — but the
run does need free space in the system temp location equal to one weight layout while a case
runs. Set `TMP`/`TMPDIR` to a roomy volume if the default temp drive is small.

## Correctness (why the two agree)

- `com.microsoft.MoE` takes **router logits** in `router_probs` and softmaxes internally, so
  both graphs are fed the same logits.
- `normalize_routing_weights=1` renormalizes the top-k weights — identical to `RouterTopK`
  (softmax over the top-k logits) and to `Softmax → TopK → Div-by-sum`. The harness keeps the
  fused and expanded routers consistent (and forces `normalize=1` when `--router-impl
  routertopk`).
- **Weight layout is transposed between the two ops** and the harness handles it: MoE
  `fc1_experts_weights` `(E, fc1_out, hidden)` (applied `x @ W.T`) vs. GroupedMatMul `weights`
  `(E, hidden, fc1_out)` (applied `x @ W`); similarly for FC2. One shared set of expert weights
  is generated and the transposed variant fed to whichever graph needs it.
- The `_rel_err` cross-check is a **hard gate**: the default tolerance is `1e-4` (fp32) /
  `5e-2` (fp16), well above the observed **~1e-7 (fp32)** agreement. `--strict` (default **on**)
  raises on divergence; `--no-strict` downgrades it to a warning. A divergent run reports
  `DIVERGED` instead of a (meaningless) speedup, so a broken configuration can never advertise a
  bogus number. Validated offline against a NumPy reference of the expanded graph for `silu`,
  `gelu`, and `swiglu`, with and without bias and renormalization.

## SwiGLU handling (which form each run uses)

SwiGLU cases exercise the CPU-supported **interleaved** layout (`swiglu_fusion=1`): the FC1
output is `2 * inter_size` wide with rows laid out `[gate0, linear0, gate1, linear1, …]`, and
the activation is `gate * sigmoid(alpha * gate) * (linear + beta)`.

- The **FUSED** side always uses the MoE op's built-in SwiGLU (`swiglu_fusion=1`).
- The **EXPANDED** side's SwiGLU form is selected with `--swiglu-impl`:
  - `expanded` (default): the standard `Sigmoid + Mul + Mul` graph on the de-interleaved
    gate/linear tensors — the **"expanded form of SwiGLU"**.
  - `fused-op`: the new fused `com.microsoft.SwiGLU` op — the **"proposed SwiGLU fused op"**.

Every run prints the active `swiglu-impl` and `router-impl` in the report header and records
them in the CSV, so it is always unambiguous which SwiGLU form produced a given number.

> Note: on the expanded side the interleaved FC1 output is de-interleaved with `Reshape` +
> `Split` + `Squeeze` before the activation. Those ops (and the `Reshape`/`Unsqueeze` around FC2
> and the combine) show up in the `--profile` per-op breakdown, so part of the expanded-vs-fused
> gap on SwiGLU cases is this layout bookkeeping rather than GEMM time — read the breakdown with
> that in mind.

## Device / dtype note

**CPU is float32-only** for this comparison: the fused `MoE` fp16 CPU kernel is compiled but
not registered, so an fp16 MoE node has no CPU kernel. `GroupedMatMul` / `SwiGLU` / `RouterTopK`
do have fp16 CPU kernels, but since the fused side can't run fp16 on CPU the default CPU dtype
is float32. `--dtype float16` is intended for the CUDA path (MoE has fp16/bf16 CUDA kernels).

## Results

Environment: 24-logical-CPU x86-64 host, ONNX Runtime 1.29.0 built from this branch
(Release, CPU EP), intra-op threads auto (~24), 5 warmup + 30 timed iterations, strict
cross-check ON (`--tol 1e-4`), `--mem-budget-gb 12`. Latency is **median** wall-clock
milliseconds per `Run` (mean/min/std are in the CSVs). The full matrix is **18 cases × 4
configs** = (`--swiglu-impl {expanded, fused-op}`) × (`--router-impl {softmax-topk,
routertopk}`), all fp32.

**Why fp32 only:** the fused `com.microsoft.MoE` fp16 CPU kernel is compiled but not
registered, so an fp16 MoE node has no CPU kernel (see *Device / dtype note* above). The
comparison is therefore CPU-fp32; fp16 is a CUDA-path concern (future work).

**What is compared:** the fused `com.microsoft.MoE` op (router + both FC GEMMs + activation +
combine in one kernel) vs. the **expanded** form built from primitives —
`RouterTopK`/`Softmax→TopK→Div` → `GroupedMatMul` (FC1) → activation → `GroupedMatMul` (FC2) →
`Mul` (route-weight scale) → `ReduceSum` (combine top-k). Both sides are validated to compute
the same result every run.

### Headline

**Fused `com.microsoft.MoE` is slower than the expanded `GroupedMatMul` form in every tested
regime on CPU** — from ~1.1× (already-saturated batched cases) up to ~4.2× (single-token
decode). Table below is the canonical config (`--swiglu-impl expanded --router-impl
softmax-topk`); the other three configs move numbers by <12% (see *Sensitivity*).
"expanded speedup" = `fused_ms / expanded_ms` (>1 ⇒ expanded is faster).

| case                | regime              |    M |   K |     F |   E | k | fused ms | expanded ms | expanded speedup | max rel err |
|---------------------|---------------------|-----:|----:|------:|----:|--:|---------:|------------:|-----------------:|------------:|
| decode-silu         | decode/launch-bound |    1 | 4096| 14336 |   8 | 2 |   78.256 |      18.599 |          **4.21×** |    1.8e-06 |
| decode-swiglu       | decode/launch-bound |    1 | 4096| 14336 |   8 | 2 |  115.049 |      29.742 |          **3.87×** |    2.1e-06 |
| decode-batch8-silu  | decode-batch        |    8 | 4096| 14336 |   8 | 2 |  122.278 |     105.754 |            1.16× |    1.7e-06 |
| decode-batch32-silu | decode-batch        |   32 | 4096| 14336 |   8 | 2 |  152.325 |     136.391 |            1.12× |    3.3e-07 |
| small-silu          | prefill/compute     |  128 | 1024|  2048 |   8 | 2 |    8.786 |       7.808 |            1.13× |    2.7e-07 |
| small-silu-bias     | prefill/compute     |  128 | 1024|  2048 |   8 | 2 |   16.000 |       8.260 |            1.94× |    2.8e-07 |
| small-swiglu        | prefill/compute     |  128 | 1024|  2048 |   8 | 2 |   18.049 |      11.869 |            1.52× |    2.8e-07 |
| small-gelu          | prefill/compute     |  128 | 1024|  2048 |   8 | 2 |   15.120 |       8.512 |            1.78× |    2.6e-07 |
| deepseek-many-silu  | prefill/compute     |  512 | 1024|  1408 |  64 | 6 |   76.775 |      56.644 |            1.36× |    2.6e-07 |
| prefill-silu        | prefill/compute     |  512 | 2048|  5632 |   8 | 2 |  155.435 |      93.638 |            1.66× |    2.8e-07 |
| prefill-swiglu      | prefill/compute     |  512 | 2048|  5632 |   8 | 2 |  220.354 |     147.892 |            1.49× |    2.7e-07 |
| prefill-gelu        | prefill/compute     |  512 | 2048|  5632 |   8 | 2 |  170.512 |      97.699 |            1.75× |    3.1e-07 |
| switch-top1-silu    | prefill/compute     | 1024 |  768|  3072 | 128 | 1 |  116.748 |      99.492 |            1.17× |    4.0e-07 |
| mixtral-silu        | prefill/compute     | 1024 | 4096| 14336 |   8 | 2 | 1397.386 |     804.437 |            1.74× |    4.2e-07 |
| mixtral-swiglu      | prefill/compute     | 1024 | 4096| 14336 |   8 | 2 | 2068.742 |    1285.123 |            1.61× |    3.8e-07 |
| large-tokens-swiglu | prefill/compute     | 2048 | 1024|  2048 |   8 | 2 |  212.220 |     113.240 |            1.87× |    2.6e-07 |
| prefill-4k-silu     | compute anchor      | 4096 | 2048|  5632 |   8 | 2 | 1148.656 |     648.608 |            1.77× |    3.1e-07 |
| prefill-8k-silu     | compute anchor      | 8192 | 2048|  5632 |   8 | 2 | 2480.652 |    1431.243 |            1.73× |    3.4e-07 |

Raw CSVs: `results_moe_vs_grouped_cpu_fp32_{expanded,fusedop}_{softmaxtopk,routertopk}.csv`;
consolidated console tables in `results_moe_vs_grouped_cpu_fp32_console.txt`.

### Gap attribution (from `--profile` per-op breakdown)

The gap is **the GEMM itself, not dispatch / graph round-trip overhead.** In the expanded form,
all the non-GEMM primitives (router `Softmax`/`TopK`/`Div`, activation, the SwiGLU
de-interleave `Reshape`/`Split`/`Squeeze`, the `Mul` scale, and the `ReduceSum` combine) sum to
only **~2%** of expanded wall time; the two `GroupedMatMul` calls are ~98%. So the expanded form
is essentially "GroupedMatMul + negligible glue", and comparing wall times is effectively
comparing the fused kernel's internal grouped-GEMM against `GroupedMatMul`.

Concrete per-iteration numbers (`profiling_moe_vs_grouped_cpu_fp32.csv`):

| case            |    M | fused MoE GEMM | expanded GroupedMatMul | expanded overhead ops | fused-GEMM / GMM |
|-----------------|-----:|---------------:|-----------------------:|----------------------:|-----------------:|
| prefill-8k-silu | 8192 |     2383 ms    |         1285 ms        |    30 ms (**2.3%**)   |     **1.86×**    |
| decode-silu     |    1 |     87.4 ms    |         25.3 ms        |   0.5 ms (**2.1%**)   |     **3.46×**    |

- **Decode M=1 is the ~4× worst case** because the fused MoE kernel pays the **full per-expert
  cost to process a single token**: with `E=8, k=2` it still does the grouped work per selected
  expert over a `K=4096 → F=14336` weight, and that per-expert launch/pack cost is not amortised
  over any token batch. `GroupedMatMul` handles the tiny `M·k` row set far more cheaply.
- **The gap shrinks with batch but does not close.** As `M` grows the fused kernel amortises its
  per-expert overhead (`decode-batch8` 1.16×, `decode-batch32` 1.12×), but even at the
  compute-bound anchors it **persists**: `M=4096` 1.77× and `M=8192` 1.73× (fused-GEMM/GMM
  1.86× at 8k). This is a steady-state grouped-GEMM efficiency gap on CPU, not a fixed startup
  cost — the fused internal grouped-GEMM is simply **~1.8–3.5× slower** than `GroupedMatMul`
  across the range.

### Sensitivity (SwiGLU form and router impl)

- **SwiGLU form (`--swiglu-impl`).** The fused side always uses MoE's built-in SwiGLU with
  `swiglu_fusion=1` (interleaved — the only CPU-supported format). The expanded side was
  benchmarked **both ways**: the fused `com.microsoft.SwiGLU` op (`fused-op`) and the expanded
  `Sigmoid + Mul + Mul` graph (`expanded`). Measured impact on the expanded side is
  **<12% and regime-dependent** (e.g. `large-tokens-swiglu` 113.2 → 124.2 ms, +9.6%;
  `decode-swiglu` 29.7 → 26.4 ms, −11%; `mixtral-swiglu` ~±0.3%). It does **not** change the
  headline — expanded wins under either SwiGLU form.
- **Router impl (`--router-impl`).** `softmax-topk` (`Softmax→TopK→Div`) vs. the fused
  `RouterTopK` op move medians by **<5%**, within run-to-run noise (the router is a tiny
  fraction of total time in every case). No config flips the fused-vs-expanded ranking.

### The one exception worth the perf team's attention

**`switch-top1-silu`** (128 experts, top-1, `K=768→F=3072`) is the **only** case where the
fused kernel's **internal grouped-GEMM is faster than `GroupedMatMul`**: in the profiled run
the fused MoE GEMM is **124.6 ms/iter vs. GroupedMatMul's 153.3 ms/iter**. Top-1 routing over
many small experts is exactly the shape the fused grouped-GEMM's per-group dispatch is built
for. Even so, **the expanded form still wins wall-clock** (99.5 vs. 116.7 ms median) because
its ~2% overhead is cheaper than whatever the fused kernel spends outside its GEMM. This is the
one shape where closing the fused GEMM gap could plausibly make fused MoE the wall-clock winner
— flagged for the perf team.

### Correctness

All **18 cases × 4 configs (72 runs)** passed the strict fused-vs-expanded cross-check
(`--tol 1e-4`): **max relative error 2.08e-6** (worst case `decode-swiglu`), no `DIVERGED`, no
`OOM-skip` (12 GiB budget, per-tensor external-data serialization keeps peak at ~1× layout).
Threads ~24 intra-op, fp32. The two implementations compute the same MoE layer within
numerical tolerance, so the latency comparison is apples-to-apples.

## Conclusions

- **On CPU, the more complex fused `com.microsoft.MoE` op is a net performance loss vs. the
  simpler `GroupedMatMul`-based expansion** — slower in all 18 regimes, by ~1.1× (saturated
  batch) to ~4.2× (single-token decode). The end-goal question ("is the heavier MoE fusion
  worth it on CPU vs. reusing GroupedMatMul?") answers **no, as currently implemented.**
- **The cost is in the fused kernel's internal grouped-GEMM, not in fusion/dispatch savings.**
  The expanded form's extra ops (router, activation, SwiGLU de-interleave, combine) are only
  ~2% of its time, so fusing them away cannot recover the gap; the fused grouped-GEMM is
  ~1.8–3.5× slower than `GroupedMatMul` at the same work. Optimization effort should target the
  MoE kernel's grouped-GEMM path (ideally sharing `GroupedMatMul`'s CPU GEMM), not the
  op-fusion boundary.
- **`switch-top1` (many experts, top-1) is the lone shape where the fused GEMM already wins**
  and is the most promising target for making fused MoE competitive on CPU.
- **CUDA is future work.** The harness fully supports `--providers cuda` (and MoE has fp16/bf16
  CUDA kernels, so the fp32-only CPU restriction lifts there). The trade-off may invert on GPU,
  where a single fused kernel avoids materialising intermediate expert tensors in HBM and saves
  launch overhead — but that must be measured on a GPU build before drawing any conclusion.
