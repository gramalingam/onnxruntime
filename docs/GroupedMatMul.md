# GroupedMatMul (contrib op)

`com.microsoft.GroupedMatMul` performs a *grouped* matrix multiplication: every token
(row of the input) is multiplied by one or more weight matrices, each selected per token
from a stack of weight matrices via an integer group index, returning the per-expert
results. This is the core computation of Mixture-of-Experts (MoE) feed-forward layers, and
corresponds to `torch.nn.functional.grouped_mm` in PyTorch.

This op is derived from the design discussion in
[onnx/onnx#7902](https://github.com/onnx/onnx/issues/7902) and the RFC in
[onnx/onnx#8193](https://github.com/onnx/onnx/pull/8193). See
[Design notes](#design-notes) below for how this specification differs from that proposal.

## Signature

### Inputs

| # | Name | Type | Required | Shape | Description |
|---|------|------|----------|-------|-------------|
| 0 | `input` | T | Yes | `[M, K]` | Row-major tokens. `M` tokens, `K` hidden (contraction) dim. |
| 1 | `weights` | T | Yes | `[num_groups, K, N]` | One `K x N` weight matrix per group. |
| 2 | `group_indices` | tensor(int64) | Yes | `[M, k]` | Group id per (token, expert-slot). Each token selects `k` experts. Values in `[0, num_groups)`. Use `k = 1` for the dense case. |
| 3 | `bias` | T | Optional | `[num_groups, N]` | Per-group bias, added to each selected expert result. |

`weights` and `bias` are shared across all tokens.

Callers with batched inputs of shape `[..., K]` should `Reshape` the leading dimensions into
`M` first. In ONNX Runtime a `Reshape` of this kind is a metadata-only view (zero-copy), so
requiring a flattened `input` costs nothing.

### Output

| # | Name | Type | Shape | Description |
|---|------|------|-------|-------------|
| 0 | `output` | T | `[M, k, N]` | Per-expert results: for each token, the result of multiplying it by each of its `k` selected experts (plus the optional bias). |

### Type constraint

`T`: `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)` (bfloat16: CUDA only).

## Semantics

```python
# input:           [M, K]
# weights:         [num_groups, K, N]
# group_indices:   [M, k]           values in [0, num_groups)
# bias:            [num_groups, N] or None
for i in range(M):
    for j in range(k):
        g = group_indices[i, j]
        r[i, j] = input[i] @ weights[g]           # [K] @ [K, N] -> [N]
        if bias is not None:
            r[i, j] += bias[g]

output = r                                        # [M, k, N]
```

### Edge cases

- **Empty groups** are valid: if no selection maps to group `g`, `weights[g]` is unused.
- **`k == 1`** is the dense case; it degenerates to selecting one weight matrix per token.
- **`num_groups == 1`** and all indices `0` is equivalent to `MatMul(input, weights[0])`.
- Out-of-range indices (`< 0` or `>= num_groups`) are an error.

## Typical MoE usage (top-k)

The two projections of an MoE feed-forward layer map cleanly onto this op, and **no `Expand`
of the tokens is needed** — the op reuses each token row across its `k` selected experts
internally. Any weighted combination of the per-expert results (for example the top-k
router-weighted sum) is expressed with standard `Mul` / `ReduceSum` ops in the surrounding
graph:

```
scores          = Softmax(MatMul(hidden, router_W))     # [B, M, E]
values, indices = TopK(scores, k)                        # [B, M, k]

h        = Reshape(hidden, [B*M, K])                     # zero-copy view
idx      = Reshape(indices, [B*M, k])
val      = Reshape(values,  [B*M, k])

# Up projection: per-expert output -> [B*M, k, F]
h        = GroupedMatMul(h, expert_W1, idx)
h        = SiLU(h)
h        = Reshape(h, [B*M*k, F])                        # zero-copy view
idx2     = Reshape(idx, [B*M*k, 1])

# Down projection: per-expert output, then router-weighted sum over the k slots
d        = GroupedMatMul(h, expert_W2, idx2)             # [B*M*k, 1, hidden]
d        = Reshape(d, [B*M, k, hidden])                  # regroup the k slots
out      = ReduceSum(d * Unsqueeze(val, -1), axis=1)     # [B*M, hidden]
```

Both projections use `GroupedMatMul` purely for the grouped matrix multiplication. The
top-k router-weighted sum in the down-projection is expressed explicitly with `Mul` +
`ReduceSum` over the regrouped `k` slots.

## Reference decomposition (for correctness only)

```
# Per-expert results, r: [M, k, N]
idx_flat  = Reshape(group_indices, [M*k])
W_sel     = Gather(weights, idx_flat, axis=0)            # [M*k, K, N]  (duplicates weights!)
X         = Reshape(Expand(Unsqueeze(input, 1), [M, k, K]), [M*k, 1, K])
r         = Reshape(MatMul(X, W_sel), [M, k, N])         # + Gather(bias, idx_flat) if present

output    = r                                            # [M, k, N]
```

The decomposition materializes one weight matrix *per selection* (`O(M*k*K*N)`) and an
`Expand` copy of the tokens, which is impractical for real MoE layers. A fused kernel reads
each weight matrix once regardless of how many tokens use it and reuses each token row across
its `k` experts without copying — that is the reason for the dedicated op.

## Implementation notes

Both kernels use the standard "sort/permute by group, then one GEMM per group" strategy so
that each group is a single dense GEMM over a contiguous block of selections. A "selection"
is a `(token, expert-slot)` pair `p` in `[0, M*k)`; its source token row is `p / k`.

- **CPU**: selections are bucketed by group; each non-empty group gathers its source token
  rows and runs one MLAS GEMM (`[count, K] x [K, N]`). Results (plus optional bias) are
  scattered into the `[M, k, N]` output. float16 is up-converted to float for the GEMM.
- **CUDA**: selections are stable-sorted by group index; each selection's source token row is
  gathered into group-contiguous order; each non-empty group runs one cuBLAS GEMM; results
  (plus optional bias) are scattered back to the `[M, k, N]` output in selection order.

## Design notes

Differences from the ONNX issue #7902 / RFC #8193 proposals, and why:

1. **Single 3D form for `group_indices` (`[M, k]`), not separate 2D/3D variants.** The dense
   case is simply `k == 1`. Callers flatten any leading batch dims into `M` first, which is a
   zero-copy `Reshape` in ONNX Runtime. This keeps the spec and every kernel to a single code
   path.
2. **Top-k is first-class.** Accepting `group_indices [M, k]` lets the op reuse each token row
   across its `k` experts internally, eliminating the `Expand` of `input` (which is a real
   data copy in ONNX Runtime, not a view). The op always returns the per-expert results
   `[M, k, N]`.
3. **No fused weighted-sum combine.** The router-weighted sum used in MoE layers is expressed
   with standard `Mul` / `ReduceSum` ops in the surrounding graph rather than folded into this
   op. In a standard MoE down-projection the `k` experts have distinct per-slot inputs, so the
   weighted sum reduces over a *different* grouping than the matmul's own `k`; folding it in
   would require decoupling the two groupings and pinning down a normative result layout. This
   matches RFC [onnx/onnx#8193](https://github.com/onnx/onnx/pull/8193), which dropped the
   fused combine for this reason. A dedicated fused down-projection operator may capture it
   more cleanly in the future.
4. **`group_indices`, not `offsets`.** Indices do not require the caller to pre-sort tokens
   and compose naturally with `TopK`/`Gather`. Runtimes sort internally (both kernels here
   do). This keeps the op declarative, matching ONNX's role as an interchange format.
5. **No fused activation or dequantization.** These stay as separate ops (`SiLU`,
   `DequantizeLinear`) to keep the op composable across the many MoE routing variants (top-k,
   expert-choice, soft, hash, shared+routed).
6. **Stacked 3D `weights` `[num_groups, K, N]`** (all groups share `K`, `N`), which is the
   common MoE case and keeps memory contiguous and shapes static.

