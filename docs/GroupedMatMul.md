# GroupedMatMul (contrib op)

`com.microsoft.GroupedMatMul` performs a *grouped* matrix multiplication: every token
(row of the input) is multiplied by one or more weight matrices, each selected per token
from a stack of weight matrices via an integer group index. This is the core computation of
Mixture-of-Experts (MoE) feed-forward layers, and corresponds to
`torch.nn.functional.grouped_mm` in PyTorch.

This op is derived from the design discussion in
[onnx/onnx#7902](https://github.com/onnx/onnx/issues/7902). See
[Design notes](#design-notes) below for how this specification differs from that proposal.

## Signature

### Inputs

| # | Name | Type | Required | Shape | Description |
|---|------|------|----------|-------|-------------|
| 0 | `input` | T | Yes | `[M, K]` | Row-major tokens. `M` tokens, `K` hidden (contraction) dim. |
| 1 | `weights` | T | Yes | `[num_groups, K, N]` | One `K x N` weight matrix per group. |
| 2 | `group_indices` | tensor(int64) | Yes | `[M, k]` | Group id per (token, expert-slot). Each token selects `k` experts. Values in `[0, num_groups)`. Use `k = 1` for the dense case. |
| 3 | `combine_weights` | T | Optional | `[M, k]` | Per-selection combine weights. When present, the op returns the weighted sum over the `k` selected experts. |
| 4 | `bias` | T | Optional | `[num_groups, N]` | Per-group bias, added before the optional combine. |

`weights` and `bias` are shared across all tokens.

Callers with batched inputs of shape `[..., K]` should `Reshape` the leading dimensions into
`M` first. In ONNX Runtime a `Reshape` of this kind is a metadata-only view (zero-copy), so
requiring a flattened `input` costs nothing.

### Output

| # | Name | Type | Shape | Description |
|---|------|------|-------|-------------|
| 0 | `output` | T | `[M, N]` (combine) or `[M, k, N]` (no combine) | See semantics. |

- When `combine_weights` **is** provided, the output is the weighted sum over the `k`
  selected experts: shape `[M, N]`.
- When `combine_weights` is **absent**, the per-expert results are returned: shape `[M, k, N]`.

### Type constraint

`T`: `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)` (bfloat16: CUDA only).

## Semantics

```python
# input:           [M, K]
# weights:         [num_groups, K, N]
# group_indices:   [M, k]           values in [0, num_groups)
# combine_weights: [M, k] or None
# bias:            [num_groups, N] or None
for i in range(M):
    for j in range(k):
        g = group_indices[i, j]
        r[i, j] = input[i] @ weights[g]           # [K] @ [K, N] -> [N]
        if bias is not None:
            r[i, j] += bias[g]

if combine_weights is not None:
    for i in range(M):
        output[i] = sum(combine_weights[i, j] * r[i, j] for j in range(k))   # [M, N]
else:
    output = r                                                                # [M, k, N]
```

### Edge cases

- **Empty groups** are valid: if no selection maps to group `g`, `weights[g]` is unused.
- **`k == 1`** is the dense case; without `combine_weights` it degenerates to selecting one
  weight matrix per token. With `combine_weights` it additionally scales each token's result.
- **`num_groups == 1`** and all indices `0` is equivalent to `MatMul(input, weights[0])`.
- Out-of-range indices (`< 0` or `>= num_groups`) are an error.

## Typical MoE usage (top-k)

The two projections of an MoE feed-forward layer map cleanly onto this op, and **no `Expand`
of the tokens is needed** — the op reuses each token row across its `k` selected experts
internally:

```
scores          = Softmax(MatMul(hidden, router_W))     # [B, M, E]
values, indices = TopK(scores, k)                        # [B, M, k]

h        = Reshape(hidden, [B*M, K])                     # zero-copy view
idx      = Reshape(indices, [B*M, k])
val      = Reshape(values,  [B*M, k])

# Up projection: per-expert output (no combine) -> [B*M, k, F]
h        = GroupedMatMul(h, expert_W1, idx)
h        = SiLU(h)
h        = Reshape(h, [B*M*k, F])                        # zero-copy view
idx2     = Reshape(idx, [B*M*k, 1])

# Down projection with fused weighted-sum combine -> [B*M, hidden]
out      = GroupedMatMul(h, expert_W2, idx2? , val)      # see note below
```

The up projection uses the **no-combine** form (the activation must run per expert before
combining). The down projection uses the **combine** form so the top-k weighted sum is fused
into the op, avoiding a separate `Mul` + `ReduceSum`.

## Reference decomposition (for correctness only)

```
# Per-expert results, r: [M, k, N]
idx_flat  = Reshape(group_indices, [M*k])
W_sel     = Gather(weights, idx_flat, axis=0)            # [M*k, K, N]  (duplicates weights!)
X         = Reshape(Expand(Unsqueeze(input, 1), [M, k, K]), [M*k, 1, K])
r         = Reshape(MatMul(X, W_sel), [M, k, N])         # + Gather(bias, idx_flat) if present

# combine present:
output    = ReduceSum(r * Unsqueeze(combine_weights, -1), axis=1)   # [M, N]
# combine absent:
output    = r                                                        # [M, k, N]
```

The decomposition materializes one weight matrix *per selection* (`O(M*k*K*N)`) and an
`Expand` copy of the tokens, which is impractical for real MoE layers. A fused kernel reads
each weight matrix once regardless of how many tokens use it, reuses each token row across
its `k` experts without copying, and (when combining) never materializes the `[M, k, N]`
result to a graph tensor — that is the reason for the dedicated op.

## Implementation notes

Both kernels use the standard "sort/permute by group, then one GEMM per group" strategy so
that each group is a single dense GEMM over a contiguous block of selections. A "selection"
is a `(token, expert-slot)` pair `p` in `[0, M*k)`; its source token row is `p / k`.

- **CPU**: selections are bucketed by group; each non-empty group gathers its source token
  rows and runs one MLAS GEMM (`[count, K] x [K, N]`). Results are either scattered into the
  `[M, k, N]` output (no combine) or accumulated as `combine_weights[p] * (result + bias)`
  into the `[M, N]` output. float16 is up-converted to float for the GEMM.
- **CUDA**: selections are stable-sorted by group index; each selection's source token row is
  gathered into group-contiguous order; each non-empty group runs one cuBLAS GEMM; results
  (plus optional bias) are scattered back to selection order. When combining, a final
  reduction kernel sums the `k` per-expert results with the combine weights into the `[M, N]`
  output.

## Design notes

Differences from the ONNX issue #7902 proposal, and why:

1. **Single 3D form for `group_indices` (`[M, k]`), not separate 2D/3D variants.** The dense
   case is simply `k == 1`. Callers flatten any leading batch dims into `M` first, which is a
   zero-copy `Reshape` in ONNX Runtime. This keeps the spec and every kernel to a single code
   path.
2. **Top-k is first-class, with an optional fused weighted-sum combine.** Accepting
   `group_indices [M, k]` lets the op reuse each token row across its `k` experts internally,
   eliminating the `Expand` of `input` (which is a real data copy in ONNX Runtime, not a
   view). The optional `combine_weights` additionally fuses the weighted sum over `k`,
   avoiding a separate `Mul` + `ReduceSum` and the round-trip of the `[M, k, N]` per-expert
   tensor through memory. The combine is **optional** so the same op serves the MoE
   up-projection (per-expert output, no combine), the down-projection (fused combine), and the
   plain non-MoE `grouped_mm` (`k = 1`, no combine).
3. **`group_indices`, not `offsets`.** Indices do not require the caller to pre-sort tokens
   and compose naturally with `TopK`/`Gather`. Runtimes sort internally (both kernels here
   do). This keeps the op declarative, matching ONNX's role as an interchange format.
4. **No fused activation or dequantization.** These stay as separate ops (`SiLU`,
   `DequantizeLinear`) to keep the op composable across the many MoE routing variants (top-k,
   expert-choice, soft, hash, shared+routed). The weighted-sum combine is fused because it is
   common to all of them and cannot otherwise avoid the `Expand`/`ReduceSum` overhead.
5. **Stacked 3D `weights` `[num_groups, K, N]`** (all groups share `K`, `N`), which is the
   common MoE case and keeps memory contiguous and shapes static.
