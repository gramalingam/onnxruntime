# GroupedMatMul (contrib op)

`com.microsoft.GroupedMatMul` performs a *grouped* matrix multiplication: every token
(row of the input) is multiplied by a weight matrix that is selected, per token, from a
stack of weight matrices via an integer group index. This is the core computation of
Mixture-of-Experts (MoE) feed-forward layers, and corresponds to
`torch.nn.functional.grouped_mm` in PyTorch.

This op is derived from the design discussion in
[onnx/onnx#7902](https://github.com/onnx/onnx/issues/7902). See
[Design notes](#design-notes) below for how this specification differs from that proposal.

## Signature

### Inputs

| # | Name | Type | Required | Shape | Description |
|---|------|------|----------|-------|-------------|
| 0 | `input` | T | Yes | `[..., K]` | Row-major tokens. Any rank ≥ 2; all leading dims are token dims. |
| 1 | `weights` | T | Yes | `[num_groups, K, N]` | One `K x N` weight matrix per group. |
| 2 | `group_indices` | tensor(int64) | Yes | `[...]` | Group id per token. Shape equals `input` shape without the last dim. Values in `[0, num_groups)`. |
| 3 | `bias` | T | Optional | `[num_groups, N]` | Per-group bias, added to the result. |

`weights` and `bias` are shared across all tokens (batches).

### Output

| # | Name | Type | Shape | Description |
|---|------|------|-------|-------------|
| 0 | `output` | T | `[..., N]` | `input` shape with the last dim replaced by `N`. |

### Type constraint

`T`: `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)` (bfloat16: CUDA only).

## Semantics

Let `M` be the number of tokens (product of all `input` dims except the last). Flattening
the leading dims:

```python
# input:         [M, K]
# weights:       [num_groups, K, N]
# group_indices: [M]           values in [0, num_groups)
# bias:          [num_groups, N] or None
for i in range(M):
    g = group_indices[i]
    output[i] = input[i] @ weights[g]         # [K] @ [K, N] -> [N]
    if bias is not None:
        output[i] += bias[g]
```

### Edge cases

- **Empty groups** are valid: if no token maps to group `g`, `weights[g]` is unused.
- **All tokens in one group** degenerates to a standard `MatMul`.
- `num_groups == 1` is equivalent to `MatMul(input, weights[0])`.
- Out-of-range indices (`< 0` or `>= num_groups`) are an error.

## Typical MoE usage (top-k)

Top-k routing is expressed by *flattening* — the op itself has no top-k or routing built in:

```
scores          = Softmax(MatMul(hidden, router_W))     # [B, M, E]
values, indices = TopK(scores, k)                        # [B, M, k]

# Repeat each token k times, flatten to [B, M*k, K]
h        = Reshape(Expand(Unsqueeze(hidden, -2), [B,M,k,K]), [B, M*k, K])
idx      = Reshape(indices, [B, M*k])

h        = GroupedMatMul(h, expert_W1, idx)              # up projection
h        = SiLU(h)
out      = GroupedMatMul(h, expert_W2, idx)              # down projection

# Weighted sum over the k experts
out      = Reshape(out, [B, M, k, hidden])
output   = ReduceSum(out * Unsqueeze(values, -1), axis=-2)
```

## Reference decomposition (for correctness only)

```
idx_flat  = Reshape(group_indices, [M])
W_sel     = Gather(weights, idx_flat, axis=0)            # [M, K, N]  (duplicates weights!)
out       = MatMul(Reshape(input, [M, 1, K]), W_sel)     # [M, 1, N]
output    = Reshape(out, [..., N])
# + Gather(bias, idx_flat) if bias present
```

The decomposition materializes one weight matrix *per token* (`O(M*K*N)`), which is
impractical for real MoE layers. A fused kernel reads each weight matrix once regardless
of how many tokens use it — that is the reason for the dedicated op.

## Implementation notes

Both kernels use the standard "sort/permute by group, then one GEMM per group" strategy so
that each group is a single dense GEMM over a contiguous block of tokens:

- **CPU**: tokens are bucketed by group; each non-empty group runs one MLAS GEMM
  (`[count, K] x [K, N]`). float16 is up-converted to float for the GEMM.
- **CUDA**: tokens are stable-sorted by group index; input rows are gathered into
  group-contiguous order; each non-empty group runs one cuBLAS GEMM; results (plus optional
  bias) are scattered back to the original token order. This matches the grouped-GEMM
  approach used by the MoE kernels and by cuBLAS/CUTLASS grouped GEMM APIs.

## Design notes

Differences from the ONNX issue #7902 proposal, and why:

1. **No 3D `group_indices` (`[B, M, k]`) special case.** The proposal offered *both* 3D
   indices and a flatten-based top-k pattern to express the same thing. That redundancy
   complicates the spec and every kernel. This op keeps a single, rank-generic form —
   `group_indices` always has the shape of `input` minus its last dim — and top-k is
   expressed by flattening (as the proposal itself recommends). This subsumes the 2D
   `[M, K]` and 3D `[B, M, K]` cases without a separate code path.
2. **`group_indices`, not `offsets`.** Indices do not require the caller to pre-sort tokens
   and compose naturally with `TopK`/`Gather`. Runtimes sort internally (both kernels here
   do). This keeps the op declarative, matching ONNX's role as an interchange format.
3. **No fused activation, scaling, or weighted-sum.** These stay as separate ops
   (`SiLU`/`Mul`/`ReduceSum`, `DequantizeLinear`) to keep the op composable across the many
   MoE routing variants (top-k, expert-choice, soft, hash, shared+routed).
4. **Stacked 3D `weights` `[num_groups, K, N]`** (all groups share `K`, `N`), which is the
   common MoE case and keeps memory contiguous and shapes static.
