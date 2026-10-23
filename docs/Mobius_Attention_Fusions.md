# Opt-in Mobius attention inverses

`rewrite::MakeMobiusAttentionRules()` in
`onnxruntime/core/optimizer/mobius_attention_fusion.h` returns experimental rules
for explicit use with `PatternRewriter`. They are **not registered in the
default transformer pipeline**. The model must already import `com.microsoft:1`.
The accepted standard-domain import range is **23-24**; newer opsets are not
implicitly assumed equivalent.

The source oracle is Mobius commit
[`d69efe03a9f4c1ada744bc31dd01077a710b3c39`](https://github.com/onnxruntime/mobius/tree/d69efe03a9f4c1ada744bc31dd01077a710b3c39).
The C++ tests reproducibly generate small standard-ONNX graphs from these bodies:

| Candidate | Pinned source | Implemented boundary |
|---|---|---|
| A5 | [`PixtralAttention.forward`](https://github.com/onnxruntime/mobius/blob/d69efe03a9f4c1ada744bc31dd01077a710b3c39/src/mobius/components/_pixtral_vision.py#L147-L202) | Cache-free, noncausal, rank-3 standard `Attention` to `com.microsoft::MultiHeadAttention`. |
| A6 | [`Qwen3VLVisionAttention._emit_standard_attention`](https://github.com/onnxruntime/mobius/blob/d69efe03a9f4c1ada744bc31dd01077a710b3c39/src/mobius/components/_qwen3_vl_vision.py#L271-L340) | Canonical segment-mask construction, three batch-axis unsqueezes, standard `Attention`, and output squeeze to `com.microsoft::PackedMultiHeadAttention`, **only for immutable `cu_seqlens=[0,N]`**. |
| A7 | [`_apply_attention`](https://github.com/onnxruntime/mobius/blob/d69efe03a9f4c1ada744bc31dd01077a710b3c39/src/mobius/components/_attention.py#L77-L231) | Maskless causal rank-3 standard `Attention` with dynamic BNSH past caches and three live outputs to `com.microsoft::GroupQueryAttention`. |

## Contracts and composition

The family has its own `PatternLibrary`. Attention bodies are composed by named
calls; the packed mask composes a `ScalarLike` interface with an alternation for
literal FP32 scalars versus `CastLike` scalars. Handwritten `AttentionView`,
`PackedView`, and `DecoderView` adapters use instance-scoped `BindingView`
access, not positional indices into flattened nodes. C++ callbacks validate the
semantic contracts and bind replacement attributes.

All admitted Q/K/V tensors are **FP32**, with statically known positive dimensions
and identical batch/sequence geometry. Explicit positive finite scale and head
counts are required. Head dimensions must agree, including V. Softcap must be
zero; softmax precision must be omitted, zero, or explicitly FLOAT. Masks,
nonpad lengths, unsupported attributes, and the optional QK output are not
silently discarded. Projections, external RoPE, casts, and layout reshapes outside
the matched boundary remain unchanged.

A5 requires equal Q/KV head counts, no cache or mask, one output, and identical
Q/K/V shapes. It does not implement cross-attention, packed QKV, or unequal V
head sizes.

A7 permits grouped heads when query heads are divisible by KV heads. Both past
caches must be FP32 `[B,Hkv,P,D]`, with identical shapes; `P=0` is allowed.
The head size `D` must be a multiple of eight, as required by the CPU kernel.
This FP32 form targets the CPU GQA kernel; assigned non-CPU nodes are rejected
even if the caller allows that provider in the generic rewriter options. The
base's CUDA GQA registrations accept FP16/BF16, not FP32.
The output caches retain the standard Attention BNSH layout and **both cache
outputs and every external consumer are preserved**. The new query and KV
sequence lengths must agree. Because no padding/mask is admitted, every cache
entry is valid: the generated INT32 metadata is exactly `seqlens_k[b]=P+S-1`
and scalar `total_sequence_length=P+S`. Overflow is rejected before generation.
The GQA schema leaves the present sequence dimension unknown when that generated
total is not available to shape inference, rather than incorrectly copying `P`.
This retains existing source output annotations of `P+S`; known shared-buffer
and sliding-window shape inference are unchanged.
For nonempty past, multi-token steps require batch size one, matching the CPU
target's input validation. `causal=1` uses bottom-right causal alignment;
`do_rotary=0` retains all external rotary/interleaving/partial-width processing.
This is not a fused-RoPE rule. Ragged padding, arbitrary bias, static
TensorScatter caches, sliding windows, windowed/shared cache interfaces, FP8
caches, and nonzero softcap remain unsupported.

The rewriter's deletion-boundary checks protect live intermediate outputs.
An output exposed by a rule retains its existing `NodeArg` and consumers.
A live packed mask, unsqueezed Attention output, or mask-construction intermediate
outside that rule's boundary causes rejection rather than deleting that value.

## Packed attention: exact restriction, not an approximation option

The packed rule is enabled separately with
`MakeMobiusAttentionRules(/*include_cuda_packed_attention=*/true)`.
This selects a **CUDA deployment**, not a CPU implementation. At this base,
`PackedMultiHeadAttention` has CUDA float/FP16 kernels but no CPU kernel.
Assigned non-CUDA Attention nodes are rejected, and the usual rewriter
execution-provider compatibility checks still apply. Unassigned export graphs
can be rewritten for later CUDA placement. The default factory omits this rule.

For `cu_seqlens=[0,N]`, `Range(0,N,1)` is always in the sole segment; the canonical
Where mask is identically zero. The replacement has one unpadded subsequence,
INT32 cumulative lengths `[0,N]`, and `token_offset=[[0,1,...,N-1]]`. This removes
the quadratic mask construction without changing which keys are attended.
Both plain constants and unfused `CastLike` scalar forms are recognized.
BF16-to-FP16-and-back conversion is **not** introduced.

**Multiple subsequences are deferred, even with constant lengths.** Mobius's
finite `-10000` bias is not an exact hard block mask. For a concrete counterexample,
take one head, head size one, scale one, `cu_seqlens=[0,1,2]`,
`Q=[[1],[1]]`, `K=[[0],[20000]]`, and `V=[[0],[1]]`.
For the first query, the standard masked logits are `[0,10000]`, so the result
is approximately `1`; packed attention excludes the second key and returns `0`.
No universal tolerance makes these operations equivalent. Checking mask
structure, monotonic lengths, or nominal dtype does not prove a bound on logits.
The implementation therefore rejects this graph instead of exposing an
approximation flag.

## B3 LATENT PagedAttention: interface blockers

The pinned
[`DeepSeekMLA.forward`](https://github.com/onnxruntime/mobius/blob/d69efe03a9f4c1ada744bc31dd01077a710b3c39/src/mobius/components/_deepseek_mla.py#L128-L252)
and
[`PagedLatentMLA.forward`](https://github.com/onnxruntime/mobius/blob/d69efe03a9f4c1ada744bc31dd01077a710b3c39/src/mobius/components/_paged_mla.py#L449-L516)
do not have interchangeable inputs/outputs. The base's
`com.microsoft::PagedAttention:1` schema exists, but a schema is not evidence of
an equivalent local rewrite or of kernel availability for all variants.

The dense source has expanded dense K/V past and present tensors. The target
requires a caller-owned paged latent key buffer, cumulative lengths, per-request
past lengths, and a block table, with no separate V cache. The pinned emitter
also supplies a token-to-slot mapping; that input is optional in the base ABI,
but deriving it still requires the caller's block table and lengths.
None of the dense source's values establishes page ownership, allocation,
slot lifetime, or the absent V-cache interface. Creating new graph inputs would
change the public signature; fabricating one-shot page tables would not preserve
cache ownership or subsequent calls. In addition,
[`absorb_mla_weights`](https://github.com/onnxruntime/mobius/blob/d69efe03a9f4c1ada744bc31dd01077a710b3c39/src/mobius/components/_paged_mla.py#L289-L360)
changes projection dimensions and floating-point operation ordering.

Consequently B3 has **no rule in this family**. It needs a coordinated
exporter/runtime cache-interface conversion with explicit paging inputs,
latent-weight provenance, and cache-consumer migration. No schema imports or
rewriter redesign are added to suggest otherwise.
The base CUDA implementation has a dedicated unfused LATENT backend with a
device-shared-memory eligibility check; ordinary Flash/MEA schema support does
not establish eligibility for this layout.

## Numerical qualification and validation

These are algebraic inference fusions, **not bitwise floating-point identities**:
standard Attention may distribute scaling over Q and K, whereas contrib kernels
can scale GEMM results or use a fused attention path. CPU FP32 comparisons use
small nontrivial inputs, explicit nondefault scale, and absolute tolerance
`2e-6`, including every output/cache consumer. This is evidence for the tested
forms, not a guarantee for NaN/Inf inputs or all dynamic ranges.

`MobiusAttentionFusionTests.*` covers both indexed and reference matching,
opsets 23 and 24, CPU A5/A7 source-versus-fused values, multi-token decode,
zero-length initial caches, multiple output consumers, canonical/folded packed
scalar forms, and negative semantic/deletion-boundary mutations. Every source
generator emits only standard-domain nodes. Packed coverage is structural and
shape/schema validation on this CPU host; **CUDA numerical execution is not
claimed**.

Run the targeted tests from the build output directory:

```powershell
.\onnxruntime_test_all.exe --gtest_filter="MobiusAttentionFusionTests.*"
```
