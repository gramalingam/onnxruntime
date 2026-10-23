# Opt-in Mobius stateful sequence fusions

`rewrite::MakeMobiusStatefulRules()` in
`core/optimizer/mobius_stateful_fusion.h` supplies a dedicated rule family for
`PatternRewriter`. It is not registered in the default optimizer pipeline.
The model must already import `com.microsoft` opset 1.

## A8: CausalConvWithState

The source is the standard-ONNX body of
[`causal_conv_nd_with_state`](https://github.com/onnxruntime/mobius/blob/d69efe03a9f4c1ada744bc31dd01077a710b3c39/src/mobius/functions/causal_conv.py#L23-L159),
not a function name or a contrib-node annotation. Run constant folding before
these rules to turn the factory's `Constant` operators into initializers.

Supported bodies concatenate `past_state` and input along axis 2, extract the
trailing K-1 positions, apply valid depthwise 1D Conv, reshape a per-channel
bias to `[1,-1,1]`, and add it. Activation is absent, `Sigmoid` plus `Mul`, or
standard `Swish(alpha=1)` (the latter requires ONNX opset 24).
Slice can use immutable indices with optional unit step or the factory's exact
`Shape/Gather/Sub/Reshape` computation. The computed width must equal K-1.
Both output identities, graph outputs, and their external consumers survive.
External consumers of internal values prevent fusion.
The family uses `GetPattern(name, true)`'s opt-in, unique-consumer sibling-output
matching for the otherwise independent Conv and state-Slice branches.
Ambiguous sibling Slices conservatively prevent fusion.
The no-activation rule declines a prefix consumed by Sigmoid or Swish so it
cannot preempt complete activation matching or bypass an unsupported Swish alpha.

The rules require fixed float32 input `[B,C,T]`, weight `[C,1,K]`, bias `[C]`,
and state `[B,C,K-1]`; B, C, T, K must be positive. K=1 is supported with
an explicitly empty history and an empty Slice, not the incorrect `[-0:]`
interpretation. Stride and dilation must be one and padding zero.
Emitted attributes explicitly select channels-first, ndim=1, dilation=1,
state_window=0, and the matched activation.

This base has the target schema in `core/graph/contrib_ops/bert_defs.cc` and a
float32 CPU kernel in `contrib_ops/cpu/bert/causal_conv_with_state.cc`.
The default execution-provider policy permits CPU-assigned or unassigned
nodes only. No GPU numerical coverage is claimed. Other dtypes, symbolic
dimensions, absent bias/state, alternate layouts, multi-dimensional Conv,
dilated or windowed state are intentionally unsupported. Standard ONNX import
range is conservatively 17-24.

## A9: LinearAttention is deferred

The target schema and CPU kernel **are present** in this base. The prerequisite
is nested-graph matching, not target availability.
`pattern_rewriter.cc`'s `MatchPattern` rejects any matched operator whose
`GetAttributeNameToSubgraphMap()` or `ImplicitInputDefs()` is nonempty **before**
calling the rule's C++ semantic check. Consequently an ordinary `Scan` pattern
cannot invoke a narrow gated-delta body verifier through this API. A standalone
body verifier does not remove that barrier. This family deliberately contains
no Scan-by-name rule, marker-based proof, or no-op A9 factory.

A follow-up needs an explicitly scoped matcher facility for subgraph owners,
with unchanged default rejection, then verification of all nested inputs,
outputs, captures, axes, directions, constants and arithmetic. Start with the
actual [gated-delta factory body](https://github.com/onnxruntime/mobius/blob/d69efe03a9f4c1ada744bc31dd01077a710b3c39/src/mobius/functions/linear_attention.py#L262-L353):
decay the old state, retrieve from the decayed state, apply beta to
`value-retrieval`, add the key/delta outer product, and query the updated state.
Outer reshapes, head replication, query scale, state precision and both output
interfaces must also be proved. Function names and attributes are not evidence:
the pinned LightningAttention call and registered body disagree on query scale.
No general matcher changes or schema upgrades are included here.

## Validation

`MobiusStatefulFusionTests.*` builds small standard-ONNX fixtures without contrib
source nodes. Structural tests cover activation and Slice forms, both output
identities and consumers, K=1, malformed constants, broadcast changes and
deletion boundaries. CPU parity tests compare all returned values before and
after fusion for decode, short prefill and longer prefill, while asserting the
native target node is present. Standalone opset-24 `Swish` has structural coverage;
CPU numerical coverage uses no activation and the canonical Sigmoid/Mul SiLU.
The second application must perform no rewrite. A standard-ONNX unrelated Scan
fixture confirms that even sibling-output opt-in does not bypass the
subgraph-owner rejection or reach its semantic callback.
