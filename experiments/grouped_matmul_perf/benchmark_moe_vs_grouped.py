#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Performance comparison: fused com.microsoft.MoE vs. an "expanded" MoE built from
com.microsoft.GroupedMatMul.

A standard top-k Mixture-of-Experts (MoE) feed-forward layer is implemented two ways and
timed against each other:

(a) FUSED    -- a single ``com.microsoft.MoE`` node. It takes the router logits and the
                stacked per-expert FC1/FC2 weights and does routing, the two grouped GEMMs,
                the activation, and the router-weighted combine all inside one kernel.

(b) EXPANDED -- the same computation spelled out with the smaller-fusion ops documented in
                ``docs/GroupedMatMul.md`` ("Typical MoE usage (top-k)"):

                    router          : Softmax + TopK (or com.microsoft.RouterTopK)
                    FC1 (up/gate)   : GroupedMatMul            -> [M, k, fc1_out]
                    activation      : SiLU / GELU / SwiGLU
                    FC2 (down)      : GroupedMatMul            -> [M, k, hidden]
                    combine         : Mul + ReduceSum over the k expert slots

The end goal is to quantify the perf gap between the complex all-in-one MoE fusion and the
simpler GroupedMatMul-based decomposition, to attribute that gap to a mechanism (op-dispatch
overhead vs. intermediate round-trips vs. GEMM efficiency) via ``--profile``, and to confirm
the two graphs compute the same result.

Correctness (why the two graphs agree)
--------------------------------------
* ``com.microsoft.MoE`` takes *router logits* in its ``router_probs`` input and applies the
  softmax internally, so both graphs are fed the same logits.
* ``normalize_routing_weights=1`` renormalizes the top-k weights (divide by their sum). This is
  numerically identical to ``RouterTopK`` (softmax over the top-k logits) and to
  ``Softmax -> TopK -> Div-by-sum``. Set it to match whichever expanded router is used.
* WEIGHT LAYOUT DIFFERS between the two ops and must be transposed: MoE applies ``x @ W.T``
  with weights ``(E, out, in)`` while GroupedMatMul applies ``x @ W`` with weights
  ``(E, in, out)``. Transposing the last two axes makes GroupedMatMul compute the identical
  linear map. ONE shared set of expert weights is generated per case (same seed for both
  graphs); the fused model embeds the MoE layout and the expanded model embeds its transpose.
  Only one layout is ever materialised at a time (see "Memory" below), and the two outputs are
  cross-checked with ``_rel_err_robust`` (see its docstring for why a *row-wise, percentile*
  metric is used instead of a plain global max: a rare row whose top-k routing lands on a
  fp16-precision tie between two experts' softmax probabilities can legitimately be broken
  differently by the fused and expanded graphs' independent softmax/TopK implementations,
  which is not a bug but dominates a max-based comparison).
* SwiGLU on the fused MoE CPU kernel only supports the interleaved format
  (``swiglu_fusion=1``): the FC1 output is ``2 * inter`` wide and the row layout is
  ``[gate0, linear0, gate1, linear1, ...]``; the activation is
  ``gate * sigmoid(alpha * gate) * (linear + beta)`` with per-element clamps when a
  ``swiglu_limit`` is set. The expanded FC1 GroupedMatMul produces a matching ``2 * inter``
  wide output which is de-interleaved into gate/linear before the activation, so the two
  graphs are apples-to-apples. ``--swiglu-impl`` selects whether the expanded side uses the
  standard Sigmoid+Mul+Mul form or the fused ``com.microsoft.SwiGLU`` op; the run header and
  README state which form was used.

Device / dtype notes
--------------------
* CPU is FLOAT32-ONLY for this comparison: the fused ``MoE`` fp16 CPU kernel is compiled but
  NOT registered in the CPU kernel registry, so an fp16 MoE node has no CPU kernel.
  GroupedMatMul / SwiGLU / RouterTopK all have fp16 CPU kernels, but since the fused side
  cannot run fp16 on CPU the default CPU dtype here is float32. ``--dtype float16`` is intended
  for the CUDA path (MoE has fp16/bf16 CUDA kernels). ``--providers cpu --dtype float16`` has
  no runnable combo and exits early with a clear message.
* Requires an ONNX Runtime build that contains the GroupedMatMul / SwiGLU / RouterTopK contrib
  ops (built from this branch). Run from OUTSIDE the repository root so the local
  ``onnxruntime/`` source directory does not shadow the installed package.

Memory
------
Expert weights are baked as raw-bytes ONNX initializers (production-representative; neither the
MoE nor the GroupedMatMul CPU kernel implements PrePack, so there is no prepack asymmetry). The
two layouts are large and identical in size regardless of M -- e.g. a Mixtral-swiglu layer is
~5.25 GiB per layout in fp32. To fit modest boxes this harness (1) builds and times the fused
model to completion, frees it, then builds and times the expanded model, so only ONE weight
layout is resident at a time, and (2) applies ``--mem-budget-gb``: any case whose single-layout
weight footprint exceeds the budget is reported as ``OOM-skip`` instead of crashing. Because
these multi-GiB layouts exceed protobuf's ~2 GiB single-message limit, each model is serialized
with ONNX external data (weights in a side file) and the session is created from that file path;
``model.SerializeToString()`` would otherwise raise on the >=2 GiB cases.

Usage
-----
    python benchmark_moe_vs_grouped.py --providers cpu --csv results_cpu_fp32.csv
    python benchmark_moe_vs_grouped.py --swiglu-impl fused-op
    python benchmark_moe_vs_grouped.py --router-impl routertopk
    python benchmark_moe_vs_grouped.py --profile            # per-op time breakdown
    python benchmark_moe_vs_grouped.py --intra-op-threads 1 # single-thread comparison
    python benchmark_moe_vs_grouped.py --providers cuda --dtype float16   # on a GPU host
"""

import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxruntime as ort

# (name, M tokens, K hidden, F inter_size, E num_experts, k top-k, activation, bias)
# Realistic MoE feed-forward shapes spanning decode (M=1) to large prefill (M=8192), with both a
# plain activation (silu/gelu) and swiglu at several shapes so one run exposes both paths.
# The M=4096/8192 "compute anchor" cases make the asymptotic (compute-bound) gap observable:
# if the gap shrinks there it is dispatch/round-trip overhead-bound; if it persists the fused
# GEMM path is fundamentally faster. Anchors grow activation memory, not weight memory.
DEFAULT_CASES = [
    # name,                   M,    K,    F,     E,   k, activation, bias
    ("decode-silu",           1,    4096, 14336, 8,   2, "silu",    False),
    ("decode-swiglu",         1,    4096, 14336, 8,   2, "swiglu",  False),
    ("decode-batch8-silu",    8,    4096, 14336, 8,   2, "silu",    False),
    ("decode-batch32-silu",   32,   4096, 14336, 8,   2, "silu",    False),
    ("small-silu",            128,  1024, 2048,  8,   2, "silu",    False),
    ("small-silu-bias",       128,  1024, 2048,  8,   2, "silu",    True),
    ("small-swiglu",          128,  1024, 2048,  8,   2, "swiglu",  False),
    ("small-gelu",            128,  1024, 2048,  8,   2, "gelu",    False),
    ("prefill-silu",          512,  2048, 5632,  8,   2, "silu",    False),
    ("prefill-swiglu",        512,  2048, 5632,  8,   2, "swiglu",  False),
    ("prefill-gelu",          512,  2048, 5632,  8,   2, "gelu",    False),
    ("mixtral-silu",          1024, 4096, 14336, 8,   2, "silu",    False),
    ("mixtral-swiglu",        1024, 4096, 14336, 8,   2, "swiglu",  False),
    ("deepseek-many-silu",    512,  1024, 1408,  64,  6, "silu",    False),
    ("switch-top1-silu",      1024, 768,  3072,  128, 1, "silu",    False),
    ("large-tokens-swiglu",   2048, 1024, 2048,  8,   2, "swiglu",  False),
    ("prefill-4k-silu",       4096, 2048, 5632,  8,   2, "silu",    False),
    ("prefill-8k-silu",       8192, 2048, 5632,  8,   2, "silu",    False),
]

_NP_DTYPE = {"float32": np.float32, "float16": np.float16}
_ONNX_DTYPE = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16}

_MS_OPSET = [helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]


def _fc1_out(F, activation):
    """FC1 output width. SwiGLU produces an interleaved gate+linear pair, so 2*inter."""
    return 2 * F if activation == "swiglu" else F


def _regime(M):
    """Label the arithmetic regime of a token count so decode is not read as compute."""
    if M == 1:
        return "decode/launch-bound"
    if M <= 32:
        return "decode-batch"
    return "prefill/compute"


def _const(name, np_array):
    """Small constant (shapes/scalars) baked via a Python list -- cheap for tiny tensors."""
    return helper.make_tensor(
        name=name,
        data_type=onnx.helper.np_dtype_to_tensor_dtype(np_array.dtype),
        dims=np_array.shape,
        vals=np_array.flatten().tolist(),
    )


def _raw_const(name, np_array, ext_dir):
    """Large constant stored as ONNX **external data** -- never baked into the proto.

    A single expert-weight tensor for a full MoE layer can exceed protobuf's hard ~2 GiB
    per-message limit (e.g. a swiglu fc1 of shape (E, K, 2F)). Baking it as ``raw_data`` makes
    ``make_tensor(raw=True)`` / ``graph.initializer.extend`` / ``SerializeToString`` raise
    ``google.protobuf.message.EncodeError``. Instead we write the bytes to ``ext_dir/<name>.bin``
    and reference them from a tiny ``EXTERNAL`` TensorProto, so the ModelProto stays small and ORT
    memory-maps the weights. The model MUST be saved into ``ext_dir`` for the relative location to
    resolve. ``tolist()`` on billion-element tensors is pathologically slow, so we use raw bytes.
    """
    contiguous = np.ascontiguousarray(np_array)
    rel_location = f"{name}.bin"
    with open(os.path.join(ext_dir, rel_location), "wb") as data_file:
        data_file.write(contiguous.tobytes())

    tensor = TensorProto()
    tensor.name = name
    tensor.data_type = onnx.helper.np_dtype_to_tensor_dtype(contiguous.dtype)
    tensor.dims.extend(contiguous.shape)
    tensor.data_location = TensorProto.EXTERNAL
    # Populate external_data entries directly: onnx.set_external_data() requires a pre-existing
    # raw_data field, which we deliberately never create (that is the whole point -- keep the
    # bytes out of the proto).
    for key, value in (("location", rel_location), ("offset", "0"),
                       ("length", str(contiguous.nbytes))):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    return tensor


def _make_model(nodes, inputs, outputs, initializers, opsets=None):
    graph = helper.make_graph(nodes, "moe", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=opsets or _MS_OPSET)
    model.ir_version = 10
    return model


# --------------------------------------------------------------------------------------------
# Shared expert weights (MoE layout, generated once per model from a fixed seed)
# --------------------------------------------------------------------------------------------

def make_moe_weights(K, F, E, activation, np_dtype, with_bias, seed=0):
    """Generate MoE-layout expert weights for one model.

    Returns a dict with fc1 (E, fc1_out, K), fc2 (E, K, F) and optional biases. The fused model
    embeds these directly; the expanded model embeds the last-two-axes transpose (see
    ``run_one_model``). Both models call this with the same seed, so they operate on identical
    weights and their outputs are comparable.
    """
    rng = np.random.default_rng(seed)
    fc1_out = _fc1_out(F, activation)
    # scale=0.05 keeps activations in a small, well-conditioned range so silu/gelu/swiglu stay
    # numerically stable and the fp32 fused/expanded outputs agree to ~1e-7. Large weights would
    # amplify the rounding differences between the two graph spellings and inflate rel-err.
    scale = 0.05

    weights = {
        "fc1": (rng.standard_normal((E, fc1_out, K)) * scale).astype(np_dtype),
        "fc2": (rng.standard_normal((E, K, F)) * scale).astype(np_dtype),
    }
    if with_bias:
        weights["fc1_b"] = (rng.standard_normal((E, fc1_out)) * scale).astype(np_dtype)
        weights["fc2_b"] = (rng.standard_normal((E, K)) * scale).astype(np_dtype)
    return weights


def make_feeds(M, K, E, np_dtype, seed=1):
    """Runtime inputs (weights are initializers, so only activations/logits are fed)."""
    rng = np.random.default_rng(seed)
    return {
        "input": rng.standard_normal((M, K)).astype(np_dtype),
        "router_logits": rng.standard_normal((M, E)).astype(np_dtype),
    }


def weight_bytes(K, F, E, activation, itemsize):
    """Single-layout expert-weight footprint in bytes (biases are negligible)."""
    fc1_out = _fc1_out(F, activation)
    return E * (fc1_out * K + K * F) * itemsize


# --------------------------------------------------------------------------------------------
# (a) Fused model: a single com.microsoft.MoE node
# --------------------------------------------------------------------------------------------

def build_fused_moe_model(M, K, F, E, k, activation, elem_type, np_dtype,
                          fc1_w, fc2_w, fc1_b, fc2_b,
                          normalize, activation_alpha, activation_beta, swiglu_limit, ext_dir):
    is_swiglu = activation == "swiglu"
    fc1_out = _fc1_out(F, activation)
    inputs = [
        helper.make_tensor_value_info("input", elem_type, [M, K]),
        helper.make_tensor_value_info("router_logits", elem_type, [M, E]),
    ]
    initializers = [
        _raw_const("fc1_experts_weights", fc1_w, ext_dir),
        _raw_const("fc2_experts_weights", fc2_w, ext_dir),
    ]

    node_inputs = ["input", "router_logits", "fc1_experts_weights"]
    if fc1_b is not None:
        initializers.append(_raw_const("fc1_experts_bias", fc1_b, ext_dir))
        node_inputs.append("fc1_experts_bias")
    else:
        node_inputs.append("")  # optional fc1_experts_bias absent
    node_inputs.append("fc2_experts_weights")
    if fc2_b is not None:
        initializers.append(_raw_const("fc2_experts_bias", fc2_b, ext_dir))
        node_inputs.append("fc2_experts_bias")

    attrs = dict(
        activation_type="swiglu" if is_swiglu else activation,
        k=k,
        normalize_routing_weights=1 if normalize else 0,
    )
    if is_swiglu:
        attrs["swiglu_fusion"] = 1  # interleaved: the only CPU-supported SwiGLU format
        attrs["activation_alpha"] = activation_alpha
        attrs["activation_beta"] = activation_beta
        if swiglu_limit is not None:
            attrs["swiglu_limit"] = swiglu_limit

    node = helper.make_node("MoE", node_inputs, ["output"], domain="com.microsoft", **attrs)
    output = helper.make_tensor_value_info("output", elem_type, [M, K])
    return _make_model([node], inputs, [output], initializers)


# --------------------------------------------------------------------------------------------
# (b) Expanded model: router -> GroupedMatMul(FC1) -> activation -> GroupedMatMul(FC2) -> combine
# --------------------------------------------------------------------------------------------

def _router_nodes(router_impl, k, normalize):
    """Return (nodes, extra_inits) producing 'val' [M,k] weights and 'idx' [M,k] int64."""
    if router_impl == "routertopk":
        # RouterTopK renormalizes the top-k softmax internally (== normalize_routing_weights=1).
        nodes = [
            helper.make_node("RouterTopK", ["router_logits"], ["val", "idx"],
                             domain="com.microsoft", k=k),
        ]
        return nodes, []
    # softmax-topk
    inits = [_const("k_const", np.array([k], dtype=np.int64))]
    nodes = [
        helper.make_node("Softmax", ["router_logits"], ["probs"], axis=-1),
        helper.make_node("TopK", ["probs", "k_const"], ["val_raw", "idx"],
                         axis=-1, largest=1, sorted=1),
    ]
    if normalize:
        inits.append(_const("router_axis", np.array([-1], dtype=np.int64)))
        nodes += [
            helper.make_node("ReduceSum", ["val_raw", "router_axis"], ["val_denom"],
                             keepdims=1),
            helper.make_node("Div", ["val_raw", "val_denom"], ["val"]),
        ]
    else:
        nodes.append(helper.make_node("Identity", ["val_raw"], ["val"]))
    return nodes, inits


def _activation_nodes(activation, in_name, out_name, np_dtype):
    """Non-swiglu elementwise activation matching moe_utils.cc ApplyActivation, on [M,k,F]."""
    if activation == "identity":
        return [helper.make_node("Identity", [in_name], [out_name])], []
    if activation == "relu":
        return [helper.make_node("Relu", [in_name], [out_name])], []
    if activation == "silu":
        # silu(x) = x * sigmoid(x)   (moe_utils uses no alpha for silu)
        return [
            helper.make_node("Sigmoid", [in_name], [in_name + "_sig"]),
            helper.make_node("Mul", [in_name, in_name + "_sig"], [out_name]),
        ], []
    if activation == "gelu":
        # tanh approximation, matching moe_utils.cc:
        # 0.5*x*(1 + tanh(0.7978845608*(x + 0.044715*x^3)))
        c0 = np.array(0.7978845608, dtype=np_dtype)
        c1 = np.array(0.044715, dtype=np_dtype)
        half = np.array(0.5, dtype=np_dtype)
        one = np.array(1.0, dtype=np_dtype)
        inits = [
            _const(in_name + "_c0", c0), _const(in_name + "_c1", c1),
            _const(in_name + "_half", half), _const(in_name + "_one", one),
        ]
        nodes = [
            helper.make_node("Mul", [in_name, in_name], [in_name + "_x2"]),
            helper.make_node("Mul", [in_name + "_x2", in_name], [in_name + "_x3"]),
            helper.make_node("Mul", [in_name + "_x3", in_name + "_c1"], [in_name + "_c1x3"]),
            helper.make_node("Add", [in_name, in_name + "_c1x3"], [in_name + "_inner0"]),
            helper.make_node("Mul", [in_name + "_inner0", in_name + "_c0"], [in_name + "_inner"]),
            helper.make_node("Tanh", [in_name + "_inner"], [in_name + "_tanh"]),
            helper.make_node("Add", [in_name + "_tanh", in_name + "_one"], [in_name + "_1pt"]),
            helper.make_node("Mul", [in_name, in_name + "_half"], [in_name + "_halfx"]),
            helper.make_node("Mul", [in_name + "_halfx", in_name + "_1pt"], [out_name]),
        ]
        return nodes, inits
    raise ValueError(f"Unsupported non-swiglu activation: {activation}")


def _swiglu_nodes(swiglu_impl, gate_name, linear_name, out_name, np_dtype,
                  activation_alpha, activation_beta, swiglu_limit):
    """SwiGLU on de-interleaved gate/linear [M,k,F] -> out [M,k,F]."""
    nodes, inits = [], []

    if swiglu_limit is not None:
        # Clamp gate above and linear on both sides, matching moe_utils.cc; the fused SwiGLU op
        # has no built-in clamp, so it is applied here via Clip for both impls.
        limit = np.array(swiglu_limit, dtype=np_dtype)
        neg_limit = np.array(-swiglu_limit, dtype=np_dtype)
        inits += [_const("swiglu_limit_hi", limit), _const("swiglu_limit_lo", neg_limit)]
        nodes.append(helper.make_node("Clip", [gate_name, "", "swiglu_limit_hi"],
                                      [gate_name + "_cl"]))
        nodes.append(helper.make_node("Clip", [linear_name, "swiglu_limit_lo", "swiglu_limit_hi"],
                                      [linear_name + "_cl"]))
        gate_name, linear_name = gate_name + "_cl", linear_name + "_cl"

    if swiglu_impl == "fused-op":
        # com.microsoft.SwiGLU has no beta term; clamps (if any) were applied above via Clip.
        if activation_beta != 0.0:
            raise ValueError("com.microsoft.SwiGLU has no beta; use --swiglu-impl expanded "
                             "for activation_beta != 0.")
        nodes.append(helper.make_node("SwiGLU", [gate_name, linear_name], [out_name],
                                      domain="com.microsoft", alpha=activation_alpha))
        return nodes, inits

    # expanded: gate * sigmoid(alpha*gate) * (linear + beta)
    gate_for_sigmoid = gate_name
    if activation_alpha != 1.0:
        alpha = np.array(activation_alpha, dtype=np_dtype)
        inits.append(_const("swiglu_alpha", alpha))
        nodes.append(helper.make_node("Mul", [gate_name, "swiglu_alpha"], [gate_name + "_a"]))
        gate_for_sigmoid = gate_name + "_a"
    nodes.append(helper.make_node("Sigmoid", [gate_for_sigmoid], [gate_name + "_sig"]))
    nodes.append(helper.make_node("Mul", [gate_name, gate_name + "_sig"], [gate_name + "_swish"]))

    linear_term = linear_name
    if activation_beta != 0.0:
        beta = np.array(activation_beta, dtype=np_dtype)
        inits.append(_const("swiglu_beta", beta))
        nodes.append(helper.make_node("Add", [linear_name, "swiglu_beta"], [linear_name + "_b"]))
        linear_term = linear_name + "_b"
    nodes.append(helper.make_node("Mul", [gate_name + "_swish", linear_term], [out_name]))
    return nodes, inits


def build_expanded_model(M, K, F, E, k, activation, elem_type, np_dtype,
                         fc1_w, fc2_w, fc1_b, fc2_b,
                         normalize, router_impl, swiglu_impl,
                         activation_alpha, activation_beta, swiglu_limit, ext_dir,
                         fused_reduce=False):
    is_swiglu = activation == "swiglu"
    fc1_out = _fc1_out(F, activation)

    inputs = [
        helper.make_tensor_value_info("input", elem_type, [M, K]),
        helper.make_tensor_value_info("router_logits", elem_type, [M, E]),
    ]
    initializers = [
        _raw_const("fc1_gmm_weights", fc1_w, ext_dir),  # (E, K, fc1_out)
        _raw_const("fc2_gmm_weights", fc2_w, ext_dir),  # (E, F, K)
        _const("shape_Mk_F", np.array([M * k, F], dtype=np.int64)),
        _const("shape_Mk_1", np.array([M * k, 1], dtype=np.int64)),
        _const("shape_M_k_K", np.array([M, k, K], dtype=np.int64)),
        _const("combine_axis", np.array([1], dtype=np.int64)),
        _const("neg1", np.array([-1], dtype=np.int64)),
    ]
    if fc1_b is not None:
        initializers.append(_raw_const("fc1_gmm_bias", fc1_b, ext_dir))
    if fc2_b is not None:
        initializers.append(_raw_const("fc2_gmm_bias", fc2_b, ext_dir))
    nodes = []

    # --- Router: produce val [M,k] and idx [M,k] int64 ---
    router_nodes, router_inits = _router_nodes(router_impl, k, normalize)
    nodes += router_nodes
    initializers += router_inits

    # --- FC1 (up/gate projection): GroupedMatMul -> [M, k, fc1_out] ---
    fc1_gmm_inputs = ["input", "fc1_gmm_weights", "idx"]
    if fc1_b is not None:
        # GroupedMatMul bias is (E, N) with N = fc1_out; MoE fc1 bias is (E, fc1_out). Same.
        fc1_gmm_inputs.append("fc1_gmm_bias")
    nodes.append(helper.make_node("GroupedMatMul", fc1_gmm_inputs, ["h1"],
                                  domain="com.microsoft"))

    # --- Activation -> act [M, k, F] ---
    if is_swiglu:
        # De-interleave h1 [M,k,2F] -> gate/linear [M,k,F].
        # Reshape to [M,k,F,2] then split the last axis into gate/linear.
        initializers.append(_const("shape_M_k_F_2", np.array([M, k, F, 2], dtype=np.int64)))
        initializers.append(_const("split_1_1", np.array([1, 1], dtype=np.int64)))
        initializers.append(_const("shape_axis3", np.array([3], dtype=np.int64)))
        nodes.append(helper.make_node("Reshape", ["h1", "shape_M_k_F_2"], ["h1_pairs"]))
        nodes.append(helper.make_node("Split", ["h1_pairs", "split_1_1"],
                                      ["gate_p", "linear_p"], axis=3))
        nodes.append(helper.make_node("Squeeze", ["gate_p", "shape_axis3"], ["gate"]))
        nodes.append(helper.make_node("Squeeze", ["linear_p", "shape_axis3"], ["linear"]))
        sw_nodes, sw_inits = _swiglu_nodes(swiglu_impl, "gate", "linear", "act", np_dtype,
                                           activation_alpha, activation_beta, swiglu_limit)
        nodes += sw_nodes
        initializers += sw_inits
    else:
        act_nodes, act_inits = _activation_nodes(activation, "h1", "act", np_dtype)
        nodes += act_nodes
        initializers += act_inits

    if fused_reduce:
        # --- FC2 + combine fused: com.microsoft.GroupedMatMulReduceSum takes "act" [M,k,F]
        # directly (one distinct row per (token, expert-slot) selection -- no reshape needed)
        # and produces the router-weighted-summed [M, K] output in a single op, replacing
        # GroupedMatMul + Reshape + Mul + ReduceSum below.
        fc2_fused_inputs = ["act", "fc2_gmm_weights", "idx", "val"]
        if fc2_b is not None:
            fc2_fused_inputs.append("fc2_gmm_bias")
        nodes.append(helper.make_node("GroupedMatMulReduceSum", fc2_fused_inputs, ["output"],
                                      domain="com.microsoft"))
    else:
        # --- FC2 (down projection): GroupedMatMul over [M*k, F] with idx_flat [M*k, 1] ---
        nodes.append(helper.make_node("Reshape", ["act", "shape_Mk_F"], ["act_flat"]))
        nodes.append(helper.make_node("Reshape", ["idx", "shape_Mk_1"], ["idx_flat"]))
        fc2_gmm_inputs = ["act_flat", "fc2_gmm_weights", "idx_flat"]
        if fc2_b is not None:
            # MoE fc2 bias is (E, K); GroupedMatMul bias (E, N=K). Same.
            fc2_gmm_inputs.append("fc2_gmm_bias")
        nodes.append(helper.make_node("GroupedMatMul", fc2_gmm_inputs, ["down_flat"],
                                      domain="com.microsoft"))  # [M*k, 1, K]
        nodes.append(helper.make_node("Reshape", ["down_flat", "shape_M_k_K"], ["down"]))  # [M, k, K]

        # --- Combine: router-weighted sum over the k expert slots -> [M, K] ---
        nodes.append(helper.make_node("Unsqueeze", ["val", "neg1"], ["val_u"]))  # [M, k, 1]
        nodes.append(helper.make_node("Mul", ["down", "val_u"], ["down_weighted"]))
        nodes.append(helper.make_node("ReduceSum", ["down_weighted", "combine_axis"], ["output"],
                                      keepdims=0))  # [M, K]

    output = helper.make_tensor_value_info("output", elem_type, [M, K])
    return _make_model(nodes, inputs, [output], initializers)


# --------------------------------------------------------------------------------------------
# Sessions / timing (SessionOptions + _provider_list + perf_counter loop reused from the
# sibling benchmark_grouped_matmul.py, extended with thread pinning and optional profiling)
# --------------------------------------------------------------------------------------------

def _provider_list(name):
    if name == "cpu":
        return ["CPUExecutionProvider"]
    if name == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise ValueError(name)


def _session_options(intra_op_threads, profile_prefix=None):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op_threads is not None:
        so.intra_op_num_threads = intra_op_threads
    if profile_prefix is not None:
        so.enable_profiling = True
        # Pin the profile file to a known location; end_profiling() otherwise returns a name
        # resolved against the (possibly unwritable) current working directory.
        so.profile_file_prefix = profile_prefix
    return so


def time_model(model_source, feeds, provider, warmup, iters, intra_op_threads):
    so = _session_options(intra_op_threads)
    sess = ort.InferenceSession(model_source, so, providers=_provider_list(provider))
    if provider == "cuda" and "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError("CUDAExecutionProvider not available in this ORT build")
    out_names = [o.name for o in sess.get_outputs()]

    for _ in range(warmup):
        result = sess.run(out_names, feeds)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = sess.run(out_names, feeds)
        times.append(time.perf_counter() - t0)
    return np.array(times), result[0]


def profile_breakdown(model_source, feeds, provider, warmup, iters, intra_op_threads):
    """Run with ORT profiling on and aggregate per-op-type kernel time (microseconds).

    Returns a list of (op_type, count, total_us) sorted by total_us descending, which attributes
    the fused-vs-expanded gap to op dispatch overhead vs. GEMM time vs. reshape/round-trips.
    """
    so = _session_options(intra_op_threads, profile_prefix=os.path.join(
        tempfile.gettempdir(), "ort_moe_prof"))
    sess = ort.InferenceSession(model_source, so, providers=_provider_list(provider))
    out_names = [o.name for o in sess.get_outputs()]
    for _ in range(max(1, warmup)):
        sess.run(out_names, feeds)
    for _ in range(max(3, iters)):
        sess.run(out_names, feeds)
    prof_file = sess.end_profiling()

    agg = {}
    if not os.path.isabs(prof_file):
        prof_file = os.path.join(tempfile.gettempdir(), prof_file)
    try:
        with open(prof_file) as f:
            events = json.load(f)
        for e in events:
            if e.get("cat") == "Node" and str(e.get("name", "")).endswith("_kernel_time"):
                op = e.get("args", {}).get("op_name", "?")
                count, total = agg.get(op, (0, 0))
                agg[op] = (count + 1, total + int(e.get("dur", 0)))
    finally:
        try:
            os.remove(prof_file)
        except OSError:
            pass
    return sorted(([op, c, t] for op, (c, t) in agg.items()), key=lambda r: -r[2])


def _rel_err(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.maximum(np.abs(b).max(), 1e-6)
    return float(np.abs(a - b).max() / denom)


def _rel_err_robust(a, b, row_percentile=99.0, row_tol=None):
    """Row-wise relative error, robust to a handful of MoE rows whose *routing* legitimately
    disagrees between the two graphs.

    The fused and expanded graphs pick their top-k experts from a Softmax(router_logits) that
    both graphs compute independently (once fused-side, once as an explicit Softmax+TopK).
    When two experts' probabilities for a row land within fp16 precision of each other, CPU and
    CUDA (or even two different kernels on the same device) can legitimately break the tie in
    opposite directions -- this is unspecified by the ONNX TopK op, not a kernel bug. A
    different top-k expert set for that one row then produces a *completely different* (but
    individually correct) output row, since the two experts' weights are unrelated. A single
    such row dominates a plain max-based relative error over the whole [M, ...] tensor and
    makes the comparison meaningless for judging correctness of the actual math.

    Instead, compute the max relative error *per row* (over all non-batch axes), then take a
    high percentile across rows. This still fails loudly on a real, broad-based bug (which
    perturbs most/all rows) while tolerating the rare row that took a different, equally valid,
    routing decision.

    Returns (relerr_row_percentile, relerr_max, num_outlier_rows) where a row counts as an
    "outlier" if its own max relative error exceeds ``row_tol`` (defaults to 10x the usual fp16
    tolerance, i.e. it looks like a genuinely different routing decision rather than noise).
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.maximum(np.abs(b).max(), 1e-6)
    per_row = np.abs(a - b).reshape(a.shape[0], -1).max(axis=1) / denom
    relerr_max = float(per_row.max())
    relerr_pct = float(np.percentile(per_row, row_percentile))
    tol = row_tol if row_tol is not None else 5e-1
    num_outliers = int((per_row > tol).sum())
    return relerr_pct, relerr_max, num_outliers


def _stats_ms(times_s):
    ms = times_s * 1e3
    return dict(mean=float(ms.mean()), median=float(np.median(ms)),
                min=float(ms.min()), std=float(ms.std()))


# --------------------------------------------------------------------------------------------
# Per-model build+run (only one weight layout is resident at a time)
# --------------------------------------------------------------------------------------------

def run_one_model(kind, M, K, F, E, k, activation, elem_type, np_dtype,
                  feeds, provider, args, normalize, with_bias):
    """Generate weights, build the graph, time it, optionally profile it; free everything.

    ``kind`` is "fused", "expanded", or "expanded_fused_reduce" (FC2 + combine fused into a
    single com.microsoft.GroupedMatMulReduceSum node). Weights are generated (and, for the
    expanded models, transposed) here and released before the session is created so that the
    MoE-layout and GroupedMatMul-layout copies are never resident simultaneously.
    """
    # Expert weights are written directly to ONNX external data files (see _raw_const): a single
    # MoE weight tensor can exceed protobuf's hard ~2 GiB per-message limit, so baking it into the
    # proto would raise google.protobuf.message.EncodeError at graph-build / serialize time. The
    # weights must live in the same directory as the model file for their relative locations to
    # resolve, so the temp dir is created up front and passed into the builders.
    tmp_dir = tempfile.mkdtemp(prefix="ort_moe_model_")
    try:
        moe = make_moe_weights(K, F, E, activation, np_dtype, with_bias)
        fc1_b = moe.get("fc1_b")
        fc2_b = moe.get("fc2_b")

        if kind == "fused":
            model = build_fused_moe_model(
                M, K, F, E, k, activation, elem_type, np_dtype,
                moe["fc1"], moe["fc2"], fc1_b, fc2_b,
                normalize, args.activation_alpha, args.activation_beta, args.swiglu_limit,
                tmp_dir)
            del moe
        else:
            # MoE applies x @ W.T (weights (E,out,in)); GroupedMatMul applies x @ W (weights
            # (E,in,out)). Transposing the last two axes makes GroupedMatMul compute the identical
            # linear map, so the two graphs are numerically comparable.
            fc1_t = np.ascontiguousarray(np.swapaxes(moe["fc1"], 1, 2))
            fc2_t = np.ascontiguousarray(np.swapaxes(moe["fc2"], 1, 2))
            del moe  # free the MoE-layout source before building the (transposed) proto
            model = build_expanded_model(
                M, K, F, E, k, activation, elem_type, np_dtype,
                fc1_t, fc2_t, fc1_b, fc2_b,
                normalize, args.router_impl, args.swiglu_impl,
                args.activation_alpha, args.activation_beta, args.swiglu_limit,
                tmp_dir, fused_reduce=(kind == "expanded_fused_reduce"))
            del fc1_t, fc2_t

        # Weights are already external; a plain save writes only the (small) ModelProto, whose
        # initializers reference the side files written into tmp_dir by _raw_const.
        model_path = os.path.join(tmp_dir, "model.onnx")
        onnx.save_model(model, model_path)
        del model  # the on-disk model + external data files are all the session needs

        times, out = time_model(model_path, feeds, provider, args.warmup, args.iters,
                                args.intra_op_threads)
        breakdown = None
        if args.profile:
            breakdown = profile_breakdown(model_path, feeds, provider, args.warmup, args.iters,
                                          args.intra_op_threads)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return times, out, breakdown


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------

def _print_breakdown(label, breakdown, total_ms):
    print(f"    [profile] {label} per-op kernel time (of {total_ms:.3f} ms mean):")
    for op, count, total_us in breakdown:
        print(f"        {op:<20}{count:>6} calls{total_us / 1e3:>12.3f} ms")


def run_benchmark(args):
    np_dtype = _NP_DTYPE[args.dtype]
    elem_type = _ONNX_DTYPE[args.dtype]
    itemsize = np.dtype(np_dtype).itemsize

    normalize = bool(args.normalize)
    if args.router_impl == "routertopk" and not normalize:
        print("[note] --router-impl routertopk always renormalizes top-k; forcing "
              "normalize_routing_weights=1 for the fused MoE to match.")
        normalize = True

    providers = []
    for p in args.providers:
        if p == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            print(f"[skip] CUDA requested but not available in ORT build "
                  f"({ort.get_available_providers()})")
            continue
        if p == "cpu" and args.dtype == "float16":
            print("[skip] float16 on CPU: the fused MoE fp16 CPU kernel is not registered. "
                  "Use float32 on CPU (float16 is for CUDA).")
            continue
        providers.append(p)

    if not providers:
        print("[error] No runnable provider/dtype combination. On CPU only float32 is "
              "supported (fused MoE has no fp16 CPU kernel); float16 requires an available "
              "CUDAExecutionProvider. Re-run with e.g. '--providers cpu --dtype float32' or on "
              "a CUDA host. Exiting.")
        return

    threads_reported = args.intra_op_threads if args.intra_op_threads is not None \
        else (os.cpu_count() or 0)
    threads_label = str(args.intra_op_threads) if args.intra_op_threads is not None \
        else f"auto(~{threads_reported})"

    print(f"onnxruntime {ort.__version__}; providers available: "
          f"{ort.get_available_providers()}")
    print(f"dtype={args.dtype} bias={args.bias} normalize={int(normalize)} "
          f"router-impl={args.router_impl} swiglu-impl={args.swiglu_impl} "
          f"intra-op-threads={threads_label}")
    print(f"warmup={args.warmup} iters={args.iters} strict={args.strict} "
          f"mem-budget={args.mem_budget_gb} GiB profile={args.profile}")
    print("SwiGLU: fused MoE uses swiglu_fusion=1 (interleaved); expanded uses "
          f"'{args.swiglu_impl}' "
          f"({'com.microsoft.SwiGLU op' if args.swiglu_impl == 'fused-op' else 'Sigmoid+Mul+Mul'})"
          ". Reported ms are MEDIAN over timed iters.\n")

    header = (f"{'case':<20}{'act':<8}{'regime':<20}{'prov':<6}{'M':>6}{'K':>6}{'F':>7}{'E':>5}"
              f"{'k':>3}{'fused_ms':>11}{'exp_ms':>11}{'fused_speedup':>14}{'max_relerr':>12}{'ok':>4}")
    if args.fused_reduce:
        header += f"{'expFR_ms':>11}{'FR_speedup':>12}{'FR_relerr':>12}{'FR_ok':>6}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, M, K, F, E, k, activation, case_bias in args.cases:
        with_bias = args.bias or case_bias
        regime = _regime(M)
        w_gib = weight_bytes(K, F, E, activation, itemsize) / (1024 ** 3)
        fits = w_gib <= args.mem_budget_gb
        feeds = make_feeds(M, K, E, np_dtype)

        for provider in providers:
            if not fits:
                print(f"{name:<20}{activation:<8}{regime:<20}{provider:<6}{M:>6}{K:>6}{F:>7}"
                      f"{E:>5}{k:>3}{'OOM-skip':>11}{'-':>11}{'-':>14}"
                      f"{f'>{args.mem_budget_gb}G ({w_gib:.1f}G)':>12}{'-':>4}")
                rows.append(dict(
                    case=name, activation=activation, regime=regime, provider=provider,
                    M=M, K=K, F=F, num_experts=E, k=k, weight_gib=w_gib, skipped="oom",
                    fused_median_ms=None, expanded_median_ms=None, fused_speedup=None,
                    max_relerr=None, ok=None, threads=threads_reported,
                    router_impl=args.router_impl, swiglu_impl=args.swiglu_impl,
                    normalize=int(normalize)))
                continue

            fused_t, fused_out, fused_bd = run_one_model(
                "fused", M, K, F, E, k, activation, elem_type, np_dtype,
                feeds, provider, args, normalize, with_bias)
            exp_t, exp_out, exp_bd = run_one_model(
                "expanded", M, K, F, E, k, activation, elem_type, np_dtype,
                feeds, provider, args, normalize, with_bias)

            fused_s = _stats_ms(fused_t)
            exp_s = _stats_ms(exp_t)
            relerr, relerr_max, num_outlier_rows = _rel_err_robust(exp_out, fused_out)
            tol = args.tol if args.tol is not None \
                else (1e-4 if args.dtype == "float32" else 5e-2)
            ok = relerr <= tol
            # A divergent run must not advertise a speedup -- the two graphs are not the same
            # function, so the ratio is meaningless.
            speedup = exp_s["median"] / fused_s["median"] if ok else float("nan")
            speedup_str = f"{speedup:.2f}x" if ok else "DIVERGED"

            line = (f"{name:<20}{activation:<8}{regime:<20}{provider:<6}{M:>6}{K:>6}{F:>7}{E:>5}"
                    f"{k:>3}{fused_s['median']:>11.3f}{exp_s['median']:>11.3f}{speedup_str:>14}"
                    f"{relerr:>12.2e}{('yes' if ok else 'NO'):>4}")

            fr_s = fr_out = fr_bd = None
            fr_speedup = fr_relerr = None
            fr_ok = None
            if args.fused_reduce:
                fr_t, fr_out, fr_bd = run_one_model(
                    "expanded_fused_reduce", M, K, F, E, k, activation, elem_type, np_dtype,
                    feeds, provider, args, normalize, with_bias)
                fr_s = _stats_ms(fr_t)
                fr_relerr, fr_relerr_max, fr_num_outlier_rows = _rel_err_robust(fr_out, fused_out)
                fr_ok = fr_relerr <= tol
                fr_speedup = fr_s["median"] / fused_s["median"] if fr_ok else float("nan")
                fr_speedup_str = f"{fr_speedup:.2f}x" if fr_ok else "DIVERGED"
                line += (f"{fr_s['median']:>11.3f}{fr_speedup_str:>12}{fr_relerr:>12.2e}"
                        f"{('yes' if fr_ok else 'NO'):>6}")

            print(line)

            if args.profile:
                if fused_bd:
                    _print_breakdown("fused", fused_bd, fused_s["mean"])
                if exp_bd:
                    _print_breakdown("expanded", exp_bd, exp_s["mean"])
                if fr_bd:
                    _print_breakdown("expanded_fused_reduce", fr_bd, fr_s["mean"])

            if num_outlier_rows:
                print(f"    [INFO] {name}: {num_outlier_rows} row(s) took a different (but "
                      f"individually valid) top-k routing decision -- a near-tie in the "
                      f"softmax(router_logits) below fp16 precision, broken differently by "
                      f"CPU/CUDA TopK. Excluded from the 99th-percentile rel err "
                      f"({relerr:.3e}); raw max rel err over all rows was {relerr_max:.3e}.")

            if not ok:
                print(f"    [WARN] {name}: 99th-pct rel err {relerr:.3e} exceeds tol {tol:.1e} "
                      f"-- fused and expanded disagree; speedup suppressed.")
                if args.strict:
                    raise AssertionError(
                        f"{name}: rel err {relerr:.3e} > tol {tol:.1e} (strict mode)")

            if args.fused_reduce and not fr_ok:
                print(f"    [WARN] {name}: expanded_fused_reduce 99th-pct rel err {fr_relerr:.3e} "
                      f"exceeds tol {tol:.1e} vs. fused MoE -- speedup suppressed.")
                if args.strict:
                    raise AssertionError(
                        f"{name}: expanded_fused_reduce rel err {fr_relerr:.3e} > tol {tol:.1e} "
                        f"(strict mode)")

            row = dict(
                case=name, activation=activation, regime=regime, provider=provider,
                M=M, K=K, F=F, num_experts=E, k=k, weight_gib=w_gib, skipped="",
                fused_median_ms=fused_s["median"], fused_mean_ms=fused_s["mean"],
                fused_min_ms=fused_s["min"], fused_std_ms=fused_s["std"],
                expanded_median_ms=exp_s["median"], expanded_mean_ms=exp_s["mean"],
                expanded_min_ms=exp_s["min"], expanded_std_ms=exp_s["std"],
                fused_speedup=(speedup if ok else None), max_relerr=relerr, ok=ok,
                threads=threads_reported, router_impl=args.router_impl,
                swiglu_impl=args.swiglu_impl, normalize=int(normalize))
            if args.fused_reduce:
                row.update(
                    expanded_fused_reduce_median_ms=fr_s["median"],
                    expanded_fused_reduce_mean_ms=fr_s["mean"],
                    expanded_fused_reduce_min_ms=fr_s["min"],
                    expanded_fused_reduce_std_ms=fr_s["std"],
                    expanded_fused_reduce_speedup=(fr_speedup if fr_ok else None),
                    expanded_fused_reduce_max_relerr=fr_relerr,
                    expanded_fused_reduce_ok=fr_ok)
            rows.append(row)

    if args.csv and rows:
        import csv
        fieldnames = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--providers", nargs="+", default=["cpu", "cuda"],
                   choices=["cpu", "cuda"],
                   help="Execution providers to benchmark (cuda auto-skips if unavailable).")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                   help="float32 is the only CPU option (fused MoE fp16 CPU kernel is not "
                        "registered); float16 is for the CUDA path.")
    p.add_argument("--router-impl", default="softmax-topk",
                   choices=["softmax-topk", "routertopk"],
                   help="Expanded router: standard Softmax+TopK(+renorm) or the fused "
                        "com.microsoft.RouterTopK op.")
    p.add_argument("--swiglu-impl", default="expanded", choices=["expanded", "fused-op"],
                   help="For swiglu cases, the expanded side uses either the standard "
                        "Sigmoid+Mul+Mul form or the fused com.microsoft.SwiGLU op.")
    p.add_argument("--bias", action="store_true",
                   help="Force optional FC1/FC2 biases on for every case (some cases enable "
                        "bias by default regardless).")
    p.add_argument("--normalize", type=int, default=1, choices=[0, 1],
                   help="normalize_routing_weights: renormalize top-k weights (default 1).")
    p.add_argument("--activation-alpha", type=float, default=1.0,
                   help="SwiGLU sigmoid gate scale (default 1.0 = SiLU gate).")
    p.add_argument("--activation-beta", type=float, default=0.0,
                   help="SwiGLU linear bias (default 0.0). Not supported by the fused SwiGLU op.")
    p.add_argument("--swiglu-limit", type=float, default=None,
                   help="Optional SwiGLU clamp limit (default: no clamp).")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30,
                   help="Timed iterations per model (default 30 for stable median/std).")
    p.add_argument("--intra-op-threads", type=int, default=None,
                   help="Pin SessionOptions.intra_op_num_threads (e.g. 1 for single-thread). "
                        "Default: ORT's automatic count (reported in the header).")
    p.add_argument("--mem-budget-gb", type=float, default=4.0,
                   help="Skip (OOM-skip) any case whose single-layout expert-weight footprint "
                        "exceeds this many GiB. Peak RSS is roughly 2x this per model.")
    p.add_argument("--profile", action="store_true",
                   help="Enable ORT profiling and print a per-op kernel-time breakdown so the "
                        "fused-vs-expanded gap can be attributed to a mechanism.")
    p.add_argument("--tol", type=float, default=None,
                   help="Cross-check tolerance (default 1e-4 fp32 / 5e-2 fp16).")
    p.add_argument("--strict", dest="strict", action="store_true", default=True,
                   help="Raise if fused and expanded disagree beyond tolerance (default ON).")
    p.add_argument("--no-strict", dest="strict", action="store_false",
                   help="Report divergence as a warning instead of raising.")
    p.add_argument("--fused-reduce", action="store_true",
                   help="Also run a third variant: 'expanded' with FC2 (down-projection) + the "
                        "router-weighted combine fused into a single com.microsoft."
                        "GroupedMatMulReduceSum node (replacing GroupedMatMul + Reshape + Mul + "
                        "ReduceSum), and compare it against the fused MoE op alongside the "
                        "plain 'expanded' baseline.")
    p.add_argument("--csv", default=None, help="Optional path to write results as CSV.")
    args = p.parse_args()
    args.cases = DEFAULT_CASES
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
