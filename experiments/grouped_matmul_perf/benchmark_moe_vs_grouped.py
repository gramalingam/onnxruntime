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
simpler GroupedMatMul-based decomposition, and to confirm they compute the same result.

Correctness (why the two graphs agree)
--------------------------------------
* ``com.microsoft.MoE`` takes *router logits* in its ``router_probs`` input and applies the
  softmax internally, so both graphs are fed the same logits.
* ``normalize_routing_weights=1`` renormalizes the top-k weights (divide by their sum). This is
  numerically identical to ``RouterTopK`` (softmax over the top-k logits) and to
  ``Softmax -> TopK -> Div-by-sum``. Set it to match whichever expanded router is used.
* WEIGHT LAYOUT DIFFERS between the two ops and must be transposed:
    - MoE ``fc1_experts_weights`` = ``(E, fc1_out, hidden)`` and is applied as ``x @ W.T``.
      GroupedMatMul ``weights`` = ``(E, hidden, fc1_out)`` and is applied as ``x @ W``.
      So the GroupedMatMul FC1 weight is the *transpose* (last two axes) of the MoE FC1 weight.
    - MoE ``fc2_experts_weights`` = ``(E, hidden, inter)``; GroupedMatMul FC2 weight is its
      transpose ``(E, inter, hidden)``.
  This script generates ONE shared set of expert weights and feeds the transposed variant to
  whichever graph needs it, then cross-checks the two outputs with ``_rel_err``.
* SwiGLU on the fused MoE CPU kernel only supports the interleaved format
  (``swiglu_fusion=1``): the FC1 output is ``2 * inter`` wide and the row layout is
  ``[gate0, linear0, gate1, linear1, ...]``; the activation is
  ``gate * sigmoid(alpha * gate) * (linear + beta)`` with per-element clamps when a
  ``swiglu_limit`` is set. The expanded FC1 GroupedMatMul produces a matching ``2 * inter``
  wide output which is de-interleaved into gate/linear before the activation, so the two
  graphs are apples-to-apples.

Device / dtype notes
--------------------
* CPU is FLOAT32-ONLY for this comparison: the fused ``MoE`` fp16 CPU kernel is compiled but
  NOT registered in the CPU kernel registry, so an fp16 MoE node has no CPU kernel.
  GroupedMatMul / SwiGLU / RouterTopK all have fp16 CPU kernels, but since the fused side
  cannot run fp16 on CPU the default CPU dtype here is float32. ``--dtype float16`` is intended
  for the CUDA path (MoE has fp16/bf16 CUDA kernels).
* Requires an ONNX Runtime build that contains the GroupedMatMul / SwiGLU / RouterTopK contrib
  ops (built from this branch). Run from OUTSIDE the repository root so the local
  ``onnxruntime/`` source directory does not shadow the installed package.

Usage
-----
    python benchmark_moe_vs_grouped.py --providers cpu --csv results_moe_vs_grouped_cpu_fp32.csv
    python benchmark_moe_vs_grouped.py --swiglu-impl fused-op
    python benchmark_moe_vs_grouped.py --router-impl routertopk
    python benchmark_moe_vs_grouped.py --providers cuda --dtype float16   # on a GPU host
"""

import argparse
import time

import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxruntime as ort

# (name, M tokens, K hidden, F inter_size, E num_experts, k top-k, activation)
# Realistic MoE feed-forward shapes spanning decode (M=1) to prefill (M=2048), with both a
# plain activation (silu/gelu) and swiglu at several shapes so one run exposes both paths.
DEFAULT_CASES = [
    # name,                     M,    K,    F,     E,   k,  activation
    ("decode-silu",             1,    4096, 14336, 8,   2,  "silu"),
    ("decode-swiglu",           1,    4096, 14336, 8,   2,  "swiglu"),
    ("small-silu",              128,  1024, 2048,  8,   2,  "silu"),
    ("small-swiglu",            128,  1024, 2048,  8,   2,  "swiglu"),
    ("small-gelu",              128,  1024, 2048,  8,   2,  "gelu"),
    ("prefill-silu",            512,  2048, 5632,  8,   2,  "silu"),
    ("prefill-swiglu",          512,  2048, 5632,  8,   2,  "swiglu"),
    ("mixtral-silu",            1024, 4096, 14336, 8,   2,  "silu"),
    ("mixtral-swiglu",          1024, 4096, 14336, 8,   2,  "swiglu"),
    ("deepseek-many-silu",      512,  1024, 1408,  64,  6,  "silu"),
    ("switch-top1-silu",        1024, 768,  3072,  128, 1,  "silu"),
    ("large-tokens-swiglu",     2048, 1024, 2048,  8,   2,  "swiglu"),
]

_NP_DTYPE = {"float32": np.float32, "float16": np.float16}
_ONNX_DTYPE = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16}

_MS_OPSET = [helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]


def _const(name, np_array):
    return helper.make_tensor(
        name=name,
        data_type=onnx.helper.np_dtype_to_tensor_dtype(np_array.dtype),
        dims=np_array.shape,
        vals=np_array.flatten().tolist(),
    )


def _make_model(nodes, inputs, outputs, initializers, opsets=None):
    graph = helper.make_graph(nodes, "moe", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=opsets or _MS_OPSET)
    model.ir_version = 10
    return model


# --------------------------------------------------------------------------------------------
# Shared expert weights (generated once, fed transposed to each graph so outputs are equal)
# --------------------------------------------------------------------------------------------

def make_shared_weights(K, F, E, activation, np_dtype, with_bias, seed=0):
    """Generate MoE-layout expert weights, plus their GroupedMatMul-layout transposes.

    Returns a dict with:
        fc1_moe : (E, fc1_out, K)   -- MoE fc1_experts_weights (fc1_out = 2F for swiglu else F)
        fc2_moe : (E, K, F)         -- MoE fc2_experts_weights
        fc1_gmm : (E, K, fc1_out)   -- GroupedMatMul FC1 weight (transpose of fc1_moe)
        fc2_gmm : (E, F, K)         -- GroupedMatMul FC2 weight (transpose of fc2_moe)
    Optional biases (MoE layout) when with_bias.
    """
    rng = np.random.default_rng(seed)
    is_swiglu = activation == "swiglu"
    fc1_out = 2 * F if is_swiglu else F
    scale = 0.05

    fc1_moe = (rng.standard_normal((E, fc1_out, K)) * scale).astype(np_dtype)
    fc2_moe = (rng.standard_normal((E, K, F)) * scale).astype(np_dtype)

    weights = {
        "fc1_moe": fc1_moe,
        "fc2_moe": fc2_moe,
        # Transpose last two axes: (E, out, in) -> (E, in, out).
        "fc1_gmm": np.ascontiguousarray(fc1_moe.transpose(0, 2, 1)),
        "fc2_gmm": np.ascontiguousarray(fc2_moe.transpose(0, 2, 1)),
    }
    if with_bias:
        weights["fc1_bias_moe"] = (rng.standard_normal((E, fc1_out)) * scale).astype(np_dtype)
        weights["fc2_bias_moe"] = (rng.standard_normal((E, K)) * scale).astype(np_dtype)
    return weights


def make_feeds(M, K, E, np_dtype, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "input": rng.standard_normal((M, K)).astype(np_dtype),
        "router_logits": rng.standard_normal((M, E)).astype(np_dtype),
    }


# --------------------------------------------------------------------------------------------
# (a) Fused model: a single com.microsoft.MoE node
# --------------------------------------------------------------------------------------------

def build_fused_moe_model(M, K, F, E, k, activation, elem_type, with_bias,
                          normalize, activation_alpha, activation_beta, swiglu_limit):
    # Expert weights are graph inputs fed as numpy arrays at run time (not baked as
    # initializers) so realistic MoE weight tensors do not have to be materialised as Python
    # lists at model-build time.
    is_swiglu = activation == "swiglu"
    fc1_out = 2 * F if is_swiglu else F
    inputs = [
        helper.make_tensor_value_info("input", elem_type, [M, K]),
        helper.make_tensor_value_info("router_logits", elem_type, [M, E]),
        helper.make_tensor_value_info("fc1_experts_weights", elem_type, [E, fc1_out, K]),
        helper.make_tensor_value_info("fc2_experts_weights", elem_type, [E, K, F]),
    ]
    initializers = []

    node_inputs = ["input", "router_logits", "fc1_experts_weights"]
    if with_bias:
        inputs.append(helper.make_tensor_value_info("fc1_experts_bias", elem_type, [E, fc1_out]))
        node_inputs.append("fc1_experts_bias")
    else:
        node_inputs.append("")  # optional fc1_experts_bias absent
    node_inputs.append("fc2_experts_weights")
    if with_bias:
        inputs.append(helper.make_tensor_value_info("fc2_experts_bias", elem_type, [E, K]))
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


def _activation_nodes(activation, in_name, out_name, elem_type, np_dtype,
                      activation_alpha):
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


def _swiglu_nodes(swiglu_impl, M, k, F, gate_name, linear_name, out_name, elem_type,
                  np_dtype, activation_alpha, activation_beta, swiglu_limit):
    """SwiGLU on de-interleaved gate/linear [M,k,F] -> out [M,k,F]."""
    nodes, inits = [], []

    if swiglu_limit is not None:
        limit = np.array(swiglu_limit, dtype=np_dtype)
        neg_limit = np.array(-swiglu_limit, dtype=np_dtype)
        inits += [_const("swiglu_limit_hi", limit), _const("swiglu_limit_lo", neg_limit)]
        # gate = min(gate, limit)
        nodes.append(helper.make_node("Clip", [gate_name, "", "swiglu_limit_hi"],
                                      [gate_name + "_cl"]))
        # linear = clamp(linear, -limit, limit)
        nodes.append(helper.make_node("Clip", [linear_name, "swiglu_limit_lo", "swiglu_limit_hi"],
                                      [linear_name + "_cl"]))
        gate_name, linear_name = gate_name + "_cl", linear_name + "_cl"

    if swiglu_impl == "fused-op":
        if swiglu_limit is not None:
            # com.microsoft.SwiGLU has no clamp; clamps were applied above via Clip, and it has
            # no beta. beta must be 0 for the fused op to be equivalent.
            pass
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


def build_expanded_model(M, K, F, E, k, activation, elem_type, with_bias,
                         normalize, router_impl, swiglu_impl,
                         activation_alpha, activation_beta, swiglu_limit, np_dtype):
    is_swiglu = activation == "swiglu"
    fc1_out = 2 * F if is_swiglu else F

    inputs = [
        helper.make_tensor_value_info("input", elem_type, [M, K]),
        helper.make_tensor_value_info("router_logits", elem_type, [M, E]),
        helper.make_tensor_value_info("fc1_gmm_weights", elem_type, [E, K, fc1_out]),
        helper.make_tensor_value_info("fc2_gmm_weights", elem_type, [E, F, K]),
    ]
    initializers = [
        _const("shape_Mk_F", np.array([M * k, F], dtype=np.int64)),
        _const("shape_Mk_1", np.array([M * k, 1], dtype=np.int64)),
        _const("shape_M_k_K", np.array([M, k, K], dtype=np.int64)),
        _const("combine_axis", np.array([1], dtype=np.int64)),
        _const("neg1", np.array([-1], dtype=np.int64)),
    ]
    if with_bias:
        inputs.append(helper.make_tensor_value_info("fc1_gmm_bias", elem_type, [E, fc1_out]))
        inputs.append(helper.make_tensor_value_info("fc2_gmm_bias", elem_type, [E, K]))
    nodes = []

    # --- Router: produce val [M,k] and idx [M,k] int64 ---
    router_nodes, router_inits = _router_nodes(router_impl, k, normalize)
    nodes += router_nodes
    initializers += router_inits

    # --- FC1 (up/gate projection): GroupedMatMul -> [M, k, fc1_out] ---
    fc1_gmm_inputs = ["input", "fc1_gmm_weights", "idx"]
    if with_bias:
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
        sw_nodes, sw_inits = _swiglu_nodes(swiglu_impl, M, k, F, "gate", "linear", "act",
                                           elem_type, np_dtype, activation_alpha,
                                           activation_beta, swiglu_limit)
        nodes += sw_nodes
        initializers += sw_inits
    else:
        act_nodes, act_inits = _activation_nodes(activation, "h1", "act", elem_type, np_dtype,
                                                 activation_alpha)
        nodes += act_nodes
        initializers += act_inits

    # --- FC2 (down projection): GroupedMatMul over [M*k, F] with idx2 [M*k, 1] ---
    nodes.append(helper.make_node("Reshape", ["act", "shape_Mk_F"], ["act_flat"]))
    nodes.append(helper.make_node("Reshape", ["idx", "shape_Mk_1"], ["idx2"]))
    fc2_gmm_inputs = ["act_flat", "fc2_gmm_weights", "idx2"]
    if with_bias:
        # MoE fc2 bias is (E, K); GroupedMatMul bias (E, N=K). Same.
        fc2_gmm_inputs.append("fc2_gmm_bias")
    nodes.append(helper.make_node("GroupedMatMul", fc2_gmm_inputs, ["d_flat"],
                                  domain="com.microsoft"))  # [M*k, 1, K]
    nodes.append(helper.make_node("Reshape", ["d_flat", "shape_M_k_K"], ["d"]))  # [M, k, K]

    # --- Combine: router-weighted sum over the k expert slots -> [M, K] ---
    nodes.append(helper.make_node("Unsqueeze", ["val", "neg1"], ["val_u"]))  # [M, k, 1]
    nodes.append(helper.make_node("Mul", ["d", "val_u"], ["d_weighted"]))
    nodes.append(helper.make_node("ReduceSum", ["d_weighted", "combine_axis"], ["output"],
                                  keepdims=0))  # [M, K]

    output = helper.make_tensor_value_info("output", elem_type, [M, K])
    return _make_model(nodes, inputs, [output], initializers)


# --------------------------------------------------------------------------------------------
# Timing (reused verbatim from benchmark_grouped_matmul.py)
# --------------------------------------------------------------------------------------------

def _provider_list(name):
    if name == "cpu":
        return ["CPUExecutionProvider"]
    if name == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise ValueError(name)


def time_model(model, feeds, provider, warmup, iters):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=_provider_list(provider)
    )
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


def _rel_err(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.maximum(np.abs(b).max(), 1e-6)
    return float(np.abs(a - b).max() / denom)


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------

def run_benchmark(args):
    np_dtype = _NP_DTYPE[args.dtype]
    elem_type = _ONNX_DTYPE[args.dtype]

    normalize = args.normalize
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

    print(f"onnxruntime {ort.__version__}; providers available: "
          f"{ort.get_available_providers()}")
    print(f"dtype={args.dtype} bias={args.bias} normalize={normalize} "
          f"router-impl={args.router_impl} swiglu-impl={args.swiglu_impl} "
          f"warmup={args.warmup} iters={args.iters}\n")

    header = (f"{'case':<20}{'act':<8}{'prov':<6}{'M':>6}{'K':>6}{'F':>7}{'E':>5}{'k':>3}"
              f"{'fused_ms':>11}{'expanded_ms':>13}{'speedup':>9}{'max_relerr':>12}{'ok':>4}")
    print(header)
    print("-" * len(header))

    rows = []
    for name, M, K, F, E, k, activation in args.cases:
        weights = make_shared_weights(K, F, E, activation, np_dtype, args.bias)
        base_feeds = make_feeds(M, K, E, np_dtype)
        fused_feeds = {
            **base_feeds,
            "fc1_experts_weights": weights["fc1_moe"],
            "fc2_experts_weights": weights["fc2_moe"],
        }
        exp_feeds = {
            **base_feeds,
            "fc1_gmm_weights": weights["fc1_gmm"],
            "fc2_gmm_weights": weights["fc2_gmm"],
        }
        if args.bias:
            fused_feeds["fc1_experts_bias"] = weights["fc1_bias_moe"]
            fused_feeds["fc2_experts_bias"] = weights["fc2_bias_moe"]
            exp_feeds["fc1_gmm_bias"] = weights["fc1_bias_moe"]
            exp_feeds["fc2_gmm_bias"] = weights["fc2_bias_moe"]
        fused = build_fused_moe_model(
            M, K, F, E, k, activation, elem_type, args.bias, normalize,
            args.activation_alpha, args.activation_beta, args.swiglu_limit)
        expanded = build_expanded_model(
            M, K, F, E, k, activation, elem_type, args.bias, normalize,
            args.router_impl, args.swiglu_impl, args.activation_alpha, args.activation_beta,
            args.swiglu_limit, np_dtype)

        for provider in providers:
            fused_t, fused_out = time_model(fused, fused_feeds, provider, args.warmup, args.iters)
            exp_t, exp_out = time_model(expanded, exp_feeds, provider, args.warmup, args.iters)
            fused_ms = fused_t.mean() * 1e3
            exp_ms = exp_t.mean() * 1e3
            speedup = exp_ms / fused_ms
            relerr = _rel_err(exp_out, fused_out)
            tol = args.tol if args.tol is not None else (2e-3 if args.dtype == "float32" else 5e-2)
            ok = relerr <= tol
            print(f"{name:<20}{activation:<8}{provider:<6}{M:>6}{K:>6}{F:>7}{E:>5}{k:>3}"
                  f"{fused_ms:>11.3f}{exp_ms:>13.3f}{speedup:>8.2f}x{relerr:>12.2e}"
                  f"{('yes' if ok else 'NO'):>4}")
            if not ok:
                print(f"    [WARN] {name}: max rel err {relerr:.3e} exceeds tol {tol:.1e} "
                      f"-- fused and expanded disagree.")
                if args.strict:
                    raise AssertionError(f"{name}: rel err {relerr:.3e} > tol {tol:.1e}")
            rows.append(dict(
                case=name, activation=activation, provider=provider, M=M, K=K, F=F,
                num_experts=E, k=k, fused_ms=fused_ms, expanded_ms=exp_ms, speedup=speedup,
                max_relerr=relerr, ok=ok, router_impl=args.router_impl,
                swiglu_impl=args.swiglu_impl, normalize=int(normalize)))

    if args.csv and rows:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
    p.add_argument("--bias", action="store_true", help="Include optional FC1/FC2 biases.")
    p.add_argument("--normalize", type=int, default=1, choices=[0, 1],
                   help="normalize_routing_weights: renormalize top-k weights (default 1).")
    p.add_argument("--activation-alpha", type=float, default=1.0,
                   help="SwiGLU sigmoid gate scale (default 1.0 = SiLU gate).")
    p.add_argument("--activation-beta", type=float, default=0.0,
                   help="SwiGLU linear bias (default 0.0). Not supported by the fused SwiGLU op.")
    p.add_argument("--swiglu-limit", type=float, default=None,
                   help="Optional SwiGLU clamp limit (default: no clamp).")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--tol", type=float, default=None,
                   help="Cross-check tolerance (default 2e-3 fp32 / 5e-2 fp16).")
    p.add_argument("--strict", action="store_true",
                   help="Raise if the fused and expanded outputs disagree beyond tolerance.")
    p.add_argument("--csv", default=None, help="Optional path to write results as CSV.")
    args = p.parse_args()
    args.cases = DEFAULT_CASES
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
