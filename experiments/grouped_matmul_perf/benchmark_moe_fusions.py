#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Microbenchmarks for the two MoE fusion candidates identified in the analysis.

Beyond the fused ``GroupedMatMul`` grouped-GEMM, a standard top-k MoE feed-forward layer
still contains two non-GEMM op clusters that are candidates for fusion. This harness measures
each cluster *in isolation* to quantify the headroom a fusion could recover, on CPU (and CUDA
if available). It uses only standard ONNX ops plus ``com.microsoft.QuickGelu`` (SiLU), so it
runs against a stock ONNX Runtime build -- no new kernels required.

Candidate A -- SwiGLU gated activation ``Mul(SiLU(g), u)``
---------------------------------------------------------
Touches the layer's largest intermediates: two ``[T, F]`` tensors (F = FFN inner dim, e.g.
14336 in Mixtral). Three variants of the same math are timed to bracket the fusion win:

* ``unfused``   : ``Sigmoid(g) -> Mul(g,.) -> Mul(.,u)``  (3 elementwise passes; what a naive
                  SiLU-then-multiply graph contains).
* ``quickgelu`` : ``QuickGelu(g, alpha=1) -> Mul(.,u)``   (2 passes; SiLU folded into one op via
                  the existing contrib op).
* ``fused_lb``  : ``Mul(g,u)``                            (1 pass; a *lower bound* for a single
                  fused SwiGLU kernel -- same memory traffic, minus the cheap sigmoid math).

The gap ``unfused - fused_lb`` is the bandwidth headroom a fused gated-activation kernel (or a
GroupedMatMul activation epilogue) targets.

Candidate B -- Router ``Softmax -> TopK (-> renormalize)``
----------------------------------------------------------
Because softmax is monotonic, ``TopK(Softmax(logits))`` picks the same experts as
``TopK(logits)``, and *renormalized* top-k of a full softmax is identical to a softmax taken
over just the top-k logits. Two numerically-equivalent strategies are timed:

* ``naive`` : ``Softmax(logits)[M,E] -> TopK(k) -> ReduceSum -> Div``  (full E-wide softmax).
* ``fused`` : ``TopK(logits, k) -> Softmax([M,k])``                    (softmax on k << E only).

``E`` (#experts) is small, so this is a launch-latency / intermediate-allocation win rather
than an arithmetic one. The harness checks the two strategies agree (max rel err).

Usage:
    python benchmark_moe_fusions.py                       # both, CPU (+CUDA if present)
    python benchmark_moe_fusions.py --bench swiglu
    python benchmark_moe_fusions.py --bench router --dtype float16 --csv router.csv
"""

import argparse
import time

import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxruntime as ort

_NP_DTYPE = {"float32": np.float32, "float16": np.float16}
_ONNX_DTYPE = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16}

# SwiGLU cluster: T = number of (token, expert-slot) rows, F = FFN inner dim.
# T = B*S*k; realistic decode/prefill token counts x Mixtral/DeepSeek/Llama-MoE inner dims.
SWIGLU_CASES = [
    # name,           T,     F
    ("small",         512,   2048),
    ("mixtral-ffn",   1024,  14336),
    ("deepseek-ffn",  2048,  1408),
    ("wide",          1024,  8192),
    ("many-tokens",   8192,  4096),
]

# Router cluster: M = tokens, E = num_experts, k = experts per token.
ROUTER_CASES = [
    # name,             M,     E,    k
    ("mixtral",         4096,  8,    2),
    ("deepseek",        4096,  64,   6),
    ("switch-many",     4096,  128,  1),
    ("large-experts",   8192,  256,  8),
    ("small-batch",     512,   32,   4),
]


def _mk_model(nodes, inputs, outputs, initializers=None, use_ms=False):
    opsets = [helper.make_opsetid("", 17)]
    if use_ms:
        opsets.append(helper.make_opsetid("com.microsoft", 1))
    graph = helper.make_graph(nodes, "g", inputs, outputs, initializers or [])
    model = helper.make_model(graph, opset_imports=opsets)
    model.ir_version = 10
    return model


# --- Candidate A: SwiGLU gated activation ---------------------------------------------------

def build_swiglu(variant, T, F, elem_type):
    g = helper.make_tensor_value_info("g", elem_type, [T, F])
    u = helper.make_tensor_value_info("u", elem_type, [T, F])
    out = helper.make_tensor_value_info("out", elem_type, [T, F])
    if variant == "unfused":
        nodes = [
            helper.make_node("Sigmoid", ["g"], ["sig"]),
            helper.make_node("Mul", ["g", "sig"], ["silu"]),
            helper.make_node("Mul", ["silu", "u"], ["out"]),
        ]
        return _mk_model(nodes, [g, u], [out])
    if variant == "quickgelu":
        nodes = [
            helper.make_node("QuickGelu", ["g"], ["silu"], domain="com.microsoft", alpha=1.0),
            helper.make_node("Mul", ["silu", "u"], ["out"]),
        ]
        return _mk_model(nodes, [g, u], [out], use_ms=True)
    if variant == "fused_lb":
        nodes = [helper.make_node("Mul", ["g", "u"], ["out"])]
        return _mk_model(nodes, [g, u], [out])
    raise ValueError(variant)


def swiglu_feeds(T, F, np_dtype, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "g": rng.standard_normal((T, F)).astype(np_dtype),
        "u": rng.standard_normal((T, F)).astype(np_dtype),
    }


# --- Candidate B: Router Softmax -> TopK ----------------------------------------------------

def build_router(variant, M, E, k, elem_type):
    logits = helper.make_tensor_value_info("logits", elem_type, [M, E])
    val = helper.make_tensor_value_info("val", elem_type, [M, k])
    idx = helper.make_tensor_value_info("idx", TensorProto.INT64, [M, k])
    k_const = helper.make_tensor("k_const", TensorProto.INT64, [1], [k])
    if variant == "naive":
        nodes = [
            helper.make_node("Softmax", ["logits"], ["probs"], axis=-1),
            helper.make_node("TopK", ["probs", "k_const"], ["tv", "idx"], axis=-1),
            helper.make_node("ReduceSum", ["tv", "axis_last"], ["denom"], keepdims=1),
            helper.make_node("Div", ["tv", "denom"], ["val"]),
        ]
        inits = [k_const, helper.make_tensor("axis_last", TensorProto.INT64, [1], [-1])]
        return _mk_model(nodes, [logits], [val, idx], inits)
    if variant == "fused":
        nodes = [
            helper.make_node("TopK", ["logits", "k_const"], ["tl", "idx"], axis=-1),
            helper.make_node("Softmax", ["tl"], ["val"], axis=-1),
        ]
        return _mk_model(nodes, [logits], [val, idx], [k_const])
    raise ValueError(variant)


def router_feeds(M, E, np_dtype, seed=0):
    rng = np.random.default_rng(seed)
    return {"logits": rng.standard_normal((M, E)).astype(np_dtype)}


# --- Timing ---------------------------------------------------------------------------------

def _provider_list(name):
    if name == "cpu":
        return ["CPUExecutionProvider"]
    if name == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise ValueError(name)


def time_model(model, feeds, provider, warmup, iters):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), so,
                                providers=_provider_list(provider))
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
    return np.array(times), result


def _rel_err(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    denom = max(np.abs(b).max(), 1e-6)
    return float(np.abs(a - b).max() / denom)


def _providers(requested):
    out = []
    for p in requested:
        if p == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            print(f"[skip] CUDA requested but not available "
                  f"({ort.get_available_providers()})")
            continue
        out.append(p)
    return out


def run_swiglu(args, providers, np_dtype, elem_type):
    print("\n=== Candidate A: SwiGLU gated activation  Mul(SiLU(g), u) ===")
    header = (f"{'case':<14}{'prov':<6}{'T':>7}{'F':>7}"
              f"{'unfused_ms':>12}{'quickgelu_ms':>14}{'fused_lb_ms':>13}"
              f"{'unfsd/lb':>10}{'qg/lb':>8}")
    print(header)
    print("-" * len(header))
    rows = []
    for name, T, F in SWIGLU_CASES:
        feeds = swiglu_feeds(T, F, np_dtype)
        models = {v: build_swiglu(v, T, F, elem_type)
                  for v in ("unfused", "quickgelu", "fused_lb")}
        for prov in providers:
            ms, outs = {}, {}
            for v, m in models.items():
                t, r = time_model(m, feeds, prov, args.warmup, args.iters)
                ms[v] = t.mean() * 1e3
                outs[v] = r[0]
            # correctness: unfused vs quickgelu (both compute true SwiGLU)
            relerr = _rel_err(outs["quickgelu"], outs["unfused"])
            print(f"{name:<14}{prov:<6}{T:>7}{F:>7}"
                  f"{ms['unfused']:>12.4f}{ms['quickgelu']:>14.4f}{ms['fused_lb']:>13.4f}"
                  f"{ms['unfused']/ms['fused_lb']:>9.2f}x{ms['quickgelu']/ms['fused_lb']:>7.2f}x")
            rows.append(dict(bench="swiglu", case=name, provider=prov, T=T, F=F,
                             unfused_ms=ms["unfused"], quickgelu_ms=ms["quickgelu"],
                             fused_lb_ms=ms["fused_lb"], swiglu_relerr=relerr))
    return rows


def run_router(args, providers, np_dtype, elem_type):
    print("\n=== Candidate B: Router  Softmax -> TopK (-> renormalize) ===")
    header = (f"{'case':<14}{'prov':<6}{'M':>6}{'E':>5}{'k':>3}"
              f"{'naive_ms':>11}{'fused_ms':>11}{'speedup':>9}{'max_relerr':>12}")
    print(header)
    print("-" * len(header))
    rows = []
    for name, M, E, k in ROUTER_CASES:
        feeds = router_feeds(M, E, np_dtype)
        m_naive = build_router("naive", M, E, k, elem_type)
        m_fused = build_router("fused", M, E, k, elem_type)
        for prov in providers:
            tn, rn = time_model(m_naive, feeds, prov, args.warmup, args.iters)
            tf, rf = time_model(m_fused, feeds, prov, args.warmup, args.iters)
            naive_ms, fused_ms = tn.mean() * 1e3, tf.mean() * 1e3
            # Compare the renormalized top-k weights (val). Indices order may tie-break
            # differently, so compare sorted-descending values per row.
            vn = np.sort(rn[0].astype(np.float64), axis=-1)[:, ::-1]
            vf = np.sort(rf[0].astype(np.float64), axis=-1)[:, ::-1]
            relerr = _rel_err(vf, vn)
            print(f"{name:<14}{prov:<6}{M:>6}{E:>5}{k:>3}"
                  f"{naive_ms:>11.4f}{fused_ms:>11.4f}{naive_ms/fused_ms:>8.2f}x{relerr:>12.2e}")
            rows.append(dict(bench="router", case=name, provider=prov, M=M, E=E, k=k,
                             naive_ms=naive_ms, fused_ms=fused_ms, speedup=naive_ms/fused_ms,
                             max_relerr=relerr))
    return rows


def run(args):
    np_dtype = _NP_DTYPE[args.dtype]
    elem_type = _ONNX_DTYPE[args.dtype]
    providers = _providers(args.providers)
    print(f"onnxruntime {ort.__version__}; providers available: "
          f"{ort.get_available_providers()}")
    print(f"dtype={args.dtype} warmup={args.warmup} iters={args.iters}")

    rows = []
    if args.bench in ("swiglu", "all"):
        rows += run_swiglu(args, providers, np_dtype, elem_type)
    if args.bench in ("router", "all"):
        rows += run_router(args, providers, np_dtype, elem_type)

    if args.csv and rows:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", default="all", choices=["all", "swiglu", "router"])
    p.add_argument("--providers", nargs="+", default=["cpu", "cuda"],
                   choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--csv", default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
