#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Performance comparison: com.microsoft.GroupedMatMul vs its function-definition expansion.

The contrib op ``com.microsoft.GroupedMatMul`` fuses a Mixture-of-Experts (MoE) grouped
matrix multiplication into a single kernel. Its documented "reference decomposition"
(``docs/GroupedMatMul.md``) expresses the same computation with standard ONNX ops:

    idx_flat = Reshape(group_indices, [M*k])
    W_sel    = Gather(weights, idx_flat, axis=0)          # [M*k, K, N]  (duplicates weights!)
    X        = Reshape(Expand(Unsqueeze(input, 1), [M,k,K]), [M*k, 1, K])
    r        = MatMul(X, W_sel)                           # [M*k, 1, N]
    r        = r + Gather(bias, idx_flat)                 # optional
    output   = Reshape(r, [M, k, N])

This script builds both graphs for a range of realistic MoE sizes, runs them under ONNX
Runtime, and reports wall-clock latency for the fused op vs. the decomposition, on CPU and
(when available) CUDA.

The decomposition materialises one weight matrix *per selection* (``W_sel`` is
``O(M*k*K*N)``), so it is both slower and far more memory hungry than the fused op. Cases
whose ``W_sel`` tensor would exceed ``--mem-budget-gb`` are still run for the fused op but the
decomposition is skipped (reported as ``OOM-skip``).

Usage:
    python benchmark_grouped_matmul.py                 # CPU (and CUDA if available)
    python benchmark_grouped_matmul.py --providers cpu cuda
    python benchmark_grouped_matmul.py --dtype float16 --csv results.csv
"""

import argparse
import time

import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxruntime as ort

# (M tokens, k top-k, K hidden, N ffn/out, num_groups experts)
DEFAULT_CASES = [
    # name,               M,    k,   K,    N,    num_groups
    ("tiny",              256,  1,   512,  512,  8),
    ("small-dense",       512,  1,   768,  768,  8),
    ("small-top2",        512,  2,   512,  512,  16),
    ("medium-dense",     1024,  1,   768,  768,  16),
    ("medium-top2",       512,  2,   768,  768,  32),
    ("large-tokens",     2048,  1,   512,  512,  32),
    ("wide-hidden",       256,  2,  1024, 1024,  8),
    ("many-experts",      512,  1,   512,  512,  64),
]

_NP_DTYPE = {"float32": np.float32, "float16": np.float16}
_ONNX_DTYPE = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16}


def _const(name, np_array):
    return helper.make_tensor(
        name=name,
        data_type=onnx.helper.np_dtype_to_tensor_dtype(np_array.dtype),
        dims=np_array.shape,
        vals=np_array.flatten().tolist(),
    )


def build_fused_model(M, k, K, N, num_groups, elem_type, with_bias):
    """Single com.microsoft.GroupedMatMul node."""
    inputs = [
        helper.make_tensor_value_info("input", elem_type, [M, K]),
        helper.make_tensor_value_info("weights", elem_type, [num_groups, K, N]),
        helper.make_tensor_value_info("group_indices", TensorProto.INT64, [M, k]),
    ]
    node_inputs = ["input", "weights", "group_indices"]
    if with_bias:
        inputs.append(helper.make_tensor_value_info("bias", elem_type, [num_groups, N]))
        node_inputs.append("bias")

    node = helper.make_node(
        "GroupedMatMul", node_inputs, ["output"], domain="com.microsoft"
    )
    output = helper.make_tensor_value_info("output", elem_type, [M, k, N])
    graph = helper.make_graph([node], "fused_grouped_matmul", inputs, [output])
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    return model


def build_decomposition_model(M, k, K, N, num_groups, elem_type, with_bias):
    """Reference decomposition from docs/GroupedMatMul.md using standard ONNX ops."""
    inputs = [
        helper.make_tensor_value_info("input", elem_type, [M, K]),
        helper.make_tensor_value_info("weights", elem_type, [num_groups, K, N]),
        helper.make_tensor_value_info("group_indices", TensorProto.INT64, [M, k]),
    ]
    if with_bias:
        inputs.append(helper.make_tensor_value_info("bias", elem_type, [num_groups, N]))

    initializers = [
        _const("shape_Mk", np.array([M * k], dtype=np.int64)),
        _const("shape_MkK", np.array([M, k, K], dtype=np.int64)),
        _const("shape_Mk1K", np.array([M * k, 1, K], dtype=np.int64)),
        _const("shape_MkN", np.array([M, k, N], dtype=np.int64)),
        _const("axis1", np.array([1], dtype=np.int64)),
    ]

    nodes = [
        # idx_flat = Reshape(group_indices, [M*k])
        helper.make_node("Reshape", ["group_indices", "shape_Mk"], ["idx_flat"]),
        # W_sel = Gather(weights, idx_flat, axis=0)  -> [M*k, K, N]
        helper.make_node("Gather", ["weights", "idx_flat"], ["W_sel"], axis=0),
        # X = Reshape(Expand(Unsqueeze(input, 1), [M,k,K]), [M*k, 1, K])
        helper.make_node("Unsqueeze", ["input", "axis1"], ["input_u"]),
        helper.make_node("Expand", ["input_u", "shape_MkK"], ["input_e"]),
        helper.make_node("Reshape", ["input_e", "shape_Mk1K"], ["X"]),
        # r = MatMul(X, W_sel)  -> [M*k, 1, N]
        helper.make_node("MatMul", ["X", "W_sel"], ["r_raw"]),
    ]

    if with_bias:
        nodes += [
            # bias_sel = Unsqueeze(Gather(bias, idx_flat, axis=0), 1) -> [M*k, 1, N]
            helper.make_node("Gather", ["bias", "idx_flat"], ["bias_sel"], axis=0),
            helper.make_node("Unsqueeze", ["bias_sel", "axis1"], ["bias_sel_u"]),
            helper.make_node("Add", ["r_raw", "bias_sel_u"], ["r_biased"]),
            helper.make_node("Reshape", ["r_biased", "shape_MkN"], ["output"]),
        ]
    else:
        nodes.append(helper.make_node("Reshape", ["r_raw", "shape_MkN"], ["output"]))

    output = helper.make_tensor_value_info("output", elem_type, [M, k, N])
    graph = helper.make_graph(
        nodes, "decomposed_grouped_matmul", inputs, [output], initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    return model


def make_feeds(M, k, K, N, num_groups, np_dtype, with_bias, seed=0):
    rng = np.random.default_rng(seed)
    feeds = {
        "input": rng.standard_normal((M, K)).astype(np_dtype),
        "weights": rng.standard_normal((num_groups, K, N)).astype(np_dtype),
        "group_indices": rng.integers(0, num_groups, size=(M, k)).astype(np.int64),
    }
    if with_bias:
        feeds["bias"] = rng.standard_normal((num_groups, N)).astype(np_dtype)
    return feeds


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


def run_benchmark(args):
    np_dtype = _NP_DTYPE[args.dtype]
    elem_type = _ONNX_DTYPE[args.dtype]
    budget_bytes = args.mem_budget_gb * (1024**3)
    itemsize = np.dtype(np_dtype).itemsize

    providers = []
    for p in args.providers:
        if p == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            print(f"[skip] CUDA requested but not available in ORT build "
                  f"({ort.get_available_providers()})")
            continue
        providers.append(p)

    print(f"onnxruntime {ort.__version__}; providers available: "
          f"{ort.get_available_providers()}")
    print(f"dtype={args.dtype} bias={args.bias} warmup={args.warmup} iters={args.iters} "
          f"mem-budget={args.mem_budget_gb} GiB\n")

    header = (f"{'case':<14}{'provider':<7}{'M':>5}{'k':>3}{'K':>6}{'N':>6}{'G':>5}"
              f"{'fused_ms':>11}{'decomp_ms':>12}{'speedup':>9}{'max_relerr':>12}")
    print(header)
    print("-" * len(header))

    rows = []
    for name, M, k, K, N, num_groups in args.cases:
        feeds = make_feeds(M, k, K, N, num_groups, np_dtype, args.bias)
        fused = build_fused_model(M, k, K, N, num_groups, elem_type, args.bias)
        wsel_bytes = M * k * K * N * itemsize
        decomp_fits = wsel_bytes <= budget_bytes
        decomp = (build_decomposition_model(M, k, K, N, num_groups, elem_type, args.bias)
                  if decomp_fits else None)

        for provider in providers:
            fused_t, fused_out = time_model(fused, feeds, provider, args.warmup, args.iters)
            fused_ms = fused_t.mean() * 1e3

            if decomp is None:
                decomp_ms = float("nan")
                speedup = float("nan")
                relerr = float("nan")
                decomp_str = f"OOM-skip(>{args.mem_budget_gb}G)"
                speedup_str = "-"
                relerr_str = "-"
            else:
                decomp_t, decomp_out = time_model(
                    decomp, feeds, provider, args.warmup, args.iters)
                decomp_ms = decomp_t.mean() * 1e3
                speedup = decomp_ms / fused_ms
                relerr = _rel_err(fused_out, decomp_out)
                decomp_str = f"{decomp_ms:.3f}"
                speedup_str = f"{speedup:.2f}x"
                relerr_str = f"{relerr:.2e}"

            print(f"{name:<14}{provider:<7}{M:>5}{k:>3}{K:>6}{N:>6}{num_groups:>5}"
                  f"{fused_ms:>11.3f}{decomp_str:>12}{speedup_str:>9}{relerr_str:>12}")
            rows.append(dict(
                case=name, provider=provider, M=M, k=k, K=K, N=N, num_groups=num_groups,
                fused_ms=fused_ms, decomp_ms=decomp_ms, speedup=speedup, max_relerr=relerr,
                wsel_gib=wsel_bytes / (1024**3)))

    if args.csv:
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
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--bias", action="store_true", help="Include the optional bias input.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--mem-budget-gb", type=float, default=3.0,
                   help="Skip the decomposition when its W_sel tensor exceeds this size.")
    p.add_argument("--csv", default=None, help="Optional path to write results as CSV.")
    args = p.parse_args()
    args.cases = DEFAULT_CASES
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
