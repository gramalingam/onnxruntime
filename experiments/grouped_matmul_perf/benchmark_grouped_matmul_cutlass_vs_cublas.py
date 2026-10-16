#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Performance comparison: GroupedMatMul's two CUDA GEMM-execution strategies.

com.microsoft.GroupedMatMul's CUDA kernel gathers selections into group-contiguous order, runs
one dense GEMM per group, then scatters (+bias) back into the output. This script compares the
kernel's two interchangeable GEMM-execution strategies (selected via the
ORT_GROUPED_MATMUL_CUDA_IMPL env var, see grouped_matmul.cc / grouped_matmul_cutlass_gemm.h):

  - "cublas"  (default): one cublasGemmHelper() launch per non-empty group, preceded by a
    blocking device->host copy + cudaStreamSynchronize to build the host-side permutation.
  - "cutlass": a single CUTLASS grouped-GEMM kernel launch (device-side scheduling) covering all
    groups at once -- the same MoeGemmRunner machinery that backs com.microsoft.MoE's GEMM1/GEMM2.

Gather/scatter/permutation-building are identical in both paths; only the GEMM-execution step
differs. Usage:

    python benchmark_grouped_matmul_cutlass_vs_cublas.py --dtype float16 \\
        --csv results_cutlass_vs_cublas_cuda_fp16.csv
"""

import argparse
import os
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
    ("large-top2",       2048,  2,   768,  768,  32),
    ("wide-hidden",       256,  2,  1024, 1024,  8),
    ("many-experts",      512,  1,   512,  512,  64),
    ("many-experts-top2", 512,  2,   512,  512,  64),
]

_NP_DTYPE = {"float32": np.float32, "float16": np.float16}
_ONNX_DTYPE = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16}


def build_model(M, k, K, N, num_groups, elem_type, with_bias):
    inputs = [
        helper.make_tensor_value_info("input", elem_type, [M, K]),
        helper.make_tensor_value_info("weights", elem_type, [num_groups, K, N]),
        helper.make_tensor_value_info("group_indices", TensorProto.INT64, [M, k]),
    ]
    node_inputs = ["input", "weights", "group_indices"]
    if with_bias:
        inputs.append(helper.make_tensor_value_info("bias", elem_type, [num_groups, N]))
        node_inputs.append("bias")

    node = helper.make_node("GroupedMatMul", node_inputs, ["output"], domain="com.microsoft")
    output = helper.make_tensor_value_info("output", elem_type, [M, k, N])
    graph = helper.make_graph([node], "grouped_matmul", inputs, [output])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
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


def time_model(model, feeds, impl, warmup, iters):
    os.environ["ORT_GROUPED_MATMUL_CUDA_IMPL"] = impl
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        model.SerializeToString(), so, providers=["CUDAExecutionProvider"]
    )
    if "CUDAExecutionProvider" not in sess.get_providers():
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

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"CUDA not available ({ort.get_available_providers()})")

    print(f"onnxruntime {ort.__version__}; providers available: {ort.get_available_providers()}")
    print(f"dtype={args.dtype} bias={args.bias} warmup={args.warmup} iters={args.iters}\n")

    header = (f"{'case':<18}{'M':>6}{'k':>3}{'K':>6}{'N':>6}{'G':>5}"
              f"{'cublas_ms':>11}{'cutlass_ms':>12}{'speedup':>9}{'max_relerr':>12}")
    print(header)
    print("-" * len(header))

    rows = []
    for name, M, k, K, N, num_groups in args.cases:
        feeds = make_feeds(M, k, K, N, num_groups, np_dtype, args.bias)
        model = build_model(M, k, K, N, num_groups, elem_type, args.bias)

        cublas_t, cublas_out = time_model(model, feeds, "cublas", args.warmup, args.iters)
        cutlass_t, cutlass_out = time_model(model, feeds, "cutlass", args.warmup, args.iters)
        cublas_ms = cublas_t.mean() * 1e3
        cutlass_ms = cutlass_t.mean() * 1e3
        speedup = cublas_ms / cutlass_ms
        relerr = _rel_err(cutlass_out, cublas_out)

        print(f"{name:<18}{M:>6}{k:>3}{K:>6}{N:>6}{num_groups:>5}"
              f"{cublas_ms:>11.4f}{cutlass_ms:>12.4f}{speedup:>8.2f}x{relerr:>12.2e}")
        rows.append(dict(
            case=name, M=M, k=k, K=K, N=N, num_groups=num_groups,
            cublas_ms=cublas_ms, cutlass_ms=cutlass_ms, speedup=speedup, max_relerr=relerr))

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
    p.add_argument("--dtype", default="float16", choices=["float32", "float16"])
    p.add_argument("--bias", action="store_true", default=True,
                   help="Include the optional bias input (default: on).")
    p.add_argument("--no-bias", dest="bias", action="store_false")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--csv", default=None, help="Optional path to write results as CSV.")
    args = p.parse_args()
    args.cases = DEFAULT_CASES
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
