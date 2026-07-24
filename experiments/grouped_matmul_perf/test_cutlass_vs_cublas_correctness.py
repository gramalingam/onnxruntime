import os
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

os.environ.setdefault("LD_LIBRARY_PATH", "")


def build_model(M, K, N, num_groups, k, dtype, has_bias):
    input_t = helper.make_tensor_value_info("input", dtype, [M, K])
    weights_t = helper.make_tensor_value_info("weights", dtype, [num_groups, K, N])
    indices_t = helper.make_tensor_value_info("group_indices", TensorProto.INT64, [M, k])
    inputs = ["input", "weights", "group_indices"]
    value_infos = [input_t, weights_t, indices_t]
    if has_bias:
        bias_t = helper.make_tensor_value_info("bias", dtype, [num_groups, N])
        inputs.append("bias")
        value_infos.append(bias_t)
    output_t = helper.make_tensor_value_info("output", dtype, [M, k, N])

    node = helper.make_node("GroupedMatMul", inputs, ["output"], domain="com.microsoft")
    graph = helper.make_graph([node], "g", [v for v in value_infos if v.name in inputs],
                               [output_t])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.microsoft", 1),
                                                      helper.make_opsetid("", 18)])
    return model


def run(model, feeds, impl):
    os.environ["ORT_GROUPED_MATMUL_CUDA_IMPL"] = impl
    so = ort.SessionOptions()
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CUDAExecutionProvider"])
    out = sess.run(None, feeds)[0]
    return out


def test_case(M, K, N, num_groups, k, dtype_name, has_bias, seed):
    rng = np.random.default_rng(seed)
    np_dtype = {"float32": np.float32, "float16": np.float16}[dtype_name]
    onnx_dtype = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16}[dtype_name]

    input_np = rng.standard_normal((M, K)).astype(np.float32).astype(np_dtype)
    weights_np = rng.standard_normal((num_groups, K, N)).astype(np.float32).astype(np_dtype)
    group_indices_np = rng.integers(0, num_groups, size=(M, k)).astype(np.int64)
    feeds = {"input": input_np, "weights": weights_np, "group_indices": group_indices_np}
    if has_bias:
        bias_np = rng.standard_normal((num_groups, N)).astype(np.float32).astype(np_dtype)
        feeds["bias"] = bias_np

    model = build_model(M, K, N, num_groups, k, onnx_dtype, has_bias)

    out_cublas = run(model, feeds, "cublas")
    out_cutlass = run(model, feeds, "cutlass")

    abs_diff = np.abs(out_cublas.astype(np.float32) - out_cutlass.astype(np.float32))
    rel_diff = abs_diff / (np.abs(out_cublas.astype(np.float32)) + 1e-3)
    max_abs = abs_diff.max()
    max_rel = rel_diff.max()
    tol = 2e-2 if dtype_name == "float16" else 1e-4
    status = "OK" if max_rel < tol or max_abs < tol else "FAIL"
    print(f"[{status}] M={M} K={K} N={N} groups={num_groups} k={k} dtype={dtype_name} bias={has_bias} "
          f"max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
    return status == "OK"


if __name__ == "__main__":
    all_ok = True
    cases = [
        (8, 16, 32, 4, 1, "float32", False, 0),
        (8, 16, 32, 4, 1, "float32", True, 1),
        (64, 128, 256, 8, 2, "float32", True, 2),
        (64, 128, 256, 8, 2, "float16", True, 3),
        (200, 96, 64, 5, 3, "float16", False, 4),
        (17, 33, 29, 6, 1, "float32", True, 5),  # odd sizes, some groups may end up empty
        (1, 16, 16, 4, 1, "float16", True, 6),   # M=1 edge case
        (128, 512, 512, 16, 2, "float16", True, 7),
    ]
    for case in cases:
        all_ok &= test_case(*case)
    print("ALL OK" if all_ok else "SOME FAILED")
