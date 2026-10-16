import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

print("ORT version:", ort.__version__)
print("Providers:", ort.get_available_providers())

def make_model(M, K, N, num_groups, k, dtype_np, dtype_onnx, use_bias):
    inputs = ["input", "weights", "group_indices", "combine_weights"]
    if use_bias:
        inputs.append("bias")
    node = helper.make_node(
        "GroupedMatMulReduceSum", inputs, ["output"], domain="com.microsoft"
    )
    graph_inputs = [
        helper.make_tensor_value_info("input", dtype_onnx, [M, k, K]),
        helper.make_tensor_value_info("weights", dtype_onnx, [num_groups, K, N]),
        helper.make_tensor_value_info("group_indices", TensorProto.INT64, [M, k]),
        helper.make_tensor_value_info("combine_weights", dtype_onnx, [M, k]),
    ]
    if use_bias:
        graph_inputs.append(helper.make_tensor_value_info("bias", dtype_onnx, [num_groups, N]))
    graph_outputs = [helper.make_tensor_value_info("output", dtype_onnx, [M, N])]
    graph = helper.make_graph([node], "g", graph_inputs, graph_outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 10
    return model

def ref(input, weights, group_indices, combine_weights, bias):
    M, k, K = input.shape
    num_groups, K2, N = weights.shape
    out = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(k):
            g = group_indices[i, j]
            v = input[i, j].astype(np.float32) @ weights[g].astype(np.float32)
            if bias is not None:
                v = v + bias[g].astype(np.float32)
            out[i] += combine_weights[i, j].astype(np.float32) * v
    return out

def run_case(M, K, N, num_groups, k, dtype_np, dtype_onnx, use_bias, provider, seed=0):
    rng = np.random.default_rng(seed)
    input = rng.standard_normal((M, k, K)).astype(np.float32).astype(dtype_np)
    weights = rng.standard_normal((num_groups, K, N)).astype(np.float32).astype(dtype_np)
    group_indices = np.array([rng.choice(num_groups, size=k, replace=False) for _ in range(M)], dtype=np.int64)
    combine_weights = rng.standard_normal((M, k)).astype(np.float32).astype(dtype_np)
    bias = rng.standard_normal((num_groups, N)).astype(np.float32).astype(dtype_np) if use_bias else None

    model = make_model(M, K, N, num_groups, k, dtype_np, dtype_onnx, use_bias)
    so = ort.SessionOptions()
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=[provider])
    feeds = {
        "input": input, "weights": weights, "group_indices": group_indices,
        "combine_weights": combine_weights,
    }
    if use_bias:
        feeds["bias"] = bias
    out = sess.run(None, feeds)[0]
    expected = ref(input, weights, group_indices, combine_weights, bias)
    # CUDA's float32 GEMMs use TF32 tensor cores by default (~1e-3 relative precision), same as
    # GroupedMatMul's own CUDA kernel -- so float32-on-CUDA needs a looser tolerance than the
    # essentially-exact CPU float32 path.
    if dtype_np == np.float32 and provider == "CUDAExecutionProvider":
        atol, rtol = 2e-2, 2e-2
    elif dtype_np == np.float32:
        atol, rtol = 1e-3, 1e-4
    else:
        atol, rtol = 3e-2, 3e-2
    ok = np.allclose(out.astype(np.float32), expected, atol=atol, rtol=rtol)
    maxdiff = np.max(np.abs(out.astype(np.float32) - expected))
    print(f"  M={M} K={K} N={N} G={num_groups} k={k} dtype={dtype_np.__name__} bias={use_bias} provider={provider}: "
          f"{'OK' if ok else 'FAIL'} maxdiff={maxdiff:.5f}")
    return ok

cases = [
    (1, 4, 4, 1, 1, np.float32, TensorProto.FLOAT),   # trivial
    (2, 8, 4, 3, 1, np.float32, TensorProto.FLOAT),   # k=1 dense
    (5, 16, 8, 4, 2, np.float32, TensorProto.FLOAT),  # multi-select
    (5, 16, 8, 4, 2, np.float16, TensorProto.FLOAT16),
    (17, 32, 24, 6, 3, np.float32, TensorProto.FLOAT),
    (17, 32, 24, 6, 3, np.float16, TensorProto.FLOAT16),
]

all_ok = True
for provider in ["CPUExecutionProvider", "CUDAExecutionProvider"]:
    print(f"--- provider {provider} ---")
    for (M, K, N, G, k, dnp, donnx) in cases:
        for use_bias in [False, True]:
            try:
                ok = run_case(M, K, N, G, k, dnp, donnx, use_bias, provider)
                all_ok = all_ok and ok
            except Exception as e:
                print(f"  EXCEPTION: {e}")
                all_ok = False

print("ALL OK" if all_ok else "SOME FAILED")
