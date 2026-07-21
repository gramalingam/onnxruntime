# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate the ONNX model that hosts the toy FeedForward custom operator.

The graph contains a single node ``FeedForward`` in the ``com.example`` domain.
The op is implemented in C++ (see ``feed_forward_model.h`` / ``.cc``) and drives
ORT's own MatMul / Add / Relu kernels to compute:

    Y = Relu(X @ W1 + b1) @ W2 + b2

Node inputs (positional):
    X  : runtime input,   float [M, K]   (M is dynamic)
    W1 : constant weight, float [K, H]
    b1 : constant bias,   float [H]
    W2 : constant weight, float [H, N]
    b2 : constant bias,   float [N]

Output:
    Y  : float [M, N]

The weights are generated from the SAME deterministic formulas used by the C++
reference implementation in ``run_feed_forward.cc`` so the two can be compared.

Usage:
    pip install onnx numpy
    python generate_model.py
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save_model
from onnx.checker import check_model

K, H, N = 4, 8, 3


def make_w1() -> np.ndarray:
    return np.array(
        [[((k * H + j) % 7 - 3) * 0.1 for j in range(H)] for k in range(K)],
        dtype=np.float32,
    )


def make_b1() -> np.ndarray:
    return np.array([0.01 * j for j in range(H)], dtype=np.float32)


def make_w2() -> np.ndarray:
    return np.array(
        [[((h * N + n) % 5 - 2) * 0.1 for n in range(N)] for h in range(H)],
        dtype=np.float32,
    )


def make_b2() -> np.ndarray:
    return np.array([0.02 * n for n in range(N)], dtype=np.float32)


def main() -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["M", K])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["M", N])

    initializers = [
        numpy_helper.from_array(make_w1(), name="W1"),
        numpy_helper.from_array(make_b1(), name="b1"),
        numpy_helper.from_array(make_w2(), name="W2"),
        numpy_helper.from_array(make_b2(), name="b2"),
    ]

    node = helper.make_node(
        "FeedForward",
        inputs=["X", "W1", "b1", "W2", "b2"],
        outputs=["Y"],
        domain="com.example",
    )

    graph = helper.make_graph([node], "feed_forward_graph", [x], [y], initializers)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 14),
            helper.make_opsetid("com.example", 1),
        ],
    )
    # The custom op has no registered ONNX schema, so skip the schema check.
    check_model(model, check_custom_domain=False)
    save_model(model, "feed_forward.onnx")
    print("Saved feed_forward.onnx")


if __name__ == "__main__":
    main()
