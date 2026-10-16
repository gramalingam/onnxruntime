# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Portable int4 decompositions for the opt-in C++ quantized rewriter examples.

local.quant.Unpack specializes onnx/onnx#8576 to bits=4; local.moe.GroupedMatMul
implements the indexed expert multiplication proposed in onnx/onnx#8193.
Both have executable standard-ONNX function bodies; no future schema is needed.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSETS = [
    helper.make_opsetid("", 21),
    helper.make_opsetid("local.quant", 1),
    helper.make_opsetid("local.moe", 1),
    helper.make_opsetid("com.microsoft", 1),
]


def unpack_function():
    nodes = []

    def constant(name, value):
        nodes.append(helper.make_node("Constant", [], [name], value=numpy_helper.from_array(np.asarray(value))))

    constant("mask", np.uint8(15))
    constant("shift", np.uint8(4))
    constant("axes", np.array([-1], np.int64))
    constant("start", np.array([0], np.int64))
    constant("last", np.array([-1], np.int64))
    constant("two", np.array([2], np.int64))
    nodes.extend(
        [
            helper.make_node("BitwiseAnd", ["x", "mask"], ["low"]),
            helper.make_node("BitShift", ["x", "shift"], ["high"], direction="RIGHT"),
            helper.make_node("Unsqueeze", ["low", "axes"], ["lo"]),
            helper.make_node("Unsqueeze", ["high", "axes"], ["hi"]),
            helper.make_node("Concat", ["lo", "hi"], ["pairs"], axis=-1),
            helper.make_node("Shape", ["x"], ["shape"]),
            helper.make_node("Slice", ["shape", "start", "last"], ["prefix"]),
            helper.make_node("Gather", ["shape", "last"], ["bytes"], axis=0),
            helper.make_node("Mul", ["bytes", "two"], ["padded_count"]),
            helper.make_node("Concat", ["prefix", "padded_count"], ["padded_shape"], axis=0),
            helper.make_node("Reshape", ["pairs", "padded_shape"], ["padded"]),
            helper.make_node("Unsqueeze", ["count", "axes"], ["end"]),
            helper.make_node("Slice", ["padded", "start", "end", "axes"], ["y"]),
        ]
    )
    return helper.make_function("local.quant", "Unpack", ["x", "count"], ["y"], nodes, [OPSETS[0]], attributes=["bits"])


def grouped_function():
    return helper.make_function(
        "local.moe",
        "GroupedMatMul",
        ["x", "w", "indices"],
        ["y"],
        [
            helper.make_node("Constant", [], ["axes"], value_ints=[1, 2]),
            helper.make_node("Constant", [], ["squeeze_axes"], value_ints=[2]),
            helper.make_node("Gather", ["w", "indices"], ["selected"], axis=0),
            helper.make_node("Unsqueeze", ["x", "axes"], ["tokens"]),
            helper.make_node("MatMul", ["tokens", "selected"], ["result"]),
            helper.make_node("Squeeze", ["result", "squeeze_axes"], ["y"]),
        ],
        [OPSETS[0]],
    )


class Builder:
    def __init__(self):
        self.nodes = []
        self.initializers = []
        self.inputs = []
        self.rng = np.random.default_rng(2026)

    def const(self, name, value):
        self.initializers.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def ints(self, name, value):
        return self.const(name, np.asarray(value, np.int64))

    def op(self, op, inputs, output, **attrs):
        self.nodes.append(helper.make_node(op, inputs, [output], **attrs))
        return output

    def input(self, name, shape):
        self.inputs.append(helper.make_tensor_value_info(name, T.FLOAT, shape))
        return name

    def weights(self, prefix, shape):
        return self.const(prefix, self.rng.integers(0, 256, shape, dtype=np.uint8))

    def scales(self, prefix, shape):
        return self.const(prefix, self.rng.uniform(0.01, 0.06, shape).astype(np.float32))

    def dequantize(self, p, packed, scale, count, rows, zp=None):
        q = self.op("Unpack", [packed, count], p + "codes", domain="local.quant", bits=4)
        f = self.op("Cast", [q], p + "float", to=T.FLOAT)
        if zp is None:
            zp = self.const(p + "offset", np.float32(8))
        else:
            zp = self.op("Unsqueeze", [zp, self.ints(p + "zaxis", [-1])], p + "zp")
        centered = self.op("Sub", [f, zp], p + "centered")
        scales = self.op("Unsqueeze", [scale, self.ints(p + "axis", [-1])], p + "scales")
        blocks = self.op("Mul", [centered, scales], p + "blocks")
        row_weights = self.op("Reshape", [blocks, self.ints(p + "rows", rows)], p + "r", allowzero=0)
        return self.op("Transpose", [row_weights], p + "dequantized", perm=[1, 0] if len(rows) == 2 else [0, 2, 1])

    def save(self, name, output_shape, moe=False, routing="topk"):
        graph = helper.make_graph(
            self.nodes,
            name,
            self.inputs,
            [helper.make_tensor_value_info("output", T.FLOAT, output_shape)],
            self.initializers,
        )
        functions = [unpack_function()] + ([grouped_function()] if moe else [])
        model = helper.make_model(graph, functions=functions, opset_imports=OPSETS, ir_version=10)
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        onnx.checker.check_model(model, full_check=True)
        validate(model, moe, routing)
        onnx.save(model, Path(__file__).with_name(name + ".onnx"))


def matmul(name, zero_point="symmetric", k=48, block=16, rank=2):
    b = Builder()
    n, blocks = 5, k // block
    shape = [k] if rank == 1 else [2, k] if rank == 2 else [2, 3, k]
    x = b.input("input", shape)
    w = b.weights("packed", (n, blocks, block // 2))
    s = b.scales("scales", (n, blocks))
    zp = None
    if zero_point == "float":
        zp = b.const("zero_points", b.rng.uniform(0, 15, (n, blocks)).astype(np.float32))
    elif zero_point == "packed":
        z = b.weights("zero_points", (n, (blocks + 1) // 2))
        z = b.op("Unpack", [z, b.ints("zero_count", blocks)], "zero_codes", domain="local.quant", bits=4)
        zp = b.op("Cast", [z], "zero_float", to=T.FLOAT)
    weights = b.dequantize("w", w, s, b.ints("count", block), [n, k], zp)
    b.op("MatMul", [x, weights], "output")
    b.save(name, [*shape[:-1], n])


def moe(name, routing="topk", hidden=32, intermediate=48, block=16, experts=3, top_k=2):
    b = Builder()
    tokens = 4
    x = b.input("input", [tokens, hidden])
    logits = b.input("logits", [tokens, experts])
    count = b.ints("count", block)
    weights = {}
    for p, n, k in [("g", intermediate, hidden), ("u", intermediate, hidden), ("d", hidden, intermediate)]:
        w = b.weights(p + "packed", (experts, n, k // 2))
        s = b.scales(p + "scale", (experts, n, k // block))
        packed = b.op(
            "Reshape",
            [w, b.ints(p + "blocked", [experts, n, k // block, block // 2])],
            p + "b",
            allowzero=0,
        )
        weights[p] = b.dequantize(p, packed, s, count, [experts, n, k])
    router = logits if routing == "topk" else b.op("Softmax", [logits], "probs", axis=-1)
    b.nodes.append(
        helper.make_node("TopK", [router, b.ints("top_k", [top_k])], ["selected", "indices"], axis=-1, largest=1)
    )
    if routing == "topk":
        mixing = b.op("Softmax", ["selected"], "mixing", axis=-1)
    elif routing == "normalized":
        total = b.op("ReduceSum", ["selected", b.ints("sum_axes", [-1])], "total", keepdims=1, noop_with_empty_axes=0)
        mixing = b.op("Div", ["selected", total], "mixing")
    else:
        mixing = "selected"
    g = b.op("GroupedMatMul", [x, weights["g"], "indices"], "gate", domain="local.moe")
    u = b.op("GroupedMatMul", [x, weights["u"], "indices"], "up", domain="local.moe")
    sigmoid = b.op("Sigmoid", [g], "sigmoid")
    activated = b.op("Mul", [g, sigmoid], "activated")
    product = b.op("Mul", [activated, u], "product")
    flat = b.op("Reshape", [product, b.ints("flat_shape", [-1, intermediate])], "flat", allowzero=0)
    indices = b.op("Reshape", ["indices", b.ints("index_shape", [-1, 1])], "flat_indices", allowzero=0)
    projected = b.op("GroupedMatMul", [flat, weights["d"], indices], "projected", domain="local.moe")
    regrouped = b.op("Reshape", [projected, b.ints("output_shape", [-1, top_k, hidden])], "regrouped", allowzero=0)
    expanded = b.op("Unsqueeze", [mixing, b.ints("mix_axes", [-1])], "expanded")
    weighted = b.op("Mul", [regrouped, expanded], "weighted")
    b.op("ReduceSum", [weighted, b.ints("reduce_axes", [1])], "output", keepdims=0, noop_with_empty_axes=0)
    b.save(name, [tokens, hidden], moe=True, routing=routing)


def validate(model, moe, routing):
    """Check the executable function decompositions against independent NumPy math."""
    parameters = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}

    def unpack(packed, count):
        result = np.stack([packed & 15, packed >> 4], axis=-1)
        return result.reshape(*packed.shape[:-1], -1)[..., :count].astype(np.float32)

    def softmax(x):
        exp = np.exp(x - x.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    evaluator = ReferenceEvaluator(model)
    rng = np.random.default_rng(2026)
    for tied in (False, True):
        feeds = {
            value.name: rng.uniform(-0.7, 0.7, [d.dim_value for d in value.type.tensor_type.shape.dim]).astype(
                np.float32
            )
            for value in model.graph.input
        }
        x = feeds["input"]
        if not moe:
            packed, scales = parameters["packed"], parameters["scales"]
            blocks = unpack(packed, int(parameters["count"]))
            zp = parameters.get("zero_points", np.float32(8))
            if zp.ndim:
                if zp.dtype == np.uint8:
                    zp = unpack(zp, scales.shape[-1])
                zp = zp[..., None]
            rows = ((blocks - zp) * scales[..., None]).reshape(packed.shape[0], -1)
            expected = x @ rows.T
        else:
            logits = feeds["logits"]
            if tied:
                logits.fill(0)
            count = int(parameters["count"])
            weights = {}
            for p in ("g", "u", "d"):
                packed, scales = parameters[p + "packed"], parameters[p + "scale"]
                blocks = unpack(packed, packed.shape[-1] * 2).reshape(*scales.shape, count)
                rows = ((blocks - 8) * scales[..., None]).reshape(*packed.shape[:-1], -1)
                weights[p] = rows.transpose(0, 2, 1)
            router = logits if routing == "topk" else softmax(logits)
            top_k = int(parameters["top_k"][0])
            indices = np.argsort(-router, axis=-1, kind="stable")[:, :top_k]
            selected = np.take_along_axis(router, indices, axis=-1)
            mixing = softmax(selected) if routing == "topk" else selected
            if routing == "normalized":
                mixing = mixing / mixing.sum(axis=-1, keepdims=True)
            gate = np.matmul(x[:, None, None, :], weights["g"][indices]).squeeze(2)
            up = np.matmul(x[:, None, None, :], weights["u"][indices]).squeeze(2)
            product = gate / (1 + np.exp(-gate)) * up
            projected = np.matmul(product[:, :, None, :], weights["d"][indices]).squeeze(2)
            expected = (projected * mixing[..., None]).sum(axis=1)
        actual = evaluator.run(None, feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    matmul("unpack_matmul_symmetric")
    matmul("unpack_matmul_float_zp", "float", k=64, block=32, rank=3)
    matmul("unpack_matmul_packed_zp", "packed", rank=1)
    moe("unpack_qmoe_topk")
    moe("unpack_qmoe_normalized", "normalized", hidden=64, intermediate=32, block=32)
    moe("unpack_qmoe_unnormalized", "unnormalized", experts=5, top_k=1)
