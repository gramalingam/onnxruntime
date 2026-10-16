# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate GroupedMatMul MoE proofs of concept using Mobius routing.

Requires mobius-onnx from onnxruntime/mobius revision
b1e9c8e9dec1dfa206183f957e77d07cfb734df2.
The grouped expert body replaces Mobius' unrolled dense expert evaluation,
following https://github.com/onnx/onnx/pull/8193. No checkpoints are downloaded.
GroupedMatMul is a local ONNX function, so no future ONNX schema is required.
"""

from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
from mobius.components._moe import SoftmaxTopKGate, TopKGate
from onnx import helper
from onnx_ir.passes import common
from onnxscript import GraphBuilder

DOMAIN = "local.moe"


def grouped_matmul_function():
    # Broadcast each token row over the selected expert weight matrices.
    return helper.make_function(
        DOMAIN,
        "GroupedMatMul",
        ["input", "weights", "indices"],
        ["output"],
        [
            helper.make_node("Constant", [], ["axes"], value_ints=[1, 2]),
            helper.make_node("Constant", [], ["squeeze_axes"], value_ints=[2]),
            helper.make_node("Gather", ["weights", "indices"], ["selected"], axis=0),
            helper.make_node("Unsqueeze", ["input", "axes"], ["tokens"]),
            helper.make_node("MatMul", ["tokens", "selected"], ["result"]),
            helper.make_node("Squeeze", ["result", "squeeze_axes"], ["output"]),
        ],
        [helper.make_opsetid("", 24)],
    )


def generate(name: str, routing: str, experts: int = 3, top_k: int = 2) -> None:
    hidden, intermediate, tokens = 8, 12, 6
    gate = (
        TopKGate(hidden, experts, top_k)
        if routing == "topk"
        else SoftmaxTopKGate(hidden, experts, top_k, norm_topk_prob=routing == "normalized")
    )
    rng = np.random.default_rng(2026)
    gate.load_state_dict({"weight": ir.tensor(rng.uniform(-0.4, 0.4, (experts, hidden)).astype(np.float32))})
    x = ir.Value(name="hidden", shape=ir.Shape([tokens, hidden]), type=ir.TensorType(ir.DataType.FLOAT))
    graph = ir.Graph([x], [], nodes=[], name=name, opset_imports={"": 24, DOMAIN: 1})
    builder = GraphBuilder(graph)
    op = builder.op
    routing_weights, indices = gate(op, x)

    def parameter(name, shape):
        value = ir.Value(
            name=name,
            shape=ir.Shape(shape),
            type=ir.TensorType(ir.DataType.FLOAT),
            const_value=ir.tensor(rng.uniform(-0.4, 0.4, shape).astype(np.float32)),
        )
        graph.initializers[name] = value
        return value

    w_gate = parameter("expert_gate", (experts, hidden, intermediate))
    w_up = parameter("expert_up", (experts, hidden, intermediate))
    w_down = parameter("expert_down", (experts, intermediate, hidden))
    g = op.GroupedMatMul(x, w_gate, indices, _domain=DOMAIN)
    u = op.GroupedMatMul(x, w_up, indices, _domain=DOMAIN)
    product = op.Mul(op.Swish(g), u)
    flat = op.Reshape(product, [-1, intermediate])
    flat_indices = op.Reshape(indices, [-1, 1])
    projected = op.GroupedMatMul(flat, w_down, flat_indices, _domain=DOMAIN)
    regrouped = op.Reshape(projected, [-1, top_k, hidden])
    y = op.ReduceSum(op.Mul(regrouped, op.Unsqueeze(routing_weights, [-1])), [1], keepdims=0)
    y.name = "output"
    graph.outputs.append(y)
    model = ir.Model(graph, ir_version=10, producer_name="mobius-onnx")
    common.LiftConstantsToInitializersPass(lift_all_constants=True)(model)
    proto = ir.to_proto(model)
    proto.functions.append(grouped_matmul_function())
    assert sum(node.op_type == "GroupedMatMul" for node in proto.graph.node) == 3
    assert all(node.op_type not in ("MoE", "QMoE", "Loop") for node in proto.graph.node)
    proto = onnx.shape_inference.infer_shapes(proto, strict_mode=True)
    onnx.checker.check_model(proto, full_check=True)
    onnx.save(proto, Path(__file__).with_name(f"{name}.onnx"))


if __name__ == "__main__":
    generate("mobius_moe_topk", "topk")
    generate("mobius_moe_softmax_topk_normalized", "normalized")
    generate("mobius_moe_softmax_topk", "unnormalized", experts=5, top_k=1)
