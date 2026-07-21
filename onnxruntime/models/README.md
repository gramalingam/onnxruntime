# Toy C++ models built on ONNX Runtime kernels

This folder explores a specific idea: **writing an ONNX Runtime execution plan
for a model directly in C++**, compiling it together with (a build/package of)
ONNX Runtime, and running it — while reusing ORT's own operator kernels for the
core numerical work (MatMul, Relu, ...).

The first exercise here is a toy feed-forward-network (FFN) layer:

```
Y = Relu(X @ W1 + b1) @ W2 + b2
```

It is deliberately small; the point is to explore **the calling interface**
between a hand-written C++ "model" and the ORT op kernels, and **how such a
model can be structured**.

## Files

| File | Purpose |
|---|---|
| `feed_forward_model.h` / `.cc` | The toy model. `FeedForwardModel` is the 2-phase C++ model; `FeedForwardOp` hosts it as a custom operator. |
| `run_feed_forward.cc` | Driver: builds a session, runs the model, verifies against a plain-C++ reference. |
| `generate_model.py` | Generates `feed_forward.onnx` (a single `FeedForward` node + weight initializers). |
| `feed_forward.onnx` | The generated model (checked in for convenience). |
| `CMakeLists.txt` | Builds the driver against a prebuilt ONNX Runtime package. |

## The 2-phase structure

The model separates *building the plan* from *running it*:

* **Phase 1 – construction.** `FeedForwardModel`'s constructor receives the
  weights and pre-creates the ORT op kernels it will use (two `MatMul`s, two
  `Add`s, one `Relu`). This is analogous to "compiling" an execution plan.
* **Phase 2 – execution.** `FeedForwardModel::Run()` receives the runtime input
  `X` and produces the output `Y` by invoking the pre-created kernels in order.

## The calling interface (what this exercise found)

ORT exposes a **standalone operator interface** for invoking a single kernel
outside of a full graph run:

* `Ort::Op::Create(info, op_type, domain, opset, type_constraints..., attrs...,
  input_count, output_count)` — locates and instantiates one kernel.
  (C API: `OrtApi::CreateOp`.)
* `op.Invoke(context, inputs, n_in, outputs, n_out)` — runs it on `OrtValue`s.
  (C API: `OrtApi::InvokeOp`.)

Two ORT-provided handles are required to use this interface:

1. **`OrtKernelInfo`** — needed by `CreateOp` to find the kernel in an execution
   provider's kernel registry. Available when ORT constructs a kernel.
2. **`OrtKernelContext`** — needed by `InvokeOp` for the allocator, thread pool,
   compute stream, etc. Available when ORT runs a kernel's `Compute`.

Because those handles are produced by ORT only when it instantiates and runs an
operator, the toy model is **hosted inside a custom operator** (`FeedForwardOp`):

* ORT calls `CreateKernel(api, info)` → we build `FeedForwardModel(info)`
  (**phase 1**, has `OrtKernelInfo`).
* ORT calls `Compute(context)` → we call `FeedForwardModel::Run(context)`
  (**phase 2**, has `OrtKernelContext`).

### Where the weights come from

The weights are the **constant inputs** of the hosting node (initializers
`W1, b1, W2, b2`). Phase 1 reads them via
`Ort::ConstKernelInfo::GetTensorConstantInput(index, &is_constant)`
(C API: `KernelInfoGetConstantInput_tensor`) and copies them into the model, so
by the time `Run()` is called only `X` still needs to be supplied. This is what
makes the constructor "accept the weights" and `Run()` "accept the inputs".

### Values and intermediate buffers

Inputs/outputs of the hosting node are obtained from
`Ort::KernelContext` (`GetInput`, `GetOutput`). Intermediate activations are
allocated as plain CPU buffers wrapped in `Ort::Value::CreateTensor<float>` and
passed between the standalone kernels. (A more sophisticated version would use
the kernel context's temp-space allocator so it works on non-CPU providers.)

## Build and run

You need a prebuilt ONNX Runtime package (headers + libs). See
[`../../samples/cxx/README.md`](../../samples/cxx/README.md) for how to obtain
one. Then:

```bash
# 1. (optional) regenerate the model
pip install onnx numpy
python generate_model.py

# 2. configure + build
cmake -S . -B build \
    -DORT_HEADER_DIR:PATH=/path/to/onnxruntime/include \
    -DORT_LIBRARY_DIR:PATH=/path/to/onnxruntime/lib
cmake --build build --config Release

# 3. run (model + ORT libs are copied next to the executable)
cd build && ./feed_forward_model
```

Expected output ends with `Result: PASS`.

## Limitations / next steps

* CPU-only: intermediate buffers use CPU memory. Using the kernel context's
  temp allocator would generalize this to other execution providers.
* The model is hosted as a custom op purely to obtain `OrtKernelInfo` /
  `OrtKernelContext`. A follow-up exercise could explore whether these handles
  can be produced more directly for a fully standalone C++ program.
* Opset versions passed to `Ort::Op::Create` must match those ORT registers for
  each op (here: `MatMul`=13, `Add`/`Relu`=14).
