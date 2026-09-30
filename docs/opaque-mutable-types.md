# Opaque mutable types in ONNX Runtime

An ONNX model can use an object whose natural implementation is a stateful C++ class
without exposing that class's private representation. The graph treats the object
as a functional value; ONNX Runtime can reuse and mutate its physical storage when
doing so cannot change an observable result.

## The problem: using a C++ object from ONNX

Consider a deterministic pseudo-random number generator (PRNG) with this illustrative
C++ interface:

```cpp
class Prng {
 public:
  explicit Prng(uint64_t seed = 0);

  void Seed(uint64_t seed);
  float Generate();
};
```

Calling `Generate()` returns one value and advances private state. An implementation
might use one `uint64_t` and SplitMix64, but callers should depend on the interface,
not the representation. A different implementation could require more complex state.

Encoding that private state as an ordinary tensor would expose an implementation
detail as part of the model contract. The same concern applies to larger stateful
abstractions, such as a key-value cache. Instead, the graph should know that it has a
PRNG value without needing to know its layout.

## Encoding the class in ONNX

ONNX opaque types identify an abstract type by domain and name:

```text
opaque(example.prng,PRNG)
```

The domain and name do not specify the object's size, memory layout, allocation,
or destruction. The runtime implementation supplies those details.

To preserve functional graph semantics, a mutating method is represented as an
operation that returns the updated state as a new logical value. Conceptually,
the PRNG has two operations:

```text
Create(seed: scalar tensor(int64)) -> PRNG
Generate(state: PRNG) -> (value: scalar tensor(float), next_state: PRNG)
```

Here, `PRNG` abbreviates the opaque type above. Both tensors are rank-0 tensors.
Successive calls explicitly thread the state through the graph:

```text
prng0 = Create(seed)
(first,  prng1) = Generate(prng0)
(second, prng2) = Generate(prng1)
```

`prng0`, `prng1`, and `prng2` are distinct logical values, even if the backend can
implement some transitions by advancing the same C++ object.

## The challenge: forked use

A functional value may be consumed more than once:

```text
prng0 = Create(seed)
(left_first,  left_prng1)  = Generate(prng0)
(right_first, right_prng1) = Generate(prng0)
(left_second,  left_prng2)  = Generate(left_prng1)
(right_second, right_prng2) = Generate(right_prng1)
```

Both initial calls must observe the same original `prng0`. Consequently,
`left_first` and `right_first` must match. The continued branches must also produce
matching second values, just as two independent generators initialized with the
same seed would.

Unconditionally advancing one shared object for the first call would change the
input seen by the other call. That would violate the graph's semantics.

An implementation therefore needs a policy for uses where mutation is unsafe:

1. Reject unsupported usage if the object cannot be cloned or otherwise evaluated
   without modifying an observable input.
2. Clone the input state when necessary, then advance the clone.

The PRNG example uses the second approach. ORT does **not** automatically clone
arbitrary opaque objects: the kernel implements the copy fallback. A type that
cannot support that fallback needs an explicit unsupported-usage policy; a reuse
annotation alone does not impose a linear-use restriction on the model.

Forking does not mean every edge must copy. After the other consumers have finished,
a last-use consumer may safely reuse the original object. Conversely, a linear chain
does not guarantee reuse if the input remains observable or other planner constraints
prevent it.

## How ONNX Runtime handles it

### In-tree type and kernel registration

The [executable PRNG example](../onnxruntime/test/framework/opaque_reuse_test.cc)
uses ORT's existing `ORT_REGISTER_OPAQUE_TYPE` and `DataTypeImpl::RegisterDataType`
facilities. Its datatype instances have process lifetime, and its schemas and CPU
kernels are registered through the existing test registry infrastructure.

The concrete example uses `opaque(ort.test.prng,State)` and the operators `Seed`
and `Generate` in the `ort.test.prng` domain, version 1. `Seed` corresponds to
`Create` in the conceptual examples above; the C++ class is `TestPrng`, and its
state-advancing method is `Next()`. These are **test-only operators**, not public
production PRNG operators.

### Permission to reuse, not a requirement

The `Generate` kernel's `KernelDefBuilder` declares:

```cpp
def.MayInplace(0, 1);
```

Input 0 is the state. Output 0 is the generated float, and output 1 is the next state.
This declaration permits output 1 to reuse input 0; it does not require aliasing.

The [allocation planner](../onnxruntime/core/framework/allocation_planner.cc)
recognizes compatible opaque input/output types by matching their domain and name
and requiring matching planned locations. This is a compatibility check, not
authorization to mutate an object. The existing `MayInplace` eligibility, liveness,
execution-order, and observability checks still apply.

In particular, caller-owned graph inputs and states exposed as graph outputs must
remain observable without being changed by a later node. Graph outputs themselves
are allocated separately rather than using this intermediate-value reuse path.
Parallel execution remains conservative: the test's parallel plan does not reuse
the opaque state. Matching types alone never enable indiscriminate reuse of
non-tensor values.

### Allocation, ownership, and kernel behavior

For a safe opaque `AllocKind::kReuse` plan, the
[execution frame](../onnxruntime/core/framework/execution_frame.cc) ensures that
the reused value is allocated and assigns its `OrtValue` to the output. This shares
ownership of the same object, not merely its address. Releasing one value does not
destroy an object still owned by another; the deleter runs once when the last owner
releases it.

When reuse is not selected, the existing non-tensor factory creates a distinct
output object. The kernel handles both cases by comparing object addresses. The
following pseudocode corresponds to the test kernel, with counters omitted:

```cpp
const auto* input = ctx->Input<TestPrng>(0);
auto* output = ctx->Output<TestPrng>(1);
if (input != output) {
  output->CopyFrom(*input);
}
*ctx->Output(0, TensorShape{})->MutableData<float>() = output->Next();
```

Copying must preserve the input's logical state independently: merely copying a
pointer to shared mutable internals would not provide a correct fallback.

The existing [opaque datatype facilities](../include/onnxruntime/core/framework/data_types.h)
provide creation and deletion. The
[optimizer execution frame](../onnxruntime/core/optimizer/optimizer_execution_frame.cc)
continues to allocate distinct objects using those same facilities; it needs no
opaque-specific allocation change.

## Tests as an executable example

`OpaqueReuseTest` in
[`opaque_reuse_test.cc`](../onnxruntime/test/framework/opaque_reuse_test.cc)
covers both functional correctness and the optimization:

- A linear chain asserts the reuse plan, checks actual input/output identity through
  reuse counters, and verifies allocation, copy, and destruction counts. An
  always-copy implementation fails this test even if its generated numbers are correct.
- A fork continues both returned states and checks both branches against the
  deterministic sequence, while allowing safe last-use reuse on one fork edge.
- Graph input and output tests verify that still-observable states are not mutated.
- Repeated runs and retained `OrtValue` ownership exercise object lifetime and
  exactly-once destruction.
- Negative cases cover a kernel without `MayInplace`, parallel execution, and
  different opaque domains or names.
- An optimizer-frame test verifies distinct allocation and shared ownership lifetime.

The central distinction is between **logical immutability** and **physical mutation**.
The graph always describes functional state transitions. Reusing a C++ object is
only an implementation optimization when its previous state can no longer be observed.

## Scope

This support is implemented in-tree in ONNX Runtime and uses its existing allocation
and deletion mechanisms. It does not add plugin APIs, dynamic type registration or
unregistration APIs, or plugin unload behavior. No separate external ORT patch is
needed for the implementation described here. Support for another opaque abstraction
still requires its own datatype, schemas, kernels, and a correct policy for cases
where reuse is unavailable.
