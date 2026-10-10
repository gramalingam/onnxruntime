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

## Mutable opaque objects at the model boundary

Opaque values can also represent state that persists across model runs. For example,
a caller may hold a PRNG or key-value cache object, pass its current state into a
model, and receive its next state as an output.

The graph remains functional:

```text
Generate(state: PRNG) -> (value: scalar tensor(float), next_state: PRNG)
```

The question at the API boundary is whether `state` and `next_state` may be backed
by the same mutable object.

### Existing tensor precedent: key-value caches

ORT already uses this pattern for tensor-backed state such as key-value caches.
For example, the
[`TensorScatter` kernel](../onnxruntime/core/providers/cpu/llm/tensorscatter.cc)
declares:

```cpp
def.MayInplace(0, 0);
```

The declaration says that `present_cache` may reuse `past_cache`. The kernel requests
its output normally, compares the input and output data addresses, copies the old
cache only when the addresses differ, and then applies the update to the output.
Its
[`InPlace_IOBinding` test](../onnxruntime/test/providers/cpu/llm/tensorscatter_op_test.cc)
binds the same tensor `OrtValue` as both `past_cache` and `present_cache`, runs the
model, and verifies that the caller's buffer was updated in place.

CUDA and WebGPU `GroupQueryAttention` kernels use the same basic convention for
key and value caches:

```cpp
def.MayInplace(3, 1);  // past_key -> present_key
def.MayInplace(4, 2);  // past_value -> present_value
```

The [CUDA implementation](../onnxruntime/contrib_ops/cuda/bert/group_query_attention.cc)
detects the shared-buffer path by comparing the past and present data pointers. Its
[I/O-binding test](../onnxruntime/test/contrib_ops/group_query_attention_op_test.cc)
binds each past cache tensor as its corresponding present output. Attention
operators may also have attributes such as `past_present_share_buffer` that define
fixed-capacity cache shapes or select specialized kernels. Those details are
operator-specific; they are not the fundamental input/output alias mechanism.

There are also operator-specific side-effect conventions. For example, an operator
with omitted cache outputs can directly update an input cache. That is useful for
specialized stateful operators, but it is not a good general contract for opaque
values. Explicit functional input and output values preserve the model's dataflow
and provide a natural distinct-output fallback.

### How caller-provided aliasing works today

When a caller supplies inputs and preallocated outputs, the execution frame places
those `OrtValue`s in their graph value slots before any kernel executes. Copying an
`OrtValue` shares ownership of its underlying tensor or object. If the caller binds
the same value as an input and output, the two slots therefore refer to the same
underlying storage.

When the producing kernel requests that output, the execution frame returns the
already allocated value rather than allocating another one. For tensors, it also
checks that the supplied output shape matches the shape requested by the kernel.
The kernel can then recognize the alias through pointer equality and take its
in-place path.

This caller-provided boundary alias is related to, but distinct from, reuse created
by the allocation planner:

- **Caller-provided alias:** Binding the same preallocated value makes the graph
  input and graph output share storage for that run.
- **Planner-selected alias:** `MayInplace` permits ORT to reuse storage for
  graph-internal values when liveness, observability, execution-order, type, and
  location checks allow it.

The planner deliberately protects graph inputs from planner-selected reuse because
they are caller-owned. That protection does not prevent a caller from explicitly
supplying the same value as an output. Existing cache kernels rely on the caller
using this facility only with an operator that supports the alias. ORT does not yet
provide a general validation rule that connects a caller-provided graph-boundary
alias to the kernel's `MayInplace` declaration.

### Common case: bind the same object as input and output

The corresponding low-friction interface for an opaque type is implemented for a
direct linear state-update path. The caller supplies the same concrete object as
both the input and the preallocated output:

```text
state_value = OrtValue(PRNG object)

BindInput("state", state_value)
BindOutput("next_state", state_value)
Run()
```

The exact API spelling may vary by language binding. The important signal is object
identity: the input and caller-supplied output `OrtValue`s refer to the same concrete
object. They do not need to be the same `OrtValue` wrapper because copying an
`OrtValue` shares ownership of its underlying object.

For a supported binding, this should mean:

- The caller permits ORT to overwrite the object's initial state during the run.
- On successful completion, that same object contains the logical value of
  `next_state`.
- Other model results have their ordinary functional meaning. Aliasing the boundary
  values does not change the graph's semantics.
- The caller must not read, modify, or use the object in another run until execution
  completes, including through another alias or a shared mutable resource owned by
  the object.
- A validation failure before execution leaves the object unchanged. A failure after
  execution starts may leave it partially advanced; no rollback is promised.
- Supplying distinct input and output objects retains the ordinary behavior in which
  the input remains unchanged.

This provides reference-parameter behavior without a separate annotation for the
common linear-state case. It is a permission to mutate the caller-owned object,
not proof that the graph is linear or that mutation is automatically safe.

For a model in which one state-update kernel directly connects the graph input to
the graph output, the execution frame returns the caller-supplied opaque object from
`Output<T>()`. The kernel uses the same address comparison and distinct-output
fallback as a tensor cache kernel. No new opaque allocation mechanism is required.

The contract describes the final logical output and object identity; it need not
forbid every internal copy. In a more complex graph, ORT and the kernels may use
temporary distinct objects as long as all graph values remain correct and the
designated output is ultimately produced in the caller-supplied object.

There is an important practical limitation: ORT has no generic opaque copy-back
operation. If the result is computed in another object, the runtime cannot transfer
arbitrary opaque state into the supplied object without help from the datatype or
kernel. The final producing kernel must be able to write the correct result into that
object through a safe plan. Otherwise, ORT should reject the aliasing configuration
before executing kernels rather than silently return a different object.

#### Why the existing mechanism needs a general contract

ORT already accepts caller-supplied inputs and preallocated outputs, including
non-tensor `OrtValue`s, and direct opaque input/output aliasing follows naturally
from the same execution-frame behavior used by tensor caches. However, the current
facility is permissive: preallocating an output with the same underlying value does
not by itself prove that the producing kernel supports aliasing or that a longer
input-to-output path can safely preserve the requested identity.

The boundary validation implements the following contract for opaque state and is
designed to align with the existing tensor convention:

1. A kernel declares `MayInplace(input_index, output_index)`.
2. The kernel produces correct results with either aliased or distinct values.
3. A caller may bind the same underlying `OrtValue` to the corresponding graph
   input and output to request boundary mutation.
4. For opaque objects, ORT validates that the requested boundary alias is a direct
   linear relationship covered by `MayInplace` before execution and rejects
   unsupported or conflicting bindings.
5. Type-specific validation remains where necessary: tensor element type, shape,
   and location versus opaque domain, name, and location.

Under this contract, `MayInplace` has two related uses: it permits planner-selected
reuse for internal values and declares that caller-provided aliasing is supported by
the kernel. It remains permission rather than a requirement; a kernel still needs a
correct distinct-output path.

The implemented direct one-kernel boundary case matches existing cache usage. A
longer linear path remains a separate optimization problem. For example:

```text
input_state -> Generate -> state1 -> Generate -> output_state
```

The planner protects `input_state`, so the first transition may create and copy into
`state1`. The final kernel may be able to copy the result into the caller-provided
`output_state`, but generic ORT cannot perform that opaque copy-back. Achieving
zero-copy aliasing through the whole chain requires per-run alias-aware allocation
behavior or a prepared alternative plan. This optimization should not be required
to support the common direct state-update case.

The target overhead for the direct path is boundary identity and compatibility
validation, not per-run whole-graph analysis or additional caller annotations.

### Extensions for more general usage

Object identity covers the most common in/out case but cannot express every useful
permission. Future APIs or annotations could support broader contracts without
complicating the simple path.

#### Consumable input with a separately returned object

A caller might permit ORT to consume or mutate an input while accepting that the
returned state is a different object:

```text
ConsumeInput("state")
Run() -> next_state
```

This can enable reuse without requiring a same-object output. It is a weaker identity
contract but a broader ownership permission. It should remain distinct from the
same-object in/out contract so callers know whether to retain or replace their state
handle after each run.

#### Explicit input/output relationships

Annotations could name an input/output pair when object identity is unavailable at
planning time, when ORT allocates the output, or when a language binding cannot
conveniently express shared object identity. Such an annotation grants permission;
it does not override liveness, type, location, execution-order, or observability
requirements.

#### Copy or assignment support

More complex graphs could benefit from an optional type-specific way to copy or
assign opaque state. That would let ORT compute into a temporary object and copy the
final value into a caller-supplied output. This would be a new datatype capability,
not generic behavior inferred by the runtime. Types for which copying is unsupported
or expensive could omit it and accept a more restricted set of in/out plans.

#### Branches, control flow, and devices

Forked graphs may still be valid. Kernels can preserve earlier logical states in
temporary objects and mutate the caller's object only at its last safe use. However,
guaranteeing that a particular output receives the original object may require
scheduling constraints or additional copies. An initial implementation should
support simple sequential CPU paths and reject unsupported branching, control-flow,
parallel, or cross-device cases rather than weaken graph semantics.

These extensions should be additive. They should not impose annotations, generic
copy callbacks, or per-run planning costs on the common same-object linear path.

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

### Object representation and type-specific factories

[`OrtValue`](../include/onnxruntime/core/framework/ort_value.h) stores a
`std::shared_ptr<void>` and an `MLDataType`, which is a `const DataTypeImpl*`.
For the PRNG, the shared pointer points directly to the concrete `TestPrng` object.
There is no extra heap-allocated PRNG handle and no tensor holding a pointer.
The shared pointer has an ownership control block; the datatype pointer refers to
shared type metadata rather than metadata allocated for each value.

The type designer supplies the concrete class. In-tree datatype registration maps
the model's opaque domain and name to its runtime type descriptor. The existing
[datatype templates](../include/onnxruntime/core/framework/data_types.h) connect
that descriptor to the class: `OpaqueType<T, ...>` inherits `NonTensorType<T>`,
whose `GetCreateFunc()` returns a factory performing `new T()` and whose
`GetDeleteFunc()` returns a deleter performing `delete static_cast<T*>(p)`.

The generic framework invokes these functions; it neither knows the concrete class
nor infers which constructor to call. The factory belongs to the type descriptor,
not an existing object, so it can create the first instance. Using the in-tree
implementation rather than a plugin does not remove these existing factories.

This template factory requires a default-constructible class. Its result is a
fully constructed object in the class-defined initial state, not uninitialized
storage. Classes requiring constructor arguments or custom allocation need a
suitable datatype implementation or a richer creation interface; this change
does not add such an interface.

### Output allocation and lifetime

The kernel requests its output by calling
[`OpKernelContext::Output<T>()`](../include/onnxruntime/core/framework/op_kernel_context.h).
That request reaches the
[execution frame](../onnxruntime/core/framework/execution_frame.cc), which consults
the allocation plan if the output has not already been allocated:

- For a safe opaque `AllocKind::kReuse` plan, the frame ensures the reused value
  is allocated and assigns its `OrtValue` to the output. Normally the producer has
  already created the object, so this shares the existing object and control block
  without allocating another one.
- Otherwise, the frame uses `AllocateTraditionalMLValue` to invoke the datatype's
  factory and initialize the output `OrtValue` with the resulting object and deleter.

Thus the framework supplies an output object **on the kernel's request**, rather
than constructing a new object unconditionally at kernel entry. Sharing an
`OrtValue` shares ownership, not just an address: releasing one value does not
destroy an object still owned by another. The registered deleter invokes the
object's destructor once the last owner releases it.

The class remains responsible for its private resources and internal buffer
allocation. ORT does not automatically pool those buffers like tensor storage.
A complex class can use a cheap default-constructed shell and allocate resources
when needed, but this does not imply automatic resource sharing or cloning.

The [optimizer execution frame](../onnxruntime/core/optimizer/optimizer_execution_frame.cc)
continues to allocate distinct objects using the same factory/deleter facilities;
it needs no opaque-specific allocation change.

### Runtime responsibilities versus kernel semantics

The runtime manages allocation, lifetime, and permission to alias. The kernel
implements the operation's semantics. The `Seed` kernel requests an output object
and then calls `Seed(...)` to initialize its logical state from the input seed.
The `Generate` kernel compares input and output addresses and handles both planner
decisions. The following code corresponds to the test kernel, with counters omitted:

```cpp
const auto* input = ctx->Input<TestPrng>(0);
auto* output = ctx->Output<TestPrng>(1);
if (input != output) {
  output->CopyFrom(*input);
}
*ctx->Output(0, TensorShape{})->MutableData<float>() = output->Next();
```

On the distinct-output path, this is **default construction followed by explicit
`CopyFrom`**, not copy construction or C++ copy assignment. `TestPrng` disables
copy and move operations; `CopyFrom` explicitly copies its private state.
On the alias path, the kernel advances the existing object without copying.

There is no generic runtime cloning step. When output and input are distinct,
the kernel must leave the input's logical state unchanged. Copy-then-update is
one strategy; another kernel could compute the output state directly without
copying first. Any copying must preserve logical independence: merely copying a
pointer to shared mutable internals would not provide a correct fallback.

### Wrapping and allocation costs

Fresh opaque allocation involves a concrete object allocation and shared-pointer
control-block storage. Aliasing avoids new object/control-block allocation in
the normal case, but still involves `OrtValue` assignment and reference-count
bookkeeping.

Tensor values already use the same outer `OrtValue` wrapper. Their shared pointer
points to a [`Tensor` header](../onnxruntime/core/framework/tensor.cc) describing
shape, element type, and a data buffer, rather than directly to a PRNG object.
Ordinary tensor buffer reuse can still create a new tensor header and control
block; opaque reuse here shares the entire object.

These are qualitative costs, not a claim of measured negligible overhead.
For a tiny PRNG, allocations and ownership bookkeeping may be significant relative
to the state update. For a larger abstraction, internal allocation or copying may
dominate instead. The appropriate tradeoff depends on the class and workload.

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
