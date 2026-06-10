# CNN Gradient Native Compilation Scalability

## Summary

The generated gradient for the crater CNN can now be lowered through the
descriptor-export path to valid, non-empty MLIR. However, compiling that MLIR
into a native shared library with `CPUFunctionExecutor` does not finish within
a practical amount of time.

This is not currently a correctness failure. The compiler does not report an
invalid HIR node, an unbound variable, malformed MLIR, or an LLVM diagnostic.
The observed failure is a compile-time scalability problem: the generated MLIR
module is extremely large, and the generic MLIR CPU pipeline spends several
minutes processing it without completing.

The same numerical problem is handled much more efficiently by TensorFlow and
PyTorch. Those systems do not normally materialize this computation as tens of
megabytes of duplicated, fully expanded scalar/tensor IR before optimization.
They retain graph structure, reuse common subexpressions and saved forward
values, and lower high-level convolution and reduction operations through
specialized kernels and compiler pipelines.

## Reproduction

The reproducer is the CNN in `examples/crater_train.py`:

```python
from examples.crater_train import _CNN_FULL_LISP_SRC, _parameter_types
from remora.ad_source import generate_gradient_function_source
from remora.runtime import CPUFunctionExecutor

param_types = _parameter_types()
gradient = generate_gradient_function_source(
    _CNN_FULL_LISP_SRC,
    "cnn-loss",
    param_types,
    differentiate_input=0,
    include_prelude=False,
    syntax="lisp",
)

artifact = CPUFunctionExecutor.compile_source(
    gradient.source,
    gradient.function_name,
    param_types,
    include_prelude=False,
    syntax="lisp",
)
```

The descriptor-lowering acceptance condition succeeds:

```python
from remora.compiler import compile_function_source

result = compile_function_source(
    gradient.source,
    gradient.function_name,
    param_types,
    include_prelude=False,
    syntax="lisp",
    verify=False,
)
assert result.mlir_text != ""
```

## Measurements

Measurements from the current implementation and MLIR 18 toolchain:

| Phase | Observed time | Output size/result |
|---|---:|---:|
| Generate gradient source | 0.01 seconds | 29,531 characters |
| Typecheck and descriptor MLIR lowering | 107.92 seconds | 47,552,499 characters |
| `mlir-opt-18` CPU pipeline | More than 252 seconds | Did not finish |
| End-to-end `CPUFunctionExecutor.compile_source` | More than 600 seconds | Timed out |

The 360-second phase-timed run spent approximately 108 seconds producing the
descriptor MLIR and then exhausted the remaining approximately 252 seconds in
`run_cpu_pipeline_text`. A separate end-to-end run was terminated after 600
seconds before it produced a shared library.

The CPU pipeline begins with:

1. `linalg-fuse-elementwise-ops`
2. `one-shot-bufferize`
3. buffer hoisting and deallocation
4. conversion of `linalg` and `scf` to loops/control flow
5. conversion to the LLVM dialect

The phase timing identifies the external `mlir-opt-18` invocation as the
dominant unfinished phase. More detailed per-pass timing has not yet been
collected, so it is not proven which individual pass is worst. Elementwise
fusion and one-shot bufferization are the leading suspects because both perform
whole-module analysis over a very large tensor program.

## Why the IR Becomes So Large

Several independent expansion mechanisms compound.

### Source AD duplicates the forward computation

The current source-to-source AD implementation emits a standalone Remora
gradient expression. Expressions needed by multiple derivative terms are
repeated in the generated source instead of being named once and reused.

For this CNN, repeated terms include convolution output, ReLU masks, linear
layer output, the final logit, and BCE subexpressions. A source file of only
about 29 KB therefore describes a DAG as a deeply duplicated expression tree.
Lowering the tree recursively emits each occurrence again.

### The gradient is compiled separately for each differentiated input

Training requires gradients for `k`, `b1`, `w2`, `b2`, `w3`, and `b3`. The
current workflow generates and compiles one function per differentiated input.
Each function contains much of the same forward computation and backward
logic. Even if one gradient eventually compiles, compiling all six repeats a
large amount of work and prevents cross-gradient sharing.

### Forward values are recomputed instead of saved

Reverse-mode AD is most efficient when the forward pass produces a tape or
explicit saved values that the backward pass consumes. The current generated
source often reconstructs forward intermediates wherever a derivative rule
needs them.

This trades runtime storage for both compile-time and runtime recomputation.
For a CNN, that trade is especially poor because convolution and dense-layer
intermediates are expensive and referenced by several gradient paths.

### High-level operations are lowered too early

`im2col`, `col2im`, folds, maps, transposes, broadcasts, and selects are lowered
into explicit tensor operations. In particular, statically shaped image and
patch operations can produce many individual extraction and insertion
operations.

Once the program has been expanded this far, generic fusion and bufferization
passes must rediscover structure that was explicit and compact in the HIR.

### The lowering does not preserve sharing

HIR is currently consumed as a tree. Structurally identical subexpressions do
not automatically become shared SSA definitions. The closed-scalar-fold
hoisting added for descriptor export removes one class of repeated scalar
computations, but it does not address repeated array-valued CNN subgraphs.

### Textual MLIR adds overhead

The descriptor lowerer constructs a very large MLIR string. The runtime then
passes that string to an external `mlir-opt` process, which must parse it,
allocate an in-memory operation graph, run global analyses, print another large
string, and later translate it to LLVM IR.

Text is not the root cause, but 47.5 MB of textual MLIR materially increases
parse time, memory traffic, subprocess I/O, and peak memory use.

## Comparison with TensorFlow and PyTorch

TensorFlow and PyTorch solve the same forward and backward CNN computation much
more efficiently because their execution and compilation models preserve
important structure.

Typical advantages include:

- Reverse-mode AD records or represents a graph with shared nodes rather than
  repeatedly substituting the entire forward expression into every derivative.
- Backward functions consume saved tensors from the forward pass.
- A single backward invocation computes all requested parameter gradients.
- Convolution, matrix multiplication, activation, and reduction remain
  high-level operations until they can be mapped to optimized kernels.
- Common-subexpression elimination and graph partitioning happen before
  low-level loop expansion.
- Mature CPU and GPU backends use specialized libraries and tuned kernels
  instead of synthesizing every convolution from scalar extraction/insertion
  operations.
- Compilation caches are keyed by graph, shape, dtype, and device, allowing
  reuse across training steps.

PyTorch eager mode also avoids ahead-of-time compilation entirely for the
ordinary training path: autograd schedules existing operator kernels directly.
`torch.compile` can capture and optimize larger graphs, but it still starts from
an operator graph with explicit sharing. TensorFlow similarly represents
gradient computations using graph operations and delegates expensive kernels to
optimized implementations.

Remora does not need to reproduce either framework wholesale, but it should
adopt the same essential principle: preserve graph sharing and high-level array
operations until the compiler has enough information to optimize them.

## Recommended Direction

The strongest fix is architectural: stop treating generated gradient source as
the optimization IR for nontrivial programs.

### 1. Generate one value-and-grad function

Generate a single function that returns the primal loss and all requested
parameter gradients. The forward computation should run once. The backward
sweep should reuse saved forward values and produce all six gradients together.

For the CNN, the desired conceptual signature is:

```text
value_and_grad_cnn(k, b1, w2, b2, w3, b3, mask, x, y)
  -> (loss, dk, db1, dw2, db2, dw3, db3)
```

This removes the six-way duplication caused by per-input gradient compilation
and creates opportunities for shared scheduling and buffer planning.

### 2. Represent AD output as a DAG or SSA IR

The AD transform should emit named bindings or SSA values for intermediates.
At minimum, source generation should introduce `let` bindings and memoize
structurally identical subexpressions. Preferably, AD should operate directly on
typed HIR or a dedicated graph/SSA representation and bypass source reparse and
tree reconstruction.

Required properties:

- Every primal intermediate has one definition.
- Derivative rules reference that definition instead of copying its expression.
- Array-valued common subexpressions are shared, not only scalar folds.
- Lexical scope and shape/type information remain explicit.

### 3. Add explicit tape and saved-value analysis

Introduce a forward/backward split:

```text
forward(inputs) -> (loss, tape)
backward(tape, dloss) -> gradients
```

Then perform saved-value analysis to decide which intermediates should be
stored and which are cheap enough to recompute. This decision should be made
deliberately rather than emerging from repeated source substitution.

An initial conservative implementation can save all array-valued intermediates.
Later work can use liveness, cost, and memory-size estimates to trade storage
against recomputation.

### 4. Keep convolution and linear algebra high-level

Do not expand CNN primitives to elementwise tensor extraction/insertion before
AD and graph simplification.

Candidate operations include:

- convolution and convolution gradients
- matrix-vector and matrix-matrix multiplication
- reductions
- broadcast
- ReLU and select
- reshape and transpose views

Add VJP rules for these operations at the high-level IR. Lower them later to
named library calls, structured `linalg` operations, or tiled loop nests. For
CPU execution, calling an optimized BLAS or convolution implementation may be a
better near-term path than relying on generic fusion to recover efficient code.

### 5. Run simplification before low-level MLIR generation

Add an AD/HIR optimization stage before descriptor lowering:

1. dead-code elimination
2. constant folding
3. algebraic simplification, especially multiplication/addition by zero or one
4. common-subexpression elimination
5. broadcast and reshape canonicalization
6. map fusion
7. map-reduce recognition
8. cancellation of transpose/reshape pairs where legal

This stage should operate on compact typed IR. Waiting until the program is a
47.5 MB MLIR module makes every optimization more expensive.

### 6. Add array-valued hoisting to descriptor lowering

As an incremental fix, descriptor lowering can memoize repeated pure HIR
subexpressions and emit one SSA value for each unique expression. This should
include array-valued folds, maps, views, and `im2col` results.

This is less robust than fixing AD generation because structural equality can
be expensive and scope-sensitive, but it may substantially reduce the current
module without redesigning the AD pipeline.

Memoization must account for:

- lexical bindings and shadowing
- function parameters and captured scalar values
- result type and shape
- purity
- prefix-independent SSA naming

### 7. Avoid expanding static copies into thousands of operations

Replace unrolled `im2col`/`col2im` generation with structured loops or a compact
`linalg.generic`/affine representation. The current static unrolling produces a
large operation count before optimization begins.

For example, emit nested `scf.for` or affine loops that compute patch and pixel
indices. This keeps IR size proportional to loop nest depth rather than image
size. A dedicated convolution operation would be better still.

### 8. Split and measure the MLIR pipeline

Add per-pass timing and peak-memory measurement. Run each CPU pass separately
on a persisted reproducer to determine whether the dominant cost is fusion,
bufferization, loop conversion, or LLVM conversion.

Useful diagnostics include:

```text
mlir-opt --mlir-timing --mlir-timing-display=tree ...
/usr/bin/time -v mlir-opt ...
```

Also compare pipelines with the initial `linalg-fuse-elementwise-ops` pass
disabled. If fusion is superlinear on this graph, an HIR-level fusion/CSE pass
followed by a simpler MLIR pipeline may be faster and more predictable.

### 9. Prefer in-process IR construction for large modules

Once IR size is under control, build MLIR operations through the Python/C++ API
and run passes in process where practical. This avoids repeated parsing and
printing of tens of megabytes of text. It will not solve expression duplication
by itself, so this should follow graph-size reductions rather than replace them.

### 10. Cache compilation artifacts

Cache generated gradient HIR, lowered MLIR, LLVM IR, and shared libraries by:

- function source/hash
- differentiated input set
- concrete parameter shapes and element types
- CPU target features
- compiler and pipeline version

Caching does not reduce first-compilation latency, but it prevents the same
large compile from being repeated for every training process or test run.

## Suggested Implementation Order

### Immediate diagnostics

1. Persist the 47.5 MB MLIR reproducer outside normal tests.
2. Collect per-pass wall time and peak RSS from `mlir-opt-18`.
3. Count HIR nodes, MLIR operations, repeated subexpressions, and occurrences of
   major forward intermediates.
4. Record compile-time budgets in a benchmark rather than a correctness test.

### Near-term reductions

1. Emit loop-based `im2col` and `col2im` instead of statically unrolled tensor
   extraction/insertion operations.
2. Add typed-HIR common-subexpression elimination and `let` introduction.
3. Add array-valued memoization in descriptor lowering as a temporary guard.
4. Simplify generated gradients before MLIR lowering.
5. Test the CPU pipeline without generic elementwise fusion.

### Medium-term AD redesign

1. Generate one multi-result value-and-grad program.
2. Add an explicit tape or saved-value representation.
3. Move AD from source trees to typed HIR/SSA.
4. Add high-level VJPs for convolution, linear algebra, reduction, broadcast,
   and view operations.

### Long-term backend work

1. Lower convolution and matrix operations to optimized CPU/GPU kernels.
2. Add cost-based fusion and recomputation decisions.
3. Add memory planning and buffer reuse across forward and backward passes.
4. Cache shape-specialized native artifacts.

## Proposed Acceptance Targets

Use explicit size and latency targets so functional success does not hide
compile-time regressions.

Initial targets for the crater CNN gradient:

- Generated optimized HIR should contain shared forward intermediates.
- Descriptor MLIR should be less than 5 MB of text.
- Descriptor lowering should complete in less than 10 seconds on the reference
  development machine.
- The CPU MLIR pipeline should complete in less than 60 seconds.
- End-to-end native compilation should complete in less than 90 seconds.
- One compiled function should produce all six trainable gradients.
- Compiled gradients should match the interpreter and finite differences within
  the existing numerical tolerance.
- Repeated training steps should reuse the same compiled artifact.

These are provisional engineering budgets, not claims that Remora should yet
match TensorFlow or PyTorch compilation and execution performance. They are
intended to force the IR-size problem to be solved before further backend
optimization.

## Current Status

The descriptor export correctness gap is resolved: scalar descriptor
parameters can be captured through tensor, map, fold, view, and scalar lowering,
and the generated CNN gradient produces non-empty MLIR.

The remaining blocker for compiled CNN training is scalability:

- lowering takes approximately 108 seconds;
- the result is approximately 47.5 MB of MLIR;
- the generic CPU MLIR pipeline does not finish within the tested time budget;
- therefore no native shared library is produced and execution cannot begin.

The next work should focus on reducing and preserving the computation graph,
not on increasing the timeout.
