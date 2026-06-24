# Future Work

Items that have a clear upgrade path for performance or completeness.
Completed items are marked; remaining items describe what is left.

---

## Language Features (Deferred or Missing)

These are features from the Remora academic papers that the compiler
rejects or does not implement.  They are **upstream of lowering** —
the typechecker or parser gates them.

### Dynamic higher-order functions (polyvariadic application)

Functions passed as arguments work via monomorphization.  Call-through-variable
(e.g. `let f = inc in f(f 5)`) works.  What does not work: passing a function
through a `map` body as a callable (`map f arr` where `f` is a let-bound
variable).  The map lowering needs to resolve the callable from the scalar
environment.

### Functions in function position (MIMD arrays-of-functions)

The typechecker defers map over function-valued arrays
(`frame.py:121`, `frame.py:176`).  This blocks the classic Remora MIMD
pattern `(define m [[square sqrt] [add1 sub1]]) (m 9)`.

### Remaining text-path deferrals

One site remains intentionally deferred in `tensor_ops.py`:
**binary map operator sections** — a unary section callable like `(* 2)`
passed to a binary map is semantically ambiguous (one input, two arrays).
The typechecker currently rejects sections in binary callable positions;
full Remora would support this with pair-type output.

### `shape` / `rank` of function values

Deferred (`hir.py:909`, `runtime.py:1783`).  Querying the shape or
rank of a function-typed value is not supported.

### Missing surface syntax from the papers

Three forms present in the Remora language spec are absent from the
current grammar:

- **`(frame [d1 … dn] expr1 … exprn)`** — explicit frame construction
- **`(array [d1 … dn] atom1 … atomn)`** — explicit array of atoms
- **`all` keyword on parameters** — consuming an entire argument as one cell

### `ComposeExpr` (`∘`) asymmetry

Function composition (`∘`) is in the ML-syntax grammar and parser but
not in the Lisp reader.  Lisp programs cannot use composition.

---

## Operations (Typechecked but Deferred in Lowering)

These operations are accepted by the typechecker but not yet lowered
to MLIR on one or more backends.

### CPU lowering gaps

| Operation | Status | Location |
|-----------|--------|----------|
| Integer division `/i` scalar | Deferred | `scalar.py:348` |
| Ternary+ maps (>2 arrays) | Deferred | `module.py:662`, `tensor_ops.py` |
| Indices-of for arbitrary ranks | Deferred | `tensor_ops.py:3255` |
| Sort for non-f32 element types | Deferred | `tensor_ops.py:3655` |
| Threaded CPU vectorization | Not supported | `pipeline.py:311` |

### GPU lowering gaps

| Operation | Status | Location |
|-----------|--------|----------|
| GPU scan limited to `+`, `*` | `min`/`max`/`&&`/`\|\|` rejected | `gpu_lowering.py:594` |
| GPU radix sort f32-only + N limit | i32 rejected | `_gpu_radix_sort.py:437` |
| GPU fused map f32-only | i32 outputs deferred | `_gpu_map_support.py` |
| 16 HIR nodes have no standalone GPU kernel | Work inside map bodies only | See below |

### GPU lowering gaps

| Operation | Status | Location |
|-----------|--------|----------|
| GPU scan limited to `+`, `*` | `min`/`max`/`&&`/`\|\|` rejected | `gpu_lowering.py:594` |
| GPU radix sort f32-only + N limit | i32 rejected | `_gpu_radix_sort.py:437` |
| GPU fused map f32-only | i32 outputs deferred | `_gpu_map_support.py` |
| 16 HIR nodes have no standalone GPU kernel | Work inside map bodies only | See below |

### HIR nodes without standalone GPU kernels

These HIR nodes have no dedicated `codegen.py` dispatch or
`gpu_lowering.py` builder.  Most work correctly **inside map bodies**
via the general GPU expression compiler (`_gpu_expr_lowering.py`), but
cannot be compiled as top-level kernels:

| Node | Inside map body? |
|------|:---:|
| `HIRSlice` | ✗ |
| `HIRReverse` | ✓ |
| `HIRRotate` | ✓ |
| `HIRSubarray` | ✓ |
| `HIRTranspose` | Limited |
| `HIRReshape` | ✓ |
| `HIRRavel` | ✓ |
| `HIRTake` | ✓ |
| `HIRDrop` | ✓ |
| `HIRAppend` | ✓ |
| `HIRWithShape` | ✓ |
| `HIRFoldRight` | ✗ |
| `HIRPair` / `HIRFirst` / `HIRSecond` | ✗ |
| `HIRCall` | ✗ |
| `HIRLambda` | ✗ |
| `HIRIota` | Limited |

### Builder API (IREE path) — disabled

The MLIR builder API path (`_builder_ops.py`, `_builder_emitter.py`,
`scalar_builder.py`) is preserved but disabled in `module.py`.  It was
the original lowering approach, but the text path is ~175x faster and
handles all patterns.  If the builder is needed again (e.g. for
structural IR verification), reorder the fallback so text is tried
first.

---

## Backend Scale Limits

### Multi-block operations (N > 1024)

- **Filter and replicate** need a multi-block i32 prefix sum.
  The f32 multi-block scan infrastructure exists; an i32 variant
  remains to be built.
- **Scatter-add** currently uses a single-block kernel with
  barrier + thread-0 add.  For N > 1024, needs a two-kernel plan
  or `llvm.atomicrmw fadd`.

### Multi-block scan (N > 1,048,576)

Recursive multi-level scan for arrays exceeding 1024 blocks.
Falls back to serial beyond 1M elements.

### Fused GPU map scale limits

The fused GPU map optimization (`_gpu_map_support.py`) currently
handles 1–10 array inputs and float outputs only.

---

## Type System

### Float64 and int64 support

Most GPU and CPU lowering paths are f32/i32-only.  Scientific
computing workloads need double precision.  Extending the descriptor
ABI, kernel generators, and type checker to support f64/i64 is
straightforward but touches many files.

### Dynamic shapes (runtime array dimensions)

Today the implementation is **static-shapes-only at the lowering boundary**.
The type front-end is genuinely dependent — index/dimension variables, `Π`
types (`define/pi`), `Σ` types, `dependent_types.py` / `index.py` — but
`compile_function_source` *specializes* (monomorphizes) every function to
concrete dimensions and then refuses to lower anything with a free dimension
variable (`compiler.py`: "compiled function … has unspecialized index
variables"). Every emitted artifact bakes the sizes in: tensor/memref types
(`tensor<3x2x2xf32>`), `scf.for` bounds, buffer sizes, GPU grid/block, and the
descriptor ranks all come from `StaticDim` constants.

Dynamic shapes means compiling a **single** function/kernel that accepts array
dimensions as *runtime* values and works for any size — true shape polymorphism,
matching Remora's semantics. Concretely:
- Thread dimension variables through to lowering instead of resolving them to
  constants (relax the "no free index variables" gate; discharge equality
  constraints from the dependent-type checker, with residual runtime checks for
  what can't be proven statically).
- Emit dynamic-shape MLIR: `tensor<?x…>` / `memref<?x…>`, `tensor.empty(%n)`
  with dynamic operands, runtime `scf.for` bounds, dynamic `linalg` ops. MLIR
  and linalg already support this, so the target is capable.
- GPU: kernels read dimensions from arguments / the descriptor ABI (which
  *already* carries runtime sizes and strides) and compute indices, loop bounds,
  and grid/block dynamically instead of from baked constants.
- Runtime: compute buffer/allocation sizes from runtime dimensions.

This is foundational but a large, cross-cutting lift (typechecker → elaborate →
HIR → lowering → runtime). A tractable on-ramp is to keep widening
compound-expression lowering coverage on static shapes first (the recent GPU
general-map and rank-≥2-cell work is part of this), so more valid dense programs
compile robustly before dimensions go dynamic.

### JIT shape specialization

`remora.define()` requires static array sizes baked into the source.
A JAX-style trace-and-specialize approach would let one definition
work for any array size by compiling a specialized kernel on first
call for each distinct shape signature, then caching it. (This is the
*specialize-per-shape* alternative to true dynamic shapes — cheap to
build on what exists, but causes per-shape recompiles and code bloat.)

### Boxes / irregular (ragged) nested arrays

Remora's regular `Array` type is **rectangular** — every sub-array along a given
axis has the same shape — which is exactly what makes implicit rank-polymorphic
lifting (frame/cell decomposition) well-defined. *Irregular* nested data — "an
array of arrays where each inner array has a different, possibly runtime,
length" — therefore cannot be a rectangular `Array`. Remora expresses it with
**boxes**: a `box` is an atom that existentially packages a value together with
its dimension witnesses (`Σ` / `SigmaType`, the dependent *sum*). You build a
regular array *of boxes*, and `unbox` opens one (binding its hidden dimensions)
before use.

The front-end plumbing already exists — `SigmaType`, `box`/`unbox` surface
syntax, `BoxExpr`/`UnboxExpr`, `HIRBox`/`HIRUnbox` — but `box`/`unbox` are
currently implemented as **type erasure with no runtime effect**. That is only
sound today because every shape is static: the existentially-hidden dimension is
actually a known constant, so the box carries no real runtime information.
Genuinely ragged data needs a box to carry a *runtime* dimension witness, which
only becomes meaningful once dynamic shapes exist.

Sequence: dynamic shapes first, then make `box`/`unbox` carry and recover
runtime dimensions for true ragged arrays-of-arrays.

### Segmented reductions

Remora papers describe segmented reductions (grouped reductions where
group boundaries are data-driven).  No grammar entries, AST nodes, or
lowering exist for these.

---

## Performance and Tooling

### Remove the builder path entirely

The MLIR builder API path (`_builder_ops.py`, `_builder_emitter.py`,
`scalar_builder.py`) is ~175x slower than the text path and handles
fewer patterns.  It is currently commented out in `module.py`.  Once
the text path has proven itself over time, the builder files can be
deleted to simplify the codebase (~1000 lines).

### Text-path MLIR caching

The text path emits MLIR strings which are then parsed via
`ir.Module.parse(text)`.  For repeated compilations of the same
program (e.g. iterative development), caching the parsed module
object or the emitted MLIR text would avoid re-generation.

### ForallType type inference robustness

`_infer_type_vars` uses a single-pass approach: TypeVar bindings from
nested FuncTypes are deferred, and concrete types from other params
resolve them.  A pathological case where a ForallType variable
appears *only* inside nested FuncType params (no top-level concrete
param to resolve it) would leave the binder unbound.  A two-pass
approach (collect all TypeVar candidates, resolve with concrete
types, fall back to TypeVar consensus) would be more robust.

### Monomorphization code duplication

`_monomorphize_hof_calls` at ~200 lines has logic for detecting HOF
calls, cloning functions, substituting, and deduplicating.  A
standalone pass with helper extraction (`_clone_function_body`,
`_substitute_params`) and sharing with `_try_monomorphize` would
reduce duplication.

### TypeVar leak prevention audit

Multiple places apply `INT` as a fallback for `TypeVar` params and
return types (`_lift_lambda`, `erase_to_hir`, `_resolve_main_return_type`).
A systematic approach — a single `_resolve_type_var(type, hint=INT)`
utility — would make the fallback behavior consistent and auditable.

### Kernel fusion

The Mandelbrot iteration calls three separate kernels per step
(`step_real`, `step_imag`, `mag_sq`), each writing to and reading
from intermediate device buffers.  When multiple element-wise maps
share the same inputs, fusing them into a single kernel eliminates
intermediate allocation and memory traffic.

Approach: detect chains of `remora.define()` calls applied to the
same arrays and compile a fused kernel with multiple outputs.  Or,
allow `remora.define()` to accept a multi-expression body that
returns a tuple.

### Host-side output arena for compiled-CPU iterative loops

The CPU analogue of the GPU buffer arena, with a much smaller payoff.
On the compiled-CPU path the per-call allocation cost is already low —
inputs are passed **zero-copy by pointer** (`array.ctypes.data`) and there
is no host↔device transfer — but `CPUFunctionExecutor.execute()` still
allocates a **fresh output array** (`_empty_output_value` → host `malloc`)
on every call.  For tight iterative loops (e.g. a stencil or optimizer that
calls one compiled function thousands of times with the same output shape),
reusing the output buffer avoids that per-call allocation and the associated
GC churn.

The building block already exists: a host `Arena` (a bump allocator over a
`bytearray` with `reset()`) can be passed via `execute(..., arena=...)`.
What is missing is an *ergonomic, automatic* mode — e.g. an executor option
that keeps a right-sized output arena and `reset()`s it each call, or a
size-classed host pool mirroring the GPU `DeviceMemoryPool` — so callers get
reuse without manually managing an arena.  Modest payoff (host `malloc` /
numpy allocation is already cheap and the OS allocator pools memory), so this
is a low-priority ergonomic win, mainly useful for very hot CPU loops.

### Benchmarks vs NumPy / JAX / Futhark

Systematic performance comparison of Remora-compiled code against
hand-written NumPy, JAX `jit`, and Futhark for common array
operations (map, fold, scan, matmul, sort, stencil).  This is the
most publishable artifact and validates whether rank polymorphism
compiles to competitive code.

---

## Interop and Ergonomics

### PyTorch tensor interop

Accept `torch.Tensor` inputs in `RemoraFunction.__call__`.  For CPU
tensors, extract `data_ptr()`.  For CUDA tensors, pass the device
pointer directly to GPU kernels.

### PyTorch autograd integration

Register Remora's AD gradient functions as custom
`torch.autograd.Function` backward passes.

### Better error messages

Type errors and lowering failures produce compiler-internal messages
(HIR node names, MLIR dialect errors).  Python users expect
NumPy-quality diagnostics with source locations and suggestions.

### Persistent full-artifact cache

`remora.define()` re-parses and re-typechecks every call even when
the native `.so` is cached by `cache.py`.  Caching the full compiled
artifact (typed AST, HIR, kernel metadata) by source hash would make
repeated `define()` calls instant after the first compilation.

### Documentation for the PL community

Remora was designed by Slepak, Shivers, and Mansky at Northeastern.
The academic papers in `docs/remora-reference/` describe the
semantics and type theory.  A companion document showing how rank
polymorphism compiles through HIR → MLIR → GPU kernels, with
concrete examples of implicit lifting and frame/cell decomposition,
would bridge the gap between the theory papers and this
implementation.

---

## Completed

### Recursive functions — typechecker, interpreter, CPU compilation
- Self-recursion (tail/non-tail), mutual recursion, deep call chains.
- Typechecker: fixpoint inference with provisional FuncType.
- Interpreter: tail-call trampoline, mutual trampoline.
- CPU: `HIRCall` → MLIR `func.call`, manual bufferization for array types.
- 15+ regression tests covering `fac`, `fib`, `sum_to`, `is_even`/`is_odd`, Ackermann.

### Higher-order functions — monomorphization, closure capture, lambda lifting
- Function values passed as arguments, stored in let bindings.
- Monomorphization pass clones HOFs per call site and substitutes concrete functions.
- Full closure conversion: lambdas with captures compile on CPU.
- ForallType HOF: `define/forall` with `(Func (t) t)` params compiles.
- 20+ HOF regression tests.

### Text-path deferrals closed (4 of 5)
- Fold operator sections, exclusive/right scans rank ≥ 2, cell-fold producer
  map sections, binary cell-map guard removed.
- One remaining (binary map operator sections) is a defensive guard for
  a pattern the typechecker rejects.

### Builder path disabled
- Builder API path was ~175x slower and fell back for 66% of programs.
  Commented out in `module.py`; text path only.

### GPU buffer arena (device memory pool)

- `DeviceMemoryPool` recycles device buffers by power-of-two size class, so
  `RemoraExecutor.execute()` reuses allocations instead of doing a
  `cuMemAlloc`/`cuMemFree` on every call.
- Lives on the `CUDARuntime` as a shared pool — buffers are reused across
  `execute()` calls *and* across executors that share the runtime (the
  iterative multi-kernel case) — and is drained (`cuMemFree`) when the runtime
  is closed.
- Runtimes without a shared pool (lightweight test fakes) get a local
  executor-owned pool drained on `close()`.  Steady-state memory is bounded by
  (distinct size classes) × (peak concurrent buffers per class), not by call
  count.

### Parallel GPU Filter and Replicate (N ≤ 1024)

- `HIRFilter`: three-kernel plan (predicate eval → i32 prefix sum →
  scatter-write).
- `HIRReplicate`: two-kernel plan (prefix sum on counts →
  scatter-replicate).
- All kernels orchestrated by `ExecutionPlan`.

### Host-Orchestrated GPU Optimization Loops

- `ad_optimize.lisp` compiles to a GPU `LoopPlan` via
  `try_compile_state_fold_gpu`.  200-step gradient descent runs on
  GPU producing the correct result `[0.512337, 0.433115, 0.911621]`.
- CSE collapses the AD source transform's 32,769-node gradient
  expression before GPU compilation.

### Tiled Shared-Memory Matmul

- TILE=16 cooperative loading.  Falls back to naive per-thread
  dot-product when the tiled version fails to compile.

### Multi-Block Parallel Scan (up to 1M elements)

- Four-kernel plan: per-block Hillis-Steele → extract block sums →
  scan block sums → propagate prefixes.

### Parallel Sort and Grade

- Single-block bitonic sort/grade for N ≤ 1024.
- Multi-block bitonic sort and grade for N > 1024 with odd-block
  reversal, double-buffered global merge, and i32 value-lookup
  grade.  Supports up to ~1M elements.

### Parallel Scatter-Add (N ≤ 1024)

- Single-block kernel: parallel copy + barrier + thread-0 add.

---

## Abandoned

### `# coding: remora` source codec

Removed.  The codec abused Python's encoding machinery, required a
`.pth` file for direct script execution, and re-invoked the Remora
compiler on every module import.  Replaced by `remora.define()` which
accepts Remora source as a Python string and returns a compiled
callable.
