# Future Work

Items that have a clear upgrade path for performance or completeness.
Completed items are marked; remaining items describe what is left.

______________________________________________________________________

## Language Features (Deferred or Missing)

These are features from the Remora academic papers that the compiler
rejects or does not implement. They are **upstream of lowering** —
the typechecker or parser gates them.

### Dynamic higher-order functions (polyvariadic application)

Functions passed as arguments work via monomorphization. Call-through-variable
(e.g. `let f = inc in f(f 5)`) works. What does not work: passing a function
through a `map` body as a callable (`map f arr` where `f` is a let-bound
variable). The map lowering needs to resolve the callable from the scalar
environment.

### Functions in function position (MIMD arrays-of-functions)

The typechecker defers map over function-valued arrays
(`frame.py:121`, `frame.py:176`). This blocks the classic Remora MIMD
pattern `(define m [[square sqrt] [add1 sub1]]) (m 9)`.

### Remaining text-path deferrals

One site remains intentionally deferred in `tensor_ops.py`:
**binary map operator sections** — a unary section callable like `(* 2)`
passed to a binary map is semantically ambiguous (one input, two arrays).
The typechecker currently rejects sections in binary callable positions;
full Remora would support this with pair-type output.

### `shape` / `rank` of function values

Deferred (`hir.py:909`, `runtime.py:1783`). Querying the shape or
rank of a function-typed value is not supported.

### Missing surface syntax from the papers

Three forms present in the Remora language spec are absent from the
current grammar:

- **`(frame [d1 … dn] expr1 … exprn)`** — explicit frame construction
- **`(array [d1 … dn] atom1 … atomn)`** — explicit array of atoms
- **`all` keyword on parameters** — consuming an entire argument as one cell

### `ComposeExpr` (`∘`) asymmetry

Function composition (`∘`) is in the ML-syntax grammar and parser but
not in the Lisp reader. Lisp programs cannot use composition.

______________________________________________________________________

## heat1d: Thomas Algorithm Status

The Thomas tridiagonal solver is fully implemented in Remora and
compiles on the CPU path.  All four gaps discovered during Stage 1
(June 2026) have been resolved:

| # | Gap | Resolution |
|---|-----|-----------|
| 1 | `iscan` with lambda step function | `_lower_body_in_loop` inlines lambda bodies; `_resolve_scan_function` resolves HIRVar refs |
| 2 | Scan init/element type constraint | Removed `init_type == element_type` from `_infer_scan`/`_infer_trace`; fixed heterogenous result type |
| 3 | Interpreter scan dtype truncation | `np.empty_like(array)` → `np.empty(shape, dtype=expr.type.element)` |
| 4 | Closure-capturing scan lambdas | `tensor_env` from let-chain threaded through `_lower_scan_rank1` / `_lower_scan_tensor_input`; `HIRIf` added to `_lower_body_in_loop`; comparison ops use operand types |

The compiled Thomas solver is used in `examples/heat1d/heat1d_model.py`
via `_compile_thomas(N)` and matches the Python reference oracle
(kept in `examples/heat1d/test_heat1d.py`) to f32 precision (~1e-5).
Parity tests for N=4, N=10 random, and identity matrices all pass.

______________________________________________________________________

## Operations (Typechecked but Deferred in Lowering)

These operations are accepted by the typechecker but not yet lowered
to MLIR on one or more backends.

### CPU lowering gaps

| Operation                                            | Status        | Location                                  |
| ---------------------------------------------------- | ------------- | ----------------------------------------- |
| Integer division `/i` scalar                         | Deferred      | `scalar.py:348`                           |
| Ternary+ maps (>2 arrays)                            | Deferred      | `module.py:662`, `tensor_ops.py`          |
| Indices-of for arbitrary ranks                       | Deferred      | `tensor_ops.py:3255`                      |
| Sort for non-f32 element types                       | Deferred      | `tensor_ops.py:3655`                      |
| Threaded CPU vectorization                           | Not supported | `pipeline.py:311`                         |
| `index-item` (`HIRIndex`) inside map body            | Deferred      | `tensor_ops.py` map body emitter          |
| Recursive array construction via `define/pi`         | Blocked       | typechecker rejects mismatched sizes      |
| Cross-function calls in closure-capturing map bodies | Deferred      | defunctionalization pass                  |

#### Index-in-map

`index-item` (`HIRIndex`) inside a `map` body is rejected by the map
body expression emitter. This blocks any `map` that needs per-element
access to a companion array (e.g. building triples for a Thomas scan
by indexing into separate `upper`, `diag`, `lower` arrays). The
heat1d Crank-Nicolson RHS assembly attemped `(map (lambda (i) ... (index-item T i)) (iota N))` and hit this gap.

Discovered: June 2026, heat1d Stage 1.

#### Recursive array construction

`define/pi` requires a concrete (static) return shape. A recursive
helper that builds an array via `(append [val] (recurse ...))` produces
size `1 + returned_size` at each level, so the typechecker sees
`float[1]` vs `float[6]` and rejects the body. Dependent types
(`define/pi ([k Dim]) ... (Array Float (- n i))`) could express this
but are not yet threaded through lowering.

Discovered: June 2026, heat1d Stage 1.

#### Closure-capturing map bodies calling top-level functions

A `map` body lambda that captures a free variable and calls a
top-level `define/pi` function hits `unknown HIR function` in the
defunctionalization pass. Direct `(map square arr)` (no closure,
function passed as callable) works; `(map (lambda (i) (peek arr i)) (iota N))` (captures `arr`, calls `peek`) does not.

Discovered: June 2026, heat1d Stage 1.

### GPU lowering gaps

| Operation                                  | Status                           | Location                 |
| ------------------------------------------ | -------------------------------- | ------------------------ |
| GPU scan limited to `+`, `*`               | `min`/`max`/`&&`/`\|\|` rejected | `gpu_lowering.py:594`    |
| GPU radix sort f32-only + N limit          | i32 rejected                     | `_gpu_radix_sort.py:437` |
| GPU fused map f32-only                     | i32 outputs deferred             | `_gpu_map_support.py`    |
| 16 HIR nodes have no standalone GPU kernel | Work inside map bodies only      | See below                |

### HIR nodes without standalone GPU kernels

These HIR nodes have no dedicated `codegen.py` dispatch or
`gpu_lowering.py` builder. Most work correctly **inside map bodies**
via the general GPU expression compiler (`_gpu_expr_lowering.py`), but
cannot be compiled as top-level kernels:

| Node                                 | Inside map body? |
| ------------------------------------ | :--------------: |
| `HIRSlice`                           |        ✗         |
| `HIRReverse`                         |        ✓         |
| `HIRRotate`                          |        ✓         |
| `HIRSubarray`                        |        ✓         |
| `HIRTranspose`                       |     Limited      |
| `HIRReshape`                         |        ✓         |
| `HIRRavel`                           |        ✓         |
| `HIRTake`                            |        ✓         |
| `HIRDrop`                            |        ✓         |
| `HIRAppend`                          |        ✓         |
| `HIRWithShape`                       |        ✓         |
| `HIRFoldRight`                       |        ✗         |
| `HIRPair` / `HIRFirst` / `HIRSecond` |        ✗         |
| `HIRCall`                            |        ✗         |
| `HIRLambda`                          |        ✗         |
| `HIRIota`                            |     Limited      |

### Complex number type + FFT primitives

Adding a `Complex` numeric type and one-dimensional FFT primitives
(`rfft`, `irfft`) would let Remora compute in the frequency domain —
matching the approach used by array-language cousins (J, APL,
Futhark).  The immediate motivator is the **heat1d Fourier-matrix
solver** (`examples/heat1d/fourier_solver.py`), which currently runs in
pure NumPy and could be expressed entirely in Remora.

**Why it fits Remora.**  Array languages in the APL family have
included Fourier transforms as primitives for decades — J has `fft`
and `ifft`, most APL implementations expose FFTW through system
functions.  Remora already follows this pattern for sort, scan, and
matmul.  Adding a frequency-domain path also unblocks the entire
thermal-quadrupole formalism (transmission matrices, circulant
admittance, rectification) used in planetary heat-flow models.

**What needs to be built.**

1. **`Complex` type in the type system.**  Add `Complex` to the type
   grammar, `Type` enum, and internal representations (`remora_type`,
   `RemoraType`, `ArrayType`).  The typechecker needs to infer
   `Complex` results for `rfft` and accept `Complex` operands for
   `irfft`.  `VarExpr` disambiguation, frame/cell typing, and
   dependent-type tracking all need `Complex` variants.

2. **Complex literals and arithmetic.**  The parser and grammar must
   accept complex notation (e.g. `1+2j`, `1.0+0.0j`).  Arithmetic
   operators (`+`, `-`, `*`, `/`) need complex-aware promotion —
   `Float + Complex = Complex` — mirroring numpy.

3. **HIR nodes.**  `HIRRfft` (real → complex, sequence length baked as
   a static dimension) and `HIRIrfft` (complex → real).  These carry
   the transform length `n` as an attribute, analogous to `HIRSort`.

4. **CPU lowering.**  Call through to FFTW (or PocketFFT for a
   header-only dependency): `HIRRfft` emits a C runtime call
   `remora_rfft_f32(n, real_input, complex_output)`.  Same pattern as
   the existing `remora_matmul_f32` / `remora_sort_f32` in
   `remora_rt.c`.

5. **GPU lowering.**  `HIRRfft` emits a cuFFT plan + execution kernel.
   For single-kernel embedding, cuFFT batch mode (many 1D transforms
   of the same length) is well-suited.  Descriptor ABI integration
   passes the plan handle and workspaces through the existing
   `ExecutionPlan` / `BufferSpec` infrastructure.

6. **Complex storage in the descriptor ABI.**  Interleaved floats
   `[re, im, re, im, …]` — this matches cuFFT's default layout and
   numpy's `.view(float).reshape(-1, 2)` convention, so zero-copy
   passes are possible.

**Pragmatic shortcut (no Complex type).**  As a low-ceremony
alternative, `rfft` can return **two `Float` arrays** (real and
imaginary components), and `irfft` can accept two.  This is the
approach used by pure-real-array FFT libraries and would unblock the
heat1d Fourier solver *without* a type-system expansion.  The cost: no
complex arithmetic in Remora (the heat1d solver needs none — it works
with real-valued temperature and flux), but all general-purpose
frequency-domain code would lack ergonomic complex support.

**Impact beyond heat1d.**  A working FFT primitive enables:
- Convolution / correlation in O(N log N) time (essential for signal
  processing and image filtering on GPU).
- Spectral PDE solvers (beyond the 1D heat equation).
- Feature-engineering pipelines that call `rfft → map → irfft` on
  sensor or audio data.
- Replacing NumPy as the host-side glue in any Remora-compiled
  pipeline that needs a Fourier transform.

### Bool type

Remora currently has `Int` and `Float` but no `Bool` type — boolean
results from comparisons are represented as `Int` (0/1).  A first-class
`Bool` type would:

- Allow type-safe `if` conditions and predicate arrays.
- Enable boolean reductions (`fold &&`, `fold ||`) with proper typing.
- Let GPU filter/replicate operations produce and consume `Bool` arrays
  instead of i32 workarounds.
- Improve error messages when a boolean is expected but a float/int is
  provided.

The main work is in the type grammar (`Bool` variant), typechecker
(comparison result types, conditional predicates), HIR (`HIRIf`
already uses a scalar condition — formalize), MLIR lowering (i1 vs i32
in `arith.select`/`scf.if`), and the descriptor ABI (bool descriptors
for GPU kernels).

The MLIR builder API path (`_builder_ops.py`, `_builder_emitter.py`,
`scalar_builder.py`) is preserved but disabled in `module.py`. It was
the original lowering approach, but the text path is ~175x faster and
handles all patterns. If the builder is needed again (e.g. for
structural IR verification), reorder the fallback so text is tried
first.

______________________________________________________________________

## GPU Dense-Subset Completion Plan

The dense subset is fully implemented on CPU — every construct the
typechecker accepts compiles and runs.  GPU coverage is ~60–70% of that
same set.  The phases below close the remaining gaps in priority order.

### Phase 1 — Scan in compound map bodies (unblocks heat1d CN on GPU)

These four items are needed for the heat1d Crank-Nicolson Thomas
solver (forward `iscan` + backward `trace-right`) to compile on GPU
inside a map body via the general expression compiler.

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 1.1 | `iscan` in map body | Medium | `GpuScan` node wrapping Hillis-Steele in shared memory; serial fallback for N > 1024 to start. |
| 1.2 | `trace-right` in map body | Medium | Reverse-direction of 1.1; shares kernel structure, reversed index ordering. |
| 1.3 | `iota` as thread coordinate in compound bodies | Small | `GpuIndexCoordinate` already exists; needs wiring as explicit `iota` source for scan indices. |
| 1.4 | `index-item` from captured arrays at computed coordinates | Small | Resolves to `GpuInputLoad` at let-bound coordinates; coordinate-from-let path partially wired. |

Milestone: `cn_step` runs on GPU, matching CPU output to f32 precision.

### Phase 2 — Scan operator generalization

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 2.1 | `min`/`max` in f32 scan | Small | `llvm.intr.minnum`/`maxnum` in scan kernel builder; wire through `GpuReduce`. |
| 2.2 | `&&`/`||` in bool scan | Small | i1 scan kernel with `llvm.and`/`llvm.or`, mirroring f32 path. |
| 2.3 | Multi-block scan: exclusive, right, multiply modes | Medium | Extend 4-kernel plan beyond inclusive-add-only. |
| 2.4 | Standalone `trace-right` kernel | Medium | Reverse-scan builder mirroring existing scan module builder. |

### Phase 3 — View ops as standalone GPU kernels

Every view op already works inside map bodies via the expression
compiler.  This phase adds top-level kernels so they appear outside
map bodies, matching the CPU path and the codegen dispatch cascade.

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 3.1 | `HIRSlice` standalone | Medium | Only view op also rejected inside map bodies; needs descriptor ABI with per-dimension offset+size. |
| 3.2 | `HIRTake` / `HIRDrop` standalone | Small | Per-dimension offset in descriptor; wrapper kernel adjusting base pointer + shape. |
| 3.3 | `HIRReverse` / `HIRRotate` standalone | Small | Descriptor-level: flip stride sign or offset; no data movement. |
| 3.4 | `HIRSubarray` standalone | Small | Per-dimension offset/size, descriptor-level. |
| 3.5 | `HIRReshape` / `HIRRavel` standalone | Small | Descriptor-only: reshape permutes shape/strides, ravel flattens. |
| 3.6 | `HIRTranspose` standalone | Medium | Stride permutation in descriptor; needs output allocation for non-contiguous result. |
| 3.7 | `HIRAppend` standalone | Medium | Concatenation: two memcpy regions or parallel copy with conditional source select. |

### Phase 4 — Multi-element-type support

All GPU operations currently hardcode f32 (a few support i32).  This
phase generalizes the kernel builders and expression compiler.

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 4.1 | i32 fused map (compound expressions) | Medium | `_gpu_map_support.py` defers i32 fused maps; needs `I32Expr` tree. |
| 4.2 | i32 reduction (fold) | Medium | Grid-strided + shmem tree reduce with i32 loads/stores. |
| 4.3 | i32 scan (single-block + multi-block) | Medium | Generalize f32 scan builders to i32. |
| 4.4 | i32 sort | Medium | Radix sort works on uint32 keys; i32→uint32 is same bitcast+sign-flip. Extend type dispatch. |
| 4.5 | i32 filter / replicate / scatter-add | Small | Extend existing f32 plans with i32 load/store variants. |
| 4.6 | f64 support (all ops) | Large | Requires f64 descriptor ABI, f64 MLIR typing, alignment; touches every GPU file. |

### Phase 5 — Scale limits (multi-block >1024)

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 5.1 | Multi-block i32 prefix sum (for filter/replicate) | Medium | f32 multi-block scan exists; i32 variant unblocks filter/replicate N > 1024. |
| 5.2 | Multi-block scatter-add (N > 1024) | Small | `llvm.atomicrmw fadd` for global-memory atomics, or two-kernel plan. |
| 5.3 | Recursive multi-level scan (N > 1M) | Medium | Three-level approach: scan blocks → scan block-sums → propagate. |
| 5.4 | Sort beyond 1M | Medium | Extend radix-sort block count or multi-level aggregation. |

### Phase 6 — Structural nodes (pairs, call, recursion)

These are the hardest remaining items — they require structural changes
to the GPU kernel model.

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 6.1 | `HIRPair` / `HIRFirst` / `HIRSecond` in map bodies | Medium | 2-component `GpuPairExpr` in expression compiler. |
| 6.2 | `HIRFoldRight` standalone | Medium | Reverse-direction reduction; same kernel structure as fold. |
| 6.3 | `HIRCall` in map bodies | Large | Cross-kernel calls need device-side function pointers or callee-body inlining. |
| 6.4 | Recursive device functions | Large | Needs bounded-stack depth + manual stack; defer until a concrete use case demands it. |

### Completed GPU milestones

- [x] Simple f32/i32/bool maps (rank 1–10)
- [x] Compound map bodies via general expression compiler
- [x] f32 reduction (fold/reduce, `+`/`*`)
- [x] f32 scan (single-block `+`/`*`, multi-block `+` up to 1M)
- [x] f32 sort/grade (bitonic + 256-bin radix up to 1M)
- [x] f32 matmul (basic + tiled 16×16)
- [x] f32 im2col + cell-fold conv
- [x] Indices-of (any rank)
- [x] f32 filter/replicate (parallel, up to 1024)
- [x] f32 scatter-add (parallel, up to 1024)
- [x] View ops inside map bodies (reverse, rotate, subarray, take, drop, append, reshape, ravel, withShape)
- [x] AD gradient descent state-fold GPU loop plan
- [x] Device memory pool + device-resident execution

______________________________________________________________________

## Backend Scale Limits

> **Superseded by [GPU Dense-Subset Completion Plan](#gpu-dense-subset-completion-plan) Phase 5 above.**
> The multi-block filter/replicate/scatter-add/scan/sort items are tracked there.

______________________________________________________________________

## Type System

### Float64 and int64 support

Most GPU and CPU lowering paths are f32/i32-only. Scientific
computing workloads need double precision. Extending the descriptor
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
group boundaries are data-driven). No grammar entries, AST nodes, or
lowering exist for these.

______________________________________________________________________

## Performance and Tooling

### Remove the builder path entirely

The MLIR builder API path (`_builder_ops.py`, `_builder_emitter.py`,
`scalar_builder.py`) is ~175x slower than the text path and handles
fewer patterns. It is currently commented out in `module.py`. Once
the text path has proven itself over time, the builder files can be
deleted to simplify the codebase (~1000 lines).

### Text-path MLIR caching

The text path emits MLIR strings which are then parsed via
`ir.Module.parse(text)`. For repeated compilations of the same
program (e.g. iterative development), caching the parsed module
object or the emitted MLIR text would avoid re-generation.

### ForallType type inference robustness

`_infer_type_vars` uses a single-pass approach: TypeVar bindings from
nested FuncTypes are deferred, and concrete types from other params
resolve them. A pathological case where a ForallType variable
appears *only* inside nested FuncType params (no top-level concrete
param to resolve it) would leave the binder unbound. A two-pass
approach (collect all TypeVar candidates, resolve with concrete
types, fall back to TypeVar consensus) would be more robust.

### Monomorphization code duplication

`_monomorphize_hof_calls` at ~200 lines has logic for detecting HOF
calls, cloning functions, substituting, and deduplicating. A
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
from intermediate device buffers. When multiple element-wise maps
share the same inputs, fusing them into a single kernel eliminates
intermediate allocation and memory traffic.

Approach: detect chains of `remora.define()` calls applied to the
same arrays and compile a fused kernel with multiple outputs. Or,
allow `remora.define()` to accept a multi-expression body that
returns a tuple.

### Host-side output arena for compiled-CPU iterative loops

The CPU analogue of the GPU buffer arena, with a much smaller payoff.
On the compiled-CPU path the per-call allocation cost is already low —
inputs are passed **zero-copy by pointer** (`array.ctypes.data`) and there
is no host↔device transfer — but `CPUFunctionExecutor.execute()` still
allocates a **fresh output array** (`_empty_output_value` → host `malloc`)
on every call. For tight iterative loops (e.g. a stencil or optimizer that
calls one compiled function thousands of times with the same output shape),
reusing the output buffer avoids that per-call allocation and the associated
GC churn.

The building block already exists: a host `Arena` (a bump allocator over a
`bytearray` with `reset()`) can be passed via `execute(..., arena=...)`.
What is missing is an *ergonomic, automatic* mode — e.g. an executor option
that keeps a right-sized output arena and `reset()`s it each call, or a
size-classed host pool mirroring the GPU `DeviceMemoryPool` — so callers get
reuse without manually managing an arena. Modest payoff (host `malloc` /
numpy allocation is already cheap and the OS allocator pools memory), so this
is a low-priority ergonomic win, mainly useful for very hot CPU loops.

### Benchmarks vs NumPy / JAX / Futhark

Systematic performance comparison of Remora-compiled code against
hand-written NumPy, JAX `jit`, and Futhark for common array
operations (map, fold, scan, matmul, sort, stencil). This is the
most publishable artifact and validates whether rank polymorphism
compiles to competitive code.

______________________________________________________________________

## Interop and Ergonomics

### PyTorch tensor interop

Accept `torch.Tensor` inputs in `RemoraFunction.__call__`. For CPU
tensors, extract `data_ptr()`. For CUDA tensors, pass the device
pointer directly to GPU kernels.

### PyTorch autograd integration

Register Remora's AD gradient functions as custom
`torch.autograd.Function` backward passes.

### Better error messages

Type errors and lowering failures produce compiler-internal messages
(HIR node names, MLIR dialect errors). Python users expect
NumPy-quality diagnostics with source locations and suggestions.

### Persistent full-artifact cache

`remora.define()` re-parses and re-typechecks every call even when
the native `.so` is cached by `cache.py`. Caching the full compiled
artifact (typed AST, HIR, kernel metadata) by source hash would make
repeated `define()` calls instant after the first compilation.

### Documentation for the PL community

Remora was designed by Slepak, Shivers, and Mansky at Northeastern.
The academic papers in `docs/remora-reference/` describe the
semantics and type theory. A companion document showing how rank
polymorphism compiles through HIR → MLIR → GPU kernels, with
concrete examples of implicit lifting and frame/cell decomposition,
would bridge the gap between the theory papers and this
implementation.

______________________________________________________________________

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
  executor-owned pool drained on `close()`. Steady-state memory is bounded by
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
  `try_compile_state_fold_gpu`. 200-step gradient descent runs on
  GPU producing the correct result `[0.512337, 0.433115, 0.911621]`.
- CSE collapses the AD source transform's 32,769-node gradient
  expression before GPU compilation.

### Tiled Shared-Memory Matmul

- TILE=16 cooperative loading. Falls back to naive per-thread
  dot-product when the tiled version fails to compile.

### Multi-Block Parallel Scan (up to 1M elements)

- Four-kernel plan: per-block Hillis-Steele → extract block sums →
  scan block sums → propagate prefixes.

### Parallel Sort and Grade

- Single-block bitonic sort/grade for N ≤ 1024.
- Multi-block bitonic sort and grade for N > 1024 with odd-block
  reversal, double-buffered global merge, and i32 value-lookup
  grade. Supports up to ~1M elements.

### Parallel Scatter-Add (N ≤ 1024)

- Single-block kernel: parallel copy + barrier + thread-0 add.

### Scientific notation in parsers

- Extended the `FLOAT` regex in both `lisp_reader.py` and `grammar.lark`
  to accept `1e5`, `1.5e-3`, `10E+3`, etc. The old regex required a
  decimal point; the new one also matches integer-with-exponent
  (`[0-9]+[eE][+-]?[0-9]+`). 7 regression tests.
  Discovered/Fixed: June 2026, heat1d Stage 1c.

### Scan lambdas (CPU compiled path)

- `_resolve_scan_function` resolves the step function from `HIRVar`
  references (post-defunctionalization).
- `_lower_scan_rank1` and `_lower_scan_tensor_input` inline lambda bodies
  via `_lower_body_in_loop`, supporting `let`, `if`, arithmetic, and
  index operations inside scan lambdas.
- Removed the `init_type == element_type` constraint from
  `_infer_scan`/`_infer_trace` in the type checker (blocked scans where
  the carry and element types differ, e.g. scalar init + array elements).
- Fixed scan result type for heterogeneous init/element.
- Fixed interpreter scan result dtype — `np.empty_like(array)` inherited
  the input array's Int32 dtype, truncating Float results; now uses
  `expr.type` to determine the output dtype.
- 7 new tests in `test_properties.py::test_scan_family`.
  Discovered/Fixed: June 2026, heat1d Stage 1.

### Closure-capturing scan lambdas + full Thomas algorithm (CPU compiled path)

- Added HIRScan dispatch in `_lower_main_result_with_tensor_env` so
  let-lowering can lower scans that appear as let-chain bodies.
- Created `_lower_scan_tensor_let_result` to thread `tensor_env` from
  enclosing let-bindings into the scan lambda lowering.
- Merged the passed-in `tensor_env` into the step-function environment
  in both `_lower_scan_rank1` and `_lower_scan_tensor_input`, so
  `(index-item upper i)` on captured arrays works inside scan lambdas.
- Added `HIRIf` handling to `_lower_body_in_loop` (scalar condition
  with `scf.if`), needed for `(if (< i 1) 0.0 ...)` guard expressions.
- Fixed comparison MLIR type annotation in `_lower_body_in_loop` to use
  operand type (not `i1`) and add the required comma after `arith.cmpi/cmpf`
  predicates.
- Added `render_blocks()` to `_MLIRMainModuleBuilder` for extracting raw
  MLIR blocks without the module wrapper.
- Fixed `input_elem_remora` access to handle `HIRVar` array nodes in
  function-parameter scan lowering.
- The full Thomas tridiagonal solver (cp forward sweep, denominator
  computation, dp forward sweep, back substitution via `trace-right`)
  compiles and runs on the CPU path, matching the Python reference to
  f32 precision.  Used in `examples/heat1d/heat1d_model.py`.
- 3 new Remora-vs-Python parity tests in `examples/heat1d/test_heat1d.py`.
  Discovered/Fixed: June 2026, heat1d Stage 1.

______________________________________________________________________

## Abandoned

### `# coding: remora` source codec

Removed. The codec abused Python's encoding machinery, required a
`.pth` file for direct script execution, and re-invoked the Remora
compiler on every module import. Replaced by `remora.define()` which
accepts Remora source as a Python string and returns a compiled
callable.
