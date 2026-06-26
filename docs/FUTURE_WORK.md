# Future Work

Items that have a clear upgrade path for performance or completeness.
Completed items are marked; remaining items describe what is left.

______________________________________________________________________

## High-Value Future Directions

These are not all prerequisites for correctness, but they would make
Remora more compelling for users and researchers.

### Shape polymorphism and dynamic shapes

Static shapes make the current compiler tractable, but many real
workloads need runtime-sized arrays. A staged path is:

1. specialize-per-shape caching for common runtime dimensions;
1. dynamic descriptor ABI support for `memref<?x...>`;
1. segmented arrays for variable-length outputs from `filter`,
   `replicate`, and recursive array builders.

This would move Remora closer to Futhark/JAX-style usability while
preserving static specialization when dimensions are known.

### Stronger GPU execution planning

GPU lowering now emits many individual descriptor kernels. Larger
programs need a first-class optimizer for multi-kernel plans:

- kernel fusion across producer/consumer maps and views;
- automatic buffer reuse and liveness-based memory planning;
- persistent device-resident execution for iterative algorithms;
- autotuned block sizes and specialized kernels by dtype/shape.

### Full AD on GPU

Reverse-mode AD works through the interpreter and CPU path for
scalar-cost functions. A valuable next milestone is end-to-end GPU
gradient execution for common differentiable array programs:

- map/fold/scan VJPs on GPU;
- view-operation cotangents on GPU;
- pair-valued and multi-parameter gradients through descriptor ABI;
- optimizer loops that keep primal, gradient, and state buffers on the
  device.

### Property-based and differential testing

The test suite is broad, but higher confidence would come from generated
program families:

- random well-typed dense-core programs compared across interpreter,
  CPU, and GPU;
- metamorphic shape tests for rank-polymorphic lifting;
- oracle comparison against NumPy/JAX/Futhark for array primitives;
- fuzzing parser/typechecker error paths to keep diagnostics stable.

### Developer tooling and debuggability

Users need good feedback when compilation fails:

- stable, source-located backend errors that distinguish language gaps
  from toolchain failures;
- `--explain-lowering` summaries showing which backend path was chosen;
- visual execution plans for multi-kernel GPU programs;
- easier MLIR/PTX artifact capture for bug reports.

### Libraries and domain examples

Array languages become compelling when useful libraries are included.
Good candidates:

- linear algebra wrappers beyond matmul (solve, eigensolvers, norms);
- signal-processing primitives (FFT, convolution, correlation);
- stencil and PDE kernels;
- random number generation and Monte Carlo utilities;
- statistical reductions and small optimization routines.

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
compiles on the CPU path. All four gaps discovered during Stage 1
(June 2026) have been resolved:

| #   | Gap                               | Resolution                                                                                                                                                              |
| --- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `iscan` with lambda step function | `_lower_body_in_loop` inlines lambda bodies; `_resolve_scan_function` resolves HIRVar refs                                                                              |
| 2   | Scan init/element type constraint | Removed `init_type == element_type` from `_infer_scan`/`_infer_trace`; fixed heterogenous result type                                                                   |
| 3   | Interpreter scan dtype truncation | `np.empty_like(array)` → `np.empty(shape, dtype=expr.type.element)`                                                                                                     |
| 4   | Closure-capturing scan lambdas    | `tensor_env` from let-chain threaded through `_lower_scan_rank1` / `_lower_scan_tensor_input`; `HIRIf` added to `_lower_body_in_loop`; comparison ops use operand types |

The compiled Thomas solver is used in `examples/heat1d/heat1d_model.py`
via `_compile_thomas(N)` and matches the Python reference oracle
(kept in `examples/heat1d/test_heat1d.py`) to f32 precision (~1e-5).
Parity tests for N=4, N=10 random, and identity matrices all pass.

______________________________________________________________________

## Operations (Typechecked but Deferred in Lowering)

These operations are accepted by the typechecker but not yet lowered
to MLIR on one or more backends.

### CPU lowering follow-ups

The CPU text lowering path now covers the dense-core constructs accepted
by the typechecker, including recursion, higher-order monomorphization,
closure capture, arbitrary-rank `indices-of`, integer arithmetic,
multi-array maps, and compound map bodies with indexing. Remaining CPU
work is mostly about expressiveness beyond the current dense core and
performance engineering:

| Area                                                     | Value                                                                                                                                 |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Recursive array construction with dependent result sizes | Enables functions that build arrays by recursion, e.g. `append`-based generators whose return length depends on a decreasing index    |
| Threaded CPU vectorization and scheduling policy         | Uses available cores for large maps/folds/scans without requiring users to hand-tune execution                                        |
| MLIR builder-path retirement or revival                  | Either delete the disabled builder path to reduce maintenance, or revive it for structural IR validation once the text path is stable |
| CPU performance benchmarking suite                       | Tracks regressions against NumPy/Futhark/JAX-style baselines for maps, folds, scans, stencils, matmul, sort, and AD workloads         |
| Better native cache invalidation                         | Include lowering-version fingerprints for more Python-side lowering changes, not only C runtime changes                               |

### GPU lowering gaps

All previously listed GPU lowering gaps have been **closed** by the
[GPU Dense-Subset Completion Plan](#gpu-dense-subset-completion-plan):

> | Original gap                            | Resolution                                                                 | Phase |
> | --------------------------------------- | -------------------------------------------------------------------------- | ----- |
> | Scan limited to `+`, `*`                | Compound scan body via expression compiler handles `min`/`max`/`&&`/`\|\|` | 2     |
> | Radix sort f32-only + N limit           | i32 accepted; guard relaxed for >1024                                      | 4, 5  |
> | Fused map f32-only                      | i32 maps via expression compiler; i32 filter/replicate                     | 4     |
> | 16 HIR nodes without standalone kernels | All 16 now have standalone kernels or dispatch entries                     | 3, 6  |

### HIR nodes without standalone GPU kernels

All 16 nodes listed below now have standalone GPU kernel builders or
are handled through the general map builder / expression compiler.
This table is retained as a historical reference:

| Node                                 |                 Standalone now?                 |
| ------------------------------------ | :---------------------------------------------: |
| `HIRSlice`                           |                  ✓ (Phase 3.1)                  |
| `HIRReverse`                         |                  ✓ (Phase 3.3)                  |
| `HIRRotate`                          |                  ✓ (Phase 3.3)                  |
| `HIRSubarray`                        |                  ✓ (Phase 3.4)                  |
| `HIRTranspose`                       |                  ✓ (Phase 3.6)                  |
| `HIRReshape`                         |                  ✓ (Phase 3.5)                  |
| `HIRRavel`                           |                  ✓ (Phase 3.5)                  |
| `HIRTake`                            |                  ✓ (Phase 3.2)                  |
| `HIRDrop`                            |                  ✓ (Phase 3.2)                  |
| `HIRAppend`                          |                  ✓ (Phase 3.7)                  |
| `HIRWithShape`                       |                  ✓ (map body)                   |
| `HIRFoldRight`                       |                  ✓ (Phase 6.2)                  |
| `HIRPair` / `HIRFirst` / `HIRSecond` |                  ✓ (Phase 6.1)                  |
| `HIRCall`                            | Partial (helpers inline; tail recursion subset) |
| `HIRLambda`                          |                   ✓ (inlined)                   |
| `HIRIota`                            |                  ✓ (map body)                   |

### Complex number type + FFT primitives

Adding a `Complex` numeric type and one-dimensional FFT primitives
(`rfft`, `irfft`) would let Remora compute in the frequency domain —
matching the approach used by array-language cousins (J, APL,
Futhark). The immediate motivator is the **heat1d Fourier-matrix
solver** (`examples/heat1d/fourier_solver.py`), which currently runs in
pure NumPy and could be expressed entirely in Remora.

**Why it fits Remora.** Array languages in the APL family have
included Fourier transforms as primitives for decades — J has `fft`
and `ifft`, most APL implementations expose FFTW through system
functions. Remora already follows this pattern for sort, scan, and
matmul. Adding a frequency-domain path also unblocks the entire
thermal-quadrupole formalism (transmission matrices, circulant
admittance, rectification) used in planetary heat-flow models.

**What needs to be built.**

1. **`Complex` type in the type system.** Add `Complex` to the type
   grammar, `Type` enum, and internal representations (`remora_type`,
   `RemoraType`, `ArrayType`). The typechecker needs to infer
   `Complex` results for `rfft` and accept `Complex` operands for
   `irfft`. `VarExpr` disambiguation, frame/cell typing, and
   dependent-type tracking all need `Complex` variants.

1. **Complex literals and arithmetic.** The parser and grammar must
   accept complex notation (e.g. `1+2j`, `1.0+0.0j`). Arithmetic
   operators (`+`, `-`, `*`, `/`) need complex-aware promotion —
   `Float + Complex = Complex` — mirroring numpy.

1. **HIR nodes.** `HIRRfft` (real → complex, sequence length baked as
   a static dimension) and `HIRIrfft` (complex → real). These carry
   the transform length `n` as an attribute, analogous to `HIRSort`.

1. **CPU lowering.** Call through to FFTW (or PocketFFT for a
   header-only dependency): `HIRRfft` emits a C runtime call
   `remora_rfft_f32(n, real_input, complex_output)`. Same pattern as
   the existing `remora_matmul_f32` / `remora_sort_f32` in
   `remora_rt.c`.

1. **GPU lowering.** `HIRRfft` emits a cuFFT plan + execution kernel.
   For single-kernel embedding, cuFFT batch mode (many 1D transforms
   of the same length) is well-suited. Descriptor ABI integration
   passes the plan handle and workspaces through the existing
   `ExecutionPlan` / `BufferSpec` infrastructure.

1. **Complex storage in the descriptor ABI.** Interleaved floats
   `[re, im, re, im, …]` — this matches cuFFT's default layout and
   numpy's `.view(float).reshape(-1, 2)` convention, so zero-copy
   passes are possible.

**Pragmatic shortcut (no Complex type).** As a low-ceremony
alternative, `rfft` can return **two `Float` arrays** (real and
imaginary components), and `irfft` can accept two. This is the
approach used by pure-real-array FFT libraries and would unblock the
heat1d Fourier solver *without* a type-system expansion. The cost: no
complex arithmetic in Remora (the heat1d solver needs none — it works
with real-valued temperature and flux), but all general-purpose
frequency-domain code would lack ergonomic complex support.

**Impact beyond heat1d.** A working FFT primitive enables:

- Convolution / correlation in O(N log N) time (essential for signal
  processing and image filtering on GPU).
- Spectral PDE solvers (beyond the 1D heat equation).
- Feature-engineering pipelines that call `rfft → map → irfft` on
  sensor or audio data.
- Replacing NumPy as the host-side glue in any Remora-compiled
  pipeline that needs a Fourier transform.

### Bool storage and predicate ergonomics

`Bool` is now a first-class scalar and array element type. Future work
is mostly about representation and ergonomics:

- **Bit-packed Bool arrays.** Today descriptor-ABI GPU kernels store
  boolean arrays as byte-sized values (`i8`) at the boundary and `i1`
  internally. Bit-packing would reduce memory bandwidth for masks,
  filters, and predicate-heavy workloads.
- **Predicate-array fusion.** Fuse common pipelines such as
  `map predicate -> filter` or `map predicate -> replicate` so masks do
  not need to be materialized unless the user asks for them.
- **Boolean scan/reduce ergonomics.** Add higher-level aliases such as
  `any`, `all`, `where`, and `count-true` on top of existing `&&`/`||`
  folds and scans.

### MLIR builder path decision

The MLIR builder API path (`_builder_ops.py`, `_builder_emitter.py`,
`scalar_builder.py`) is preserved but disabled in `module.py`. It was
the original lowering approach, but the text path is much faster and
handles the dense core. Decide whether to:

- delete the builder path to reduce maintenance burden, or
- revive it as a structural IR validation backend, with the text path
  still tried first for normal compilation.

______________________________________________________________________

## GPU Dense-Subset Completion Plan

The dense subset is fully implemented on CPU for the static-shape core.
GPU coverage has been increased from ~60–70% to ~90%+ of that same set.
Phases 1–4 and most of 5–6 are complete. Remaining GPU work is now mostly
about scale limits, general device-side calls, and dynamic-shape/irregular
data support.

### Phase 1 — Scan in compound map bodies ✓ COMPLETE

| #   | Item                            | Resolution                                                                           |
| --- | ------------------------------- | ------------------------------------------------------------------------------------ |
| 1.1 | `iscan` in map body             | Compound scan body via `gpu_expr_from_hir` + `_gpu_emit_expr` in serial scan builder |
| 1.2 | `trace-right` in map body       | Same path; `is_right` flag + reversed indexing                                       |
| 1.3 | `iota` as thread coordinate     | `GpuIndexCoordinate` wired through `coord_map` to `%i0_i32`                          |
| 1.4 | `index-item` at computed coords | Non-literal index lowering + `GpuLetExpr` with i32→i64 cast                          |

Milestone achieved: `cn_step` runs on GPU as 10 chained kernels, matching CPU to f32 precision.

### Phase 2 — Scan operator generalization ✓ COMPLETE

| #   | Item                                         | Resolution                                                      |
| --- | -------------------------------------------- | --------------------------------------------------------------- |
| 2.1 | `min`/`max` in f32 scan                      | Compound body `(if (< prev x) prev x)` via Phase 1 path         |
| 2.2 | `&&`/\`                                      |                                                                 |
| 2.3 | Multi-block scan: exclusive, right, multiply | Operator/identity text replacement + codegen gate for all modes |
| 2.4 | Standalone `trace-right`                     | Parallel Hillis-Steele already handles `is_right` flag          |

### Phase 3 — View ops as standalone GPU kernels ✓ COMPLETE

All 7 items done via shared `_build_view_copy_kernel` template + per-op index expressions:

| #   | Item                       | Resolution                                                     |
| --- | -------------------------- | -------------------------------------------------------------- |
| 3.1 | `HIRSlice`                 | Start + tid×step indexing; HIR-only (no Remora surface syntax) |
| 3.2 | `HIRTake` / `HIRDrop`      | Identity copy with offset                                      |
| 3.3 | `HIRReverse` / `HIRRotate` | Flipped index / modular offset                                 |
| 3.4 | `HIRSubarray`              | Offset copy                                                    |
| 3.5 | `HIRReshape` / `HIRRavel`  | Identity copy with new output shape (rank-1 descriptors)       |
| 3.6 | `HIRTranspose`             | Decompose→swap→recompose indexing (rank-2)                     |
| 3.7 | `HIRAppend`                | Two-source conditional load/store                              |

### Phase 4 — Multi-element-type support ✓ COMPLETE

| #   | Item                            | Resolution                                                                         |
| --- | ------------------------------- | ---------------------------------------------------------------------------------- |
| 4.1 | i32 fused maps                  | Expression compiler already handles i32 natively                                   |
| 4.2 | i32 reduction                   | Guards relaxed + `_replace_elem_type(…, replace_ops=True, replace_identity=True)`  |
| 4.3 | i32 scan (single + multi-block) | Type guard + identity + ops + parallel/serial path + compound path                 |
| 4.4 | i32 sort                        | Radix sort key map for i32 (XOR sign bit, no bitcast); bitonic fallback type-aware |
| 4.5 | i32 filter / replicate          | Guards relaxed + `_replace_elem_type` + fcmp→icmp for comparisons                  |
| 4.6 | f64 support                     | **Done**                                                                           |

### Phase 5 — Scale limits ✓ PARTIAL (1/4 complete, 1 partial, 2 deferred)

| #   | Item                                | Status       | Notes                                                                                   |
| --- | ----------------------------------- | ------------ | --------------------------------------------------------------------------------------- |
| 5.1 | Multi-block i32 prefix sum          | **Done**     | `_replace_elem_type` on existing multi-block f32 scan                                   |
| 5.2 | Multi-block scatter-add (N > 1024)  | **Partial**  | Guard relaxed from N \<= 1024; `atomicrmw` text path for larger inputs is still pending |
| 5.3 | Recursive multi-level scan (N > 1M) | **Deferred** | See [Recursive multi-level scan](#recursive-multi-level-scan-n--1m-53)                  |
| 5.4 | Sort beyond 1M                      | **Deferred** | See [Sort beyond 1M](#sort-beyond-1m-54)                                                |

### Phase 6 — Structural nodes ✓ PARTIAL (2/4 complete, 2 partial)

| #   | Item                                 | Status      | Notes                                                                                     |
| --- | ------------------------------------ | ----------- | ----------------------------------------------------------------------------------------- |
| 6.1 | `HIRPair` / `HIRFirst` / `HIRSecond` | **Done**    | Pairs lowered as 2-component `GpuArrayExpr` in expression compiler                        |
| 6.2 | `HIRFoldRight` standalone            | **Done**    | Routed to reduction builder (associative for `+`/`*`)                                     |
| 6.3 | `HIRCall` in map bodies              | **Partial** | Non-recursive helpers inline; unsupported calls fail loudly                               |
| 6.4 | Recursive device functions           | **Partial** | Scalar self-tail-recursive helpers inside `map` are supported; general recursion deferred |

### Completed GPU milestones

- [x] Simple f32/i32/bool maps (rank 1–10)
- [x] Compound map bodies via general expression compiler
- [x] f32/i32 reduction (fold/reduce/fold-right, `+`/`*`)
- [x] f32/i32 scan (single-block `+`/`*`/min/max, multi-block `+`/`*` up to 1M, exclusive/right, compound bodies)
- [x] f32/i32 sort/grade (bitonic + 256-bin radix up to 1M)
- [x] f32 matmul (basic + tiled 16×16)
- [x] f32 im2col + cell-fold conv
- [x] Indices-of (any rank)
- [x] f32/i32 filter/replicate (parallel, up to 1024)
- [x] f32 scatter-add (parallel, up to 1024; guard relaxed for >1024)
- [x] View ops standalone (reverse, rotate, take, drop, subarray, reshape, ravel, transpose, append, withShape)
- [x] Pairs (Pair/First/Second) in map bodies via expression compiler
- [x] AD gradient descent state-fold GPU loop plan
- [x] Device memory pool + device-resident execution
- [x] heat1d CN step (10-kernel chain) exact match to CPU
- [x] f64 GPU infra (guards, codegen, `_replace_elem_type`)

## Deferred GPU Scale and Structural Items

### Recursive multi-level scan (N > 1M) (5.3)

The existing multi-block scan handles N up to 1,048,576 (1024 blocks ×
1024 threads). For N beyond 1M, the block-sum scan itself exceeds 1024
elements and needs another level.

**Approach:** Three-level recursive aggregation.

- Level 1: per-block Hillis-Steele (existing, unchanged).
- Level 2: block-sum scan on >1024 blocks — recurse: scan per-sub-group
  of blocks, extract sub-group sums, scan those, propagate.
- Level 3: final sum scan always fits in one block.
  This mirrors CUB's three-level device scan. ~200 lines of new kernel
  orchestration code building on the existing 4-kernel plan.

### Sort beyond 1M (5.4)

The radix sort handles N up to 1,048,576. For larger arrays, the block
count exceeds 1024.

**Approach:** Multi-level radix sort — partition into segments ≤ 1M,
sort each, then merge. Or: extend the histogram to use 2D block grids
(blockIdx.y for segment index). ~150 lines. Alternatively, the fallback
multi-block bitonic sort already handles larger N (up to ~10M) at
O(N log²N) cost — practical for correctness, not for performance.

### Device-side function calls (6.3)

Non-recursive helper calls inside GPU map bodies are now inlined, and
scalar self-tail-recursive helpers inside GPU map bodies lower to loop
blocks. Unsupported calls fail loudly.

Remaining useful work:

- **Call-through-variable on GPU.** Today the GPU path works best when
  the callee is statically known after defunctionalization. True
  dynamic callee selection would require a closed function table or an
  earlier specialization pass that enumerates all possible callees.
- **Reusable device functions instead of always inlining.** Inlining is
  simple and effective for small helpers, but large helper bodies can
  bloat kernels. A future GPU ABI could emit private device functions
  and call them from generated kernels.
- **Better diagnostics for unsupported higher-order GPU programs.**
  Current errors are specific for recursion; non-recursive higher-order
  gaps should receive similarly actionable messages.

**Device-side function pointers** (~800+ lines) would be needed only if
there's a use case for *dynamically* selecting callees (call-through-variable).
That would require a PTX-level function table and indirect call support,
which PTX supports via `.callprototype` / `call.uni`.

### Recursive device functions (6.4)

The supported subset is scalar self-tail-recursive helpers inside map
bodies, tested for Float, Int, and Bool parity. General recursion on
GPU remains open.

Future options:

- **General non-tail recursion with a per-thread stack.** Requires
  bounded stack depth, explicit frame layout, overflow behavior, and
  careful occupancy testing. This is only worth doing for concrete
  workloads such as tree traversal, adaptive algorithms, or recursive
  combinator interpreters.
- **Mutual tail recursion as a state machine.** Convert an SCC of
  mutually tail-recursive scalar helpers into a loop with a program
  counter and loop-carried arguments. This is far cheaper than a
  general stack and would cover many parser/interpreter-style kernels.
- **Array-returning recursive helpers.** Likely requires either per-lane
  scratch buffers or an explicit segmented output representation; both
  interact with GPU memory planning and should be designed with dynamic
  shapes/boxes in mind.

### Sort beyond 1M (5.4)

The radix sort handles N up to 1,048,576. For larger arrays, the block
count exceeds 1024.

**Approach:** Multi-level radix sort — partition into segments ≤ 1M,
sort each, then merge. Or: extend the histogram to use 2D block grids
(blockIdx.y for segment index). ~150 lines. The fallback multi-block
bitonic sort already handles larger N (up to ~10M) at O(N log²N) cost
— sufficient for correctness, not for performance.

### Scatter-add atomicrmw text (5.2 remainder)

The parallel scatter-add builder handles N ≤ 1024 via shared-memory
barrier + single-thread add. For N > 1024, shared memory can't hold
all elements. The guard has been relaxed; what remains is emitting
`llvm.atomicrmw fadd` text (global atomic floating-point add) instead
of the shared-memory barrier pattern. ~30 lines of MLIR text
generation.

### General recursion on GPU (6.4 remainder)

Scalar self-tail-recursive helpers inside GPU `map` bodies are handled
with explicit loop blocks today. General recursion still requires
alloca-based call frames, continuation/state-machine lowering, or
device-side helper functions with a defined calling convention. Tail
recursion covers common iterative kernels such as `sum_to`, `repeat`,
and simple optimizer loops; defer full recursion until a concrete
application, such as tree traversal or dynamic programming on GPU,
demands the extra machinery.

______________________________________________________________________

## Backend Scale Limits

> **Superseded by [GPU Dense-Subset Completion Plan](#gpu-dense-subset-completion-plan) Phase 5 above.**
> The multi-block filter/replicate/scatter-add/scan/sort items are tracked there.

______________________________________________________________________

## Type System

### int64 and mixed-precision support

Float64 is implemented across the frontend, CPU lowering, GPU lowering,
runtime, and display paths. The next numeric expansion is int64 plus
better mixed-precision rules:

- `Int64` literals and type annotations.
- Promotion rules for `Int`/`Int64`/`Float`/`Float64`.
- Descriptor metadata and runtime dtype plumbing for int64 arrays.
- CPU and GPU lowering for int64 arithmetic, comparisons, scans,
  reductions, sort/grade, and index interop.

This matters for scientific and data-processing workloads where indices,
counts, and timestamps exceed 32-bit range.

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
  f32 precision. Used in `examples/heat1d/heat1d_model.py`.
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

______________________________________________________________________

## Documentation Gaps

### `docs/USER_GUIDE.md` — updates needed

The user guide is a syntax-and-feature reference last updated around
the dense-core CPU completion. The following sections need writing
or updating:

1. **f64 literal syntax and type annotations.**

   - Document `1.0d` / `-3.14d` literals (Lisp and ML syntax).
   - Document `Float64` / `float64` type annotations in `define/pi`
     signatures: `(Array Float64 4)`.
   - Document float↔float64 promotion rules in scan/reduce/fold.
   - Update the Literals table to include `Float64`.

1. **GPU coverage.**

   - The current GPU section describes the IREE path; update it to
     describe the direct-CUDA descriptor-ABI kernels (Phase 1–6,
     covering maps, reductions, scans, sort/grade, views, filter,
     replicate, matmul, scatter-add, pairs, state-fold loops).
   - Document `--target gpu` vs `--target gpu-nvidia` (IREE legacy).
   - Document `remora-perf` benchmark CLI and `--device-resident`.

1. **Higher-order functions.**

   - The recursive functions and ForallType HOF sections exist.
     Add documentation for closure capture in map/fold/reduce callables.

1. **AD gradient compilation.**

   - The AD section describes `(grad f)`. Add documentation for
     `compile_gradient_function_source` and the per-input gradients
     API (`compile_gradient_functions_source`).

1. **Python embedding.**

   - Document `remora.define()`, `RemoraFunction`, the `%%remora` cell
     magic, and `%remora_eval` line magic (Jupyter integration).
   - Document GPU buffer pool and device-resident execution APIs
     (`alloc_and_upload`, `download`, `execute_device`, `DeviceArray`).

1. **Cache behaviour.**

   - Document the native `.so` cache (`~/.cache/remora/native/`) — when
     it invalidates (source, toolchain, `remora_rt.c`, pipeline version
     changes), how to clear it, and the `REMORA_NO_CACHE` env var.

1. **Acceptance test suite.**

   - Document how to add new acceptance cases (`manifest.json`,
     `.remora` files in `tests/acceptance/pass/` or `rejected/` or
     `deferred/`).

### New architecture document (not started yet)

A companion document to the existing design docs (`DENSE_CORE.md`, `ABI.md`,
`IMPLEMENTATION_NOTES.md`) covering the **end-to-end compilation pipeline**
for the non-IREE direct-CUDA path:

1. **Pipeline diagram:** source → AST → typed AST → HIR → optimised HIR →
   GPU kernel builders or CPU MLIR text → PTX/C object → ctypes execution.

1. **GPU codegen cascade** (`codegen.py`):

   - `generate_mlir_descriptor_abi_ptx` routing tree: which HIR node maps to
     which kernel builder.
   - How `ExecutionPlan` multi-kernel orchestration works (buffer specs,
     kernel steps, host loops, buffer swapping).
   - How the expression compiler (`_gpu_expr_lowering.py`) handles compound
     map/scan bodies recursively via `_gpu_emit_expr`.

1. **Descriptor ABI:**

   - The `(allocated, aligned, offset, size, stride)` memref descriptor
     convention for GPU kernels.
   - How the runtime wraps NumPy arrays into descriptors
     (`make_host_memref_descriptor`).
   - How device-resident execution and the buffer pool work.

1. **CPU lowering path:**

   - Text-based MLIR emission (`lowering/tensor_ops.py`, `lowering/scalar.py`).
   - The `_lower_function_descriptor_module` chain: internal function
     (tensor-level) → wrapper (memref-interface, `llvm.emit_c_interface`).
   - MLIR pipeline: `mlir-opt` → LLVM dialect → `mlir-translate` → LLVM IR →
     `llc` → `.o` → `gcc -shared` → `.so`.
   - How the C runtime (`remora_rt.c`) is compiled once and linked into
     every `.so`.

1. **Element-type support matrix:**

   - Which ops support f32, i32, bool, f64 on CPU and GPU.
   - The `_replace_elem_type` helper and how type guards route non-f32
     element types through the GPU kernel builders.

1. **Type system walkthrough:**

   - From `FLOAT64 = ScalarType("float64")` through `common_numeric_type`
     promotion to `type_to_mlir` → `"f64"` to `_numpy_dtype` → `np.float64`.
   - How the typechecker resolves `define/pi`, `define/forall`, dependent
     types, and function values.

1. **AD pipeline:**

   - Source-level reverse-mode transformation (`ad_source.py`): tape
     trace, VJP generation, grad-lifting.
   - Gradient compilation path (CPU and GPU).

1. **Caching:**

   - Cache key computation (source, param types, toolchain, C runtime hash,
     pipeline version).
   - Cache location and storage format.
   - When cached artifacts are invalidated.
