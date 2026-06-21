# Compilation Approaches: RemoraC vs Futhark

A technical comparison of two functional array language compilers
targeting CPU and GPU execution.

## Overview

**Remora** (Slepak, Shivers, Mansky — Northeastern) is a
rank-polymorphic array language where scalar functions lift
implicitly to operate on arrays of any rank.  **Futhark**
(Henriksen, Elsman, Oancea — DIKU Copenhagen) is a data-parallel
functional language with explicit nested parallelism and in-place
array updates via uniqueness types.

RemoraC is this project's compiler for Remora's dense core.
Futhark has a mature, extensively benchmarked compiler with
backends for OpenCL, CUDA, HIP, multicore C, and WebGPU.

Both compile pure functional array programs to GPU kernels.
Their approaches differ fundamentally in how parallelism is
expressed, discovered, and mapped to hardware.

## Language-Level Differences

### Parallelism model

Remora expresses parallelism through **rank polymorphism**: a
scalar function applied to an array implicitly maps over the
array's frame (leading dimensions).  The programmer never writes
`map` explicitly for lifting — the type system infers where
element-wise application occurs based on the ranks of the
arguments and the function's cell type.

Futhark expresses parallelism through **explicit combinators**:
`map`, `reduce`, `scan`, `scatter`, and `reduce_by_index`
(histogram).  The programmer chooses where parallelism occurs.
Nested `map`s express nested parallelism.

The AUTOMAP paper (OOPSLA '24) bridges these approaches by
inferring where to insert Futhark's explicit `map` calls to
recover Remora-style rank-polymorphic semantics, using integer
linear programming to find optimal map insertion points.

### In-place updates

Futhark supports in-place array updates via uniqueness types
(linear typing).  A function that consumes its input array can
update it without copying, and the type system guarantees no
aliasing.  This is critical for GPU memory efficiency.

RemoraC has no uniqueness types.  All operations produce new
arrays.  The MLIR lowering may introduce buffer reuse through
MLIR's bufferization passes, but this is not controlled by the
source language.

### Type system

Remora's type system tracks array rank and shape statically.
Functions have explicit cell types (the shape of elements they
operate on) and frame types (the shape they map over).  Rank
polymorphism is the central typing discipline.

Futhark uses size-dependent types to track array dimensions
statically.  Size expressions can appear in types (e.g.,
`[n]i32` for an array of `n` integers).  Size variables are
inferred and checked at compile time, with runtime checks
inserted where sizes cannot be resolved statically.

## Compilation Pipeline

### Futhark

```
Source → Parser → Type Checker → Internalise
  → SOACS (Second-Order Array Combinators)
  → Kernel IR (explicit GPU parallelism)
  → ImpCode (imperative)
  → OpenCL / CUDA C / multicore C / HIP / WebGPU
```

Key transformations:
1. **Internalisation**: source-level constructs lowered to a
   small set of parallel combinators (SOACS).
2. **Moderate/incremental flattening**: nested `map`-`reduce`
   compositions transformed into flat GPU kernels.  Multiple
   kernel versions may be generated for different input sizes
   (see below).
3. **ImpCode generation**: the kernel IR is lowered to an
   imperative representation with explicit memory allocation,
   index calculations, and loop nests.
4. **Backend code generation**: ImpCode emitted as OpenCL
   kernel strings, CUDA C, or C with OpenMP pragmas.

### RemoraC

```
Source (.remora / .lisp) → Parser / Lisp Reader → AST
  → Type Checker → Typed AST
  → Elaboration → Core IR
  → HIR (High-level IR)
  → Defunctionalisation → Optimised HIR
  → MLIR Lowering (tensor ops, scalar, views, module builder)
  → CPU: mlir-opt → LLVM IR → .so (via llc + gcc)
  → GPU: LLVM dialect → LLVM IR → PTX (via mlir-translate + llc)
```

Key transformations:
1. **Elaboration**: typed AST lowered to a core IR that makes
   rank-polymorphic lifting explicit.
2. **HIR erasure**: core IR simplified to HIR, a flat
   representation with `HIRMap`, `HIRFold`, `HIRScan`, etc.
3. **Defunctionalisation**: higher-order functions resolved to
   named first-order functions.
4. **MLIR lowering**: HIR nodes mapped to MLIR tensor/arith/scf
   dialect operations for CPU, or to LLVM dialect for GPU
   descriptor-ABI kernels.

## GPU Code Generation

### Futhark: Incremental Flattening

Futhark's signature contribution is **incremental flattening**
(PPoPP '19).  For a nested parallel program like
`map (\xs -> reduce (+) 0 xs) xss`, Futhark generates multiple
kernel versions:

- **Version A**: outer `map` parallelised, inner `reduce` runs
  sequentially per thread (good when outer dimension is large).
- **Version B**: both levels flattened into a single kernel with
  segmented reduction (good when inner dimension is large).
- **Version C**: fully sequentialised fallback.

A **threshold parameter** (auto-tuned per hardware) selects the
version at runtime based on actual input sizes.  This avoids the
classic flattening problem where a single strategy is optimal
for some inputs but catastrophic for others.

### RemoraC: Pattern-Matching Kernel Templates

RemoraC uses a **priority-chain dispatch** in `codegen.py` that
pattern-matches the HIR body against known kernel shapes:

1. `HIRIm2col` → im2col kernel
2. `HIRScatterAdd` → parallel scatter-add
3. `HIRMatmul` → tiled shared-memory matmul
4. `HIRSort` / `HIRGrade` → bitonic sort/grade
5. `HIRFilter` → three-kernel plan (pred + scan + scatter)
6. `HIRReplicate` → two-kernel plan
7. `HIRFold` → parallel tree reduction or scan
8. General `HIRMap` / `HIRApply` with compound body →
   `build_descriptor_abi_general_map_gpu_module`
   (compiles arbitrary element-wise expression trees to
   per-thread GPU code via `gpu_expr_from_hir`)
9. Simple f32/i32/bool element-wise maps → template kernels

Each match generates MLIR LLVM dialect text for a GPU kernel
with the Remora descriptor ABI (aligned pointer + offset +
sizes + strides).  The MLIR is compiled to LLVM IR and then
to PTX via the installed NVPTX toolchain.

For multi-kernel operations (filter, scan, sort), an
`ExecutionPlan` describes the kernel launch sequence with
named device buffers and optional host-side loops with
buffer swapping.

### Key Difference

Futhark generates **multiple kernel versions** and selects at
runtime.  RemoraC generates **one kernel per pattern** with no
multi-versioning.  Futhark's approach adapts to input sizes;
RemoraC's approach relies on static shape information to
select the right kernel template at compile time.

Futhark's IR is designed for transformation (flattening,
fusion, tiling are IR-to-IR passes).  RemoraC's GPU path
generates MLIR text directly from HIR — there is no
intermediate GPU-specific IR that optimisation passes operate
on.

## Memory Management

### Futhark

Futhark has a sophisticated memory management system
(SC '22, "Memory Optimizations in an Array Language"):

- **Last-use analysis**: arrays are freed as soon as their
  last consumer has read them.
- **Short-circuiting**: when an array is produced by one
  kernel and consumed by the next with the same shape,
  the allocation is reused without copying.
- **In-place map**: when a `map` produces an array of the
  same size as its input, the compiler can write output
  elements directly into the input buffer.
- **Memory block merging**: distinct arrays with
  non-overlapping lifetimes share the same allocation.

These optimisations are performed as compiler passes on the
ImpCode IR before backend code generation.

### RemoraC

RemoraC's memory management is minimal:

- **CPU**: output arrays are allocated by the caller (NumPy),
  passed as descriptors, and filled by the compiled `.so`.
  No buffer reuse across calls.
- **GPU `execute()`**: each call does
  `alloc → copy H→D → launch → copy D→H → free`.
  No buffer reuse.
- **GPU `execute_plan()`**: pre-allocates all plan buffers,
  executes kernel steps, frees at the end.  Better for
  multi-kernel operations but still no reuse across plan
  executions.
- **Cache**: compiled `.so` files cached by source hash
  (`cache.py`).  Avoids recompilation but not re-allocation.

This is the largest gap between RemoraC and Futhark.  A buffer
arena and last-use analysis would bring RemoraC closer to
Futhark's memory efficiency.

## Automatic Differentiation

### Futhark

Futhark added AD as a compiler-integrated feature (SC '22,
"AD for an Array Language with Nested Parallelism"):

- Forward and reverse mode AD built into the compiler.
- AD of `map`, `reduce`, `scan`, `scatter`, and
  `reduce_by_index` handled with specialised rules.
- The main challenge: AD of nested parallelism must produce
  code that is itself parallel.  A naïve reverse-mode pass
  would sequentialise the backward pass.
- Futhark's solution: custom VJP rules for each parallel
  combinator that preserve parallelism in the adjoint.

### RemoraC

RemoraC implements AD via source-to-source transformation
(`ad_source.py`):

- Reverse-mode AD applied at the AST level before HIR
  lowering.
- `(grad f)` expressions resolved by the grad-lifting pass
  at any depth in the program.
- The AD transform produces an expanded expression tree
  (32,769 nodes for `ad_optimize.lisp`) that is collapsed
  by the existing CSE pass (`hir_opt.py`) before GPU
  compilation.
- GPU-compiled AD: the gradient descent step function is
  compiled as a single GPU kernel via the general map
  module, with `HIRScatterAdd` for gradient accumulation.

The key difference: Futhark's AD preserves parallelism in the
adjoint by design (custom rules per combinator).  RemoraC's AD
is a mechanical source expansion that relies on CSE to recover
efficiency — it does not have combinator-specific adjoint rules.

## Defunctionalisation

Both languages defunctionalise higher-order functions for GPU
compilation, since GPU hardware does not support function
pointers.

Futhark's approach (TFP '18): higher-order functions are
replaced by a dispatch on a tag value representing the
function.  Lambda expressions become constructors of a sum
type, and application sites become case matches.

RemoraC's approach (`defunc.py`): lambda expressions used as
arguments to higher-order combinators (map, fold) are lifted
into top-level named functions.  The compiler monomorphises
at call sites, specialising each function for its concrete
argument types.

## Maturity and Benchmarking

Futhark has been developed since 2013, has published
extensive GPU benchmarks against hand-written CUDA/OpenCL
(rodinia, finpar, parboil suites), and is used in production
for financial computing and scientific simulation.

RemoraC is a research prototype.  It has 13 GPU integration
tests verified on an RTX 5090 but no systematic performance
benchmarks against other array language compilers.
Establishing competitive performance on standard benchmarks
would be the most impactful next step for academic
credibility.

## Where They Converge: AUTOMAP

The AUTOMAP paper (OOPSLA '24) is the most direct connection
between the two projects.  It implements **rank-polymorphic
function application inference** for Futhark: given a function
with explicit parameter ranks, AUTOMAP infers where to insert
`map` calls so that the function can be applied to
higher-ranked arguments — exactly what Remora does implicitly.

AUTOMAP uses integer linear programming to find the optimal
map insertion, handling cases where multiple valid liftings
exist.  It cites the Slepak/Shivers Remora papers as the
semantic foundation and validates that Remora's approach to
rank polymorphism can be recovered in a language with explicit
parallelism.

This suggests a potential synthesis: Remora's implicit lifting
semantics as the user-facing language, compiled through
AUTOMAP-style map insertion, then through Futhark-style
incremental flattening for GPU code generation.  RemoraC
currently skips the flattening step entirely, generating
GPU kernels directly from HIR patterns.

## Summary

| Aspect | Futhark | RemoraC |
|--------|---------|---------|
| Parallelism | Explicit combinators | Implicit rank polymorphism |
| GPU strategy | Incremental flattening + multi-versioning | Pattern-matching kernel templates |
| Memory | Last-use analysis, short-circuiting, block merging | Alloc/free per call |
| AD | Compiler-integrated with parallel adjoint rules | Source-to-source expansion + CSE |
| Defunc | Tag-based dispatch (sum types) | Lambda lifting + monomorphisation |
| Backend | OpenCL / CUDA C / HIP / multicore C / WebGPU | MLIR → LLVM IR → PTX (GPU) or `.so` (CPU) |
| Maturity | 12+ years, extensive benchmarks, production use | Research prototype, 13 GPU tests |
| Rank poly | Added via AUTOMAP inference (2024) | Core language feature |

## References

Papers in `docs/remora-reference/`:

- `futhark-pldi17.pdf` — "Futhark: Purely Functional
  GPU-Programming with Nested Parallelism and In-Place Array
  Updates" (PLDI 2017)
- `futhark-incremental-flattening-ppopp19.pdf` — "Incremental
  Flattening for Nested Data Parallelism" (PPoPP 2019)
- `futhark-ad-sc22.pdf` — "AD for an Array Language with Nested
  Parallelism" (SC 2022)
- `futhark-memory-sc22.pdf` — "Memory Optimizations in an Array
  Language" (SC 2022)
- `futhark-automap-rank-poly-oopsla24.pdf` — "AUTOMAP: Inferring
  Rank-Polymorphic Function Applications with Integer Linear
  Programming" (OOPSLA 2024)
- `futhark-defunctionalisation-tfp18.pdf` — "High-Performance
  Defunctionalisation in Futhark" (TFP 2018)
- `slepak-dissertation.pdf` — Slepak's dissertation on Remora
- `semantics-of-rank-polymorphism.pdf` — Slepak, Shivers, Mansky
