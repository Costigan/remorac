# RemoraC Roadmap

This document collects user-visible milestones, research directions, and
forward-looking engineering work. For backend support gaps, see
`docs/BACKEND_GAPS.md`. For completed historical work, see
`docs/IMPLEMENTATION_LOG.md`. For documentation tasks, see `docs/DOCS_TODO.md`.

## High-Value Directions

These are not all prerequisites for correctness, but they would make RemoraC
more useful to users and more interesting as a programming-language research
platform.

### Shape Polymorphism And Dynamic Shapes

Static shapes make the current compiler tractable, but many real workloads need
runtime-sized arrays. A staged path:

1. specialize-per-shape caching for common runtime dimensions;
1. dynamic descriptor ABI support for `memref<?x...>`;
1. segmented arrays for variable-length outputs from `filter`, `replicate`, and
   recursive array builders.

This would move Remora closer to Futhark/JAX-style usability while preserving
static specialization when dimensions are known.

### Stronger GPU Execution Planning

GPU lowering now emits many individual descriptor kernels. Larger programs need
a first-class optimizer for multi-kernel plans:

1. kernel fusion across producer/consumer maps and views;
1. automatic buffer reuse and liveness-based memory planning;
1. persistent device-resident execution for iterative algorithms;
1. autotuned block sizes and specialized kernels by dtype/shape.

### Full AD On GPU

Reverse-mode AD works through the interpreter and CPU path for scalar-cost
functions. A valuable next milestone is end-to-end GPU gradient execution for
common differentiable array programs:

1. map/fold/scan VJPs on GPU;
1. view-operation cotangents on GPU;
1. pair-valued and multi-parameter gradients through the descriptor ABI;
1. optimizer loops that keep primal, gradient, and state buffers on the device.

### Property-Based And Differential Testing

The test suite is broad, but higher confidence would come from generated program
families:

1. random well-typed dense-core programs compared across interpreter, CPU, and
   GPU;
1. metamorphic shape tests for rank-polymorphic lifting;
1. oracle comparison against NumPy/JAX/Futhark for array primitives;
1. parser/typechecker fuzzing for stable diagnostics.

### Developer Tooling And Debuggability

Users need good feedback when compilation fails:

1. stable, source-located backend errors that distinguish language gaps from
   toolchain failures;
1. `--explain-lowering` summaries showing which backend path was chosen;
1. visual execution plans for multi-kernel GPU programs;
1. easier MLIR/PTX artifact capture for bug reports.

### Libraries And Domain Examples

Array languages become compelling when useful libraries are included. Good
candidates:

1. linear algebra wrappers beyond matmul: solve, eigensolvers, norms;
1. signal-processing primitives: FFT, convolution, correlation;
1. stencil and PDE kernels;
1. random number generation and Monte Carlo utilities;
1. statistical reductions and small optimization routines.

## Type System And Language Work

### Full Implementation Of The Remora Language

The current compiler implements a large dense, statically shaped subset of
Remora. A full implementation should make the paper language available across
the frontend, interpreter, CPU backend, and GPU backend where feasible.

Major milestones:

1. implement true runtime dynamic shapes in compiled code;
1. make boxes carry runtime dimension witnesses and support ragged arrays;
1. add segmented reductions and other irregular parallel primitives;
1. implement missing paper surface forms: explicit `frame`, explicit `array`,
   and `all` parameter syntax;
1. complete dynamic higher-order function semantics, including arrays of
   functions and call-through-variable in compound contexts;
1. support `shape` and `rank` for function values if retained from the full
   language model;
1. remove syntax asymmetries between ML and Lisp frontends, including
   composition support;
1. define and test CPU/interpreter/GPU parity boundaries with an executable
   support matrix;
1. keep unsupported backend features as loud, source-located errors until they
   can be implemented correctly.

This is larger than a backend project. It cuts through parser, typechecker,
elaboration, HIR, runtime representation, CPU lowering, GPU lowering, tests, and
documentation. The practical sequence is dynamic shapes first, runtime boxes and
ragged arrays second, segmented operations third, and full higher-order/MIMD
function semantics in parallel with better support-matrix tooling.

### Dynamic Shapes

Today the implementation is static-shapes-only at the lowering boundary. The
front end is genuinely dependent: index/dimension variables, `define/pi`,
`SigmaType`, `dependent_types.py`, and `index.py` exist. But
`compile_function_source` specializes every function to concrete dimensions and
then refuses to lower anything with free dimension variables.

Dynamic shapes means compiling a single function or kernel that accepts array
dimensions as runtime values and works for any size. Concretely:

1. thread dimension variables through lowering instead of resolving them to
   constants;
1. emit dynamic-shape MLIR: `tensor<?x...>`, `memref<?x...>`,
   `tensor.empty(%n)`, runtime `scf.for` bounds, and dynamic `linalg` ops;
1. make GPU kernels read dimensions from arguments or descriptors instead of
   baked constants;
1. compute runtime allocation sizes from runtime dimensions.

A tractable on-ramp is to continue widening static-shape compound-expression
coverage first, so more valid dense programs compile robustly before dimensions
go dynamic.

### JIT Shape Specialization

`remora.define()` currently requires static array sizes baked into source. A
JAX-style trace-and-specialize approach would let one definition work for many
array sizes by compiling a specialized kernel on first call for each distinct
shape signature, then caching it. This is cheaper than true dynamic shapes but
can cause per-shape recompiles and code bloat.

### Boxes And Irregular Nested Arrays

Remora's regular `Array` type is rectangular. Irregular nested data, such as an
array of arrays where each inner array has a different runtime length, needs
boxes: existential packages that carry values with hidden dimension witnesses.

The front-end plumbing already exists: `SigmaType`, `box`/`unbox`, `BoxExpr`,
`UnboxExpr`, `HIRBox`, and `HIRUnbox`. Today these are effectively type erasure
because every shape is static. True ragged arrays need boxes to carry runtime
dimension witnesses, which becomes meaningful after dynamic shapes exist.

Suggested sequence: dynamic shapes first, then runtime boxes, then irregular
arrays of boxes and segmented operations.

### Segmented Reductions

Remora papers describe segmented reductions, but no grammar entries, AST nodes,
or lowering exist for these yet. Segmented reductions are a natural companion to
runtime boxes and irregular arrays.

### Int64 And Mixed Precision

Float64 is implemented across the frontend, CPU lowering, GPU lowering, runtime,
and display paths. A next numeric expansion is int64 plus better mixed-precision
rules:

1. `Int64` literals and type annotations;
1. promotion rules for `Int`/`Int64`/`Float`/`Float64`;
1. descriptor metadata and runtime dtype plumbing for int64 arrays;
1. CPU/GPU lowering for int64 arithmetic, comparisons, scans, reductions,
   sort/grade, and index interop.

This matters for scientific and data-processing workloads where indices, counts,
and timestamps exceed 32-bit range.

### Bool Storage And Predicate Ergonomics

`Bool` is a first-class scalar and array element type. Future work is mostly
about representation and higher-level predicate workflows:

1. bit-packed bool arrays to reduce memory bandwidth for masks, filters, and
   predicate-heavy workloads;
1. predicate-array fusion for pipelines such as `map predicate -> filter` or
   `map predicate -> replicate`, avoiding materialized masks when possible;
1. higher-level aliases such as `where` and `count-true` on top of existing
   boolean folds/scans and `any`/`all`.

### Complex Numbers And FFT Primitives

Adding complex numbers would unlock frequency-domain workloads, including FFTs,
convolutions, spectral PDE solvers, and a Remora-native version of the heat1d
Fourier solver.

The full version is a substantial cross-cutting feature:

1. add `Complex` or explicit `Complex64`/`Complex128` scalar types;
1. define literal or constructor syntax;
1. add promotion rules, such as `Float + Complex64 -> Complex64`;
1. represent complex values in HIR and lowering;
1. support CPU ABI/runtime storage, likely interleaved real/imag floats;
1. support GPU descriptor ABI storage for complex arrays;
1. define arithmetic, casts, display, NumPy interop, cache metadata, and
   operation-specific behavior;
1. decide comparison policy, since complex values are not naturally ordered;
1. decide AD semantics, such as real-valued losses over complex inputs or
   Wirtinger derivatives.

Recommended first step: add `Complex64` as an array element type with
constructor/accessors before full numeric integration:

```lisp
(complex 1.0 2.0)
(real z)
(imag z)
(conj z)
(abs z)
```

Then support arrays, `map`, `+`, `-`, `*`, `/`, and perhaps `rfft`/`irfft`.
This would unlock FFT and spectral examples without forcing all numeric,
comparison, and AD policies to be solved at once.

As a lower-ceremony alternative, `rfft` could return two `Float` arrays, real and
imaginary components, and `irfft` could accept two arrays. That would unblock
some signal-processing and heat-flow use cases without introducing a complex
type, at the cost of worse general-purpose complex ergonomics.

### J/K-Like Syntax

A J- or K-inspired syntax could be valuable as an experimental frontend, not as
a replacement for the ML and Lisp syntaxes. Remora already has many of the
semantic ingredients that make J/K expressive: rank polymorphism, frame/cell
decomposition, lifting, reranking, maps, folds, scans, composition, and dense
array literals.

Interesting ideas to explore:

1. compact rank-operator syntax for Remora's `rerank ~(...)` forms;
1. adverbs that map directly to `map`, `fold`, `scan`, `trace`, and related
   higher-order array combinators;
1. tacit programming over a statically typed rank-polymorphic core;
1. shape-aware trains whose inferred frame/cell types can be checked;
1. a concise surface that expands to existing AST/HIR, paired with
   `--emit-hir` or `--explain-lowering` so users can see the expansion;
1. comparison with traditional dynamically typed array languages: what does a
   shape-safe, statically typed J-like language look like when backed by MLIR
   and GPU code generation?

Risks:

1. J/K notation is expert-friendly but can be hostile to new users;
1. a third syntax increases documentation and testing cost;
1. terse syntax can obscure the already subtle frame/cell semantics.

Suggested first step: build a small experimental parser that lowers to the
existing AST and initially supports literals, arithmetic verbs, composition,
rank/rerank, fold, scan, and map. Do not add backend behavior at first.

## Performance And Tooling

### Benchmark Strategy And Performance Targets

The benchmark plan should emphasize representative end-to-end programs, not only
single-operation microbenchmarks. Single-op numbers are still useful for
diagnosing kernel quality, but real Remora programs chain maps, views,
reductions, scans, sorts, and AD-generated kernels.

Benchmark goals:

1. compare Remora against NumPy, JAX/XLA, and Futhark on both single operations
   and pipelines;
1. separate transfer time, launch overhead, kernel time, and CPU compilation
   overhead;
1. benchmark device-resident execution separately from host round-trip
   execution;
1. report composed-vs-manually-fused performance for map-heavy chains;
1. track representative application kernels: gradient descent, convolution
   pipelines, N-body/all-pairs kernels, stencils/PDEs, FFT/signal pipelines, and
   sort/scan/reduce pipelines;
1. keep benchmark results in machine-readable JSON so regressions are visible.

Useful target goals from the benchmark work:

1. CPU maps/folds/scans should aim for vectorized or threaded throughput in the
   low billions of elements/second where memory bandwidth allows;
1. GPU map/fold/scan/sort/stencil kernels should be benchmarked at million- and
   ten-million-element scales;
1. matmul should use high-quality CPU/GPU kernels rather than naive loop nests;
1. pipeline benchmarks should show whether Remora keeps data on device and
   avoids materializing unnecessary intermediates.

### Representative Pipeline Benchmarks

Add and maintain end-to-end benchmarks that look like real programs:

1. AD optimization loops, such as `ad_optimize.lisp`, with interpreter/CPU/GPU
   variants where supported;
1. convolution pipelines: im2col or direct stencil convolution, activation, and
   pooling;
1. N-body or other all-pairs kernels, with numeric parity tests before any GPU
   result is trusted;
1. `sort -> scan`, `normalize -> sort -> prefix-sum`, and future
   `sort -> segmented-reduce` pipelines;
1. stencil/PDE update loops;
1. FFT/spectral pipelines once complex numbers or `rfft`/`irfft` exist.

For each benchmark, record:

1. correctness oracle and tolerance;
1. compile time separately from execution time;
1. host-transfer time separately from device execution;
1. whether the pipeline is CPU-only, GPU round-trip, or fully device-resident;
1. number of kernels and intermediate buffers.

### Device-Resident Execution And Plan Composition

The executor can run multi-kernel plans with device-resident intermediate
buffers. The remaining roadmap item is compiler-level composition: source
programs such as `map f (sort xs)` should lower to one combined device-resident
plan where supported, downloading only the final result.

Work items:

1. recursively lower supported sub-expressions into sub-plans;
1. merge `ExecutionPlan` objects with buffer-name prefixing and kernel-name
   disambiguation;
1. allow multi-module loading or module merging for composed plans;
1. start with single-array producer to single consumer patterns such as
   `map f (sort xs)` and `fold + 0 (scan + 0 xs)`;
1. verify that intermediates do not touch the host;
1. report the number of launches and allocated intermediate buffers in
   `--explain-lowering` or a future `--explain-schedule`.

This is composition, not fusion: it removes host round-trips but still launches
separate kernels and materializes intermediates. Fusion remains a separate
optimization.

### Kernel Fusion

The Mandelbrot iteration calls three separate kernels per step: `step_real`,
`step_imag`, and `mag_sq`. When multiple element-wise maps share the same
inputs, fusing them into one kernel eliminates intermediate allocation and
memory traffic.

Potential approaches:

1. detect chains of `remora.define()` calls applied to the same arrays and
   compile a fused kernel with multiple outputs;
1. allow `remora.define()` to accept a multi-expression body that returns a
   tuple;
1. fuse at the execution-plan level for producer/consumer maps and views;
1. add a lazy array or graph-recording mode for Python-hosted workflows, then
   lower recorded element-wise graphs to fused HIR;
1. measure composed-vs-manually-fused chains as a fusion efficiency metric.

Initial fusion targets:

1. chains of element-wise maps;
1. map/view/map pipelines;
1. map followed by reduction when the reduction body is simple;
1. multi-output fused kernels for workflows such as Mandelbrot-style update
   steps.

### GPU Plan Profiling And IR Introspection

Add tooling to attribute performance gaps to kernel quality, launch overhead,
transfer overhead, memory bandwidth, and resource limits.

Work items:

1. optional per-step timing for `ExecutionPlan`, using CUDA events or gated
   host-side synchronization;
1. `RemoraExecutor.profile_plan(plan, inputs)` returning per-kernel timing
   summaries;
1. a benchmark flag such as `remora-perf --profile` that prints kernel name,
   calls, total time, median time, and percent of total;
1. generated MLIR/PTX dumps for benchmark operations;
1. `ptxas --verbose` resource reports: registers/thread, shared memory/block,
   spills, occupancy-relevant data;
1. optional `nsys`/`ncu` instructions for ground-truth profiling.

This profiling should guide whether a performance gap is best addressed by
kernel optimization, launch reduction, plan composition, or fusion.

### CPU Vectorization And Threaded Reductions

CPU performance remains important because the CPU backend is the complete
compiled implementation target. Improve CPU codegen before relying on GPU
fallback decisions.

Work items:

1. evaluate MLIR vectorization passes for `linalg.generic` and generated loop
   nests;
1. keep vectorization behind `--cpu-vectorize` until the supported surface is
   stable;
1. implement or improve threaded reductions using OpenMP or `scf.parallel` /
   `scf.reduce`;
1. benchmark map, fold, scan, matmul, sort, stencils, and AD workloads against
   NumPy and JAX;
1. preserve scalar/static lowering paths when vectorization or threading is not
   profitable.

### Text-Path MLIR Caching

The text path emits MLIR strings which are then parsed with
`ir.Module.parse(text)`. For repeated compilations of the same program, caching
the emitted MLIR text or parsed module object could avoid regeneration.

### MLIR Builder Path Decision

The MLIR builder API path is preserved but disabled. It was the original lowering
approach, but the text path is faster and handles more of the dense core. Decide
whether to delete the builder path to reduce maintenance burden or revive it as
a structural IR validation backend, with the text path still used for normal
compilation.

### ForallType Inference Robustness

`_infer_type_vars` currently uses a single-pass approach. TypeVar bindings from
nested `FuncType`s are deferred, and concrete types from other params resolve
them. A pathological case where a `ForallType` variable appears only inside
nested function parameters could leave the binder unbound. A two-pass approach
could collect all TypeVar candidates, resolve with concrete types, and fall back
to TypeVar consensus when needed.

### Monomorphization Code Duplication

`_monomorphize_hof_calls` contains logic for detecting higher-order calls,
cloning functions, substituting parameters, and deduplicating results. A
standalone pass with helper extraction, shared with `_try_monomorphize`, would
reduce duplication and make higher-order behavior easier to audit.

### TypeVar Leak Prevention Audit

Multiple places apply `INT` as a fallback for TypeVar params and return types.
A single `_resolve_type_var(type, hint=INT)` utility would make fallback behavior
consistent and easier to audit.

### Persistent Full-Artifact Cache

`remora.define()` re-parses and re-typechecks every call even when the native
`.so` is cached. Caching full compiled artifacts by source hash, including typed
AST, HIR, kernel metadata, and native objects, would make repeated definitions
instant after the first compilation.

### Host-Side Output Arena

The compiled CPU path passes inputs zero-copy by pointer, but
`CPUFunctionExecutor.execute()` still allocates a fresh output array on every
call. A host-side output arena or size-classed host pool could reduce allocation
churn for tight iterative loops, mirroring the GPU device memory pool. The payoff
is modest, so this is a low-priority ergonomic/performance win.

### Benchmarks Against NumPy, JAX, And Futhark

A systematic benchmark suite should compare Remora-compiled code against
hand-written NumPy, JAX `jit`, and Futhark for maps, folds, scans, matmul, sort,
stencils, AD workloads, and irregular/segmented operations when implemented.
This is one of the most publishable artifacts because it tests whether
rank-polymorphic abstractions compile to competitive code.

Futhark comparison work:

1. create a small `benchmarks/futhark/` suite for map, fold, scan, matmul, sort,
   and stencil kernels;
1. support both Futhark CPU and CUDA backends where available;
1. call compiled Futhark libraries through `ctypes` or subprocess using
   Futhark data formats;
1. report Futhark results alongside Remora CPU/GPU, NumPy, and JAX.

## Interop And Ergonomics

### PyTorch Tensor Interop

Accept `torch.Tensor` inputs in `RemoraFunction.__call__`. For CPU tensors,
extract `data_ptr()`. For CUDA tensors, pass the device pointer directly to GPU
kernels.

### PyTorch Autograd Integration

Register Remora AD gradient functions as custom `torch.autograd.Function`
backward passes.

### Better Error Messages

Type errors and lowering failures currently expose compiler-internal concepts,
HIR node names, or MLIR dialect errors. Python users expect NumPy-quality
diagnostics with source locations and suggestions.

### PL Community Documentation

The Remora papers describe the semantics and type theory. A companion document
showing how rank polymorphism compiles through HIR to MLIR and GPU kernels would
bridge the gap between the theory papers and this implementation.
