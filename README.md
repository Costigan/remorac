# RemoraC

Remora is an array programming language for high-performance numerical
computation on CPUs and GPUs. It belongs to the same family as APL, J,
and Futhark where programs operate on whole arrays at once. Remora
adds a static type system that catches shape errors before execution
and enables efficient compilation to parallel hardware via MLIR.

Remora was introduced by Shivers, Slepak and Manolis in *An
Introduction to Rank-polymorphic Programming in Remora*
(arxiv.org/abs/1912.13451) and further defined by Justin Slepak in his
dissertation at Northeastern University.  Spelak's Racket
implementation is at github.com/jrslepak/Remora.

RemoraC compiles a subset of the Remora language. It currently
implements the dense, statically shaped core of Remora and compiles
ML-syntax and Lisp-syntax programs to CPU and NVIDIA GPU via MLIR and
LLVM.
		
While RemoraC is useful now for regular dense numeric programs, tests,
examples, and compiler experiments, we do intend to fully implement
the language.  Here are the known gaps:

1. true dynamic shapes
2. runtime boxes and existential dimension witnesses
3. ragged arrays and irregular data
4. segmented reductions
5. ordered structural records and data-frame-style arrays of records
6. full dynamic higher-order semantics
7. arrays of functions / MIMD function application
8. CPU lowering for dynamic and irregular values

This project was created to achieve these goals:

1. To learn and practice programming via LLM-based coding harnesses
2. To explore whether Remora would be an effective foundation for
   performing the map algebra operations underlying an existing system for
   finding landing sites for lunar missions that currently uses numpy
   and custom code in C#.NET.
3. To explore using information from static shape analysis to make
   choices between scheduling calculations on the GPU vs CPU.
4. To provide a foundation for learning how JAX, Julia and other
   languages have integrated automatic differentiation into array
   processing.

## Status

The dense core supports static rectangular arrays with ranks 0 through 10,
`Int`, `Float`, `Float64`, and `Bool` values, rank-polymorphic lifting,
`map`, `fold`/`reduce`, scans, conditionals, views, top-level definitions,
monomorphized higher-order calls, recursion, and a small standard library.

Backend support is strongest on the interpreter and compiled CPU paths. The CPU
backend covers the current dense-core typechecker surface, including recursion,
closure conversion, higher-order monomorphization, views, scans, sort/grade,
matmul, pairs, boxes as static type-erasure, and automatic differentiation for
scalar-cost functions.

The GPU backend is narrower. It supports many dense numeric patterns,
including element-wise maps, reductions, scans within current scale
limits, view operations, f32/i32 sort and grade within current limits,
f32 matmul, filter/replicate, scatter-add within current limits,
device-resident execution, and several compound map-body
expressions. Unsupported GPU programs are expected to fail loudly
rather than silently falling back or miscompiling.

## Architecture

```text
Source (.remora / .lisp)
  -> parser.py / lisp_reader.py  -> AST
  -> typechecker.py              -> Typed AST
  -> elaborate.py                -> Core IR
  -> hir.py                      -> HIR
  -> hir_opt.py / defunc.py      -> Optimized HIR
  -> lowering/                   -> MLIR for CPU
  -> gpu_lowering.py / codegen.py -> LLVM dialect / PTX for GPU
  -> runtime.py / executor.py    -> native execution
```

The interpreter remains the semantic oracle for tests. CPU and GPU compiled
results are expected to match interpreter results for supported programs.

The CPU and GPU runtimes use a descriptor ABI for arrays: aligned pointer,
offset, sizes, and strides. The ABI is documented in
[docs/ABI.md](docs/ABI.md).

## Quick Start

RemoraC uses Python 3.11+ and `uv`.

```bash
uv sync
uv run remorac examples/prelude_sum.remora
uv run remorac --target interp examples/conditional.remora
uv run remorac --syntax lisp --target interp examples/ad_optimize.lisp
```

Inspect compiler stages:

```bash
uv run remorac --emit-ast examples/prelude_sum.remora
uv run remorac --emit-typed-ast examples/prelude_sum.remora
uv run remorac --emit-hir examples/prelude_sum.remora
uv run remorac --emit-mlir examples/prelude_sum.remora
```

Run the REPL:

```bash
uv run remorac --repl
```

## Testing

Run the whole suite:

```bash
uv run pytest
```

By default, GPU tests are first-class: if CUDA is expected but unavailable, the
test session fails clearly. To tolerate a missing GPU while still running GPU
tests on machines where CUDA is available:

```bash
REMORA_TEST_GPU=0 uv run pytest
```

Fast compile check:

```bash
uv run python -m compileall -q remora
```

The test strategy emphasizes numeric parity, not compile-only smoke tests. GPU
lowering changes should include tests that run kernels and compare results
against the interpreter oracle.

## Benchmarks

Current benchmark notes are in [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md).
The snapshot there was taken on Linux x86_64 with 24 CPU cores and an NVIDIA
RTX 5090 Laptop GPU.

### Where RemoraC is competitive

1. **Scan on CPU.** At 100K, Remora-CPU (777M elem/s) beats NumPy (514M)
   by 1.5x. The compiled scalar loop with fused carry avoids NumPy's
   double-buffered `cumsum` strategy.
2. **Stencil on CPU.** 2-2.5x over NumPy at all sizes. MLIR's
   `linalg.generic` fusion of `im2col → fold-dot` into a single
   pass eliminates the 9-element intermediate allocation NumPy pays.
3. **Sort on GPU at scale.** 656M elem/s at 1M, 2x NumPy. The 256-bin
   radix sort implemented in `remora/_gpu_radix_sort.py` delivers
   competitive throughput once launch overhead amortizes.
5. **Fold on GPU.** 4.20G elem/s at 1M, within 2x of NumPy. The GPU
   reduction kernel is efficient at scale.

### Where improvement is needed

1. **Map and fold on CPU** At 450-500M elem/s, Remora-CPU is 6-20x
   behind NumPy. The vectorized CPU pipeline needs calibration and
   tuning. Target: >1G elem/s at 1M.
2. **GPU launch overhead for small N** GPU sort at
   10K (13M elem/s) and scan at 10K (82M elem/s) are slowed by
   per-kernel launch cost. Multi-kernel plans like sort (18 launches)
   compound this.
3. **Map and fold on GPU for small N** At 10K, GPU is 2-3x slower than
   CPU due to launch overhead.
4. **CPU sort** At 124-152M elem/s, the C runtime LSD radix sort is
   2-4x behind NumPy's radix-sort-based `np.sort`.
5. **Matmul on CPU for large N** At N=512, 32M elem/s. The tiled C
   kernel is single-threaded. Threading (OpenMP `scf.parallel`) or
   linking against an optimized BLAS would deliver 10-50x improvement.
6. **Kernel fusion** The `fusion` benchmark (chain-of-3 maps) shows no
   fusion at the `linalg.generic` level.
7. **GPU plan composition** The `pipeline` benchmark (`cumsum ∘ sort`)
   currently requires separate host-side calls. Compiler plan
   composition (lowering a sub-expression GPU op into a combined
   device-resident plan) would eliminate host round-trips for
   source-level pipelines.

## Roadmap

Near-term engineering work:

- Reconcile backend support documentation into one executable support matrix.
- Improve diagnostics for unsupported language and backend features.
- Add `--explain-lowering`-style visibility into GPU/CPU lowering path
  selection.
- Broaden property-based and differential tests across interpreter, CPU, and
  GPU.
- Continue modularizing large lowering files and shared emitters.

Longer-term implementation directions:

- Runtime dynamic shapes and shape-specialized caching.
- Runtime boxes, ragged arrays, and segmented reductions.
- Stronger GPU execution planning: fusion, buffer reuse, device-resident loops,
  and autotuned kernels.
- Fuller AD support on GPU with parallel adjoint rules.
- Broader numeric and library coverage, including Int64, complex numbers, FFTs,
  linear algebra, random generation, statistics, and scientific examples.

See [docs/ROADMAP.md](docs/ROADMAP.md) and
[docs/BACKEND_GAPS.md](docs/BACKEND_GAPS.md) for the detailed backlog.

## Documentation

- [docs/PROJECT_OVERVIEW_AND_ARCHITECTURE.md](docs/PROJECT_OVERVIEW_AND_ARCHITECTURE.md)
  is the best deep project overview.
- [docs/ABI.md](docs/ABI.md) specifies the descriptor ABI.
- [docs/BACKEND_GAPS.md](docs/BACKEND_GAPS.md) tracks known language and backend
  gaps.
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) documents user-facing syntax and
  commands.
- [docs/remorac-vs-futhark.md](docs/remorac-vs-futhark.md) compares RemoraC's
  approach with Futhark.
