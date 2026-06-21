# Future Work

Items that have a clear upgrade path for performance or completeness.
Completed items are marked; remaining items describe what is left.

## High-Impact

### GPU buffer arena
Every `RemoraExecutor.execute()` call currently does
`alloc → copy H→D → launch → copy D→H → free`.  Iterative
Python+Remora workflows pay this overhead per call: the Mandelbrot
example does 240 alloc/free cycles (80 iterations × 3 kernels).
`execute_plan()` pre-allocates plan buffers but frees them after
each plan execution — no reuse across calls.

A device memory arena that persists across calls and recycles
buffers by size class would eliminate allocation overhead for
iterative algorithms.  The `GPUPtxContext` pool is a partial
prototype but is only used by `execute_program_from_ptx`.

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

### Benchmarks vs NumPy / JAX / Futhark
Systematic performance comparison of Remora-compiled code against
hand-written NumPy, JAX `jit`, and Futhark for common array
operations (map, fold, scan, matmul, sort, stencil).  This is the
most publishable artifact and validates whether rank polymorphism
compiles to competitive code.

### Float64 and int64 support
Most GPU and CPU lowering paths are f32/i32-only.  Scientific
computing workloads need double precision.  Extending the descriptor
ABI, kernel generators, and type checker to support f64/i64 is
straightforward but touches many files.

### JIT shape specialization
`remora.define()` requires static array sizes baked into the source.
A JAX-style trace-and-specialize approach would let one definition
work for any array size by compiling a specialized kernel on first
call for each distinct shape signature, then caching it.

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

## Completed

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

## Remaining

### Multi-block filter and replicate (N > 1024)
Requires a multi-block i32 prefix sum.  The f32 multi-block scan
infrastructure exists; an i32 variant and integration into the
filter/replicate plans is the remaining work.

### Multi-block scan (N > 1,048,576)
Recursive multi-level scan for arrays exceeding 1024 blocks.
The current implementation falls back to serial for N > 1M.

### Parallel scatter-add (N > 1024)
Use a two-kernel `ExecutionPlan` (parallel copy + single-thread add)
or `llvm.atomicrmw fadd` for the add step.

### PyTorch tensor interop
Accept `torch.Tensor` inputs in `RemoraFunction.__call__`.  For CPU
tensors, extract `data_ptr()`.  For CUDA tensors, pass the device
pointer directly to GPU kernels.  Deferred to future work.

### PyTorch autograd integration
Register Remora's AD gradient functions as custom
`torch.autograd.Function` backward passes.  Deferred to future work.

## Abandoned

### `# coding: remora` source codec
Removed.  The codec abused Python's encoding machinery, required a
`.pth` file for direct script execution, and re-invoked the Remora
compiler on every module import.  Replaced by `remora.define()` which
accepts Remora source as a Python string and returns a compiled
callable.
