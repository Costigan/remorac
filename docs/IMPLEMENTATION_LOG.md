# Implementation Log

This document records completed implementation phases, historical plans, and
abandoned work. For active roadmap items, see `docs/ROADMAP.md`. For current
backend gaps, see `docs/BACKEND_GAPS.md`.

## heat1d: Thomas Algorithm Status

The Thomas tridiagonal solver is fully implemented in Remora and compiles on the
CPU path. All four gaps discovered during Stage 1 have been resolved:

1. `iscan` with lambda step functions: `_lower_body_in_loop` inlines lambda
   bodies and `_resolve_scan_function` resolves `HIRVar` references.
1. Scan init/element type constraint: removed `init_type == element_type` from
   scan/trace inference and fixed heterogeneous result type.
1. Interpreter scan dtype truncation: replaced `np.empty_like(array)` with
   allocation based on `expr.type.element`.
1. Closure-capturing scan lambdas: threaded `tensor_env` from let chains
   through scan lowering, added `HIRIf` to `_lower_body_in_loop`, and fixed
   comparison op types.

The compiled Thomas solver is used in `examples/heat1d/heat1d_model.py` via
`_compile_thomas(N)` and matches the Python reference oracle to f32 precision.
Parity tests for `N=4`, `N=10` random, and identity matrices pass.

## GPU Dense-Subset Completion Plan

The dense subset is fully implemented on CPU for the static-shape core. GPU
coverage has been expanded substantially. Remaining GPU items are tracked in
`docs/BACKEND_GAPS.md`.

### Phase 1: Scan In Compound Map Bodies

Complete.

1. `iscan` in map bodies lowers through `gpu_expr_from_hir` and `_gpu_emit_expr`.
1. `trace-right` uses the same path with an `is_right` flag and reversed
   indexing.
1. `iota` can act as a thread coordinate.
1. `index-item` supports computed coordinates through non-literal index
   lowering and i32-to-i64 casts.

Milestone: `cn_step` runs on GPU as a 10-kernel chain and matches CPU to f32
precision.

### Phase 2: Scan Operator Generalization

Complete.

1. `min`/`max` in f32 scan compile as compound `if` bodies.
1. boolean scan support was added for relevant paths.
1. multi-block scan supports exclusive, right, and multiply modes through
   operator/identity replacement and codegen gates.
1. standalone `trace-right` uses the existing parallel Hillis-Steele path.

### Phase 3: View Ops As Standalone GPU Kernels

Complete via shared `_build_view_copy_kernel` template and per-op index
expressions:

1. `HIRSlice`;
1. `HIRTake` / `HIRDrop`;
1. `HIRReverse` / `HIRRotate`;
1. `HIRSubarray`;
1. `HIRReshape` / `HIRRavel`;
1. `HIRTranspose`;
1. `HIRAppend`.

### Phase 4: Multi-Element-Type Support

Complete.

1. i32 fused maps through the expression compiler;
1. i32 reduction through `_replace_elem_type`;
1. i32 scans for single-block and multi-block paths;
1. i32 sort key mapping;
1. i32 filter/replicate guards and comparison replacement;
1. f64 infrastructure.

### Completed GPU Milestones

1. simple f32/i32/bool maps, ranks 1-10;
1. compound map bodies through the general expression compiler;
1. f32/i32 reductions;
1. f32/i32 scans up to current scale limits;
1. f32/i32 sort/grade up to current scale limits;
1. f32 matmul, including tiled 16x16 path;
1. f32 im2col and cell-fold convolution paths;
1. indices-of for any rank;
1. f32/i32 filter and replicate up to current scale limits;
1. f32 scatter-add up to current scale limits;
1. standalone view ops and descriptor reinterpretation paths;
1. pairs in map bodies through the expression compiler;
1. AD gradient-descent state-fold GPU loop plan;
1. device memory pool and device-resident execution;
1. heat1d Crank-Nicolson step as a 10-kernel chain;
1. f64 GPU infrastructure.

## Completed Language And Compiler Work

### Recursive Functions

Complete on typechecker, interpreter, and CPU compiled path:

1. self recursion, tail and non-tail;
1. mutual recursion;
1. deep call chains;
1. fixpoint inference with provisional `FuncType`;
1. interpreter tail-call trampoline;
1. CPU `HIRCall` to MLIR `func.call`;
1. manual bufferization for array types;
1. regression tests for factorial, Fibonacci, `sum_to`, even/odd, and
   Ackermann-style programs.

### Higher-Order Functions

Complete for the dense CPU/interpreter subset:

1. function values passed as arguments and stored in let bindings;
1. monomorphization pass that clones higher-order functions per call site;
1. closure conversion for lambdas with captures on CPU;
1. `ForallType` higher-order functions with `(Func (t) t)` parameters.

### Text-Path Deferrals Closed

Closed items:

1. fold operator sections;
1. exclusive/right scans at rank >= 2;
1. cell-fold producer map sections;
1. binary cell-map guard removal.

One remaining binary map operator-section guard is tracked in
`docs/BACKEND_GAPS.md`.

### Builder Path Disabled

The MLIR builder API path was measured as much slower and less capable than the
text path. It is commented out in `module.py`; normal compilation uses the text
path.

### GPU Buffer Arena

`DeviceMemoryPool` recycles device buffers by power-of-two size class. It lives
on `CUDARuntime` as a shared pool, so buffers are reused across `execute()` calls
and across executors that share a runtime. Lightweight runtimes without a shared
pool get an executor-owned local pool drained on `close()`.

### Parallel GPU Filter And Replicate

1. `HIRFilter`: predicate evaluation, i32 prefix sum, scatter-write.
1. `HIRReplicate`: prefix sum on counts, scatter-replicate.
1. Both are orchestrated by `ExecutionPlan`.

### Host-Orchestrated GPU Optimization Loops

`ad_optimize.lisp` compiles to a GPU `LoopPlan` through
`try_compile_state_fold_gpu`. A 200-step gradient descent run executes on GPU
and produces the expected result. CSE collapses the AD source transform's large
gradient expression before GPU compilation.

### Tiled Shared-Memory Matmul

The tiled matmul path uses `TILE=16` cooperative loading. It falls back to a
naive per-thread dot-product path when the tiled version fails to compile.

### Multi-Block Parallel Scan

Implemented as a four-kernel plan:

1. per-block Hillis-Steele scan;
1. extract block sums;
1. scan block sums;
1. propagate prefixes.

Current scale limits are tracked in `docs/BACKEND_GAPS.md`.

### Parallel Sort And Grade

Implemented:

1. single-block bitonic sort/grade for `N <= 1024`;
1. multi-block bitonic sort/grade for larger inputs up to current limits;
1. odd-block reversal, double-buffered global merge, and i32 value-lookup
   grade;
1. radix sort path for supported cases.

### Parallel Scatter-Add

Single-block kernel implemented for `N <= 1024`: parallel copy, barrier, and
thread-0 add. Larger-input atomic text path remains in `docs/BACKEND_GAPS.md`.

### Scientific Notation In Parsers

The `FLOAT` regex in both `lisp_reader.py` and `grammar.lark` accepts forms such
as `1e5`, `1.5e-3`, and `10E+3`, including integer-with-exponent syntax.

### Scan Lambdas On CPU Compiled Path

Implemented:

1. `_resolve_scan_function` resolves step functions from `HIRVar`;
1. `_lower_scan_rank1` and `_lower_scan_tensor_input` inline lambda bodies via
   `_lower_body_in_loop`;
1. scan lambdas support `let`, `if`, arithmetic, and index operations;
1. heterogeneous scan init/element type support;
1. interpreter scan result dtype fix.

### Closure-Capturing Scan Lambdas And Thomas Solver

Implemented:

1. `HIRScan` dispatch in `_lower_main_result_with_tensor_env`;
1. `_lower_scan_tensor_let_result`;
1. threading captured tensor environment into scan lambda lowering;
1. `HIRIf` handling in `_lower_body_in_loop`;
1. fixed comparison MLIR type annotations;
1. `render_blocks()` for raw MLIR block extraction;
1. full Thomas tridiagonal solver on compiled CPU path.

## Abandoned

### `# coding: remora` Source Codec

Removed. The codec abused Python's encoding machinery, required a `.pth` file
for direct script execution, and re-invoked the Remora compiler on every module
import. It was replaced by `remora.define()`, which accepts Remora source as a
Python string and returns a compiled callable.
