# Documentation TODO

This document tracks documentation work split out of `docs/FUTURE_WORK.md`.

## `docs/USER_GUIDE.md` Updates

### Float64 Syntax And Type Annotations

1. Document `1.0d` / `-3.14d` literals in Lisp and ML syntax.
1. Document `Float64` / `float64` annotations in `define/pi` signatures, such
   as `(Array Float64 4)`.
1. Document float/float64 promotion rules in scan, reduce, and fold.
1. Update the literals table to include Float64.

### GPU Coverage

1. Replace legacy IREE-oriented wording with the direct-CUDA descriptor ABI
   path.
1. Document current GPU coverage: maps, reductions, scans, sort/grade, views,
   filter, replicate, matmul, scatter-add, pairs-in-map-bodies, and state-fold
   loops.
1. Document target naming, including `--target cuda` and any legacy GPU target
   aliases that still exist.
1. Document benchmark CLI behavior and device-resident execution if exposed.

### Higher-Order Functions

Add examples and limitations for closure capture in map/fold/reduce callables,
including backend differences between CPU, interpreter, and GPU.

### AD Gradient Compilation

1. Document `(grad f)` separately from host-side gradient compilation APIs.
1. Document `compile_gradient_function_source`.
1. Document `compile_gradient_functions_source` for per-input gradients.
1. Document GPU gradient compilation limits.

### Python Embedding

1. Document `remora.define()`.
1. Document `RemoraFunction`.
1. Document `%%remora` cell magic and `%remora_eval` line magic.
1. Document GPU buffer pool and device-resident APIs: `alloc_and_upload`,
   `download`, `execute_device`, and `DeviceArray`.

### Output Metadata Behavior

Document sidecar rebuild metadata:

1. metadata location: `a.json` or `<output>.json` next to the compiled artifact;
1. invalidation inputs: source hashes, toolchain fingerprint, CPU thread count,
   and vectorization flag;
1. how to force rebuild by deleting the output artifact or metadata file;
1. clarify that the old `~/.cache/remora/native/` cache is no longer used by
   the CLI.

### Acceptance Test Suite

Document how to add acceptance cases:

1. `tests/acceptance/manifest.json`;
1. `.remora` files under pass/rejected/deferred categories;
1. expected exit codes;
1. stdout/stderr checks.

## New Architecture Document

Create a companion to `DENSE_CORE.md`, `ABI.md`, and
`IMPLEMENTATION_NOTES.md` covering the end-to-end compiler pipeline for the
direct-CUDA path.

### Pipeline Diagram

Document:

```text
source -> AST -> typed AST -> elaborated core -> HIR -> optimized HIR
       -> CPU MLIR or GPU kernel builders -> native object/PTX -> ctypes
```

### GPU Codegen Cascade

Document:

1. `generate_mlir_descriptor_abi_ptx`;
1. which HIR nodes map to which kernel builders;
1. how `ExecutionPlan` works: buffer specs, kernel steps, host loops, and buffer
   swapping;
1. how `_gpu_expr_lowering.py` handles compound map/scan bodies.

### Descriptor ABI

Document:

1. allocated pointer, aligned pointer, offset, sizes, and strides;
1. how runtime wraps NumPy arrays into descriptors;
1. how device-resident execution and the buffer pool work.

### CPU Lowering Path

Document:

1. text-based MLIR emission in `lowering/tensor_ops.py` and
   `lowering/scalar.py`;
1. the internal function to wrapper function lowering chain;
1. `llvm.emit_c_interface`;
1. `mlir-opt` to LLVM dialect to `mlir-translate` to LLVM IR to `llc` to object
   to `gcc -shared`;
1. how `remora_rt.c` is compiled and linked.

### Element-Type Support Matrix

Create a current matrix for f32, f64, i32, bool, and any future numeric types by
operation and backend. Long-term, this should be generated from or checked
against an executable support matrix.

### Type System Walkthrough

Document:

1. scalar type definitions such as `FLOAT64`;
1. numeric promotion through `common_numeric_type`;
1. lowering to MLIR types;
1. NumPy dtype mapping;
1. `define/pi`, `define/forall`, dependent types, and function values.

### AD Pipeline

Document the path from `(grad f)` or host-side gradient compilation through AD
tape construction, source generation, HIR verification/optimization, and backend
lowering.
