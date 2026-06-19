# Python Integration Plan

## Goal

Bring Remora's rank-polymorphic semantics into the Python data science
ecosystem by using Python's native extension points (`# coding:` codec
and `%%remora` cell magic) to intermix Remora S-expression source with
Python code.  Compiled Remora functions appear as ordinary Python
callables that accept NumPy arrays with zero-copy interop.

The existing interpreter, CPU (MLIR), and GPU (LLVM/PTX) backends
remain the compilation targets.  This plan adds the Python-facing API
layer, not a new backend.

## Architecture

```
Python source (.py or Jupyter cell)
  → Ingestion (codec or cell magic)
  → Remora S-expression blocks extracted
  → Lisp reader (remora/lisp_reader.py) → AST
  → Type checker (remora/typechecker.py) → Typed AST
  → Compiler pipeline (remora/compiler.py) → native .so
  → Python callable wrapper (RemoraFunction)
  → Registered in user's namespace
```

At call time:

```
Python: result = my_remora_fn(np_array_1, np_array_2)
  → JIT rank/shape check against Remora type signature
  → Extract __array_interface__ (pointer, shape, strides)
  → Build descriptor structs
  → Call compiled native function via ctypes
  → Wrap output as NumPy array
  → Return to Python
```

## Phase 1 — Source Codec and Core Wrapper (`# coding: remora`)

Build the `RemoraFunction` callable wrapper and the source codec for
standalone `.py` files.

- [x] **1.1 — `RemoraFunction` callable wrapper**: A Python class that
      wraps a compiled native function.  `__call__` accepts NumPy arrays,
      extracts `__array_interface__` metadata (pointer, shape, strides),
      builds Remora descriptor structs, calls the compiled `.so` via
      ctypes, and wraps the output as a NumPy array.  The existing
      `CPUFunctionExecutor` already does most of this — refactor it
      into a clean public API.

- [x] **1.2 — JIT boundary checking**: When `RemoraFunction.__call__`
      is invoked, verify that the input arrays' ranks and shapes satisfy
      the Remora function's type signature.  Raise a clear
      `RemoraRankMismatchError` before any computation or allocation if
      frames fail to align.

- [x] **1.3 — Codec registration**: Register a Python codec named
      `remora` via `codecs.register`.  When a `.py` file starts with
      `# coding: remora`, the codec intercepts the raw bytes before
      CPython's tokenizer.

- [x] **1.4 — Block delimiter convention**: Define how Remora blocks
      are delimited within Python source.  Candidates:
      - `# remora:begin` / `# remora:end` comment markers
      - Indentation-based detection after a `remora:` header
      The chosen convention must produce valid Python after
      transformation so that linters and formatters don't choke on
      non-transformed source.

- [x] **1.5 — Source transformation**: The codec reads the file,
      identifies Remora blocks, compiles each via the Remora pipeline,
      and replaces them with Python source that registers
      `RemoraFunction` wrappers.  The rest of the file passes through
      unchanged.

- [x] **1.6 — Import hooks**: Register an `importlib` finder/loader so
      that `import my_remora_module` works for `.py` files with
      `# coding: remora`.  The loader applies the codec transformation
      at import time.

- [x] **1.7 — Caching**: Compiled native artifacts (`.so` files) should
      be cached on disk (e.g., in `__pycache__/` or a `.remora_cache/`
      directory) keyed by source hash, so recompilation only happens
      when the Remora source changes.

- [x] **1.8 — Error reporting**: Parse errors, type errors, and
      lowering errors from the Remora compiler should produce clear
      Python exceptions with source locations pointing into the
      original Remora source, not into generated code.

- [x] **1 tests** — End-to-end tests: write a `.py` file with
      `# coding: remora`, import it from another Python script, call
      Remora functions with NumPy arrays, verify results.

## Phase 2 — Jupyter Cell Magic (`%%remora`)

Uses the `RemoraFunction` wrapper from Phase 1.  Users write Remora
in one cell, call compiled functions from Python in the next.

- [x] **2.1 — Extend `remora/jupyter/magics.py`**: Add a `%%remora`
      cell magic that receives raw cell text (S-expression source),
      compiles it, and registers callable wrappers into the notebook's
      `user_ns` global namespace.

- [x] **2.2 — Multi-definition cells**: A single `%%remora` cell may
      contain multiple `define` / `define/pi` forms.  Each definition
      becomes a separate `RemoraFunction` registered in the namespace.
      Values (non-function definitions) are evaluated and registered as
      NumPy arrays.

- [x] **2.3 — Target selection**: Support `%%remora --target cpu` and
      `%%remora --target interp` (default: cpu).

- [x] **2.4 — Error reporting**: Compiler errors should display
      clearly in the notebook output with source locations pointing
      into the cell text.

- [x] **2 tests** — Notebook-style tests using `IPython.testing` or
      `pytest-jupyter`.  Verify: define a function in a `%%remora`
      cell, call it from Python with NumPy arrays, check results.

## Phase 3 — Developer Experience

- [x] **3.1 — `remora.define()` Python API**: For users who prefer not
      to use the codec or magic, provide a programmatic API:
      `fn = remora.define("(define/pi () (f [x ...] ...) ...)")`.
      Returns a `RemoraFunction` directly.

- [x] **3.2 — REPL integration**: The existing `remora` REPL
      (`remora/repl.py`) should be usable from IPython via
      `%remora_eval` line magic or by importing `remora.repl`.

- [x] **3.3 — Inline type display**: In Jupyter, `%%remora --types`
      prints the inferred types of all definitions alongside the
      compilation output.

- [x] **3 tests** — API tests for `remora.define()`, REPL integration.

## Future Work

### GPU Target in Cell Magic and Codec

`%%remora --target gpu` and GPU compilation from `.py` files.
Compile to PTX, launch via `RemoraExecutor`, return results as
NumPy arrays (device→host copy).

### PyTorch Tensor Interop

Accept `torch.Tensor` inputs in `RemoraFunction.__call__`.  For CPU
tensors, extract the data pointer via `tensor.data_ptr()`.  For CUDA
tensors, pass the device pointer directly to GPU kernels (true
zero-copy).

### PyTorch Autograd Integration

Register Remora's AD gradient functions as custom
`torch.autograd.Function` backward passes, enabling Remora functions
to participate in PyTorch computation graphs.  This is a substantial
feature that deserves its own design work.

## Existing Infrastructure to Reuse

| Component | Location | What it provides |
|-----------|----------|-----------------|
| Lisp reader | `remora/lisp_reader.py` | S-expression parsing → AST |
| Type checker | `remora/typechecker.py` | Static rank/shape deduction |
| Compiler | `remora/compiler.py` | AST → MLIR → native code |
| CPU executor | `remora/runtime.py` | `.so` loading, ctypes dispatch |
| GPU executor | `remora/executor.py` | PTX loading, CUDA launch |
| Descriptor ABI | `remora/abi.py` | Pointer + offset + sizes + strides structs |
| Jupyter magics | `remora/jupyter/magics.py` | Existing magic infrastructure |
| AD | `remora/ad.py`, `remora/ad_source.py` | Gradient computation |

## Design Constraints

- **No string wrapping for DSL code.** Remora source is written as
  first-class syntax, not inside Python string literals.
- **Preserve Remora semantics.** Implicit lifting, explicit reranking,
  and rank-polymorphic function application work exactly as in
  standalone `.remora` / `.lisp` files.
- **Zero-copy data interchange.** NumPy arrays are passed by pointer,
  not copied.  The descriptor ABI already supports this layout.
- **Existing backends.** No new compilation backend.  The MLIR CPU
  pipeline and LLVM GPU pipeline are the execution engines.
