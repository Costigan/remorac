# Remora GPU Compiler: Detailed Implementation Plan

## 1. Goals and Deliverables

This document is a concrete engineering plan for building a Remora compiler. The long-term deliverables are:

1. **`remorac`** — an ahead-of-time (AOT) compiler that takes a Remora source file, compiles it to a GPU binary, and runs it, producing a result.
2. **`remora`** — an interactive REPL using the exact same compilation pipeline, initially by recompiling a self-contained temporary program for each expression. Incremental compilation is an optimization for later, not a prerequisite for the first REPL.

Both tools share the same parser, type checker, lowering, MLIR pass pipeline, execution engine, and result display. The first prototype still proves one end-to-end compiler path, but it should expose that path through a CPU-first REPL as soon as parser/typechecker/lowering/runtime pieces are usable. The REPL maintains accumulated source definitions in session state and recompiles the current expression with those definitions; it does not require incremental MLIR module linking.

The compilation backend is MLIR via the `mlir-python-bindings` package, targeting NVIDIA CUDA via the PTX path (with AMD ROCm as a secondary target). See `MLIR_ARCHITECTURE.md` for the full rationale and comparison.

### 1.1 First Prototype Contract

The first implementation is **Remora Dense Core**, not the full language surface described in the research references. It proves dense rectangular rank-polymorphic array execution before adding ragged arrays, boxes, runtime shape polymorphism, or fully dynamic higher-order values.

The first cut must prove one end-to-end path before broadening scope:

1. Parse, type-check, lower, compile, and execute `iota`, unary `map`, and scalar `fold`.
2. Support dense rank-0, rank-1, rank-2, and rank-3 arrays with static rank, static dimensions, and static element types.
3. Verify every program on the CPU lowering path before enabling GPU execution.
4. Target NVIDIA CUDA only.
5. Use one stable kernel ABI: Remora external kernels receive pointers to rank-specialized **memref descriptor structs**, and the Python runtime constructs and passes those descriptors.
6. Generate PTX and launch it manually through `cuda-python`; do not depend on MLIR-generated host `gpu.launch_func` execution in the first prototype.
7. Provide a non-incremental CPU-first `remora` REPL once the CPU execution path is available. GPU REPL support reuses the same interface after the NVIDIA ABI/runtime is stable.

Incremental REPL compilation, a broad standard library, AMD lowering, dynamic shapes, dynamic higher-order functions, and performance tiling are follow-on work. They should not block the first executable compiler or the first interactive shell.

### 1.2 Remora Dense Core Architecture Contracts

These contracts are binding for the first implementation. If a later phase changes one, it must update tests and ABI documentation at the same time.

#### Dense Arrays and Static Shapes

Remora Dense Core supports only dense rectangular arrays. Every array rank and every dimension extent must be known at compile time after constant folding. Shape-producing expressions may exist for inspection, but they do not introduce runtime-dependent dimensions in the first cut.

Allowed:

```remora
iota 10
map (* 2.0) [1.0, 2.0, 3.0]
map (\row -> fold (+) 0.0 row) [[1.0, 2.0], [3.0, 4.0]]
```

Deferred:

```remora
def f n = iota n       -- rejected unless n is a compile-time constant
```

Rank-0 through rank-3 arrays are in scope. Rank-4+ should be rejected with a clear "rank limit exceeded in Dense Core" diagnostic until the lowering and ABI tests are extended.

#### Numeric Policy

`iota n` returns `int[n]`. Arithmetic supports `int`, `float`, and explicit result typing:

- `int op int -> int` for `+`, `-`, `*`
- `float op float -> float`
- mixed `int`/`float` arithmetic promotes the `int` operand to `float`
- `/` always returns `float`
- comparisons return `bool`

The type checker inserts explicit typed cast nodes for promotions so HIR and MLIR lowering never guess.

#### Static Function Values

The prototype supports static function values: named functions, lambdas, operator functions, and operator sections may be passed to `map`, `fold`, and directly called higher-order helpers when the concrete callable is known at compile time. These are lambda-lifted and/or monomorphized before MLIR lowering.

The prototype does not support dynamic function values: returning closures, storing functions in arrays, choosing functions through runtime conditionals, closure structs with runtime tags, or device-side indirect calls. These cases must produce a diagnostic that says dynamic higher-order functions are deferred.

#### Views and Strides

The runtime array model is view-capable from day one. A value may describe a non-contiguous view by carrying `offset`, `sizes`, and `strides`, even though the first lowering only needs to allocate contiguous row-major arrays. This prevents transpose, slicing, and subarray support from requiring an ABI replacement later.

Contiguous row-major rank-3 shape `[d0, d1, d2]` has strides `[d1*d2, d2, 1]`. Transpose/slice operations are deferred as language features, but the descriptor representation must already be able to express them.

#### MLIR Strategy

Do not build a custom Remora MLIR dialect in the first implementation. Preserve Remora-specific semantics in a strongly typed Python HIR through type checking, shape resolution, function-value analysis, and view normalization. Lower to standard MLIR `tensor`, `linalg`, `arith`, `math`, `memref`, `func`, `scf`, `gpu`, and LLVM/NVVM dialects only after those invariants are explicit.

This keeps the GPU backend reachable quickly while avoiding early erasure of frame/cell semantics inside the Python compiler.

---

## 2. Language Subset (Prototype Scope)

### 2.1 What Is Implemented

| Feature | Notes |
|---|---|
| Scalar types: `float`, `int`, `bool` | `f32`, `i32`, `i1` in MLIR |
| Rank-0 through rank-3 dense arrays with static shapes | `tensor<f32>`, `tensor<d0xf32>`, `tensor<d0xd1xf32>`, and `tensor<d0xd1xd2xf32>` |
| Lifted application (`map f arr`) | Unary static callable over scalar, vector, matrix, or rank-3 dense arrays |
| Reduction (`fold f init arr`) | Scalar accumulator over the outermost dimension of rank-1 through rank-3 arrays |
| Lambda expressions | Static lambdas for `map`, `fold`, and monomorphized direct higher-order calls |
| `let` bindings | Lexically scoped |
| Top-level function definitions | Named functions; static function values allowed when resolved at compile time |
| Built-in `iota n` | `[0, 1, ..., n-1]` |
| Arithmetic and comparison | `+`, `-`, `*`, `/`, comparisons, and explicit typed casts inserted for int/float promotion |

### 2.2 What Is Deferred (Out of Scope for Prototype)

- Incremental REPL compilation and long-session module caching
- Standard library beyond compiler built-ins
- Rank-4+ array programs
- Transpose, slicing, and matrix-specific operators as surface-language features
- Dynamic shapes and dynamic dimensions
- Dynamic rank (rank not known at compile time)
- Dynamic first-class function values, closure structs, arrays of functions, and dynamic HOF dispatch
- Ragged arrays, boxes, dependent sums, and hidden-shape arrays
- AMD ROCm target
- MLIR-generated host-side GPU launch flow
- Mutable state / in-place updates
- File I/O (only stdout print)
- Strings as first-class values
- Recursive data structures
- Exceptions / error handling at the Remora level

---

## 3. Implementation Language and Tool Choices

| Component | Choice | Rationale |
|---|---|---|
| Compiler host language | **Python 3.11+** | mlir-python-bindings are Python-native; fast prototyping |
| Parser | **Lark** (LALR or Earley) | Clean EBNF grammar syntax, good error messages |
| MLIR API | **mlir-python-bindings** (from `mlir` pip package built with LLVM) | Official Python API for MLIR construction and pass management |
| CUDA execution | **`cuda-python`** package | Official NVIDIA Python CUDA driver bindings |
| Testing | **pytest** | Standard; plays well with Python project |
| Numeric arrays (host-side) | **numpy** | Universal Python array library; used for result marshaling |
| LLVM version | **LLVM/MLIR 18** (pin to a release) | Stable; mlir-python-bindings packages available |
| CUDA version | **CUDA 12+** | Required for modern PTX features |

### 3.1 Why Not Mojo (Yet)

As of 2026-05-29, Mojo is a much stronger GPU option than it was in early planning. The Mojo documentation now covers GPU kernels, host/device memory management, `DeviceContext` launch APIs, NVIDIA/AMD/Apple GPU support, warp/block synchronization, warp-level reductions, and low-level GPU intrinsics. The reason to avoid Mojo for the first prototype is therefore **not** that Mojo lacks basic GPU programming capability.

The first prototype still uses Python + direct MLIR bindings because this plan needs exact control over:

1. Emitting `tensor`/`linalg`/`gpu` MLIR for Remora's rank-lifting semantics.
2. Validating `linalg.generic` fusion and bufferization behavior pass by pass.
3. Pinning an LLVM/MLIR 18 lowering pipeline and inspecting the IR at every stage.
4. Defining a stable runtime ABI around MLIR-lowered memref descriptors.

Using Mojo as the implementation language or as a source-code target would add another compiler layer between Remora and the MLIR being tested. That is valuable later, but it makes the first vertical slice harder to debug: failures could come from Remora lowering, generated Mojo code, Mojo compiler lowering, or the GPU runtime. Mojo is also still pre-1.0, source stability is not guaranteed, and cross-compilation/linking support remains a work in progress.

Revisit Mojo after M9/M10, when the direct MLIR path has proven the Remora semantics, ABI, AOT path, and fusion expectations. At that point, Mojo should be evaluated in two concrete roles:

- **Runtime/kernel host**: replace Python CUDA runtime code with Mojo `DeviceContext` and typed buffer management.
- **Compilation target**: generate Mojo for selected kernels only if it can preserve the required linalg/fusion control or produce equivalent verified IR.

### 3.2 Installing `mlir-python-bindings`

The MLIR Python package is not on PyPI; it must be built from source or obtained from an LLVM nightly release artifact:

```bash
# Option A: Build from LLVM source (slow but exact)
cmake -DLLVM_ENABLE_PROJECTS="mlir" \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
      ../llvm-project/llvm
make -j$(nproc) mlir-python-bindings

# Option B: Use pre-built wheels from LLVM nightly (faster)
pip install mlir-python-bindings --find-links https://github.com/llvm/llvm-project/releases/...

# Option C: Use the mlir package bundled with iree-compiler (quickest for getting started)
pip install iree-compiler  # pulls in mlir as a dependency
```

### 3.3 Remora Language References

The Remora references needed by implementation agents are checked into `remora-reference/` with both PDFs and searchable `.txt` sidecars. Start with `remora-reference/README.md`, which maps references to implementation tasks.

| Reference | Local copy | Source URL | Implementation use |
|---|---|---|---|
| Introduction to Rank-polymorphic Programming in Remora | `remora-reference/remora-tutorial-draft.pdf` and `.txt` | https://www.ccs.neu.edu/home/shivers/papers/remora-tutorial-draft.pdf | User-facing language model, examples, arrays, frames, cells, rank lifting, and shape consistency |
| Introduction to Rank-polymorphic Programming in Remora (arXiv entry) | `remora-reference/intro-rank-polymorphic-programming-remora.pdf` and `.txt` | https://arxiv.org/abs/1912.13451 | Citation metadata and alternate copy of the tutorial material |
| The Semantics of Rank Polymorphism | `remora-reference/semantics-of-rank-polymorphism.pdf` and `.txt` | https://arxiv.org/abs/1907.00509 | Formal typing and dynamic semantics for frame/cell decomposition, lifting, shape soundness, and type-driven execution |
| A Typed Programming Language for Rank-Polymorphic Array Processing | `remora-reference/slepak-dissertation.pdf` and `.txt` | https://ccs.neu.edu/~jrslepak/Dissertation.pdf | Deep reference for type inference, bidirectional typing, shape constraints, type erasure, and translation toward explicit iteration |

Use these references by phase:

- **Phases 1-2**: tutorial first, then semantics paper for precise parser/type rules.
- **Phases 3-5**: semantics paper for frame/cell lowering; dissertation for type-erasure-to-iteration guidance.
- **Phases 9-10**: tutorial examples and dissertation base-environment material for early REPL and standard library behavior.

---

## 4. Repository Structure

```
remora-gpu/
├── remora/                      # Main Python package
│   ├── __init__.py
│   ├── ast_nodes.py             # AST dataclass definitions
│   ├── grammar.lark             # Lark grammar for Remora
│   ├── parser.py                # Parser: source text → AST
│   ├── types.py                 # Type system: RemoraType, ArrayType, etc.
│   ├── typechecker.py           # Type inference, shape resolution, rank lifting
│   ├── hir.py                   # High-level IR (between typed AST and MLIR)
│   ├── defunc.py                # Defunctionalization pass
│   ├── lowering.py              # HIR → MLIR linalg.generic lowering
│   ├── pipeline.py              # MLIR pass pipeline: CPU and GPU configurations
│   ├── codegen.py               # PTX generation from lowered GPU MLIR module
│   ├── runtime.py               # CUDA driver API wrappers (device alloc, launch)
│   ├── executor.py              # High-level execution engine (AOT and JIT)
│   ├── repl.py                  # REPL: state management, loop, commands
│   ├── display.py               # Result formatting for REPL output
│   └── errors.py                # RemoraError, SourceLocation, pretty diagnostics
│
├── bin/
│   ├── remorac                  # AOT compiler entry point script
│   └── remora                   # REPL entry point script
│
├── stdlib/
│   ├── prelude.rem              # Core Dense Core helpers: sum, scale, etc.
│   └── linalg.rem               # Deferred linear algebra: dot, matmul, norm, transpose
│
├── tests/
│   ├── conftest.py              # Shared fixtures (MLIRContext, CUDARuntime)
│   ├── test_parser.py           # Parser unit tests
│   ├── test_typechecker.py      # Type checker and shape inference tests
│   ├── test_defunc.py           # Defunctionalization tests
│   ├── test_lowering.py         # HIR → MLIR lowering tests
│   ├── test_pipeline.py         # MLIR pass pipeline tests
│   ├── test_abi.py              # Rank-0..3 descriptor ABI tests
│   ├── test_execution.py        # End-to-end GPU execution tests
│   ├── test_repl.py             # REPL interaction tests
│   ├── golden_mlir/             # Checked MLIR fixtures for builder/pipeline validation
│   │   ├── iota_rank1.mlir
│   │   ├── map_rank1.mlir
│   │   ├── map_rank2.mlir
│   │   ├── map_rank3.mlir
│   │   └── fold_rank1.mlir
│   └── programs/                # .rem source files for end-to-end tests
│       ├── scalar_add.rem
│       ├── vector_scale.rem
│       ├── matrix_scale.rem
│       ├── tensor3_scale.rem
│       ├── vector_cell_map.rem
│       ├── nested_map.rem       # Map chain fusion verification
│       ├── hof_param.rem        # Function as parameter
│       ├── large_vector.rem     # Performance test (10M elements)
│       └── deferred/
│           ├── dot_product.rem
│           └── matmul.rem
│
├── pyproject.toml
├── docs/
│   └── ABI.md                   # Remora external kernel ABI, rank-0..3 descriptors
└── README.md
```

---

## 5. Architecture Overview

The two tools share a single compilation pipeline that diverges only at execution time:

```
                         ┌─────────────────────────────────────────────┐
                         │            Shared Compilation Pipeline        │
                         │                                               │
  source.rem             │  Parser → Typed AST → HIR → MLIR Linalg     │
  ──────────► parse ─────►  → MLIR Pass Pipeline → PTX + KernelMeta    │
                         │                                               │
                         └──────────────────┬──────────────────────────┘
                                            │
                          ┌─────────────────┴─────────────────┐
                          │                                     │
                   AOT mode (`remorac`)        Interactive mode (`remora` REPL)
                          │                                     │
                   Run main() kernel              Compile one expression at a time
                   Print result                  Marshal result → display
                   Exit                          Loop back for next input
```

### 5.1 AOT Compilation Flow

```
Remora source file
        │
        ▼  [parser.py]
    Remora AST
        │
        ▼  [typechecker.py]
    Typed AST
    (every node carries its RemoraType;
     all ranks and frame shapes resolved)
        │
        ▼  [defunc.py]
    Defunc'd AST
    (no function values remain;
     each HOF call site is monomorphized
     or carries a closure struct + dispatch)
        │
        ▼  [hir.py + lowering.py]
    MLIR Module
    (func + linalg + tensor dialects)
        │
        ▼  [pipeline.py — linalg fusion pass]
    MLIR Module
    (linalg chains fused; fewer generics)
        │
        ▼  [pipeline.py — bufferization]
    MLIR Module
    (memref dialect; alloc/dealloc inserted)
        │
        ▼  [pipeline.py — GPU mapping]
    MLIR Module
    (device kernel outlined; host launch path discarded for prototype)
        │
        ▼  [pipeline.py — NVVM lowering]
    MLIR Module
    (nvvm + llvm dialects)
        │
        ▼  [codegen.py — LLVM NVPTX backend]
    PTX text  +  KernelMeta list
        │
        ▼  [executor.py]
    Load PTX via CUDA driver
    Allocate device buffers
    Build memref descriptors matching the lowered MLIR ABI
    Launch kernel
    Copy result to host
    Display result
```

The prototype uses the generated **device kernel** as the source of truth and launches it manually through CUDA. Any MLIR-generated host wrapper or `gpu.launch_func` flow is an implementation detail used only to outline and lower kernels; it is not the runtime ABI.

### 5.2 Non-Incremental REPL Compilation Flow

The first REPL compiles each expression as a self-contained program by wrapping it in a `main()` function that includes all definitions accumulated so far. This makes the REPL available early without incremental MLIR module linking, GPU module relinking, or cross-input kernel caching.

The initial target is CPU-only, because it can reuse parser, type checker, HIR, MLIR lowering, CPU pipeline, executor, diagnostics, and result formatting before the CUDA descriptor ABI is stable. Once the NVIDIA runtime path is complete, the same REPL interface can add `--target gpu-nvidia`.

```
REPL input (text)
        │
        ├─ Is it a definition?
        │       │
        │       ▼
        │   Parse + type-check + defunc the definition
        │   Add to accumulated definition list (no execution)
        │   Print "Defined: name"
        │
        └─ Is it an expression?
                │
                ▼
            Parse + type-check the expression against accumulated env
            Wrap in main(): definitions + expression as final stmt
            Run full compilation pipeline for selected target
            Execute on CPU, or launch GPU kernel and copy result to host
            Display result
            Loop
```

---

## 6. Phase-by-Phase Implementation Plan

### Phase 0: Infrastructure and Hello World

**Goal**: Prove that the toolchain works. Generate a trivial GPU kernel via MLIR Python bindings, execute it on the GPU, retrieve the result.

**Tasks**:

0.1. Create Python project structure (`pyproject.toml`, `remora/` package, `tests/`, `bin/`)

0.2. Set up virtual environment; install:
```bash
pip install lark cuda-python numpy pytest
# plus mlir package (see section 3.2)
```

0.3. Write `tests/test_infra.py`:
- Test that `from mlir.ir import Context` succeeds
- Test that `from cuda import cuda` succeeds
- Test that `import numpy` succeeds

0.4. Write `hello_mlir.py` (standalone script, not part of package):
- Create `MLIRContext`
- Build a `func.func` with a scalar `f32` add using `arith.addf`
- Apply `-convert-arith-to-llvm`, `-convert-func-to-llvm` passes
- Execute via `mlir.execution_engine.ExecutionEngine`
- Verify result is correct

0.5. Write `hello_linalg.py`:
- Build a `linalg.generic` that doubles each element of a `tensor<10xf32>`
- Lower to CPU via the CPU pipeline (see Phase 7)
- Execute and verify the output array

0.6. Write `hello_gpu.py`:
- Same `linalg.generic`, but route through the GPU pipeline
- Extract the PTX text output
- Use `cuda-python` to load PTX as a CUDA module
- Launch the kernel; copy result back; verify

**Milestone M0**: `python hello_gpu.py` completes with correct output on a GPU machine.

---

### Phase 1: Parser and AST

**Goal**: Parse any Remora program in the defined subset and produce a clean AST.

#### 1.1 Grammar (`remora/grammar.lark`)

```lark
// Top-level
program    : def_* expr?
definition : def_
expression : expr

def_       : "def" NAME params "=" expr   -> func_def
           | "def" NAME "=" expr           -> val_def

params     : NAME+

// Expressions (precedence via grammar levels)
expr       : let_expr | if_expr | lambda_expr | compose_expr

let_expr   : "let" NAME "=" expr "in" expr
if_expr    : "if" expr "then" expr "else" expr
lambda_expr: ("\\" | "λ") NAME+ ("->" | "→") expr
compose_expr: compose_expr "∘" app_expr  -> compose
           | app_expr

app_expr   : app_expr atom+              -> application
           | "map"  callable atom        -> map_expr
           | "fold" atom atom atom       -> fold_expr
           | "iota" atom                 -> iota_expr
           | "shape" atom               -> shape_expr
           | "rank"  atom               -> rank_expr
           | atom

callable   : atom
           | "(" infix_op ")"            -> operator_func
           | "(" infix_op atom ")"       -> left_section
           | "(" atom infix_op ")"       -> right_section

infix_op   : "+" | "-" | "*" | "/" | "<" | "<=" | "==" | "!=" | "&&" | "||"

atom       : "(" expr ")"               -> paren
           | "[" (expr ("," expr)*)? "]" -> array_lit
           | FLOAT                       -> float_lit
           | INT                         -> int_lit
           | BOOL                        -> bool_lit
           | NAME                        -> var
           | atom "[" expr ("," expr)* "]" -> index_expr

// Infix ops (handled via app_expr application of built-in op names,
// or via a separate infix layer — see implementation note)
infix_expr : ...  // standard precedence climbing for +, *, -, /, <, <=, ==, &&, ||

BOOL       : "true" | "false"
INT        : /-?[0-9]+/
FLOAT      : /-?[0-9]+\.[0-9]*/
NAME       : /[a-zA-Z_][a-zA-Z0-9_']*/

%ignore /\s+/
%ignore /--[^\n]*/   // line comments
```

**Implementation note**: Standard infix operators (`+`, `*`, etc.) are parsed through a precedence-climbing layer or a Pratt parser. Operator sections such as `(*)`, `(* 2.0)`, and `(2.0 *)` are explicit grammar forms so examples like `map (* 2.0) xs` are valid unary maps, not accidental binary `map` syntax. Binary zipping maps such as `map (*) a b` are out of scope until `zip` or multi-input `linalg.generic` lowering is implemented.

The parser exposes separate entry points:

- `parse_program`: accepts zero or more definitions followed by an optional final expression for AOT files.
- `parse_definition`: accepts exactly one definition for REPL/file loading.
- `parse_expr`: accepts exactly one expression.

The REPL must use `parse_definition` first and then `parse_expr`; it must not rely on a `start: def_* expr` grammar that makes definition-only input invalid.

#### 1.2 AST Node Definitions (`remora/ast_nodes.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class SourceLoc:
    file: str
    line: int
    col: int

@dataclass
class Program:
    definitions: list[FuncDef | ValDef]
    body: Expr
    loc: SourceLoc

@dataclass
class FuncDef:
    name: str
    params: list[str]
    body: Expr
    loc: SourceLoc

@dataclass
class ValDef:
    name: str
    value: Expr
    loc: SourceLoc

@dataclass
class LetExpr:
    name: str
    value: Expr
    body: Expr
    loc: SourceLoc

@dataclass
class IfExpr:
    condition: Expr
    then_branch: Expr
    else_branch: Expr
    loc: SourceLoc

@dataclass
class LambdaExpr:
    params: list[str]
    body: Expr
    loc: SourceLoc

@dataclass
class AppExpr:
    func: Expr
    args: list[Expr]
    loc: SourceLoc

@dataclass
class MapExpr:
    func: Expr
    array: Expr
    loc: SourceLoc

@dataclass
class FoldExpr:
    func: Expr
    init: Expr
    array: Expr
    loc: SourceLoc

@dataclass
class IoTaExpr:
    size: Expr
    loc: SourceLoc

@dataclass
class ShapeExpr:
    array: Expr
    loc: SourceLoc

@dataclass
class VarExpr:
    name: str
    loc: SourceLoc

@dataclass
class IntLit:
    value: int
    loc: SourceLoc

@dataclass
class FloatLit:
    value: float
    loc: SourceLoc

@dataclass
class BoolLit:
    value: bool
    loc: SourceLoc

@dataclass
class ArrayLit:
    elements: list[Expr]
    loc: SourceLoc

@dataclass
class IndexExpr:
    array: Expr
    indices: list[Expr]
    loc: SourceLoc

Expr = (LetExpr | IfExpr | LambdaExpr | AppExpr | MapExpr | FoldExpr |
        IoTaExpr | ShapeExpr | VarExpr | IntLit | FloatLit | BoolLit |
        ArrayLit | IndexExpr)
```

#### 1.3 Parser Implementation (`remora/parser.py`)

```python
from lark import Lark, Transformer
from pathlib import Path
from .ast_nodes import *

_GRAMMAR = (Path(__file__).parent / "grammar.lark").read_text()

class ASTBuilder(Transformer):
    """Lark transformer: parse tree → AST nodes."""
    
    def program(self, items):    return Program(_defs(items), _optional_body(items), ...)
    def func_def(self, items):   return FuncDef(str(items[0]), list(items[1:-1]), items[-1], ...)
    def let_expr(self, items):   return LetExpr(str(items[0]), items[1], items[2], ...)
    def lambda_expr(self, items):return LambdaExpr([str(p) for p in items[:-1]], items[-1], ...)
    def map_expr(self, items):   return MapExpr(items[0], items[1], ...)
    def fold_expr(self, items):  return FoldExpr(items[0], items[1], items[2], ...)
    def application(self, items):
        if len(items) == 1: return items[0]
        return AppExpr(items[0], items[1:], ...)
    def var(self, items):        return VarExpr(str(items[0]), ...)
    def float_lit(self, items):  return FloatLit(float(items[0]), ...)
    def int_lit(self, items):    return IntLit(int(items[0]), ...)
    def bool_lit(self, items):   return BoolLit(items[0] == "true", ...)
    def array_lit(self, items):  return ArrayLit(list(items), ...)
    # ... etc.

_parser = Lark(_GRAMMAR, parser="earley", ambiguity="resolve")

def parse(source: str, filename: str = "<input>") -> Program:
    tree = _parser.parse(source)
    return ASTBuilder().transform(tree)

def parse_file(path: str) -> Program:
    source = Path(path).read_text()
    return parse(source, filename=path)

def parse_repl_input(text: str) -> FuncDef | ValDef | Expr:
    """Parse a single REPL line: either a definition or an expression."""
    # Try parsing as exactly one definition first; fall back to exactly one expression.
    # Do not wrap expression input in a fake definition.
    ...
```

#### 1.4 Tasks

- [x] Write `grammar.lark` covering all constructs in section 2.1
- [x] Write `ASTBuilder` transformer for all grammar rules
- [x] Implement `parse()` and `parse_file()`
- [x] Implement `parse_repl_input()` with definition/expression discrimination
- [x] Write `tests/test_parser.py`:
  - Integer and float literals
  - Array literals
  - Lambda with multiple parameters
  - `let` binding
  - `map` expression
  - `fold` expression
  - `iota`
  - Function application (single and curried)
  - Top-level function definition
  - Nested expressions
  - Infix operators
  - `if then else`
  - Error cases (malformed syntax)

**Milestone M1**: `parse("let x = map (\\ a -> a + 1.0) (iota 10) in fold (+) 0.0 x")` returns a correct AST.

---

### Phase 2: Type System and Shape Inference

**Goal**: Determine the type and shape of every expression, resolving Remora's rank-polymorphic lifting.

#### 2.1 Type Representation (`remora/types.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union

@dataclass(frozen=True)
class StaticDim:
    value: int
    def __str__(self): return str(self.value)

DimExpr = StaticDim

@dataclass(frozen=True)
class ScalarType:
    name: str     # "float", "int", "bool"

@dataclass(frozen=True)
class ArrayType:
    element: RemoraType
    shape: tuple[DimExpr, ...]    # outermost first

    @property
    def rank(self) -> int:
        return len(self.shape)

    def with_frame(self, frame: tuple[DimExpr, ...]) -> 'ArrayType':
        """Prepend frame dimensions to this array type."""
        return ArrayType(self.element, frame + self.shape)

    def drop_outer(self, n: int) -> RemoraType:
        """Remove n outermost dimensions. Returns ScalarType if rank drops to 0."""
        if self.rank - n == 0:
            return self.element
        return ArrayType(self.element, self.shape[n:])

@dataclass(frozen=True)
class FuncType:
    params: tuple[RemoraType, ...]
    result: RemoraType

RemoraType = Union[ScalarType, ArrayType, FuncType]

FLOAT = ScalarType("float")
INT   = ScalarType("int")
BOOL  = ScalarType("bool")
```

Dense Core intentionally has no dynamic dimension type. Every dimension must be a non-negative integer known after compile-time constant folding. If a later phase adds symbolic or runtime dimensions, it must introduce a real shape-expression and constraint-solving layer rather than smuggling dynamic extents into `StaticDim`.

```python
def eval_static_dim(expr: Expr, env: ConstEnv, loc: SourceLoc) -> StaticDim:
    """
    Evaluate a shape expression at compile time.
    Raises RemoraTypeError if the result is not a non-negative integer constant.
    """
    ...
```

`iota n`, array literal shapes, reshape-like future operations, and rank-limited lowering all use `eval_static_dim`.

#### 2.2 Rank Lifting Rule

The central shape-inference rule for Remora:

Given:
- `f : cell_type → result_cell_type`
- `arr : T` where `T` is an `ArrayType` whose innermost dimensions match `cell_type`

Then:
- Frame shape = `arr.shape[:arr.rank - cell_rank]`
- `map f arr : result_cell_type.with_frame(frame_shape)`

Dense Core caps every expression result at rank 3. Any `map`, literal, or future shape operation whose result rank would exceed 3 is rejected until the ABI and lowering tests are extended.

In code:
```python
def infer_lifting(f_type: FuncType, arr_type: RemoraType) -> tuple[tuple[DimExpr,...], RemoraType]:
    """
    Returns (frame_shape, result_type) for map f arr.
    Raises TypeError if f's input type is incompatible with arr's element structure.
    """
    cell_type = f_type.params[0]
    cell_rank = cell_type.rank if isinstance(cell_type, ArrayType) else 0
    
    if isinstance(arr_type, ScalarType):
        if cell_rank != 0:
            raise TypeError(f"Function expects {cell_rank}D cells but got scalar array")
        frame_shape = ()
    else:
        total_rank = arr_type.rank
        if total_rank < cell_rank:
            raise TypeError(f"Array rank {total_rank} too low for cell rank {cell_rank}")
        frame_shape = arr_type.shape[:total_rank - cell_rank]
    
    result_cell = f_type.result
    if frame_shape:
        result_type = ArrayType(
            result_cell.element if isinstance(result_cell, ArrayType) else result_cell,
            frame_shape + (result_cell.shape if isinstance(result_cell, ArrayType) else ()))
    else:
        result_type = result_cell
    
    return frame_shape, result_type
```

#### 2.3 Bidirectional Type Checking Strategy

Use bidirectional typing for lambdas, operator sections, and higher-order operations. Do not infer a standalone polymorphic type for every lambda before seeing its use site.

For `map f arr`:

1. Infer `arr`.
2. Choose the expected cell type from an annotation or known callable type when available. Otherwise, Dense Core defaults unary scalar maps to the array element scalar type.
3. Check `f` against `FuncType((cell_type,), result_cell_type)` using the array cell type as expected input.
4. Compute the frame shape and result type.

For `fold f init arr`:

1. Infer `arr` and `init`.
2. The element type is `arr.drop_outer(1)`.
3. Check `f` against `FuncType((init_type, elem_type), init_type)`.

Operator sections are checked against expected function types. For example, in `map (* 2.0) (iota 10)`, `iota 10 : int[10]`, the expected input is `int`, `2.0` forces mixed numeric promotion, and the section type becomes `int -> float` with an inserted cast.

#### 2.4 Numeric Promotion and Cast Nodes

The type checker inserts explicit typed casts for numeric promotion:

```python
@dataclass
class TypedCast:
    value: TypedExpr
    from_type: ScalarType
    to_type: ScalarType
    type: ScalarType
    loc: SourceLoc
```

HIR lowering preserves these as `HIRCast`, and MLIR lowering emits `arith.sitofp`, `arith.index_cast`, or other explicit conversion operations. Lowering passes must not infer numeric promotions implicitly.

#### 2.5 Type Checker (`remora/typechecker.py`)

```python
class TypeEnv:
    """Immutable mapping from names to types. Supports extension."""
    def __init__(self, bindings: dict[str, RemoraType] | None = None):
        self._b = bindings or {}
    
    def extend(self, name: str, ty: RemoraType) -> 'TypeEnv':
        return TypeEnv({**self._b, name: ty})
    
    def lookup(self, name: str, loc: SourceLoc) -> RemoraType:
        if name not in self._b:
            raise RemoraTypeError(f"Unbound variable '{name}'", loc)
        return self._b[name]

class TypeChecker:
    def check(self, prog: Program) -> TypedProgram:
        env = self._build_prelude_env()
        for defn in prog.definitions:
            env = self._add_definition(defn, env)
        typed_body, body_ty = self.infer(prog.body, env)
        return TypedProgram(prog.definitions, typed_body, body_ty)
    
    def infer(self, expr: Expr, env: TypeEnv) -> tuple[TypedExpr, RemoraType]:
        match expr:
            case IntLit(v): 
                return TypedIntLit(v, INT), INT
            
            case FloatLit(v): 
                return TypedFloatLit(v, FLOAT), FLOAT
            
            case VarExpr(name, loc):
                ty = env.lookup(name, loc)
                return TypedVarExpr(name, ty), ty
            
            case ArrayLit(elems, loc):
                typed_elems = []
                elem_ty = None
                for e in elems:
                    te, ty = self.infer(e, env)
                    if elem_ty is None: elem_ty = ty
                    elif ty != elem_ty: raise RemoraTypeError("Inconsistent element types or shapes", loc)
                    typed_elems.append(te)
                arr_ty = ArrayType(elem_ty, (StaticDim(len(elems)),))
                return TypedArrayLit(typed_elems, arr_ty), arr_ty
            
            case LambdaExpr(params, body, loc):
                # Lambdas in expression position: types of params must be inferrable
                # Use fresh type variables; unify on body inference
                param_tyvars = [fresh_tyvar(p) for p in params]
                inner_env = env
                for name, tv in zip(params, param_tyvars):
                    inner_env = inner_env.extend(name, tv)
                typed_body, body_ty = self.infer(body, inner_env)
                param_tys = [self.resolve(tv) for tv in param_tyvars]
                fn_ty = FuncType(tuple(param_tys), body_ty)
                return TypedLambda(params, typed_body, fn_ty), fn_ty
            
            case MapExpr(f_expr, arr_expr, loc):
                # Infer the array first so lambdas and operator sections can
                # be checked against the expected cell type.
                typed_arr, arr_ty = self.infer(arr_expr, env)
                expected_cell_ty = self.default_cell_type(arr_ty, loc)
                typed_f, f_ty = self.check_callable_for_map(f_expr, expected_cell_ty, env, loc)
                frame_shape, result_ty = infer_lifting(f_ty, arr_ty)
                self.enforce_rank_limit(result_ty, loc)
                return TypedMap(
                    typed_f, typed_arr, frame_shape,
                    cell_shape_of(f_ty.params[0]), result_ty, loc), result_ty
            
            case FoldExpr(f_expr, init_expr, arr_expr, loc):
                typed_init, init_ty = self.infer(init_expr, env)
                typed_arr, arr_ty = self.infer(arr_expr, env)
                if not isinstance(arr_ty, ArrayType) or arr_ty.rank < 1:
                    raise RemoraTypeError("fold expects a non-scalar array", loc)
                elem_ty = arr_ty.drop_outer(1)
                expected_f_ty = FuncType((init_ty, elem_ty), init_ty)
                typed_f, f_ty = self.check(f_expr, expected_f_ty, env)
                result_ty = init_ty
                return TypedFold(typed_f, typed_init, typed_arr, result_ty, loc), result_ty
            
            case LetExpr(name, val, body, loc):
                typed_val, val_ty = self.infer(val, env)
                inner_env = env.extend(name, val_ty)
                typed_body, body_ty = self.infer(body, inner_env)
                return TypedLet(name, val_ty, typed_val, typed_body, body_ty), body_ty
            
            case AppExpr(func, args, loc):
                typed_f, f_ty = self.infer(func, env)
                if not isinstance(f_ty, FuncType):
                    raise RemoraTypeError(f"Not a function: {f_ty}", loc)
                typed_args = []
                for arg, param_ty in zip(args, f_ty.params):
                    ta, aty = self.infer(arg, env)
                    self.unify(aty, param_ty, loc)
                    typed_args.append(ta)
                return TypedApp(typed_f, typed_args, f_ty.result), f_ty.result
```

Array literal typing must recursively enforce shape consistency, not only element scalar type consistency. For example, `[[1, 2], [3]]` is a type error because the nested element shapes differ. Empty array literals are deferred until the language has explicit type annotations.

Dense Core rejects any `fold` where the accumulator type, init expression type, and reduction function result type do not unify exactly after explicit numeric promotion. Generalized reductions over array cells can be added later after rank-1 through rank-3 scalar folds are stable.

#### 2.6 Typed AST

The typed AST mirrors the AST but every node carries its `RemoraType`, and `TypedMap` additionally carries its resolved `frame_shape` (critical for the lowering pass):

```python
@dataclass
class TypedMap:
    func: TypedExpr
    array: TypedExpr
    frame_shape: tuple[DimExpr, ...]   # NEW: frame dimensions resolved by type checker
    cell_shape: tuple[DimExpr, ...]    # NEW: cell dimensions (from f's input type)
    type: RemoraType
    loc: SourceLoc
```

#### 2.7 Tasks

- [x] Implement `types.py` with all type variants and `with_frame`, `drop_outer`, etc.
- [x] Implement compile-time dimension evaluation with clear diagnostics for non-constant shape expressions
- [x] Implement `infer_lifting` for rank-polymorphic map
- [x] Implement `TypeEnv`, `TypeChecker.infer`, and `TypeChecker.check` for bidirectional typing
- [x] Implement explicit numeric promotion and `TypedCast`
- [x] Enforce Dense Core rank limit: rank 0 through rank 3 only
- [ ] Implement `TypeChecker.check` for full programs including top-level definitions
  - Partial: top-level value definitions are supported. Top-level function definitions are supported for statically known direct calls and unary `map` callables by specializing the function body at the call site from concrete argument types. Recursive functions, generalized annotations, and dynamic function values remain deferred.
- [ ] Implement `_build_prelude_env()` for built-in functions (+, *, etc.)
- [x] Write `tests/test_typechecker.py`:
  - Scalar literal typing
  - Rank-1, rank-2, and rank-3 array literal typing
  - Lambda checking from expected map/fold types
  - map with scalar function: `map (\x -> x + 1.0) : float[n] → float[n]`
  - map with scalar function over rank-2 and rank-3 arrays
  - map with vector function: `map (\row -> fold (+) 0.0 row) : float[m,n] → float[m]`
  - fold on vector: `fold (+) 0.0 : float[n] → float`
  - Nested map: `map (map f) : float[m,n] → float[m,n]`
  - `map (* 2.0) (iota 10)` inserts an int-to-float cast and returns `float[10]`
  - Rank-4 result is rejected
  - Type error for mismatched element types

**Milestone M2**: Type-check `fold (+) 0.0 (map (\x -> x * x) (iota 10))` and resolve all types and frame shapes.

---

### Phase 3: High-Level IR (HIR)

**Goal**: Define a simplified intermediate representation that sits between the typed AST and MLIR. HIR is:
- Explicit about all shapes (no symbolic inference variables remain)
- Frame/cell decomposed for every lifting operation
- Free of syntactic sugar
- Still functional and side-effect-free

#### 3.1 HIR Node Definitions (`remora/hir.py`)

```python
@dataclass
class HIRProgram:
    functions: list[HIRFunction]
    main: HIRExpr
    return_type: RemoraType

@dataclass
class HIRFunction:
    name: str
    params: list[HIRParam]
    body: HIRExpr
    return_type: RemoraType

@dataclass
class HIRParam:
    name: str
    type: RemoraType

@dataclass
class HIRMap:
    """Rank-polymorphic lifting; frame/cell explicitly decomposed."""
    frame_shape: tuple[DimExpr, ...]     # outer parallel dimensions
    cell_shape: tuple[DimExpr, ...]      # dimensions f operates on
    func: HIRCallable                    # function to apply (after defunc)
    array: HIRExpr
    result_type: RemoraType

@dataclass
class HIRFold:
    """Reduction along outermost dimension."""
    reduction_dim: DimExpr
    func: HIRCallable
    init: HIRExpr
    array: HIRExpr
    result_type: RemoraType

@dataclass
class HIRLet:
    name: str
    value_type: RemoraType
    value: HIRExpr
    body: HIRExpr
    result_type: RemoraType

@dataclass
class HIRCall:
    """Direct function call (static dispatch)."""
    func_name: str
    args: list[HIRExpr]
    result_type: RemoraType

@dataclass
class HIRDispatch:
    """Dynamic dispatch via closure tag (after defunctionalization)."""
    tag: HIRExpr
    cases: list[tuple[int, HIRExpr]]    # tag_value → body
    closure_args: list[HIRExpr]         # captured variables
    result_type: RemoraType

@dataclass
class HIRLambda:
    """Static function value before defunctionalization."""
    params: list[HIRParam]
    body: HIRExpr
    result_type: FuncType

@dataclass
class HIRPrimOp:
    """Primitive scalar operation."""
    op: str      # "+f", "*f", "-f", "/f", "+i", "neg", "sqrt", "exp", "<f", "=f", "and", "or", ...
    args: list[HIRExpr]
    result_type: RemoraType

@dataclass
class HIRIota:
    size: DimExpr
    result_type: RemoraType

@dataclass
class HIRCast:
    value: HIRExpr
    from_type: ScalarType
    to_type: ScalarType
    result_type: ScalarType

@dataclass
class HIRArrayLit:
    elements: list[HIRExpr]
    result_type: RemoraType

@dataclass
class HIRVar:
    name: str
    type: RemoraType

@dataclass
class HIRLit:
    value: int | float | bool
    type: RemoraType

HIRExpr = (HIRMap | HIRFold | HIRLet | HIRCall | HIRDispatch | HIRLambda |
           HIRPrimOp | HIRIota | HIRCast | HIRArrayLit | HIRVar | HIRLit)
HIRCallable = str | HIRLambda | HIRDispatch   # before defunc; str only after static defunc
```

Before defunctionalization, HIR may contain `HIRLambda` and static function references. After defunctionalization, `HIRMap.func` and `HIRFold.func` must be plain function names for Dense Core. `HIRDispatch` is a future representation and should not appear in a successfully lowered Dense Core program.

#### 3.2 Typed AST → HIR Lowering

This is a straightforward structural transformation (no type inference needed — the typed AST already has all types):

```python
def lower_to_hir(typed_prog: TypedProgram) -> HIRProgram:
    functions = [lower_func_def(d) for d in typed_prog.definitions]
    main = lower_expr(typed_prog.body, {})
    return HIRProgram(functions, main, typed_prog.body_type)

def lower_expr(expr: TypedExpr, env: dict[str, RemoraType]) -> HIRExpr:
    match expr:
        case TypedMap(f, arr, frame_shape, cell_shape, ty, loc):
            return HIRMap(frame_shape, cell_shape,
                          lower_callable(f), lower_expr(arr, env), ty)
        case TypedFold(f, init, arr, ty, loc):
            return HIRFold(arr.type.shape[0], lower_callable(f),
                           lower_expr(init, env), lower_expr(arr, env), ty)
        case TypedLet(name, val_ty, val, body, ty):
            return HIRLet(name, val_ty, lower_expr(val, env), lower_expr(body, env), ty)
        case TypedApp(f, args, ty):
            if isinstance(f, TypedVarExpr):
                return HIRCall(f.name, [lower_expr(a, env) for a in args], ty)
            # ...
        case TypedLit(v, ty):
            return HIRLit(v, ty)
        case TypedVarExpr(name, ty):
            return HIRVar(name, ty)
        # ...
```

---

### Phase 4: Defunctionalization

**Goal**: Eliminate all function values from Dense Core HIR, so no HOF call site carries a runtime function pointer. After defunctionalization, every `HIRMap.func` and `HIRFold.func` must be a static `str` naming a known function.

If a function value is stored, returned, captured in a closure that cannot be lambda-lifted, selected by a runtime conditional, stored in an array, or passed through a non-inlineable higher-order parameter, the compiler emits a clear "dynamic higher-order functions are deferred" diagnostic. `HIRDispatch` is reserved for a later runtime-function-value implementation and is not accepted by Dense Core lowering.

#### 4.1 Analysis Pass

```python
class DefuncAnalyzer:
    """
    For each HIRMap and HIRFold site, determine the set of function values
    that can flow to the 'func' argument.
    """
    def analyze(self, prog: HIRProgram) -> dict[str, set[str]]:
        """Returns: call-site-id → set of possible function names."""
        # Walk the HIR; collect all MapExpr/FoldExpr nodes
        # For each, track which HIRCallable values can reach it
        # For direct references (HIRVar pointing at a known function): static
        # For lambdas defined in-scope: collect and name them
        ...
```

#### 4.2 Static Case (Most Common)

When the function argument to `map` or `fold` is a literal lambda or a named top-level function:

```python
# Input HIR:
HIRMap(frame_shape=(StaticDim(10),), ...,
       func=HIRLambda(["x"], HIRPrimOp("+f", [HIRVar("x"), HIRLit(1.0, FLOAT)], FLOAT)),
       ...)

# After defunc: the lambda is lifted to a named HIRFunction
# map site becomes:
HIRMap(..., func="__lambda_0", ...)
# New top-level function added:
HIRFunction("__lambda_0", [HIRParam("x", FLOAT)],
            HIRPrimOp("+f", [HIRVar("x"), HIRLit(1.0, FLOAT)], FLOAT), FLOAT)
```

#### 4.3 Static Higher-Order Parameters

When a higher-order function receives a function value as a parameter (e.g., `def apply_twice f x = f (f x)`), Dense Core accepts it only when every call site passes a concrete statically known callable. The compiler handles this by **monomorphizing at the call site**: whenever `apply_twice` is called with a specific function, it generates a specialized version of `apply_twice` with that function inlined. This is equivalent to C++ template instantiation.

```python
class Monomorphizer:
    def specialize(self, func: HIRFunction, func_arg_name: str, 
                   concrete_func: str) -> HIRFunction:
        """Produce a copy of func with func_arg_name replaced by concrete_func inline."""
        ...
```

For genuinely dynamic cases, defer to a post-Dense-Core phase.

#### 4.4 Tasks

- [ ] Implement `DefuncAnalyzer.analyze` to classify each HOF site as static or rejected-deferred
  - Partial: current `defunctionalize` pass statically rewrites accepted HOF sites and rejects captured/dynamic lambdas, but there is no separate analyzer object.
- [x] Implement lambda lifting: collect all inline lambdas; assign names; add to function table
- [ ] Implement monomorphization for direct HOF parameters only when the concrete function is known at the call site
- [x] Write `tests/test_defunc.py`:
  - Inline lambda in map: lifted to named function
  - Named function passed to map: trivial (already a string)
  - Lambda capturing a variable from outer scope: either lambda-lifted with explicit captured scalar args or rejected with the deferred-dynamic-HOF diagnostic

**Milestone M3**: Defunctionalize `map (\x -> x * 2.0) (iota 10)` to `HIRMap(func="__lambda_0", ...)` with `__lambda_0` as a top-level HIR function.

---

### Phase 5: HIR → MLIR Linalg Lowering

**Goal**: Generate MLIR operations from HIR using `mlir-python-bindings`. This is the core of the compiler.

The snippets in this phase are design sketches until validated against the pinned LLVM/MLIR 18 Python bindings. Implementation must proceed by creating tiny checked builders in this order: `iota`, unary scalar `map`, scalar `fold`, then composition of `map` into `fold`. Each builder must round-trip through `mlir-opt --verify-diagnostics` before being used by the compiler.

#### 5.1 MLIR Module Setup

```python
from mlir.ir import (Context, Module, Location, InsertionPoint,
                     F32Type, IntegerType, IndexType, FunctionType,
                     RankedTensorType, ShapedType, AffineMap,
                     AffineMapAttr, ArrayAttr, StringAttr, IntegerAttr)
from mlir.dialects import (func as func_d, linalg, tensor, arith,
                            math as math_d, memref, index as index_d)

class MLIRLowering:
    def __init__(self):
        self.ctx = Context()
        # Register all dialects we'll use
        self.ctx.enable_multithreading(False)
        self.loc = Location.unknown(self.ctx)
    
    def lower_program(self, prog: HIRProgram) -> Module:
        with self.ctx, self.loc:
            module = Module.create()
            with InsertionPoint(module.body):
                for fn in prog.functions:
                    self._lower_function(fn)
                # Lower main as a special entry function
                self._lower_main(prog.main, prog.return_type)
        return module
```

#### 5.2 Type Lowering

```python
def lower_type(self, ty: RemoraType) -> mlir.Type:
    match ty:
        case ScalarType("float"):
            return F32Type.get()
        case ScalarType("int"):
            return IntegerType.get_signless(32)
        case ScalarType("bool"):
            return IntegerType.get_signless(1)
        case ArrayType(elem, shape):
            dims = []
            for d in shape:
                if isinstance(d, StaticDim):
                    dims.append(d.value)
                else:
                    dims.append(ShapedType.get_dynamic_size())
            return RankedTensorType.get(dims, self.lower_type(elem))
        case FuncType(params, result):
            return FunctionType.get(
                [self.lower_type(p) for p in params],
                [self.lower_type(result)])
```

#### 5.3 Map Lowering (Central Operation)

For `HIRMap(frame_shape, cell_shape, func, array, result_type)`:

**Case A: Scalar cell function** (frame covers all dimensions of array)

```python
def _lower_map_scalar(self, node: HIRMap, env: ValueEnv) -> Value:
    """Map a scalar→scalar function over all elements of an array."""
    arr_val = self._lower_expr(node.array, env)
    result_ty = self.lower_type(node.result_type)
    
    # Create empty output tensor
    dynamic_dims = self._dynamic_dim_values(node.frame_shape, env)
    empty = tensor.EmptyOp(result_ty, dynamic_dims).result
    
    # Build affine maps: identity for both input and output
    n = len(node.frame_shape)
    identity = AffineMap.get_identity(n)
    
    # Build iterator types: all parallel
    iters = ArrayAttr.get([StringAttr.get("parallel")] * n)
    
    # Create linalg.generic
    generic = linalg.GenericOp(
        result_tensors=[result_ty],
        inputs=[arr_val],
        outputs=[empty],
        indexing_maps=AffineMapAttr.get([identity, identity]),
        iterator_types=iters)
    
    # Build region body: call func on the element
    elem_ty = self.lower_type(node.result_type.element 
                               if isinstance(node.result_type, ArrayType)
                               else node.result_type)
    block = generic.regions[0].blocks.append(elem_ty, elem_ty)
    with InsertionPoint(block):
        in_elem = block.arguments[0]
        result = self._lower_function_call(node.func, [in_elem], env)
        linalg.YieldOp([result])
    
    return generic.result
```

**Case B: Vector/matrix cell function** (some dimensions are frame, rest are cell)

Dense Core must support this for total input/output rank up to 3 after scalar-cell rank-0..3 maps are working. This is needed for real Remora frame/cell behavior over matrices and rank-3 tensors, but it should be implemented after the scalar-cell builder has golden MLIR coverage.

This requires indexing maps that split frame dimensions (parallel) from cell dimensions (sequential within the body):

```python
def _lower_map_cell(self, node: HIRMap, env: ValueEnv) -> Value:
    """Map a cell function over the frame dimensions of an array."""
    n_frame = len(node.frame_shape)
    n_cell = len(node.cell_shape)
    total = n_frame + n_cell
    
    # Input indexing map: full identity over (frame + cell) dimensions
    input_map = AffineMap.get_identity(total)
    # Output indexing map: only frame dimensions (output has frame shape only if result is scalar,
    # or frame+result_cell if result is non-scalar)
    output_map = AffineMap.get_minor_identity(total, n_frame)
    
    iters = (["parallel"] * n_frame) + (["reduction"] * n_cell)
    # ... build linalg.generic with appropriate maps
```

#### 5.4 Fold Lowering

```python
def _lower_fold(self, node: HIRFold, env: ValueEnv) -> Value:
    """Reduce array along outermost dimension."""
    arr_val = self._lower_expr(node.array, env)
    init_val = self._lower_expr(node.init, env)
    result_ty = self.lower_type(node.result_type)
    
    # Wrap scalar init in a 0D tensor
    init_tensor = tensor.FromElementsOp(
        RankedTensorType.get([], self.lower_type(node.result_type)),
        [init_val]).result
    
    # Input map: (i) → (i),  output map: (i) → ()  (scalar)
    n = 1  # single reduction dimension (outermost)
    input_map = AffineMap.get_identity(n)
    output_map = AffineMap.get(n, 0, [])  # maps all dims to nothing → scalar
    
    generic = linalg.GenericOp(
        result_tensors=[RankedTensorType.get([], result_ty)],
        inputs=[arr_val],
        outputs=[init_tensor],
        indexing_maps=AffineMapAttr.get([input_map, output_map]),
        iterator_types=ArrayAttr.get([StringAttr.get("reduction")]))
    
    block = generic.regions[0].blocks.append(result_ty, result_ty)
    with InsertionPoint(block):
        elem = block.arguments[0]
        acc  = block.arguments[1]
        result = self._lower_function_call(node.func, [acc, elem], env)
        linalg.YieldOp([result])
    
    # Extract scalar from 0D tensor
    return tensor.ExtractOp(generic.result, []).result
```

#### 5.5 Scalar Operation Lowering

```python
def _lower_prim_op(self, node: HIRPrimOp, env: ValueEnv) -> Value:
    args = [self._lower_expr(a, env) for a in node.args]
    match node.op:
        case "+f":  return arith.AddFOp(args[0], args[1]).result
        case "-f":  return arith.SubFOp(args[0], args[1]).result
        case "*f":  return arith.MulFOp(args[0], args[1]).result
        case "/f":  return arith.DivFOp(args[0], args[1]).result
        case "negf": return arith.NegFOp(args[0]).result
        case "+i":  return arith.AddIOp(args[0], args[1]).result
        case "*i":  return arith.MulIOp(args[0], args[1]).result
        case "sqrt": return math_d.SqrtOp(args[0]).result
        case "exp":  return math_d.ExpOp(args[0]).result
        case "log":  return math_d.LogOp(args[0]).result
        case "<f":  return arith.CmpFOp(arith.CmpFPredicate.OLT, args[0], args[1]).result
        case "=f":  return arith.CmpFOp(arith.CmpFPredicate.OEQ, args[0], args[1]).result
        case "and": return arith.AndIOp(args[0], args[1]).result
        case "or":  return arith.OrIOp(args[0], args[1]).result
        case "not": return arith.XOrIOp(args[0],
                        arith.ConstantOp(IntegerType.get_signless(1),
                                         IntegerAttr.get(IntegerType.get_signless(1), 1)).result).result
```

#### 5.6 Iota Lowering

`iota n` generates an array `[0, 1, ..., n-1]`. Lower to a `linalg.generic` that uses `linalg.index`:

```python
def _lower_iota(self, node: HIRIota, env: ValueEnv) -> Value:
    n = self._resolve_dim(node.size, env)
    result_ty = RankedTensorType.get([n.value], IntegerType.get_signless(32))
    empty = tensor.EmptyOp(result_ty, []).result
    
    generic = linalg.GenericOp(
        result_tensors=[result_ty],
        inputs=[],
        outputs=[empty],
        indexing_maps=AffineMapAttr.get([AffineMap.get_identity(1)]),
        iterator_types=ArrayAttr.get([StringAttr.get("parallel")]))
    
    block = generic.regions[0].blocks.append(IntegerType.get_signless(32))
    with InsertionPoint(block):
        idx = linalg.IndexOp(IntegerAttr.get(IntegerType.get_signless(64), 0)).result
        # Cast from index to i32
        i32_idx = arith.IndexCastOp(IntegerType.get_signless(32), idx).result
        linalg.YieldOp([i32_idx])
    
    return generic.result
```

#### 5.7 Tasks

- [x] Implement `MLIRLowering` class with context and module setup
- [x] Implement `lower_type` for all `RemoraType` variants
- [ ] Validate exact MLIR Python builder patterns for `tensor.empty`, `linalg.generic`, `linalg.index`, `tensor.extract`, and `func.func` against checked-in golden MLIR fixtures
  - Partial: checked-in fixtures now validate the current textual, parse-checked MLIR output for `iota`, scalar maps, rank-2 literal maps, and map-then-fold programs. Python builder pattern validation remains deferred until the dialect builder dependencies are stable.
- [x] Implement `_lower_map_scalar` for rank-0, rank-1, rank-2, and rank-3 elementwise maps
- [ ] Implement `_lower_map_cell` for static frame/cell maps whose total result rank is <= 3
  - Partial: textual MLIR lowering supports the rank-1-cell reduction pattern over rank-2/rank-3 inputs, such as `map (\row -> fold (+) 0 row) xs`. General cell maps whose body is not a fold remain deferred.
- [x] Implement `_lower_fold` for reductions over the outermost dimension of rank-1, rank-2, and rank-3 arrays
- [x] Implement `_lower_prim_op` for all scalar operations
- [x] Implement `_lower_cast` for explicit numeric promotions
- [x] Implement `_lower_iota`
- [x] Implement `_lower_function_call` (dispatches to function body or named call)
- [ ] Implement `_lower_let` (introduce SSA value into env)
  - Partial: `_lower_let` exists and lowers scalar lets through an SSA environment. Tensor lets still use simple HIR inlining before textual emission.
- [ ] Implement `_lower_function` for top-level HIR functions → `func.func`
  - Partial: scalar HIR functions lower to `func.func private`; user-authored top-level function definitions now work in the CPU evaluator through typed static lambdas, but are not yet emitted as top-level HIR/MLIR functions.
- [x] Implement `_lower_main`: create a `main()` `func.func` that wraps the program body
- [x] Reject dynamic dimensions until Dense Core static-shape rank-0..3 programs execute end-to-end
- [x] Write `tests/test_lowering.py`:
  - Check generated MLIR text for `map (\x -> x * 2.0) (iota 10)` contains the expected `iota` and scalar-map `linalg.generic` operations
  - Check rank-2 and rank-3 scalar maps produce the expected number of parallel iterators
  - Check rank-0 scalar maps over scalar inputs lower as scalar function application
  - Check nested scalar map chains over `iota` and static array literals
  - Check a vector-cell map over a rank-2 array splits frame and cell dimensions correctly
  - Check rank-2/rank-3 outermost folds over array cells
  - Check rank-2/rank-3 rank-1-cell maps whose lifted lambda body is a fold
  - Check scalar let SSA lowering and scalar HIR function calls
  - Check fold generates `iterator_types = ["reduction"]`
  - Check iota generates `linalg.index`
  - Check current textual MLIR output against checked-in golden fixtures
  - Check standalone scalar literals, primitive operations, numeric comparisons, boolean operations, division, and explicit numeric casts

**Milestone M4**: `lower_program(hir)` for rank-1, rank-2, and rank-3 scalar maps plus `fold (+) 0.0 (map (* 2.0) (iota 10))` produces valid MLIR that passes `mlir-opt --verify-diagnostics`.

Current status: the generated textual MLIR is parse-validated through
`iree.compiler.ir.Module.parse` and covered by golden fixtures/tests. The
`mlir-opt --verify-diagnostics` command is not available on the current PATH,
so external verifier round-tripping remains an environment setup task before
Phase 6 pipeline validation.

---

### Phase 6: MLIR Pass Pipeline

**Goal**: Drive the MLIR module through the full lowering pipeline, from `tensor + linalg` all the way to PTX.

The pass lists below are starting points, not authoritative API. LLVM/MLIR pass names, nesting, and options change across releases. The implementation must pin one LLVM/MLIR 18 build and record the exact command-line-equivalent pipeline that works with that build.

Current implementation note: Phase 6 is implemented against the installed
`iree-compiler` toolchain because this repository does not currently have
standalone LLVM/MLIR tools (`mlir-opt`, `mlir-translate`, `llc`) on `PATH`.
`remora.pipeline` still records the starter CPU/NVIDIA MLIR pass strings, but
the executable PTX-producing path uses `iree-compile` and is covered by tests.
Pinning standalone LLVM/MLIR 18 remains deferred toolchain work, not a blocker
for the current IREE-backed Phase 6 validation slice.

#### 6.1 Pass Pipeline Configurations (`remora/pipeline.py`)

```python
from mlir.passmanager import PassManager
from mlir.ir import Context, Module

def build_cpu_pipeline(ctx: Context) -> PassManager:
    """
    Lower to CPU LLVM for testing and REPL fallback.
    Runs on any machine, no GPU needed.
    """
    pipeline = ",".join([
        "linalg-fuse-elementwise-ops",
        "linalg-generalize-named-ops",
        "one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops}",
        "buffer-deallocation-pipeline",
        "lower-affine",
        "convert-linalg-to-loops",
        "convert-scf-to-cf",
        "convert-arith-to-llvm",
        "convert-math-to-llvm",
        "convert-func-to-llvm",
        "convert-index-to-llvm",
        "reconcile-unrealized-casts",
    ])
    return PassManager.parse(f"builtin.module({pipeline})", ctx)

def build_gpu_nvidia_pipeline(ctx: Context, 
                               tile_size: int = 256) -> PassManager:
    """
    Full GPU lowering pipeline for NVIDIA (PTX via NVVM).
    """
    pipeline = ",".join([
        # Step 1: Linalg-level fusion (before bufferization)
        "linalg-fuse-elementwise-ops",
        "linalg-generalize-named-ops",
        
        # Step 2: Bufferization
        "one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops}",
        "buffer-deallocation-pipeline",
        
        # Step 3: Affine-level fusion and parallelization
        "affine-loop-fusion",
        "affine-parallelize",
        
        # Step 4: Lower to parallel loops
        "lower-affine",
        "convert-linalg-to-parallel-loops",
        
        # Step 5: Map to GPU threads
        f"gpu-map-parallel-loops",
        "convert-parallel-loops-to-gpu",
        
        # Step 6: GPU kernel outlining
        "gpu-kernel-outlining",
        
        # Step 7: Lower everything to LLVM + NVVM
        "lower-affine",
        "convert-scf-to-cf",
        "convert-gpu-to-nvvm{index-bitwidth=64}",
        "convert-arith-to-llvm",
        "convert-math-to-llvm",
        "convert-func-to-llvm",
        "gpu-to-llvm",
        "reconcile-unrealized-casts",
    ])
    return PassManager.parse(f"builtin.module({pipeline})", ctx)

def build_gpu_amd_pipeline(ctx: Context) -> PassManager:
    """Deferred: GPU lowering pipeline for AMD (ROCDL)."""
    pipeline = ",".join([
        "linalg-fuse-elementwise-ops",
        "linalg-generalize-named-ops",
        "one-shot-bufferize{bufferize-function-boundaries}",
        "buffer-deallocation-pipeline",
        "affine-loop-fusion",
        "affine-parallelize",
        "lower-affine",
        "convert-linalg-to-parallel-loops",
        "gpu-map-parallel-loops",
        "convert-parallel-loops-to-gpu",
        "gpu-kernel-outlining",
        "lower-affine",
        "convert-scf-to-cf",
        "convert-gpu-to-rocdl{index-bitwidth=64}",
        "convert-arith-to-llvm",
        "convert-math-to-llvm",
        "convert-func-to-llvm",
        "gpu-to-llvm",
        "reconcile-unrealized-casts",
    ])
    return PassManager.parse(f"builtin.module({pipeline})", ctx)

def run_pipeline(module: Module, pm: PassManager, 
                 debug: bool = False) -> None:
    """Run a pass pipeline, optionally printing MLIR at each stage."""
    if debug:
        # Enable per-pass MLIR dumps
        pm.enable_ir_printing()
    pm.run(module)
```

Before this pipeline is used by `remorac`, add a `tools/validate_mlir_pipeline.py` script that:

1. Prints `mlir-opt --version`.
2. Runs each pipeline stage against a checked-in minimal `iota_map_fold.mlir`.
3. Fails on unknown pass names, verifier errors, or unexpected dialects remaining after each major stage.
4. Stores the known-good expanded pipeline in `docs/mlir-pipeline-llvm18.txt`.

Any future LLVM upgrade starts by updating this validation artifact.

#### 6.2 PTX Generation (`remora/codegen.py`)

After the GPU pipeline, the MLIR module contains two sub-modules:
- **Host module**: calls `gpu.launch_func`; contains `func.func` for the main program
- **Device module**: contains the `gpu.func` kernels in NVVM/LLVM IR

```python
import subprocess
import tempfile
import os

def generate_ptx(module: Module, sm_version: str = "sm_80") -> tuple[str, list['KernelMeta']]:
    """
    Extract device MLIR module, serialize to LLVM IR, run `llc` to produce PTX.
    Returns (ptx_text, kernel_metadata).
    """
    # Find the gpu.module inside the top-level module
    gpu_module = _extract_gpu_module(module)
    
    # Serialize to LLVM IR text via mlir-translate
    llvm_ir = _mlir_to_llvmir(gpu_module)
    
    # Run llc to produce PTX
    with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
        f.write(llvm_ir)
        ll_path = f.name
    
    try:
        result = subprocess.run(
            ["llc", "-mcpu", sm_version, "-mattr", "+ptx75",
             "-filetype", "asm", ll_path, "-o", "-"],
            capture_output=True, text=True, check=True)
        ptx = result.stdout
    finally:
        os.unlink(ll_path)
    
    meta = _extract_kernel_metadata(module, gpu_module)
    return ptx, meta

@dataclass
class KernelMeta:
    name: str                     # PTX kernel function name
    grid_dims: int                # number of grid dimensions (1 or 2)
    block_size: int               # threads per block (default 256)
    num_inputs: int               # number of input tensor arguments
    num_outputs: int              # number of output tensor arguments
    input_elem_types: list[str]   # element types of input tensors
    output_elem_types: list[str]  # element types of output tensors
```

**Alternative to subprocess `llc`**: Use the LLVM Python bindings (`llvmlite` or the official `llvm` package) to run the NVPTX backend in-process. This is faster and avoids file I/O:

```python
import llvmlite.binding as llvm

llvm.initialize()
llvm.initialize_nvptx_target()

def llvmir_to_ptx(ir_text: str, sm: str = "sm_80") -> str:
    module = llvm.parse_assembly(ir_text)
    target = llvm.Target.from_triple("nvptx64-nvidia-cuda")
    machine = target.create_target_machine(
        cpu=sm, features="+ptx75", opt=2)
    return machine.emit_assembly(module)
```

#### 6.3 Tasks

- [ ] Implement `build_cpu_pipeline` and verify with `ExecutionEngine`
  - Partial: `remora.pipeline` contains the starter CPU pipeline string and reports `PipelineUnavailable` when the installed pass registry cannot parse it. ExecutionEngine verification is deferred until the standalone MLIR toolchain is pinned.
- [ ] Pin an LLVM/MLIR 18 build and commit the exact validated CPU and NVIDIA pipelines
- [ ] Implement `build_gpu_nvidia_pipeline` only after validating pass names/options with `mlir-opt`
  - Partial: `remora.pipeline` contains the starter NVIDIA pipeline string, gated behind pass-manager parsing. The installed IREE pass registry does not accept the starter standalone MLIR pipeline, so direct pass-pipeline execution remains gated.
- [x] Implement `run_pipeline` with debug mode
- [x] Implement external MLIR verification for emitted modules
  - Current: `verify_module_text` uses `mlir-opt` when available and otherwise uses `.venv/bin/iree-opt --verify-diagnostics -`.
- [x] Implement `generate_ptx`
  - Current: `remora.codegen.generate_ptx` invokes `iree-compile` with the CUDA HAL backend and dumps generated `.ptx` files. The generated PTX is an IREE HAL dispatch kernel, not yet the final direct Remora ABI kernel.
- [x] Implement `_extract_kernel_metadata` to collect kernel names and argument info
  - Current: metadata extraction records PTX entry name, PTX parameter count, and `.maxntid` block size. Final input/output element type metadata is deferred until the direct Remora kernel ABI path exists.
- [x] Write `tests/test_pipeline.py`:
  - Current: toolchain detection, validation-pipeline execution, direct `run_pipeline`, external verifier execution when available, unavailable-pass diagnostics, CPU pipeline gating, and CUDA PTX generation through `iree-compile` when available.
  - CPU pipeline: MLIR passes cleanly; ExecutionEngine executes correctly
  - GPU pipeline: MLIR passes cleanly; PTX contains expected kernel function
  - Fusion test: map chain produces a single `linalg.generic` after fusion pass
  - Verify PTX is syntactically valid (parse it back with a PTX parser or run `ptxas --dry-run`)

**Milestone M5**: Generate valid PTX from `fold (+) 0.0 (map (* 2.0) (iota 1000))`.
Current: the IREE-backed PTX path generates PTX for lowered Dense Core modules;
tests cover both `map (* 2) (iota 4)` and
`fold (+) 0.0 (map (* 2.0) (iota 1000))`.

---

### Phase 7: Runtime and Execution Engine

**Goal**: Load compiled PTX, execute on GPU, copy result back to host.

The runtime ABI is descriptor-based and is specified normatively in `docs/ABI.md`. Remora external kernels receive pointers to rank-specialized memref descriptor structs, not ad hoc raw pointer plus dimension lists. For a ranked memref, the descriptor contains:

```text
allocated pointer
aligned pointer
offset
sizes[rank]
strides[rank]
```

The Python CUDA launcher must build C-compatible descriptor structs matching `docs/ABI.md`. For the first prototype, allocated arrays are contiguous row-major, but descriptors must support arbitrary offsets and strides:

- `allocated == aligned`
- `offset == 0`
- `sizes[i]` is the static/runtime extent
- `strides` are computed from the trailing dimensions

View-producing operations are deferred, but the ABI is view-capable from day one. Kernel metadata must describe descriptor arguments and scalar arguments explicitly. Do not launch kernels with only `[input_ptrs..., output_ptr, shapes...]`.

The MLIR lowering may use internal memref conventions, but the exported CUDA kernel entry points used by `RemoraExecutor` must conform to `docs/ABI.md`. If the pinned MLIR pipeline cannot emit that ABI directly, codegen must generate a thin wrapper/adaptor kernel and test it.

#### 7.1 CUDA Runtime Wrapper (`remora/runtime.py`)

Using the `cuda-python` package (official NVIDIA Python bindings):

```python
from cuda import cuda, nvrtc
import numpy as np

class CUDAError(Exception): pass

def _check(err, msg="CUDA error"):
    if isinstance(err, tuple): err = err[0]
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise CUDAError(f"{msg}: {err}")

class CUDARuntime:
    """Manages a CUDA context and provides device memory operations."""
    
    def __init__(self, device_idx: int = 0):
        _check(cuda.cuInit(0))
        err, self._device = cuda.cuDeviceGet(device_idx)
        _check(err)
        err, self._ctx = cuda.cuCtxCreate(0, self._device)
        _check(err)
    
    def load_ptx(self, ptx: str) -> 'CUDAModule':
        ptx_bytes = ptx.encode("utf-8")
        err, module = cuda.cuModuleLoadData(ptx_bytes)
        _check(err, "Failed to load PTX")
        return CUDAModule(module, self)
    
    def alloc(self, nbytes: int) -> int:
        """Allocate nbytes on device. Returns device pointer as int."""
        err, ptr = cuda.cuMemAlloc(nbytes)
        _check(err, "cuMemAlloc failed")
        return int(ptr)
    
    def free(self, ptr: int):
        _check(cuda.cuMemFree(ptr))
    
    def copy_host_to_device(self, host_array: np.ndarray, device_ptr: int):
        err = cuda.cuMemcpyHtoD(device_ptr, host_array.ctypes.data, host_array.nbytes)
        _check(err, "H2D copy failed")
    
    def copy_device_to_host(self, device_ptr: int, host_array: np.ndarray):
        err = cuda.cuMemcpyDtoH(host_array.ctypes.data, device_ptr, host_array.nbytes)
        _check(err, "D2H copy failed")
    
    def synchronize(self):
        _check(cuda.cuCtxSynchronize())
    
    def __del__(self):
        try: cuda.cuCtxDestroy(self._ctx)
        except: pass

class CUDAModule:
    def __init__(self, module, runtime: CUDARuntime):
        self._module = module
        self._rt = runtime
    
    def get_function(self, name: str) -> 'CUDAKernel':
        err, func = cuda.cuModuleGetFunction(self._module, name.encode())
        _check(err, f"Function '{name}' not found in PTX module")
        return CUDAKernel(func, self._rt)
    
    def __del__(self):
        try: cuda.cuModuleUnload(self._module)
        except: pass

class CUDAKernel:
    def __init__(self, func, runtime: CUDARuntime):
        self._func = func
        self._rt = runtime
    
    def launch(self, grid: tuple[int,int,int], block: tuple[int,int,int],
               args: list, shared_mem: int = 0):
        """Launch this kernel with the given grid/block dimensions and arguments."""
        import ctypes
        kernel_args = (ctypes.c_void_p * len(args))()
        arg_ptrs = []
        for i, arg in enumerate(args):
            if isinstance(arg, int):
                c_arg = ctypes.c_uint64(arg)
            elif isinstance(arg, float):
                c_arg = ctypes.c_float(arg)
            else:
                raise TypeError(f"Unsupported kernel arg type: {type(arg)}")
            arg_ptrs.append(c_arg)
            kernel_args[i] = ctypes.cast(ctypes.addressof(c_arg), ctypes.c_void_p)
        
        err = cuda.cuLaunchKernel(
            self._func,
            *grid, *block,
            shared_mem, 0,   # sharedMem, stream
            kernel_args, None)
        _check(err, "Kernel launch failed")
```

#### 7.2 High-Level Executor (`remora/executor.py`)

```python
class RemoraExecutor:
    """
    Orchestrates the end-to-end execution of a compiled Remora program.
    Manages device buffers, launches kernels, marshals results.
    """
    
    def __init__(self, ptx: str, meta_list: list[KernelMeta], 
                 runtime: CUDARuntime):
        self._rt = runtime
        self._cuda_module = runtime.load_ptx(ptx)
        self._kernels = {m.name: self._cuda_module.get_function(m.name) 
                         for m in meta_list}
        self._meta = {m.name: m for m in meta_list}
    
    def execute(self, kernel_name: str, 
                inputs: list[np.ndarray]) -> np.ndarray:
        """Run a single kernel on the given numpy inputs, return numpy output."""
        meta = self._meta[kernel_name]
        
        # Allocate device buffers and copy inputs
        device_inputs = []
        for arr in inputs:
            ptr = self._rt.alloc(arr.nbytes)
            self._rt.copy_host_to_device(arr, ptr)
            device_inputs.append(ptr)
        
        # Allocate output buffer
        output_shape = self._compute_output_shape(meta, inputs)
        output_dtype = np.float32  # TODO: derive from meta.output_elem_types
        output = np.empty(output_shape, dtype=output_dtype)
        output_ptr = self._rt.alloc(output.nbytes)
        
        # Calculate grid dimensions
        n_elements = int(np.prod(output_shape))
        block_size = meta.block_size
        grid_size = (n_elements + block_size - 1) // block_size
        
        # Build kernel arguments: pointers to memref descriptors for each
        # input/output, followed by any scalar arguments required by metadata.
        input_descs = [
            make_memref_descriptor(ptr, arr.shape, element_strides(arr), arr.dtype)
            for ptr, arr in zip(device_inputs, inputs)
        ]
        output_desc = make_memref_descriptor(
            output_ptr, output.shape, element_strides(output), output.dtype)
        args = [*input_descs, output_desc]
        
        # Launch
        kernel = self._kernels[kernel_name]
        kernel.launch((grid_size, 1, 1), (block_size, 1, 1), args)
        self._rt.synchronize()
        
        # Copy result back
        self._rt.copy_device_to_host(output_ptr, output)
        
        # Free device memory
        for ptr in device_inputs:
            self._rt.free(ptr)
        self._rt.free(output_ptr)
        
        return output
```

#### 7.3 CPU Fallback Executor

For testing without a GPU, for debugging, and for the early REPL, use MLIR's `ExecutionEngine` with the CPU pipeline. The CPU executor follows the same logical ABI as the CUDA executor:

- inputs are numpy arrays described by rank-0..3 descriptors
- outputs are allocated by the executor from `KernelMeta`
- `main` writes into output descriptors rather than returning heap-allocated arrays
- scalar results are represented as rank-0 descriptors

This keeps REPL, CPU tests, and GPU tests aligned. Current implementation note:
because the standalone MLIR CPU pipeline and `ExecutionEngine` are not pinned
yet, `remora.runtime` provides an interim typed-AST CPU evaluator so examples
can run now. That evaluator is a user-facing bridge for the prototype, not the
final production CPU backend.

```python
from mlir.execution_engine import ExecutionEngine
import ctypes

class CPUExecutor:
    """Execute compiled Remora programs on CPU via MLIR ExecutionEngine."""
    
    def __init__(self, module: Module):
        # Module must have been lowered through the CPU pipeline
        self._engine = ExecutionEngine(module, opt_level=2)
    
    def execute_main(self, inputs: list[np.ndarray]) -> np.ndarray:
        # Allocate output numpy arrays from KernelMeta.
        # Build rank-specialized memref descriptors for inputs and outputs.
        # Call self._engine.invoke("main", *descriptor_ptrs, *scalar_args)
        # Return the output numpy array or scalar rank-0 value.
        ...
```

#### 7.4 Tasks

- [ ] Implement `CUDARuntime`: init, alloc, free, H2D/D2H copy, synchronize
- [ ] Implement `CUDAModule` and `CUDAKernel`
- [x] Implement Remora memref descriptor structs with ctypes for rank-0, rank-1, rank-2, and rank-3 buffers
- [x] Implement descriptor construction from numpy arrays and Remora view metadata, including byte-stride to element-stride conversion and nonzero offsets for future views
- [ ] Implement `CUDAKernel.launch` with descriptor-aware ctypes argument packing
- [ ] Implement `RemoraExecutor.execute` for a single kernel
- [ ] Implement output shape computation from kernel metadata
- [ ] Implement `CPUExecutor` for GPU-free testing
  - Current: `remora.runtime.evaluate_source` is an interim typed-AST CPU evaluator covering the checked-in Dense Core examples. MLIR `ExecutionEngine` execution remains deferred until the CPU pipeline is pinned.
- [ ] Write `tests/test_execution.py`:
  - Current: `tests/test_runtime.py` covers the interim CPU evaluator, all checked-in examples, compiler facade helpers, and `remorac` CPU output. Rename or expand into `tests/test_execution.py` when the MLIR `CPUExecutor` exists.
  - Double all elements of `[1.0, 2.0, 3.0]` → `[2.0, 4.0, 6.0]`
  - Double all elements of a 2D matrix and a 3D tensor
  - Sum vector `[1.0, ..., 10.0]` → `55.0`
  - Dot product `[1,2,3] · [4,5,6]` → `32.0`
  - Verify on CPU executor first, then GPU executor
  - ABI tests in `tests/test_abi.py` launch tiny rank-0, rank-1, rank-2, and rank-3 kernels that read and write through descriptors

**Milestone M6**: descriptor ABI tests pass for rank 0 through rank 3, and `executor.execute("main", [np.arange(10, dtype=np.float32)])` returns `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]` for a `map (* 2.0)` program.
Current: the ABI descriptor tests already pass, and the interim CPU evaluator
can run `map (* 2.0) (iota 10)` and every checked-in example. The
`executor.execute(...)` MLIR/CUDA ABI path remains deferred.

---

### Phase 8: User-Facing Entry Points

**Goal**: expose the compiler through command-line tools: `remorac prog.rem` for file-based execution and `remora --target cpu` for interactive expression evaluation.

For the first prototype, `remorac --target` supports `cpu` and `gpu-nvidia` only. The REPL starts with `cpu` and adds `gpu-nvidia` after the descriptor ABI and CUDA runtime are stable. `gpu-amd`, `--output`, and broad debug flags can be added after the vertical slice is stable.

Current implementation note: `remorac` is registered as a Python console script
and defaults to `--target cpu`, using the interim typed-AST evaluator. It also
supports `--emit-ast`, `--emit-typed-ast`, `--emit-hir`, `--emit-mlir`,
`--emit-ptx`, plus `--target mlir` and `--target ptx` for developer inspection
of the current lowering/codegen artifacts.

#### 8.1 AOT Command-Line Interface (`bin/remorac`)

```
Usage: remorac [OPTIONS] SOURCE_FILE

Options:
  -o, --output PATH      Output binary path (default: run immediately)
  --target BACKEND       gpu-nvidia (default) | cpu
  --emit-ast             Print AST and exit
  --emit-typed-ast       Print typed AST and exit
  --emit-hir             Print HIR and exit
  --emit-mlir            Print MLIR before passes and exit
  --emit-mlir-after      Print MLIR after each pass
  --emit-ptx             Print PTX and exit
  --debug                Enable all debug output
  --sm-version VERSION   CUDA SM version (default: sm_80)
  --block-size N         GPU block size (default: 256)
  -h, --help             Show help
```

#### 8.2 AOT Implementation (`bin/remorac`)

```python
#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from remora.parser import parse_file
from remora.typechecker import TypeChecker
from remora.defunc import defunctionalize
from remora.hir import lower_to_hir
from remora.lowering import MLIRLowering
from remora.pipeline import build_gpu_nvidia_pipeline, build_cpu_pipeline, run_pipeline
from remora.codegen import generate_ptx
from remora.runtime import CUDARuntime
from remora.executor import RemoraExecutor, CPUExecutor
from remora.display import format_result
from remora.errors import RemoraError

def main():
    p = argparse.ArgumentParser(description="Remora GPU compiler")
    p.add_argument("source", help="Remora source file")
    p.add_argument("-o", "--output", help="Output binary path")
    p.add_argument("--target", default="gpu-nvidia",
                   choices=["gpu-nvidia", "cpu"])
    p.add_argument("--emit-ast", action="store_true")
    p.add_argument("--emit-typed-ast", action="store_true")
    p.add_argument("--emit-hir", action="store_true")
    p.add_argument("--emit-mlir", action="store_true")
    p.add_argument("--emit-mlir-after", action="store_true")
    p.add_argument("--emit-ptx", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--sm-version", default="sm_80")
    p.add_argument("--block-size", type=int, default=256)
    args = p.parse_args()
    
    try:
        # Parse
        ast = parse_file(args.source)
        if args.emit_ast or args.debug:
            print("=== AST ==="); print(ast); print()
        
        # Type check
        checker = TypeChecker()
        typed_ast = checker.check(ast)
        if args.emit_typed_ast or args.debug:
            print("=== Typed AST ==="); print(typed_ast); print()
        
        # Lower to HIR
        hir = lower_to_hir(typed_ast)
        
        # Defunctionalize
        hir = defunctionalize(hir)
        if args.emit_hir or args.debug:
            print("=== HIR ==="); print(hir); print()
        
        # Lower to MLIR
        lowering = MLIRLowering()
        mlir_module = lowering.lower_program(hir)
        if args.emit_mlir or args.debug:
            print("=== MLIR (before passes) ===")
            print(mlir_module); print()
        
        # Run pipeline
        if args.target == "cpu":
            pm = build_cpu_pipeline(mlir_module.context)
        else:
            pm = build_gpu_nvidia_pipeline(mlir_module.context)
        run_pipeline(mlir_module, pm, debug=args.emit_mlir_after or args.debug)
        
        if args.target == "cpu":
            executor = CPUExecutor(mlir_module)
            result = executor.execute_main([])
            print(format_result(result, typed_ast.body_type))
        else:
            # Generate PTX
            ptx, meta = generate_ptx(mlir_module, args.sm_version)
            if args.emit_ptx or args.debug:
                print("=== PTX ==="); print(ptx[:2000], "..."); print()
            
            # Execute
            runtime = CUDARuntime()
            executor = RemoraExecutor(ptx, meta, runtime)
            result = executor.execute("main", [])
            print(format_result(result, typed_ast.body_type))
    
    except RemoraError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

main()
```

#### 8.3 Result Display (`remora/display.py`)

```python
import numpy as np
from .types import *

def format_result(data: np.ndarray | float | int, ty: RemoraType) -> str:
    """Format a Remora result value for display."""
    match ty:
        case ScalarType("float"):
            v = float(data) if not isinstance(data, float) else data
            return f"{v:.6g}"
        case ScalarType("int"):
            return str(int(data))
        case ScalarType("bool"):
            return "true" if bool(data) else "false"
        case ArrayType(ScalarType("float"), (StaticDim(n),)):
            return "[" + "  ".join(f"{x:.6g}" for x in data) + "]"
        case ArrayType(ScalarType("int"), (StaticDim(n),)):
            return "[" + "  ".join(str(int(x)) for x in data) + "]"
        case ArrayType(_, shape) if len(shape) == 2:
            rows = [format_result(row, ArrayType(ty.element, shape[1:])) for row in data]
            return "[\n  " + "\n  ".join(rows) + "\n]"
        case _:
            # Generic numpy formatting
            return np.array2string(np.asarray(data), precision=6, separator="  ")
```

#### 8.4 AOT Tasks

- [ ] Implement `bin/remorac` with all flags
  - Current: the `remorac` console script supports the CPU-first user path: default CPU execution, `--emit-ast`, `--emit-typed-ast`, `--emit-hir`, `--emit-mlir`, `--emit-ptx`, and `--target mlir`/`--target ptx` aliases. Non-CPU execution flags, `--output`, `--emit-mlir-after`, GPU tuning flags, and broad debug mode are deferred.
- [x] Implement `format_result` for scalars, vectors, matrices, and higher-rank arrays
  - Current: `remora.display.format_result` handles int/float/bool scalars and rank-1 through rank-3 arrays with Remora-style scalar spelling.
- [ ] Implement error handling with source-location-annotated messages
  - Current: `remorac` catches parser/type/runtime/compiler errors, prints `remorac: ...` to stderr, and exits 1. Precise source spans remain deferred.
- [ ] Add `--output` flag: write PTX + a thin Python launcher script
- [ ] End-to-end test: `remorac tests/programs/tensor3_scale.rem` prints the expected rank-3 tensor
- [x] End-to-end test: `remorac --target cpu tests/programs/vector_scale.rem` works without GPU
  - Current: `tests/test_cli.py` runs every checked-in `examples/*.remora` through the `remorac` CPU entry point.

**Milestone M7**: generated programs execute through the shared executor API on CPU and NVIDIA.
Current: output formatting is shared by the interim CPU evaluator, `remorac`,
and the REPL. The final shared executor API is still deferred until the MLIR CPU
or direct Remora ABI execution path exists.

---

### Phase 9: Early REPL

**Goal**: Interactive `remora` REPL using the same compilation pipeline, supporting accumulated definitions and expression evaluation without incremental compilation.

The early REPL is not deferred behind GPU execution, fusion validation, or expanded language support. Implement it as soon as the parser, type checker, HIR/lowering path, and CPU executor can run the prototype expression subset. Each expression is compiled as a fresh temporary program containing all accumulated definitions plus a generated `main`; this is deliberately simple and keeps incremental compilation out of the critical path.

The first target is CPU-only. Add `gpu-nvidia` as a REPL target after the descriptor ABI and CUDA runtime are stable.

#### 9.1 REPL State (`remora/repl.py`)

```python
@dataclass
class ReplState:
    """All state maintained across REPL interactions."""
    type_env: TypeEnv                    # accumulated type environment
    hir_functions: list[HIRFunction]     # accumulated compiled function definitions
    compile_cache: dict[str, object]     # optional later cache; empty for first REPL
    cuda: CUDARuntime | None             # None when target is CPU
    target: str                          # "gpu-nvidia" | "cpu"
    sm_version: str                      # "sm_80" etc.
    debug: bool                          # print MLIR/PTX when True

def make_initial_state(target: str = "cpu", 
                       sm_version: str = "sm_80") -> ReplState:
    from .typechecker import make_prelude_env
    cuda = CUDARuntime() if target != "cpu" else None
    return ReplState(
        type_env=make_prelude_env(),
        hir_functions=[],
        compile_cache={},
        cuda=cuda,
        target=target,
        sm_version=sm_version,
        debug=False)
```

#### 9.2 Full-Program Recompilation

Each REPL expression is compiled as a self-contained program:

```python
def compile_and_execute(expr: TypedExpr, state: ReplState) -> np.ndarray:
    """
    Compile a typed expression (with all accumulated definitions) to a runnable kernel.
    Returns the result as a numpy array.
    """
    # Build a full HIRProgram: accumulated definitions + main expression
    hir = HIRProgram(
        functions=state.hir_functions,
        main=lower_expr_to_hir(expr),
        return_type=expr.type)
    
    hir = defunctionalize(hir)
    
    lowering = MLIRLowering()
    mlir_module = lowering.lower_program(hir)
    
    if state.debug:
        print("=== MLIR ==="); print(mlir_module)
    
    if state.target == "cpu":
        pm = build_cpu_pipeline(mlir_module.context)
        run_pipeline(mlir_module, pm)
        executor = CPUExecutor(mlir_module)
        return executor.execute_main([])
    else:
        pm = build_gpu_nvidia_pipeline(mlir_module.context)
        run_pipeline(mlir_module, pm)
        ptx, meta = generate_ptx(mlir_module, state.sm_version)
        if state.debug:
            print("=== PTX ==="); print(ptx[:1000])
        executor = RemoraExecutor(ptx, meta, state.cuda)
        return executor.execute("main", [])
```

#### 9.3 REPL Loop (`remora/repl.py`)

```python
import readline   # enables line history, Ctrl-A, Ctrl-E, etc.

REPL_COMMANDS = {
    ":quit", ":q", ":type", ":debug", ":target", ":load", ":reset", ":help"
}

class ReplSession:
    def __init__(self, target: str = "cpu", sm_version: str = "sm_80"):
        self.state = make_initial_state(target, sm_version)
        self._setup_readline()
    
    def _setup_readline(self):
        import os
        histfile = os.path.expanduser("~/.remora_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        import atexit
        atexit.register(readline.write_history_file, histfile)
    
    def _collect_full_input(self, first_line: str) -> str:
        """
        Accumulate lines until the expression is syntactically complete
        (balanced parentheses and brackets).
        """
        buf = first_line
        while not self._is_complete(buf):
            try:
                cont = input("...... ")
                buf = buf + "\n" + cont
            except (EOFError, KeyboardInterrupt):
                raise
        return buf
    
    def _is_complete(self, text: str) -> bool:
        """Heuristic: balanced parens/brackets and no trailing backslash."""
        depth_paren = text.count("(") - text.count(")")
        depth_bracket = text.count("[") - text.count("]")
        return depth_paren <= 0 and depth_bracket <= 0
    
    def eval_input(self, text: str) -> str | None:
        """Evaluate one REPL input. Returns string to print, or None."""
        text = text.strip()
        if not text: return None
        
        # Handle special commands
        if text.startswith(":"):
            return self._handle_command(text)
        
        try:
            item = parse_repl_input(text)
        except ParseError as e:
            return f"Parse error: {e}"
        
        try:
            if isinstance(item, (FuncDef, ValDef)):
                return self._process_definition(item)
            else:
                return self._process_expression(item)
        except RemoraError as e:
            return f"Error: {e}"
    
    def _process_definition(self, defn: FuncDef | ValDef) -> str:
        """Type-check and add a definition to the environment."""
        checker = TypeChecker(self.state.type_env)
        typed_defn = checker.check_definition(defn)
        
        # Update type environment
        if isinstance(defn, FuncDef):
            self.state.type_env = self.state.type_env.extend(
                defn.name, typed_defn.type)
        
        # Add to HIR function list
        hir_fn = lower_func_def_to_hir(typed_defn)
        self.state.hir_functions.append(hir_fn)
        
        return f"Defined: {defn.name} : {typed_defn.type}"
    
    def _process_expression(self, expr_ast) -> str:
        """Compile and execute an expression; return formatted result."""
        checker = TypeChecker(self.state.type_env)
        typed_expr, ty = checker.infer(expr_ast, self.state.type_env)
        
        result_arr = compile_and_execute(typed_expr, self.state)
        return format_result(result_arr, ty)
    
    def _handle_command(self, cmd: str) -> str:
        parts = cmd.split(None, 1)
        match parts[0]:
            case ":quit" | ":q":
                raise SystemExit(0)
            case ":help":
                return _HELP_TEXT
            case ":debug":
                self.state.debug = not self.state.debug
                return f"Debug mode: {'on' if self.state.debug else 'off'}"
            case ":target":
                if len(parts) < 2:
                    return f"Current target: {self.state.target}"
                new_target = parts[1].strip()
                if new_target not in ("gpu-nvidia", "cpu"):
                    return f"Unknown target: {new_target}"
                self.state.target = new_target
                if new_target != "cpu" and self.state.cuda is None:
                    self.state.cuda = CUDARuntime()
                return f"Target: {new_target}"
            case ":type":
                if len(parts) < 2:
                    return "Usage: :type <expr>"
                try:
                    ast = parse_repl_input(parts[1])
                    _, ty = TypeChecker(self.state.type_env).infer(ast, self.state.type_env)
                    return f"{parts[1]} : {ty}"
                except RemoraError as e:
                    return f"Error: {e}"
            case ":load":
                if len(parts) < 2:
                    return "Usage: :load <file.rem>"
                return self._load_file(parts[1].strip())
            case ":reset":
                from .typechecker import make_prelude_env
                self.state.type_env = make_prelude_env()
                self.state.hir_functions = []
                self.state.compile_cache = {}
                return "State reset."
            case _:
                return f"Unknown command: {parts[0]}. Type :help for help."
    
    def _load_file(self, path: str) -> str:
        """Load a Remora source file into the current session."""
        try:
            source = open(path).read()
        except OSError as e:
            return f"Cannot open {path}: {e}"
        prog = parse_file(path)
        lines = []
        for defn in prog.definitions:
            msg = self._process_definition(defn)
            lines.append(msg)
        if prog.body:
            result = self._process_expression(prog.body)
            lines.append(result)
        return "\n".join(lines)
    
    def run(self):
        """Main REPL loop."""
        print(f"Remora REPL  [target: {self.state.target}]")
        print("Type :help for commands, :quit to exit.\n")
        
        while True:
            try:
                line = input("remora> ")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
            
            try:
                full_input = self._collect_full_input(line)
            except (EOFError, KeyboardInterrupt):
                print()
                break
            
            result = self.eval_input(full_input)
            if result is not None:
                print(result)

_HELP_TEXT = """
Remora REPL commands:
  :quit, :q          Exit the REPL
  :type <expr>       Show the inferred type of an expression without executing
  :debug             Toggle debug output (MLIR, PTX dumps)
  :target <t>        Switch execution target: gpu-nvidia | cpu
  :load <file>       Load a .rem source file into this session
  :reset             Clear all definitions and return to initial state
  :help              Show this message

Examples:
  remora> let v = iota 10 in map (* 2) v
  [0  2  4  6  8  10  12  14  16  18]
  
  remora> def square x = x * x
  Defined: square : float -> float
  
  remora> map square [1.0, 2.0, 3.0]
  [1  4  9]
"""
```

#### 9.4 REPL Entry Point (`bin/remora`)

```python
#!/usr/bin/env python3
import argparse
from remora.repl import ReplSession

def main():
    p = argparse.ArgumentParser(description="Remora REPL")
    p.add_argument("--target", default="cpu",
                   choices=["gpu-nvidia", "cpu"])
    p.add_argument("--sm-version", default="sm_80")
    args = p.parse_args()
    
    session = ReplSession(target=args.target, sm_version=args.sm_version)
    session.run()

main()
```

#### 9.5 Tasks

- [x] Implement `ReplState` and `make_initial_state`
  - Current: CPU-only state stores accumulated value-definition source strings.
- [x] Implement CPU `compile_and_execute` for expressions using full-program recompilation
  - Current: expressions are evaluated through the interim typed-AST CPU evaluator using full temporary source programs. `:mlir` uses the compiler facade for MLIR inspection.
- [x] Implement `ReplSession.eval_input` dispatcher
- [x] Implement `_process_definition` (type-check, add to env, no execution)
  - Current: top-level value and function definitions are supported for the CPU evaluator. Function definitions are persisted as source and specialized at direct use sites.
- [x] Implement `_process_expression` (compile, run, display)
- [x] Implement `_collect_full_input` for multi-line continuation
- [x] Set up `readline` history
  - Current: enabled for interactive `remora`, disabled in tests.
- [x] Implement all REPL commands (`:type`, `:debug`, `:target`, `:load`, `:reset`, `:help`)
  - Current: `:mlir` is also implemented; non-CPU targets report a clear deferred message.
- [x] Implement `_load_file`
  - Current: loads current one-line top-level value/function definitions and evaluates the body if present.
- [ ] Add `gpu-nvidia` REPL target after the CUDA descriptor ABI is stable
- [ ] Add compile caching only if measured REPL latency makes it necessary
- [x] Write `tests/test_repl.py`:
  - Define a function; check it appears in type env
    - Current: tests cover persistent top-level function definitions, direct calls, function use as a `map` callable, and recursive-function deferral.
  - Evaluate a scalar expression; check result
  - Multi-line expression (two lines, parentheses)
  - `:type` command
  - `:load` a stdlib file
  - `:reset` clears definitions
  - Error recovery: bad expression followed by correct one

**Milestone M8**: `remora --target cpu` supports expression evaluation, top-level definitions, `:type`, `:load`, `:reset`, and error recovery for the prototype language subset. Dot-product examples can wait until the language subset includes the required rank-2 or zip/stdlib support.
Current: `remora --target cpu` is registered and covered by tests for the
implemented prototype language subset.

---

### Phase 10: Standard Library

**Goal**: Provide core Remora combinators that users expect, implemented in Remora itself.

Deferred until after the vertical slice. Before then, `iota`, scalar arithmetic, unary `map`, and scalar `fold` are compiler built-ins with direct tests.

#### 10.1 `stdlib/prelude.rem`

```remora
-- Arithmetic shorthands (if not already built-in as operators)
def add x y = x + y
def mul x y = x * y
def neg x = 0.0 - x
def abs x = if x < 0.0 then neg x else x

-- Reductions
def sum     = fold (+) 0.0
def product = fold (*) 1.0
def maximum = fold (\ a b -> if a > b then a else b) (neg 1e38)
def minimum = fold (\ a b -> if a < b then a else b) 1e38
def any     = fold (||) false
def all     = fold (&&) true
def count   = fold (\ n _ -> n + 1) 0

-- Array combinators
def zip f a b  = map (\ i -> f (a[i]) (b[i])) (iota (shape a)[0])
def zipwith    = zip   -- alias

-- Linear algebra
def dot a b    = sum (map (* ) (zip mul a b))
def norm v     = sqrt (dot v v)
def scale s v  = map (* s) v

-- Functional utilities  
def compose f g x = f (g x)
def flip f x y    = f y x
def const x _     = x
def id x          = x
def twice f x     = f (f x)
def on f g x y    = f (g x) (g y)
```

#### 10.2 Built-In Operations That Require Compiler Support

Some operations need direct support in the lowering pass (not just sugar):

| Operation | Remora syntax | MLIR lowering |
|---|---|---|
| `iota n` | Built-in | `linalg.generic` with `linalg.index` |
| `shape arr` | Built-in | `tensor.dim` for each dimension |
| `rank arr` | Built-in | Constant from type info |
| `transpose arr` | Built-in | `linalg.transpose` |
| `reshape arr dims` | Built-in | `tensor.reshape` |
| `slice arr start end` | Built-in | `tensor.extract_slice` |
| `cat a b` | Built-in | `tensor.concat` |
| `arr[i, j]` | Indexing syntax | `tensor.extract` |

These are implemented directly in `lowering.py` as special cases in `_lower_expr`.

#### 10.3 Loading the Standard Library

The prelude is automatically loaded at startup in both `remorac` and `remora`:

```python
def _build_prelude_env() -> TypeEnv:
    """Load and type-check stdlib/prelude.rem; return resulting type environment."""
    prelude_path = Path(__file__).parent.parent / "stdlib" / "prelude.rem"
    prog = parse_file(str(prelude_path))
    checker = TypeChecker()
    typed = checker.check(prog)
    return typed.final_env  # TypeEnv after all definitions
```

#### 10.4 Tasks

- [ ] Write `stdlib/prelude.rem` with combinators listed above
- [ ] Implement `iota` lowering in `lowering.py`
- [ ] Implement `shape` / `rank` lowering using `tensor.dim` and compile-time rank constant
- [ ] Implement `transpose` lowering using `linalg.transpose` or swapped affine map
- [ ] Implement array indexing lowering using `tensor.extract`
- [ ] Load prelude automatically in `TypeChecker.__init__` / startup
- [ ] Test prelude functions end-to-end: `sum (iota 10)` → `45`, `dot [1,2,3] [4,5,6]` → `32`

---

### Phase 11: Dynamic Shape and Rank Roadmap

**Goal**: Extend beyond Dense Core's static shape/rank restrictions without losing the predictable GPU path.

Dynamic dimensions and dynamic rank are separate features. Dynamic dimensions are the natural next step; fully dynamic rank is much harder because MLIR `linalg.generic` requires the number of loop dimensions to be known when the operation is built.

#### 11.1 Stage A: Static Rank, Dynamic Dimensions

Support types such as `float[?, 3]` or `float[m, n]` where the rank is known but one or more extents are runtime values.

Required changes:

- Reintroduce `DynDim` or a real `ShapeExpr` representation in `types.py`
- Add shape constraints and equalities, e.g. two operands of `zip` must have equal frame shapes
- Pass dynamic extents as `index` values into MLIR functions
- Lower arrays to `tensor<?x?xf32>` / `memref<?x?xf32>` where needed
- Extend descriptor construction so runtime sizes come from values, not only type constants
- Keep one generated MLIR function per static rank

This stage should still use standard `tensor`/`linalg` MLIR because dynamic extents are well supported when rank is fixed.

#### 11.2 Stage B: Static Rank With Symbolic Shape Constraints

Add a small shape constraint solver before attempting dynamic rank:

```text
m == n
shape(a) == shape(b)
rank(a) == 2
frame(a) == frame(b)
```

The solver should operate before HIR lowering and produce either:

- concrete static dimensions,
- runtime dimension parameters with equality constraints, or
- a typed error with source locations.

This stage is where Remora's shape-polymorphic typing starts to become explicit in the implementation.

#### 11.3 Stage C: Bounded Dynamic Rank Dispatch

Support values whose rank is known only at runtime by dispatching to rank-specialized implementations for a bounded range:

```text
case rank(x):
  0 -> call f_rank0(x)
  1 -> call f_rank1(x)
  2 -> call f_rank2(x)
  3 -> call f_rank3(x)
  _ -> error unsupported rank
```

Required changes:

- Add a runtime array descriptor that includes `rank`, pointers to `sizes`, and pointers to `strides`, or use a fixed-maximum-rank descriptor
- Generate rank-specialized MLIR functions for each supported rank
- Insert runtime dispatch in the host/executor layer or in generated wrapper code
- Preserve GPU kernels as rank-specialized kernels; do not attempt unbounded-rank `linalg.generic`

This is the recommended first implementation of dynamic rank.

#### 11.4 Stage D: Fully Dynamic Rank

Fully unbounded dynamic rank requires generic runtime loops over packed descriptors or a kernel template that interprets shape/stride arrays. This is substantially less compatible with MLIR `linalg` optimization and should be treated as a research/runtime project after the bounded-dispatch path has proven useful.

Required changes:

- Packed dynamic-rank value representation
- Runtime loop/indexing helpers
- Generic GPU kernels or generated kernels with loop nests over rank metadata
- New optimization strategy, since `linalg` fusion and static iterator reasoning are no longer directly available

#### 11.5 Tasks

- [ ] Add `ShapeExpr`, `DynDim`, and shape equality constraints
- [ ] Extend type checking for static-rank dynamic-dimension programs
- [ ] Extend MLIR lowering for `tensor<?x...>` with static rank
- [ ] Extend `docs/ABI.md` with dynamic-dimension descriptor rules
- [ ] Implement bounded dynamic-rank host dispatch for rank 0 through rank 3
- [ ] Add tests for dynamic dimensions, shape equality failures, and rank dispatch

**Milestone M12**: Static-rank dynamic-dimension programs execute on CPU and NVIDIA for rank 1 through rank 3.

**Milestone M13**: Bounded dynamic-rank dispatch chooses rank-specialized kernels for rank 0 through rank 3.

---

### Phase 12: Automatic Differentiation Extension

**Goal**: Add automatic differentiation as a typed HIR transformation that runs before MLIR lowering and therefore works for both CPU and GPU backends.

AD is not a runtime feature. It is a compiler pass that transforms typed, rank-resolved Remora/HIR functions into derivative HIR functions.

#### 12.1 AD Surface

Initial built-ins:

```remora
grad f
value_and_grad f
jvp f
vjp f
```

Staged support:

1. `grad` for `float -> float`
2. `grad` for scalar-output functions over `float` arrays
3. `value_and_grad`
4. `jvp` and `vjp`
5. `jacobian` after the basic AD representation is stable

#### 12.2 Differentiability Rules

Differentiable:

- `float`
- dense arrays whose elements are `float`

Non-differentiable:

- `int`
- `bool`
- shape values
- indices
- function values

Mixed functions are allowed only when gradients flow through differentiable inputs. Non-differentiable values may be used as static parameters or control/shape data, but the AD pass must reject attempts to differentiate with respect to them.

#### 12.3 HIR-Level AD Strategy

Implement AD over typed HIR, after defunctionalization/monomorphization and before MLIR lowering.

Forward mode is the first proof:

```python
def jvp_expr(expr: HIRExpr, tangent_env: dict[str, HIRExpr]) -> tuple[HIRExpr, HIRExpr]:
    """Return (primal, tangent) HIR."""
    ...
```

Reverse mode is needed for scalar-output functions over arrays:

```python
def reverse_ad_function(fn: HIRFunction, wrt: list[str]) -> HIRFunction:
    """Generate a gradient function for differentiable inputs."""
    ...
```

Generated gradient HIR must pass through the same lowering and execution pipeline as ordinary Remora programs.

#### 12.4 Primitive Rules

Required first rules:

| Primitive | Derivative |
|---|---|
| `x + y` | `dx + dy` |
| `x - y` | `dx - dy` |
| `x * y` | `dx * y + x * dy` |
| `x / y` | `(dx * y - x * dy) / (y * y)` |
| `neg x` | `neg dx` |
| `exp x` | `exp x * dx` |
| `log x` | `dx / x` |
| `sqrt x` | `dx / (2 * sqrt x)` |

Comparisons, boolean operators, and integer operations are non-differentiable.

#### 12.5 Array Combinator Rules

Required first rules:

- `map`: differentiate elementwise; result tangent has the same frame shape
- `fold (+) 0.0`: reverse-mode gradient broadcasts/scatters the output adjoint to each element
- `sum`: derivative is an array of ones with the input shape
- `iota`: non-differentiable
- explicit casts: `int -> float` has no gradient with respect to the integer source
- views/transposes/slices: deferred until the view syntax lands; reverse mode must route adjoints through inverse view/scatter behavior

Reverse-mode indexing and slicing require scatter-add or equivalent accumulation. Do not add them until mutable buffer lowering or a functional scatter representation is designed.

#### 12.6 Tasks

- [ ] Add `remora/ad.py`
- [ ] Add differentiability classification to `types.py`
- [ ] Implement forward-mode HIR AD for scalar `float -> float`
- [ ] Implement forward-mode through `map`
- [ ] Implement reverse-mode for scalar-output functions over dense float arrays
- [ ] Implement AD rules for primitive float arithmetic and elementary math
- [ ] Implement `grad` and `value_and_grad` in the type checker and HIR lowering
- [ ] Add CPU tests first, then GPU tests for generated gradient programs
- [ ] Add diagnostics for non-differentiable values and unsupported primitives

**Milestone M14**: `grad` works for scalar `float -> float` functions on CPU.

**Milestone M15**: `grad` works for scalar-output functions over rank-1 through rank-3 dense float arrays and lowers to GPU.

---

### Phase 13: Testing and Verification

#### 13.1 Unit Test Coverage Targets

| Module | Target coverage | Key test cases |
|---|---|---|
| `parser.py` | 95% | All grammar rules; error recovery |
| `typechecker.py` | 90% | All typing rules; shape arithmetic; lifting inference |
| `defunc.py` | 85% | Static HOF; monomorphization |
| `lowering.py` | 80% | Each HIR node type → expected MLIR patterns |
| `pipeline.py` | 70% | CPU pipeline passes; GPU pipeline passes; PTX validity |
| `runtime.py` | 60% | H2D/D2H correctness; kernel launch |
| `repl.py` | 75% | All commands; definition/expression discrimination |

#### 13.2 Dense Core Acceptance Suite

Create a language-level acceptance suite in `tests/acceptance/`. This suite is distinct from unit tests and MLIR pattern tests: each case is a Remora source program plus either expected stdout or an expected diagnostic.

The original Remora project does not appear to provide a packaged compiler conformance suite. Use it as a source of semantic examples, then translate only the parts that fit Remora Dense Core:

- Original repository: <https://github.com/jrslepak/Remora>
- Redex semantic model README: <https://github.com/jrslepak/Remora/blob/master/semantics/Readme.md>
- Dynamic Racket semantics with embedded `module+ test` examples: <https://github.com/jrslepak/Remora/blob/master/remora/dynamic/lang/semantics.rkt>
- Tutorial/reference examples in this repo: `docs/remora-reference/remora-tutorial-draft.txt`
- Formal semantics reference in this repo: `docs/remora-reference/semantics-of-rank-polymorphism.txt`
- Dissertation examples and typing discussion in this repo: `docs/remora-reference/slepak-dissertation.txt`

Acceptance directory layout:

```
tests/acceptance/
├── pass/
│   ├── scalar_add.rem
│   ├── vector_iota.rem
│   ├── rank1_map_float.rem
│   ├── rank1_map_iota_promote.rem
│   ├── rank2_map_scalar.rem
│   ├── rank3_map_scalar.rem
│   ├── fold_sum_int.rem
│   ├── fold_sum_float.rem
│   ├── vector_cell_map_rows.rem
│   ├── nested_map.rem
│   ├── static_lambda.rem
│   └── static_hof_param.rem
├── fail/
│   ├── ragged_array.rem
│   ├── rank4_rejected.rem
│   ├── dynamic_dim_rejected.rem
│   ├── dynamic_function_rejected.rem
│   ├── fold_accumulator_mismatch.rem
│   ├── map_non_function.rem
│   ├── map_cell_rank_mismatch.rem
│   └── unbound_variable.rem
├── deferred/
│   ├── dot_product.rem
│   ├── matrix_multiply.rem
│   ├── transpose_view.rem
│   ├── slice_view.rem
│   ├── dynamic_shape.rem
│   └── dynamic_rank.rem
└── manifest.toml
```

`manifest.toml` records expected behavior:

```toml
[[case]]
path = "pass/rank2_map_scalar.rem"
target = "cpu"
expect_stdout = "[[2 4]\n [6 8]]"

[[case]]
path = "fail/rank4_rejected.rem"
target = "typecheck"
expect_diagnostic_contains = "rank limit exceeded in Dense Core"
```

Acceptance tests must run on CPU by default. GPU acceptance runs use the same cases but are gated by `REMORA_TEST_GPU=1`.

Tasks:

- [ ] Create `tests/acceptance/pass`, `tests/acceptance/fail`, and `tests/acceptance/deferred`
- [ ] Create `tests/acceptance/manifest.toml`
- [ ] Add `tests/test_acceptance.py` to load the manifest and run `remorac --target cpu`
- [ ] Add GPU acceptance mode gated by `REMORA_TEST_GPU=1`
- [ ] Mine Dense Core-compatible examples from the tutorial, Redex README, and Racket `module+ test` blocks
- [ ] Preserve links to the source example in comments or manifest metadata when a case is derived from upstream material
- [ ] Keep `deferred/` cases in the tree but do not fail CI for them until the corresponding feature phase is active

#### 13.3 End-to-End Test Programs

```
tests/programs/
├── 01_scalar_literal.rem        def main = 42
├── 02_float_arith.rem           def main = 3.14 * 2.0
├── 03_vector_iota.rem           def main = iota 10
├── 04_vector_scale.rem          def main = map (* 2.0) (iota 10)
├── 05_vector_sum.rem            def main = fold (+) 0.0 (iota 10)
├── 06_matrix_scale.rem          def main = map (* 2.0) [[1,2],[3,4]]
├── 07_tensor3_scale.rem         rank-3 scalar map
├── 08_vector_cell_map.rem       map sum over rows of a matrix
├── 09_hof_lambda.rem            map (\x -> x * x + 1.0) (iota 5)
├── 10_hof_param.rem             def apply f x = f x; apply (* 2.0) 3.0
├── 11_nested_map.rem            map (* 2.0) (map (+ 1.0) (iota 100))
├── 12_large_vector.rem          fold (+) 0.0 (map (* 2.0) (iota 10000000))
├── 13_repl_session.rem          REPL session transcript (used by test_repl.py)
└── deferred/
    ├── dot_product.rem          requires stdlib zip/dot policy
    └── matmul.rem               requires matrix-specific lowering or stdlib expansion
```

#### 13.4 Critical Correctness Tests

```python
def test_map_chain_fusion():
    """
    map f (map g arr) must produce a SINGLE linalg.generic after fusion,
    not two. This verifies that intermediate arrays are eliminated.
    """
    source = "map (* 2.0) (map (+ 1.0) (iota 100))"
    mlir_before_passes = compile_to_pre_pass_mlir(source)
    mlir_after_fusion = run_fusion_pass_only(mlir_before_passes)
    
    # Count linalg.generic ops: should be exactly 1 after fusion
    assert mlir_after_fusion.count('linalg.generic') == 1, \
        "Map chain fusion failed: more than one linalg.generic remains"

def test_fold_after_map_fusion():
    """
    fold f (map g arr) should fuse: no intermediate array for the mapped values.
    """
    source = "fold (+) 0.0 (map (* 2.0) (iota 100))"
    mlir = compile_to_pre_pass_mlir(source)
    fused = run_fusion_pass_only(mlir)
    assert fused.count('linalg.generic') == 1

def test_numerical_correctness_dot_product():
    """Deferred until the stdlib defines dot/zip for Dense Core."""
    result = run_remorac("dot [1.0,2.0,3.0] [4.0,5.0,6.0]")
    assert abs(float(result) - 32.0) < 1e-5

def test_large_vector_sum():
    """Verify correctness on 10M elements — catches integer overflow in grid sizing."""
    result = run_remorac("fold (+) 0.0 (map (* 1.0) (iota 10000000))")
    expected = 10000000 * 9999999 / 2  # sum of 0..9999999
    assert abs(float(result) - expected) / expected < 1e-4
```

---

## 7. Key Design Decisions

### 7.0 Prototype Execution Model and ABI

The first prototype has exactly one GPU execution path:

1. Lower Remora to MLIR device code.
2. Generate PTX.
3. Load PTX with `cuda-python`.
4. Launch exported adapter kernels manually with ctypes-packed descriptor pointers following `docs/ABI.md`.

The MLIR host-side `gpu.launch_func` path is deferred. This avoids mixing two incompatible launch models and makes the runtime boundary testable. If a later implementation uses MLIR-generated host code, it must replace `docs/ABI.md` deliberately rather than coexist with the manual launcher.

### 7.1 Static Rank (Monomorphization) for the Prototype

All ranks are resolved at compile time by the type checker. Each `(function, rank)` combination generates a separate MLIR function. This means a rank-2 `map` and a rank-3 `map` produce different `linalg.generic` operations, and both are compiled in the same MLIR module.

Dynamic rank (rank as a runtime value) is deferred. If needed later, it can be handled via the "packed 1D memref + shape struct" encoding described in `MLIR_ARCHITECTURE.md`.

### 7.2 Immutable Functional Semantics

Remora arrays are immutable. The MLIR `tensor` dialect enforces this at the type level (SSA values; updates produce new tensors). Bufferization inserts `alloc`/`dealloc` automatically at the correct liveness points. Users never manage memory.

After fusion, most intermediate arrays (those consumed only by the next operation in a chain) are eliminated before any allocation is inserted — they exist only as registers.

### 7.3 Numeric Types

For the prototype:
- `float` → `f32` (single precision)
- `int` → `i32` (32-bit signed)
- `bool` → `i1`
- `iota` returns `int[n]`
- mixed `int`/`float` arithmetic inserts explicit `int -> float` casts
- `/` returns `float`

`f64` and `i64` can be added by extending `lower_type` with a type flag. This is trivial to implement later.

### 7.4 GPU Thread Configuration

Fixed block size of 256 threads for flattened 1D iteration; 16×16 blocks for native 2D iteration; 8×8×4 blocks for native 3D iteration if the selected MLIR mapping preserves three loop dimensions. Grid size is computed from the output descriptor sizes. MLIR's `-gpu-map-parallel-loops` handles the mapping after the pipeline is validated.

Tiled shared memory optimization (for reductions and matrix operations) is a Phase 5 improvement in the `MLIR_ARCHITECTURE.md` plan — not required for a correct prototype.

### 7.5 REPL Compilation Latency

Each REPL expression re-compiles all accumulated definitions plus the new expression. For a session with few definitions (the common case in early exploration), this is acceptable. If compile latency becomes an issue in long sessions, cache compiled CPU artifacts or PTX by function name and content hash; only recompile definitions that have changed.

### 7.6 Error Handling

All compiler phases raise `RemoraError` (a typed exception hierarchy) rather than crashing with Python tracebacks. The REPL catches `RemoraError` and displays a user-friendly message with source location before continuing. The `remorac` binary catches `RemoraError`, prints it to stderr, and exits with code 1.

---

## 8. Milestone Summary

### 8.1 Phase Outcomes

This table describes the practical state of the system after each phase. "User can do" means the workflow is available without reaching into internal compiler objects, unless explicitly described as a developer workflow.

| Phase | What exists after completion | User can do |
|---|---|---|
| Phase 0: Infrastructure and Hello World | Python project skeleton, dependency checks, CPU MLIR hello-world, and a manually launched CUDA/PTX smoke test | Confirm the local machine can import MLIR/CUDA packages and run a trivial generated kernel |
| Phase 1: Parser and AST | Remora grammar, AST nodes, parser entry points for programs/definitions/expressions, and parser tests | Parse source containing literals, arrays, lambdas, `let`, `iota`, unary `map`, scalar `fold`, and top-level definitions |
| Phase 2: Type System and Shape Inference | Static shape evaluation, rank-0..3 type representation, bidirectional checking for lambdas/operator sections, numeric casts, and rank-limited map/fold typing | Ask the compiler to type-check Dense Core programs and get useful diagnostics for mismatched shapes, rank-4 values, non-constant dimensions, or fold accumulator errors |
| Phase 3: High-Level IR (HIR) | A typed, simplified HIR with explicit shapes and desugared primitive operations | Inspect a compiler-friendly representation of parsed and typed programs before MLIR lowering |
| Phase 4: Defunctionalization | Static lambda lifting and rejection of deferred dynamic higher-order patterns | Compile maps/folds using inline lambdas or named functions without runtime function pointers |
| Phase 5: HIR to MLIR Linalg Lowering | Verified MLIR generation for `iota`, rank-0..3 scalar maps, static frame/cell maps up to rank 3, scalar folds, and map-fold composition | Emit MLIR for Dense Core programs and validate it against golden fixtures plus the MLIR verifier |
| Phase 6: MLIR Pass Pipeline | Pinned and validated LLVM/MLIR 18 CPU and NVIDIA lowering pipelines | Lower generated MLIR through CPU LLVM or NVIDIA PTX-producing pipelines with reproducible pass behavior |
| Phase 7: Runtime and Execution Engine | CUDA runtime wrapper, rank-0..3 descriptor ABI packing, CPU executor, and single-kernel execution support | Execute generated kernels and copy results back to host memory through the documented Remora Dense Core ABI |
| Phase 8: User-Facing Entry Points | Shared result display, CLI plumbing for `remorac`, and REPL entry point scaffold | Exercise the same execution/display path from command-line tools instead of internal compiler objects |
| Phase 9: Early REPL | CPU-first, non-incremental `remora` interactive session reusing the compiler pipeline | Enter expressions interactively, define simple functions, inspect types, load files, reset state, and recover from errors without restarting |
| Phase 10: Standard Library | Initial `prelude.rem` plus direct lowering support for required built-ins | Use named library functions such as `sum`, `scale`, and eventually `dot` instead of spelling every program with raw `map`/`fold` |
| Phase 11: Dynamic Shape and Rank Roadmap | Static-rank dynamic dimensions, shape constraints, bounded dynamic-rank dispatch, and a documented path toward fully dynamic rank | Run programs with runtime extents first, then dispatch dynamic-rank values to rank-specialized kernels |
| Phase 12: Automatic Differentiation Extension | HIR-level AD pass, differentiability rules, primitive derivative rules, and generated gradient functions | Use `grad`/`value_and_grad` for scalar and dense-array float programs on CPU and later GPU |
| Phase 13: Testing and Verification | Unit, acceptance, pipeline, fusion, runtime, CLI, AD, dynamic-shape, and end-to-end tests with CPU-by-default and GPU-gated suites | Run the test suite to verify language behavior, parser/type/lowering correctness, CPU parity, GPU execution, AD behavior, dynamic-shape behavior, and fusion expectations |

### 8.2 Milestones

| # | Milestone | Deliverable | Verifiable by |
|---|---|---|---|
| M0 | Infrastructure ready | CPU and CUDA toolchains import; one hand-written PTX launch works | `pytest tests/test_infra.py` |
| M1 | Parser slice complete | Parse definitions, expressions, operator sections, `iota`, unary `map`, scalar `fold` | `pytest tests/test_parser.py` |
| M2 | Type checker slice complete | Resolve rank-0..3 Dense Core map/fold, insert numeric casts, and reject bad shapes, rank-4 values, and non-constant dimensions | `pytest tests/test_typechecker.py` |
| M3 | HIR slice complete | Lower only static lambdas/named functions needed by map/fold | `pytest tests/test_hir.py` |
| M4 | MLIR lowering slice complete | Produce verified MLIR for `iota`, rank-1/2/3 maps, static frame/cell maps, scalar fold, and explicit casts | `pytest tests/test_lowering.py` |
| M5 | Pinned pipelines complete | Validated CPU and NVIDIA LLVM-18 pipelines are committed | `python tools/validate_mlir_pipeline.py` |
| M6 | ABI/runtime complete | Rank-0..3 descriptor ABI tests pass on CPU and CUDA where available | `pytest tests/test_abi.py && pytest tests/test_execution.py` |
| M7 | Shared execution/display complete | Generated programs run through the executor API and format scalar/vector/matrix/rank-3 results consistently | `pytest tests/test_execution.py` |
| M8 | Early REPL | Interactive expression evaluation, definitions, `:type`, `:load`, `:reset`, and error recovery on CPU | `remora --target cpu` |
| M9 | AOT vertical slice complete | `remorac` runs `fold (+) 0.0 (map (* 2.0) (iota 1000))` on CPU and NVIDIA | `remorac tests/programs/iota_map_fold.rem` |
| M9.5 | Dense Core acceptance suite | Pass/fail acceptance manifest covers Dense Core language behavior on CPU | `pytest tests/test_acceptance.py` |
| M10 | Fusion checks | Map chains and map-fold avoid materialized intermediates where MLIR supports it | `pytest tests/test_fusion.py` |
| M11 | Expanded language | Stdlib subset, dot product, transpose/slice syntax over the existing view-capable ABI | `remorac tests/programs/dot_product.rem` |
| M12 | Static-rank dynamic dimensions | Rank-1/2/3 programs with runtime extents compile and execute | `pytest tests/test_dynamic_shapes.py` |
| M13 | Bounded dynamic rank | Runtime rank dispatch chooses rank-specialized kernels for rank 0..3 | `pytest tests/test_dynamic_rank.py` |
| M14 | Scalar AD | `grad` works for scalar `float -> float` functions on CPU | `pytest tests/test_ad.py -k scalar` |
| M15 | Array AD on GPU | `grad` works for scalar-output rank-1/2/3 dense float array functions and lowers to GPU | `pytest tests/test_ad.py` |

---

## 9. Risk Areas and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Runtime ABI mismatch between MLIR-lowered kernels and CUDA launcher | High | High | Use `docs/ABI.md`; write descriptor layout and adapter-kernel tests for rank 0..3 before launching generated kernels |
| Manual CUDA launch path diverges from MLIR host `gpu.launch_func` path | Medium | High | Prototype supports only manual PTX launch; generated host launch is deferred |
| MLIR Python bindings API drift between LLVM versions | Medium | High | Pin LLVM version in `pyproject.toml`; test against one pinned wheel |
| Pass pipeline names/options differ from examples | High | High | Validate with `mlir-opt` and commit the known-good LLVM-18 pipeline artifact |
| GPU pipeline lowering produces incorrect PTX | Medium | High | Always verify on CPU pipeline first; use `--emit-mlir-after` to bisect failures |
| Bufferization failures for unusual Remora patterns | Medium | Medium | Use `-allow-return-allocs-from-loops` flag; fallback to explicit copy insertion |
| Affine fusion does not fire (pass preconditions not met) | Medium | Low | Fusion is an optimization; correctness doesn't depend on it; verify separately |
| CUDA driver initialization fails on CI (no GPU) | High (CI) | Low | `--target cpu` path for all CI tests; GPU tests run only with `REMORA_TEST_GPU=1` |
| Defunctionalization misses a pattern | Low-Medium | Medium | Start with static-only (error on dynamic HOF); add dynamic dispatch incrementally |
| Grid size integer overflow for very large arrays | Low | Medium | Use `int64` for element counts; test with arrays >2^31 elements |
| REPL compile latency unacceptable (> 5 seconds) | Low (small programs) | Low | Add content-hash compile caching; profile and optimize the Python hot path |

---

## 10. Development Workflow

### 10.1 Recommended Phase Order

Given the dependency structure, implement phases in this order:

```
Phase 0 (Infrastructure)
    │
    ├── Phase 1 (Parser)
    │       │
    │       └── Phase 2 (Type Checker)
    │               │
    │               ├── Phase 3 (HIR)
    │               │       │
    │               │       └── Phase 4 (Defunc)
    │               │               │
    │               │               └── Phase 5 (MLIR Lowering)
    │               │                       │
    │               │                       └── Phase 6 (Pipeline) → Phase 7 (Runtime)
    │               │                                                       │
    │               └── (feed back to fix type errors found during lowering) │
    │                                                                        │
    └───────────────────────────────────────────────────────────────────────►
                                                                    Phase 8 (Entry points)
                                                                    Phase 9 (Early REPL)
                                                                    Phase 10 (Stdlib)
                                                                    Phase 11 (Dynamic shapes/rank)
                                                                    Phase 12 (AD extension)
                                                                    Phase 13 (Tests)
```

### 10.2 Implementation Notes Maintenance

Keep `docs/IMPLEMENTATION_NOTES.md` up to date during implementation. Every phase
or meaningful design change must update that file in the same change set as the
code/tests. Record concrete implementation decisions, deviations from this plan,
known limitations, and deferred work. Do not leave important decisions only in
chat history, commit messages, or local memory.

### 10.3 Testing Strategy Per Phase

- **Always implement tests alongside the code** — do not defer testing.
- After each phase, run `pytest` on all existing tests before moving on.
- Use `--target cpu` during Phases 1–6 to avoid needing a GPU for development.
- Add a GPU test suite gated by `REMORA_TEST_GPU=1` environment variable.

### 10.4 Debugging Tools

During development, the following are invaluable:

```bash
# Dump MLIR at every pass boundary
remorac --emit-mlir-after prog.rem 2>&1 | less

# Dump PTX and inspect
remorac --emit-ptx prog.rem > out.ptx
ptxas --sm-version 80 out.ptx  # verify PTX syntax

# Run on CPU (no GPU, use Python debugger)
remorac --target cpu prog.rem
python -m pdb -c continue bin/remorac tests/programs/tensor3_scale.rem

# MLIR verifier (run automatically by pipeline but can be forced)
mlir-opt --verify-diagnostics out.mlir

# REPL in CPU mode (no GPU needed)
remora --target cpu
```

---

*This plan is intended to be a living document. Update it as implementation proceeds and new design decisions are made.*
