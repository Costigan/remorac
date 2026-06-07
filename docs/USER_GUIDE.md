# Remora Dense Core User Guide

## Installation

```bash
git clone <repo>
cd remorac
uv sync
```

Verify the toolchain (optional, needed for GPU and threaded CPU):

```bash
# Checks iree-compile, ptxas, CUDA availability
uv run python tools/validate_mlir_toolchain.py
```

## Quick Start

### REPL

```bash
uv run remora
```

```
Remora REPL [target: cpu]
> 1 + 2
3
> map (* 2) [1, 2, 3]
[2, 4, 6]
> :type map (* 2) (iota 5)
int[5]
> :target interp           # switch to interpreter
> fold (+) 0 (iota 5)
10
```

### Compile and run a file

```bash
uv run remorac --target cpu program.remora
uv run remorac --target gpu-nvidia program.remora    # validate GPU compilation
```

## Language Reference (Dense Core subset)

### Scalar Types

`int` (i32), `float` (f32), `bool`

### Operators

`+` `-` `*` `/` `<` `<=` `==` `!=` `&&` `||`

### Expressions

```remora
-- literals
42
3.14
true
[1, 2, 3]
[[1.0, 2.0], [3.0, 4.0]]

-- arithmetic
1 + 2 * 3
(1.0 + 2.0) / 3.0

-- let bindings
let x = 5 in x + 1

-- if (scalar or tensor condition)
if true then 1 else 2
if [true, false] then [1, 2] else [10, 20]

-- iota (integer range)
iota 5          -- [0, 1, 2, 3, 4]

-- map (element-wise)
map (* 2) [1, 2, 3]              -- unary
map (+) [1, 2] [10, 20]           -- binary
map (\x -> x + 1) [1, 2]          -- lambda

-- fold (reduction)
fold (+) 0 [1, 2, 3]              -- scalar result (sum)
fold (+) [0, 0] [[1, 2], [3, 4]]  -- array-cell fold

-- indexing
xs[0]
xs[1 + 2]                         -- dynamic index
xs[0, 1]                          -- full-rank literal index

-- views
reverse [1, 2, 3]                 -- [3, 2, 1]
transpose [[1, 2], [3, 4]]        -- [[1, 3], [2, 4]]
reshape [1, 2, 3, 4] [2, 2]       -- [[1, 2], [3, 4]]
ravel [[1, 2], [3, 4]]            -- [1, 2, 3, 4]
take 2 [1, 2, 3, 4]               -- [1, 2]
drop 2 [1, 2, 3, 4]               -- [3, 4]

-- shape and rank
shape [1, 2, 3]                   -- [3]
rank [1, 2, 3]                    -- 1
```

### Cell Maps

```remora
-- reduce each row (cell = rank-1 sub-array)
map (\row -> fold (+) 0 row) [[1, 2], [3, 4]]

-- access cell elements by index
map (\row -> row[0] + row[1]) [[1, 2], [3, 4], [5, 6]]
```

### Top-level definitions

```remora
-- Named functions (compiled to JIT .so or GPU PTX)
def add x y = x + y
def double x = map (* 2) x
def sum xs = fold (+) 0.0 xs

-- Use a definition
add 1 2
```

### Prelude functions

Built-in functions available in all programs:

| Category | Functions |
|----------|-----------|
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `id`, `const` |
| Reductions | `sum`, `product` |
| Vector | `scale`, `dot` |
| Aggregate | `max`, `min` |
| Boolean | `any`, `all` |
| Identity | `abs` |

## CLI Reference

```
remorac [options] file.remora
```

### Targets

| Flag | Description |
|------|-------------|
| `--target cpu` | Compile and JIT-execute on CPU (default) |
| `--target interp` | Run through the typed-AST interpreter |
| `--target gpu-nvidia` | Validate GPU compilation to PTX |
| `--target mlir` | Print lowered MLIR and exit |
| `--target ptx` | Print compiled PTX and exit |

### Emit / inspect

| Flag | Description |
|------|-------------|
| `--emit-ast` | Print parsed AST |
| `--emit-typed-ast` | Print type-checked AST |
| `--emit-hir` | Print defunctionalized HIR |
| `--emit-mlir` | Print lowered MLIR module |
| `--emit-ptx` | Print generated PTX (GPU only) |

### CPU performance

| Flag | Description |
|------|-------------|
| `--cpu-threads N` | Use N OpenMP threads (also reads `REMORA_NUM_THREADS`) |
| `--cpu-vectorize` | Enable affine/vector CPU lowering |
| `--no-cpu-vectorize` | Disable vectorization (default) |

### GPU function calling

```bash
# Call a named function with .npy array inputs
remorac --target gpu-nvidia --call scale --input xs.npy program.remora
```

## REPL Reference

```
uv run remora [--target cpu|interp|gpu-nvidia]
```

### Commands

| Command | Description |
|---------|-------------|
| `:target [name]` | Show or set target (`cpu`, `interp`, `gpu-nvidia`) |
| `:type <expr>` | Infer and print the type of an expression |
| `:mlir <expr>` | Print lowered MLIR for an expression |
| `:load <file>` | Load definitions from a `.remora` file |
| `:defs` | List user-defined functions |
| `:prelude` | List built-in prelude functions |
| `:reset` | Clear all user definitions |
| `:help` | Show command summary |
| `:quit` | Exit |

## GPU (NVIDIA)

### Requirements

- NVIDIA GPU with CUDA driver (RTX series tested)
- `ptxas` from CUDA toolkit (auto-detected at `/usr/local/cuda/bin/ptxas`)
- `iree-compile` from IREE for MLIR-based compilation

### Supported operations

| Operation | Element types | Ranks |
|-----------|---------------|-------|
| Element-wise maps (unary/binary) | `f32`, `i32`, `bool` | 1–10 |
| Scalar reductions | `f32`, `i32` | 1–10 |
| Bool maps (byte-backed `i8` ABI) | `bool` | 1–10 |

### Unsupported on GPU

- View operations (transpose, slice, reshape, ravel, take, drop, reverse)
- Full-rank or partial indexing
- Dynamic index expressions
- Cell maps
- Lambda functions with closures
- Higher-order functions (compose, flip)

Unsupported GPU programs emit a clear diagnostic rather than silently falling back to CPU.

### Example

```bash
# Validate a map+fold program on GPU
$ remorac --target gpu-nvidia examples/reduce_iota.remora
45.0

# Compile and print PTX
$ remorac --target ptx examples/reduce_iota.remora
.version 7.8
.target sm_80
...
```

## Performance

### Benchmarking

```bash
# Run a single benchmark
uv run remora-bench examples/reduce_iota.remora

# Run the benchmark suite against baseline
uv run remora-bench --suite --baseline docs/BENCHMARK_BASELINES.json
```

### Threaded execution

```bash
# Use 4 OpenMP threads
uv run remorac --cpu-threads 4 program.remora

# Or set environment variable
REMORA_NUM_THREADS=4 uv run remorac program.remora
```

### Vectorized execution

```bash
uv run remorac --cpu-vectorize program.remora
```

Vectorization uses the affine-super-vectorize pass with virtual vector sizes of 128 for f32, 64 for i32.

## Limits (Dense Core)

| Feature | Status |
|---------|--------|
| Arrays: rank 0–10, static dimensions | Supported |
| Rank 11+ | Rejected at typecheck |
| Dynamic rank / dimensions | Deferred, produces diagnostic |
| Recursive functions | Deferred, produces diagnostic |
| Higher-order functions (compose, flip) | Deferred, produces diagnostic |
| Lambda closures | Deferred, produces diagnostic |
| Empty array literals | Deferred |

## Jupyter

```python
%load_ext remora.jupyter

%%remora --target cpu
map (* 2) [1, 2, 3]
```

## Examples

See `examples/` directory for runnable programs:
- `bool_logic.remora` — boolean operators
- `chained_maps.remora` — chained map pipeline
- `rank10_map.remora` — rank-10 map
- `reduce_iota.remora` — fold over iota (CPU + GPU)
- `three_dimensional_transform.remora` — 3D array transformation
- ...and 20 more

```bash
# Run any example
uv run remorac --target cpu examples/rank10_map.remora
```
