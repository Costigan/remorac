# Remora Dense Core — User Guide

## What is Remora?

Remora is an array programming language designed for high-performance numerical
computation on CPUs and GPUs. It belongs to the same family as APL, J, and
Futhark: programs operate on whole arrays at once rather than looping over
individual elements. Remora adds a static type system that catches shape errors
before execution and enables efficient compilation to parallel hardware.

## What is Dense Core?

Dense Core is the current implementation milestone — a statically-typed,
explicitly parallel subset focused on dense rectangular arrays. It compiles
through MLIR to CPU JIT code and NVIDIA PTX.

**Dense Core vs. full Remora.** The full Remora language (described in
Shivers, Slepak, and Manolios's tutorial draft in `docs/remora-reference/`)
uses Lisp-style s-expression syntax with implicit rank-polymorphic lifting:
you write `(+ xs ys)` and the language automatically maps the addition
across arrays of any rank. Dense Core uses a simpler ML-like surface syntax
where iteration is always explicit — you write `map (+) xs ys` to add two
arrays element-wise, and `fold (+) 0 xs` to reduce. This trade-off makes
compilation straightforward while preserving the data-parallel semantics.

| Full Remora | Dense Core equivalent |
|---|---|
| `(+ xs ys)` | `map (+) xs ys` |
| `(reduce + 0 xs)` | `fold (+) 0 xs` |
| `(λ ([x 0]) (* x 2))` | `\x -> x * 2` |
| `(frame [n] e1 ... en)` | `[e1, e2, ..., en]` |
| `(array [d1 ...] a1 ...)` | `[[a1, a2, ...], ...]` |
| `(define (f [x 0]) ...)` | `def f x = ...` |
| `(if cond then else)` | `if cond then then else` |
| `~(1 1) +` | (no equivalent; use cell maps) |

The full Remora language adds rank polymorphism, dependent types, boxes,
reranking, and higher-order combinators. Those features are deferred until
Dense Core is solid on both CPU and GPU.

## Installation

```bash
git clone https://github.com/your-org/remorac
cd remorac
uv sync
```

Verify your toolchain:

```bash
uv run python tools/validate_mlir_toolchain.py
```

This checks for `mlir-opt`, `iree-compile`, `ptxas`, and CUDA driver
availability. GPU features need `ptxas` and the CUDA driver; CPU-only
development works with just `mlir-opt`.

## Your first program

Create a file `hello.remora`:

```remora
1 + 2 * 3
```

Run it:

```bash
$ uv run remorac hello.remora
7
```

A `.remora` file contains a single **body expression**. The compiler evaluates
it and prints the result. Try more complex expressions directly in the REPL:

```bash
$ uv run remora
Remora REPL [target: cpu]
> iota 5
[0, 1, 2, 3, 4]
> map (* 2) (iota 5)
[0, 2, 4, 6, 8]
```

---

## Language overview

### Types

Dense Core has three scalar types and one aggregate type:

| Type    | MLIR equivalent | Examples            |
|---------|----------------|---------------------|
| `int`   | `i32`          | `42`, `-7`, `0`     |
| `float` | `f32`          | `3.14`, `-2.5`, `0.0` |
| `bool`  | `i1`           | `true`, `false`     |
| array   | `tensor<...>`  | `[1, 2, 3]`, `[[1, 2], [3, 4]]` |

Every array has a **rank** (number of dimensions) and a **shape** (size along
each dimension). A scalar is just an array of rank 0.

```
value            rank   shape
42               0      (none)
[1, 2, 3]        1      [3]
[[1, 2], [3, 4]] 2      [2, 2]
```

### Syntax

Remora uses standard arithmetic operator precedence (not APL-style right-to-left).

**Operator precedence, lowest to highest:**

| Precedence | Operators | Associativity |
|-----------|-----------|---------------|
| Boolean or | `\|\|` | left |
| Boolean and | `&&` | left |
| Comparison | `<`, `<=`, `==`, `!=` | left |
| Addition | `+`, `-` | left |
| Multiplication | `*`, `/` | left |
| Application | juxtaposition | left |
| Indexing | `[ ... ]` | postfix |

So `1 + 2 * 3` is `1 + (2 * 3)` — multiplication binds tighter than addition,
exactly like standard math. Comparisons chain naturally: `1 < x && x < 5`.

**What parentheses do:** Parentheses serve two purposes.

1. **Grouping** — override precedence:
   ```
   (1 + 2) * 3    -- 9, not 7
   ```

2. **Operator sections** — turn an infix operator into a callable for `map`,
   `fold`, or application:
   ```
   (* 2)         -- a function that multiplies by 2
   (+ 1)         -- a function that adds 1
   (< 5)         -- a function that tests if less than 5
   ```

   You can also fix the left operand instead of the right:
   ```
   (2 *)         -- a function that multiplies 2 by its argument
   ```

**Function application** is written with juxtaposition (just a space between
function and arguments):
```
f x             -- apply f to x
add 1 2         -- apply add to 1 and 2
map (* 2) xs    -- apply map to (* 2) and xs
```

Multi-argument functions just list arguments after the function name.

**Keywords** (`let`, `if`, `then`, `else`, `in`, `def`, `map`, `fold`, `iota`,
etc.) are reserved and cannot be used as variable names.

**Comments** start with `--` and run to the end of the line:
```remora
-- This is a comment
let x = 5  -- inline comment
```

### Values and literals

Scalar literals are written directly:

```remora
42          -- int
3.14        -- float
true        -- bool
false       -- bool
```

Array literals use square brackets, with commas between elements at the same
level:

```remora
[1, 2, 3]                              -- rank 1, shape [3]
[[1.0, 2.0], [3.0, 4.0]]               -- rank 2, shape [2, 2]
[[[1], [2]], [[3], [4]]]               -- rank 3, shape [2, 2, 1]
```

All siblings must have the same shape — ragged arrays are rejected.

### Arithmetic

Arithmetic operators follow standard precedence. Mixed `int`/`float` promotes
to `float`. Division always returns `float`.

```remora
1 + 2           -- 3      (int)
3.0 * 4.0       -- 12.0   (float)
5 / 2           -- 2.5    (float, even with int operands)
1 + 2.0         -- 3.0    (int promoted to float)
```

Comparisons return `bool`:

```remora
1 < 2           -- true
3 == 3          -- true
4 != 5          -- true
```

Boolean logic:

```remora
true && false   -- false
true || false   -- true
```

### Variables with `let`

`let` binds a name to a value for use in an expression:

```remora
let x = 5 in x + 1             -- 6
let a = [1, 2, 3] in a[0]      -- 1
```

### Conditionals with `if`

`if` chooses between two values based on a boolean condition:

```remora
if true then 10 else 20         -- 10 (scalar condition)
```

With tensor conditions, `if` selects element-wise:

```remora
if [true, false, true] then [1, 2, 3] else [10, 20, 30]
-- [1, 20, 3]
```

### Iota — generating integer ranges

`iota n` produces the array `[0, 1, 2, ..., n-1]` of type `int[n]`:

```remora
iota 4          -- [0, 1, 2, 3]
```

`iota` also supports multi-dimensional shapes. Write dimensions separated by
spaces:

```remora
iota 2 3        -- [[0, 1, 2], [3, 4, 5]]      (rank 2, shape [2, 3])
```

### Map — applying a function to every element

`map` is the primary parallel operation. It applies a function to each element
of an array.

```remora
map (* 2) [1, 2, 3]              -- unary map: [2, 4, 6]
map (+) [1, 2] [10, 20]           -- binary map: [11, 22]
map (\x -> x + 1) [1, 2, 3]       -- lambda: [2, 3, 4]
```

The first argument to `map` is a **callable** — one of:
- An operator section: `(* 2)`, `(+ 1)`, `(/)`
- A variable name referring to a function
- A lambda: `\x -> x * 2`

### Fold — reducing an array to a single value

`fold` combines all elements of an array using an accumulator:

```remora
fold (+) 0 [1, 2, 3, 4]          -- 10    (0 + 1 + 2 + 3 + 4)
fold (*) 1 [1, 2, 3, 4]          -- 24    (1 * 1 * 2 * 3 * 4)
```

The first argument is the combining function, the second is the initial
accumulator value, and the third is the array to reduce.

**Array-cell fold** reduces over the leading dimension, keeping trailing
dimensions:

```remora
fold (+) [0, 0] [[1, 2], [3, 4]]
-- [4, 6]    ([0,0] + [1,2] + [3,4])
```

### Indexing

Access individual elements with square brackets:

```remora
let xs = [10, 20, 30] in xs[0]           -- 10   (full-rank on rank-1)
xs[1]                                       -- 20
```

For multi-dimensional arrays, provide one index per dimension:

```remora
let m = [[1, 2], [3, 4]] in m[0, 1]      -- 2
```

Dynamic indices (computed at runtime) are supported:

```remora
let xs = iota 10 in let idx = 7 in xs[idx]   -- 7
```

Partial indexing (fewer indices than dimensions) drops the indexed dimensions:

```remora
let m = [[1, 2], [3, 4]] in m[0]          -- [1, 2]
```

### Views — reshaping and rearranging

Views produce a new array that shares data with the original, just with
different shape or element order:

```remora
reverse [1, 2, 3]                   -- [3, 2, 1]
transpose [[1, 2], [3, 4]]          -- [[1, 3], [2, 4]]
reshape [1, 2, 3, 4] [2, 2]        -- [[1, 2], [3, 4]]
ravel [[1, 2], [3, 4]]              -- [1, 2, 3, 4]  (flatten)
take 2 [1, 2, 3, 4]                 -- [1, 2]
drop 2 [1, 2, 3, 4]                 -- [3, 4]
```

### Shape and rank

```remora
shape [1, 2, 3]        -- [3]       (the number of elements)
shape [[1, 2], [3, 4]] -- [2, 2]    (one size per dimension)
rank [1, 2, 3]         -- 1         (the number of dimensions)
rank [[1, 2], [3, 4]]  -- 2
```

### Cell maps

When you `map` over an array, the default is to process individual scalar
elements. Cell maps let you process sub-arrays ("cells") instead:

```remora
-- Sum each row (cell = rank-1 sub-array)
map (\row -> fold (+) 0 row) [[1, 2], [3, 4]]
-- [3, 7]

-- Access cell elements by index
map (\row -> row[0] + row[1]) [[1, 2], [3, 4], [5, 6]]
-- [3, 7, 11]
```

### Lambda functions

Lambdas are written with a backslash:

```remora
\param1 param2 -> body
```

Examples:

```remora
map (\x -> x * x + 1) [1, 2, 3]         -- [2, 5, 10]
fold (\acc x -> acc + x) 0 [1, 2, 3]    -- 6
```

---

## Program structure

A `.remora` file contains a sequence of definitions followed by a body
expression. The body expression is the last non-definition in the file.

### Top-level definitions

```remora
-- Define functions
def double x = x * 2
def add x y = x + y

-- Define values
def my_array = [1, 2, 3, 4, 5]

-- Body expression (uses the definitions above)
map double my_array
```

Running this file produces `[2, 4, 6, 8, 10]`.

### The prelude

Remora ships with built-in functions available in every program. You don't
need to import or define them:

| Category    | Functions |
|-------------|-----------|
| Arithmetic  | `add`, `sub`, `mul`, `div`, `neg`, `id`, `const` |
| Reductions  | `sum`, `product` |
| Vector ops  | `scale`, `dot` |
| Aggregate   | `max`, `min` |
| Boolean     | `any`, `all` |
| Absolute    | `abs` |

Prelude functions work like any other Remora function:

```remora
sum (iota 10)                   -- 45
dot [1.0, 2.0] [3.0, 4.0]      -- 11.0
abs (-5)                        -- 5
```

---

## CLI reference

```
uv run remorac [options] file.remora
```

### Execution targets

| Flag | What it does |
|------|-------------|
| `--target cpu` | Compile to native code via MLIR→LLVM, JIT-execute (default) |
| `--target interp` | Run through the typed-AST interpreter (correctness oracle) |
| `--target gpu-nvidia` | Compile to PTX and validate (requires `ptxas`) |
| `--target mlir` | Print lowered MLIR and exit |
| `--target ptx` | Print generated PTX and exit |

### Inspecting compilation

```bash
$ remorac --emit-mlir examples/reduce_iota.remora
#map = affine_map<(d0) -> (d0)>
module {
  func.func @main() -> f32 { ... }
}
```

### Performance flags

```bash
# Multi-threaded CPU execution
remorac --cpu-threads 4 program.remora
REMORA_NUM_THREADS=4 remorac program.remora    # or via env

# SIMD vectorization
remorac --cpu-vectorize program.remora
```

### Calling named functions on GPU

```bash
# Call `scale` with a numpy array as input
remorac --target gpu-nvidia --call scale --input xs.npy program.remora
```

---

## REPL reference

```
uv run remora [--target cpu|interp|gpu-nvidia]
```

### Commands

| Command | Action |
|---------|--------|
| `:target [cpu\|interp\|gpu-nvidia]` | Show or change execution target |
| `:type <expr>` | Show the inferred type of an expression |
| `:mlir <expr>` | Print the lowered MLIR |
| `:load <file.remora>` | Load definitions from a file |
| `:defs` | List user-defined functions |
| `:prelude` | List built-in prelude functions |
| `:reset` | Clear all user definitions |
| `:help` | Show command summary |
| `:quit` | Exit |

Example REPL session:

```
Remora REPL [target: cpu]
> :type map (* 2) (iota 5)
int[5]
> :target interp
> fold (+) 0 (iota 10)
45
> :target gpu-nvidia
> fold (+) 0.0 (map (* 2.0) (iota 1000))
2000.0
```

---

## GPU (NVIDIA)

Installation requirements:
- `ptxas` from CUDA toolkit (usually at `/usr/local/cuda/bin/`)
- `iree-compile` from IREE
- CUDA driver and `cuda-python` package for execution

### What runs on GPU

| Operation | Types | Ranks |
|-----------|-------|-------|
| Unary maps | `f32`, `i32`, `bool` | 1–10 |
| Binary maps | `f32`, `i32`, `bool` | 1–10 |
| Scalar reductions | `f32`, `i32` | 1–10 |

### What doesn't run on GPU (yet)

Views, indexing, dynamic indices, cell maps, and lambdas with closures are not
supported on GPU. These programs emit a clear diagnostic with the reason,
rather than silently falling back to CPU.

---

## Benchmarks

```bash
# Run a single benchmark
uv run remora-bench examples/reduce_iota.remora

# Run the full baseline suite
uv run remora-bench --suite --baseline docs/BENCHMARK_BASELINES.json
```

---

## Limits (Dense Core)

| Feature | Status |
|---------|--------|
| Arrays rank 0–10, static dimensions | Supported |
| Rank 11+ | Rejected at typecheck |
| Dynamic dimensions | Deferred (produces diagnostic) |
| Recursive functions | Deferred (produces diagnostic) |
| Higher-order functions (`compose`, `flip`) | Deferred (produces diagnostic) |
| Lambda closures | Deferred (produces diagnostic) |
| Empty array literals | Deferred (produces diagnostic) |

---

## Examples

The `examples/` directory contains 25 runnable programs demonstrating the full
Dense Core surface:

| File | What it demonstrates |
|------|---------------------|
| `bool_logic.remora` | Boolean operators |
| `chained_maps.remora` | Composing multiple maps |
| `conditional.remora` | Tensor `if` |
| `dot_product.remora` | `dot` prelude function |
| `indexing.remora` | Full-rank indexing on rank-2 |
| `lift_map.remora` | Map over a view |
| `nested_let.remora` | Multiple let bindings |
| `prelude_scale.remora` | `scale` and `sum` prelude |
| `rank10_map.remora` | Map on rank-10 array |
| `reduce_iota.remora` | Fold+map pipeline (CPU + GPU) |
| `scalar_arithmetic.remora` | Mixed int/float arithmetic |
| `three_dimensional_transform.remora` | 3D array transform |

Run any example:

```bash
uv run remorac --target cpu examples/rank10_map.remora
```

For GPU examples:

```bash
uv run remorac --target gpu-nvidia examples/reduce_iota.remora
```
