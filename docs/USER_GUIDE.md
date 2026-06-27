# Remora — User Guide

## What is Remora?

Remora is an array programming language for high-performance numerical computation
on CPUs and GPUs. It belongs to the same family as APL, J, and Futhark: programs
operate on whole arrays at once. Remora adds a static type system that catches shape
errors before execution and enables efficient compilation to parallel hardware via
MLIR.

Remora was designed by Justin Slepak as part of his dissertation at Northeastern
University. The original Racket/OCaml implementation is at github.com/jrslepak/Remora.

Here are key papers (local copies in `docs/remora-reference/`):

- *Remora Tutorial Draft* — best first read; covers the
  rank-polymorphic programming model, arrays, frames, cells, lifting,
  reranking, and static types
  (www.ccs.neu.edu/home/shivers/papers/remora-tutorial-draft.pdf)
- *An Introduction to Rank-polymorphic Programming in Remora* —
  published arXiv entry for the tutorial
  (arxiv.org/abs/1912.13451)
- *The Semantics of Rank Polymorphism* — formal dynamic/static
  semantics, frame/cell typing rules, shape soundness
  (arxiv.org/abs/1907.00509)
- *Slepak Dissertation* — deep reference for Remora's type system,
  bidirectional typing, shape inference, constraint solving, and type erasure
  (ccs.neu.edu/~jrslepak/Dissertation.pdf)

## What is remorac?

This implementation, remorac for Remora Compiler, implements the
dense, statically shaped core of Remora, plus several extensions.  It
does not yet implement the full language from the papers.

What is currently implemented:

- Dense rectangular arrays only, ranks 0-10.
- Scalar types: Int, Float, Float64, Bool.
- Static shapes at lowering time: compiled CPU/GPU artifacts specialize all dimensions.
- Two syntaxes: ML-like (.remora files) and Lisp s-expression (.lisp)
- Rank-polymorphic behavior, especially through Lisp syntax auto-lifting and explicit define/pi / define/forall.
- Core array operators: map, fold, reduce, fold-right, scan, iscan,
  escan, right/left variants, trace, trace-right, rerank
- Views and shape operations: indexing, slicing, transpose, reshape,
  ravel, reverse, take, drop, rotate, subarray, shape, rank, length
- Higher-order functions on CPU/interpreter: lambdas, functions as
  arguments, closure capture, define/forall, monomorphization of
  higher-order calls
- Recursion on CPU/interpreter: self recursion, mutual recursion, tail
  and non-tail recursion
- Special operations: sort, grade, matmul, im2col, col2im, pair,
  first, second, box, unbox as mostly type-erasure/static-shape
  machinery, reverse-mode grad for scalar-cost functions

Backend coverage differs:

- Interpreter: broadest semantic coverage; used as oracle.
- CPU compiled backend: essentially complete for the current dense 
  static core accepted by the typechecker.
- GPU backend: a subset. Strong for maps, reductions, scans, views,
  sort/grade, matmul, some AD/state-fold loops, and descriptor-based
  kernels; limited for closures, full higher-order behavior, boxes,
  general recursion, irregular/dynamic data, and some dtype/op
  combinations.

What is not implemented from full Remora:

- True dynamic shapes in compiled code.
- Runtime ragged/irregular arrays via real existential boxes.
- MIMD arrays of functions / function-valued arrays.
- Full dynamic higher-order function dispatch, especially call-through-variable in all contexts.
- Full GPU support for all CPU/interpreter features.
- Segmented reductions from the papers.
- Some paper surface forms, such as explicit frame, explicit array, and all parameter syntax.
- Full dynamic shape polymorphism without per-shape specialization.

## Installation

```bash
git clone <url>
cd remorac
uv sync
```

Verify:

```bash
uv run remorac examples/scalar_arithmetic.remora
```

## Quick start

### ML syntax (`.remora` files)

```remora
def double x = x * 2
def sum_all xs = fold (+) 0 xs
fold (+) 0 (iota 10)
```

### Lisp syntax (`.lisp` files)

```lisp
(+ [1 2 3] [4 5 6])
(define (sum-vec [v]) (fold + 0 v))
(sum-vec [1 2 3 4 5])
```

### CLI

`remorac` is a compiler-and-runner: it compiles Remora source to a native
executable and runs it in one step. By default it produces `a.out` (an
ELF executable you can `./a.out`).

```bash
# ML syntax (default — inferred from .remora extension)
remorac file.remora                    # compile to a.out, run it
remorac --emit-mlir file.remora        # Show generated MLIR
remorac --emit-ast file.remora         # Show parsed AST
remorac --emit-hir file.remora         # Show HIR before lowering
remorac --emit-ptx file.remora         # Show generated PTX

# Lisp syntax (inferred from .lisp extension)
remorac file.lisp
remorac --syntax lisp file.txt         # Explicit syntax override

# Target selection
remorac --target cpu file.remora       # Compile and run on CPU (default)
remorac --target interp file.remora    # Reference interpreter (no compilation)
remorac --target cuda file.remora      # GPU execution (CUDA)

# Output control
remorac -o myprog file.remora          # Name the output binary
remorac --shared file.remora           # Produce a.so instead of a.out
remorac --shared -o libfoo.so file.remora  # Named shared library
remorac --compile-only file.remora     # Compile to a.out/a.json, don't run
remorac --cleanup file.remora          # Delete a.out + metadata after running

# Multiple source files (concatenated; later defs shadow earlier ones)
remorac lib.remora main.remora

# Load files and start REPL
remorac --repl                         # REPL with just the prelude
remorac --repl lib.remora              # REPL with loaded definitions
```

### REPL

```bash
uv run remorac --repl
remora> :syntax lisp
remora> (+ [1 2 3] [4 5 6])
[5 7 9]
remora> :target interp
remora> (iscan + 0 [2 10 5])
[2 12 17]
```

## Syntax reference

### Literals

| Lisp        | ML               | Type / Description     |
| ----------- | ---------------- | ---------------------- |
| `42`        | `42`             | Integer                |
| `3.14`      | `3.14`           | Float (32-bit)         |
| `3.14d`     | `3.14d`          | Float64 (64-bit)       |
| `-7`        | `-7`             | Negative integer       |
| `#t` / `#f` | `true` / `false` | Boolean                |
| `[1 2 3]`   | `[1, 2, 3]`      | Array (ML uses commas) |

### let

`let` / `let*` (Lisp) and `let ... in` (ML) create local bindings.

| Lisp                           | ML                   | Notes                            |
| ------------------------------ | -------------------- | -------------------------------- |
| `(let ((x 5)) (+ x 1))`        | `let x = 5 in x + 1` | Parallel bindings (simultaneous) |
| `(let* ((x 5) (y (+ x 1))) y)` | —                    | Sequential bindings (sequential) |

### if / select

Conditional expressions. `if` takes a bool and two branches; `select` is an
N-ary variant that checks each condition in order.

| Lisp                               | ML                         | Notes                    |
| ---------------------------------- | -------------------------- | ------------------------ |
| `(if (< 1 2) 10 20)`               | `if 1 < 2 then 10 else 20` |                          |
| `(select #t 10 20)`                | `select true 10 20`        |                          |
| `(select #f 10 (select #t 20 30))` | —                          | Multi-condition dispatch |

### Arithmetic — Prefix (Lisp)

| Lisp       | ML equivalent | Description           |
| ---------- | ------------- | --------------------- |
| `(+ a b)`  | `a + b`       | Addition              |
| `(- a b)`  | `a - b`       | Subtraction           |
| `(* a b)`  | `a * b`       | Multiplication        |
| `(/ a b)`  | `a / b`       | Division              |
| `(exp x)`  | —             | Exponential           |
| `(log x)`  | —             | Natural logarithm     |
| `(sqrt x)` | —             | Square root           |
| `(* -1 x)` | —             | Unary negation (Lisp) |

### Arithmetic — Infix (ML)

| ML      | Operator | Type         |
| ------- | -------- | ------------ |
| `a + b` | `+`      | Float or Int |
| `a - b` | `-`      | Float or Int |
| `a * b` | `*`      | Float or Int |
| `a / b` | `/`      | Float or Int |

### Boolean operators

| Lisp       | ML       | Notes              |
| ---------- | -------- | ------------------ |
| `(&& a b)` | `a && b` | Short-circuit `&&` |
| \`(        |          | a b)\`             |
| `(not x)`  | —        | Negation (Lisp)    |

### Comparison

| Lisp       | ML       | Notes                 |
| ---------- | -------- | --------------------- |
| `(< a b)`  | `a < b`  | Less than             |
| `(<= a b)` | `a <= b` | Less than or equal    |
| `(> a b)`  | `a > b`  | Greater than          |
| `(>= a b)` | `a >= b` | Greater than or equal |
| `(== a b)` | `a == b` | Equality              |
| `(!= a b)` | `a != b` | Inequality (Lisp)     |

### Definitions

| Lisp                                                       | ML               | Notes                            |
| ---------------------------------------------------------- | ---------------- | -------------------------------- |
| `(define (f [x]) body)`                                    | `def f x = body` | Named function                   |
| `(define xs body)`                                         | `def xs = body`  | Value definition                 |
| `(define/pi () (f [x Float] Float) body)`                  | —                | Explicit types for scalar params |
| `(define/pi ([n Dim]) (f [x (Array Float n)] Float) body)` | —                | Parametric (rank-polymorphic)    |
| `(define/forall (t) (f [x (Array t 3)] (Array t 3)) body)` | —                | Type-variable (HOF)              |

### Lambda

Anonymous functions.

| Lisp                  | ML                               |
| --------------------- | -------------------------------- |
| `(lambda (x) body)`   | `\x -> body`                     |
| `(lambda (x y) body)` | `\x y -> body`                   |
| `(\ (x) body)`        | `\x -> body` (`λ` Unicode alias) |

Lambda parameters are **not** automatically rank-polymorphic in ML syntax. They are
plain (scalar) unless wrapped in a top-level `define`/`define/pi` with explicit
annotations. Rank-polymorphic lambdas work via operator sections and `define/forall`.

### Built-in Functions (Prelude)

The Remora prelude (`stdlib/prelude.rem`) auto-prepends every program with these
functions. They are available in both syntaxes without explicit import.

| Function  | Description                 | Signature                                 |
| --------- | --------------------------- | ----------------------------------------- |
| `add`     | Element-wise addition       | `(Float, Float) -> Float`                 |
| `sub`     | Element-wise subtraction    | `(Float, Float) -> Float`                 |
| `mul`     | Element-wise multiplication | `(Float, Float) -> Float`                 |
| `div`     | Element-wise division       | `(Float, Float) -> Float`                 |
| `neg`     | Unary negation              | `Float -> Float`                          |
| `id`      | Identity                    | `Float -> Float`                          |
| `const`   | Const function (curried)    | `Float -> (Float -> Float)`               |
| `sum`     | Sum reduction               | `(Array Float) -> Float`                  |
| `product` | Product reduction           | `(Array Float) -> Float`                  |
| `scale`   | Scalar multiplication       | `(Float, Array Float) -> Array Float`     |
| `dot`     | Dot product                 | `(Array Float 1, Array Float 1) -> Float` |
| `max`     | Reduction to max            | `(Array Float) -> Float`                  |
| `min`     | Reduction to min            | `(Array Float) -> Float`                  |
| `abs`     | Absolute value              | `Float -> Float`                          |
| `any`     | Any-true                    | `(Array Bool) -> Bool`                    |
| `all`     | All-true                    | `(Array Bool) -> Bool`                    |

All of these can be passed as values to higher-order functions and used inside
`map`, `fold`, `scan`, and other operators.

## Map and Fold

### Map

`map` applies a function element-wise over one or more arrays.

| Lisp             | ML              | Description                 |
| ---------------- | --------------- | --------------------------- |
| `(map f xs)`     | `map f xs`      | Unary map                   |
| `(map f xs ys)`  | `map f xs ys`   | Binary map                  |
| `(map (+ 1) xs)` | `map (+ 1) xs`  | With operator section       |
| `map (+) xs ys`  | `map (+) xs ys` | Binary operator as function |

**Note:** In the Lisp DSL, `map` with two arguments calls a user-defined function `map`.
The Lisp syntax form is `(map f xs)` (unary), and `(map (+ 1) xs)` passes an
operator section as the function. Binary maps use `define`-level auto-lifting for
cross-product and principal-frame semantics.

In the ML syntax, `map f xs ys` applies the binary function `f` element-wise across
both arrays.

### Fold / Reduce

Aggregate arrays to a scalar. `fold` consumes the whole array;
`fold-right` processes from the right; `reduce` is alias for `fold`.

| Lisp                   | ML                     | Type                                |
| ---------------------- | ---------------------- | ----------------------------------- |
| `(fold + 0 xs)`        | `fold (+) 0 xs`        | Element-wise reduce                 |
| `(fold-right + 0 xs)`  | `fold-right (+) 0 xs`  | Right fold                          |
| `(reduce + 0 xs)`      | `reduce (+) 0 xs`      | Alias for fold                      |
| `(reduce/zero + 0 xs)` | `reduce/zero (+) 0 xs` | Always uses the neutral element `0` |
| `(reduce/1 + 0 xs)`    | `reduce/1 (+) 0 xs`    | First element as base               |

### Scan

Scan produces a running-reduction array (output same shape as input).

| Lisp                  | ML                    | Description     |
| --------------------- | --------------------- | --------------- |
| `(iscan + 0 xs)`      | `iscan (+) 0 xs`      | Inclusive scan  |
| `(escan + 0 xs)`      | `escan (+) 0 xs`      | Exclusive scan  |
| `(scan + 0 xs)`       | `scan (+) 0 xs`       | Alias for iscan |
| `(scan-left + 0 xs)`  | `scan-left (+) 0 xs`  | Alias for iscan |
| `(scan-right + 0 xs)` | `scan-right (+) 0 xs` | Right scan      |

With `scan-right-with`:

| Lisp                       | ML                         | Description                            |
| -------------------------- | -------------------------- | -------------------------------------- |
| `(scan-right-with + 0 xs)` | `scan-right-with (+) 0 xs` | Right scan with explicit initial value |

Scan also supports `/zero` and `/1` variants (same semantics as reduce) and
the `trace` family (see below).

### Trace

Trace is like scan but uses a *seed* function to compute the initial carry.
`trace` walks left-to-right; `trace-right` walks right-to-left.

| Lisp                   | ML                     | Description          |
| ---------------------- | ---------------------- | -------------------- |
| `(trace + 0 xs)`       | `trace (+) 0 xs`       | Left scan with seed  |
| `(trace-right + 0 xs)` | `trace-right (+) 0 xs` | Right scan with seed |

### compose

`compose` (ML) and `(compose f g)` (Lisp) create a new function that applies `g`
then `f`.

| Lisp            | ML            |
| --------------- | ------------- |
| `(compose f g)` | `compose g f` |

Note the argument order: ML's `compose g f` means "first `g`, then `f`" (standard
right-to-left composition). In ML syntax: `compose f . g`.

## Views, Primitives, and Box Operations

### Iota

Generate integer or boolean arrays.

| Lisp                   | ML                       | Description                               |
| ---------------------- | ------------------------ | ----------------------------------------- |
| `(iota n)`             | `iota n`                 | Int array `[0, 1, ..., n-1]`              |
| `(iota-n n)`           | `iota-n n`               | Float64 array of shape `n`                |
| `(iota1 n)`            | `iota1 n`                | Boxed dynamic iota: `(Σ (len) [int len])` |
| `(iota bool n)` — (ML) | `(iota-bool n)` — (Lisp) | Bool array                                |

### Primitive Queries

Extract shape metadata from arrays.

| Lisp          | ML          | Description               |
| ------------- | ----------- | ------------------------- |
| `(shape xs)`  | `shape xs`  | Shape vector (Int array)  |
| `(rank xs)`   | `rank xs`   | Number of dimensions      |
| `(length xs)` | `length xs` | Size of leading dimension |

### Indexing

Access elements by position. ML syntax supports index suffixes on identifiers: `xs {0} {1}`
for multi-dimensional indexing, or `xs {3}` for leading-dim index.

| Lisp                       | ML                           | Description                           |
| -------------------------- | ---------------------------- | ------------------------------------- |
| `(index-item xs 0)`        | `index-item xs 0` or `xs{0}` | Leading-dim pick                      |
| `(index xs 0 1)`           | `index xs 0 1` or `xs{0}{1}` | Multi-dim lookup                      |
| `(subarray m [1 0] [2 2])` | —                            | Extract sub-region at start with size |

Slices are specified in ML syntax as index suffixes:

| Syntax    | Description                                     |
| --------- | ----------------------------------------------- |
| `xs{1:3}` | Slice from index 1 (exclusive) to 3 (inclusive) |
| `xs{1:}`  | Slice from index 1 to end                       |
| `xs{:3}`  | Slice from start to index 3 (exclusive)         |
| `xs{::2}` | Every second element (stride 2)                 |

### View Operations

| Lisp                    | ML                    | Description             |
| ----------------------- | --------------------- | ----------------------- |
| `(reverse xs)`          | `reverse xs`          | Reverse leading dim     |
| `(transpose m)`         | `transpose m`         | Transpose matrix        |
| `(reshape xs [2 2])`    | `reshape [2, 2] xs`   | Reshape to new shape    |
| `(ravel m)`             | `ravel m`             | Flatten to 1D           |
| `(take n xs)`           | `take n xs`           | First `n` elements      |
| `(drop n xs)`           | `drop n xs`           | Drop first `n` elements |
| `(rotate xs n)`         | `rotate xs n`         | Circular shift          |
| `(rerank ~[0 0] xs ys)` | `rerank ~[0 0] xs ys` | Align ranks             |

### Reranking

The `~` (tilde) operator lifts scalar-cell functions to operate on array cells.

```lisp
(~(0 0) +)        ; desugars to (lambda ([x0 0] [x1 0]) (+ x0 x1))
(map (~(0 0) +) [1 2 3] [4 5 6])  ; element-wise addition of two arrays
```

Reranking works in both syntaxes. The ML syntax uses `rerank`:

```remora
rerank ~(0 0) (+) xs ys
```

### Pair / first / second

| Lisp         | ML         | Description            |
| ------------ | ---------- | ---------------------- |
| `(pair a b)` | `pair a b` | Construct a pair       |
| `(first p)`  | `first p`  | Extract first element  |
| `(second p)` | `second p` | Extract second element |

### Boxes (Phase 6)

Type-erasure boxing/unboxing. Boxes allow runtime-sized data. They support `iota1`,
`filter`, and `replicate`.

| Lisp                     | ML         | Description                              |
| ------------------------ | ---------- | ---------------------------------------- |
| `(box x)`                | `box x`    | Wrap in a box                            |
| `(unbox b (len v) body)` | —          | Unbox, binding shape-length and contents |
| `(boxes xs)`             | `boxes xs` | Map box over array                       |
| `(boxes-add b1 b2)`      | —          | Scatter-add boxes                        |

Box operations in ML syntax:

```remora
def b = box [1 2 3]
```

### Sort / Grade

| Lisp           | ML                                                                 | Description                | Types      |
| -------------- | ------------------------------------------------------------------ | -------------------------- | ---------- |
| `(sort < xs)`  | `sort xs` (using default `<`) or `sort (lambda (a b) (<= a b)) xs` | Sort ascending             | Float, Int |
| `(grade < xs)` | `grade xs`                                                         | Return permutation indices | Float, Int |

Sort uses the `<` operator as the default comparison. For non-default comparisons,
use a lambda:

| Lisp                               | ML                         |
| ---------------------------------- | -------------------------- |
| `(sort (lambda (a b) (> a b)) xs)` | `sort (\ a b -> a > b) xs` |

## Higher-Order Functions

Functions can be passed as arguments to other functions. Supports:

- Plain `define` (type inferred)
- `define/forall` with explicit `Func` types
- Closure capture (lambdas referencing outer variables)

```lisp
;; Plain define — type inferred
(define (apply_twice [f x]) (f (f x)))
(apply_twice (lambda (x) (+ x 1)) 5)  ;; → 7

;; Closure capture
(let ((z 3)) (apply_twice (lambda (x) (+ x z)) 5))  ;; → 11

;; define/forall with explicit Func type
(define/forall (t) (apply_twice [f (Func (t) t) x t] t) (f (f x)))
(define (square [x]) (* x x))
(apply_twice square 5)  ;; → 25
```

## Rank Polymorphism (Lisp syntax)

Functions expecting scalars auto-lift when applied to arrays:

```lisp
(+ [1 2 3] [4 5 6])          ;; → [5 7 9] (implicit map)
(* 2 [1 2 3])                ;; → [2 4 6] (scalar broadcasting)
(+ [10 20] [[1 2] [3 4]])    ;; → [[11 12] [23 24]] (principal frame)
```

Vector-cell functions auto-lift to matrix inputs:

```lisp
(define (sum-vec [v 1]) (fold + 0 v))
(sum-vec [[1 2 3] [4 5 6]])  ;; → [6 15]
```

## Recursive Functions

Self-recursion and mutual recursion work on CPU and in the interpreter. Scalar
tail-recursive function groups lower to stack-safe CPU loops, including mutual
tail recursion. Non-tail-recursive scalar functions still compile to CPU through
ordinary native calls, and recursive functions may take array parameters or
return arrays.

### Self-recursion (tail)

```lisp
(define (sum_to [n acc])
  (if (== n 0) acc (sum_to (- n 1) (+ acc n))))
(sum_to 10000 0)  ;; → 50005000
```

### Self-recursion (non-tail)

```lisp
(define (fib [n])
  (if (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))
(fib 10)  ;; → 55
```

### Mutual recursion

```lisp
(define (is_even [n])
  (if (== n 0) true (is_odd (- n 1))))
(define (is_odd [n])
  (if (== n 0) false (is_even (- n 1))))
(is_even 4)  ;; → true
```

### Recursive define/pi with array parameters

```lisp
(define/pi ()
  (sum_with_base [a (Array Float 4) n Float] Float)
  (if (== n 0.0)
      (fold + 0.0 a)
      (+ n (sum_with_base a (- n 1.0)))))
(sum_with_base [1.0 2.0 3.0 4.0] 4.0)  ;; → 20.0
```

### GPU recursion

GPU supports a narrower subset: scalar tail-recursive helper functions used inside a
`map`. Self-recursive helpers and mutually recursive helper groups lower to a
per-thread state machine when every recursive call in the group is in tail
position.

```lisp
(define/pi ()
  (sum_to [n Float acc Float] Float)
  (if (== n 0.0) acc (sum_to (- n 1.0) (+ acc n))))

(define/pi ()
  (f [xs (Array Float 4)] (Array Float 4))
  (map (lambda (x) (sum_to x 0.0)) xs))
```

This works for `Float`, `Int`, and `Bool` helpers. Tail-recursive helpers (self
or mutual) are now also threaded through higher-order step functions:

- `scan` step functions on the single-block `f32` path, and
- serial rank-1 `f32` `fold`/`reduce`/`fold-right` step functions (an inline
  `lambda` taking an accumulator and an element, with a literal `f32`
  initializer over a direct rank-1 `f32` array parameter).

Non-tail recursion, array-parameter recursive helpers, and array-returning
recursive helpers are still rejected on GPU. The compound `fold` path is
deliberately conservative: it runs serially on a single thread and is limited to
rank-1 `f32` inputs and scalar `f32` results; other shapes/dtypes fall back to
the existing reduction/scan paths or are rejected loudly.

## Automatic Differentiation

Remora supports reverse-mode automatic differentiation via the `grad` operator.
`grad f` returns the gradient function of `f` with respect to its Float parameters.

### Basic usage

```lisp
;; Scalar function
(define/pi () (sq [x Float] Float) (* x x))
((grad sq) 3.0)                        ;; → 6.0

;; Vector loss
(define/pi ([n Dim])
  (sq-loss [x (Array Float n)] Float)
  (fold + 0.0 (* x x)))
((grad (iapp sq-loss 5)) [1 2 3 4 5])  ;; → [2 4 6 8 10]

;; Binary function — returns pair in interpreter, single gradient via compiled CPU
(define/pi ([n Dim])
  (dot-loss [x (Array Float n) w (Array Float n)] Float)
  (fold + 0.0 (* x w)))
((grad (iapp dot-loss 4)) [1 2 3 4] [5 6 7 8])
;; → ([5 6 7 8], [1 2 3 4])

;; Conditional
(define/pi () (relu [x Float] Float) (if (> x 0.0) x 0.0))
((grad relu) 3.0)  ;; → 1.0
((grad relu) -3.0) ;; → 0.0
```

### Supported differentiable operations

| Operation              | VJP                                   | Notes                      |
| ---------------------- | ------------------------------------- | -------------------------- |
| `+ - * /`              | Standard                              | Scalar and elementwise     |
| `fold + 0.0`           | Broadcast adjoint                     | Sum reduction              |
| `reshape`, `ravel`     | Reshape cotangent                     | Shape-preserving           |
| `transpose`, `reverse` | Inverse view on cotangent             | Elementwise                |
| `take`, `drop`         | Zero-pad cotangent                    | Leading-dimension only     |
| `append`               | Split cotangent via `take`/`drop`     | Rank-N (axis 0)            |
| `subarray`             | Scatter cotangent via zero-pad        | Rank-1 only                |
| `rotate`               | Counter-rotate cotangent              | Elementwise                |
| `index`                | Scatter via `scatter-add`             | Compile-time-known indices |
| `if` / `select`        | Route cotangent through active branch | Both branches traced       |

### Compilation options

```python
from remora.compiler import compile_gradient_function_source

# Compile a single gradient function (CPU)
cpu = compile_gradient_function_source(source, "loss", (param_type,))

# Compile for GPU (elementwise and select gradients)
gpu = compile_gradient_function_source_to_supported_gpu_artifacts(
    source, "loss", (param_type,))

# Compile per-input gradients for multi-parameter functions
from remora.compiler import compile_gradient_functions_source
grads = compile_gradient_functions_source(source, "dot-loss", (tx, tw))
```

### Limitations

- `grad f` requires `f` to return a scalar `Float`
- All differentiated parameters must be `Float` or `Array Float`
- GPU kernels support elementwise and select operations only (structured views run on CPU)
- Multi-parameter `grad` returns a pair in the interpreter but a single gradient
  via compiled CPU; use `compile_gradient_functions_source` for both
- Inline lambdas inside `map` that capture array variables may resolve to scalar-cell
  instead of vector-cell lifting. Use a named helper with explicit parameter types to
  force vector-cell behavior

## Feature Status by Backend

### CPU Backend

| Feature                                                              | Status              |
| -------------------------------------------------------------------- | ------------------- |
| Scalar arithmetic (+, -, \*, /)                                      | Full support        |
| Arrays (rank 1–10, Int, Float, Float64, Bool)                        | Full support        |
| `let` bindings (scalar and array)                                    | Full support        |
| `if` / `select` conditionals                                         | Full support        |
| `map` (unary, binary, operator sections)                             | Full support        |
| `fold` / `reduce` / `fold-right`                                     | Full support        |
| `scan` (inclusive, exclusive, left, right)                           | Full support        |
| `trace` / `trace-right`                                              | Full support        |
| `lambda` expressions                                                 | Full support        |
| `define` (plain, `define/pi`, `define/forall`)                       | Full support        |
| Function values as arguments                                         | Full support        |
| Closure capture                                                      | Full support        |
| Operator sections (`(* 2)`, `(+ 1)`)                                 | Full support        |
| Recursion (self, mutual, tail, non-tail)                             | Full support        |
| `rerank`                                                             | Full support        |
| Views (index, slice, transpose, reshape, ravel, reverse, take, drop) | Full support        |
| `iota` / `iota-n` / `iota1`                                          | Full support        |
| `sort` / `grade`                                                     | Full (f32/i32)      |
| `matmul`                                                             | Full                |
| `im2col` / `col2im`                                                  | Full                |
| `pair` / `first` / `second`                                          | Full                |
| `box` / `unbox`                                                      | Full (type erasure) |
| `grad` (automatic differentiation)                                   | Full                |
| `compose`                                                            | Full                |

### GPU Backend

| Feature                                                                     | Status                        |
| --------------------------------------------------------------------------- | ----------------------------- |
| Element-wise maps (Float, Float64, Int, Bool)                               | Full support                  |
| Compound map bodies (arithmetic, comparisons, lets, conditionals, indexing) | Full support                  |
| Closures and first-class functions                                          | Limited                       |
| `map` with scalar cells                                                     | Full support                  |
| `fold` / `reduce` / `scan`                                                  | Full support                  |
| `trace` / `trace-right`                                                     | Full support                  |
| Rotate, take/drop, reverse, subarray, indices-of                            | Full (descriptor kernels)     |
| Transpose, reshape, ravel, append, with-shape                               | Full (descriptor reinterpret) |
| Type-aware i32 arithmetic and comparisons                                   | Supported                     |
| `sort` / `grade`                                                            | Float32 only                  |
| `matmul`                                                                    | Float32 only                  |
| `im2col` / `col2im`                                                         | Limited                       |
| `pair` / `first` / `second`                                                 | Not supported                 |
| `box` / `unbox`                                                             | Not supported                 |
| `grad`                                                                      | Elementwise/select only       |
| Recursion                                                                   | Tail-helper subset only       |

### Interpreter

| Feature                       | Status                  |
| ----------------------------- | ----------------------- |
| All dense-core constructs     | Full support            |
| Rank-polymorphic auto-lifting | Supported (Lisp syntax) |
| Deep non-tail recursion       | Supported (trampolined) |
| Dynamic shapes                | Supported               |
| Boxes                         | Full runtime support    |

## Feature Status by Phase

| Phase | Feature                                                           | Status                                                 |
| ----- | ----------------------------------------------------------------- | ------------------------------------------------------ |
| 1     | Lisp syntax + ML syntax                                           | Full                                                   |
| 2     | Rank polymorphism                                                 | Scalar/vector auto-lift, broadcasting                  |
| 3     | Reduce/scan/fold/trace                                            | Full (7 operators with all variants)                   |
| 4     | Primitives (iota/shape/rank/length)                               | Full                                                   |
| 4     | Views (reverse/transpose/reshape/ravel/take/drop/rotate/subarray) | Full                                                   |
| 4     | pair/first/second, boxes, sort/grade                              | Full                                                   |
| 4     | im2col/col2im, matmul, append, indices-of, with-shape             | Full                                                   |
| 5     | Reranking                                                         | Full                                                   |
| 6     | Boxes                                                             | Type erasure (no runtime effect)                       |
| AD    | Automatic differentiation                                         | Reverse-mode via tape + source gen                     |
| 7     | Recursive functions                                               | Full on CPU/interp; tail-helper on GPU                 |
| HOF   | Higher-order functions                                            | Monomorphization, closure capture, ForallType          |
| 8     | GPU lowering                                                      | Maps, reductions, views, scan; sort, matmul (f32 only) |

## Examples

See the `examples/` directory:

| Example                       | Description                                   |
| ----------------------------- | --------------------------------------------- |
| `scalar_arithmetic.remora`    | Basic arithmetic operations                   |
| `scalar_branching.remora`     | If/select in scalar context                   |
| `function_application.remora` | Function composition                          |
| `dot_product.remora`          | Dot product via prelude                       |
| `prelude_sum.remora`          | Sum via prelude                               |
| `prelude_scale.remora`        | Scaling via prelude                           |
| `factorial.remora`            | Factorial (scalar loop)                       |
| `fibonacci.remora`            | Fibonacci (non-tail recursion)                |
| `nested_let.remora`           | Nested let in ML syntax                       |
| `shape_rank.remora`           | Shape and rank queries                        |
| `reduce_iota.remora`          | Fold over iota                                |
| `matrix_row_reduce.remora`    | Row-level reductions                          |
| `threshold_mask.remora`       | Conditional masking                           |
| `chained_maps.remora`         | Chained map operations                        |
| `lift_map.remora`             | Map lifting examples                          |
| `rank3_map.remora`            | 3D map                                        |
| `rank4_map.remora`            | 4D map                                        |
| `rank10_*`                    | Rank-10 examples (shape, rank, indexing, map) |
| `indexing.remora`             | Array indexing in ML syntax                   |
| `rank_polymorphism.lisp`      | Auto-lifting and broadcasting                 |
| `scans.lisp`                  | All scan, reduce, fold variants               |
| `views.lisp`                  | View operations and primitives                |
| `conditional.lisp`            | If/select/branching                           |
| `tail_recursion.lisp`         | Tail-recursive patterns                       |
| `integration.lisp`            | Multi-phase calculus integration              |
| `ad_*.lisp`                   | Automatic differentiation examples            |
| `ad_optimize.lisp`            | AD optimization (grad-lifting + state fold)   |
| `cnn.lisp`                    | Convolution examples                          |
| `bool_logic.remora`           | Boolean logic in ML syntax                    |
| `section_right.remora`        | Right sections in ML syntax                   |

Run any example:

```bash
uv run remorac examples/scalar_arithmetic.remora          # ML, CPU
uv run remorac --syntax lisp examples/scans.lisp          # Lisp, CPU
uv run remorac --target interp examples/cnn.lisp          # Interpreter
uv run remorac --target cuda examples/scans.lisp    # GPU
```

## Inspecting Compiler Stages

```bash
uv run remorac --emit-ast file.remora                    # Parse tree (AST)
uv run remorac --emit-typed-ast file.remora              # Type-checked AST
uv run remorac --emit-hir file.remora                    # HIR before lowering
uv run remorac --emit-mlir file.remora                   # MLIR before passes
```

## Reference Interpreter

The reference interpreter (`--target interp`) supports the full dense core and is
more tolerant than the CPU backend. Use it to test programs or as an oracle: it is
the ground truth for correct numeric output.

```bash
uv run remorac --syntax lisp --target interp examples/integration.lisp
```
