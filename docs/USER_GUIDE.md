# Remora — User Guide

## What is Remora?

Remora is an array programming language for high-performance numerical
computation on CPUs and GPUs. It belongs to the same family as APL, J, and
Futhark: programs operate on whole arrays at once. Remora adds a static type
system that catches shape errors before execution and enables efficient
compilation to parallel hardware via MLIR.

The current implementation supports two syntaxes — a Dense Core ML-like syntax
(`.remora` files) and a Lisp s-expression syntax (`.lisp` files) with
rank-polymorphic auto-lifting.

## Installation

```bash
git clone https://github.com/your-org/remorac
cd remorac
uv sync
```

Verify:
```bash
uv run remorac examples/hello.remora
```

## Quick start

### ML syntax (`.remora` files)
```remora
-- Dense Core: explicit maps and folds
def double x = x * 2
def sum xs = fold (+) 0 xs
map double (iota 10)
```

### Lisp syntax (`.lisp` files)
```lisp
;; Rank-polymorphic: auto-lifting
(+ [1 2 3] [4 5 6])
(define (double [x 0]) (* x 2))
(map double [1 2 3 4 5])
```

### CLI
```bash
remorac file.remora                  # ML syntax (default)
remorac --syntax lisp file.lisp      # Lisp syntax
remorac --emit-mlir file.remora      # Show generated MLIR
remorac --target interp file.remora  # Reference interpreter
remorac --target gpu-nvidia file.remora  # GPU execution (IREE)
```

### REPL
```
remora> :syntax lisp
remora> (+ [1 2 3] [4 5 6])
[5 7 9]
remora> :target interp
remora> (iscan + 0 [2 10 5])
[2 12 17]
```

## Syntax reference

### Literals
| Lisp | ML | Meaning |
|------|----|---------|
| `42` | `42` | Integer |
| `3.14` | `3.14` | Float |
| `#t` / `#f` | `true` / `false` | Boolean |
| `[1 2 3]` | `[1, 2, 3]` | Array |

### Let and If
| Lisp | ML |
|------|-----|
| `(:: x 5 (+ x 1))` | `let x = 5 in x + 1` |
| `(if (< 1 2) 10 20)` | `if 1 < 2 then 10 else 20` |
| `(select #t 10 20)` | `select true 10 20` |

### Arithmetic and Comparison
| Lisp | ML |
|------|-----|
| `(+ 1 2)` | `1 + 2` |
| `(< x 5)` | `x < 5` |
| `(&& a b)` | `a && b` |

### Definitions
| Lisp | ML |
|------|-----|
| `(define (f [x]) body)` | `def f x = body` |
| `(define (f [x 0]) body)` | rank-annotated param |
| `(define xs [1 2 3])` | `def xs = [1, 2, 3]` |

### Lambda
| Lisp | ML |
|------|-----|
| `(lambda (x) body)` | `\x -> body` |
| `(lambda (x y) body)` | `\x y -> body` |
| `(λ (x) body)` | same |

### Map and Fold
| Lisp | ML |
|------|-----|
| `(map (+ 2) xs)` | `map (+ 2) xs` |
| `(map f xs ys)` | `map f xs ys` |
| `(fold + 0 xs)` | `fold (+) 0 xs` |

### Reduce and Scan (Phase 3)
| Lisp | ML |
|------|-----|
| `(reduce + 0 xs)` | `reduce (+) 0 xs` |
| `(reduce/zero + 0 xs)` | `reduce/zero (+) 0 xs` |
| `(reduce/1 + 0 xs)` | `reduce/1 (+) 0 xs` |
| `(iscan + 0 xs)` | `iscan (+) 0 xs` |
| `(escan + 0 xs)` | `escan (+) 0 xs` |
| `(scan + 0 xs)` | alias for iscan |
| `(fold-right + 0 xs)` | `fold-right (+) 0 xs` |
| `(trace + 0 xs)` | `trace (+) 0 xs` |
| `(trace-right + 0 xs)` | `trace-right (+) 0 xs` |

Scan/zero and /1 variants available for all scan forms.

### Views and Primitives (Phase 4)
| Lisp | ML | Description |
|------|-----|-------------|
| `(iota 5)` | `iota 5` | Range 0..4 |
| `(iota1 n)` | — | Boxed dynamic iota |
| `(shape xs)` | `shape xs` | Shape vector |
| `(rank xs)` | `rank xs` | Dimensionality |
| `(length xs)` | `length xs` | Leading dim size |
| `(reverse xs)` | `reverse xs` | Reverse |
| `(transpose m)` | `transpose m` | Transpose |
| `(reshape xs [2 2])` | `reshape [2,2] xs` | Reshape |
| `(ravel m)` | `ravel m` | Flatten |
| `(take 2 xs)` | `take 2 xs` | First N elements |
| `(drop 2 xs)` | `drop 2 xs` | Drop first N |
| `(rotate xs 2)` | `rotate xs 2` | Circular shift |
| `(subarray m [1 0] [2 2])` | — | Extract sub-region |
| `(append xs ys)` | — | Concatenate |
| `(indices-of xs)` | — | Coordinate tensor |
| `(with-shape 5 [3 2])` | — | Broadcast replicate |
| `(index xs 0 1)` | — | Multi-dim index |
| `(index-item xs 0)` | — | Leading-dim index |
| `(filter pred xs)` | — | Boxed filter |
| `(replicate [2 1] xs)` | — | Boxed repeat |
| `(sort < xs)` | — | Sort (typecheck only) |
| `(grade < xs)` | — | Grade (typecheck only) |

### Reranking (Phase 5)
```lisp
(~(0 0) +)        ; desugars to (lambda ([x0 0] [x1 0]) (+ x0 x1))
(map (~(0 0) +) [1 2] [3 4])
```

### Boxes (Phase 6)
```lisp
(box [1 2 3])                              ; wrap in box
(unbox b (len v) body)                     ; open box, bind len and v
(iota1 5)                                  ; boxed iota: (Σ (len) [int len])
(filter (> 0) [1 -2 3])                    ; boxed filter
(replicate [2 1] [10 20])                  ; boxed repeat
```

## Rank Polymorphism (Phase 2)

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

## GPU Support (Phase 8)

Operations compilable to GPU via IREE:
- Element-wise maps (f32, i32, bool)
- Reductions (f32)
- Rotate, subarray, indices-of
- Scan, append (descriptor-ABI kernels)

```bash
remorac --target gpu-nvidia program.remora
```

## Automatic differentiation

Remora supports reverse-mode automatic differentiation via the `grad` operator.
`grad f` returns the gradient function of `f` with respect to its Float
parameters.

### Basic usage (Lisp syntax)

```lisp
;; Scalar function
(define/pi () (sq [x Float] Float) (* x x))
((grad sq) 3.0)                        ;; → 6.0

;; Vector loss
(define/pi ([n Dim])
  (sq-loss [x (Array Float n)] Float)
  (fold + 0.0 (* x x)))
((grad (iapp sq-loss 5)) [1 2 3 4 5])  ;; → [2 4 6 8 10]

;; Binary function — returns pair in interpreter, single gradient via compiler
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

| Operation | VJP | Notes |
|---|---|---|
| `+ - * /` | Standard | Scalar and elementwise |
| `fold + 0.0` | Broadcast adjoint | Sum reduction |
| `reshape`, `ravel` | Reshape cotangent | Shape-preserving |
| `transpose`, `reverse` | Inverse view on cotangent | Elementwise |
| `take`, `drop` | Zero-pad cotangent | Leading-dimension only |
| `append` | Split cotangent via `take`/`drop` | Rank-N (axis 0) |
| `subarray` | Scatter cotangent via zero-pad | Rank-1 only |
| `rotate` | Counter-rotate cotangent | Elementwise |
| `index` | Scatter via `scatter-add` | Compile-time-known indices |
| `if` / `select` | Route cotangent through active branch | Both branches traced |

### Compilation options

```python
# Compile a single gradient function (CPU)
from remora.compiler import compile_gradient_function_source
cpu = compile_gradient_function_source(source, "loss", (param_type,))

# Compile for GPU (elementwise and select gradients)
gpu = compile_gradient_function_source_to_supported_gpu_artifacts(
    source, "loss", (param_type,))

# Compile per-input gradients for multi-parameter functions
from remora.compiler import compile_gradient_functions_source
grads = compile_gradient_functions_source(source, "dot-loss", (tx, tw))
```

### Limitations

- `(grad f)` requires `f` to return a scalar `Float`
- All differentiated parameters must be `Float` or `Array Float`
- GPU kernels support elementwise and select operations only (structured views run on CPU)
- Multi-parameter `(grad f)` returns a pair in the interpreter but a single gradient via compiled CPU; use `compile_gradient_functions_source` for both
- Inline lambdas inside `map` that capture array variables may resolve to scalar-cell instead of vector-cell lifting. Use a named helper with explicit parameter types (e.g., `(define ([dot row x]) ...)`) to force vector-cell behavior

## Feature status by phase

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Lisp syntax | Full |
| 2 | Rank polymorphism | Scalar/vector auto-lift, broadcasting |
| 3 | Reduce/scan/fold/trace | 14/14 operators |
| 4 | Additional primitives | 11/12 (sort/grade typecheck only) |
| 5 | Reranking | Full |
| 6 | Boxes | Core done, execution deferred |
| AD | Automatic differentiation | Reverse-mode via tape + source gen |
| 8 | GPU | 8/14 (maps, reductions, views, scan, append) |

## Examples

See `examples/` directory:
- `examples/rank_polymorphism.lisp` — auto-lifting and broadcasting
- `examples/scans.lisp` — all scan/reduce/fold variants
- `examples/views.lisp` — view operations and primitives
- `examples/integration.lisp` — multi-phase integration

Run any example:
```bash
remorac --syntax lisp --target cpu examples/scans.lisp
```
