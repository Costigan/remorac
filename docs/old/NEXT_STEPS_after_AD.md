# Next Steps After AD

AD (Phases 0-5) is complete. The Remora compiler now differentiates a useful
numerical-programming subset through tape-based reverse mode, source-to-source
gradient generation, interpreter execution, and compiled CPU execution with
partial GPU support.

## What You Can Do Today

```lisp
;; Unary scalar
(define/pi () (sq [x Float] Float) (* x x))
((grad sq) 3.0)  ;; → 6.0

;; Unary vector
(define/pi ([n Dim]) (sq-loss [x (Array Float n)] Float)
  (fold + 0.0 (* x x)))
((grad (iapp sq-loss 5)) [1.0 2.0 3.0 4.0 5.0])  ;; → [2,4,6,8,10]

;; Binary (pair-returning in interpreter)
(define/pi ([n Dim])
  (dot-loss [x (Array Float n) w (Array Float n)] Float)
  (fold + 0.0 (* x w)))
((grad (iapp dot-loss 4)) [1.0 2.0 3.0 4.0] [5.0 6.0 7.0 8.0])
;; → ([5,6,7,8], [1,2,3,4])  (interpreter pair)

;; Conditional (piecewise)
(define/pi () (piecewise [x Float] Float)
  (if (> x 0.0) (* x x) (- 0.0 x)))
((grad piecewise) 3.0)  ;; → 6.0
((grad piecewise) -3.0) ;; → -1.0
```

## What's Missing

### Language Gaps

| Feature | Why it matters |
|---|---|
| **Hessian / higher-order AD** | `grad(grad(f))` for Newton methods, curvature-aware optimizers |
| **`value-and-grad`** | Avoid recomputing the primal when you need both the loss and gradient |
| **`jacobian`** | Vector-valued functions; needed for ODE solvers, implicit layers |
| **Dimension multiplication in Pi** | `ravel` of a Pi-typed array needs `DimMul` for symbolic flattened length |
| **Array of function** | Differentiable `map` over function arrays |
| **Module / parameter system** | Manage trainable parameters separate from input data |

### Compiler Gaps

| Feature | Why it matters |
|---|---|
| **GPU execution for structured VJPs** | Append, subarray, index gradients run on CPU only |
| **GPU execution for n-ary `(grad f)`** | Descriptor-ABI GPU path currently handles unary only |
| **Buffer pooling (beyond liveness)** | `EvalTape` reference-count liveness releases dead buffers; a free-list would reuse them and avoid repeated allocations |
| **Checkpointing / recomputation** | Trade memory for compute: discard intermediate primals, recompute them during reverse pass |
| **AD-specific MLIR passes** | Fuse primal + gradient computation, eliminate redundant operations |
| **Shape-Pi specialization in source gen** | Generate concrete-shape gradient functions to avoid `define/pi` overhead |

### Ecosystem Gaps

| Feature | Why it matters |
|---|---|
| **Optimizer primitives** | SGD, Adam, etc. in Remora or via FFI |
| **Data loading** | Tensors from files, streaming batches |
| **Training loop** | `(train model data optimizer loss epochs)` |
| **ONNX / MLIR export** | Deploy trained models to inference runtimes |
| **Debugging / visualization** | Tape inspection, gradient checking tools, computation graph viz |

## Suggested Priority Order

### Short-term (1-2 sessions each)

1. **`value-and-grad`** — simple change: return `(pair primal gradient)` from the tape instead of just gradient
2. **Buffer pooling** — add a free-list to `EvalTape` that reuses buffers of matching shapes across `reverse()` calls
3. **AD user docs** — tutorial-style docs with worked examples for each supported operation

### Medium-term (3-5 sessions each)

4. **Hessian / `grad(grad(f))`** — apply AD to the gradient function itself; requires tape-to-source to emit a differentiable function
5. **GPU n-ary gradient** — extend descriptor-ABI path to multi-parameter functions
6. **Checkpointing** — add a `checkpoint` decorator that recomputes primals instead of storing them

### Long-term (design + implement)

7. **Module / parameter system** — `(Parameter name shape)`, `(module ...)`, `(parameters m)`
8. **Training loop** — `(train loss optimizer data)`, works with both CPU and GPU
9. **Shape-Pi specialization** — generate monomorphic gradient functions to avoid Pi overhead at runtime

## Suggested Next Concrete Task

**`value-and-grad`** — the smallest useful addition. Change `grad_via_tape` to return
both the primal output and the gradient. The API becomes:

```lisp
((value-and-grad f) x)  ;; → (pair (f x) (grad f)(x))
```

This requires: a `ValueAndGradExpr` AST node, Lisp reader support, typechecker
rule (`FuncType(A, Float) → FuncType(A, PairType(Float, A))`), and tape
modification to capture the primal output alongside the gradient.
