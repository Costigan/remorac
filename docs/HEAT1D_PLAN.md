# Heat1D Implementation Plan (RemoraC)

## Overview

Implement Haynes' 1D lunar heat flow model (Hayne et al. 2017) using RemoraC
for the numerical core (array computations, tridiagonal solve, CN coefficients)
and Python for orchestration (time stepping, Picard/Newton convergence, I/O).

Reference oracle: `/e/projects/heat1d` (Python implementation by Paul Haynes).

## Architecture

```
Python (orchestration):
  ├── setup_spatial_grid(N, H, skin_depth, ...) → z[N], dz[N-1], rho[N]
  ├── compute_initial_temp(N)                      → T[N]
  ├── for each time step t:
  │   ├── newton_surface_temp(Q_s, T, ...)         → T_s_new
  │   ├── compute_cn_coefficients(T, rho, dz, dt)  → sub, diag, super, rhs
  │   │   └── (Remora: interior CN coeff assembly)
  │   ├── [if Picard] loop until convergence:
  │   │   ├── update_K_Cp(T_guess)                  → K_half[N-1], Cp[N]
  │   │   ├── compute_cn_coefficients(...)           → tridiagonal system
  │   │   ├── thomas_solve(sub, diag, super, rhs)   → T_new
  │   │   │   └── (Remora: Thomas tridiagonal solver)
  │   │   └── check_convergence(T_new, T_guess)
  │   └── T = T_new

Remora (compiled CPU):
  ├── compute_cn_coefficients  : computes interior CN tridiagonal coefficients
  ├── thomas_solve             : Thomas algorithm (forward sweep + back sub)
  └── compute_K(T), compute_Cp(T)  : temperature-dependent material properties
```

## Stage 1: Single 1D Column in Remora

### 1a. Constant coefficients, uniform grid (N=30)

**Goal:** Get end-to-end pipeline working with simplest physics.

- Python: grid setup, boundary conditions, time loop, Picard loop
- Remora: CN coefficient assembly, Thomas tridiagonal solver
- Validation: compare step-by-step against NumPy implementation

**Remora functions to implement:**

#### `thomas_solve` (Lisp syntax, uses `iscan`/`escan`/`trace-right`)
```
(define/pi () (thomas_solve
  [lower (Array Float 29) diag (Array Float 30) upper (Array Float 29) rhs (Array Float 30)]
  (Array Float 30))
  ;; Forward sweep: compute cp[N-1] then dp[N] via scans
  ;; Back substitution: compute x[N] via reverse scan
  ...)
```

Forward sweep uses two `iscan` passes:
1. Compute `cp[i] = upper[i] / (diag[i] - lower[i-1] * cp[i-1])`, with `lower[-1]=0`
2. Compute `dp[i] = (rhs[i] - lower[i-1] * dp[i-1]) / m[i]` where `m[i] = diag[i] - lower[i-1] * cp[i-1]`

Back substitution uses `trace-right` or manual right-fold:
3. `x[n-1] = dp[n-1]`; `x[i] = dp[i] - cp[i] * x[i+1]` for i = n-2..0

Inputs to both scans are "zipped" tuples of (upper/diag/lower_prev) or
(rhs/m/lower_prev) constructed via `map`.

#### `assemble_cn_interior` (Lisp syntax)
```
(define/pi () (assemble_cn_interior
  [T_old       (Array Float 30)
   T_bc_top    Float
   T_bc_bot    Float
   rho         (Array Float 30)
   Cp          (Array Float 30)
   K_half      (Array Float 29)
   dz          (Array Float 29)
   dt          Float]
  (Pair (Pair (Array Float 29) (Array Float 30))  ;; (lower, diag)
        (Pair (Array Float 29) (Array Float 30)))) ;; (upper, rhs)
  ;; Computes Crank-Nicolson tridiagonal system for interior nodes
  ;; Returns: lower[N-1], diag[N], upper[N-1], rhs[N]
  ...)
```

The CN discretization for node i (interior):
- `g1[i] = 2*dz[i+1] / (dz[i]*dz[i+1]*(dz[i]+dz[i+1]))` for i=0..N-3
- `g2[i] = 2*dz[i]   / (dz[i]*dz[i+1]*(dz[i]+dz[i+1]))` for i=0..N-3
- `a[i] = dt * g1[i-1] * K_half[i-1] / (rho[i] * Cp[i])` for i=1..N-2
- `b[i] = dt * g2[i-1] * K_half[i]   / (rho[i] * Cp[i])` for i=1..N-2
- CN half-coeffs: `ha = 0.5*a, hb = 0.5*b`
- LHS: `sub[i] = -ha[i+1], diag[i] = 1+ha[i]+hb[i], sup[i] = -hb[i]`
- RHS: `rhs[i] = ha[i]*T_old[i-1] + (1-ha[i]-hb[i])*T_old[i] + hb[i]*T_old[i+1]`
- BC rows modified: row 0 uses T_bc_top (Dirichlet), row N-1 uses T_bc_bot

### 1b. Move Thomas solve into Remora

The Thomas algorithm uses `iscan` (inclusive scan) for the forward sweep. Scan
is fully supported in the CPU compiled path (`_lower_scan_module`,
`_lower_scan_rank1`). The algorithm architecture:

1. **Prepare scan inputs:** For cp computation, zip `(upper, diag[:n-1], shifted_lower)`
   into `[n-1, 3]` array. For dp computation, zip `(rhs, m, shifted_lower)` into
   `[n, 3]` array (where m is precomputed denominator).

2. **Forward sweep (cp):** `cp = iscan cp_step 0.0 triples_cp`
   where `cp_step(prev_cp, [u, d, l]) = u / (d - l * prev_cp)`

3. **Forward sweep (dp):** `dp = iscan dp_step 0.0 triples_dp`
   where `dp_step(prev_dp, [r, m, l]) = (r - l * prev_dp) / m`

4. **Back substitution:** `x = escan-right back_step dp_last paire_dp_cp_padded`
   where `back_step(x_next, [dp_c, cp_c]) = dp_c - cp_c * x_next`

Fallback if scan proves problematic: recursive Remora function with `append`
(array construction is O(n^2) but fine for N≤100). Or: Python-side Thomas.

### 1c. Temperature-dependent K(T), Cp(T)

After constant-coeff version works:

- `compute_Cp(T)`: 4th-order polynomial `c0 + c1*T + c2*T^2 + c3*T^3 + c4*T^4`
- `compute_K(T, rho)`: `K_c(z) * (1 + chi/350^3 * T^3)` where `K_c(z)` scales
  with `rho(z)`
- Both implemented as Remora functions using `map` over temperature array
- Picard loop: Python recomputes K/Cp with current guess, re-calls Remora for
  CN assembly and Thomas solve

### 1d. Non-uniform grid

After uniform grid works:

- Python computes non-uniform grid via geometric progression (as in Haynes)
- `g1[i], g2[i]` geometric coefficients computed in Python
- Pass g1/g2 arrays to Remora CN assembly (or compute in Remora using `map`
  over dz triples)

## Stage 2: Rank-Polymorphic Lifting to 2D (deferred)

- Reshape data to [Y, X, N] tensors
- Each (y,x) is an independent 1D column
- Try automatic lifting: pass [Y, X, N] arrays to the Stage-1 functions
- Remora's rank polymorphism should lift map/fold/scan over the [Y,X] frame
- If sequential Thomas algorithm (loop-carried dependency along N) doesn't
  compose with automatic lifting, fall back to Python for-loops calling
  compiled Stage-1 functions per column

## Stage 3: Performance and Validation (deferred)

- N=100, Y=32, X=32, 100 time steps
- Compare against Haynes' reference implementation
- Time Remora-compiled path vs pure Python

## File Layout

```
remorac/
  docs/HEAT1D_PLAN.md          # This plan
  examples/heat1d/
    __init__.py                 # (empty)
    heat1d_model.py             # Python orchestration
    remora_solvers.rem          # Remora source: Thomas + CN assembly
    test_heat1d.py              # Verification tests
```

## Implementation Order

1. `examples/heat1d/remora_solvers.rem` — Thomas algorithm (scan-based)
2. `examples/heat1d/remora_solvers.rem` — CN coefficient assembly
3. `examples/heat1d/heat1d_model.py` — Python orchestration
4. `examples/heat1d/test_heat1d.py` — verification against NumPy
5. Add K(T), Cp(T) temperature dependence
6. Non-uniform grid geometry

## Key Remora Constructs Needed

| Construct | CPU Support | Used For |
|-----------|-------------|----------|
| `map` | ✓ | Element-wise arithmetic, coefficient computation |
| `fold` | ✓ | Reductions |
| `iscan` / `escan` | ✓ | Thomas forward sweep / back substitution |
| `trace` / `trace-right` | ✓ | Alternative to escan for back substitution |
| `let` | ✓ | Local bindings, intermediate values |
| `if` | ✓ | Boundary condition branching |
| `lambda` | ✓ | Scan step functions |
| `iota` | ✓ | Index array construction |
| Array indexing `xs[i]` | ✓ | Element access |
| Array slicing | ✓ | Trimming arrays |
| `append` | ✓ | Array construction (fallback) |
| Recursion | ✓ | Fallback for sequential algorithms |

## Stage 1 Results (2026-06)

### What works (12/12 tests passing)

- **Thomas tridiagonal solver** (Python): Verified against `np.linalg.solve` for
  random diagonally-dominant systems, identity matrices, and known 3×3 systems.
- **Crank-Nicolson time stepping**: Single-step structure validated (surface BC
  enforced, no NaN/inf, temperatures bounded).
- **Steady-state convergence**: Two test cases — a thin domain (z_max=0.02 m,
  high diffusivity) and a moderate domain (z_max=0.1 m) — both converge to the
  analytical isothermal solution within the step budget.
- **Heat conservation**: With adiabatic boundaries, total heat content is
  conserved to machine precision.
- **Grid scaling**: Works for N = 10, 30, 100 grid points.

### Remora lowering gaps discovered, pursued, and resolved

#### Gaps fixed (June 2026)

| Gap | Fix | Location |
|-----|-----|----------|
| `iscan`/`escan`/`trace` with lambda step function | Lambda body inlined via `_lower_body_in_loop`; `_resolve_scan_function` resolves HIRVar refs | `tensor_ops.py` |
| Scan init/element type constraint (init ≠ element when element is ArrayType) | Removed over-constraint from `_infer_scan`/`_infer_trace`; fixed result type for heterogenous init/element | `typechecker.py` |
| Interpreter scan dtype (Float truncated to Int when scanning over Int indices) | Use `expr.type` instead of `np.empty_like(array)` for result dtype | `runtime.py` |
| No scientific notation in Lisp reader | Extended FLOAT regex in both parsers | `lisp_reader.py`, `grammar.lark` |

#### Gaps still open

| Attempt | Blocker | Detail |
|---------|---------|--------|
| Closure-capturing scan lambdas (compiled path) | `_lower_tensor_let_module` rejects scan in let chain | Scan lambda references external arrays via let-bindings; lowering doesn't thread captures |
| Recursive array construction with `define/pi` | Type checker rejects mismatched sizes | `(append [val] (recursive_call))` produces size k+1 but return annotation says size N |
| `map` over `iota` calling recursive helper | `DefunctionalizationError: unknown HIR function` | Cross-function calls in closure-capturing map bodies aren't resolved |
| `map` with `index-item` in body | `RemoraLoweringError: cannot lower HIRIndex in map body` | `tensor_ops.py` map body emitter rejects HIRIndex |

#### Thomas algorithm status

The Thomas forward sweep (cp computation) works correctly in the **interpreter**:

```lisp
(let ((upper [1.0 2.0 3.0])
      (diag [10.0 15.0 20.0])
      (lower [3.0 5.0 7.0]))
  (iscan (lambda (prev i)
    (let ((u (index-item upper i))
          (d (index-item diag i))
          (l (if (< i 1) 0.0 (index-item lower (- i 1)))))
      (/ u (- d (* l prev)))))
    0.0 (iota 3)))
; ⇒ [0.1, 0.136, 0.155]
```

The **compiled** path is blocked by the closure-capturing scan lambda gap:
the scan body references `upper`, `diag`, `lower` captured from enclosing
let-bindings, and the let-lowering pass rejects the resulting HIR.

#### Takeaway

For Stage 1a (constant coefficients, uniform grid), the per-step computation is
trivially fast in NumPy. The Thomas algorithm and CN coefficient arithmetic are
O(N) with small constant factors. Moving them to compiled Remora would bring no
meaningful speedup. The value of Remora compilation emerges in later stages:

- **Stage 1c (K(T), Cp(T)):** Element-wise polynomial/map evaluation over N
  points — a natural fit for `map` in Remora. The temperature-dependent K and
  Cp computation is compiled and used inside the Picard loop.
- **Stage 2 (rank-polymorphic lifting):** `map` over [Y,X,N] tensors where each
  column is independent. The element-wise K/Cp computation lifts automatically.

### Files created

```
examples/heat1d/
  __init__.py          # Empty package marker
  heat1d_model.py      # Heat1DModel class + Remora-compiled K(T)/Cp(T) + thomas_solve
  test_heat1d.py       # 14 tests (3 Thomas, 4 Remora properties, 5 model, 2 Picard)
```

### Next steps

1. **Stage 1c (done):** Temperature-dependent K(T) and Cp(T) compile and run
   via `CPUFunctionExecutor`. Python calls them inside the Picard loop.
2. **Stage 1d:** Non-uniform grid geometry (compute g1/g2 coefficients in
   Python, pass to Remora for CN coefficients).
3. **Closure-capturing scan lambdas:** Fix the let-lowering to thread captured
   arrays through scan lambdas. This would enable the full Thomas algorithm
   on the compiled path.
4. **Stage 2:** Rank-polymorphic lifting test. Arrange data as [Y,X,N] tensors
   and pass to the compiled K/Cp functions.
