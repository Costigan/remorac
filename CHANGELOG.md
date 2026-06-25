# Changelog

All notable changes to RemoraC are documented here, organized by
feature area.  See also the per-phase changelog in the git history.

## f64 CPU Lowering + f64 Literal Syntax (June 2026)

### f64 CPU lowering pipeline

Extended the CPU compiled backend to support `float64` element types
end-to-end through typechecking, HIR lowering, MLIR emission, C runtime,
and display formatting.

- **Typechecker** (`typechecker.py`): `_infer_scan`, `_infer_trace`,
  `_infer_fold`, `_infer_reduce`, `_infer_fold_right` now use
  `common_numeric_type(init_type, element_type)` when both are
  `ScalarType` and `is_numeric`, so `float` init + `float64` array
  produces a `float64` result.  `is_numeric` guard prevents spurious
  `common_numeric_type` calls on bool operations.
- **HIR** (`hir.py`): `_prim_op_name` maps `FLOAT64` → suffix `"d"`
  (e.g. `+d` for `arith.addf` with f64 operands).
- **MLIR type mapping** (`lowering/types.py`): `type_to_mlir` maps
  `FLOAT64` → `"f64"`.
- **Arithmetic operations** (`operators.py`): Added `_ARITH_OPS_F64`,
  `_LLVM_OPS_F64`, `"f64"` entries in dispatch tables, `f64` handling in
  `comparison_predicate`.
- **CPU scalar lowering** (`lowering/scalar.py`): `_literal_value` handles
  `"f64"`; `_cast_if_needed` adds i32→f64 and f32→f64 casts; `_hir_prim_op`
  and `_emit_prim_op` accept `+d`/`-d`/`*d`/`/d`/`expd`/`logd` ops;
  `_prim_op_operand` emits f64 constants.
- **Tensor ops** (`lowering/tensor_ops.py`): Suffix stripping recognises
  `"d"`; `_cmp_op_to_mlir` handles all suffixes.
- **Module builder** (`lowering/module.py`): Auto-detect regex for
  `remora_sort_*` externs updated to include `f64`.
- **GPU expression compiler** (`_gpu_expr_lowering.py`):
  `_scalar_type_to_mlir` maps `FLOAT64` → `"f64"`.
- **Runtime** (`runtime.py`): `_cast_scalar` / `_numpy_dtype` support
  `FLOAT64`; `_eval_expr_node` handles `Float64Lit`.
- **AD tracer** (`ad.py`): `_trace_node` handles `Float64Lit` → `push_const`.
- **Display** (`display.py`): FLOAT64 formatting (`.15g`); array formatter
  selects FLOAT64 for `float64` element arrays.
- **Executor** (`executor.py`): `_PLAN_DTYPES` includes `"f64"`/`"float64"`.

### f64 C runtime extensions (`remora_rt.c`)

- Added `_cmp_f64_asc` comparison helper.
- Added `remora_sort_f64` (qsort-based).
- Added `_cmp_grade_f64` and `remora_grade_f64`.
- Added 1d aliases `remora_sort_1d_f64` and `remora_grade_1d_f64`.

### f64 literal syntax (`1.0d`)

- **AST node** (`ast_nodes.py`): New `Float64Lit` dataclass, added to
  `Expr` type union.
- **Lisp reader** (`lisp_reader.py`): `FLOAT64` terminal with `d` suffix
  regex; `float64_lit` transformer strips suffix before `float()`.
- **ML grammar** (`grammar.lark`): `FLOAT64` terminal + `float64_lit`
  rule in `primary`.
- **ML parser** (`parser.py`): `float64_lit` method, `Float64Lit` import.
- **Type checker**: `Float64Lit` → `FLOAT64` type.
- **HIR**: `Float64Lit` → `HIRLit(value, FLOAT64)`.
- **Interpreter**: `Float64Lit` → `float(value)`.

### Cache invalidation on C runtime changes

- `compute_cache_key` now includes a hash of `remora_rt.c` so changes
  to the C runtime invalidate stale cached `.so` files.

### Verified f64 CPU operations

identity, reduce, map, fold, scan, reverse, rotate, append, sort — all
produce correct numeric results on the CPU compiled path.

## GPU f64 Support — General Map, Scan, Reduce, Parity Tests (June 2026)

GPU descriptor-ABI kernels now fully support `float64` element types
through maps (simple and compound), reductions, and scans.  f32 was
silently assumed in several code paths; all have been corrected.

### General map builder f64 support (`gpu_lowering.py`)

- **Element type detection**: `input_elem_types` gains `"f64"` for
  `FLOAT64` params; `_slot_etypes` and `result_elem_type` now
  recognise `_FLOAT64`.
- **Primitive callable suffix**: `+d` / `-d` / `*d` / `/d` recognised
  alongside `+f` / `+i` for f64 primitive callables in map bodies.
- **Scalar all-mode routing**: `_use_serial` is forced True for
  non-f32 scans (f64/i32/bool), preventing the parallel Hillis-Steele
  path from silently miscompiling non-f32 scans.
- **Scan body text**: non-compound serial scan body blocks use
  `_scan_elem_type` instead of hardcoded `f32`.

### codegen KernelMeta f64 support (`codegen.py`)

- Both the first-pass and fallback `KernelMeta` construction paths
  recognise `"float64"` output element type (was silently
  defaulting to `"f32"`, causing f64 kernels to write to
  4-byte-wide output buffers).

### GPU expression compiler f64 support (`_gpu_expr_lowering.py`)

- `_lower_prim_op` strips `d` suffix for f64, mapping to `"f64"`
  element type.
- `_lower_fold_to_gpu`: iota→accumulator cast uses
  `_scalar_type_to_mlir(result_type)` instead of hardcoded `"f32"`.
- `_scalar_type_to_mlir` maps `FLOAT64` → `"f64"`.

### General map text emitter f64 fixes (`gpu_lowering.py`)

- `_emit_scalar_reduce` and `_emit_multi_reduce`: loop signatures,
  select types, fold ops, and done branches use the element type
  from the init expression instead of hardcoded `"f32"`.
- Added `i64 → f64` cast via `llvm.sitofp`.

### GPU f64 parity tests (6 tests, 6 passing)

New tests in `TestGPUNumericParity` (`tests/test_gpu_general_lowering.py`),
comparing GPU descriptor-ABI output against the reference interpreter:

- `test_f64_map_primitive` — `(map (+ 1.0d) a)`
- `test_f64_map_lambda` — `(map (lambda (x) (+ x 1.0d)) a)`
- `test_f64_map_binary` — `(map + a a)`
- `test_f64_reduce` — `(fold + 0.0d a)`
- `test_f64_scan` — `(iscan + 0.0d a)`
- `test_f64_map_compound_fold` — `(map (lambda (x) (fold + 0.0d (iota 2))) a)`

Parity harness upgraded: `_run_parity` accepts `dtype` parameter;
`_format_input` emits `1.0d`-suffixed literals for f64 oracle source.

## heat1d Lunar Thermal Model + CPU Lowering Gaps Closed (June 2026)

### Scientific notation in Lisp reader
- Extended `FLOAT` regex in both `lisp_reader.py` and `grammar.lark`
  to accept `1e5`, `1.5e-3`, `10E+3`, etc.  Plain integers (`10`)
  still match `INT` only (dimension contexts unchanged).

### Scan lambdas (CPU compiled path)
- `_resolve_scan_function` resolves step functions from `HIRVar`
  references (post-defunctionalization).
- `_lower_scan_rank1` and `_lower_scan_tensor_input` inline lambda
  bodies via `_lower_body_in_loop`, supporting `let`, `if`, arithmetic,
  and index operations inside scan lambdas.
- Removed `init_type == element_type` constraint from
  `_infer_scan`/`_infer_trace` (blocked scans with different carry
  and element types, e.g. scalar init + array elements).
- Fixed scan result type computation for heterogeneous init/element.
- Fixed interpreter scan result dtype — `np.empty_like(array)`
  inherited input's Int32 dtype, truncating Float results.

### Closure-capturing scan lambdas + full Thomas algorithm
- Added `HIRScan` dispatch in `_lower_main_result_with_tensor_env` so
  let-lowering can lower scans with captured arrays.
- Threaded `tensor_env` from enclosing let-bindings into scan lambda
  lowering via `_lower_scan_tensor_let_result`.
- Added `HIRIf` handling to `_lower_body_in_loop` (scalar conditions
  with `scf.if`), needed for guard expressions like
  `(if (< i 1) 0.0 ...)` in scan lambdas.
- Fixed comparison MLIR type annotation in `_lower_body_in_loop` to
  use operand type and add comma after `arith.cmpi/cmpf` predicates.
- Added `render_blocks()` to `_MLIRMainModuleBuilder` for extracting
  raw MLIR blocks without the module wrapper.
- Fixed `input_elem_remora` access for `HIRVar` array nodes in
  function-parameter scan lowering.
- The full Thomas tridiagonal solver (cp sweep → denominator → dp
  sweep → back-substitution via `trace-right`) compiles and runs on
  the CPU path, matching the Python reference to f32 precision.

### Index-in-map body (HIRIndex inside map lambdas)
- Added `_map_body_needs_tensor_lowering` to detect compound bodies
  (handles both `HIRLambda` and `HIRVar`-referenced functions).
- Added compound-body routing check in `_lower_iota_scalar_map_result`;
  redirects to `_lower_map_body_with_loops` when needed.
- `_lower_map_body_with_loops` resolves `HIRVar` function refs via
  `_function_as_lambda`.
- Exclude inlined lambda from separate function lowering when loop
  path is used (avoids double-lowering compile errors).

### Integer division `/i` scalar
- Added `/i` to accepted ops in `_hir_prim_op` and `_emit_prim_op`.

### Indices-of for arbitrary ranks
- Generalized from hardcoded rank 2/3 to arbitrary rank via
  `linalg.index` + `arith.select` chain.

### Ternary+ maps
- Type checker: `_infer_nary_map` handles 3+ arrays with arithmetic
  operators.
- Interpreter: `_nary_map_value` applies callable element-wise to
  N arrays.
- Compiler: `_lower_nary_map_module`/`_lower_nary_map_result` emits
  `linalg.generic` with N inputs and chained binary ops.
- Binary map path unchanged (len==2 goes through existing dispatches).

### heat1d model (`examples/heat1d/`)
- `Heat1DModel` class with Remora-compiled K(T), Cp(T), and Thomas
  solver.  Python handles CN coefficient assembly, Picard iteration,
  boundary conditions, and time stepping.
- 15 regression tests: 4 Thomas solve (Python + Remora parity),
  4 material properties, 5 model correctness, 2 Picard iteration.
- Python `thomas_solve` preserved as oracle in `test_heat1d.py`.

## heat1d — Orbit Model, CFL Warmup, and Reference Validation (June 2026)

The lunar heat flow model now faithfully reproduces the Hayne et al. (2017)
reference implementation, matching surface temperatures to 0.1 K and
depth swings to <0.5 K over a 708-Earth-day (24 lunar day) simulation.

### Verification against Hayne et al. (2017) reference

Validated against the open-source reference model by P. O. Hayne
([github.com/pog1990/heat1d](https://github.com/pog1990/heat1d)),
which was fitted to Diviner LRO radiometer data.  Comparison at
latitude −85° (lunar south pole), dt=360 s, Fourier-matrix equilibration,
3-cycle CFL warmup, 24 lunar days of Crank-Nicolson output:

| Metric | Reference | Our model | Δ |
|--------|-----------|-----------|-----|
| T_surf min | 56.6 K | 56.5 K | −0.1 K |
| T_surf max | 177.9 K | 177.9 K | 0.0 K |
| T_surf mean | 99.3 K | 99.2 K | −0.1 K |
| Swing at 0.33 m | 8.0 K | 7.6 K | −0.5 K |
| Swing at 0.66 m | 6.2 K | 5.7 K | −0.5 K |
| Mean at 0.66 m | 103.6 K | 102.1 K | −1.5 K |

The subsurface mean runs ~1.5 K cooler at depth, attributable to a
small orbit-phase offset accumulated during the warmup phase (our
single CFL-dt computation vs. the reference's per-step CFL recomputation).

### Component-level verification

- **Crank-Nicolson step**: single-step Remora TDMA vs reference NumPy Thomas
  solver matches to 3×10⁻⁵ K across all 19 grid nodes.
- **Surface Newton solver**: standalone comparison to 3×10⁻⁵ K.
- **Absorbed solar flux**: identical to machine precision (matching Keihm
  1984 angle-dependent albedo and cosine solar zenith).
- **Fourier-matrix solver**: surface temperature extrema match to 0.1 K.
- **Kcs consistency fix**: surface contact conductivity corrected from
  7.0×10⁻⁴ to 7.4×10⁻⁴ W/m/K in `fourier_solver.py` (matched the value in
  `heat1d_model.py` and the reference `Moon.ks`).

### Bug fix — CN upper-array indexing

Fixed a one-off error in the Crank-Nicolson tridiagonal assembly
(`_SOURCE` f-string): `upper[0]` was incorrectly `-hb[0]` instead of
`0.0`, and `upper[i]` for interior nodes used `hb[i]` instead of
`hb[i-1]`.  This corrupted the surface Dirichlet BC and caused
T_new[0] to drift from the imposed T_surf.  The fix restores the
correct Thomas tridiagonal structure.

### Orbit model — eccentric Earth orbit (`orbit.py`)

New module computing heliocentric distance r_au(t) and solar declination
dec(t) via Kepler's equation:

- Solves `M = E - e*sin(E)` for eccentric anomaly E via Newton-Raphson
- True anomaly nu = atan2(sqrt(1-e²)·sin(E), cos(E) - e)
- r_au = (1-e²) / (1 + e·cos(nu)) — gives perihelion boost of 3.4%
- dec = asin(sin(obliq) · sin(nu + Lp)) — seasons cycle at Earth's
  orbital rate

Parameters: e=0.0167 (Earth), obliquity=1.54° (Moon), YEAR=365.256 d.

### Driver improvements

- **Orbit-aware flux**: `absorbed_flux()` now receives r_au from
  `orbit_state()` rather than a constant 1.0 AU.  Both the warmup and
  output phases track absolute time `t_global` with continuous orbit
  advancement.  The equilibration phase preserves the Fourier initial
  state by using a fixed r_au and dec at the epoch.
- **CFL warmup phase** (matching reference Phase 1b): runs 3 diurnal
  cycles at CFL-limited time step (~2400 s) after equilibration to
  bridge the Fourier solution to the output time step.  Uses
  `Heat1DModel.step(dt_override=...)` for variable step size.
- **`compute_cfl_ts(model)`**: new helper computing `min(F·ρ·cp·dz²/K)`
  with F=0.5, matching the reference's `computeCFL`.
- CLI: `--no-warmup` flag to skip the CFL warmup phase,
  `--n-warmup` to set the number of warmup cycles (default 3).
- **Declination cycling removed from driver**: orbital declination is
  now computed by `orbit_state()`, replacing the standalone
  `_declination()` sinusoidal approximation.

### Documentation (`examples/heat1d/README.md`)

Added comprehensive README covering model physics, grid construction,
solver architecture, CLI usage, side-by-side validation results against
the Hayne reference, and a future-work section proposing Remora
migration paths (bottom BC, surface Newton, Picard loop), Numba
acceleration of Python loops, FFT primitives in Remora for the Fourier
solver, GPU time-stepping path, and adaptive dt.

### Tests
- 7 scan lambda tests in `test_properties.py::test_scan_family`.
- 3 Remora-vs-Python Thomas parity tests in
  `examples/heat1d/test_heat1d.py`.
- Total: 290 tests passing (non-GPU), 0 regressions.

## Dense Core CPU — Complete

Full closure conversion, ForallType HOF monomorphization, all text-path
deferrals closed, builder path disabled.  The compiled-CPU path now
covers every construct the dense-core typechecker accepts.

### Full closure conversion
- `_ensure_lambda_typed` in `_infer_top_level_function_app` type-checks
  lambda bodies with their containing environment, producing `TypedLambda`
  with resolved free variables (`remora/typechecker.py`).
- `lower_expr` handles `TypedLambda` values directly, avoiding the
  re-typecheck-with-empty-env path that lost outer bindings (`remora/hir.py`).
- `_monomorphize_hof_calls` propagates concrete return types through
  HIRLet chains; `_replace_calls` updates `result_type` on HIRLet nodes
  when bodies are replaced (`remora/compiler.py`).
- `erase_to_hir` resolves TypeVar params to INT so monomorphized
  functions don't leak type variables into MLIR (`remora/erase.py`).
- `_free_vars` and `_free_vars_callable` exclude FuncType HIRVars —
  lifted function references are global, not closures (`remora/defunc.py`).
- Verified: `(let ((z 3)) (apply_twice (lambda (x) (+ x z)) 5)) = 11`,
  multi-capture variants, `row_norms.remora` now compiles.

### ForallType HOF monomorphization
- Lisp parser extended with `(Func (t1 t2 ...) ret)` type expression
  in `define/forall` signatures (`remora/lisp_reader.py`).
- `_infer_type_vars` walks `FuncType.result` for binder resolution;
  TypeVar-vs-TypeVar conflicts defer to concrete bindings from other
  parameters; concrete types replace provisional TypeVar bindings
  (`remora/typechecker.py`).
- Verified: `apply_twice inc 5`, `compose inc inc 5`, `apply2 square 5`
  with single- and multi-variable ForallType, all CPU compiled.

### Text-path deferrals closed
- **Fold operator sections**: resolved via `_lower_primitive_callable_result`
  (`tensor_ops.py:3173`).
- **Exclusive/right scans rank ≥ 2**: `_lower_scan_multirank_exclusive_right`
  emits `scf.for` with reversed loop or inverted carry logic
  (`tensor_ops.py:3757`).
- **Cell-fold producer map sections**: resolved via
  `_lower_primitive_callable_result` with section value threading
  (`tensor_ops.py:1461`).
- **Binary cell-map**: guard removed — cell dimensions are parallel
  iterators handled by the scalar lowering (`tensor_ops.py:1075`).
- **Cell-fold producer map lifted lambdas**: `HIRVar` callables resolved
  to `HIRPrimCallable` when the function body is a simple `HIRPrimOp`;
  arity mismatch for patterns like `(lambda (x) (* x x))` handled by
  operand replication (`tensor_ops.py`).
- **HIRVar.result_type bug**: fixed to use `.type` for HIRVar nodes in
  `_lower_if_tensor_input` (`tensor_ops.py:801`).

### Builder path disabled
- Builder API path (Stream E7) was ~175x slower than text path and
  fell back for 66% of programs.  Commented out with rationale;
  lowering always uses the text path now (`remora/lowering/module.py`).
- `arith.select` → `scf.if` in `test_lowering.py` — the text path
  correctly uses `scf.if` (eager `arith.select` would infinite-loop
  on recursive calls; the builder had this bug masked by fallback).

### Regression tests
- `test_closure_capture_in_hof_arg_compiled` added to
  `test_phase7_dependent_functions.py`.
- `row_norms.remora` moved from expected failure to passing.
- 988 non-GPU tests pass, 1 skipped (OpenMP availability check).

## Recursive Functions — Typechecker, Interpreter, CPU Compilation

Recursive `def` functions now typecheck, interpret (with tail-call
optimisation), and **compile to CPU** — covering self-recursion
(tail and non-tail), mutual recursion, and deep call chains.

### Typechecker (`remora/typechecker.py`)
- Removed the one-line recursion gate; self-referential calls now
  typecheck via fixpoint inference with a provisional `FuncType`.
- `_require` skips `TypeVar` comparisons during body inference; a
  post-inference `_substitute_type_var` pass resolves the provisional
  type variable to the concrete return type.
- Mutual recursion works automatically — when `f` calls `g` calls `f`,
  the fixpoint chain begins at `f`, extends through `g`, and resolves
  back to `f`'s provisional type.
- Non-index-arg `_typed_lambda_cache` avoids redundant re-inference.

### Interpreter (`remora/runtime.py`)
- `_gather_func_lambdas` extracts `FuncDef`-wrapping `TypedLambda`
  nodes from the typed AST and binds them as callables in the
  interpreter environment.
- `_lambda_callable` now captures `env` by reference so closures see
  their own name (enabling recursive self-calls).
- Self-tail-call trampoline (`_eval_expr_tail` + `_TailCall` exception)
  gives O(1) stack space; verified at 100k+ recursive calls.
- `TypeVar` bypass in `_coerce_runtime_value`.

### CPU compilation (`remora/hir.py`, `remora/erase.py`, lowering)
- `lower_to_hir` and `erase_to_hir` now emit `HIRFunction` nodes for
  every `FuncDef` in the typed program body (via `_gather_func_def_lambdas`).
- `_has_recursive_call` detects functions whose body contains a
  non-primitive call via `TypedExprNode(VarExpr)` (self or mutual);
  those emit `HIRCall` while prelude/utility functions still inline.
- HIRCall is lowered to MLIR `func.call @name` in the scalar path;
  mutual calls can resolve through the full `functions` dict (fix in
  `_lower_function`).
- `scf.if` replaces `arith.select` for if-expressions in the scalar
  emitter, avoiding eager evaluation of both branches that would
  infinite-loop on recursive calls.
- `func.func private` definitions are emitted for each `HIRFunction`,
  producing native recursive MLIR modules.

### REPL (`remora/repl.py`)
- `strict=False` on `evaluate_source_compiled` so the REPL gracefully
  falls back to the interpreter when compilation is unavailable.

### Tests
- Updated 3 rejection tests (typechecker, CLI, REPL) to assert correct
  recursive evaluation.
- Acceptance test `recursive_function` moved from `rejected` to
  `supported` (interp target); new passing `.remora` file added.
- 278 tests pass; no regressions.

### Verified patterns
```
def fac n = if n <= 1 then 1 else n * fac (n - 1)     → fac 5 = 120
def sum_to n acc = let r = n+acc in sum_to (n-1) r     → sum_to 10000 0 = 50005000
def fib n = if n <= 1 then n else fib(n-1)+fib(n-2)    → fib 12 = 144
def is_even n = ... is_odd (n-1)                        → mutual, is_even 5000 = True
my-def is_odd n = ... is_even (n-1)
```
All patterns work on the interpreter AND compiled CPU path.

### Array-valued recursive functions — CPU compilation

Array-typed recursive functions (e.g. `double arr n = ... double
(map (* 2) arr) (n - 1)`) previously failed in the CPU pipeline:
`bufferize-function-boundaries` rejected the `@double → @double`
callgraph cycle.  Fixed via manual bufferization:

- **`_lower_recursive_tensor_function`** (`module.py`): emits two
  functions — `@__{name}_mref` (memref-interface internal function,
  no tensor args/results) and a thin tensor wrapper `@{name}` that
  copies tensors→memrefs before the call and reads back after.
- **`_lower_mref_call`** (`tensor_ops.py`): detects memref-interface
  callees (name starts with `__`, ends with `_mref`) and wraps
  tensor args in `memref.alloc` + copy loops before the call.
- **`scf.if` branch placement**: `_lower_if_tensor_input_scalar_cond`
  now puts branch computations *inside* `scf.if` regions (not eagerly
  before them), so recursive calls are control-dependent and don't
  infinite-loop.
- **Module builder**: `_lower_iota_scalar_map_module` and friends
  pass `functions` to `_MLIRMainModuleBuilder` so recursive functions
  referenced from map bodies are included in the module.
- **`_lower_functions` filter**: only emits texts starting with
  `func.func` — scalar-returning fold/reduce bodies (raw code) stay
  inlined via callers.
- Verified: `double (iota 3) 2 = [0, 4, 8]`, `triple (iota 3) 3`,
  `scale_mult (iota 3) 3 2`, `ack 3 3 = 61`.

### `define/pi` mutual recursion — typechecker fix

Mutual recursion with `define/pi` failed because dimension-variable
binders matched trivially across call sites (both sides `DimVar("n")`)
but the constraint solver short-circuited without recording a binding.
- **Fix**: `_infer_index_bindings` now accepts remaining unbound
  binders when their name appears in the free index variables of the
  actual parameter types (dimension already resolved by caller context).

### Runtime: PipelineUnavailable fallback

- `evaluate_source_compiled(..., strict=False)` now catches
  `PipelineUnavailable` in addition to `RemoraLoweringError`, falling
  back to the interpreter when the MLIR toolchain is missing.

### Regression tests

- 15 regression tests across `test_execution.py` and
  `test_phase7_dependent_functions.py` covering scalar recursion,
  array-valued recursion, mutual recursion (2-way, 3-way, deep),
  `define/forall` recursive typecheck, `define/pi` mutual typecheck
  and interpreter, map-over-recursive (Lisp + ML), and Ackermann.
- 545 tests pass; no regressions.

## Higher-Order Recursion — Monomorphization, Closure Capture, Lambda Lifting

Higher-order functions (`def apply_twice f x = f (f x)`) now typecheck,
interpret, and **compile to CPU**.  Function values can be passed as
arguments, stored in let bindings, and used as map/fold callables.

### Typechecker — function values as arguments
- `VarExpr` handler falls back to `self._functions` when `env.lookup`
  fails, producing a `FuncType` with `TypeVar` params/result for plain
  `define` functions, or the declared type for `define/pi`/`define/forall`.
- `_infer_app` handles calls through `TypeVar`-typed variables by
  generating a fresh `FuncType` and proceeding with inference.
- `_require_numeric` and the `/` division check skip `TypeVar` types.
- `LambdaExpr` in value position returns a generic `FuncType` (no longer
  rejected as `"lambda expressions require an expected function type"`).
- `_infer_index_bindings` accepts unbound `define/pi` binders when their
  name appears in the free index variables of actual param types —
  enabling mutual recursion with `define/pi`.

### Interpreter — lazy FuncDef callables, lambda closure env
- `_bind_definition` creates a lazy callable (`_make_func_def_callable`)
  for `FuncDef`s not already bound by `_gather_func_lambdas`.
- `_make_func_def_callable` lazily typechecks the function body on first
  call via `specialize_top_level_function`.
- `_eval_expr_node` handles `LambdaExpr` by creating a closure that
  passes captured variable types to `check_callable`.

### Monomorphization pass (`remora/compiler.py`)
- **`_monomorphize_hof_calls`**: scans all `HIRCall` nodes whose callee
  has `FuncType` params.  For each concrete call site, clones the
  callee's `HIRFunction`, substitutes the `FuncType` param with the
  resolved function name, replaces `HIRCall` nodes to call-through-variable
  with direct calls, resolves `TypeVar` return types, and removes the
  original HOF function.
- **`_ensure_function`**: materializes missing `HIRFunction` entries for
  functions only used as values (via the typechecker).
- **`_substitute_hir`**: extended to handle `HIRCall`, `HIRIf`,
  `HIRFold`/`HIRReduce` with `TypeVar` result-type resolution via a
  `functions` lookup dict.
- **`_strip_hof_args`**: removes `FuncType` args from recursive self-calls
  in monomorphized bodies.
- **Let-bound alias resolution**: `HIRLet` bindings that alias known
  functions are resolved by redirecting `HIRCall` targets and dropping
  the let binding.  Handles nested aliases via substitution chaining.
- **`CompilerArtifact.return_type`**: falls back to the resolved HIR
  return type when the typed program's type contains `TypeVar`s.
- Verified: `apply_twice inc 5 = 7`, `compose inc inc 5 = 7`,
  `apply_twice (lambda (x) (+ x 1)) 5 = 7`,
  `repeat inc 3 0 = 3` (recursive HOF), all CPU compiled.

### Closure capture in map/fold callables (`remora/defunc.py`)
- `_rewrite_callable` checks `_free_vars_of_lambda` before lifting.
  If the lambda captures outer variables, it stays inline — the map/fold
  lowering resolves captures from the local tensor/scalar env.
- `_rewrite_function` passes function params in `scalar_env` so lambdas
  inside function bodies can reference them without being rejected.
- `_rewrite_call_arg` lifts capture-free `HIRLambda` in `HIRCall`
  argument positions to named functions.
- `_rewrite_lambda_value` handles `HIRLambda` in non-callable positions.
- `_lift_lambda` resolves `TypeVar` params and return type from the
  body expression; standalone function (formerly a `_Defunctionalizer`
  method, now module-level).

### HIR lambda body lowering (`remora/hir.py`)
- `_lower_typed_node` for `LambdaExpr` now lowers the body via
  `check_callable` instead of a dummy `HIRLit(0)` placeholder, so
  inline lambdas carry their actual computation into MLIR.

### Map over function variables
- Function names can be passed as `map`/`fold` callables:
  `map inc (iota 3) = [1, 2, 3]` (interpreter + CPU).

### Function-valued typechecker gates removed
- `frame.py:121`: `"map over function values is deferred"` — relaxed;
  `FuncType` treated as scalar for cell/frame decomposition.
- `frame.py:179`: `"function-valued map results are deferred"` — relaxed;
  passes through unchanged.
- `typechecker.py:1529`: same gate in binary map — relaxed.
- `typechecker.py:1141`: `"arrays of functions are deferred"` — relaxed;
  creates scalar-typed arrays for `FuncType` elements.

### Recursive HOF support
- Monomorphization correctly handles self-recursive HOFs by stripping
  `FuncType` args from recursive self-calls and hard-coding the concrete
  function in the body.
- `function_application.remora` example now compiles (was an expected
  failure).

### Regression tests
- 22 new regression tests across `test_execution.py` and
  `test_phase7_dependent_functions.py` covering: array-valued recursion
  (3), Ackermann, scalar regression (4), map-over-recursive (ML + Lisp),
  `define/forall` typecheck, three-function mutual (2), deep mutual,
  `define/pi` mutual (2), HOF apply_twice/compose/let/lambda/repeat (7),
  closure capture in map (2), and map lambda regression.
- 595 non-GPU tests pass; 1 pre-existing failure (`test_im2col.py`,
  unrelated `HIRVar.result_type` bug).

## GPU Radix Sort (256-bin, 4-pass)

New `remora/_gpu_radix_sort.py`: a device-resident LSD radix sort for
f32 arrays that replaces the O(N log²N) bitonic GPU sort for
1024 < N ≤ 1024² (bitonic remains the fallback below 1024).  Built and
validated kernel-by-kernel against a NumPy oracle before integration.

- **12-kernel `ExecutionPlan`**, wired into the GPU sort dispatch in
  `generate_mlir_descriptor_abi_ptx` (`codegen.py`) ahead of bitonic.
  Pipeline per sort: f32→uint32 monotonic key map (`llvm.bitcast` +
  sign flip) → 4 × [ per-block digit-major histogram (shared-memory
  `atomicrmw`) → exclusive prefix scan (digit-major decomposition into
  single-block Hillis-Steele scans) → stable scatter ] → key→f32, with
  ping-pong key buffers.
- **Stable per-digit local rank via warp intrinsics**:
  `rank = ctpop(match.any.sync(digit) & lanemask.lt)` per warp,
  aggregated across the 32 warps in shared memory; out-of-range lanes
  use a sentinel digit.  This is the O(N) equivalent of CUB/XLA's
  approach and is correct on duplicate-heavy input (stability verified).
- **Performance**: **607M elem/s at 1M** through the official
  `remora-perf` path (with H↔D transfer + 18 per-step plan syncs),
  **~1.35G elem/s device-resident** — **~9× the old bitonic (68M)**,
  above the 500M target and within ~3× of JAX/XLA.
- **Correctness**: bit-exact vs `np.sort` for N = 2K–1M across random,
  heavy-duplicate, negative, zero, and ±inf inputs.  New test
  `test_gpu_radix_sort_matches_numpy_when_available`; all 13 existing
  GPU sort tests and 87 executor/GPU-lowering tests still pass.
- **De-risking note**: the fast 256-bin design was initially deferred
  as "highest-risk" (hand-written warp-intrinsic MLIR).  A 5-minute
  empirical probe proved the warp intrinsics (`match.any.sync`,
  `vote.ballot`, `shfl.sync`, `ctpop`) lower cleanly through the repo's
  `mlir-translate-18 → llc-18 → ptxas` path (validated to a cubin),
  retiring the core risk and making the design tractable.

## Benchmark Improvement Plan (Phases 1–3)

Work driven by `docs/BENCHMARK_IMPROVEMENT_PLAN.md`.  New numbers in
`benchmarks/results/REPORT.md` (RTX 5090 Laptop GPU).

### CPU matmul — tiled C kernel (Phase 1.1)
- Replaced the naive `linalg.matmul`→loops path with a C runtime call
  `remora_matmul_f32` (`remora/remora_rt.c`): a register-tiled (4 rows
  of C per A load), cache-blocked SGEMM compiled `-O3 -march=native`.
  Calls `cblas_sgemm` when an optimized BLAS (OpenBLAS/BLIS) is found;
  the reference Netlib BLAS is intentionally skipped (no faster than
  the tiled C kernel).
- `_lower_matmul_tensor_input` (`lowering/tensor_ops.py`) emits the
  tensor→memref copies + `func.call @remora_matmul_f32` for rank-2 f32;
  `_lower_function_descriptor_module` (`lowering/module.py`) auto-detects
  the extern.  `runtime.py` adds `_find_optimized_blas`, `-march=native`,
  and conditional `-DREMORA_HAVE_BLAS`/`-lopenblas` link.
- 512×512: 248ms → 7.00ms (~34x; 1.1M → 37.4M elem/s; >50M up to 256).

### CPU sort — LSD radix sort (Phase 1.2)
- `remora_sort_f32` is now a 4-pass 8-bit LSD radix sort over a
  monotonic uint32 key mapping (sign-bit/negative flip), replacing
  `qsort`.  Profiling showed `qsort`'s indirect comparator — not the
  tensor→memref copy (~0.01ms for 100K) — was the bottleneck.
- 100K: 13.5M → 167.8M elem/s (~12.6x); 1M: 11.2M → 143.0M.

### GPU scan — multi-block enabled (Phase 1.3)
- The single-block scan builder (`build_descriptor_abi_f32_scan_gpu_module`)
  now raises `GPUScaffoldError` for inclusive left-to-right add scans
  with 1024 < N ≤ 1024², routing them to the existing four-kernel
  multi-block `ExecutionPlan` (previously dead code masked by a serial
  fallback).  Exclusive/right/mul/oversized scans keep the serial path.
- `bench_scan_remora_gpu` now compiles to HIR then calls
  `generate_mlir_descriptor_abi_ptx` directly, using `execute_plan`.
- GPU scan now works for n > 1024: 1M reaches 1.50G elem/s.

### Benchmark CLI: larger sizes, pool toggle (Phases 1.4–1.5)
- `DEFAULT_SIZES` gains 10M; `MATMUL_SIZES`/`STENCIL_SIZES` gain 1024.
- `RemoraExecutor.set_pool_enabled(bool)` + `remora-perf --no-pool`
  bypass the device memory pool; measured ~53us/call allocator saving.

### Benchmark coverage (Phase 2)
- **Application benchmarks**: `grad_descent` (numpy/jax/remora-cpu/gpu;
  all converge to `[0.512337, 0.433115, 0.911621]`), `conv_pipeline`
  (conv→relu→sum-pool; numpy/jax/remora-cpu match within 5e-7),
  `nbody` (all-pairs gravity; numpy/jax/remora-cpu match within 1e-3).
- **Fusion benchmarks**: composed vs hand-fused op-chains (`mapchain`,
  `triple`, `dot`); map_chain fuses perfectly, the 3-map `triple` chain
  shows a 1.46x penalty at 100K (not fully fused).
- New ops registered in `ALL_OPS`; per-op size lists added.

### Device-resident execution (Phases 2.2, 3.2)
- `RemoraExecutor`: `alloc_and_upload`, `download`, `free_device`,
  `execute_device` (launch on device-resident pointers, no H↔D copy),
  and `execute_to_device` returning a `DeviceArray`.
- New `DeviceArray` class (ptr/shape/dtype/nbytes, pool-allocated) with
  `from_numpy`, `to_numpy`, `free`.
- `remora-perf --device-resident` isolates transfer overhead (map 1M:
  690→86us; fold: 250→41us).  `examples/device_resident_iter.py`: a
  100-step on-device recurrence runs 9.87ms vs 72.14ms with per-call
  transfer (7.31x).
- Test: `test_device_array_round_trip_and_iteration_when_available`.

### GPU radix sort (Phase 3.1, `remora/_gpu_radix_sort.py`)
- New 256-bin (8-bit-digit, 4-pass) LSD radix sort for f32 arrays,
  exposed as a 12-kernel `ExecutionPlan` and wired into the GPU sort
  dispatch (`codegen.py`) for 1024 < N ≤ 1024²; bitonic remains the
  fallback below 1024.
- Pipeline: f32→uint32 monotonic key map (`llvm.bitcast` + sign flip)
  → per pass: per-block digit-major histogram (shared `atomicrmw`) →
  exclusive scan (digit-major decomposition) → stable scatter, with
  the per-digit local rank from warp intrinsics
  (`match.any.sync(digit) & lanemask.lt → ctpop`, aggregated across
  warps) → key→f32.  Built and validated kernel-by-kernel against a
  NumPy oracle.
- **607M elem/s at 1M** through the official benchmark (with H↔D
  transfer + per-step syncs), ~1.35G device-resident — ~9x the old
  bitonic (68M), above the 500M target.
- An empirical probe confirmed the warp intrinsics lower cleanly
  through the repo's `mlir-translate-18 → llc-18 → ptxas` path (to a
  cubin), which made the fast 256-bin design tractable; the original
  "deferred, highest-risk" assessment was over-cautious.
- Test: `test_gpu_radix_sort_matches_numpy_when_available` (N=2K..100K,
  random/duplicates/negatives, exact vs `np.sort`).

### Known limitations surfaced (documented, not fixed)
- CPU lowering gap: a `/` division inside a `map` body drops the
  `_mlir_ciface_remora_call` export symbol (blocks average pooling;
  conv_pipeline uses sum pooling instead).
- GPU general-map miscompile: a vector-valued (3-component) cell fold
  collapses to a broadcast scalar, so the N-body GPU output is wrong
  (`[s s s]` rows); `bench_nbody` omits remora-gpu.

## GPU Device Memory Pool

### Buffer arena for `RemoraExecutor` (`remora/executor.py`)
- Added `_pool: dict[int, list[int]]` keyed by allocation size,
  with `_pool_alloc()` and `_pool_free()` methods.
- `execute()` and `execute_plan()` now reuse pooled device buffers
  instead of calling `cudaMalloc`/`cudaFree` on every kernel launch.
- `close()` drains the pool, freeing all cached device pointers
  before closing modules and the CUDA runtime.

## Benchmark Suite (`remora/benchmark_suite.py`)

### Runtime performance benchmarks: Remora vs NumPy vs JAX
- New `remora-perf` CLI (`remora/benchmark_suite.py`) measuring
  execution time (not compilation time) for six array operations:
  map, fold, scan, matmul, sort, and stencil (3×3 box blur via
  im2col + fold-dot).
- Four backends: NumPy (CPU), JAX (GPU/XLA), Remora compiled CPU
  (MLIR), Remora compiled GPU (custom CUDA kernels).
- Compile-once-execute-many pattern: compiled artifacts are reused
  across warmup and timed iterations.
- GPU sort and matmul benchmarks construct HIR directly
  (`HIRSort`, `HIRMatmul`) and compile via
  `generate_mlir_descriptor_abi_ptx`.
- JSON output (`--json FILE`) and configurable `--ops`, `--backends`,
  `--sizes`, `--warmup`, `--trials` flags.
- Benchmark plan documented in `docs/BENCHMARK_PLAN.md`.
- Full report with analysis in `benchmarks/results/REPORT.md`.

### Key results (RTX 5090 Laptop GPU, 2026-06-20)
- Remora CPU stencil outperforms NumPy 2.5× (58M vs 24M elem/s)
  via MLIR loop fusion of im2col + fold-dot.
- Remora GPU stencil scales to 1.14G elem/s at 512×512 (49×
  faster than NumPy).
- Remora GPU matmul (tiled TILE=16) reaches 735M elem/s at
  512×512, within 6× of JAX/cuBLAS.
- Remora GPU fold reaches 3.9G elem/s at 1M elements.
- Remora GPU bitonic sort (62M elem/s at 1M) is the weakest
  result, 120× slower than JAX's radix sort.

## Examples

- `examples/embedded_remora.py`: scale, dot product, and negate
  functions compiled with `remora.define()` and called with NumPy.
- `examples/mandelbrot.py`: 800×600 Mandelbrot set rendered with
  three Remora-compiled kernels (`step_real`, `step_imag`, `mag_sq`)
  driving the z = z² + c iteration from Python.

## Removals

### `# coding: remora` source codec removed
- The custom Python source codec (`remora/codec.py`) and its
  `# remora:begin` / `# remora:end` block markers have been removed.
  The codec abused Python's encoding machinery for DSL embedding,
  required a `.pth` file for direct script execution, and re-invoked
  the Remora compiler on every module import.
- Examples now use `remora.define()` for embedding Remora in Python.
- Codec tests removed from `tests/test_api.py`.

## Bug Fixes

### N-body test fixes
- Fixed extra `)` in `_nbody_source_compiled` (unbalanced parens
  caused a Lisp parse error).
- Added `exp`, `log`, `sqrt` to `_ARITH_OPS_F32` in `operators.py`
  so math intrinsics compile in state-fold loop bodies.

### Multi-block bitonic merge direction
- Local sort kernels for multi-block sort and grade now reverse
  odd blocks' output to form bitonic sequences at block boundaries.
  Without this, the global merge produced per-block sorted output
  but failed to merge across blocks.

### HIRIota catch-all guard
- Added explicit `isinstance(array_expr, HIRIota)` check before
  the computed-expression CSE branch in the general map binding
  loop.  Without this, iota arrays were intercepted by the CSE
  handler instead of mapping to thread coordinates.

## GPU Expression Compiler (AD on GPU)

### `ad_optimize.lisp` runs on GPU
- 200-step gradient descent executes entirely on GPU via `LoopPlan`,
  producing the same result as CPU: `[0.512337, 0.433115, 0.911621]`.
- The AD source transform's 32,769-node gradient expression is
  collapsed by CSE before GPU compilation.

### HIRScatterAdd support (`_gpu_expr_lowering.py`)
- Static path: when the target is a `GpuArrayExpr` and the index is a
  compile-time literal, the scatter-add modifies the component directly.
- Dynamic path: when the target is a per-element expression, emits a
  runtime `GpuSelect(coord == index, target + update, target)` using
  the thread coordinate.

### HIRLet input alias propagation
- When `let coeffs = params` and `params` is in `input_map`, the let
  name `coeffs` is added to `input_map` as an alias.  This allows
  `HIRIndex(HIRVar('coeffs'), 0)` to resolve through the alias.
- Save/restore ensures the alias doesn't leak outside the let scope.

### `hir_optimize` bug fix (`hir_opt.py`)
- `hir_optimize` was discarding the CSE bindings returned by `hir_cse`
  — it passed only the rewritten expression (with dangling variable
  references) to DCE.  Fixed to wrap bindings as nested `HIRLet` before
  running DCE.

### CSE on computed map operands (`gpu_lowering.py`)
- When a map/apply operand is a non-trivial expression (not a variable,
  literal, or view op), it is CSE-optimized via `hir_optimize` before
  being wrapped as an `HIRLet` in the lambda body.
- Coordinate env seeded in `_gpu_emit_expr` so `GpuIndexCoordinate`
  resolves during MLIR emission.

### SSA name uniqueness fix
- `%gen_zidx_N` constants replaced with `_fresh_ssa()` counter to
  avoid duplicate SSA definitions when multiple scatter-adds emit
  the same zero-index constant.

### `GpuCompareOp` i64 support
- Integer comparisons (`i32`, `i64`) now emit `llvm.icmp` with
  signed predicates instead of `llvm.fcmp`.

## GPU Performance Kernels

### Tiled Shared-Memory Matmul
- TILE=16 cooperative loading with shared memory tiles.
- Each thread block computes one 16×16 output tile; edge tiles are
  bounds-checked and zero-padded for non-aligned dimensions.
- Falls back to the naive per-thread dot-product if the tiled version
  fails to compile.

### Bitonic Sort and Grade
- **Sort** (N ≤ 1024): O(N log²N) parallel bitonic sort in shared
  memory.  Non-power-of-2 sizes padded with +∞ sentinels.
- **Grade / argsort** (N ≤ 1024): same algorithm with a paired index
  array; outputs i32 permutation indices.
- **Multi-block sort** (N > 1024): per-block local sort followed by
  global compare-swap merge steps orchestrated via `ExecutionPlan`
  with double-buffered global memory.  Odd blocks reverse output to
  form bitonic sequences at boundaries.  Supports up to ~1M elements.
- **Multi-block grade** (N > 1024): per-block local grade followed by
  global merge steps that compare `values[idx]` and swap i32 indices.
  Same odd-block reversal for correct bitonic merge.

### Parallel Scatter-Add
- Single-block kernel (N ≤ 1024): all threads copy target→output in
  parallel, then thread 0 performs the scalar add after a barrier.
- Falls back to serial for N > 1024.

### Multi-Block Parallel Scan
- Four-kernel plan for N > 1024: per-block Hillis-Steele scan →
  extract block sums → scan block sums → propagate prefixes.
- Supports up to 1,048,576 elements (1024 blocks × 1024 threads).
- Single-block Hillis-Steele scan unchanged for N ≤ 1024.

### Parallel Filter
- Three-kernel `ExecutionPlan`: predicate evaluation → i32 prefix
  sum (Hillis-Steele) → scatter-write.
- Replaces the serial single-thread kernel for N ≤ 1024.

### Parallel Replicate
- Two-kernel plan: prefix sum on counts → scatter-replicate.
- Each thread writes `counts[i]` copies of `values[i]` to computed
  output positions.
- Replaces the serial single-thread kernel for N ≤ 1024.

## Multi-Kernel Orchestration

### ExecutionPlan Infrastructure (`remora/execution_plan.py`)
- `BufferSpec`: named device buffers with shape, dtype, optional
  `init` data for pre-populating from host.
- `KernelStep`: single GPU kernel launch with named buffer refs.
- `LoopPlan`: host-side iteration with buffer swapping (double-buffer
  pattern for optimization loops).
- `ExecutionPlan`: ordered sequence of steps with validation.

### RemoraExecutor Extensions (`remora/executor.py`)
- `execute_plan(plan, inputs)`: runs multi-step plans with named
  device buffers, kernel launches, host-side loops, and buffer
  swapping.
- `add_module(ptx, kernels)`: load kernels from additional PTX
  modules.
- `KernelMeta.grid_size`: explicit grid size override for 2D tiling.

### State-Fold GPU Detection (`remora/codegen.py`)
- `try_compile_state_fold_gpu(program)`: detects
  `fold body init (iota N)` where the body ignores the step variable,
  compiles the body as a GPU map kernel, and emits a `LoopPlan` with
  N iterations and buffer swapping.
- Wired into `execute_program_on_gpu()` as a first-pass attempt
  before the IREE pipeline.

## GPU Codegen Improvements

### HIRApply Support
- `build_descriptor_abi_general_map_gpu_module` now accepts `HIRApply`
  (the generalized rank-polymorphic application node) alongside
  `HIRMap`.
- Codegen fallback also accepts `HIRApply` with any callable type
  (not just `HIRLambda`).

### ArrayType Import Fix
- `ArrayType` imported at module level in `codegen.py` to avoid
  `UnboundLocalError` from Python's scoping rules when local imports
  inside try blocks shadowed the name.

### Tiled Matmul i1/i64 Fix
- `llvm.and` of `llvm.icmp` results uses `: i1` (not `: i64`) in
  the tiled matmul bounds-checking logic.

## Python Integration (Phases 1-3)

### Phase 1 — Core Wrapper
- `RemoraFunction` callable wrapper with JIT rank/shape checking.
- `compile_function()`, `compile_all()`, and `remora.define()` APIs.
- `RemoraRankMismatchError` for clear boundary violations.

### Phase 2 — Jupyter Cell Magic
- `%%remora` cell magic with `--target cpu|interp|gpu`, `--syntax`,
  `--out`, `--types` flags.
- Multi-definition cells via `compile_all()`.
- Error reporting: compiler errors propagate to notebook output.

### Phase 3 — Developer Experience
- `remora.define(source)`: returns `RemoraFunction` (single def) or
  `dict` (multiple defs).
- `%remora_eval` line magic: persistent REPL session in IPython with
  `--target`, `--syntax`, `--reset` flags.
- `%%remora --types`: inline type display for compiled definitions.

## AD and Optimization

- Five AD example files: `ad_polynomial`, `ad_circle`, `ad_spring`,
  `ad_softmax`, `ad_optimize`.
- `ad_optimize.lisp`: 200-step gradient descent on polynomial
  curve-fitting loss; works on interpreter, compiled CPU, and GPU
  (result: `[0.512337, 0.433115, 0.911621]`).
- Grad-lifting pass (`_rewrite_applied_source_gradient`):
  `(grad f)` resolved at any depth via `_collect_typed_grads`.
- State fold lowering (`_lower_state_fold_result`): `scf.for` with
  scalar decomposition for array-valued fold accumulators.

## Lisp Syntax

- `::` let-form replaced entirely with standard `let`/`let*`
  (Scheme-style).  Grammar, transformer, all tests and docs updated.

## Test Infrastructure

- Pre-existing test failures fixed: restored `docs/DENSE_CORE.md`,
  `docs/ABI.md`, `docs/IMPLEMENTATION_NOTES.md`, and
  `docs/BENCHMARK_BASELINES.json` from `docs/old/`.
- GPU integration tests (`tests/test_gpu_integration.py`): 13 tests
  verified on NVIDIA RTX 5090 (compute capability 12.0) covering
  map, sort, grade, matmul, reduction, AD gradient descent, and
  multi-block grade.
- Execution plan unit tests (`tests/test_execution_plan.py`):
  19 tests for plan construction, validation, and state-fold
  detection.
- Python integration tests (`tests/test_api.py`, `tests/test_magics.py`):
  22 tests for `RemoraFunction`, `define()`, cell magic, and
  REPL integration.
- N-body tests (`tests/test_nbody.py`): 5 tests passing after
  paren-balance and math-ops fixes.
- Total: 938 non-GPU + 13 GPU tests passing, 0 xfails, 0 regressions.
