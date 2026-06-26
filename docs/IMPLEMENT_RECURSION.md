# Recursion Completion Plan

## Goal

Implement full support for recursion, tail and otherwise, in:

- the interpreter
- the CPU compiled backend

and implement broad-enough GPU recursion support for common use cases:

- recursive scalar helpers used inside GPU map bodies
- tail-recursive scalar helpers inside GPU map bodies
- clear, tested rejection for unsupported recursive GPU shapes

This document starts from the current implementation state. It is not a
record of the earlier plan.

## Current State

### Already Working

The following are covered by passing tests today:

- Recursive definitions typecheck.
- Interpreter evaluates recursive functions for ordinary recursion depths.
- Interpreter has an O(1) Python-stack trampoline for tail-recursive self
  calls and tail-position mutual recursion.
- CPU compiled backend supports scalar self-recursion:
  `fac`, `fib`, `sum_to`, `ack`.
- CPU compiled backend supports scalar mutual recursion:
  two-function and three-function examples.
- CPU compiled backend supports array-valued self-recursion with array
  parameters through the memref-interface shim in
  `remora/lowering/module.py::_lower_recursive_tensor_function`.
- CPU compiled backend supports scalar-returning self-recursive functions
  with array parameters through the same memref-interface path.
- CPU compiled backend supports some higher-order recursive forms after HOF
  monomorphization.
- GPU can inline some non-recursive helper calls into general map kernels.
- GPU has a narrow `_GpuTailLoop` implementation for one self-tail-recursive
  helper shape inside map lowering.

Relevant passing tests:

| Area | Tests |
|---|---|
| CPU scalar recursion | `tests/test_execution.py::*recursion*`, `test_ackermann_compiled` |
| CPU array recursion | `test_array_valued_recursion_compiled`, `test_array_valued_recursion_with_multiple_params`, `test_array_valued_recursion_deeper` |
| CPU scalar return + array params | `test_scalar_valued_recursion_with_array_param_compiled` |
| Mutual recursion | `test_mutual_recursion_regression_even_odd`, `test_three_function_mutual_recursion_interpreted_and_compiled`, `test_mutual_recursion_deep_interpreted_and_compiled` |
| HOF recursion | `test_hof_recursive_repeat_compiled` |

### Known Gaps

These gaps block the stated goal:

1. Interpreter non-tail recursion is not stack-safe. Deep non-tail recursion
   still uses Python call stack and can raise `RecursionError`.

2. CPU recursion is not yet full across all array/top-level forms.
   Known failing shapes include:
   - top-level compiled calls that pass array literals into recursive
     `define/pi` functions
   - CPU compiled mutual `define/pi` recursion with array parameters
   - recursive array-param cases where calls are not self-recursive or not
     shaped as the current memref shim expects

3. CPU recursive tensor lowering is still special-case oriented.
   `_lower_recursive_tensor_function` handles important self-recursive
   shapes, but the implementation is not yet a general recursion-group
   lowering for arbitrary SCCs with array parameters.

4. GPU recursion support is narrow and under-tested.
   `_GpuTailLoop` exists, but there are no direct GPU tests asserting numeric
   parity for recursive helpers. The recognizer currently handles only a
   direct shape like:

   ```remora
   if cond then base else self_call(updated_args)
   ```

   It does not handle:
   - non-tail recursion
   - mutual recursion
   - top-level scalar recursive kernels
   - recursive helpers with array parameters
   - common syntactic variants where the recursive call is under a `let`,
     cast, or simple wrapper expression

5. GPU failure messages can be misleading. The general map recursion failure
   can be masked by earlier specialized fallback errors from `codegen.py`.

## Target Semantics

### Interpreter

The interpreter is the semantic oracle.

It must support:

- self-recursion, tail and non-tail
- mutual recursion, tail and non-tail
- recursive `define/pi` and `define/forall`
- higher-order recursion after typechecking
- recursive functions with scalar and array parameters

The interpreter does not need to make all non-tail recursion O(1) memory, but
it must not fail because of Python's recursion limit for reasonable program
depths used in tests.

### CPU Backend

The CPU backend must compile and execute the same recursive Remora programs
as the interpreter, subject only to existing non-recursion language gaps.

It must support:

- scalar-returning recursive functions
- array-returning recursive functions
- recursive functions with scalar parameters
- recursive functions with array parameters
- self-recursion
- mutual recursion
- tail recursion
- non-tail recursion
- higher-order recursion after monomorphization
- descriptor ABI function compilation through `CPUFunctionExecutor`
- top-level source compilation through `evaluate_source_compiled`

### GPU Backend

The GPU backend target is intentionally narrower.

It must support common cases:

- non-recursive helper calls inside GPU map bodies by inlining
- self-tail-recursive scalar helper calls inside GPU map bodies
- accumulator-style helpers such as:

  ```lisp
  (define/pi ()
    (sum_to [n Float acc Float] Float)
    (if (== n 0.0) acc (sum_to (- n 1.0) (+ acc n))))

  (define/pi ()
    (f [xs (Array Float 4)] (Array Float 4))
    (map (lambda (x) (sum_to x 0.0)) xs))
  ```

- the same supported shape for `Float`, `Int`, and `Bool` where the scalar
  operations are otherwise supported by GPU expression lowering
- numeric parity against the interpreter on an actual GPU

It may reject:

- non-tail recursion
- mutual recursion
- recursion requiring a per-thread stack
- recursive functions that return arrays
- recursive helpers with array parameters
- top-level scalar recursive kernels, until scalar GPU kernels are a first
  class backend target

Unsupported GPU recursion must fail loudly with a recursion-specific error,
not a misleading fallback error.

## Architecture Decisions

### Interpreter: Trampoline Recursion

Current tail-call trampoline remains the fast path.

To support deep non-tail recursion, add an interpreter-level trampoline for
recursive function groups. This is a runtime evaluator transformation, not a
source or HIR rewrite.

Preferred approach:

1. Detect recursive SCCs among top-level functions at interpreter binding
   time.
2. For each SCC, evaluate function bodies through an explicit frame loop.
3. Tail calls replace the active frame.
4. Non-tail calls push a continuation frame.
5. Returns pop and resume the continuation.

The continuation frame only needs to support typed AST constructs used in
recursive function bodies. Implement it incrementally with tests:

- `TypedIf`
- `TypedLet`
- `TypedApp`
- `TypedCast`
- scalar primitive operations
- array values as opaque Python values passed through frames

This removes Python call-stack dependence for non-tail recursive calls.

### CPU: Native `func.call` Plus General Memref SCC Wrappers

For scalar-only recursive functions, the current native MLIR `func.call`
approach is acceptable. MLIR/LLVM handles the call stack.

For recursive functions with array parameters or array returns, direct tensor
function recursion is unsafe because `bufferize-function-boundaries` can hit
tensor callgraph cycles. The current memref-interface shim is the right
direction, but it must be generalized from self-recursive functions to
recursive SCCs.

Required design:

1. Build a HIR call graph for functions included in the compilation unit.
2. Compute SCCs.
3. For each recursive SCC touching arrays, emit one memref-interface function
   per HIR function:

   ```mlir
   func.func private @__f_mref(%out: ..., %arg0_memref: ..., %arg1: ...)
   func.func private @__g_mref(%out: ..., %arg0_memref: ..., %arg1: ...)
   ```

4. Emit public/private tensor wrappers with original names:

   ```mlir
   func.func private @f(%arg0: tensor<...>, %arg1: f32) -> ...
   func.func private @g(%arg0: tensor<...>, %arg1: f32) -> ...
   ```

5. Inside each memref-interface implementation, recursive calls to any SCC
   member lower to the corresponding `@__name_mref`.
6. Scalar return values use `memref<scalar>` output buffers.
7. Array return values use plain statically shaped output memrefs.
8. Array parameters inside memref implementations are available both as
   memrefs for recursive calls and as tensors when existing tensor lowering
   needs a tensor view.

This preserves existing tensor lowering while removing the self-recursion
restriction.

### GPU: Tail-Recursive Helper Loops

For GPU, do not implement a general recursive stack first. The target is
common, bounded, tail-recursive scalar helpers inside per-element map bodies.

Current `_GpuTailLoop` should become a tested, documented subset:

1. Detect self-tail-recursive helper functions before inlining.
2. Normalize simple tail-recursive bodies to a canonical loop form:

   ```text
   if done(args) then result(args) else self(update(args))
   ```

3. Accept benign wrappers around the tail call:
   - `let` whose body is the tail call
   - scalar casts that preserve supported scalar lowering
   - condition with recursive call in either branch, as long as exactly one
     branch is the tail call and the other is the base result

4. Reject non-tail recursive calls with:

   ```text
   GPU recursion supports tail-recursive scalar helpers inside map bodies only
   ```

5. Emit LLVM block-loop code through `_GpuTailLoop`.
6. Test numeric parity on GPU.

## Milestones

### Milestone 1: Baseline Audit and Regression Locks

Goal: lock in the current behavior and document known failures as xfail or
rejected tests before changing implementation.

Tasks:

- [ ] Add an interpreter test proving deep non-tail recursion currently fails
  or mark the desired behavior as xfail.
- [ ] Add CPU tests for known supported recursion shapes, if any are only
  covered by ad hoc probes.
- [ ] Add CPU rejected/xfail tests for current gaps:
  - top-level recursive `define/pi` call with array literal argument
  - mutual `define/pi` recursion with array parameters on CPU
- [ ] Add GPU rejected tests for unsupported recursion with recursion-specific
  error messages:
  - non-tail helper inside map
  - mutual helper inside map
  - recursive helper with array parameter
- [ ] Add at least one GPU accepted compile test for the current supported
  tail-recursive helper-inside-map shape, then convert it to numeric parity in
  Milestone 5.

Verification:

```bash
REMORA_TEST_GPU=0 uv run pytest tests/test_execution.py tests/test_phase7_dependent_functions.py
uv run python -m compileall -q remora
```

On a GPU machine:

```bash
uv run pytest tests/test_gpu_numeric_parity.py tests/test_gpu_general_lowering.py
```

### Milestone 2: Interpreter Full Recursion

Goal: interpreter supports deep tail and non-tail recursion without relying on
Python recursion depth.

Tasks:

- [ ] Add SCC discovery for top-level interpreted functions.
- [ ] Keep the existing tail-call trampoline for simple tail calls.
- [ ] Add explicit continuation frames for non-tail recursive calls.
- [ ] Support continuation frames for:
  - `if`
  - `let`
  - primitive scalar ops
  - function application
  - casts
- [ ] Ensure mutual non-tail recursion uses the same frame loop.
- [ ] Preserve normal Python call behavior for non-recursive functions.
- [ ] Add a guard or test strategy for intentionally diverging recursion.

Acceptance tests:

- [ ] `sum_to_tail 50000 0` returns expected result.
- [ ] `sum_to_non_tail 5000` returns expected result without
  `RecursionError`.
- [ ] `fib 20` returns expected result.
- [ ] mutual non-tail recursion of depth 5000 returns expected result.
- [ ] recursive function with array parameter interprets at depth 5000.

### Milestone 3: CPU Full Scalar Recursion

Goal: scalar-only recursion stays complete and stable while later tensor work
is added.

Tasks:

- [ ] Add SCC-aware function emission for scalar-only recursive groups, even
  if it continues to emit native `func.call`.
- [ ] Verify mutually recursive scalar functions are always emitted together
  in descriptor compilation.
- [ ] Add CPU tests for:
  - self tail recursion
  - self non-tail recursion
  - mutual tail recursion
  - mutual non-tail recursion
  - three-function mutual recursion
  - higher-order recursive function after monomorphization
- [ ] Add descriptor ABI tests using `CPUFunctionExecutor.compile_source`,
  not only `evaluate_source_compiled`.

Acceptance tests:

- [ ] All scalar recursion interpreter tests have CPU parity tests.
- [ ] `CPUFunctionExecutor` works for recursive scalar functions with one,
  two, and three scalar parameters.

### Milestone 4: CPU Recursive Functions with Arrays

Goal: CPU backend supports recursive functions with array parameters and/or
array returns for self and mutual recursion.

Tasks:

- [ ] Add call-graph SCC computation to module lowering.
- [ ] Replace self-only `_lower_recursive_tensor_function` routing with
  recursive-SCC routing.
- [ ] Generate memref-interface implementations for every function in a
  recursive SCC that touches arrays.
- [ ] Route calls between SCC members to `@__name_mref`.
- [ ] Keep calls from outside the SCC going through original tensor wrappers.
- [ ] Preserve scalar-only recursive calls as native `func.call` unless they
  belong to a mixed scalar/tensor SCC.
- [ ] Lower scalar-typed branches containing nested recursive calls without
  assuming the call is directly under `HIRIf`.
- [ ] Support `HIRLet` in scalar recursive branches without mutating shared
  scalar environments unsafely.
- [ ] Fix top-level compiled array literal arguments into recursive
  `define/pi` calls.
- [ ] Support mutual `define/pi` recursion with array parameters.
- [ ] Add tests for array parameters that are used and unused in recursive
  bodies, so accidental dead-parameter behavior is caught.

Acceptance tests:

- [ ] Self-recursive scalar return with array param:

  ```lisp
  (define/pi ()
    (rec_sum [a (Array Float 4) n Float] Float)
    (if (== n 0.0) 0.0 (+ n (rec_sum a (- n 1.0)))))
  ```

- [ ] Same function called from top-level source with an array literal.
- [ ] Same function compiled through `CPUFunctionExecutor`.
- [ ] Array-returning recursive function with array and scalar params.
- [ ] Mutual `define/pi` recursion with array params compiles and runs.
- [ ] Non-tail recursive scalar result with array params compiles and runs.
- [ ] Recursive function where the array param is indexed or folded in the
  base case, proving the array parameter is not only carried through.

### Milestone 5: GPU Tail-Recursive Helper Support

Goal: common tail-recursive scalar helpers inside GPU maps compile and pass
numeric parity tests.

Tasks:

- [ ] Make `_GpuTailLoop` detection explicit and tested.
- [ ] Normalize tail-recursive helper bodies before GPU lowering.
- [ ] Support accumulator-style helpers with one input value and one or more
  scalar accumulators.
- [ ] Support recursive call in either `if` branch.
- [ ] Support simple `let` wrappers around update expressions.
- [ ] Ensure loop-carried values preserve scalar types (`f32`, `f64`, `i32`,
  `i1`) instead of assuming `f32`.
- [ ] Add compile-time rejection for non-tail self calls.
- [ ] Add compile-time rejection for mutual recursion.
- [ ] Add compile-time rejection for recursive array-returning helpers.
- [ ] Fix `codegen.py` fallback error propagation so recursion-specific
  errors are not masked by bool/int/f32 specialized fallback errors.

Acceptance tests:

- [x] GPU numeric parity for `sum_to` helper inside `map`, `Float`.
- [x] GPU numeric parity for `countdown`/identity helper inside `map`, `Int`.
- [x] GPU numeric parity for boolean tail-recursive helper inside `map`, if
  bool map support is available for the shape.
- [ ] GPU numeric parity for helper with two accumulators.
- [ ] GPU rejected-not-silent test for non-tail recursion:

  ```lisp
  (define/pi ()
    (sum_to [n Float] Float)
    (if (== n 0.0) 0.0 (+ n (sum_to (- n 1.0)))))
  ```

- [ ] GPU rejected-not-silent test for mutual recursion.
- [ ] GPU rejected-not-silent test for recursive helper with array parameter.

### Milestone 6: GPU Documentation and User-Facing Behavior

Goal: GPU recursion support is honest, predictable, and easy to debug.

Tasks:

- [ ] Document supported GPU recursion subset in `docs/PROJECT_OVERVIEW.md`.
- [ ] Add CLI/API error text that states:

  ```text
  GPU recursion supports tail-recursive scalar helpers inside map bodies only.
  ```

- [ ] Ensure unsupported recursive GPU programs fail before PTX generation
  with `CodegenUnavailable` or `GPUScaffoldError`.
- [ ] Add tests that assert error messages for unsupported GPU recursion.
- [ ] Add examples for supported GPU recursion and unsupported alternatives.

## Test Matrix

Each row must have interpreter and CPU coverage unless explicitly marked GPU
only.

| Shape | Interpreter | CPU | GPU |
|---|---:|---:|---:|
| self tail scalar | required | required | helper-in-map required |
| self non-tail scalar | required | required | rejected |
| mutual tail scalar | required | required | rejected |
| mutual non-tail scalar | required | required | rejected |
| higher-order recursive scalar | required | required | rejected unless monomorphized to supported helper |
| self tail array return | required | required | rejected |
| self non-tail array return | required | required | rejected |
| scalar return + array params | required | required | rejected |
| mutual recursion + array params | required | required | rejected |
| recursive helper inside map | required | required | tail-only numeric parity |

## Done Criteria

The goal is complete only when all of the following are true:

- Interpreter tests prove deep tail and deep non-tail recursion do not hit
  Python recursion depth.
- CPU tests prove parity with the interpreter for scalar, array-param,
  array-return, self-recursive, mutual-recursive, tail, non-tail, and
  higher-order recursive cases.
- GPU tests include numeric parity for supported tail-recursive helper cases.
- GPU tests include rejected-not-silent coverage for unsupported recursive
  cases.
- GPU unsupported recursion errors are specific and not masked by fallback
  builder errors.
- `uv run python -m compileall -q remora` passes.
- `REMORA_TEST_GPU=0 uv run pytest` passes on a non-GPU machine.
- `uv run pytest` passes on the GPU development machine, including the GPU
  recursion parity tests.

## Implementation Notes

### Files Likely to Change

| File | Reason |
|---|---|
| `remora/runtime.py` | interpreter non-tail trampoline / continuation frames |
| `remora/compiler.py` | call graph/SCC plumbing, function collection fixes |
| `remora/lowering/module.py` | recursive SCC memref-interface lowering |
| `remora/lowering/tensor_ops.py` | memref call support and scalar/tensor recursive call routing |
| `remora/lowering/scalar.py` | scalar call routing where memref-interface calls appear under scalar expressions |
| `remora/_gpu_expr_lowering.py` | tail-recursive helper detection and rejection |
| `remora/gpu_lowering.py` | `_GpuTailLoop` emission and scalar type support |
| `remora/codegen.py` | preserve recursion-specific GPU errors through fallback cascade |
| `tests/test_execution.py` | interpreter/CPU recursion matrix |
| `tests/test_phase7_dependent_functions.py` | mutual/HOF recursion |
| `tests/test_gpu_numeric_parity.py` | GPU recursion parity |
| `tests/test_gpu_general_lowering.py` | GPU compile/rejection coverage |

### Avoiding False Confidence

- Compile-only GPU tests are not enough. Supported GPU recursion requires
  numeric parity against the interpreter.
- A top-level CPU test with an array literal is not equivalent to
  `CPUFunctionExecutor`; both paths must be covered.
- A self-recursive test is not enough for mutual recursion.
- A tail-recursive test is not enough for non-tail recursion.
- An unused array parameter is not enough to prove array-param recursion.

## Risks

1. Interpreter continuation frames can become a second evaluator. Keep the
   supported frame set small and grow only under tests.

2. CPU memref-interface SCC lowering can duplicate or omit functions if the
   function collection logic is incomplete. Build and test call graph SCCs
   explicitly.

3. Recursive tensor lowering can accidentally execute branch computations
   eagerly. All recursive `HIRIf` lowering must keep branch computations
   inside `scf.if` regions.

4. GPU tail-recursion support can silently miscompile loop-carried scalar
   types if it assumes `f32`. Test `f32`, `i32`, and `bool` where supported.

5. GPU fallback code can hide the real error. Preserve the most specific
   general-map recursion error when all fallback builders fail.
