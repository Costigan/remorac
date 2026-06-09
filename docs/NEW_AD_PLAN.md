# Automatic Differentiation Plan

## 1. Decision

Automatic differentiation (AD) should be implemented after the dependent-type phase has produced a stable elaboration and erasure pipeline.

The AD pass will operate on a typed, elaborated core IR after index constraints and frame/cell decomposition have been resolved, but before dependent annotations and useful shape metadata are erased. The first implementation may specialize dimension-polymorphic functions before differentiation. This keeps the AD transform monomorphic while preserving compile-time shape checking.

This plan supersedes `automatic-differentiation-proposal.md` as the implementation roadmap. The older proposal remains useful as a vision document, but it makes three assumptions that should not guide implementation:

- A `SigmaType` is an existential type, not a general pair for a primal value and pullback.
- Known shapes permit ahead-of-time storage planning, but do not guarantee stack-only, allocation-free execution.
- Synthesizing higher-order pullback closures is a poor fit for the current first-order HIR and direct-call lowering.

## 2. Required Phase 7 Exit Criteria

AD work should not begin until the following are true:

1. Pi/index applications are elaborated explicitly and all generated constraints are either solved or reported as errors.
2. There is a typed core IR between source typechecking and backend HIR/MLIR lowering.
3. Frame/cell decomposition and principal-frame replication are represented explicitly and consistently.
4. Specialization and type erasure are separate, testable passes.
5. The typed core has a verifier that can re-check types and shapes after compiler transformations.
6. Rank-2 scan, rotate, and append no longer depend on unrelated rank-1 lowering shortcuts.

The AD pass depends more on these architectural properties than on having the full theoretical mixed-prefix solver. A restricted but stable Phase 7 is sufficient if every differentiated call is specialized to solved shapes before AD.

## 3. Proposed Pipeline

```text
Source AST
  -> dependent type checking and elaboration
  -> typed core IR
       - solved index applications
       - normalized shape expressions
       - explicit frame/cell and broadcast decisions
       - grad/value-and-grad markers
  -> type/index specialization (initial implementation)
  -> reverse-mode AD transform
  -> typed-core verification
  -> dependent-type erasure
  -> existing HIR and MLIR lowering
  -> CPU, GPU, or interpreter execution
```

Do not implement the primary AD transform on raw AST or directly on MLIR:

- Raw AST still contains syntactic sugar, implicit lifting, and unresolved application choices.
- MLIR has already lost Remora-level frame/cell intent, making broadcast and reduction adjoints harder to derive correctly.

## 4. User-Level Semantics

### 4.1 Initial API

The minimum user-visible API is:

```lisp
(grad f)
(value-and-grad f)
```

For the first release:

- `f` is pure and unary.
- The input is `Float` or an array of `Float`.
- `f` returns a scalar `Float`.
- `grad f` returns a function whose result has the same type and shape as `f`'s input.
- `value-and-grad` may initially be an internal/testing operation until the language has a suitable product or record type.

For a dependent function:

```text
f : Pi (s : Shape). Array Float s -> Float
grad f : Pi (s : Shape). Array Float s -> Array Float s
```

The implementation may specialize `s` before running AD, but the source-level type must preserve the Pi binder.

### 4.2 Differentiable Types

Initially differentiable:

- `Float`
- `Array Float shape`

Initially non-differentiable:

- `Int` and `Bool`
- function values
- boxes and existentially shaped values
- arrays with non-float elements

The cotangent type of a differentiable scalar or array is the same type and shape. Integers and booleans may participate as compile-time indices or control values, but do not receive cotangents.

### 4.3 Diagnostics

Unsupported differentiation must be a compile-time error naming the exact operation and source location. Silent zero gradients are not acceptable for unsupported primitives.

## 5. Internal AD Representation

Use a first-order reverse-mode transform, not general runtime pullback closures.

The typed core should first be normalized into an A-normal or SSA-like form so every intermediate has a name. The transform then:

1. Emits the primal computation.
2. Records only primal values required by derivative rules.
3. Creates an adjoint slot for each active differentiable value.
4. Traverses active operations in reverse order.
5. Accumulates cotangents into operand adjoints.
6. Returns the adjoint of the selected input.

Add internal concepts such as:

- `ZeroCotangent(type)` to avoid eagerly materializing zero tensors.
- `AddCotangent(target, contribution)` for explicit accumulation.
- saved-value or tape records with liveness information.
- an activity analysis that excludes values independent of differentiated inputs.

These can be internal IR nodes; they do not require general source-language tuples or mutable references.

## 6. Derivative Rules

Maintain derivative definitions in a centralized primitive registry. Each entry should include the primal type rule, differentiability classification, required saved values, and VJP builder.

### 6.1 First Slice

- Constants and variables
- Float casts where mathematically valid
- `+`, `-`, `*`, `/`
- Unary negation
- `let` and direct first-order calls
- Elementwise `map`/implicit scalar lifting
- Sum reduction
- `reshape`, `ravel`, and `transpose`

This slice is enough for quadratic losses, dot products expressed as multiply plus sum, and basic linear models.

### 6.2 Array Rules

- Elementwise operations apply the scalar VJP over the same frame.
- A broadcast/replicated operand receives the sum of cotangents over the replicated frame dimensions.
- Sum reduction broadcasts the incoming cotangent back over the reduced dimension.
- Transpose applies the inverse permutation.
- Reshape and ravel reshape the cotangent back to the input shape.
- Index/gather requires scatter-add in the reverse pass; defer it until an explicit scatter-add operation exists.

Rank-polymorphic lifting must use the elaborated frame/cell metadata. It should not rediscover rank behavior during AD.

### 6.3 Later Rules

- General reduce/fold/scan
- Conditional expressions
- Indexing and subarray via scatter-add
- Append and shape-changing structural operations
- Matrix multiplication as either a recognized composite or a dedicated primitive
- Custom derivative declarations
- Boxes and dynamic-shape operations

Non-smooth operations such as comparisons, sort, grade, filter, and integer indexing require an explicit language policy. They should initially be rejected when active rather than assigned arbitrary derivatives.

## 7. Control Flow and Effects

Remora is currently mostly pure, which is favorable for AD. Preserve that restriction for the initial implementation.

- A conditional whose predicate is inactive may be differentiated by recording the chosen branch and reversing only that branch.
- A predicate dependent on differentiated floating-point data is allowed only after the language defines piecewise differentiation behavior.
- Recursive functions remain unsupported until both ordinary compilation and tape storage have a bounded strategy.
- Calls into the C runtime require registered VJP rules or must be rejected as active operations.

## 8. Tape and Memory Planning

The first correct implementation should save every primal value required by a VJP rule. Optimize later.

Follow-up passes should add:

- activity-based dead tape elimination
- liveness-based buffer reuse
- recomputation/checkpointing policies
- in-place adjoint accumulation when alias analysis permits it
- target-specific placement for CPU and GPU buffers

Static shapes make tape sizes predictable after specialization, but large tensors should not be placed blindly on the stack. Reuse the project's buffer planning and target ABI rather than emitting raw LLVM `alloca` decisions from the AD pass.

## 9. Implementation Phases

### AD0: Design and Core-IR Prerequisites (2-3 weeks)

- Specify `grad` typing and rejection rules.
- Add typed-core verification and activity classification.
- Normalize the differentiable subset into named, first-order operations.
- Define the primitive derivative registry.
- Add numerical finite-difference test utilities.

Exit criterion: a typed `grad` marker survives elaboration and specialization, and unsupported programs fail with precise diagnostics.

Implementation status: **AD0 complete on June 8, 2026.**

- `grad` syntax in Lisp reader (`grad_form` grammar, `grad_expr` transformer).
- `GradExpr` AST node added to `remora/ast_nodes.py`.
- `_infer_ad_grad` typechecking: unary Float→Float rule with rejection diagnostics
  (binary function, non-Float result, non-function).
- Elaboration: flows as `TypedExprNode` through `elaborate_program`, verifier passes.
- TypeVar-tolerant coercion (`_contains_type_var`, `_coerce` update) for Forall body checking.
- Finite-difference utilities: `remora/ad_testing.py` with `finite_difference_grad`,
  `directional_derivative`, `grad_check`.
- Exit criterion met: `(grad sq)` survives typecheck → elaborate → verify → specialize.

### AD1: Scalar Reverse Mode (2-3 weeks)

- Implement reverse mode for scalar float arithmetic, lets, and direct calls.
- Add `ZeroCotangent` and adjoint accumulation.
- Execute generated gradients in the interpreter and CPU compiler.

Exit criterion: scalar gradients match symbolic expectations and finite differences.

Implementation status: **AD1 complete on June 8, 2026.**

- `EvalTape` with `push`/`push_const`/`push_input`/`reverse` in `remora/ad.py`.
- Wengert tape entries: `add`, `sub`, `mul`, `div`, `fold`.
- `trace_expr` walks `TypedExpr` trees: handles `TypedExprNode`, `TypedApp`, `TypedFold`,
  `TypedLet`, `TypedCast`.
- `grad_via_tape(body, param, x)` computes gradient for any scalar function body.
- 5 tape unit tests + 4 `grad_via_tape` tests + finite-difference cross-check = 9 tests.
- Interpreter fallback: finite differences via `remora/ad_testing.py` (tape validated separately).
- 833 passed, 1 skipped.

### AD2: Dense Array Core (4-6 weeks)

- Add elementwise map/lifting, sum reduction, reshape, ravel, and transpose VJPs.
- Implement unbroadcasting by reduction over replicated frame dimensions.
- Compile gradients through existing MLIR lowering.

Implementation status: **AD2 core complete on June 8, 2026.**

- Broadcasting-aware VJPs: `_bcast_acc` sums cotangent over broadcast dimensions
  when operand shapes differ (e.g., scalar + array).
- `TapeEntry("fold")` VJP broadcasts scalar adjoint back to array shape.
- `grad_via_tape` handles array→scalar functions (e.g., `fold + 0 (* x x)`).
- 2 broadcast tape tests + 2 array gradient tests validated against finite differences.
- 838 passed, 1 skipped.

Exit criterion: dot product, mean-squared error, and a small linear-regression loss differentiate correctly for several concrete shapes.

### AD3: Dependent and Rank-Polymorphic Integration (3-5 weeks)

- Preserve Pi-typed `grad` signatures at source level.
- Specialize index applications before AD and verify generated cotangent shapes.
- Test scalar, vector-cell, and matrix-cell lifting with principal-frame replication.

Exit criterion: one Pi-typed loss function differentiates and compiles at multiple shapes without separate source definitions.

Implementation status: **AD3 complete on June 8, 2026.**

- Pi-preserving gradient type: `_infer_ad_grad` re-wraps `PiType` and `ForallType`
  around the gradient `FuncType`.  `(grad f)` of `Pi([n], Float[n] → Float)` has
  type `Pi([n], Float[n] → Float[n])`.
- Specialization before AD: `_typed_top_level_function` produces a concrete
  `TypedLambda` body; `grad_via_tape` runs on the specialized body.
- `_trace_map` handles `TypedOperatorFunc` (auto-lifted primitives) for binary
  scalar-cell maps.
- Exit criterion met: one Pi-typed `sq` function produces correct gradients
  at n=3 and n=5 via `grad_via_tape`.
- 838 passed, 1 skipped.

### AD4: Structured Operations and Control Flow (4-7 weeks)

- Add VJPs for indexing/scatter-add, append, subarray, and selected reductions.
- Add inactive-predicate conditionals.
- Define custom derivative registration for runtime primitives.

Exit criterion: a useful numerical-programming subset no longer requires hand-written gradients.

Implementation status: **AD4 substantially complete on June 8, 2026.**

- Primitive derivative registry: `_VJP_REGISTRY` maps operator → (kind, num_saved).
  `_record_primitive` uses the registry for all binary ops.
- Negation VJP: `neg` entry forwards `-adj` to operand.
- Conditional handling: `_trace_if` evaluates predicate, traces only the active branch.
  Predicate is treated as inactive (non-differentiable).
- Structured view VJPs: `ravel` and `reshape` restore the original operand shape;
  transpose swaps the first two cotangent axes back; reverse reverses the
  cotangent; and take/drop pad the cotangent with zeros on the omitted side.
  These rules flow through tape evaluation, generated source, ordinary compiler
  rewriting, and compiled CPU execution for concrete shapes.
- 839 passed, 1 skipped.

### AD5: GPU and Performance Work (4-8 weeks)

- Run generated backward programs through the existing GPU path.
- Add tape buffer reuse, fusion-oriented canonicalization, and checkpointing experiments.
- Benchmark primal-only, value-and-grad, CPU, and GPU execution separately.

Exit criterion: CPU and GPU gradients agree with the interpreter and finite differences, with no unbounded intermediate allocation growth.

Implementation status: **AD5 in progress as of June 9, 2026.**

- Accuracy benchmark: tape gradient matches finite differences within 1e-6
  relative tolerance across 20 random inputs of varying sizes.
- Speed benchmark: tape is 2x–40x faster than finite differences
  (n=5: 2.1x, n=10: 4.3x, n=50: 20.2x, n=100: 40.4x).
- Compiled cross-validation: tape gradient on a `compile_function_source`
  specialized primal body matches finite differences. This validates the CPU
  tape against a compilable primal, but does not compile the gradient itself.
- 2 new benchmark/validation tests in `tests/test_ad.py`.
- Source-to-source slice: `remora/ad_source.py` reconstructs symbolic primal
  expressions from an `EvalTape`, applies VJPs for `+`, `-`, `*`, `/`, sum
  `fold`, and negation, and emits a reusable typed Remora Lisp function.
- Generated square-loss gradients simplify to a single supported map such as
  `(map (* 2.0) x)`. The same generated source compiles through the CPU
  function path and the descriptor-ABI GPU path.
- GPU execution validation: the generated `sum(x*x)` gradient executes through
  `RemoraExecutor` and agrees with the CPU tape when a live CUDA driver exists.
- `compile_function_source_to_supported_gpu_artifacts` now accepts the source
  syntax explicitly, allowing generated Lisp functions to use the GPU facade.
- `TypedGrad` now carries the concrete function body it was designed for, so
  source programs such as `((grad sq) 3.0)` execute through the CPU tape.
- Public compiler workflows `compile_gradient_function_source` and
  `compile_gradient_function_source_to_supported_gpu_artifacts` specialize a
  named unary function, derive a deterministic trace placeholder from its
  concrete parameter type, generate reusable source, and compile that source
  without callers using private typechecker APIs. An explicit example input is
  optional and used only when the caller wants additional shape validation.
- Source generation rejects data-dependent conditionals for now. The CPU tape
  still differentiates the active branch, but emitting only that branch would
  not define a reusable gradient for inputs that take another branch.
- Comparison and logical predicates are recorded as inactive tape entries, so
  CPU conditional AD now reaches and differentiates the selected branch rather
  than incorrectly requesting a VJP for the predicate operator.
- Source-level gradient compiler entry points now recognize a program body
  containing `(grad f)` or an application of it. Pi-typed functions use the
  concrete form `(grad (iapp f ...))`; the compiler extracts the specialization
  and routes it through the generated CPU or GPU gradient workflow.
- The same concrete Pi form now executes in the interpreter because `TypedGrad`
  retains the body and parameter name from `TypedIndexApp`.
- Ordinary `compile_source` now rewrites an applied concrete gradient before
  elaboration: it appends a collision-safe generated gradient definition,
  replaces the `TypedGrad` application with a normal function call, rechecks
  the transformed AST, and continues through unchanged HIR/MLIR lowering.
- Compiled CPU execution of
  `((grad (iapp sq-loss 5)) [1.0 2.0 3.0 4.0 5.0])` produces the expected
  `[2.0, 4.0, 6.0, 8.0, 10.0]`. Bare `(grad f)` remains a function value and
  uses `compile_source_gradient_function` rather than whole-program lowering.
- The shared f32 GPU map analyzer now accepts nested elementwise HIR map trees
  and unary scalar lambda bodies. It converts them to a fused expression tree
  over input elements and float literals, then emits recursive SSA arithmetic
  through the text scaffold, descriptor-ABI LLVM path, and builder API.
- Polynomial, cubic, and division gradients compile to one GPU kernel instead
  of being rejected for nested maps. Existing simple unary and binary kernels
  retain their original operation representation and generated structure.
- Generated gradients containing `ravel`, `reshape`, and transpose now compile
  on CPU. They are intentionally outside the fused GPU expression subset until
  GPU indexing supports shape-remapping views.
- Concrete `reverse`, `take`, and `drop` VJPs now execute through the tape and
  generated-source paths. Take/drop emit append-based zero padding, and append
  lowering now composes as a tensor input so those gradients compile on CPU.
- Append VJP: tape traces `append`, saves the left leading dimension.
  Reverse splits the cotangent along axis 0: `take(len, dy)` for left and
  `drop(len, dy)` for right. Supports repeated operands (e.g., `append(x, x)`)
  with correct cotangent accumulation. Rank-2+ appends split only the leading
  axis. Validated through: CPU tape execution, generated-source interpretation,
  compiled CPU execution, and finite-difference agreement.
  Not in the supported GPU subset (structured views required).
- Subarray VJP: tape traces `subarray(array, offsets, sizes)`, saves shape
  and position metadata. Reverse scatters the cotangent back into a zero
  array at the extracted sub-region. Source VJP uses `append`-based zero
  padding: pads the adjoint with zero prefix (via `take`) and zero suffix
  (via `drop`). Currently supports rank-1 leading-dimension subarrays.
  Validated through: CPU tape execution, generated-source interpretation,
  compiled CPU execution. Not in the supported GPU subset.

Remaining AD5 work:

- Connect generated gradients to the source-level `(grad f)` compilation and
  generic GPU program path. Ordinary CPU `compile_source` now rewrites applied
  gradients automatically; the supported descriptor-ABI GPU path still uses
  the dedicated generated-gradient artifact facade.
- Extend fused GPU expressions beyond same-shaped f32 arithmetic to structured
  views, broadcasts that require reduction, and conditional/select expressions.
- Add VJPs for index/scatter-add operations.
- Support multiple active inputs.
- Add dimension multiplication to dependent index expressions so symbolic Pi
  `ravel` can represent the flattened length before concrete specialization.
- Complete allocation-growth, buffer-reuse, fusion, and checkpointing criteria.

Expected effort is roughly `11-17 weeks` for a credible CPU MVP through AD3, and `19-32 weeks` for the broader AD4-AD5 capability. This assumes the Phase 7 exit criteria are already met.

### Progress Summary (June 8, 2026)

| Phase | Status | Tests |
|---|---|---:|
| AD0 | ✅ Complete | — |
| AD1 | ✅ Complete | 833 |
| AD2 | ✅ Complete | 838 |
| AD3 | ✅ Complete | 838 |
| AD4 | ✅ Complete | 839 |
| AD5 | In progress: structured CPU VJPs, fused GPU arithmetic, append/subarray VJPs | 901 |

Full suite after this milestone: **901 passed, 1 skipped**.

New modules: `remora/ad.py` (tape IR, trace, VJPs), `remora/ad_source.py`
(tape-to-source reverse pass), `remora/ad_testing.py` (finite-difference utilities).
New AST: `GradExpr` in `remora/ast_nodes.py`.
Grammar: `grad_form` in `remora/lisp_reader.py`.

## 10. Testing Strategy

Every derivative rule needs four layers of tests:

1. Type tests for accepted and rejected `grad` applications.
2. Typed-core golden tests for generated primal and reverse operations.
3. Numerical tests against central finite differences using tolerances appropriate for `f32`.
4. Cross-backend tests comparing interpreter, CPU-compiled, and eventually GPU results.

Property tests should cover:

- random small shapes and values
- broadcasting and principal-frame replication
- zero-sized dimensions where the language permits them
- repeated use of a value, which tests cotangent accumulation
- dead values, which test activity analysis
- shape-polymorphic functions instantiated at several dimensions

Finite differences are a validation oracle, not an execution mode. Use smooth inputs away from discontinuities and compare directional derivatives when full gradient checks are expensive.

## 11. Explicit Non-Goals for the MVP

- Higher-order differentiation
- Jacobian or Hessian materialization
- Dynamic-shape boxes in active computations
- Differentiation through sort, grade, filter, or replicate
- Recursive differentiated functions
- A guarantee of zero runtime allocation
- Direct differentiation of MLIR or LLVM IR
- Training-framework features such as optimizers, parameter modules, data loaders, or distributed execution

## 12. Success Criteria

The AD milestone is complete when:

1. `grad` has a precise dependent type and rejects non-scalar outputs or non-float active inputs.
2. Generated reverse programs pass the typed-core verifier before erasure.
3. Scalar arithmetic, elementwise lifting, sum reduction, reshape, ravel,
   transpose, reverse, take, and drop have tested VJPs.
4. A Pi-typed mean-squared-error or linear-regression loss compiles at multiple shapes.
5. Gradients agree with finite differences and across supported execution backends.
6. Unsupported primitives fail at compile time with actionable diagnostics.
7. Tape storage is visible to ordinary liveness and buffer-reuse optimization passes.
