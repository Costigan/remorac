# Dependent Types Implementation Plan

## 1. Decision

Phase 7 should be implemented as a dependent-type elaboration pipeline, not as a direct extension of the existing backend-oriented HIR.

The project is ready to start Phase 7 because the existing interpreter, typechecker, HIR lowering, CPU runtime, GPU path, and regression suite are stable enough to provide a baseline. The baseline confirmed before this plan was:

```text
690 passed, 1 skipped
```

However, the old Section 6 plan in `PROJECT_STATUS_before_phase_7.md` should not be followed literally. In particular, do not add symbolic `HIRPiType` or `HIRIApp` nodes to the current HIR. Current HIR is already close to backend lowering and should receive specialized, erased types. Dependent constructs belong in a new typed elaborated core.

## 2. Goals

Phase 7 adds shape-indexed types and dependent products over dimensions and shapes.

User-visible goals:

- Write functions once over dimension variables, such as vector dot product over any length.
- Typecheck rank-polymorphic and shape-parametric code with useful diagnostics.
- Specialize Pi-typed programs to concrete shapes before MLIR lowering.
- Preserve the existing compiler and interpreter behavior for all Phase 1-6 programs.
- Leave the compiler with a typed core suitable for future whole-program transforms, especially automatic differentiation.

Non-goals for the first implementation:

- Full dissertation-level Presburger solving.
- General dependent pattern matching.
- Runtime symbolic shapes in MLIR.
- Higher-order arrays of functions.
- Full Forall type polymorphism.
- User-visible AD.

## 3. Architecture

The target pipeline is:

```text
Source AST
  -> parser / Lisp reader
  -> dependent typechecking and elaboration
  -> typed core IR
       - Pi/index applications explicit
       - shape/index expressions normalized
       - frame/cell decomposition recorded
       - principal-frame broadcasting recorded
  -> specialization
       - concrete dimensions substituted at use sites
       - Pi applications erased
       - residual shapes made StaticDim-only for current backend
  -> backend HIR
       - HIRApply / HIRMap / HIRReduce / HIRScan / ...
       - no symbolic Pi or index application nodes
  -> defunctionalization
  -> MLIR lowering
  -> runtime execution
```

The important boundary is between typed core and backend HIR.

Typed core owns dependent constructs. Backend HIR owns executable array operations after all dependent choices are solved or specialized.

## 4. New Modules

Add new modules rather than expanding `typechecker.py` until it becomes unmaintainable.

Recommended files:

- `remora/index.py`: index sorts and index expression AST.
- `remora/index_normalize.py`: simplification and alpha-equivalence for index expressions.
- `remora/constraints.py`: constraint representation and solvers.
- `remora/dependent_types.py`: PiType, ForallType placeholder, type substitution helpers.
- `remora/elaborated.py`: typed core IR.
- `remora/elaborate.py`: source typed AST to typed core.
- `remora/specialize.py`: instantiate Pi-typed functions at concrete index arguments.
- `remora/erase.py`: erase dependent annotations into backend-compatible types.
- `remora/core_verify.py`: typed-core verifier.

Existing files to touch deliberately:

- `remora/types.py`
- `remora/typechecker.py`
- `remora/ast_nodes.py`
- `remora/lisp_reader.py`
- `remora/parser.py` if ML syntax gets type/index syntax
- `remora/compiler.py`
- `remora/hir.py`
- `remora/defunc.py`
- `remora/lowering/types.py`
- `tests/test_typechecker.py`
- `tests/test_hir.py`
- new `tests/test_index.py`
- new `tests/test_elaborate.py`
- new `tests/test_dependent_types.py`

## 5. Type And Index Model

### 5.1 Index Sorts

Implement two index sorts first:

```python
DimSort      # natural-number dimension
ShapeSort    # finite sequence of dimensions
```

Do not model dimensions as ordinary runtime `int` values. A dimension variable is a compile-time index variable. Runtime `Int` remains a scalar value type.

### 5.2 Index Expressions

Initial expression forms:

```python
DimLit(value: int)
DimVar(name: str)
ShapeLit(dims: tuple[IndexExpr, ...])
ShapeVar(name: str)
ShapeConcat(left: IndexExpr, right: IndexExpr)
DimAdd(left: IndexExpr, right: IndexExpr)
DimSub(left: IndexExpr, right: IndexExpr)  # later; guarded by non-negativity constraints
```

Phase 7a implements `DimLit`, `DimVar`, and fixed-rank shape literals containing dimensions. `ShapeVar`, `ShapeConcat`, and arithmetic are introduced in later milestones.

### 5.3 Types

Extend the type model to include:

```python
PiType(binders: tuple[IndexBinder, ...], body: RemoraType)
ForallType(...)  # placeholder only until element-type polymorphism is implemented
```

Update `ArrayType.shape` carefully. The current code assumes:

```python
ArrayType.shape: tuple[StaticDim, ...]
```

The migration should happen in two layers:

- In source/dependent typing, shapes may contain index expressions.
- In backend-erased types, shapes must be `tuple[StaticDim, ...]` until MLIR lowering supports dynamic tensors more generally.

Do not make every backend caller handle arbitrary symbolic shapes in one step.

## 6. Syntax

### 6.1 Lisp Syntax First

Implement dependent type syntax in the Lisp reader first. It is already closer to Remora's reference notation and avoids overloading the ML grammar too early.

Phase 7a syntax:

```lisp
(define/pi ([len Dim])
  (dot-product
    [xs (Array Float len)
     ys (Array Float len)]
    Float)
  (fold + 0.0 (* xs ys)))
```

This narrow form keeps dependent annotations separate from existing rank
annotations. The result type is mandatory so the elaborated `PiType` never
contains a placeholder result.

Later, allow explicit type aliases or standalone annotations when needed:

```lisp
(: dot-product
   (Pi ([len Dim])
     (-> (Array Float len)
         (Array Float len)
         Float)))
```

If the existing definition syntax cannot support this cleanly, add a narrow form for Phase 7 tests rather than redesigning all syntax.

### 6.2 ML Syntax Later

Defer ML syntax for Pi annotations until the Lisp path works. The ML parser can continue using monomorphic annotations during 7a.

## 7. Typed Core IR

The typed core should represent elaborated programs after typechecking but before backend lowering.

Minimum core nodes:

```python
CoreProgram
CoreFunction
CoreParam
CoreVar
CoreLit
CoreLet
CoreIf
CoreArray
CoreApply
CoreIndexApply
CoreLambda
CorePrimOp
CoreMapLikeApply
CoreReduce
CoreFold
CoreScan
CoreBox
CoreUnbox
```

The initial implementation can mirror existing typed AST nodes where practical. The important additions are:

- Core nodes carry fully elaborated types.
- Pi applications are explicit.
- Index substitutions are explicit.
- Frame/cell decomposition decisions are recorded once.
- Broadcasting/principal-frame decisions are recorded once.
- The verifier can re-check all type and shape annotations.

Backend HIR should be produced only after specialization and erasure.

## 8. Constraint System

### 8.1 Constraint Representation

Represent constraints explicitly even in Phase 7a:

```python
DimEq(left, right)
ShapeEq(left, right)
SortConstraint(expr, sort)
NonNegative(expr)
```

Keep source locations on constraints. Diagnostics need to point to the application or annotation that generated the failed obligation.

### 8.2 Phase 7a Solver

Phase 7a solver supports:

- equality between dimension variables and static dimensions
- repeated occurrences of a dimension variable
- exact matching of fixed-rank shape literals
- prefix extraction only where the cell rank is known

Required examples:

```text
Array Float len  applied to shape [5]      => len = 5
Array Float len  applied to shape [5, 2]   => fail unless cell/frame split allows it
Array Float m n  applied to shape [3, 4]   => m = 3, n = 4
Array Float len, Array Float len
  applied to [3], [4]                      => fail: len cannot be both 3 and 4
```

Do not call this full dependent inference. It is exact dimension elaboration.

### 8.3 Phase 7b Solver

Add:

- shape variables
- trailing rest variables
- shape concatenation
- free-monoid simplification
- finite split search for equations with concrete shapes

This enables useful types like shape-preserving identity over unknown rank and append with common suffix/rest shape.

### 8.4 Phase 7c Solver

Add:

- dimension arithmetic
- linear equality constraints
- non-negativity checks
- enough Presburger reasoning for append, take/drop, subarray, and reshape products if feasible

Do not implement a full general-purpose solver before the language has enough tests to justify it.

## 9. Frame/Cell Decomposition

Before or during 7a, extract a shared frame/cell module from the typechecker.

It should own:

- cell-rank validation
- array suffix matching
- frame extraction
- principal-frame selection
- broadcasting/replication obligations
- result type framing

Current logic is spread across map/app inference and lowering. Phase 7 should centralize it so dependent elaboration and future AD do not rediscover the same decisions.

Suggested module:

```python
remora/frame.py
```

Core API sketch:

```python
decompose_argument(actual: RemoraType, expected_cell: RemoraType) -> FrameCell
principal_frame(frames: list[ShapeExpr]) -> ShapeExpr
apply_frame(result_cell_type: RemoraType, frame: ShapeExpr) -> RemoraType
```

For Phase 7a, these functions can operate on fixed-rank shape literals. Later phases generalize them to shape variables and concatenation.

## 10. Specialization And Erasure

Specialization instantiates Pi-typed functions at concrete index bindings.

Initial strategy:

- Key specializations by function name plus concrete index arguments.
- Generate unique internal function names such as `dot_product__len_5`.
- Substitute index variables in parameter and result types.
- Produce backend-erased types with only `StaticDim` shapes.
- Reuse existing `lower_to_hir`, defunctionalization, and MLIR lowering once erased.

Do not force MLIR lowering to evaluate symbolic expressions. If a shape cannot be reduced to `StaticDim`, fail before backend HIR.

Erasure rules:

- `PiType` disappears after specialization.
- `CoreIndexApply` disappears after substitution.
- `DimVar` and `ShapeVar` must not appear in backend HIR types.
- `SigmaType` erasure remains compatible with the current boxed/dynamic output convention, but hidden dimension escape checks should move into typed-core verification.

## 11. Milestones

### 7.0 Foundation: Index And Core Infrastructure (2-3 weeks)

Deliverables:

- `remora/index.py`
- index parser support for Lisp-only tests
- index normalizer for literals and variables
- `PiType` data model
- typed core skeleton
- typed-core verifier skeleton
- compiler pipeline flag or internal path to produce typed core
- no behavior changes for existing programs

Acceptance criteria:

- Existing suite passes.
- Unit tests cover index AST equality, substitution, and simple normalization.
- A monomorphic program can round-trip through typed core and erase to existing HIR.

### 7.1 Exact-Dimension Pi Types (4-6 weeks)

Deliverables:

- Lisp syntax for `Pi` and index binders
- TypeEnv support for index bindings
- bidirectional checking for annotated Pi functions
- exact dimension constraint generation
- exact-dimension solver
- specialization of Pi functions at concrete dimensions
- backend erasure to current `ArrayType(... StaticDim ...)`

Acceptance criteria:

- Dot product with `len` typechecks and compiles for multiple concrete vector lengths.
- Repeated dimension variables reject mismatched shapes.
- Existing implicit rank-polymorphic programs behave unchanged.
- Generated MLIR for a specialized Pi program resembles the equivalent monomorphic program.

Estimate: 4-6 weeks after 7.0.

Implementation status: **Phase 7.1 complete on June 8, 2026.**

- 7.0 index expressions, substitution, normalization, typed-core checkpoint,
  verifier, compiler routing, and erasure are implemented.
- The exact fixed-rank dimension solver is implemented.
- Lisp `define/pi` supports `Dim` binders plus scalar and fixed-rank array
  parameter/result annotations.
- Direct calls infer concrete dimension arguments from parameter shapes, check
  the body against the specialized result type, and erase through existing HIR.
- Dot product executes through both the interpreter and CPU compiler.
- Explicit `iapp` now records concrete specialization evidence as
  `CoreIndexApplication`, supports dependent result shapes, and erases to HIR
  and MLIR identical to the equivalent monomorphic program.
- Dependent bodies are checked bidirectionally against their declared result
  under symbolic parameter types when the definition is processed.
- Concrete instances are cached and named deterministically, for example
  `dot__n_2`; inferred and explicit applications share the same specialization.
- The core records deduplicated `CoreSpecialization` entries and verifies that
  every explicit index application resolves to a matching concrete instance.
- `CoreProgram` is now structural: definitions, expressions, explicit index
  applications, and concrete specialization bodies are represented and
  verified independently of `TypedProgram`.
- Core-to-HIR erasure walks the structural core and rejects free index
  variables before backend lowering.
- `TypeEnv` has separate value and index namespaces.
- `compile_function_source` uses the same specialization path and exposes the
  deterministic specialization name and concrete index arguments.

Phase 7.1 intentionally ends at exact fixed-rank dimension polymorphism:

- only `Dim` binders are accepted by `define/pi`
- explicit `iapp` arguments must be non-negative dimension literals
- all dimensions must be concrete before HIR lowering
- cross-module specialization is not applicable yet because the language has
  no separate module/import system; standalone function compilation from a
  source unit is supported
- shape variables, shape concatenation, rest shapes, and dimension arithmetic
  begin in Phases 7.3 and 7.4

### 7.2 Shared Frame/Cell Elaboration (3-5 weeks)

Deliverables:

- `remora/frame.py`
- typed-core representation of frame/cell decisions
- migration of map/app frame logic out of ad hoc typechecker paths
- principal-frame broadcasting constraints in typed core
- rank-2 compiled support for scan, rotate, and append where feasible

Acceptance criteria:

- Current map/rerank tests still pass.
- Rank-2 scan/rotate/append no longer fail solely because lowering lacks frame/cell abstraction.
- Dependent and non-dependent applications use the same frame/cell decision code.

Estimate: 3-5 weeks.

Implementation status: **Phase 7.2 substantially complete on June 8, 2026.**

- `remora/frame.py` created as the single owner of frame/cell decomposition:
  - `scalar_cell_and_frame` – simple element+frame decomposition
  - `cell_matches_array_suffix` – suffix matching
  - `cell_type_candidates` – generates valid cell types for lifting
  - `decompose_argument` – central frame/cell decomposition from function type
  - `apply_frame` – result-type framing
  - `principal_frame` – principal-frame selection for broadcasting
  - `infer_lifting` – legacy-compatible wrapper
  - `validate_cell_rank` – cell-rank validation
  - `broadcasting_obligations` – binary broadcasting resolution
  - `FrameCell` record for downstream reuse

- `remora/typechecker.py` migrated to route through `remora/frame.py`:
  - `_scalar_cell_and_frame` delegates to `scalar_cell_and_frame`
  - `_cell_type_candidates` delegates to `cell_type_candidates`
  - `_principal_frame` delegates to `principal_frame`
  - Map and implicit-application inference calls `frame_infer_lifting` and `apply_frame`

- `FrameCellDecision` added to `remora/elaborated.py` typed core:
  - Records `frame_shape`, `cell_shape`, `cell_type`, `is_implicit`, `is_binary`
  - `CoreExpr` carries an optional `frame` field
  - `remora/elaborate.py` extracts frame decisions from `TypedMap` nodes
  - `remora/core_verify.py` verifies frame/cell shapes are concrete

- Rank-N append lowering (`remora/lowering/tensor_ops.py`):
  - Removed `rank != 1` guard
  - Generates N-D offsets, sizes, and strides for `tensor.insert_slice`
  - Tested at rank-2 through interpreter and compiled CPU path

- Rank-N rotate lowering (`remora/lowering/tensor_ops.py`):
  - Removed `rank != 1` guard
  - Generates N-D affine maps, iterator types, and multi-dimensional extracts
  - `linalg.index` collects all trailing dimension indices
  - Tested at rank-2 through interpreter and compiled CPU path

- Rank-N scan lowering: **investigated, deferred.**
  - The interpreter correctly handles scan at rank-2+.
  - Lowering requires array-typed carries (`tensor.extract_slice`/`tensor.insert_slice`)
    and nested linalg operations for carry updates, which is a larger effort.
  - The `rank != 1` guard remains with a clear diagnostic.

- GPU scan/append rank-1 guards (`remora/gpu_lowering.py`) remain unchanged;
  those paths are independent work streams.

- 28 unit tests for `remora/frame.py` in `tests/test_frame.py`.
- 4 integration tests for rank-2 append and rotate (correctness + compiled).
- Full regression suite: **771 passed, 1 skipped.**

Remaining Phase 7.2 work:
- Rank-N compiled scan support (requires cell-typed carry lowering).
- GPU rank-N support for scan and append (independent GPU work stream).


### 7.3 Shape Variables And Concatenation (5-8 weeks)

Deliverables:

- `ShapeVar`
- `ShapeConcat`
- free-monoid normalizer
- shape equality solver with finite split search
- rest-variable syntax
- tests for shape-preserving functions and common-rest constraints

Acceptance criteria:

- Identity-like and shape-preserving functions work over unknown-rank shapes.
- Common-rest constraints reject incompatible suffixes.
- Append can express same-rest typing, excluding leading-dimension arithmetic if arithmetic is not yet implemented.

Estimate: 5-8 weeks.

### 7.4 Dimension Arithmetic (6-10 weeks)

Deliverables:

- `DimAdd` and selected arithmetic operators
- linear constraint representation
- arithmetic normalization
- solver for common linear equalities
- non-negativity diagnostics
- result-shape arithmetic for append and take/drop

Acceptance criteria:

- Append type can express leading dimension `(+ da db)`.
- Take/drop/subarray result shapes can be checked where operands are statically known or constrained.
- Arithmetic failures produce useful diagnostics.

Estimate: 6-10 weeks.

### 7.5 Forall And Element-Type Polymorphism (optional, 4-8 weeks)

Deliverables:

- `ForallType`
- type variable environment
- type application/elaboration
- primitive signatures generalized over element types where valid

Acceptance criteria:

- Shape-polymorphic and element-polymorphic identity functions typecheck.
- Append can be expressed with both element and shape parameters.

This is optional for shape-dependent Phase 7 success. It should not block Pi-over-shape work.

### 7.6 Cleanup And Stabilization (3-5 weeks)

Deliverables:

- old ad hoc shape logic removed or quarantined
- typed-core verifier required in the compiler pipeline
- clear diagnostics for unsolved constraints
- documentation and examples
- all existing tests plus dependent tests passing

Acceptance criteria:

- Full regression suite passes.
- Dependent examples compile and run through interpreter and CPU compiled execution.
- Backend HIR contains no symbolic dependent constructs.
- AD prerequisite checklist in `NEW_AD_PLAN.md` is satisfied except AD-specific items.

## 12. Test Plan

Add focused tests at every layer.

Index tests:

- sort checking
- substitution
- alpha-renaming
- normalization
- equality and inequality cases

Type tests:

- simple Pi annotations
- index binder scope
- repeated dimension variables
- mismatch diagnostics
- hidden Sigma dimension non-escape
- interaction with lambdas and top-level functions

Elaboration tests:

- explicit CoreIndexApply generation
- frame/cell decisions recorded
- principal-frame broadcasting recorded
- typed-core verifier catches corrupted shapes

Specialization tests:

- unique specialized names
- cache/reuse repeated specializations
- no symbolic index variables after erasure
- monomorphic equivalent types after specialization

End-to-end tests:

- Pi-typed dot product at multiple lengths
- Pi-typed vector magnitude
- shape-preserving map
- append after arithmetic milestone
- reduce over Pi-typed vectors
- compiled-vs-interpreter comparisons

Regression:

- Run `env UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` before and after each milestone.

## 13. Diagnostics Requirements

Every constraint failure should include:

- source location
- expected shape/type
- actual shape/type
- unsolved variable or conflicting binding
- operation/application that generated the constraint

Example:

```text
shape mismatch in call to dot-product:
  len was inferred as 3 from argument xs
  but argument ys has leading dimension 4
```

Avoid generic errors such as `expected int[3], got int[4]` when the real issue is a failed dependent binding.

## 14. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Typechecker becomes unmaintainable | High | High | Add index/core/constraint modules; keep solver out of recursive inference paths |
| Solver scope expands too early | High | High | Ship exact-dimension Pi first; defer concat/arithmetic |
| Backend receives symbolic shapes | Medium | High | Require erasure verifier before `lower_to_hir` |
| Existing rank polymorphism regresses | Medium | High | Centralize frame/cell logic and run full suite every milestone |
| Error messages become opaque | Medium | Medium | Attach source locations to constraints from day one |
| Specialization causes code explosion | Medium | Medium | Cache by function plus index arguments; measure generated function count |
| Sigma/box semantics conflict with Pi | Medium | Medium | Move hidden-dimension escape checks into typed-core verifier |
| AD prerequisites get missed | Medium | Medium | Keep typed core stable and documented; satisfy `NEW_AD_PLAN.md` exit criteria |

## 15. Updated Estimate

A credible restricted Phase 7, through exact-dimension Pi types and a typed core, is about 8-12 weeks.

A useful user-visible dependent type implementation, including shape variables and arithmetic sufficient for append-like programs, is about 22-37 weeks.

Rough breakdown:

| Milestone | Estimate |
|---|---:|
| 7.0 Foundation | 2-3 weeks |
| 7.1 Exact-dimension Pi | 4-6 weeks |
| 7.2 Frame/cell elaboration | 3-5 weeks |
| 7.3 Shape variables/concat | 5-8 weeks |
| 7.4 Dimension arithmetic | 6-10 weeks |
| 7.5 Forall optional | 4-8 weeks |
| 7.6 Stabilization | 3-5 weeks |

## 16. Immediate Next Steps

1. Create `remora/index.py` with index sorts, expressions, substitution, and tests.
2. Create a minimal `remora/elaborated.py` typed core that can represent current monomorphic programs.
3. Add a typed-core verifier and a no-op erase-to-HIR path for existing programs.
4. Wire the compiler pipeline so existing programs can optionally pass through typed core.
5. Add exact-dimension Pi syntax in the Lisp reader.
6. Implement exact-dimension constraint solving and specialization.

The first implementation PR should not touch MLIR lowering except to prove that erased programs still reach the same backend path.
