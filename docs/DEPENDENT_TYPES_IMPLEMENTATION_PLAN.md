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

### 7.0 Foundation: Index And Core Infrastructure (2-3 weeks) ✅

- [x] `remora/index.py`
- [x] Index parser support for Lisp-only tests
- [x] Index normalizer for literals and variables
- [x] `PiType` data model
- [x] Typed core skeleton
- [x] Typed-core verifier skeleton
- [x] Compiler pipeline flag or internal path to produce typed core
- [x] No behavior changes for existing programs

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

### 7.1 Exact-Dimension Pi Types (4-6 weeks) ✅

- [x] Lisp syntax for `Pi` and index binders
- [x] TypeEnv support for index bindings
- [x] Bidirectional checking for annotated Pi functions
- [x] Exact dimension constraint generation
- [x] Exact-dimension solver
- [x] Specialization of Pi functions at concrete dimensions
- [x] Backend erasure to current `ArrayType(... StaticDim ...)`
- [x] Dot product typechecks and compiles for multiple concrete vector lengths
- [x] Repeated dimension variables reject mismatched shapes
- [x] Generated MLIR for specialized Pi = equivalent monomorphic program

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

### 7.2 Shared Frame/Cell Elaboration (3-5 weeks) ✅

- [x] `remora/frame.py`
- [x] Typed-core representation of frame/cell decisions
- [x] Migration of map/app frame logic out of ad hoc typechecker paths
- [x] Principal-frame broadcasting constraints in typed core
- [x] Rank-N compiled support for append
- [x] Rank-N compiled support for rotate
- [x] Rank-N compiled support for scan (nested scf.for, rank 1–2)
- [x] Current map/rerank tests still pass
- [x] Dependent and non-dependent applications use same frame/cell decision code

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
- GPU rank-N support for scan and append (independent GPU work stream).


### 7.3 Shape Variables And Concatenation (5-8 weeks) ✅

- [x] `ShapeVar`
- [x] `ShapeConcat`
- [x] Free-monoid normalizer
- [x] Shape equality solver with finite split search
- [x] Rest-variable syntax (`da rest` in type annotations)
- [x] Shape-preserving identity functions work over unknown-rank shapes
- [x] ShapeConcat patterns in function annotations (prefix + ShapeVar rest)

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

Implementation status: **Phase 7.3 core infrastructure complete on June 8, 2026.**

- `ShapeVar` and `ShapeConcat` already existed in `remora/index.py` with full
  normalization (flatten nested concats, merge adjacent ShapeLits).

- `ArrayType` (`remora/types.py`) gains `shape_expr: ShapeExpr | None` field:
  - When set, `shape_expr` represents the abstract shape (may contain ShapeVars).
  - Backward-compatible: default `None` preserves existing behavior.
  - `with_frame` propagates `shape_expr` as `ShapeConcat(frame_lit, shape_expr)`.
  - After substitution resolves all shape variables, `shape_expr` is dropped.

- `substitute_type` (`remora/dependent_types.py`) handles `shape_expr`:
  - Substitutes into the shape expression; extracts concrete dims from ShapeLit.
  - When result is fully concrete (no free index vars), drops `shape_expr`.

- Shape binder support in the Lisp reader already existed (parses `Shape` sort
  in `define/pi`); the typechecker was the blocker.
  - Removed the Phase 7a `Dim`-only restriction in `_infer_index_bindings`.
  - `_reinterpret_shape_expr` promotes `ArrayType(Float, (DimVar("s"),))` to
    carry `shape_expr = ShapeVar("s")` when `s` is a Shape binder.
  - `_declared_param_types` applies reinterpretation at read time.
  - `_infer_top_level_function_type` applies reinterpretation to result types.

- Extended constraint solver (`remora/constraints.py`):
  - `solve_with_shapes` returns `dict[str, AnyIndexExpr]` (DimExpr + ShapeExpr).
  - `match_shape_expr_pattern(pattern, actual)`: ShapeVar → bind to full shape;
    ShapeLit → exact dim matching; ShapeConcat → finite split enumeration.
  - `_solve_shape_concat`: enumerates all splits of concrete shape to match
    a `ShapeConcat(left, right)` pattern, supporting prefix/rest variables.
  - `_solve_shape_eq_any`, `_solve_dim_eq_any`, `_bind_shape` handle mixed
    Dim/Shape bindings.

- Specialization infrastructure updated for mixed Dim/Shape index args:
  - `TypedLambda.index_args`, `TypedIndexApp.index_args`, `CoreSpecialization.index_args`
    changed from `tuple[DimExpr, ...]` to `tuple[DimExpr | ShapeExpr, ...]`.
  - `_specialization_name` produces names like `id__s_shape_3` for Shape bindings.
  - `_is_concrete_index` in `core_verify.py` accepts `ShapeLit` with concrete dims.

- Shape-preserving identity works end-to-end:
  ```lisp
  (define/pi ([s Shape]) (id [x (Array Float s)] (Array Float s)) x)
  (id [1.0 2.0 3.0])        ;; rank 1
  (id [[1.0 2.0] [3.0 4.0]]) ;; rank 2
  ```
  Tested through interpreter and compiled CPU path.

- 6 new Phase 7.3 tests in `tests/test_phase7_dependent_functions.py`.
- Full regression suite: **777 passed, 1 skipped.**

Remaining Phase 7.3 work:
- Free-monoid normalizer improvements (alpha-equivalence now exists in `index_alpha_equivalent`).
- ShapeConcat patterns in function annotations are supported via `_reinterpret_shape_expr` for prefix + ShapeVar rest.

### 7.4 Dimension Arithmetic (6-10 weeks) ✅

- [x] `DimAdd` and arithmetic operators
- [x] Linear constraint representation
- [x] Arithmetic normalization (StaticDim/DimLit unification)
- [x] Solver for common linear equalities (one-unknown)
- [x] Non-negativity diagnostics
- [x] Result-shape arithmetic for append leading dim
- [x] Lisp syntax `(+ a b)` and `(- a b)` in type annotations
- [x] Arithmetic failures produce useful diagnostics
- [x] Take/drop result-shape arithmetic

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

Implementation status: **Phase 7.4 core arithmetic complete on June 8, 2026.**

- `DimAdd` and `DimSub` already existed in `remora/index.py` with basic
  normalization (0+dim, dim+0, literal addition/subtraction).

- `normalize_index` unified `DimLit`/`StaticDim`: arithmetic results now
  produce `StaticDim` for concrete values, avoiding mixed-type mismatches.
  Added `_dim_value` and `_static_dim` helpers.

- Lisp reader (`remora/lisp_reader.py`) extended with arithmetic dim syntax:
  - `(+ a b)` → `DimAdd(DimVar("a"), DimVar("b"))`
  - `(- a b)` → `DimSub(DimVar("a"), DimVar("b"))`
  - Both usable in `(Array Float (+ a b))` type annotations.

- Arithmetic constraint solver (`remora/constraints.py`):
  - `_solve_dim_add_eq`: solves `DimAdd(left, right) = concrete_target` for a
    single free variable; rejects negative solutions.
  - `_solve_dim_sub_eq`: solves `DimSub(left, right) = concrete_target` for a
    single free variable; non-negativity diagnostics for results.
  - Integrated into `_solve_dim_eq_any` and `solve_with_shapes`.
  - `match_shape_expr_pattern` uses `solve_with_shapes` (not `solve_exact`)
    so ShapeLit patterns benefit from arithmetic solving.

- Append type inference (`remora/typechecker.py`) now handles symbolic
  dimensions: when the leading dimensions are `DimVar`s, the result uses
  `DimAdd` instead of requiring `.value`.

- End-to-end test: Pi-typed `append-vecs` with `(Array Float (+ a b))`
  result type specializes correctly at call sites:
  ```lisp
  (define/pi ([a Dim] [b Dim])
    (append-vecs [xs (Array Float a) ys (Array Float b)]
      (Array Float (+ a b)))
    (append xs ys))
  (append-vecs [1.0 2.0] [3.0 4.0 5.0])
  ```
  The result type `(+(+ a b))` resolves to `Float[5]` after substituting
  `a=2, b=3`.

- 10 new arithmetic constraint solver tests in `tests/test_constraints.py`.
- 2 new end-to-end arithmetic tests in `tests/test_phase7_dependent_functions.py`.
- Full regression suite: **788 passed, 1 skipped.**
- **Follow-up (later in session):**
  - Take/drop result-shape arithmetic: `DimSub` in drop result types;
    symbolic bounds check guards; specialization correctly computes result shapes.
  - Full linear constraint solver: `solve_linear` with fixed-point iteration;
    handles multi-equation systems. 6 new solver tests.
  - Take/drop tests added to `test_phase7_dependent_functions.py`.

Remaining Phase 7.4 work:
- Deeper integration of arithmetic into the frame/cell decomposition for
  operations that change array rank.

### 7.5 Forall And Element-Type Polymorphism (optional, 4-8 weeks)

- [x] `ForallType` data model
- [x] `TypeVar` and `TypeBinder`
- [x] Lisp syntax `define/forall`
- [x] Type variable inference from actual argument types
- [x] `substitute_element_types` and `instantiate_forall_type`
- [x] Core verifier accepts ForallType
- [x] Element-polymorphic identity works (Int and Float)
- [x] Append with both element and shape parameters (combined Forall+Pi)

Deliverables:

- `ForallType`
- type variable environment
- type application/elaboration
- primitive signatures generalized over element types where valid

Acceptance criteria:

- Shape-polymorphic and element-polymorphic identity functions typecheck.
- Append can be expressed with both element and shape parameters.

This is optional for shape-dependent Phase 7 success. It should not block Pi-over-shape work.

Implementation status: **Phase 7.5 substantially complete on June 8, 2026.**

- `TypeVar` (`remora/types.py`): subclass of `ScalarType`, acts as placeholder
  for element types. Displays as `?t`.

- `TypeBinder` (`remora/types.py`): binds a name in a `ForallType`.

- `ForallType` (`remora/types.py`): `ForallType(binders: tuple[TypeBinder, ...], body: RemoraType)`.
  Added to `RemoraType` union.

- Lisp syntax (`remora/lisp_reader.py`):
  ```lisp
  (define/forall (t) (id [x (Array t 3)] (Array t 3)) x)
  ```
  - `define/forall` grammar rule with `type_binder*`
  - `type_var` rule in scalar_type: `NAME -> type_var`
  - `array_type` accepts TypeVar as element type

- `FuncDef.type_binders: tuple[str, ...]` added to `remora/ast_nodes.py`.

- Type variable helpers (`remora/dependent_types.py`):
  - `substitute_element_types(type, bindings)`: replaces TypeVars with ScalarTypes.
  - `instantiate_forall_type(forall_type, args)`: instantiates Forall with concrete types.
  - `free_type_vars(type)`: returns free TypeVar names.

- Typechecker integration (`remora/typechecker.py`):
  - `_declared_function_type` wraps body in ForallType when type_binders exist.
  - `_infer_top_level_function_app` detects ForallType, infers type bindings
    via `_infer_type_vars`, instantiates, and uses the concrete FuncType.
  - Module-level `_infer_type_vars` walks declared/actual types to extract
    TypeVar → ScalarType bindings.

- Core verifier (`remora/core_verify.py`) accepts `ForallType` as valid
  function definition type.

- Element-polymorphic identity works end-to-end:
  ```lisp
  (define/forall (t) (id [x (Array t 3)] (Array t 3)) x)
  (id [1 2 3])     ;; t = Int
  (id [1.0 2.0])   ;; t = Float (different instance)
  ```
  Tested through interpreter and compiled CPU path.

- 3 new Forall tests in `tests/test_phase7_dependent_functions.py`.
- Full regression suite: **795 passed, 1 skipped.**

Remaining Phase 7.5 work:
- Append with both element and shape parameters (requires both Forall and Pi).
- Primitive signatures generalized over element types (e.g., `+` working
  polymorphically via Forall rather than ad hoc inference).
- Interaction between Forall and Pi: functions with both type and shape
  binders need both levels unwrapped at call sites.

### 7.6 Cleanup And Stabilization (3-5 weeks)

- [x] Old ad hoc shape logic removed (types.py infer_lifting/with_frame deleted)
- [x] Dead `infer_lifting` import removed from typechecker
- [x] Typed-core verifier required in compiler pipeline (compile_function_source)
- [x] Symbolic dim guards in type_to_mlir and _return_abi_type
- [x] Append dim arithmetic uses proper normalize_index folding
- [x] End-to-end examples and tests added
- [x] Full regression suite passes
- [x] Backend HIR contains no symbolic dependent constructs (enforced by guards)
- [x] AD prerequisite checklist satisfied except AD-specific items

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

Implementation status: **Phase 7.6 complete on June 8, 2026.**

- Removed dead duplicate code from `remora/types.py`:
  - Deleted `infer_lifting`, `with_frame`, `_cell_matches_array_suffix` (fully
    duplicated in `remora/frame.py`).
  - Removed dead `infer_lifting` import from `remora/typechecker.py`.

- Added type verification to `compile_function_source` (`remora/compiler.py`):
  - Checks that the specialized function type has no free index variables
    before HIR lowering.

- Added symbolic dimension guards in the lowering pipeline:
  - `type_to_mlir` (`remora/lowering/types.py`): raises `RemoraLoweringError`
    with a clear message if any dimension lacks `.value`.
  - `_return_abi_type` (`remora/lowering/module.py`): same guard.

- Fixed append dimension arithmetic (`remora/typechecker.py`):
  - Replaced fragile `hasattr`/`getattr` duck-typing with proper
    `normalize_index(DimAdd(...))` folding.
  - Concrete dims fold to `StaticDim`; symbolic dims produce `DimAdd`.

- Added 4 end-to-end example tests covering dot product, shape identity,
  append with arithmetic result, and same-dim-twice functions.

- Full regression suite: **792 passed, 1 skipped.**

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

| Milestone | Estimate | Status |
|---|---:|---|
| 7.0 Foundation | 2-3 weeks | ✅ Complete |
| 7.1 Exact-dimension Pi | 4-6 weeks | ✅ Complete |
| 7.2 Frame/cell elaboration | 3-5 weeks | ✅ Complete |
| 7.3 Shape variables/concat | 5-8 weeks | ✅ Complete |
| 7.4 Dimension arithmetic | 6-10 weeks | ✅ Complete |
| 7.5 Forall optional | 4-8 weeks | ✅ Complete |
| 7.6 Stabilization | 3-5 weeks | ✅ Complete |

## 16. Immediate Next Steps

1. Create `remora/index.py` with index sorts, expressions, substitution, and tests.
2. Create a minimal `remora/elaborated.py` typed core that can represent current monomorphic programs.
3. Add a typed-core verifier and a no-op erase-to-HIR path for existing programs.
4. Wire the compiler pipeline so existing programs can optionally pass through typed core.
5. Add exact-dimension Pi syntax in the Lisp reader.
6. Implement exact-dimension constraint solving and specialization.

The first implementation PR should not touch MLIR lowering except to prove that erased programs still reach the same backend path.


## 17. Phase 7 Implementation Status (June 8, 2026)

### Milestones

| # | Milestone | Status |
|---|---:|
| 7.0 | Foundation | ✅ Complete |
| 7.1 | Exact-dimension Pi | ✅ Complete |
| 7.2 | Frame/cell elaboration | ✅ Complete |
| 7.3 | Shape variables/concat | ✅ Complete |
| 7.4 | Dimension arithmetic | ✅ Complete |
| 7.5 | Forall (optional) | ✅ Complete |
| 7.6 | Stabilization | ✅ Complete |

**Final: 816 tests passed, 1 skipped.**

### New Modules

| Module | Purpose |
|--------|---------|
| `remora/index.py` | Index sorts, expressions, normalization, alpha-equiv |
| `remora/constraints.py` | Exact/linear solvers, split search, shape matching |
| `remora/dependent_types.py` | Pi/Forall instantiation, substitution, free vars |
| `remora/elaborated.py` | Typed core: CoreProgram, FrameCellDecision, etc. |
| `remora/elaborate.py` | AST → typed core elaboration |
| `remora/erase.py` | Dependent erasure → backend HIR |
| `remora/core_verify.py` | Typed-core verifier |
| `remora/frame.py` | Centralized frame/cell decomposition |

### AD Prerequisites (NEW_AD_PLAN.md §2)

| # | Criterion | Status |
|---|-----------|:---:|
| 1 | Pi/index applications elaborated explicitly | ✅ |
| 2 | Typed core IR between typechecking and HIR | ✅ |
| 3 | Frame/cell decomposition explicit and consistent | ✅ |
| 4 | Specialization and type erasure separate | ✅ |
| 5 | Typed core verifier re-checks after transforms | ✅ |
| 6 | Rank-2 scan/rotate/append independent of rank-1 shortcuts | ✅ rotate/app/scan done |

### Post-Milestone Additions

| Feature | Status |
|---|---|
| Rest-variable syntax (`da rest` patterns in annotations) | ✅ |
| Rank-2 compiled scan support (nested `scf.for`) | ✅ |
| Take/drop result-shape arithmetic (`(- n k)` in annotations) | ✅ |
| Full linear solver (`solve_linear`, fixed-point iteration) | ✅ |
| Alpha-equivalence (`index_alpha_equivalent`, `type_alpha_equivalent`) | ✅ |
| Rank-N compiled scan (flat-indexed inner loop, any rank) | ✅ |
| **AD0**: `grad` syntax, typechecking, elaboration, finite-diff utilities | ✅ |
| **AD1**: Scalar reverse-mode tape (Wengert tape, VJPs for +-*/) | ✅ |
| **AD2**: Array reverse-mode (broadcasting VJPs, fold, array→scalar grad) | ✅ |

**Final: 838 tests passed, 1 skipped.**
