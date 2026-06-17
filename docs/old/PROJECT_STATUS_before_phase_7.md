# Project Status Before Phase 7: Dependent Types

_Generated 2026-06-08. 690 tests, 0 failures._

---

## 1. Executive Summary

The Remora compiler has completed Phases 1–6 of the [implementation plan](FULL_REMORA_PLAN.md) plus major portions of Phase 8 (GPU). The compiler supports Lisp and ML syntax, full rank polymorphism, the complete reduce/scan/fold/trace family, 12 additional primitives, reranking, boxes/existential types, and GPU execution for core operations. The **only remaining major feature** is Phase 7: a dependent type system with Presburger shape reasoning.

---

## 2. Completed Work by Phase

### Phase 1 — Lisp Syntax (16/16 ✓)

Two syntaxes coexist: ML (`.remora`) and Lisp (`.lisp`). The Lisp reader produces the same AST as the ML parser. All phases support both syntaxes.

- `(map (+ 2) xs)`, `(fold + 0 xs)`, `(sort < xs)`, `(filter (> 0) xs)`, `(with-shape 5 [3 2])`
- Operator sections: `(> 0)` → left section, `(< 10)` → right section
- `(define (fn [x INT]) ...)` with optional `NAME TYPE` rank annotations

### Phase 2 — Rank Polymorphism (18/20 ✓)

Auto-lifting, broadcasting, principal-frame, and vector-cell semantics. `(+ xs ys)` works on any-rank arrays without explicit `map`.

- `HIRApply` and `HIRReduce` nodes created parallel to `HIRMap`/`HIRFold`
- Compiled + interpreter paths for auto-lifting and broadcasting
- **Deferred (2/20):** Matrix-cell ops auto-lift `(m*m a b)` — typechecking done, lowering needs nested-fold support. Functions of functions in function position (MIMD) — deferred to Phase 6+.

### Phase 3 — Scan/Fold/Reduce (14/14 ✓)

Complete family: iscan, escan, trace, trace-right, fold-right, reduce/zero, reduce/1, and all variants. Compiled + interpreter paths.

### Phase 4 — Primitives (12/12 ✓)

length, rotate, subarray, indices-of, with-shape, select, append, index-item, filter, replicate, sort, grade. All compile via the CPU path. Sort and grade use `qsort` via the C runtime (O(n log n)).

### Phase 5 — Reranking (8/8 ✓)

`~(r1 r2) expr` desugaring syntax fully supported.

### Phase 6 — Boxes and Existential Types (15/16 ✓)

- SigmaType, box, unbox, iota1–5, boxes, filter/replicate typechecking: all done
- Box/Unbox are type-erased at runtime (unwrapped in `type_to_mlir`)
- Filter and replicate now compile to native code via the C runtime with dynamic output sizing (count-at-position-0 convention)
- **Deferred (1/16):** Variable-size iota boxes `(iota1 n)` where `n` is a runtime variable. Static iota boxes (literal sizes) compile correctly.

### Phase 8 — GPU (8/14 done)

Maps, reductions, rotate, subarray, indices-of compile via IREE. Scan and append via descriptor-ABI kernels. Buffer caching via `GPUPtxContext` pool. GPU fold execution reduced from 9ms to 0.5ms (fixed linear Python scan→`np.nonzero` vector search).

- **GPU remaining (6/14):** box support, filter/replicate dynamic output, sort, MIMD dispatch, float reduction fix

### Infrastructure

- **C runtime (`remora_rt.c`)**: 30 functions — sort, grade, filter, replicate, scan, rotate, append — using flattened LLVM ABI. Compiled once and linked into every CPU `.so`.
- **MLIR lowering**: text-based pipeline with `_MLIRMainModuleBuilder` supporting external function declarations and `func.call`.
- **GPU executor**: `execute_program_from_ptx` + `GPUPtxContext` with reusable buffer pools.

---

## 3. Deferred Items

These items are not blockers for Phase 7 but represent known gaps in the current system:

### 3.1 Rank-2 support for scan, rotate, append

**What:** These operations only support rank-1 in the compiled path. The typechecker and interpreter handle arbitrary ranks.

**Why deferred:** Scan/rotate/append use Remora's frame/cell semantics, not simple per-row iteration. The per-row pattern used for sort/grade/filter/replicate doesn't apply because these operations treat higher-rank arrays as having cell types (e.g., scan over rank-2 treats each row as a single cell). Implementing rank-2 support requires understanding the Remora cell abstraction, which is a Phase 7 concern.

**Fix approach (Phase 7):** Once the frame/cell decomposition is implemented as part of the explicit iteration translation, scan/rotate/append at higher ranks become straightforward compositions of the rank-1 lowering over the cell dimensions.

### 3.2 Variable-size iota boxes

**What:** `(iota1 n)` where `n` is not a compile-time literal. Static iota boxes (e.g., `(iota1 5)`) compile correctly.

**Why deferred:** Requires `memref.alloc(%n)` with a runtime-dynamic size in the iota lowering. This needs the dynamic-sizing ABI work (same pattern solved for filter/replicate with count-at-position-0 convention) but applied to the iota operation. The MLIR for dynamic allocation exists and parses correctly through IREE; the blocker is the export path's handling of dynamic-sized tensors in the ciface wrapper.

**Fix approach:** Apply the same count-at-position-0 convention: compile the size expression to a value, allocate `n+1` elements, store `n` at position 0, create a dynamic subview of size `n+1`. The host slices via SigmaType detection.

### 3.3 Builder API gaps

**What:** 10+ HIR node types (scalar maps, cell maps, array-cell folds, indexing, box/unbox, filter, replicate, sort, grade, scan, fold-right, rotate, subarray, indices-of, with-shape, append) lack builder API equivalents.

**Impact:** Non-blocking. The text-based MLIR lowering handles all of these operations correctly. The builder API is a performance optimization (avoids string concatenation overhead), not a correctness issue.

### 3.4 GPU completion items

| Item | Status |
|---|---|
| Box support (pre-allocated device buffers) | Deferred |
| Filter/replicate dynamic output on GPU | Deferred |
| GPU sort (device-side sorting) | Deferred |
| MIMD dispatch (arrays of functions → indirect calls) | Deferred |
| Float reduction (IREE buffer type mismatch) | Requires IREE fix |

---

## 4. Current Test Coverage

| Module | Tests | Coverage |
|---|---|---|
| `test_properties.py` | Compiled-vs-interpreter for all primitives | Full |
| `test_execution.py` | CPU compiled execution for all Dense Core ops | Full |
| `test_typechecker.py` | Type checking for all Phases 1-6 | Full |
| `test_lisp_reader.py` | Lisp parser for all syntax | Full |
| `test_executor.py` | GPU executor round-trips | Full |
| `test_benchmarks.py` | CPU/GPU/interpreter performance | Basic |
| **Total** | **690** (+ 1 skipped) | |

---

## 5. Phase 7: Dependent Type System

### 5.1 What Phase 7 adds

Phase 7 implements the type system described in Slepak's dissertation:

| Concept | MLIR/Remora representation |
|---|---|
| **Index sorts** | `Dim` (natural numbers) and `Shape` (sequences of naturals) |
| **Index language** | `+`, `++`, `Shp`, dimension literals, index variables |
| **Pi types** | `(Π (len) (→ ([int len] [int len]) int))` — dot product parametric in length |
| **Sigma types** | `(Σ (len) [int len])` — existentially quantified dimensions (boxes generalize this) |
| **Forall types** | `(∀ (t) (→ ([t]) [t]))` — polymorphic over element types |
| **Application** | `i-app f dim` (index application) and `t-app f type` (type application) |

The key capability: a function like `dot-product` can be written once with type `(Π (len) (→ ([int len] [int len]) int))` and applied to vectors of any length, with the compiler proving shape compatibility through Presburger arithmetic.

### 5.2 Key components

#### 5.2.1 Index language

**Index expressions** describe array shapes: `(shape 3 4)`, `(++ @s1 (shape 5))`, `(+ d 1)`. The parser extends the existing Lark grammar for index terms.

**Index normalizer** simplifies index expressions using free monoid identities. For example, `(++ (shape 3) (shape 4))` normalizes to `(shape 3 4)`, and `(++ @s (shape 0))` normalizes to `@s`.

**Equivalence checker** determines when two index expressions describe the same shape. Under the free monoid theory, `(++ @s @t)` ∼ `(++ @t @s)` for shape variables but not for dimension variables.

#### 5.2.2 Type system

The core types:

```
PiType(name: str, sort: IndexSort, body: RemoraType)
SigmaType(name: str, body: RemoraType)  ← already partially implemented
ForallType(name: str, body: RemoraType)
ArrayType(elem: RemoraType, shape: IndexExpr)  ← shape becomes index expression
FuncType(params: list[RemoraType], result: RemoraType)
```

Array types change from `StaticDim` (compile-time integer) to `IndexExpr` (arbitrary index expression). This means array shapes can reference type variables, enabling parametric polymorphism over dimensions.

#### 5.2.3 Bidirectional type inference

Two inference modes:

- **Synthesis** (`Γ ⊢ e ⇒ τ`): given a term, infer its type. Used for variables, applications (when the function type is known), and literals.
- **Checking** (`Γ ⊢ e ⇐ τ`): given a term and an expected type, verify the term has that type. Used for function arguments, let-bound values, and explicit annotations.

Application synthesis is the critical piece: it decomposes argument shapes into frame/cell prefixes, generates Presburger constraints, and solves for the decomposition. If `f : (Π (d1 d2 @rest) (...))` is applied to an argument of shape `(shape 5 3 7)`, the solver finds `d1 = 5, d2 = 3, @rest = (shape 7)`.

#### 5.2.4 Constraint solver

The mixed-prefix fragment: a theory of the free monoid on ℕ (string equations with concatenation and natural number arithmetic). Constraints are generated during application synthesis and solved using:

- **Free monoid unification**: solves equations like `X ++ Y = A ++ B ++ C` by enumerating prefix splits
- **Presburger arithmetic**: solves linear equations over naturals for dimension constraints
- **Combined solving**: handles equations mixing concatenation and arithmetic, e.g., `d1 ++ d2 = 3 ++ 4` meaning `d1 = 3, d2 = 4`

#### 5.2.5 Type erasure

After type checking, dependent type annotations are erased from the runtime representation. The residual types characterize only what the dynamic semantics needs — element types and static dimensions. This enables the "partially erased" execution mode where most of the compilation pipeline is unchanged.

#### 5.2.6 Explicit iteration translation

Rank-polymorphic function applications are translated into explicit nested loops:

- Frame/cell decomposition determines the loop nest structure
- Cell replication generates broadcasting operations
- Reductions generate parallel reduce operations
- All of these map to existing HIR nodes (HIRMap, HIRFold, HIRReduce, HIRApply)

### 5.3 Implementation strategy: restricted subset first

A full dissertation-level implementation is 1-2+ person-years. The practical approach is a **restricted dependent type system** that handles the common cases:

**Phase 7a (8-10 weeks): Rank-parametric Pi types with literal dimensions**

- Pi types where index parameters appear only as dimension literals in array shapes
- No concatenation or shape-variable arithmetic in index expressions
- Constraint solving: simple prefix matching (no Presburger solver)
- This enables: `dot-product : (Π (len) (→ ([int len] [int len]) int))`

**Phase 7b (6-8 weeks): Shape variables and concatenation**

- Index expressions with `++`, `@s` variables, and `(shape d1 d2 ...)`
- Full index normalizer and equivalence checker
- Free monoid unification for constraint solving
- This enables: `append : (Π (da db @rest) (∀ (t) (→ ([t da @rest] [t db @rest]) [t (+ da db) @rest])))`

**Phase 7c (8-10 weeks): Presburger arithmetic and full constraint solving**

- Dimension arithmetic in index expressions: `(+ d1 d2)`
- Combined free monoid + Presburger constraint solver
- Full bidirectional inference with error reporting
- This enables: functions that compute shapes from dimension arithmetic

**Phase 7d (4-6 weeks): Type erasure and explicit iteration**

- Erase dependent type annotations for the compilation pipeline
- Extend HIR lowering with frame/cell decomposition
- Translate Pi-typed applications to nested HIRMap/HIRReduce
- Full integration with existing MLIR lowering

---

## 6. Detailed Implementation Plan: Phase 7a (Restricted Pi Types)

### 6.1 Changes to the AST and parser

**New AST nodes:**
```
IndexExpr -- base class for index expressions
  | DimLit(int)          -- dimension literal: 3
  | ShapeLit(tuple)      -- shape literal: (shape 3 4)
  | DimVar(str)          -- dimension variable: d
  | ShapeVar(str)        -- shape variable: @s
  | IndexConcat(l,r)     -- concatenation: (++ expr expr)
  | IndexAdd(l,r)        -- arithmetic (Phase 7c): (+ d1 d2)

PiExpr -- Pi-type expression in source
  binders: list[(str, IndexSort)]  -- (len) or (len d2 @rest)
  body: Expr                        -- the function type
```

**Type syntax (Lisp):**
```lisp
(define (dot-product [v1 v2])
  (:: [len Dim])
  ... body ...
```

The `::` form introduces index bindings before the body.

**Parser changes:** Extend `lisp_reader.py` grammar with index expression rules and the `::` binder form.

### 6.2 Changes to the typechecker

**New Remora types:**
```
PiType(name: str, sort: IndexSort, body: RemoraType)
ForallType(name: str, body: RemoraType)  -- deferred to Phase 7b
DimensionType  -- the type of dimension values
ShapeType      -- the type of shape values
```

**Type environment extensions:**
```
TypeEnv extends with:
  index_bindings: dict[str, IndexSort]  -- dimension/shape variables in scope
```

**Inference rules (synthesis):**
- `Γ ⊢ (:: [d Dim] e) ⇒ (Π (d) τ)` where `Γ, d:Dim ⊢ e ⇒ τ`
- `Γ ⊢ (f a) ⇒ τ[a/d]` where `Γ ⊢ f ⇒ (Π (d) τ)` and `Γ ⊢ a dim`

**Inference rules (checking):**
- `Γ ⊢ e ⇐ (Π (d) τ)` extends with `d:Dim` and checks body
- `Γ ⊢ [e1..en] ⇐ (A t (shape d1..dn))` checks each element against `t` and infers dims from literal lengths

**Constraint generation (simple prefix matching for Phase 7a):**
- Application `f a` where `f : (Π (d1 d2) (→ ([int d1 d2]) τ))` and `a` has shape `(shape 3 4)`: match prefix, bind `d1=3, d2=4`

### 6.3 Constraint solver (Phase 7a — simple prefix matching)

The restricted solver handles only the common case where all index parameters appear as explicit dimensions in array shapes, in the same order.

**Algorithm:**
1. Collect the dimension sequence from the argument's actual shape
2. Collect the dimension parameters from the function's Pi-type binders
3. Attempt a prefix match: first N unknown dims against first N actual dims
4. If match succeeds, return bindings; if fails, report "shape mismatch"

**No Presburger arithmetic, no concatenation, no free monoid unification** — pure prefix matching.

### 6.4 HIR changes

**New HIR nodes:**
```
HIRIndexExpr -- index expression in HIR
  | HIRDimLit(int)
  | HIRDimVar(str)
  | HIRShapeLit(tuple)

HIRPiType(name, sort, body)  -- Pi type wrapper (erased before MLIR)
HIRIApp(func, dim_expr)       -- index application (erased before MLIR)
```

**Defunctionalization:** Pi-typed callables are instantiated with concrete dimensions at call sites, producing monomorphic HIRFunction nodes with fixed shapes.

### 6.5 MLIR lowering

**Minimal changes.** Since Pi types and index applications are erased before MLIR lowering, the existing pipeline handles everything. The HIR already supports maps, folds, reduces, and scans over arbitrary ranks. The only change is that `ArrayType.shape` becomes `tuple[IndexExpr, ...]` instead of `tuple[StaticDim, ...]`, and shape lookup requires evaluating index expressions.

**Shape evaluation:** At HIR-to-MLIR time, all index variables are substituted with their concrete values (determined during constraint solving). The MLIR lowering sees only `StaticDim` shapes as before.

### 6.6 Test plan

1. **Unit tests for index language:** parser, normalizer (Phase 7a: trivial), equivalence
2. **Type inference tests:** synthesis for simple Pi types, checking for annotations
3. **End-to-end tests:**
   - `(define (double [x]) (* x 2))` — no Pi needed, basic function
   - `(define (dot [v INT]) (fold + 0 v))` — Pi over vector length
   - `dot-product` with Pi-typed function applied to vectors of different lengths
4. **Compilation tests:** verify compiled output matches interpreter for Pi-typed programs

### 6.7 Estimated effort

| Component | Effort |
|---|---|
| Index language parser | 3-5 days |
| PiType / ForallType classes | 2-3 days |
| Type environment extensions | 2-3 days |
| Bidirectional inference (restricted) | 5-7 days |
| Simple constraint solver (prefix matching) | 3-5 days |
| HIR extensions (Pi/IApp) | 2-3 days |
| Defunctionalization for Pi types | 5-7 days |
| Shape evaluation and MLIR integration | 3-5 days |
| Tests and debugging | 5-7 days |
| **Total Phase 7a** | **5-7 weeks** |

---

## 7. Detailed Implementation Plan: Phases 7b-7d

### Phase 7b: Shape variables and concatenation (6-8 weeks)

- Full index normalizer with free monoid simplifications
- Index equivalence checker
- Shape variable support in function types: `(append [xs ys]) : (Π (da db @rest) ...)`
- Free monoid unification solver: enumerates prefix splits for `@rest` variables
- Tests: append, concat, reshape with parametric shapes

### Phase 7c: Presburger arithmetic (8-10 weeks)

- Dimension arithmetic in index expressions: `(+ d1 d2)`, `(- d1 1)`
- Combined constraint solver: free monoid + Presburger arithmetic
- Full bidirectional inference with rank polymorphism
- Error reporting: "cannot determine frame/cell split for f applied to shape ..."
- Tests: computing result shapes from arithmetic operations

### Phase 7d: Type erasure and explicit iteration (4-6 weeks)

- Erase Pi/Forall/Sigma type wrappers before MLIR lowering
- Frame/cell decomposition → nested HIRMap/HIRReduce generation
- Integration with existing MLIR lowering pipeline
- End-to-end tests: programs with Pi-typed functions that compile to native code

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Constraint solver complexity | Medium | High | Start with prefix matching only (Phase 7a); iterate |
| Breaking existing type inference | Low | High | Extensive regression suite (690 tests); add Pi types incrementally |
| Presburger solver performance | Medium | Medium | Cache results; limit to small constraint sets |
| Interaction with rank polymorphism (Phase 2) | Medium | Medium | Reuse existing frame/cell decomposition from Phase 2 |
| Type erasure breaking MLIR lowering | Low | Medium | Erase before HIR→MLIR; test on existing programs first |

---

## 9. Success Criteria for Phase 7

1. **Phase 7a:** `dot-product` written with Pi type `(Π (len) (→ ([int len] [int len]) int))`, typechecked, and compiled to native code producing correct results for any concrete vector length.

2. **Phase 7b:** `append` with full parametric shape `(Π (da db @rest) (∀ (t) (→ ([t da @rest] [t db @rest]) [t (+ da db) @rest])))` typechecks and compiles.

3. **Phase 7c:** Constraint solver handles dimension arithmetic in shapes; functions can compute result shapes from input dimensions.

4. **Phase 7d:** Every Phase 1-6 test continues to pass; Pi-typed programs compile to the same MLIR structure as their monomorphic equivalents.

---

## 10. Summary of Current Deferred Items

These are not blockers for Phase 7 but should be addressed during or after:

| Item | Category | Effort | When |
|---|---|---|---|
| Rank-2 scan/rotate/append | Frame/cell semantics | 2-3 days each | During Phase 7d (uses frame/cell decomposition) |
| Variable-size iota boxes | Dynamic allocation | 1-2 days | Anytime (solved pattern from filter) |
| Builder API gaps | Performance optimization | 1 week | Post-Phase 7 |
| GPU completion items | Separate work stream | 4-6 weeks | Post-Phase 7 |
| Float reduction GPU fix | IREE bug | Unknown | Depends on IREE |

---

_This document should be reviewed by the team before starting Phase 7 implementation. The restricted Phase 7a approach (prefix matching only, 5-7 weeks) is the recommended starting point._
