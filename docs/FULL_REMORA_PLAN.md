# Full Remora Implementation Plan

_Generated 2026-06-07 from Dense Core completion status._

**Starting point**: Dense Core is complete (530 tests, 47 acceptance cases, 25 example
programs, CPU+GPU, MLIR builder API). This plan describes the path from Dense
Core to a full Remora implementation with Lisp syntax.

---

## 1. What full Remora adds

Full Remora, as described in Shivers, Slepak, and Manolios's tutorial and
Slepak's dissertation, adds these capabilities beyond Dense Core:

| Feature | Dense Core | Full Remora |
|---|---|---|
| Syntax | ML-like (`def f x = ...`) | Lisp s-expressions `(define (f [x 0]) ...)` |
| Iteration | Explicit (`map f xs`) | Implicit: `(f xs)` auto-lifts |
| Ranks | Fixed by `map`/`fold` | Annotated on function parameters |
| Frame/cell split | Manual via `map` | Automatic from type signatures |
| Broadcasting | None | Principal-frame cell replication |
| Type system | Rank-monomorphic, shape-erased | Dependent (Pi, Sigma), shape-indexed |
| Type inference | Simple AST inference | Bidirectional + Presburger constraint solving |
| Boxes | N/A | Existential types for ragged/dynamic shapes |
| Reduce/scan family | `fold` only | 3 reduces, 8 scans, 2 folds, 2 traces |
| Reranking | Cell maps | `~(r1 r2) func` notation |
| Higher-order | Lambdas in `map`/`fold` | Arrays of functions, MIMD dispatch |

---

## 2. Phase 1: Lisp syntax reader (explicit-only)

**Goal**: Users write `(map (+ 2) xs)` and get the same Dense Core compiler.

**Scope**: Parser-only — no new semantics. All existing Dense Core forms are
expressible in Lisp syntax. The reader is a separate parser that produces the
same AST as the existing Lark parser, making this a purely syntactic layer.

**Syntax mapping**:

```lisp
; Literals
42                        → 42
3.14                      → 3.14
#t                        → true
[1 2 3]                   → [1, 2, 3]

; Let and if
(:: x 5 (+ x 1))          → let x = 5 in x + 1
(if (< 1 2) 10 20)        → if 1 < 2 then 10 else 20

; Arithmetic / comparison
(+ 1 2)                   → 1 + 2
(< x 5)                   → x < 5

; Map and fold
(map (+ 2) xs)             → map (+ 2) xs       ; operator section
(map (lambda (x) (* x 2)) xs)  →  map (\x -> x * 2) xs
(fold + 0 xs)             → fold (+) 0 xs

; Iota and views
(iota 5)                  → iota 5
(iota 2 3)                → iota 2 3
(reverse xs)              → reverse xs
(transpose m)             → transpose m
(reshape xs [2 2])        → reshape xs [2, 2]
(ravel m)                 → ravel m
(take 2 xs)               → take 2 xs
(drop 2 xs)               → drop 2 xs

; Indexing
(index xs 0)              → xs[0]
(index xs 0 1)            → xs[0, 1]

; Shape/rank
(shape xs)                → shape xs
(rank xs)                 → rank xs

; Function definitions
(define (double [x]) (* x 2))        → def double x = x * 2
(define (add [x y]) (+ x y))         → def add x y = x + y
(define xs [1 2 3])                  → def xs = [1, 2, 3]
```

**Implementation**:
- New `remora/lisp_reader.py` module with a Lark grammar for s-expressions
- Produces the same `Program` AST nodes as the existing parser
- Entry point: `parse_lisp(source) -> Program`
- CLI flag: `--syntax lisp` (default remains ML syntax for backward compat)

**Estimate**: 2–3 days.

---

## 3. Phase 2: Rank polymorphism

**Goal**: `(+ xs ys)` automatically works on any-rank arrays without an explicit
`map`. The compiler determines the frame/cell split from the function's type
and generates the loop nest.

This is the fundamental mechanism that distinguishes full Remora from Dense
Core. It has three sub-systems:

### 3.1 Rank-annotated function types

Every function carries cell ranks in its type signature. The type `(-> int int int)`
means "takes two scalar cells, returns a scalar." The type `(-> (Vec int) int)`
means "takes a vector cell, returns a scalar."

```
; Scalars: rank 0
(define (add [x 0] [y 0]) (+ x y))      ; type: (-> int int int)

; Vectors: rank 1
(define (vmag [v 1]) (sqrt (reduce + 0 (square v))))  ; type: (-> (Vec int) int)

; Matrices: rank 2
(define (m*m [a 2] [b 2]) (reduce + 0 (* a b)))      ; type: (-> (Mat int) (Mat int) (Mat int))
```

### 3.2 Frame/cell decomposition at application

When `(f arg)` is applied, if `arg` has rank `r` and `f` expects cells of rank
`c` (where `r >= c`):

1. The last `c` dimensions of `arg`'s shape become the **cell shape**
2. The remaining `r - c` dimensions become the **frame shape**
3. The function is applied independently to each cell in the frame
4. Results are collected back into the frame shape

```
(vmag [[1 2 2]      ; shape [2, 3], rank 2
       [2 3 6]])    ; vmag expects rank 1 cells
→ frame shape [2], cell shape [3]
→ apply vmag to each row (cell)
→ collect results into frame shape [2]
→ [3 7]
```

### 3.3 Principal-frame cell replication

When multiple arguments have different frame shapes, the longest frame is the
**principal frame**. Shorter frames are replicated into the missing dimensions.

```
(+ [10 20] [[8 1 3]      ; frame shapes: [2] and [2, 3]
            [5 0 9]])    ; principal frame: [2, 3]
→ vector frame [2] replicated to [2, 3]
→ add first element of vector to first row, second to second row
→ [[18 11 13], [25 20 29]]
```

### 3.4 Implementation approach

**Type system changes**:
- Extend `RemoraType` with rank-annotated function types `FuncType(params, result, cell_ranks)`
- Typechecker infers cell ranks from function definitions
- Typechecker determines frame/cell split at application sites

**HIR changes**:
- Replace `HIRMap`/`HIRFold` with general `HIRApply(func, args)` + `HIRReduce(op, init, array, dim)`
- `HIRApply` carries the frame shape and cell shape computed by the typechecker
- Frame/cell decomposition becomes a lowering pass, not a surface construct

**Lowering changes**:
- Frame/cell decomposition generates nested loops (the "explicit iteration" translation)
- Cell replication generates broadcasting loads
- Reduction over leading dimension maps to `linalg.generic` with reduction iterator

**Estimate**: 6–8 weeks for the core mechanism; 10–12 weeks with all edge cases.

---

## 4. Phase 3: The full reduce/scan/fold/trace family

**Goal**: Beyond `fold`, provide the 15 operators from full Remora.

| Family | Operators | Description |
|--------|-----------|-------------|
| Reduce | `reduce`, `reduce/zero`, `reduce/1` | Associative parallel reduction |
| Scan | `scan`, `scan/zero`, `scan/1`, `iscan`, `iscan/zero`, `iscan/1`, `escan`, `escan/zero` | Prefix-sum with interior/exterior variants |
| Fold | `fold`, `fold-right` | Serial accumulation (left or right) |
| Trace | `trace`, `trace-right` | Prefix-sum of serial folds |

Implementation approach:
- `reduce` → `linalg.generic` with reduction iterator (already handled by Dense Core fold)
- `scan` → `linalg.generic` with a careful implementation using `scf.for` or dedicated passes
- `fold` → same as current `fold`
- Zero variants → constant initializer; `1` variant → requires non-empty leading dimension

**Estimate**: 3–4 weeks.

---

## 5. Phase 4: Additional primitives

**Goal**: Add the remaining full-Remora primitives.

| Primitive | Description | Lowering approach |
|-----------|-------------|-------------------|
| `append` | Concatenate along leading axis | `tensor.insert_slice` or loop-based copy |
| `length` | Size of leading dimension | `tensor.dim` |
| `rotate` | Circular shift per-axis | `tensor.extract_slice` with modulo indexing |
| `indices-of` | Coordinate array for each position | `linalg.generic` with `linalg.index` |
| `with-shape` | Replicate to match shape | `linalg.generic` with broadcast |
| `filter` | Select subarrays by boolean mask | Requires dynamic output size → boxes (Phase 6) |
| `select` | Element-wise ternary | `arith.select` (already supported via `if` tensors) |
| `replicate` | Repeat items by count | Requires dynamic output size → boxes |
| `sort` / `grade` | Sort / permutation index | `linalg.sort` or custom implementation |
| `subarray` | Extract rectangular region | `tensor.extract_slice` |

**Estimate**: 4–5 weeks.

---

## 6. Phase 5: Reranking

**Goal**: The `~(r1 r2) func` notation for adjusting frame/cell splits.

Reranking is syntactic sugar for η-expansion with different cell ranks:

```lisp
(~(1 1) + v m)
; desugars to:
((lambda ([x 1] [y 1]) (+ x y)) v m)
```

This requires the rank polymorphism machinery from Phase 2 to be working.
Implementation is primarily in the parser → typechecker pipeline:
- Parser recognizes `~(...)` syntax
- Desugars to a lambda with annotated cell ranks
- Typechecker handles the λ's frame/cell split at the application site

**Estimate**: 1 week (mostly syntactic, depends on Phase 2).

---

## 7. Phase 6: Boxes and existential types

**Goal**: Support `(box expr)` for ragged arrays and runtime-dependent shapes.

Boxes require:
- **Existential types** (`(Σ (len) [int len])`): a type that says "there exists some dimension"
- **Box construction**: wrapping an array with its hidden dimensions
- **Unbox**: opening a box to access its contents, with the constraint that hidden dimension information cannot leak into the result type
- **Filter/replicate results**: these produce boxed arrays since the output size isn't known at compile time
- **iota with runtime shape**: `(iota1 n)` produces a boxed vector

This is a significant type system extension. Approach:
- Add `SigmaType` to the type system
- `box` wraps an `ir.Value` with metadata → represented at runtime
- `unbox` is a let-like form that opens the existential
- For GPU, boxes require device-side dynamic allocation or pre-allocated max-size buffers

**Estimate**: 6–8 weeks.

---

## 8. Phase 7: Dependent type system and inference

**Goal**: Full dependent types with Presburger shape reasoning.

This is the most ambitious phase — essentially implementing Slepak's
dissertation. The full type system has:

- **Index sorts**: `Dim` (natural numbers) and `Shape` (sequences of naturals)
- **Index language**: `+`, `++`, `Shp`, dimension literals, index variables
- **Pi types**: `(Π (len) (→ ([int len] [int len]) int))` — dot product over vectors of any length
- **Sigma types**: `(Σ (len) [int len])` — existentially quantified dimensions (boxes)
- **Forall types**: `(∀ (t) (→ ([t]) [t]))` — polymorphic over element types

Key implementation components:

### 8.1 Index language
- Parser for index expressions: `(shape 3 4)`, `(++ @s1 (shape 5))`, `(+ d 1)`
- Index normalizer: simplifies `(++ (shape 3) (shape 4))` → `(shape 3 4)`
- Equivalence checker: `(++ @s @t)` ∼ `(++ @t @s)` under the free monoid theory

### 8.2 Bidirectional type inference
- **Synthesis** (`Γ ⊢ e ⇒ τ`): infer type from term
- **Checking** (`Γ ⊢ e ⇐ τ`): check term against given type
- Application synthesis: decomposes argument shapes into frame/cell prefixes

### 8.3 Constraint solving
- Generates string equations over Presburger arithmetic
- Solves for unknown dimensions in frame/cell decomposition
- Handles the mixed-prefix fragment (free monoid on ℕ with concatenation and addition)

### 8.4 Type erasure
- Removes detailed type annotations from runtime representations
- Residual types characterize only what the dynamic semantics needs
- Enables the "partially erased" execution mode

### 8.5 Explicit iteration translation
- Translates rank-polymorphic function application into explicit nested loops
- Frame/cell decomposition → loop nest structure
- Cell replication → broadcasting
- Reductions → parallel reduce operations

This phase is a multi-person-year research effort. A practical first step would
be a **restricted dependent type system** that handles the common cases
(rank-0, rank-1, and rank-2 cells, prefix computation of frame shapes from
argument shapes) without the full Presburger constraint solver.

**Estimate**: 6–12 months for a practical restricted version; full dissertation-level implementation is 1–2+ person-years.

---

## 9. Phase 8: GPU completion

**Goal**: Every full-Remora form that compiles on CPU also compiles on GPU.

Starting from Dense Core GPU (maps + reductions), add:
- Multi-dimensional reductions on GPU (rank-parametric)
- GPU box support via pre-allocated buffers
- GPU scan operators via parallel prefix-sum algorithms
- GPU append / rotate / subarray via descriptor arithmetic
- GPU filter / replicate with dynamic output sizing

**Estimate**: 6–8 weeks spread across phases.

---

## 10. Milestone roadmap

```
Phase 1  ── Lisp reader (2-3 days)
    │
Phase 2  ── Rank polymorphism (10-12 weeks)
    │
    ├── Phase 3 ── Full reduce/scan family (3-4 weeks)
    │
    ├── Phase 4 ── Additional primitives (4-5 weeks)
    │
    ├── Phase 5 ── Reranking (1 week, depends on Phase 2)
    │
    ├── Phase 6 ── Boxes + existential types (6-8 weeks, depends on Phase 2)
    │
    └── Phase 7 ── Dependent types + inference (6-12+ months, depends on Phase 2)
         │
         └── Phase 8 ── Full GPU (6-8 weeks, depends on Phase 7)

Total calendar time (1 engineer):
  Phase 1:          0.5 weeks
  Phase 2:         12 weeks
  Phase 3:          4 weeks (can overlap with Phase 2 tail)
  Phase 4:          5 weeks (can overlap with Phase 3)
  Phase 5:          1 week
  Phase 6:          8 weeks (partial overlap with Phases 3-4)
  Phase 7:        6-12+ months (core of the project)
  Phase 8:          8 weeks (after Phase 7 stabilizes)

  Best case:      ~12 months to restricted dependent types
  Conservative:   ~18-24 months to full dissertation-level system
```

---

## 11. Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Rank polymorphism generates incorrect frame/cell splits | High | High | Extensive property tests against interpreter; keep explicit `map` as fallback |
| Dependent type inference is too complex to implement fully | High | High | Start with restricted subset (rank 0-2 cells, prefix shapes); defer full Presburger |
| Boxes require major compiler redesign | Medium | Medium | Implement as a runtime-only mechanism first (no compile-time shape tracking) |
| GPU scan/filter performance is poor | Medium | Medium | Use well-known parallel algorithms (Blelloch scan); fall back to CPU for small sizes |
| Lisp syntax adoption conflicts with existing ML syntax | Low | Low | Both syntaxes coexist; CLI flag selects |

---

## 12. Success criteria for full Remora

- [ ] 1. `uv run pytest` passes all Dense Core + new tests
- [ ] 2. Every form in the Remora tutorial draft has a passing test:
    - [ ] `(+ xs ys)` auto-lifts without explicit `map`
    - [ ] `(reduce + 0 xs)` works as a parallel reduction
    - [ ] `(vmag [[1 2 2] [2 3 6]])` produces `[3 7]`
    - [ ] `(define (mean [xs 1]) (/ (reduce + xs) (length xs)))` typechecks and runs
    - [ ] `(vector-convolve v w)` produces correct convolution
    - [ ] `(matrix-multiply a b)` produces correct result
    - [ ] `(filter (> nums 0) nums)` returns filtered array
    - [ ] `(~(1 1) + v m)` adds vector to matrix columns
- [ ] 3. Both ML and Lisp syntax work on all programs
- [ ] 4. CPU and GPU backends handle all supported forms
- [ ] 5. Type errors are clear and source-located (not backend crashes)
- [ ] 6. Performance on rank-polymorphic programs matches or exceeds equivalent explicit-map Dense Core programs


## 13. Progress tracker

### Dense Core (prerequisite)
- [x] Integer, float, boolean literals
- [x] Array literals with consistent rectangular shape
- [x] `let` bindings (scalar and tensor)
- [x] `if` over scalar booleans and boolean tensors
- [x] Primitive arithmetic, comparison, and boolean operators
- [x] `int` → `float` numeric promotion
- [x] `iota` with compile-time integer dimension
- [x] `shape` and `rank` static metadata operations
- [x] `reverse` over statically shaped arrays
- [x] Full-rank and static partial indexing
- [x] Dynamic indexing with runtime-computed expressions
- [x] `map` over statically known callables (unary and binary)
- [x] Scalar `fold` over statically known accumulator callables
- [x] Array-cell `fold` over static callables
- [x] Cell maps with index-based body expressions
- [x] Top-level value definitions
- [x] Top-level function definitions
- [x] View ops: `transpose`, `slice`, `reshape`, `ravel`, `take`, `drop`
- [x] Starter prelude: `add`, `sub`, `mul`, `div`, `neg`, `id`, `const`, `sum`, `product`, `scale`, `dot`, `max`, `min`, `abs`, `any`, `all`
- [x] Multicore CPU threading (`--cpu-threads`)
- [x] CPU vectorization (`--cpu-vectorize`)
- [x] GPU maps (f32, i32, bool, rank 1–10)
- [x] GPU scalar reductions (f32, i32, rank 1–10)
- [x] MLIR builder API (CPU scalar/tensor/views, simple GPU scaffold)
- [x] Golden MLIR fixtures regenerated
- [x] `lowering.py` split into focused modules
- [x] Operator dispatch centralized
- [x] Text-processing MLIR hacks removed
- [x] Rank-1 GPU special cases generalized
- [x] Acceptance manifest complete (47 cases, CPU + GPU)
- [x] Example regression tests (27 compile + 25 CPU exec + 8 GPU PTX)
- [x] Toolchain validator with CUDA device detection
- [x] User guide with syntax reference
- [x] Full Remora plan documented

### Phase 1: Lisp syntax reader (explicit-only)
- [x] `remora/lisp_reader.py` — Lark grammar for s-expressions
- [x] Desugar `(define (f [x]) body)` → `def f x = body`
- [x] Desugar `(:: x v body)` → `let x = v in body`
- [x] Desugar `(if cond then else)` → `if cond then then else`
- [x] Desugar `(lambda (x) body)` → `\x -> body`
- [x] Desugar `(map callable arg)` → `map callable arg`
- [x] Desugar `(fold callable init arg)` → `fold callable init arg`
- [x] Desugar operator application `(+ 1 2)` → `1 + 2`
- [x] Desugar square brackets `[a b c]` → `[a, b, c]`
- [x] Desugar views: `(reverse xs)`, `(transpose m)`, `(iota 5)`, etc.
- [x] Desugar indexing: `(index xs 0 1)` → `xs[0, 1]`
- [x] Desugar `(shape xs)`, `(rank xs)`
- [x] Add `--syntax lisp` CLI flag
- [x] Add Lisp-syntax to REPL via `:syntax lisp` command
- [x] Tests: all existing `.remora` files have Lisp equivalents
- [x] Tests: Lisp and ML syntax produce identical AST

### Phase 2: Rank polymorphism
- [x] `FuncType` extended with cell-rank annotations per parameter
- [x] Typechecker infers cell ranks from function definition body
- [x] Typechecker computes frame/cell split at application sites
- [ ] Typechecker determines principal frame from argument shapes
- [ ] Typechecker validates frame agreement (prefix-ordering check)
- [ ] Typechecker handles cell replication for shorter frames
- [ ] `HIRApply` replaces `HIRMap` for general application
- [ ] `HIRApply` carries frame shape and cell shape
- [ ] `HIRReduce` replaces `HIRFold` for leading-dimension reduction
- [ ] Lowering pass: frame/cell decomposition → nested loops
- [ ] Lowering pass: cell replication → broadcast loads
- [x] Scalar ops auto-lift: `(+ xs ys)` works without `map`
- [ ] Vector-cell ops auto-lift: `(vmag matrix)` works
- [ ] Matrix-cell ops auto-lift: `(m*m a b)` matrix multiply
- [ ] Functions of functions in function position (MIMD)
- [x] Tests: `(+ xs ys)` produces same result as `map (+) xs ys`
- [ ] Tests: `(vmag matrix)` produces correct per-row magnitudes
- [ ] Tests: principal-frame replication mirrors tutorial examples
- [ ] Tests: type errors for failed frame agreement are clear
- [ ] Property tests: rank-polymorphic programs match explicit-map equivalents

### Phase 3: Full reduce/scan/fold/trace family
- [ ] `reduce` — associative parallel reduction over leading dimension
- [ ] `reduce/zero` — with explicit zero value for empty arrays
- [ ] `reduce/1` — requires non-empty leading dimension
- [ ] `iscan` — interior inclusive scan (prefix-sum including element)
- [ ] `iscan/zero` — interior scan with zero
- [ ] `iscan/1` — interior scan, non-empty required
- [ ] `escan` — exterior exclusive scan (prefix-sum excluding element)
- [ ] `escan/zero` — exterior scan with zero
- [ ] `scan` — alias for `iscan`
- [ ] `scan/zero` — alias for `iscan/zero`
- [ ] `scan/1` — alias for `iscan/1`
- [ ] `fold-right` — right-to-left serial fold
- [ ] `trace` — serial prefix-sum (like `iscan` but serial)
- [ ] `trace-right` — right-to-left trace
- [ ] Tests: each operator on rank-1, rank-2, rank-3 inputs
- [ ] Tests: zero variants handle empty leading dimension
- [ ] Tests: `iscan` + on `[2 10 5]` → `[2 12 17]`
- [ ] Tests: `escan/zero` + 0 on `[2 10 5]` → `[0 2 12 17]`

### Phase 4: Additional primitives
- [ ] `append` — concatenate along leading axis (MLIR lowering)
- [ ] `length` — size of leading dimension (`tensor.dim`)
- [ ] `rotate` — circular shift with per-axis rotation vector
- [ ] `indices-of` — coordinate array for each position
- [ ] `with-shape` — replicate scalar/array to match target shape
- [ ] `subarray` — extract rectangular region by offset and shape
- [ ] `filter` — select subarrays by boolean mask (→ boxes)
- [ ] `select` — element-wise ternary (already via tensor `if`)
- [ ] `replicate` — repeat items by count vector (→ boxes)
- [ ] `sort` / `grade` — sort with stable comparison function
- [ ] `index-item` — index by scalar along leading dimension
- [ ] Tests: `(append m1 m2)` for vectors, matrices, 3D arrays
- [ ] Tests: `(rotate v [2])` produces correct circular shift
- [ ] Tests: `(indices-of m)` produces coordinate tensor
- [ ] Tests: `(filter (> nums 0) nums)` returns filtered result
- [ ] Tests: `(sort < [3 1 4 1])` → `[1 1 3 4]`

### Phase 5: Reranking
- [ ] Parser recognizes `~(r1 r2 ... rn) expr` syntax
- [ ] Desugaring: `~(r1 r2) f` → `(lambda ([x1 r1] [x2 r2]) (f x1 x2))`
- [ ] Reranked reduce: `(~(0 1) reduce + m)` sums rows instead of columns
- [ ] Reranked append: `(~(1 1) append m1 m2)` appends side-by-side
- [ ] Reranked map: `(~(2 1) f x)` adjusts frame/cell partition
- [ ] Tests: `(~(1 1) + v m)` adds vector to matrix columns
- [ ] Tests: `(~(0 1) reduce + matrix)` sums each row
- [ ] Tests: Reranking with no-op cell ranks is identity

### Phase 6: Boxes and existential types
- [ ] `SigmaType` — existential type in the type system
- [ ] `box` — wraps array with hidden dimension witnesses
- [ ] `boxes` — constructs arrays of boxes with per-box witnesses
- [ ] `unbox` — opens box, binds contents and witnesses, evaluates body
- [ ] `iota1` — produces boxed vector `(Σ (len) [int len])`
- [ ] `iota2` — produces boxed matrix
- [ ] `iota0` through `iota9` — rank-monomorphic boxed iota
- [ ] `filter` result is boxed (unknown count)
- [ ] `replicate` result is boxed
- [ ] Typechecker: witness information cannot leak into result type
- [ ] Typechecker: unbox body's result shape must not depend on witness
- [ ] GPU: box storage in pre-allocated device buffers
- [ ] Tests: `(iota1 5)` produces boxed `[0 1 2 3 4]`
- [ ] Tests: `(unbox (iota1 5) (v len) v)` returns unboxed vector
- [ ] Tests: `(filter (> nums 0) nums)` typechecks and runs
- [ ] Tests: ragged array construction with `boxes`
- [ ] Tests: `(define weekdays (boxes (len) [char len] [5] ...))`

### Phase 7: Dependent type system and inference
#### 7.1 Index language
- [ ] Index parser: `(shape d1 d2 ...)`, `(++ @s1 @s2)`, `(+ d1 d2)`
- [ ] Index normalizer: free monoid simplifications
- [ ] Index equivalence checker: `(++ @s @t) ∼ (++ @t @s)`
- [ ] Splicing-shape notation: `[d @s 5]` → `(++ (shape d) @s (shape 5))`
#### 7.2 Type system
- [ ] `PiType` — dependent product over dimension/shape indices
- [ ] `ForallType` — parametric polymorphism over element types
- [ ] Index-application (`i-app f dim`) and type-application (`t-app f type`)
- [ ] Array type: `(A t shape)` with shape-indexed dimensions
- [ ] Function type: `(→ (τ ...) τ)` with rank annotations
#### 7.3 Bidirectional type inference
- [ ] Synthesis judgment: `Γ ⊢ e ⇒ τ`
- [ ] Checking judgment: `Γ ⊢ e ⇐ τ`
- [ ] Application synthesis: frame/cell decomposition + constraint generation
- [ ] Type and index abstraction handling
#### 7.4 Constraint solver
- [ ] String equation generation over mixed-prefix fragment
- [ ] Presburger arithmetic solver for dimension constraints
- [ ] Free monoid unification for shape constraints
- [ ] Error reporting: "cannot determine frame/cell split for ..."
#### 7.5 Type erasure
- [ ] Erase dependent type annotations from runtime values
- [ ] Residual types for dynamic semantics
#### 7.6 Explicit iteration
- [ ] Translate rank-polymorphic applications to nested loops
- [ ] Reduce/scan to parallel reduce/scan operations
- [ ] Cell replication to broadcasting
- [ ] Tests: `dot-product` type: `(Π (len) (→ ([int len] [int len]) int))`
- [ ] Tests: `append` type: `(Π (da db @rest) (∀ (t) (→ ([t da @rest] [t db @rest]) [t (+ da db) @rest])))`
- [ ] Tests: `reduce` type with @item-pad and @cell-shape parameters
- [ ] Tests: type inference resolves unknown dimensions from argument shapes

### Phase 8: Full GPU
- [ ] GPU multi-dimensional reductions (rank 2–10)
- [ ] GPU scan operators (parallel prefix-sum in shmem)
- [ ] GPU box support (pre-allocated device buffers)
- [ ] GPU append / rotate via descriptor arithmetic
- [ ] GPU subarray via strided descriptor slicing
- [ ] GPU filter / replicate (dynamic output, pre-allocated max-size)
- [ ] GPU sort (device-side sorting)
- [ ] GPU MIMD dispatch (arrays of functions → indirect calls)
- [ ] Tests: every Phase 3–6 primitive has a GPU acceptance test
- [ ] Tests: `REMORA_TEST_GPU=1 uv run pytest` passes all GPU-gated tests
- [ ] Benchmarks: GPU performance within 2× of hand-tuned CUDA for common ops
