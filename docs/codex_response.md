# Phase 7 Review

## Bottom line

Phase 7 is the right next milestone, but the current split is optimistic in two ways:

1. Phase 7a is under-scoped for user value and over-scoped for implementation risk.
2. The plan treats dependent types as a mostly isolated add-on, but the current compiler already has several ad hoc shape/rank special cases that Phase 7 will expose immediately.

My recommendation is to keep the goal, but reframe the first slice as an elaboration pass plus a minimal solver, not as a broad user feature release.

## 1. Phase 7a realism

The `5-7 weeks` estimate for Phase 7a is optimistic. I would budget closer to `8-12 weeks` for the first stable slice if you want it to coexist with the current compiler without turning the typechecker into a pile of special cases.

The main underestimate is not the parser or the Pi-type syntax. It is the interaction work:

- The current typechecker already has a lot of shape/rank search and coercion logic in one file, especially in the `map`/application paths and the rank-polymorphic candidate search in [`/e/projects/remorac/remora/typechecker.py`](/e/projects/remorac/remora/typechecker.py#L961) and [`/e/projects/remorac/remora/typechecker.py`](/e/projects/remorac/remora/typechecker.py#L991).
- Existing lowering already assumes concrete, erased shapes by the time it reaches MLIR, and rank-limited special cases are still common in [`/e/projects/remorac/remora/lowering/tensor_ops.py`](/e/projects/remorac/remora/lowering/tensor_ops.py#L1729) and [`/e/projects/remorac/remora/lowering/tensor_ops.py`](/e/projects/remorac/remora/lowering/tensor_ops.py#L1959).
- Phase 7 adds a new kind of compile-time object model: `Pi`, `Sigma`, `forall`, index substitution, and constraint solving. Even a small solver has a lot of edge cases around source locations, ambiguity reporting, and interaction with existing coercions.

What is undercounted:

- AST/parser changes for index syntax and type syntax.
- Elaboration and substitution machinery.
- Error reporting for unsolved or ambiguous applications.
- Regression tests for old code paths that currently rely on implicit rank inference.
- Refactoring time to keep the typechecker maintainable after adding a second inference mode.

My blunt take: if Phase 7a is supposed to be a real milestone, it should be “dot-product and other exact-length parametric kernels typecheck and compile reliably,” not “dependent types are now supported.”

## 2. Architecture

Resolving Pi applications before MLIR is the right direction. I would not lower index application directly to MLIR.

The better split is:

- Parse Pi/index syntax into a dedicated index AST.
- Elaborate or solve index application during typechecking or a dedicated elaboration pass.
- Preserve a typed, elaborated core IR after solving, with normalized shapes and explicit frame/cell decisions.
- Erase dependent annotations before MLIR so the backend only sees specialized `ArrayType` shapes and ordinary lowering IR.

That is already close to how the compiler behaves today: the current HIR treats `HIRApply` as a lowered rank-polymorphic form and keeps the backend focused on iteration and tensor ops in [`/e/projects/remorac/remora/hir.py`](/e/projects/remorac/remora/hir.py#L112) and [`/e/projects/remorac/remora/hir.py`](/e/projects/remorac/remora/hir.py#L453). However, Phase 7 should make the elaboration/erasure boundary explicit rather than collapsing directly from typechecking into backend-oriented HIR. A typed elaborated core will also be the correct future input for whole-program transformations such as automatic differentiation.

My recommendation is:

- Do index application in a type-elaboration layer, not in MLIR.
- Only lower to backend-oriented HIR after Pi applications are solved and specialization has made runtime tensor shapes representable by the existing lowering pipeline.
- Keep MLIR free of dependent constructs.

If you try to push symbolic index application into MLIR, you will end up duplicating typechecker logic in the backend and fighting the current tensor lowering, which is already tuned for concrete shapes.

## 3. Constraint solver scope

Pure prefix matching is useful, but only as a very small first slice.

What it can do well:

- Dot-product over a single literal or exact length variable.
- Exact frame/cell decomposition when the cell rank is already known.
- Some identity-style parametric functions where the shape is preserved.

What it cannot do:

- Express `append` result shapes.
- Express shape arithmetic like “left length + right length.”
- Express most of the interesting shape relationships users expect from Remora once they move beyond fixed-rank kernels.

For the three examples you called out:

- `dot-product`: prefix matching or exact dimension equality is enough.
- `reduce`: if the cell rank is already known, you mostly need frame/cell decomposition, not arithmetic.
- `append`: you need at least concatenation or addition on the leading dimension. Pure prefix matching is not enough for a general append type.

So the minimum viable solver depends on what you want to claim:

- If the claim is “we can typecheck exact-length Pi-typed kernels,” prefix matching is enough.
- If the claim is “users can write useful shape-parametric array code,” you need at least one of:
  - shape concatenation with a trailing rest variable, or
  - arithmetic on leading dimensions.

I would not present Phase 7a as practically useful unless it includes at least exact dimension variables plus a way to reconstruct a result shape from the argument shape. Otherwise it is mostly a proof of concept.

## 4. Missed gaps

Yes, there are a few deferred items that should be addressed before or alongside Phase 7, not after it.

The biggest one is the frame/cell abstraction itself.

- The typechecker already relies on rank and suffix reasoning for map-like inference, but the backend still has rank-1 special cases scattered through lowering, especially for rotate/append/scan in [`/e/projects/remorac/remora/lowering/tensor_ops.py`](/e/projects/remorac/remora/lowering/tensor_ops.py#L1734) and [`/e/projects/remorac/remora/lowering/tensor_ops.py`](/e/projects/remorac/remora/lowering/tensor_ops.py#L1937).
- The current status doc correctly says rank-2 scan/rotate/append are deferred because they need frame/cell semantics, but that is also a warning sign that the abstraction boundary is not clean yet.

My read:

- This is not a reason to block Phase 7.
- It is a reason to factor a shared frame/cell decomposition utility or IR now, before dependent types multiply the number of shape cases.

In other words, the deferred rank-2 items do suggest a deeper design issue, but it is not “dependent types are wrong.” It is “shape semantics are still encoded in too many places.”

The other item I would not leave too loose is `Sigma`/boxing semantics. The current implementation already erases boxes aggressively, so Phase 7 should make the erasure contract explicit rather than discovering it piecemeal.

## 5. Risk blind spots

The biggest risk is maintainability, and it is real.

The current typechecker is already doing a lot in one place: candidate search, rank inference, coercions, top-level function instantiation, and special casing for many primitives. You can see that structure in [`/e/projects/remorac/remora/typechecker.py`](/e/projects/remorac/remora/typechecker.py#L560), [`/e/projects/remorac/remora/typechecker.py`](/e/projects/remorac/remora/typechecker.py#L663), and [`/e/projects/remorac/remora/typechecker.py`](/e/projects/remorac/remora/typechecker.py#L1398).

If you add bidirectional inference and a constraint solver directly into that recursion, the code will become hard to reason about quickly.

The main blind spots I would call out:

- Ambiguity explosion. More candidate derivations means more backtracking and worse error messages unless you separate synthesis from solving.
- Substitution hygiene. Index variables need a clear representation and substitution story, or you will get hard-to-debug leakage between value-level and type-level names.
- Regression risk in old rank-polymorphic paths. Existing inference is permissive in a lot of places; adding Pi types may change which branch wins.
- Test matrix growth. You need positive cases, negative cases, and ambiguity cases, not just “works on dot-product.”

My recommendation is to keep the current typechecker mostly syntax-directed and introduce a separate elaboration/constraint module. Do not bury the solver inside the existing `infer` recursion.

## 6. Practicality

Restricted Phase 7a is useful, but only as a stepping stone.

If Phase 7a means “literal dimensions only, exact matching only, no concatenation, no arithmetic,” then it is useful mainly for:

- Dot-product-style examples.
- A correctness proof that Pi-type parsing and erasure work.
- Internal validation of the elaboration pipeline.

It is not yet enough to feel like a meaningful language upgrade for users.

What would actually let users write more expressive programs is some combination of:

- A real length variable that can appear in function types.
- Exact equality plus a trailing rest variable for shapes.
- Generalized application synthesis that can recover frame/cell splits.
- At least one shape-forming operation beyond exact matching, usually concatenation or leading-dimension arithmetic.

So my practical verdict is:

- Yes, ship a restricted 7a if you want to de-risk the implementation.
- No, do not advertise it as “dependent types are here” unless it can handle more than one exact-length example.
- If you want users to feel the benefit, the smallest meaningful visible feature set is probably 7a plus part of 7b.

## Recommendation

I would revise the plan to:

1. Make Phase 7a an elaboration-and-erasure milestone with exact length variables, not a broad user feature.
2. Extract a shared frame/cell decomposition layer before or alongside 7a.
3. Introduce a stable typed elaborated core and a separate erasure/specialization step; do not make backend HIR the only durable representation of solved programs.
4. Keep Pi application resolution out of MLIR.
5. Treat append-like shape arithmetic as the first genuinely user-visible dependent-type win, which probably means 7a is not the final “ship” slice.
