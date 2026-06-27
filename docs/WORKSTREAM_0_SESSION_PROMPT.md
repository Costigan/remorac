# Workstream 0 — New-Session Implementation Prompt

*Paste this as the opening prompt for a fresh session tasked with implementing
Workstream 0. It is self-contained but points to the authoritative docs rather
than duplicating them.*

---

You are implementing **Workstream 0 (Technical-Debt Paydown)** of the RemoraC
"full Remora" effort. This is infrastructure/debt-paydown work that must make the
compiler auditable **without changing any observable compilation behavior**.

## Start by reading (in this order)

1. `AGENTS.md` — build/test commands, GPU testing policy, coverage rules.
2. `docs/WORKSTREAM_0_PLAN.md` — **your spec.** Tracks A–I, the PR breakdown
   (PR1–PR10), sequencing, and exit criteria. Follow it.
3. `docs/PLAN_TO_IMPLEMENT_FULL_REMORA.md` (Workstream 0 §, Phase 0, Milestone A)
   and `docs/CODEX_PROJECT_REVIEW.md` — the motivation behind the plan.
4. `docs/PROJECT_OVERVIEW_AND_ARCHITECTURE.md` — pipeline and module map.

## Project orientation (one paragraph)

RemoraC compiles the dense, statically-shaped core of the Remora array language
(ML + Lisp syntax) to CPU and GPU via MLIR. Pipeline:
`parser/lisp_reader → typechecker → elaborate → erase → hir → hir_opt/defunc →
lowering/ (MLIR) → pipeline/codegen → CPU a.out/.so or GPU PTX`. The interpreter
(`evaluate_source`) is the semantic oracle. Python 3.11+, `uv`, pytest only (no
plugins, no linter/type-checker). Fast check: `uv run python -m compileall -q
remora`. Tests: `uv run pytest` (runs CPU **and** GPU locally by default).

## Non-negotiable constraints

- **Behavior-preserving.** No Workstream 0 change may alter compilation results.
  **Golden MLIR/PTX must stay byte-identical** and numeric parity must hold.
  Run the relevant golden + parity tests before and after each change.
- **No new language features.** Dynamic shapes, boxes, ragged arrays, segmented
  reductions, records, dynamic HOFs are all out of scope (later workstreams).
  Cost structures (Track H) are inert definitions only — wire them into nothing.
- **Do not turn `int(d.value)` into a runtime `DimValue`.** Track C only
  *centralizes* those call sites behind `static_dim`/`static_shape` helpers; the
  semantic swap is Workstream 2.
- **Refactors and semantics never mix in one commit.** One idiom / one concern
  per commit.
- **Verify every unit:** `uv run python -m compileall -q remora` then
  `uv run pytest`. For any change touching a lowering/route path
  (`codegen.py`, `gpu_lowering.py`, `_gpu_expr_lowering.py`, `lowering/*`,
  `route_registry.py`), you must run the GPU parity suite locally and state that
  you did — CI skips GPU (`REMORA_TEST_GPU=0`).
- **Do not commit or open PRs unless I explicitly ask.** Stage work, show
  `git status`/`git diff`, and wait. (The "PR1–PR10" breakdown in the plan is
  your *unit-of-work* sizing, not an instruction to push.)
- **No comments in code unless asked; follow existing conventions** (mimic
  neighboring modules; check imports before assuming a library exists).

## Resolve these decisions with me BEFORE coding the affected track

The plan ends (§8) with three open questions. Ask me up front; do not guess:

1. **Track D (builder path):** delete the disabled MLIR builder modules
   (recommended) or keep a minimal validation-only surface? **Track D is
   destructive — do not delete anything until I confirm.**
2. **Track G (`--explain-lowering`):** text only, or also a JSON form / metadata
   sidecar field?
3. **Track E (diagnostics):** add an optional `span` field to AST nodes, or use a
   side-table keyed by node identity?

You may begin the tracks that don't depend on these answers (A, C, B, F, H, I)
while waiting.

## Execution method

1. Re-validate the plan's "Current-State Findings" table (§3) against the live
   code — **line numbers may have drifted**, so re-grep for the anchors
   (`generate_mlir_descriptor_abi_ptx` in `remora/codegen.py`,
   `_descriptor_load_lines`/`_descriptor_type` in `remora/gpu_lowering.py`,
   `RemoraError` in `remora/errors.py`, the `--emit-*` args in `remora/cli.py`,
   the "currently disabled" comment in `remora/lowering/module.py`). Confirm
   `remora/capabilities.py` still does not exist.
2. Create a todo list mirroring the PR breakdown (PR1–PR10 in §5) and work them
   in order, marking exactly one in-progress at a time.
3. Recommended order (critical path A → B → G; do C early to keep routes thin):
   - **PR1 / Track A** — `remora/capabilities.py` + `tests/test_capabilities.py`.
   - **PR2 / Track C** — centralize `static_dim`/`static_shape` and the
     translate-and-return boilerplate; golden output unchanged.
   - **PR3 / Track C** — remaining descriptor/index/dtype/store idiom extraction.
   - **PR4 / Track B** — `remora/route_registry.py` wrapping the `codegen.py`
     cascade as named routes; add a route-selection snapshot test;
     `generate_mlir_descriptor_abi_ptx` becomes a thin shim; behavior unchanged.
   - **PR5 / Track G** — `--explain-lowering` + `compiler.explain_lowering(...)`.
   - **PR6 / Track E** — `SourceSpan`/`located()` in `errors.py`, span plumbing,
     four located failure sites, `tests/test_diagnostics.py`.
   - **PR7 / Track F** — `tests/_dense_gen.py` + `tests/test_differential_dense.py`,
     matrix-gated GPU parity-or-rejection.
   - **PR8 / Track I** — reconcile docs to the matrix +
     `tests/test_docs_match_capabilities.py`.
   - **PR9 / Track H** — `remora/cost.py` dataclasses + a trivial unit test.
   - **PR10 / Track D** — only after I confirm delete-vs-keep.
4. After each unit: run the fast check + targeted tests, then the full suite; show
   me the diff and the test result; pause for review before moving on.

## Definition of done

The track's **Exit** bullet in `docs/WORKSTREAM_0_PLAN.md` is met, its tests
pass, `uv run pytest` is green (CPU+GPU locally), and golden MLIR/PTX is
byte-identical. The whole workstream is done when the nine **Global Exit
Criteria** in §6 hold. Update `docs/IMPLEMENTATION_LOG.md` with the Track D
decision rationale when it lands.

## Watch out for

- The GPU dispatch cascade (`generate_mlir_descriptor_abi_ptx`) has ~30 return
  points and a strict top-to-bottom priority (im2col first … general-map
  fallback last). **Preserve that order exactly** when wrapping it as routes.
- `_descriptor_load_lines` already exists and is reused ~80×; consolidate around
  it, don't reinvent it.
- The differential harness must respect `tests/conftest.py` `REMORA_TEST_GPU`
  semantics and consult `capabilities.py` to decide parity-vs-loud-rejection.
- Keep each PR-sized change small and reviewable; never bundle a refactor with a
  behavior change.
