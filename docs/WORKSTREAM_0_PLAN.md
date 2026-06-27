# Workstream 0 Plan: Technical-Debt Paydown For Full-Language Work

*Detailed implementation plan for Workstream 0 of
[`PLAN_TO_IMPLEMENT_FULL_REMORA.md`](PLAN_TO_IMPLEMENT_FULL_REMORA.md). Read that
document and [`CODEX_PROJECT_REVIEW.md`](CODEX_PROJECT_REVIEW.md) first for
context. This plan also realizes "Phase 0: Stabilize Planning Infrastructure"
and "Milestone A: Auditable Current Compiler".*

## 1. Purpose

Full Remora support (dynamic shapes, runtime boxes, ragged arrays, segmented
reductions, records, dynamic higher-order semantics) will touch the largest and
most complex modules in the project — `gpu_lowering.py` (6.7 kLOC),
`tensor_ops.py` (5.0 kLOC), `typechecker.py` (3.5 kLOC), `codegen.py` (1.3 kLOC),
`runtime.py` (2.3 kLOC). Today, backend routing is an implicit `isinstance`
cascade, support claims live only in prose tables, errors carry no source
location, and a half-retired MLIR builder path adds cognitive load.

Workstream 0 turns **implicit backend behavior into explicit, testable
contracts** so the later workstreams can be reviewed and trusted. It is not
cosmetic: every later change to dynamic shapes and boxes needs a stable place to
register capabilities, an explainable routing decision, source-located failures,
and a differential safety net.

This is **debt paydown and infrastructure only**. It must not change observable
compilation behavior for the existing dense core. Golden-MLIR output must remain
byte-identical; numeric parity must be preserved.

## 2. Scope

### In scope (the seven Workstream 0 deliverables)

1. A **backend route registry** recording candidate lowerings, predicates,
   priority, capability requirements, and failure reasons.
2. **Extracted lowering utilities** for repeated descriptor loads, index
   decomposition, dtype-specific ops, bounds checks, and result stores.
3. A **decision on the MLIR builder path**: delete it or keep a deliberately
   small validation-only surface.
4. **Source-located errors** for typechecker, lowering, runtime-shape, and
   backend unsupported-feature failures.
5. A **generated/differential test harness** for a small dense subset.
6. **Updated docs** whose backend/status claims are backed by the capability
   matrix.
7. (From Phase 0) An **executable capability matrix**, `--explain-lowering`, and
   **cost-annotation data structures** (definitions only, no scheduling).

### Out of scope (explicitly deferred to later workstreams)

- Any new language feature: dynamic shapes, boxes, ragged arrays, segmented
  reductions, records, dynamic HOFs (Workstreams 1.5–6).
- Any cost-based **scheduling decision** (Workstream 7 / Phase 5). Workstream 0
  only *defines the data structures*.
- The lowering-module *file splits* for dynamic shapes (Phase 0.5 / Milestone
  A.5). Workstream 0 extracts helpers and isolates assumptions; the operation-
  family file split is a separate milestone.
- Changing the `int(d.value)` static-dim assumption into a runtime `DimValue`
  (Workstream 2). Workstream 0 only **centralizes** those call sites behind a
  helper so the later refactor is local.

## 3. Current-State Findings (grounding)

These findings drove the task breakdown below.

| Area | Observation | Reference |
| --- | --- | --- |
| GPU routing | One ~930-line function is an `if isinstance(function.body, …)` cascade with ~30 `return ptx, meta, plan` points and many `raise CodegenUnavailable`. Each block = predicate + shape/dtype guard + `build_descriptor_abi_*` call + `KernelMeta` + MLIR→LLVM→PTX translate + return. The general-map builder is the universal fallback; `CodegenUnavailable` is raised when nothing fits. | `remora/codegen.py:142` (`generate_mlir_descriptor_abi_ptx`) |
| Capability matrix | `remora/capabilities.py` does **not exist**. Support is claimed only in prose tables. | `docs/PROJECT_OVERVIEW_AND_ARCHITECTURE.md:23-50`, `docs/USER_GUIDE.md`, `docs/BACKEND_GAPS.md` |
| Diagnostics | `errors.py` is a 5-line base `RemoraError(Exception)`. ~19 subclasses exist (`HIRLoweringError`, `CodegenUnavailable`, `GPUScaffoldError`, `RemoraTypeError`, `RemoraLoweringError`, `CoreVerificationError`, `ConstraintError`, …) but **none carry a source location**. | `remora/errors.py:4`; subclasses across modules |
| CLI | argparse with `--emit-ast/-typed-ast/-hir/-mlir/-ptx`; no `--explain-lowering`. | `remora/cli.py:41`, `:76-80` |
| Builder path | "currently disabled"; `_builder_emitter.py` (686), `_builder_ops.py` (1190), `scalar_builder.py` (199), `_gpu_builder.py` (323). Tests still exercise it (`test_lowering.py:838-996`, `test_gpu_lowering.py:839-917`). | `remora/lowering/module.py:282` |
| Shared helpers | `_descriptor_load_lines(prefix, name, rank)` already exists and is reused ~80×. Other idioms remain duplicated: `_descriptor_type`, op-expr emitters, `tuple(int(d.value) for d in shape)`, `KernelMeta(...)` + translate-and-return boilerplate. | `remora/gpu_lowering.py:2268`, `:2262`, `:2334`, `codegen.py:194` |
| Test oracle | Interpreter `evaluate_source(...)`; CPU `CPUFunctionExecutor.compile_source()` / `evaluate_source_compiled()`; GPU `TestGPUNumericParity._run_parity` in `tests/test_gpu_general_lowering.py`. pytest only, no plugins. | `AGENTS.md`, `tests/test_gpu_general_lowering.py` |
| Cost types | `CostShape` / `ScheduleCandidate` sketched in the parent plan; not yet defined in code. | `PLAN_TO_IMPLEMENT_FULL_REMORA.md:776-793` |

## 4. Tracks

Nine tracks (A–I), one per deliverable. Each is independently reviewable. The
recommended ordering and dependency graph is in §5.

---

### Track A — Executable Capability Matrix (`remora/capabilities.py`)

**Goal.** One structured, importable source of truth for operation support and
cost-relevant properties, consumed by docs, tests, `--explain-lowering`, and the
route registry.

**New module.** `remora/capabilities.py`.

**Data model** (matches parent plan Workstream 1; superset of what Phase 0
needs):

```python
class Backend(enum.Enum):
    INTERP = "interp"
    CPU = "cpu"
    GPU = "gpu"

class Context(enum.Enum):
    TOP_LEVEL = "top-level"
    MAP_BODY = "map-body"
    FOLD_BODY = "fold-body"
    SCAN_BODY = "scan-body"
    AD_GENERATED = "ad-generated"

@dataclass(frozen=True)
class Capability:
    op: str                       # HIR node / op key, e.g. "HIRMap", "sort"
    backend: Backend
    dtypes: frozenset[str]        # {"f32","i32","bool","f64"}
    ranks: tuple[int, ...] | None # None = any up to MAX_DENSE_RANK
    static_shape: bool
    dynamic_shape: bool           # all False initially
    boxed_ragged: bool            # all False initially
    contexts: frozenset[Context]
    work_estimate: str | None     # asymptotic note, no scheduling yet
    memory_estimate: str | None
    requires_launch_transfer: bool
    fallback: Backend | None
    unsupported_reason: str | None  # user-facing text when not supported
    status: Literal["supported", "limited", "unsupported"]
```

**Steps.**

1. Define the enums/dataclasses and a `CAPABILITIES: tuple[Capability, ...]`
   registry plus query helpers: `lookup(op, backend, *, dtype, rank, context)`,
   `supported_ops(backend)`, `as_rows()` (for doc/table generation).
2. Populate entries for the **current** dense-core ops by reading the existing
   PROJECT_OVERVIEW / USER_GUIDE / BACKEND_GAPS tables and, where ambiguous,
   the actual lowering code (`codegen.py` predicates, `_gpu_map_support.py`,
   `gpu_lowering.py` builders). Encode the *real* GPU narrowness (e.g. `sort`
   GPU f32-only, `matmul` GPU f32-only, closures GPU-unsupported, im2col/col2im
   GPU-unsupported, pairs/boxes GPU-unsupported).
3. Add **record/data-frame** placeholder entries (Phase 0 task 12), all
   `status="unsupported"`, except any already-existing pair/tuple-adjacent
   behavior, with clear `unsupported_reason`.
4. Mark all `dynamic_shape=False`, `boxed_ragged=False` to make the current
   static reality explicit.
5. Keep it pure data + queries — **no** routing logic and **no** cost math here.

**Tests.** `tests/test_capabilities.py`:

- Registry integrity: no duplicate `(op, backend, context, dtype)`; every
  `unsupported`/`limited` entry has a non-empty `unsupported_reason`; every dtype
  string is from the known set; ranks within `MAX_DENSE_RANK`.
- Query helpers behave (lookup hit/miss, fallback resolution).

**Exit.** `capabilities.lookup(...)` answers "is `(op, backend, dtype, rank,
context)` supported, and if not, why?" for every current op.

---

### Track B — Backend Route Registry (migrate the `codegen.py` cascade)

**Goal.** Replace the implicit GPU dispatch cascade with an explicit registry of
route candidates, **without changing behavior**.

**New module.** `remora/route_registry.py`.

**Design.**

```python
@dataclass(frozen=True)
class RouteResult:
    ptx: str
    metas: list[KernelMeta]
    plan: ExecutionPlan | None

@dataclass(frozen=True)
class RouteDecision:        # for --explain-lowering and tests
    route_name: str
    accepted: bool
    reason: str             # why accepted, or why rejected
    capability_keys: tuple[str, ...]

@dataclass(frozen=True)
class Route:
    name: str
    priority: int                                   # lower = tried first
    predicate: Callable[[HIRFunction, RouteContext], bool]
    capability_keys: tuple[str, ...]                # entries in capabilities.py
    build: Callable[[HIRFunction, RouteContext], RouteResult]

def select_route(function, ctx) -> tuple[Route, list[RouteDecision]]: ...
```

`select_route` walks routes in priority order, recording a `RouteDecision` per
candidate (accepted/rejected + reason), returning the first acceptor and the full
trace (the trace feeds `--explain-lowering`).

**Steps.**

1. **Wrap, do not rewrite.** Each existing `if isinstance(function.body, …):`
   block in `generate_mlir_descriptor_abi_ptx` becomes a `Route` whose
   `predicate` is the extracted `isinstance`/shape/dtype guard and whose `build`
   is the extracted body (call `build_descriptor_abi_*`, construct `KernelMeta`,
   translate to PTX, return `RouteResult`). The builder functions and their
   internals are **not touched**.
2. Preserve the existing **priority order exactly** (im2col first … general-map
   fallback last). Capture the order from the current top-to-bottom cascade.
3. The current "nothing matched → `raise CodegenUnavailable`" becomes "no route
   accepted → raise `CodegenUnavailable` whose message lists the rejected routes
   and reasons" (richer, but same failure condition).
4. Re-implement `generate_mlir_descriptor_abi_ptx` as a thin shim over
   `select_route(...).build(...)` so all current callers (`compiler.py`,
   `cli.py`, `repl.py`) are unchanged.
5. Cross-link each route to its `capabilities.py` keys so a route cannot claim
   support the matrix denies (assert in a test).

**Safety / behavior-preservation.**

- **Golden-MLIR / golden-PTX must be byte-identical** after the migration. Run
  the full lowering + GPU parity suite before/after.
- Add a characterization step: snapshot, for a corpus of programs, which route
  *name* each program selects, then assert stability across the refactor.

**Tests.** `tests/test_route_registry.py`:

- Priority ordering is deterministic and total.
- For a representative corpus, `select_route` picks the route equivalent to the
  pre-migration cascade (route-selection snapshot test).
- Every route's `capability_keys` exist in `capabilities.py` and agree on dtype/
  rank support.
- Rejected-not-silent: an unsupported body yields a `CodegenUnavailable` whose
  message enumerates rejected routes.

**Exit.** The registry can report which lowering path accepted or rejected a HIR
program; behavior and golden outputs are unchanged.

---

### Track C — Extracted Lowering Utilities

**Goal.** Push repeated GPU/CPU text-emission idioms behind small, tested
helpers so later dynamic-shape work edits one helper, not dozens of sites.

**Target module.** `remora/lowering/_emit_helpers.py` (new) — or extend the
existing helper cluster in `gpu_lowering.py` and re-export. Keep helpers
**pure** (string in/out, no global state).

**Idioms to extract / consolidate** (several already partly exist):

1. **Descriptor loads** — `_descriptor_load_lines` already exists (`gpu_lowering.py:2268`, ~80 uses). Audit for stray inline copies and route all through it.
2. **Descriptor types** — `_descriptor_type(rank)` (`:2262`): confirm single source.
3. **Index decomposition** — linear-index → multi-index (row-major) used across
   map/fold/view emitters. Extract `emit_delinearize(linear, sizes) -> lines`.
4. **Dtype-specific scalar ops** — unify the parallel `_descriptor_unary_op_expr`
   / `_descriptor_i32_unary_op_expr` / `_descriptor_bool_unary_op_expr` (and the
   binary variants) behind a dtype-parameterized emitter.
5. **Bounds / guard checks** — `if tid < N` style guards.
6. **Result stores** — descriptor-relative store sequences.
7. **Translate-and-return boilerplate** — the `extract_gpu_module_body_as_module`
   → `translate_mlir_to_llvmir` → `translate_llvmir_to_nvptx_text` →
   `return ptx, [meta], plan` triple repeated at every route. Extract
   `module_to_route_result(gpu_module, metas, plan)`.
8. **Static-dim access** — the `int(d.value)` / `tuple(int(d.value) for d in shape)`
   pattern (`codegen.py:194`, and dozens across `gpu_lowering.py`,
   `_gpu_expr_lowering.py`, `lowering/tensor_ops.py`, `_builder_ops.py`,
   `view_ops.py`). Centralize behind `static_dim(d) -> int` and
   `static_shape(t) -> tuple[int, ...]` **now**, with an explicit
   `assert`/raise documenting "static-only here". This is the single most
   valuable extraction: it makes the Workstream 2 `DimValue` swap a local change
   and is the constant-site-sprawl mitigation named in the parent plan's Risks.

**Steps.**

1. Introduce helpers one idiom at a time; each commit replaces call sites for a
   single idiom and asserts golden output is unchanged.
2. Do **not** combine extraction with any semantic change.
3. Add a module-level note in each large lowering file pointing to the shared
   helpers.

**Tests.** `tests/test_emit_helpers.py` (unit tests on emitted text) **plus**
the existing golden-MLIR/PTX suite as the regression gate (must stay
byte-identical).

**Exit.** Repeated descriptor/index/dtype/store idioms have one home; static-dim
access is centralized and labeled.

---

### Track D — MLIR Builder Path Decision

**Goal.** End the half-retired builder path's cognitive load with a documented
decision.

**Current footprint.** `lowering/_builder_emitter.py` (686),
`lowering/_builder_ops.py` (1190), `lowering/scalar_builder.py` (199),
`lowering/_gpu_builder.py` (323); disabled per `lowering/module.py:282`; still
referenced by `test_lowering.py:838-996` and `test_gpu_lowering.py:839-917`.

**Decision process.**

1. Inventory: is the builder path reachable from any production entry point
   (compiler/cli/repl), or only from tests? (Grep + call-graph from
   `compile_source*`.)
2. Decide between two options and record the decision (and rationale) in this
   doc and in `docs/IMPLEMENTATION_LOG.md`:
   - **(D1) Delete.** Remove the four modules and the builder-only tests; drop
     the `module.py:282` dead branch and its comment.
   - **(D2) Validation-only surface.** Keep a *minimal* builder path, gate it
     behind an explicit `validate-only` flag, document it as a structural cross-
     check (not a production backend), and delete everything outside that
     surface.

   Recommendation: prefer **D1 (delete)** unless the builder provides a unique
   structural-validation value the text path cannot, since PROJECT_OVERVIEW
   already documents the builder as "slower and less capable" and the text path
   as the single supported path.
3. Use [`question`] with the user to confirm D1 vs D2 before deleting code, since
   this is destructive.

**Tests.** If D1: delete builder-only tests, confirm the suite is green and
golden outputs unchanged. If D2: convert builder tests into the documented
validation-only suite.

**Exit.** No half-retired path; a single documented rationale; suite green.

---

### Track E — Source-Located Diagnostics

**Goal.** Attach source locations to compiler/runtime errors so failures
(especially the upcoming dynamic-shape and box failures) are explainable. The
parent plan flags this as "easier to retrofit while the dense static core is
still the main target."

**Steps.**

1. Add `SourceSpan` (file, start line/col, end line/col) and a `Located` mixin to
   `remora/errors.py`:

   ```python
   @dataclass(frozen=True)
   class SourceSpan:
       file: str | None
       line: int
       col: int
       end_line: int | None = None
       end_col: int | None = None
       def format(self) -> str: ...  # "file:line:col"

   class RemoraError(Exception):
       span: SourceSpan | None = None
       def located(self, span) -> "RemoraError": ...   # attach + return self
       def __str__(self) -> str: ...  # prefix "file:line:col: " when span set
   ```

2. **Capture spans at parse time.** Verify the Lark parsers
   (`parser.py`, `lisp_reader.py`) expose token line/col; thread a `SourceSpan`
   onto AST nodes (add an optional `span` field on AST nodes, or a side-table
   keyed by node identity if adding fields is too invasive).
3. **Propagate** spans through typed AST → elaborated → HIR for the nodes most
   likely to raise (so `RemoraTypeError`, `HIRLoweringError`,
   `CodegenUnavailable`, `GPUScaffoldError`, runtime shape-mismatch can call
   `.located(span)`).
4. **Four required failure sites** (deliverable 4): typechecker errors, lowering
   errors, runtime-shape errors, and backend unsupported-feature errors all
   carry a span when one is available; otherwise degrade gracefully (no span =
   current behavior).
5. CLI prints `remorac: file:line:col: message` (extend the existing handler at
   `cli.py:109`).

**Scope discipline.** Do **not** rewrite every raise site. Thread spans where
they are cheap (parser → AST) and attach at the four required boundaries. A
follow-up can widen coverage.

**Tests.** `tests/test_diagnostics.py`:

- A "diagnostics are stable" suite (per CODEX review): for a set of canonical bad
  programs (type error, unknown var, unsupported GPU construct), assert the error
  is the expected subclass and the message contains `file:line:col`.
- Golden-message snapshots kept small and intentional.

**Exit.** Typechecker, lowering, runtime-shape, and backend-unsupported failures
produce source-located messages.

---

### Track F — Generated / Differential Test Harness

**Goal.** A small, generated differential-test harness over a deliberately narrow
dense subset, comparing interpreter (oracle) vs CPU vs GPU-where-supported —
landed *before* dynamic features, as a safety net for all later work.

**New file.** `tests/test_differential_dense.py` plus a generator
`tests/_dense_gen.py`.

**Generator scope (narrow on purpose).**

- Element types: `f32`, `i32`, `bool` where relevant.
- Shapes: scalar, rank-1, rank-2 at small fixed sizes.
- Constructs: scalar arithmetic, `map` (unary/binary/sections), `fold`/`reduce`,
  `if`/`select`, `let`, and a couple of views (`reverse`, `take`, `drop`).
- Deterministic seeding; bounded program size; reproducible corpus.

**Harness.**

- Oracle = `evaluate_source(...)`.
- CPU = `CPUFunctionExecutor.compile_source()` / `evaluate_source_compiled()`.
- GPU = reuse `TestGPUNumericParity._run_parity` from
  `tests/test_gpu_general_lowering.py`; **consult `capabilities.py`** to decide
  whether to assert GPU parity or assert a loud rejection
  (rejected-not-silent), so the harness self-aligns with the matrix.
- Respect `REMORA_TEST_GPU` semantics (GPU parity only asserted where supported;
  unavailable GPU degrades per `conftest.py`).

**Steps.**

1. Build the generator (pure-Python AST/source emitter for the subset).
2. Wire interpreter/CPU parametrized parity.
3. Add GPU parity/rejection driven by the capability matrix.
4. Keep runtime modest (cap corpus size; mark a larger sweep `slow` if needed).

**Tests.** The harness *is* the test; add a meta-test that the generator only
emits well-typed programs (it should never produce a type error).

**Exit.** A generated dense-subset suite runs interpreter/CPU parity and
GPU parity-or-rejection where the matrix says so.

---

### Track G — `--explain-lowering`

**Goal.** Make backend routing visible and testable from the CLI/API (CODEX
"move it up in priority"). Depends on Tracks A and B.

**Steps.**

1. Add `--explain-lowering` to `cli.py` (next to `--emit-*`, `cli.py:76-80`).
2. When set, run selection (Track B `select_route`) and print, per the program /
   per target function:
   - which route/pattern was selected (or that CPU/interpreter was chosen);
   - the candidate routes considered with accept/reject **reasons** (the
     `RouteDecision` trace);
   - the dtype/rank/shape guards discharged;
   - capability-matrix keys backing the decision;
   - what was rejected and why (for GPU-rejected regions).
3. Provide a programmatic entry (`compiler.explain_lowering(source, …) ->
   structured object`) so tests don't scrape stdout.
4. (Optional, low cost) mirror the structured object into the existing output
   metadata JSON sidecar so artifacts are self-describing.

**Tests.** `tests/test_explain_lowering.py`: for representative programs, the
structured explanation names the expected route and lists the expected
rejections; one CLI smoke test of the text output.

**Exit.** Every backend selection has an explainable, testable reason.

---

### Track H — Cost-Annotation Data Structures (definitions only)

**Goal.** Land the cost/schedule **data structures** so later passes have a stable
place to attach metadata. **No scheduling decisions, no cost math, no behavior.**

**New module.** `remora/cost.py`.

```python
StaticOrSymbolic = int | str   # symbolic dims as names for now

@dataclass(frozen=True)
class CostShape:
    elem_count: StaticOrSymbolic
    bytes_read: StaticOrSymbolic
    bytes_written: StaticOrSymbolic
    flops: StaticOrSymbolic
    temporary_bytes: StaticOrSymbolic
    irregularity: Literal["regular", "segmented", "ragged", "dynamic-call"]

@dataclass(frozen=True)
class ScheduleCandidate:
    backend: Literal["cpu", "gpu"]
    plan_kind: str
    estimated_cost: object              # CostExpression placeholder
    requirements: list[str]             # capability keys
    fallback_reason: str | None
```

**Steps.** Define the dataclasses + a trivial constructor/`__repr__`. Add a unit
test that they instantiate and are frozen. Do **not** wire them into any pass.

**Exit.** Cost annotation structures exist and are importable, ready for
Workstream 7.

---

### Track I — Documentation Reconciliation

**Goal.** Make backend/status claims agree with the capability matrix; fix the
specific drifts CODEX called out. Depends on Track A.

**Known drifts to fix (Phase 0 task 11).**

- GPU `sort`/`grade` dtype claims (USER_GUIDE says f32-only; other docs imply
  i32 accepted) — state precisely, backed by the matrix.
- Dynamic-shape wording: USER_GUIDE interpreter table vs PROJECT_OVERVIEW
  ("dynamic shapes not implemented") — make terminology precise (interpreter
  flexibility/boxes vs true runtime dynamic lowering).
- `pairs`, `im2col`/`col2im`, and `f64` support claims per backend.

**Steps.**

1. Add a doc-generation/validation helper (e.g. `capabilities.as_rows()` →
   markdown) used to regenerate or validate the support tables in
   `PROJECT_OVERVIEW_AND_ARCHITECTURE.md` and `USER_GUIDE.md` and
   `BACKEND_GAPS.md`.
2. A **test** (`tests/test_docs_match_capabilities.py`) parses the docs-visible
   support table(s) and asserts each `(op, backend, status)` matches
   `capabilities.py` (CODEX "turn support tables into parametrized tests").
3. Update prose to match the matrix; record record/data-frame entries as
   unsupported.

**Exit.** Docs and tests agree on operation/backend support; the support tables
are matrix-backed.

---

## 5. Sequencing & Dependencies

```
A (capabilities)  ─┬─> B (route registry) ─┬─> G (--explain-lowering)
                   │                        │
                   ├─> I (docs reconcile)   │
                   └─> F (differential harness, uses matrix to gate GPU)
C (lowering helpers) ── independent, but do BEFORE/with B to keep routes thin
D (builder decision) ── independent (needs user confirm D1 vs D2)
E (diagnostics)      ── independent (parser→AST span plumbing)
H (cost structs)     ── independent, trivial
```

**Critical path:** A → B → G. **Highest-leverage early:** C (static-dim
centralization) and A (matrix). **Needs a user decision before coding:** D.

### Suggested PR breakdown (small, reviewable, behavior-preserving)

1. **PR1 — Track A.** `capabilities.py` + tests. No behavior change.
2. **PR2 — Track C (static-dim + translate-return).** Centralize `int(d.value)`
   and the translate-and-return boilerplate. Golden outputs unchanged.
3. **PR3 — Track C (descriptor/index/dtype/store).** Remaining idiom extraction.
   Golden outputs unchanged.
4. **PR4 — Track B.** Route registry wrapping the cascade; route-selection
   snapshot test; golden outputs unchanged.
5. **PR5 — Track G.** `--explain-lowering` + structured API + tests.
6. **PR6 — Track E.** Source spans + four located failure sites + diagnostics
   suite.
7. **PR7 — Track F.** Differential dense harness (matrix-gated GPU).
8. **PR8 — Track I.** Docs reconciliation + docs-vs-matrix test.
9. **PR9 — Track H.** Cost dataclasses + unit test.
10. **PR10 — Track D.** Builder path delete-or-validation-only (after user
    confirms).

Each PR runs the **fast verification step** (`uv run python -m compileall -q
remora`), the full `uv run pytest` (CPU+GPU locally), and — for PRs touching any
lowering/route path (PR2–PR5, PR10) — states in the PR that **local GPU parity
was run** (per AGENTS.md / CI gap).

## 6. Global Exit Criteria (Phase 0 / Milestone A)

Workstream 0 is complete when:

1. Every backend selection has an **explainable reason** (`--explain-lowering` +
   route registry).
2. **Docs and tests agree** on operation/backend support (matrix-backed tables +
   `test_docs_match_capabilities`).
3. **Unsupported GPU paths remain loud and source-located** (rejected-not-silent
   tests; `CodegenUnavailable`/`GPUScaffoldError` carry spans where available).
4. The **route registry** can show which lowering path accepted or rejected a
   HIR program.
5. New dynamic-shape work has a **stable place** to register capabilities and
   unsupported-feature reasons (`capabilities.py` + route `capability_keys`).
6. The **generated differential suite** runs interpreter/CPU parity and GPU
   parity-where-supported.
7. **No behavior/golden-output change** for the existing dense core (verified by
   golden-MLIR/PTX byte-identity and numeric parity before/after each PR).
8. The **MLIR builder path** is resolved (deleted or validation-only) with a
   recorded rationale.
9. **Cost-annotation data structures** exist (definitions only).

## 7. Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| Refactor silently changes GPU routing or output | Route-selection snapshot test + byte-identical golden MLIR/PTX gate on every PR; run GPU parity locally. |
| Constant-site sprawl during Track C | Centralize `static_dim`/`static_shape` first (PR2); golden tests catch drift; one idiom per commit. |
| Diagnostics scope creep | Thread spans only parser→AST + four required boundaries; degrade gracefully without a span. |
| Differential harness too slow / flaky | Cap corpus, deterministic seeds, mark large sweeps `slow`; gate GPU via matrix + `REMORA_TEST_GPU`. |
| Deleting the builder path loses value | Confirm D1 vs D2 with the user; inventory reachability before deletion; keep history in `IMPLEMENTATION_LOG.md`. |
| Matrix duplicates/contradicts docs | `test_docs_match_capabilities` makes the matrix authoritative; docs generated/validated from it. |
| Mixing refactor with semantics | Hard rule: no Workstream 0 PR adds or changes a language feature; cost structs are inert; route registry only wraps. |

## 8. Open Questions for the User

1. **Builder path (Track D):** delete entirely (recommended) or keep a minimal
   validation-only surface?
2. **`--explain-lowering` output:** human-readable text only, or also a
   `--explain-lowering=json` machine form / metadata-sidecar field?
3. **Span plumbing depth (Track E):** add an optional `span` field to AST nodes
   (more invasive, cleaner) or a side-table keyed by node identity (less
   invasive)?
