# Plan To Implement Full Remora

This plan targets a specific goal:

1. implement the full Remora language in the interpreter;
2. implement the full Remora language in the compiled CPU backend;
3. do not require full GPU semantic parity;
4. use the GPU as an accelerator for common, profitable patterns;
5. explore type-system support for estimating runtime cost and selecting CPU or
   GPU schedules.

The GPU should be correct for every program it accepts, but it may reject or
fall back for programs outside its profitable subset. The CPU backend and
interpreter are the semantic baseline.

## Current Position

RemoraC already has a strong dense-core compiler:

1. two frontends, ML syntax and Lisp syntax, lowering to one AST;
2. dependent-shape machinery in the typechecker;
3. an interpreter used as semantic oracle;
4. CPU lowering for the dense, statically shaped core;
5. substantial direct-CUDA GPU lowering for dense numeric patterns;
6. numeric-parity testing culture for GPU paths.

The remaining full-language gaps are not mostly syntax. They are runtime
representation and compilation-model gaps:

01. true dynamic shapes;
02. runtime boxes and existential dimension witnesses;
03. ragged arrays and irregular data;
04. segmented reductions;
05. ordered structural records and data-frame-style arrays of records;
06. full dynamic higher-order semantics;
07. arrays of functions / MIMD function application;
08. missing paper surface forms;
09. CPU lowering for dynamic and irregular values;
10. cost/scheduling metadata to decide when GPU acceleration is worthwhile.

## Design Principles

### Interpreter And CPU Define The Language

The interpreter and compiled CPU backend should eventually accept the same full
Remora programs. The interpreter remains the executable specification. The CPU
backend is the complete compiled implementation.

### GPU Is A Profitable Accelerator

The GPU backend should not chase complete language parity. It should accelerate
patterns where GPU execution is likely to beat CPU execution after accounting
for:

1. host/device transfer;
2. kernel launch overhead;
3. intermediate allocation;
4. parallel work size;
5. memory bandwidth;
6. operation intensity;
7. data already resident on device;
8. fusion opportunities.

Unsupported or unprofitable GPU regions should remain on CPU or fail loudly only
when the user explicitly requires GPU.

### Runtime Representation Comes Before Backend Completion

Dynamic shapes, boxes, and ragged arrays require a shared runtime model. Do not
patch each backend independently. Define the values, descriptors, metadata, and
ownership model first, then implement interpreter and CPU against it, then expose
profitable GPU subsets.

### Cost Should Be A Typed Compiler Artifact

Cost and schedule information should be produced by type/elaboration/lowering
passes as structured metadata, not guessed late from generated MLIR text. The
type system already knows ranks, cells, frames, dimensions, and many static
constraints; those are exactly the facts needed for cost bounds.

## Target Semantics

### Full Interpreter

The interpreter should support:

01. all dense static programs currently supported;
02. missing paper surface forms: `frame`, `array`, and `all` parameter syntax;
03. true dynamic shapes;
04. runtime boxes and unboxing with dimension witnesses;
05. ragged arrays as arrays of boxes;
06. segmented reductions;
07. ordered structural records as scalar values;
08. first-class record constructors, field accessors, and record updates that
    lift over arrays of records;
09. simple data-frame examples represented as arrays of records;
10. dynamic higher-order functions;
11. arrays of functions and MIMD function application;
12. `shape` and `rank` for function values if retained from the full language
    model;
13. full composition support in both ML and Lisp syntax.

### Full CPU Backend

The CPU backend should compile the same language surface as the interpreter
unless a construct is intentionally interpreter-only for debugging. CPU should
support:

1. dynamic memrefs and runtime loop bounds;
2. runtime allocation for dynamic results;
3. boxed values and ragged arrays;
4. segmented reductions;
5. ordered structural records, arrays of records, and column projection/update;
6. dynamic higher-order dispatch or specialization;
7. arrays of functions;
8. recursive array construction with dependent result sizes;
9. source-located errors for genuinely unsupported runtime/toolchain cases.

### GPU Accelerator Subset

GPU should prioritize:

1. large element-wise maps and fused map pipelines;
2. reductions and scans over dense numeric arrays;
3. stencils and PDE kernels;
4. matmul and linear algebra kernels;
5. sort/grade where sizes and dtypes are supported;
6. FFT/signal-processing kernels when added;
7. segmented reductions when profitable and regular enough;
8. device-resident iterative loops;
9. AD-generated dense numeric kernels.

GPU should not initially target:

1. general dynamic function dispatch;
2. arrays of functions;
3. general boxes/ragged arrays;
4. non-tail general recursion;
5. small kernels dominated by launch/transfer overhead;
6. irregular programs with poor occupancy or high divergence unless a concrete
   use case justifies it.

## Architecture Workstreams

### Workstream 0: Technical-Debt Paydown For Full-Language Work

Full Remora support will touch the largest, most complex parts of the project.
Before adding dynamic shapes and runtime boxes, pay down the debt that would make
those changes hard to review.

Priorities from `docs/CODEX_PROJECT_REVIEW.md`:

1. make backend routing visible and data-driven;
2. reduce monolithic lowering risk;
3. decide the disabled MLIR builder path;
4. centralize support/capability metadata;
5. improve source-located diagnostics;
6. add generated differential tests;
7. reconcile documentation/status drift.

This work is not cosmetic. It makes full Remora implementation more reliable by
turning implicit backend behavior into explicit, testable contracts.

Deliverables:

1. a backend route registry that records candidate lowerings, predicates,
   priority, capability requirements, and failure reasons;
2. extracted lowering utilities for repeated descriptor loads, index
   decomposition, dtype-specific ops, bounds checks, and result stores;
3. a decision on the MLIR builder path: delete it, or keep a deliberately small
   validation-only surface;
4. source-located errors for typechecker, lowering, runtime-shape, and backend
   unsupported-feature failures;
5. a generated/differential test harness for a small dense subset before adding
   dynamic features;
6. updated docs whose backend/status claims are backed by the capability matrix.

### Workstream 1: Executable Support And Cost Matrix

Create one structured source of truth for operation support and cost-relevant
properties.

Recommended module:

```text
remora/capabilities.py
```

Initial data model:

01. operation or HIR node;
02. backend: interpreter, cpu, gpu;
03. supported dtypes;
04. supported ranks;
05. static-shape support;
06. dynamic-shape support;
07. boxed/ragged support;
08. accepted contexts: top-level, map body, fold body, scan body, AD-generated;
09. asymptotic work estimate;
10. memory traffic estimate;
11. launch/transfer requirements;
12. fallback backend;
13. user-facing unsupported reason.

Use this for:

1. docs generation;
2. parametrized tests;
3. `--explain-lowering`;
4. backend route selection;
5. GPU scheduling decisions;
6. clearer unsupported-feature errors.

Deliverables:

1. static support matrix for current features;
2. tests that compare docs-visible claims against matrix entries;
3. CLI/API hook that explains why a backend was selected or rejected;
4. route-registry integration for `codegen.py` and GPU lowering paths;
5. compatibility layer so existing lowering code can be migrated incrementally.

### Workstream 1.5: Records And Data Frames

Add the record/data-frame ideas from `docs/remora-reference/Records with Rank Polymorphism.txt` early enough that later runtime and backend work can account
for heterogeneous scalar values. Do not wait until all dynamic shapes, boxes,
and segmented operations are complete, because records affect the type system,
value model, HIR, display, backend capability matrix, and layout decisions.

The initial goal is **not** a Pandas clone and **not** full row-polymorphic
lenses. The initial goal is a small, precise language feature:

1. records are scalar values from rank polymorphism's point of view;
2. record field order is preserved and observable in display;
3. a record type is an ordered list of unique field names and field types;
4. two record types with the same fields in different orders are different
   types for display and layout purposes;
5. a vector/table of records is an ordinary array whose element type is that
   record type;
6. record constructors, field accessors, and record updates are ordinary
   functions, so Remora's existing lifting rules turn row operations into
   column operations.

#### Surface Syntax

Implement both syntaxes, but keep the first version deliberately small.

Lisp syntax:

```scheme
((record loc day month hi lo)
 "Dallas" 28 3 74 57)

(view (field hi) weather-row)
(set (field hi) 75 weather-row)
(over (field hi) f->c weather-row)
```

Recommended sugar after the core forms work:

```scheme
{(loc "Dallas") (day 28) (month 3) (hi 74) (lo 57)}
#_(hi)     ; expands to (view (field hi))
#=(hi)     ; expands to (set (field hi))
#^(hi)     ; expands to (over (field hi))
```

ML syntax can start with explicit forms if parser work for record literals is
too large:

```remora
record loc day month hi lo "Dallas" 28 3 74 57
view (field hi) weather_row
set (field hi) 75 weather_row
over (field hi) f_to_c weather_row
```

Later ML syntax can add record literals and dot projection if the grammar can
support them cleanly:

```remora
{loc = "Dallas", day = 28, month = 3, hi = 74, lo = 57}
weather_row.hi
```

#### Type Model

Add these type forms:

```text
RecordType(fields: tuple[RecordField, ...])
RecordField(name: str, type: Type)
LensType(path: tuple[str, ...], source_type: Type | None, field_type: Type | None)
```

Implementation rules:

1. reject duplicate field names in a single record type;
2. preserve source field order in the type;
3. use structural equality including field order for the first version;
4. make records scalar cells with rank 0;
5. allow `ArrayType(element=RecordType(...), shape=...)` or the closest local
   equivalent if `ArrayType` currently assumes scalar dtypes;
6. do not implement open rows in the first version;
7. do not infer a polymorphic type for `(field hi)` alone unless the current
   typechecker can represent the required constraint;
8. allow field access once the record argument type is known;
9. represent `(record a b c)` as a function whose argument types are inferred
   from use and whose result is `RecordType([(a, Ta), (b, Tb), (c, Tc)])`.

If the existing function/type variable machinery cannot express the constructor
type directly, use a dedicated AST/HIR node for record construction first, then
desugar to a function later.

#### AST, HIR, And Interpreter

Add explicit nodes rather than hiding records inside tuples or dictionaries:

```text
RecordConstructorExpr(field_names, args)
RecordLiteralExpr(fields)
FieldLensExpr(path)
ViewExpr(lens, record)
SetFieldExpr(lens, new_value, record)
OverFieldExpr(lens, function, record)

HIRRecord(fields)
HIRProject(record, field_name)
HIRSetField(record, field_name, value)
HIROverField(record, field_name, function)
```

Interpreter values:

```text
RecordValue(fields: tuple[(name, Value), ...])
LensValue(path: tuple[str, ...])
```

Interpreter behavior:

1. record construction evaluates fields left to right;
2. projection fails with a source-located error if the field is absent;
3. update returns a new record and does not mutate the old one;
4. `over` evaluates the update function on the old field value and stores the
   returned value;
5. nested lenses can be added after shallow fields work;
6. array lifting should be tested by applying projection/update functions to an
   array of records, not by adding special table code.

#### CPU Representation And Lowering

Make a deliberate layout decision before lowering records:

1. scalar record values may lower as MLIR tuples, LLVM structs, or multiple
   parallel SSA values;
2. arrays of records should preferably lower internally as a columnar
   record-of-arrays representation when the array is dense and rectangular;
3. column projection from an array of records should be a view or cheap
   descriptor operation where possible, not a per-row copy;
4. column update should allocate or construct only the changed column plus a
   rebuilt record/table descriptor where possible;
5. the Python/API boundary may materialize rows for display, but backend HIR
   should retain column structure.

Recommended implementation path:

1. interpreter-only records;
2. CPU scalar record construction/projection for non-array records;
3. CPU arrays of records with a simple row-major/AoS representation if that is
   much faster to land;
4. migrate or optimize arrays of records to a columnar SoA representation before
   adding data-frame performance benchmarks;
5. add capability entries that distinguish scalar records, arrays of records,
   projection, update, nested lenses, and row-polymorphic field access.

#### Data-Frame Operations

Treat data frames as arrays of records, not as a separate table object.

First data-frame milestone:

1. construct a table from rows;
2. construct the same table from columns by lifting the record constructor;
3. project a column by applying a field accessor to the table;
4. update a column by applying `set` or `over` to the table;
5. filter rows using a boolean mask and preserve the record element type;
6. print arrays of records with field names shown once per table-like value
   where practical.

Deferred until boxes, ragged arrays, and segmented operations:

1. `filter*` that returns boxed partitions when result sizes differ;
2. partitioning a table into boxed groups;
3. grouping by a key column;
4. summarizing grouped rows with segmented reductions;
5. unique/nub over non-numeric record fields;
6. ragged/nested table columns.

#### Row Polymorphism And Lenses

Closed records are enough for the first implementation. Row polymorphism should
be a later type-system extension.

Closed-record behavior:

1. a function that projects `hi` from `{loc, day, hi}` does not automatically
   typecheck for `{loc, day, month, hi, lo}`;
2. users can still write useful code when the exact record type is known;
3. diagnostics should say whether a field is missing or whether the whole
   record type is too specific.

Later row-polymorphic behavior:

```text
view(field hi) : forall row T. Record(row + hi:T) -> T
over(field hi) : forall row T U. (T -> U) -> Record(row + hi:T) -> Record(row + hi:U)
```

Do not start with this unless the typechecker already has a natural place for
row constraints. Implementing records without row polymorphism is still useful
and provides the test corpus needed to design rows carefully.

#### GPU Policy

The GPU should not implement general records first. Add GPU support only for
profitable columnar patterns after CPU semantics are stable:

1. projection of numeric columns from large dense tables;
2. column-wise `map`/`over` pipelines;
3. row filtering when the predicate and copied columns are GPU-supported;
4. grouped/segmented reductions over numeric columns after segment descriptors
   exist;
5. device-resident table pipelines where transfer cost is amortized.

Unsupported record features should fall back to CPU unless the user explicitly
requires GPU.

#### Acceptance Tests

Add tests in this order:

01. parser round trips for record constructor, record literal, field lens,
    `view`, `set`, and `over`;
02. typechecker rejects duplicate fields and absent fields;
03. interpreter constructs a scalar record and projects each field;
04. interpreter updates a field and leaves the original record unchanged;
05. interpreter constructs a table from rows;
06. interpreter constructs the same table from columns via lifted constructor;
07. interpreter projection over the table returns the expected column;
08. interpreter `over` updates one column using another column in the predicate
    when the exact closed record type is known;
09. CPU parity for every interpreter-supported closed-record feature once CPU
    lowering exists;
10. explicit deferred/rejected tests for row-polymorphic field access, nested
    lenses, grouped tables, and GPU record lowering until those are implemented.

### Workstream 2: Dynamic Shape Runtime Model

Define runtime shapes as first-class values carried through interpreter, HIR,
CPU lowering, and descriptors.

**Guiding principle:** land one vertical slice first — a single op
(rank-1 element-wise map) that compiles once and runs correctly at multiple
runtime sizes on interpreter, CPU, and GPU — before widening op-by-op.

Core decisions:

1. representation of runtime dimensions and shapes;
2. equality/constraint checks at runtime;
3. ownership and allocation of dynamically sized arrays;
4. memref descriptor conventions for dynamic dimensions;
5. interaction with existing static-shape specialization cache.

#### The `DimValue` abstraction (tactical foundation)

The central refactor: code that today does `int(dim.value)` must instead
produce a *dimension value* that is either a compile-time constant **or** a
runtime SSA value (from a descriptor size field or a dimension argument).

```python
# Conceptual model for lowering:
# DimValue = Const(int) | Runtime(ssa_ref, source)
#   source ∈ { DescriptorSize(input_idx, axis), ExplicitDimParam(name) }
```

- [ ] Define `DimValue` in lowering: `Const(int) | Runtime(ssa, source)`.
- [ ] Decide how each Π-bound dimension is supplied at runtime:
  - **Derived dims** (common case): equals an input's axis length → read from
    that input's descriptor `size{axis}`.
  - **Free dims** not derivable from any input → passed as explicit scalar
    arguments (extend the function/kernel ABI with leading `i64` dim params).
- [ ] Build a per-function **dimension environment**: map each index variable
  → `DimValue`, populated from input descriptors / dim args at function entry.
- [ ] Audit and centralize the ~dozens of `int(d.value)` / `StaticDim.value`
  call sites (across `gpu_lowering.py`, `codegen.py`, `_gpu_expr_lowering.py`,
  `lowering/tensor_ops.py`, `lowering/_builder_ops.py`, `lowering/view_ops.py`)
  and route them through a helper that returns either a literal or an SSA
  reference. Keep the constant fast-path when the dim is statically known.

#### Relaxing the specialization gate

- [ ] `compiler.py`: stop hard-rejecting free dimension variables. Instead
  collect them, classify each as derived/free, and record them on the artifact
  so lowering and the runtime can bind them.
- [ ] Keep static specialization as an *option*, but make dynamic the
  supported path when free dims remain.
- [ ] Type/shape checker: where a dimension equality cannot be discharged
  statically, either reject with a clear message or emit a **residual runtime
  check**. Reuse existing index machinery (`dependent_types.py`:
  `index_alpha_equivalent`, `substitute_index`, `instantiate_pi`).
- [ ] Decide and document the policy for unprovable constraints (reject vs
  runtime assert); add a runtime "shape mismatch" error path.

#### CPU lowering for dynamic shapes

MLIR supports dynamic extents natively (`tensor<?x…>`, `memref<?x…>`,
`tensor.empty(%d)`, dynamic `scf.for`). This is largely "replace baked
constants with dim SSA values and `?` in types."

- [ ] Extend typechecker output to preserve residual dimension variables and
  constraints instead of forcing all dimensions to constants.
- [ ] Extend elaborated/core/HIR types with dynamic dimension markers.
- [ ] Teach interpreter to evaluate all shape expressions and constraints at
  runtime.
- [ ] Emit dynamic tensor/memref types (`tensor<?x…>`, `memref<?x…>`)
  driven by the dimension environment.
- [ ] Dynamic allocation: `tensor.empty(%d0, %d1, …)`; compute output buffer
  sizes from runtime dims.
- [ ] Dynamic control flow: `scf.for` with runtime bounds; dynamic
  `tensor.extract_slice` / `tensor.insert_slice` sizes.
- [ ] Runtime/ABI: compiled artifact reads input sizes from descriptors;
  runtime allocates outputs from computed dims; thread explicit dim args if
  used.
- [ ] **Vertical slice (CPU):** dynamic rank-1 element-wise map compiles once,
  runs at multiple sizes.
- [ ] Widen: rank-1 fold, rank-2 element-wise map, then nested map/fold.
- [ ] Keep static-specialized lowering for known dimensions as an optimization.

#### GPU lowering for dynamic shapes

Two sub-paths with different difficulty.

**Descriptor-ABI map/reduce kernels (easier — plumbing exists):**

The descriptor ABI already loads runtime sizes and strides at kernel entry
(`_descriptor_load_lines` in `gpu_lowering.py`). Index math already uses
runtime strides. The gap is that loop bounds, grid/block size, multi-index
"plane" sizes, and `tensor`/`memref` types are baked constants.

- [ ] Replace static `total_size` / `frame_size` / multi-index "plane"
  constants with the already-loaded `%inN_size{axis}` SSA values.
- [ ] Compute grid/block at launch time from runtime input sizes
  (`executor.py` / `runtime.py`), not from `KernelMeta.output_shape` constants.
- [ ] Output descriptor allocation from runtime dims; D2H copy uses runtime
  sizes.

**The general-expr emitter (`_gpu_expr_lowering.py`) — the hard part:**

The current model **statically unrolls** cells (a rank-1 cell of size 3 → 3
`GpuArrayExpr` components; a fold over `dim*K` materialized components).
You cannot unroll a dynamic-length cell.

- [ ] Introduce **loop-based emission** for axes whose extent is dynamic:
  emit an in-kernel `scf.for` over the runtime dimension instead of
  unrolling into N components. Static unrolling where the dim is known
  (common and faster) → a clean fork: *static cell ⇒ unroll, dynamic cell
  ⇒ loop*.
- [ ] Make `GpuArrayExpr` / `GpuReduce` carry a shape that may be dynamic;
  the reduce/store paths handle dynamic counts via loops.
- [ ] Reject (loudly) any dynamic-cell construct not yet covered.
- [ ] **Vertical slice (GPU):** dynamic rank-1 element-wise map + fold compile
  once, run at multiple sizes.
- [ ] Op sweep for dynamic support with compile-once-run-many-sizes parity
  on both CPU and GPU: element-wise map, fold/reduce, scan, matmul, views
  (reverse/rotate/drop/take/transpose/reshape), iota. Element-type sweep
  (f32 **and** i32, bool where relevant) at dynamic sizes.

**Test pattern for dynamic shapes:** "compile once, run at many sizes." Each
dynamic-shape op gets a test that builds a single artifact and executes at
sizes like `{1, 3, 17, 1024, 4096}`, comparing against the interpreter
oracle on both CPU and GPU.

Acceptance criteria:

1. one compiled CPU function accepts different lengths without recompilation;
2. dynamic `map`, `fold`, `scan`, `reshape`, `take`, `drop`, and indexing match
   interpreter results;
3. residual shape-constraint failures produce source-located runtime errors;
4. GPU parity for dense element-wise maps and folds at multiple dynamic sizes.

### Workstream 3: Runtime Boxes And Ragged Arrays

Implement boxes as runtime existential packages.

Depends on Workstream 2: `unbox` recovers runtime dimensions, which need the
dynamic-shape infrastructure to be operational. The front-end is already
plumbed (`SigmaType`, `box`/`unbox` syntax, `BoxExpr`/`UnboxExpr`,
`HIRBox`/`HIRUnbox`) but `box`/`unbox` are currently **type-erased (no
runtime effect)** — sound only because every shape is a known constant.

#### Box ABI design

- [ ] Design the **box ABI**: a box must carry, at runtime, its existentially
  hidden dimension witnesses **plus** the value (a descriptor, or inline
  dims + descriptor). `box` packs; `unbox` reads dims into the dimension
  environment and exposes the inner value's descriptor.
- [ ] Design the **array-of-boxes** layout for heterogeneous nesting. Options:
  (a) array of box-descriptors (pointers + per-element dims), or
  (b) CSR-style flattened values + offsets array. Document trade-offs
  (random access vs compactness vs GPU-friendliness).
- [ ] Define C/runtime ABI for boxed values.

#### Front-end → HIR

- [ ] Replace the type-erasure `HIRBox`/`HIRUnbox` with nodes that
  materialize / recover runtime dimensions.
- [ ] Typechecker: ensure `unbox` binds the hidden dimension variables into
  the runtime dimension environment for its body; verify `Σ`
  formation/elimination rules against the reference semantics
  (`docs/remora-reference/`).

#### Interpreter

- [ ] Make `box` allocate or package runtime witnesses.
- [ ] Make `unbox` bind hidden dimensions and value.
- [ ] Support arrays of boxes with different hidden dimensions.
- [ ] Add display/debug printing for boxed values.

#### CPU lowering

- [ ] Lower `box` (store dims + value per ABI) and `unbox` (load dims as
  runtime `DimValue`s, then lower the body with dynamic shapes).
- [ ] Array-of-boxes construction and indexing/`unbox` of an element.
- [ ] `map` over an array of boxes where each iteration unboxes a different
  shape (the headline ragged operation) — interpreter + CPU first.
- [ ] Support recursive array construction where output sizes are boxed or
  existential.

#### GPU (deferred)

- [ ] Host-orchestrated or restricted forms initially (per-thread divergent
  shapes are hard); reject unsupported forms loudly.
- [ ] Only target GPU boxes if a concrete workload justifies it.

Acceptance criteria:

1. interpreter and CPU compile examples with arrays of vectors of different
   lengths;
2. unboxed dimensions can drive folds/maps inside the unbox body;
3. invalid uses fail with source-located errors.

### Workstream 4: Segmented Reductions And Irregular Primitives

Add segmented reductions after boxes/dynamic shapes have a stable model.

Design choices:

1. surface syntax and type rules;
2. segment descriptor representation: offsets, lengths, or boxed arrays;
3. result shape rules;
4. neutral-element behavior;
5. interaction with rank polymorphism.

Implementation steps:

1. add parser and AST forms;
2. add typechecker rules;
3. implement interpreter semantics;
4. add HIR nodes;
5. lower CPU implementation using loops over segment descriptors;
6. add GPU implementation for large regular segment descriptors as an optional
   accelerator path.

Acceptance criteria:

1. interpreter/CPU parity for varied segment sizes, empty segments, and
   different dtypes;
2. GPU parity for accepted dense numeric segmented reductions;
3. loud GPU fallback/rejection for unprofitable or unsupported irregular cases.

### Workstream 5: Full Higher-Order And Function-Array Semantics

The current CPU path works well for statically resolvable higher-order calls.
Full Remora needs dynamic cases.

Interpreter steps:

1. represent function values uniformly;
2. support function values in arrays;
3. support arrays of functions in function position;
4. complete call-through-variable in map/fold/scan bodies;
5. reconcile `shape`/`rank` of function values with the paper semantics.

CPU strategies:

1. keep monomorphization for statically known callees;
2. add closure records for dynamic values;
3. add function tables for closed sets of callees;
4. use indirect calls only where specialization is impossible or not worth it;
5. specialize arrays of functions when contents are statically known.

GPU policy:

1. no general dynamic higher-order GPU support initially;
2. inline or specialize known helper functions;
3. use CPU fallback for dynamic calls and arrays of functions;
4. consider GPU function tables only if a concrete workload needs them.

Acceptance criteria:

1. interpreter and CPU support MIMD arrays-of-functions examples;
2. dynamic call-through-variable works in compound contexts;
3. GPU scheduler keeps these regions on CPU unless a known-safe specialization
   exists.

### Workstream 6: Missing Surface Forms And Syntax Parity

Implement full paper syntax and remove frontend asymmetries.

Tasks:

1. add `frame` form;
2. add `array` form;
3. add `all` parameter syntax;
4. add Lisp composition support equivalent to ML composition;
5. add syntax tests for ML and Lisp;
6. document exact source semantics.

Acceptance criteria:

1. parsed forms lower to existing or new AST nodes;
2. interpreter and CPU agree on examples from the Remora papers;
3. syntax errors and type errors include source locations.

### Workstream 7: Cost-Aware Type And Schedule System

This is the main research extension.

The goal is not to prove exact runtime. The goal is to estimate enough cost to
make good scheduling choices and explain them.

#### Cost Information To Track

At type/elaboration time:

01. element type;
02. rank;
03. static dimensions where known;
04. symbolic dimensions where dynamic;
05. frame/cell decomposition;
06. result shape;
07. operator kind: map, reduce, scan, view, sort, matmul, segmented operation;
08. purity/effect status;
09. associativity/commutativity where known;
10. memory layout/view status: contiguous, strided, transposed, reversed,
    boxed/ragged;
11. device residency of inputs where known.

At HIR/planning time:

01. estimated element count;
02. estimated arithmetic work;
03. estimated memory reads/writes;
04. temporary allocation size;
05. fusion opportunities;
06. kernel count;
07. host/device transfer size;
08. expected branch divergence or irregularity flag;
09. CPU fallback cost estimate;
10. GPU launch/transfer overhead estimate.

#### Proposed Types

Add an internal cost annotation rather than changing user-facing types first:

```python
@dataclass
class CostShape:
    elem_count: StaticOrSymbolic
    bytes_read: StaticOrSymbolic
    bytes_written: StaticOrSymbolic
    flops: StaticOrSymbolic
    temporary_bytes: StaticOrSymbolic
    irregularity: Literal["regular", "segmented", "ragged", "dynamic-call"]

@dataclass
class ScheduleCandidate:
    backend: Literal["cpu", "gpu"]
    plan_kind: str
    estimated_cost: CostExpression
    requirements: list[CapabilityRequirement]
    fallback_reason: str | None
```

Expose this through compiler internals first. Add user-facing annotations only
after the model proves useful.

#### Scheduling Rules

Initial conservative rules:

1. keep scalar and small-array work on CPU;
2. use GPU for large dense maps/reductions/scans with supported dtypes;
3. prefer GPU when input/output arrays are already device-resident;
4. fuse producer/consumer maps before estimating cost;
5. avoid GPU for dynamic calls, boxes, ragged arrays, and tiny segmented
   reductions;
6. use GPU for regular segmented reductions only when segment count and total
   element count exceed thresholds;
7. keep explicit user override flags for debugging and benchmarking.

The scheduler should produce a plan, not just a backend choice:

```text
source program
  -> typed AST with shape/cost facts
  -> HIR
  -> candidate CPU plan
  -> candidate GPU subplans
  -> schedule graph
  -> execution plan with CPU/GPU boundaries
```

#### Calibration

Cost estimates need calibration from benchmarks:

1. CPU loop throughput by dtype/op;
2. GPU launch overhead;
3. host/device bandwidth;
4. device memory bandwidth;
5. GPU throughput for maps, reductions, scans, sort, matmul;
6. transfer penalties for non-resident arrays;
7. benefits of fusion and device-resident loops.

Store calibrated constants in a versioned profile:

```text
~/.cache/remora/device_profiles/<machine>.json
```

Provide defaults when calibration is unavailable.

#### Explainability

Every scheduled program should be explainable:

```text
map/fold pipeline over 10,000,000 f32 elements
  CPU estimate: 3.8 ms
  GPU estimate: 0.7 ms kernel + 0.0 ms transfer (device-resident)
  selected: GPU fused_map_reduce
```

For rejected GPU regions:

```text
unbox body contains ragged array fold
  GPU rejected: runtime box payload layout unsupported
  selected: CPU dynamic_box_loop
```

## Phased Implementation Plan

### Phase 0: Stabilize Planning Infrastructure

Purpose: make future work auditable and reduce technical debt before the
full-language implementation begins.

Tasks:

01. create an executable capability matrix;
02. add a backend route registry for CPU/GPU lowering candidates;
03. migrate the current `codegen.py` GPU dispatch cascade behind the registry
    without changing behavior;
04. add `--explain-lowering`;
05. add structured, source-located unsupported-feature errors;
06. update docs to consume or mirror the matrix;
07. add parametrized tests for support claims;
08. add a generated differential-test harness for a small dense subset;
09. extract repeated GPU text-emission idioms into shared helpers;
10. decide whether to delete the disabled MLIR builder path or keep it as a
    validation-only backend;
11. reconcile stale doc/status claims around GPU dtype support, dynamic-shape
    wording, pairs, im2col/col2im, and f64 support;
12. add record/data-frame capability entries with all entries marked unsupported
    except any already-existing pair/tuple-adjacent behavior;
13. define cost annotation data structures without scheduling decisions yet.

Exit criteria:

1. every backend selection has an explainable reason;
2. docs and tests agree on operation/backend support;
3. unsupported GPU paths remain loud and source-located;
4. the route registry can show which lowering path accepted or rejected a HIR
   program;
5. new dynamic-shape work has a stable place to register capabilities and
   unsupported-feature reasons;
6. the small generated test suite runs interpreter/CPU parity and GPU parity
   where supported.

### Phase 0.5: Lowering Modularization For Dynamic Shapes

Purpose: prepare the CPU and GPU lowering code for dynamic dimensions without
mixing refactoring and semantic changes in the same patches.

Tasks:

1. split large lowering files by operation family where it reduces review risk:
   maps, folds/scans, views, calls/functions, boxes, and runtime allocation;
2. isolate descriptor ABI helpers from operation-specific code;
3. isolate static-shape assumptions behind helper APIs;
4. add tests that assert static-shape lowering is unchanged after extraction;
5. add internal assertions documenting where ranks/dimensions must currently be
   static;
6. move repeated CPU text-lowering fragments into small emitters before adding
   dynamic memref variants.

Exit criteria:

1. static dense-core test behavior is unchanged;
2. dynamic-shape implementation points are localized;
3. reviewers can identify which files own each operation family;
4. the compiler can report when a static-only assumption blocks dynamic lowering.

### Phase 1: Full Interpreter Surface

Purpose: create the full-language oracle.

Tasks:

01. implement `frame`, `array`, and `all`;
02. fix composition parity across ML/Lisp;
03. implement interpreter dynamic shapes consistently;
04. implement runtime boxes in interpreter;
05. implement arrays of boxes and ragged examples;
06. implement segmented reductions in interpreter;
07. implement closed ordered records in the interpreter;
08. implement first-class record constructors, field accessors, shallow `view`,
    shallow `set`, and shallow `over`;
09. implement table construction from rows and from lifted column arguments;
10. add interpreter tests for column projection and column update over arrays of
    records;
11. implement dynamic higher-order/function-array semantics in interpreter.

Exit criteria:

1. examples from the Remora papers run in the interpreter;
2. simple examples from `Records with Rank Polymorphism` run in the
   interpreter without row polymorphism;
3. dynamic/ragged/segmented programs have tests;
4. interpreter behavior is documented as the semantic baseline.

### Phase 2: CPU Dynamic Shapes

Purpose: compile runtime-sized dense programs.

Tasks:

1. preserve symbolic dimensions through typechecker/elaboration/HIR;
2. emit dynamic memrefs and runtime loops;
3. add runtime allocation for dynamic results;
4. add residual shape checks;
5. support dynamic dense `map`, `fold`, `scan`, views, indexing, and shape
   queries;
6. prepare the CPU value ABI so non-numeric scalar values, including records,
   have an explicit unsupported or supported path instead of falling through
   scalar dtype assumptions;
7. keep static specialization as an optimization path.

Exit criteria:

1. one compiled CPU function handles multiple input lengths;
2. dynamic dense CPU results match interpreter;
3. failure diagnostics are source-located.

### Phase 2.5: CPU Closed Records And Simple Data Frames

Purpose: compile the closed-record subset before adding ragged/grouped table
operations.

Tasks:

1. lower scalar record construction and field projection on CPU;
2. lower shallow `set` and `over` on CPU;
3. support arrays of records using a simple representation first if needed;
4. decide and document the long-term dense table layout: array-of-structs,
   struct-of-arrays, or an internal conversion between the two;
5. make column projection from dense arrays of records cheap in the chosen
   representation;
6. compile table construction from row literals;
7. compile table construction from column arrays via lifted record constructor;
8. compile row filtering for arrays of records when result shape is representable
   by existing dense/dynamic mechanisms;
9. add CPU-vs-interpreter parity tests for every closed-record feature.

Exit criteria:

1. interpreter and CPU agree for scalar records, arrays of records, projection,
   update, and simple row filtering;
2. the capability matrix distinguishes implemented closed-record behavior from
   deferred row polymorphism, nested lenses, grouping, and GPU lowering;
3. display and Python/API materialization are good enough for test debugging.

### Phase 3: CPU Boxes, Ragged Arrays, And Segments

Purpose: complete irregular data on CPU.

Tasks:

1. implement boxed runtime ABI;
2. compile `box`/`unbox`;
3. compile arrays of boxes;
4. compile ragged examples;
5. compile segmented reductions;
6. compile `filter*`-style boxed partitions for arrays of records;
7. compile grouped-table examples whose groups are represented as arrays of
   boxes or segment descriptors;
8. compile grouped summaries over numeric record fields using segmented
   reductions;
9. compile recursive array builders with dependent result sizes where practical.

Exit criteria:

1. interpreter and CPU agree for ragged arrays and segmented reductions;
2. interpreter and CPU agree for boxed table partitions and grouped summaries;
3. memory ownership and lifetime are tested;
4. boxed values are usable from Python APIs.

### Phase 4: CPU Dynamic Higher-Order Completion

Purpose: complete function-valued semantics on CPU.

Tasks:

1. add closure records and/or function tables;
2. compile call-through-variable in compound contexts;
3. compile arrays of functions;
4. support MIMD function-position examples;
5. keep monomorphization for statically resolvable cases;
6. add cost metadata for dynamic calls so scheduler keeps them on CPU.

Exit criteria:

1. CPU and interpreter agree for dynamic HOF examples;
2. static HOF programs still use monomorphized fast paths;
3. GPU scheduler rejects/falls back for dynamic calls unless specialized.

### Phase 5: Selective GPU Acceleration Scheduler

Purpose: accelerate common patterns without full GPU parity.

Tasks:

1. add cost estimation over typed AST/HIR;
2. add plan graph with CPU and GPU regions;
3. implement GPU profitability thresholds;
4. fuse dense map/view/reduction pipelines before scheduling;
5. preserve device residency across iterative loops;
6. add CPU fallback at schedule boundaries;
7. add user controls: prefer CPU, prefer GPU, require GPU, explain schedule.

Exit criteria:

1. dense numeric workloads choose GPU only when estimated profitable;
2. small workloads stay on CPU;
3. device-resident workloads avoid unnecessary transfers;
4. scheduler decisions are tested and explainable.

### Phase 6: GPU Pattern Expansion

Purpose: improve acceleration coverage for useful patterns, not full semantics.

Priority patterns:

01. dynamic-shape dense maps/reductions/scans when descriptors provide sizes;
02. segmented reductions for regular large segment descriptors;
03. numeric column projection/update for dense arrays of records when the
    internal representation is columnar or cheaply projectable;
04. row filtering for dense records when predicate and copied columns are
    GPU-supported;
05. grouped numeric summaries when segment descriptors are regular and large
    enough to beat CPU;
06. FFT and convolution pipelines after complex/FFT work;
07. AD-generated dense numeric kernels;
08. stencil/PDE kernels;
09. sort/grade scale limits;
10. scatter-add atomic path;
11. fused multi-output kernels.

Exit criteria:

1. each GPU pattern has numeric parity tests against interpreter/CPU;
2. each pattern has benchmark evidence for profitability thresholds;
3. unsupported full-language constructs fall back to CPU.

### Phase 7: Calibration, Benchmarks, And Research Evaluation

Purpose: make the scheduling type-system idea measurable.

Tasks:

1. build calibration benchmark suite;
2. compare estimated vs measured runtime;
3. compare scheduled Remora programs against CPU-only Remora, NumPy, JAX, and
   Futhark where appropriate;
4. evaluate fusion and device-residency decisions;
5. document cases where static cost facts are insufficient;
6. decide whether user-visible cost annotations are useful.

Exit criteria:

1. scheduler improves representative workloads without hurting small programs;
2. estimates are accurate enough to choose the right backend in common cases;
3. research writeup can describe the type/cost/schedule model.

## Testing Strategy

### Semantic Tests

1. interpreter golden tests for every full-language feature;
2. CPU-vs-interpreter parity for every compiled feature;
3. dynamic-shape tests with multiple runtime sizes per compiled artifact;
4. closed-record tests for scalar records, arrays of records, projection,
   update, and row filtering;
5. ragged/boxed tests with varied hidden dimensions;
6. boxed table partition tests with varied group sizes;
7. segmented reduction tests with empty, singleton, uneven, and large segments;
8. dynamic higher-order tests including arrays of functions.

### GPU Tests

1. numeric parity for every GPU-accepted pattern;
2. rejected-not-silent tests for unsupported constructs;
3. scheduler tests proving unprofitable programs stay on CPU;
4. device-resident tests proving transfers are avoided;
5. benchmark-backed threshold tests where practical.

### Generated Tests

Add property-based or generated differential tests in stages:

1. small dense static programs;
2. dynamic-shape dense programs;
3. closed-record and simple data-frame programs;
4. boxed/ragged programs;
5. boxed table partitions and grouped summaries;
6. segmented reductions;
7. higher-order programs with finite closed function sets.

The interpreter remains the oracle.

## Documentation Deliverables

1. update `USER_GUIDE.md` for full-language features;
2. document dynamic shapes and runtime boxes;
3. document records, field lenses, and data-frame idioms;
4. document CPU/GPU scheduling behavior;
5. document `--explain-lowering` and `--explain-schedule`;
6. generate or validate backend support tables from the capability matrix;
7. add examples for records/data frames, ragged arrays, segmented reductions,
   dynamic HOFs, and scheduled GPU acceleration.

## Suggested Milestones

### Milestone A: Auditable Current Compiler

Capability matrix, backend route registry, explain lowering,
support-matrix tests, structured diagnostics, generated dense-subset tests, and
clean docs.

### Milestone A.5: Modular Lowering Base

Large lowering modules have clear operation-family ownership, shared descriptor
and text-emission helpers, localized static-shape assumptions, and unchanged
static dense-core behavior. This milestone should land before dynamic CPU
lowering work begins.

### Milestone B: Full Interpreter

All full-language features run in the interpreter. This is the semantic target.

### Milestone C: Dynamic Dense CPU

Compiled CPU supports runtime-sized dense arrays.

### Milestone C.5: Closed Records And Simple Data Frames

Interpreter and CPU support ordered closed records, first-class constructors,
field projection, shallow field update, arrays of records, table construction
from rows, table construction from columns, and simple row filtering.

### Milestone D: Irregular CPU

Compiled CPU supports boxes, ragged arrays, segmented reductions, and recursive
dependent-size array builders, including boxed table partitions and grouped
numeric summaries over record fields.

### Milestone E: Full Higher-Order CPU

Compiled CPU supports dynamic higher-order calls and arrays of functions.

### Milestone F: Cost-Aware Scheduler

Compiler emits CPU/GPU schedule plans with explainable cost estimates and CPU
fallback.

### Milestone G: Profitable GPU Acceleration

GPU accelerates dense numeric, device-resident, AD-generated, stencil, FFT, and
regular segmented workloads where benchmarks show it beats CPU.

## Risks

### Dynamic Shape Scope Creep

Dynamic shapes affect every layer. Mitigation: start with rank-1/rank-2 dense
arrays and a small operator set, then expand.

### GPU Static-Unroll Model (`_gpu_expr_lowering.py`)

The static cell-unrolling model is pervasive. The static-vs-dynamic emission
fork (loop-based for dynamic cells) is the single largest risk. Prototype on
the narrow vertical slice before broad rollout.

### Constant-Site Sprawl

The `int(d.value)` pattern appears across dozens of sites in `gpu_lowering.py`,
`codegen.py`, `_gpu_expr_lowering.py`, `lowering/tensor_ops.py`,
`lowering/_builder_ops.py`, and `lowering/view_ops.py`. The `DimValue` refactor
must be done centrally or regressions will leak. Golden-MLIR tests (static dims
must remain byte-identical) will catch drift.

### Shape-Constraint Soundness

Wrong static discharge of a dimension equality is a silent miscompile. Prefer
a runtime check when not provably equal.

### Box Runtime Complexity

Boxes can become a second object system. Mitigation: define a minimal ABI for
dimension witnesses and payload descriptors before optimizing.

### CPU Performance Regression

Dynamic paths may slow static programs. Mitigation: preserve static-specialized
paths and use dynamic lowering only when dimensions are genuinely dynamic.

### GPU Scheduler Wrong Choices

A bad cost model can make programs slower. Mitigation: conservative thresholds,
calibration, device-residency tracking, and user overrides.

### Record Representation Lock-In

Records can be implemented quickly with row-major arrays of structs, but
data-frame workloads and GPU acceleration strongly prefer columnar structure.
Mitigation: keep semantic HIR distinct from physical layout, document the chosen
CPU layout, and add projection/update benchmarks before declaring the record
backend complete.

### Higher-Order Compilation Complexity

General function values are hard to compile efficiently. Mitigation: keep
monomorphization for static cases, use function tables for closed dynamic sets,
and use generic indirect calls only as a fallback.

## Research Questions

1. Can Remora's rank/cell/frame typing provide useful static cost bounds?
2. Can a type-directed scheduler reliably decide CPU vs GPU without profiling
   every program?
3. How should cost types represent symbolic dimensions and residual runtime
   constraints?
4. What is the right abstraction for device residency in a mostly functional
   array language?
5. Can irregular features such as boxes and segmented reductions expose enough
   structure for profitable GPU acceleration?
6. Can ordered structural records plus rank-polymorphic lifting provide a small
   typed data-frame core without a separate table language?
7. How much row polymorphism is needed before record lenses become ergonomic
   enough for exploratory data-frame programming?
8. How much of JAX/Futhark-style scheduling can be achieved while preserving
   Remora's explicit rank-polymorphic semantics?

## File Reference Map

Where dynamic-shape and box implementation work lands:

| Area                         | Files                                                                            |
| ---------------------------- | -------------------------------------------------------------------------------- |
| Dependent types / index vars | `remora/index.py`, `remora/dependent_types.py`, `remora/types.py`                |
| Specialization gate          | `remora/compiler.py` (`specialize_top_level_function`, `free_type_index_vars`)   |
| Typechecker                  | `remora/typechecker.py`                                                          |
| CPU/dense lowering           | `remora/lowering/tensor_ops.py`, `module.py`, `_builder_ops.py`, `view_ops.py`   |
| GPU lowering                 | `remora/gpu_lowering.py`, `remora/codegen.py`, `remora/_gpu_expr_lowering.py`    |
| Descriptor ABI / runtime     | `remora/abi.py`, `remora/runtime.py`, `remora/remora_rt.c`, `remora/executor.py` |
| Box/unbox front-end          | `remora/ast_nodes.py`, `remora/lisp_reader.py`, `remora/hir.py`                  |
| Reference semantics          | `docs/remora-reference/`                                                         |

Where records and data-frame work lands:

| Area                    | Files                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------- |
| Syntax                  | `remora/grammar.lark`, `remora/parser.py`, `remora/lisp_reader.py`                                      |
| AST and HIR             | `remora/ast_nodes.py`, `remora/hir.py`, `remora/elaborate.py`                                           |
| Types                   | `remora/types.py`, `remora/typechecker.py`                                                              |
| Interpreter             | `remora/interpreter.py`                                                                                 |
| CPU lowering            | `remora/lowering/tensor_ops.py`, `remora/lowering/module.py`, `remora/runtime.py`, `remora/remora_rt.c` |
| GPU scheduling/lowering | `remora/capabilities.py`, `remora/codegen.py`, `remora/gpu_lowering.py`, `remora/_gpu_expr_lowering.py` |
| Display/API             | `remora/interpreter.py`, `remora/runtime.py`, Python API wrappers                                       |
| Tests                   | parser/typechecker tests, interpreter execution tests, compiled CPU parity tests, acceptance manifest   |

## Non-Goals

1. Full GPU implementation of every Remora feature.
2. GPU support for general dynamic higher-order dispatch unless a concrete use
   case demands it.
3. GPU support for arbitrary ragged boxed data in the first full-language
   implementation.
4. Full row-polymorphic records or general lenses in the first record
   implementation.
5. A separate Pandas-like table object independent of ordinary Remora arrays.
6. Exact static runtime prediction.
7. User-facing cost annotations before internal scheduling has proven useful.

## Practical First Step

Start with Phase 0. The capability matrix and `--explain-lowering` are small
relative to dynamic shapes, but they make every later change safer. They also
provide the substrate for the cost-aware scheduler: before the compiler can
choose profitable GPU schedules, it must be able to state what each backend can
compile, what it costs approximately, and why a particular plan was selected.
