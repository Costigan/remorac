# Plan: Dynamic Shapes, then Boxed Arrays

Status legend: `- [ ]` not started · `- [x]` done · `- [~]` in progress.

This is the strategic roadmap for moving RemoraC from its **dense, static-shape
core** toward **full Remora**. The two language-completeness pillars are *dynamic
shapes* (runtime array dimensions, the `Π` / dependent-product side) and *boxed
arrays* (irregular / ragged nested data, the `Σ` / dependent-sum side). Dynamic
shapes comes first because boxes are built on top of it (a box is "a value plus
its runtime dimensions").

Two smaller items run **before** this work and are tracked here only for
sequencing — they are independent and low-risk:

- [ ] **GPU buffer arena** (see `FUTURE_WORK.md` → "GPU buffer arena"): a device
  memory arena that persists across `RemoraExecutor.execute()` calls and recycles
  buffers by size class, eliminating the per-call `alloc → H2D → launch → D2H →
  free` tax for iterative workloads.
- [ ] **Heat-equation example**: a new iterative stencil example (1-D or 2-D heat
  diffusion) under `examples/`, exercised end-to-end on interpreter / CPU / GPU.
  Doubles as a realistic driver for the buffer arena and, later, a natural
  dynamic-shape demo (one definition, any grid size).

---

## Strategic direction

### Why this order
1. **Near-term (now):** GPU buffer arena + heat-equation example. Small, concrete,
   high bang-for-buck; the example becomes a recurring test driver for everything
   that follows.
2. **Dynamic shapes:** the single biggest gap between "static dense core" and
   "full Remora", and the prerequisite for boxes. Highest leverage — it
   strengthens benchmarks, the PL-community doc, and removes the per-shape
   recompile that makes the compiler feel like a fixed-size DSL.
3. **Boxed arrays:** irregular nested data, built directly on runtime dimensions.

### Where we are (the gap, concretely)
- The type **front-end is already dependent**: dimension/index variables and
  shape expressions (`remora/index.py`: `DimExpr`, `ShapeExpr`, `IndexBinder`),
  `Π` types via `define/pi`, `Σ`/`SigmaType` and `box`/`unbox`
  (`remora/types.py`, `remora/dependent_types.py`).
- But the **lowering boundary is static-only**: `compile_function_source`
  *specializes* (monomorphizes) each function to concrete dimensions and then
  **refuses** to lower anything with a free dimension variable
  (`remora/compiler.py`: raises `compiled function … has unspecialized index
  variables`). Every artifact bakes sizes in — `tensor<3x2x2xf32>`, constant
  `scf.for` bounds, static buffer sizes, fixed GPU grid/block, and descriptor
  ranks all come from `StaticDim` constants.

### The encouraging part (plumbing that already exists)
- The **descriptor ABI already carries runtime sizes and strides** (aligned ptr +
  offset + per-axis sizes + per-axis strides). On GPU, `_descriptor_load_lines`
  (`remora/gpu_lowering.py`) **already loads** `%inN_size{axis}` and
  `%inN_stride{axis}` as SSA values at kernel entry, and index math already uses
  the runtime strides. The gap is that loop bounds, grid/block, multi-index
  "plane" sizes, and `tensor`/`memref` types are computed from **baked constants**
  instead of those already-loaded runtime sizes. Much of dynamic shapes on the
  map/reduce kernels is "consume `%inN_size{axis}` instead of a constant."
- MLIR/`linalg` natively support dynamic extents (`tensor<?x…>`, `memref<?x…>`,
  `tensor.empty(%d)`, dynamic loops), so the **target is ready**.

### Non-goals / explicitly deferred
- JIT shape specialization (compile-and-cache per concrete shape) is a *separate*
  item; dynamic shapes is the "one kernel for all sizes" approach. They can
  coexist later but are not coupled here.
- Full ragged-array support **on GPU** with per-thread divergent shapes is
  deferred; first box milestones are interpreter/CPU and host-orchestrated.
- Performance parity of dynamic kernels with specialized ones is not a Phase-1
  goal (dynamic kernels may be slower; correctness first).

### Guiding principle: one vertical slice first
Before broad coverage, land a **single op end-to-end** (rank-1 element-wise map)
that compiles **once** and runs correctly at **multiple runtime sizes** on
interpreter, compiled CPU, and GPU. That proves the whole vertical
(typecheck → relax-gate → lower → runtime allocation → execute) and de-risks the
rest. Only then widen op-by-op.

### New testing pattern (applies throughout)
"**Compile once, run at many sizes.**" Each dynamic-shape op gets a test that
builds a single artifact and executes it at sizes like `{1, 3, 17, 1024, 4096}`,
comparing against the interpreter oracle — on **both** CPU and GPU. Per
`AGENTS.md` coverage rules, GPU numeric parity at non-trivial dynamic sizes is
mandatory, and unsupported dynamic cases must fail **loudly**, never silently.

---

## Part 1 — Dynamic shapes

### Phase 0 — Representation & the `DimValue` abstraction
The central refactor: code that today does `int(dim.value)` must instead produce
a *dimension value* that is either a compile-time constant **or** a runtime SSA
value (from a descriptor size field or a dimension argument).

- [ ] Define a `DimValue` notion in lowering: `Const(int)` | `Runtime(ssa, source)`,
  where `source` records how to obtain it (descriptor `sizeK` of input *i*, or an
  explicit scalar dim parameter).
- [ ] Decide how each Π-bound dimension is supplied at runtime:
  - [ ] **Derived dims** (the common case): the dim equals some input array's
        axis length → read from that input's descriptor `size{axis}`.
  - [ ] **Free dims** not derivable from any input → passed as explicit scalar
        arguments (extend the function/kernel ABI with leading `i64` dim params).
- [ ] Build a per-function **dimension environment**: map each index variable name
  → `DimValue`, populated from input descriptors / dim args at function entry.
- [ ] Audit and centralize the ~dozens of `int(d.value)` / `StaticDim.value`
  call sites (`grep` shows them across `gpu_lowering.py`, `codegen.py`,
  `_gpu_expr_lowering.py`, `lowering/tensor_ops.py`, `lowering/_builder_ops.py`,
  `lowering/view_ops.py`) and route them through a helper that returns either a
  literal or an SSA reference. (Keep the constant fast-path when the dim is
  statically known.)

### Phase 1 — Relax the specialization gate + checker support
- [ ] `compiler.py`: stop hard-rejecting free dimension variables. Instead collect
  them, classify each as derived/free (Phase 0), and record them on the artifact
  so lowering and the runtime can bind them.
- [ ] Keep static specialization as an *option* (fast path / hot shapes), but make
  "dynamic" the supported path when free dims remain.
- [ ] Type/shape checker: where a dimension equality cannot be discharged
  statically (e.g. `map (+) xs ys` with `xs : Array n`, `ys : Array m`), either
  reject with a clear message or emit a **residual runtime check**. Reuse the
  existing index machinery (`dependent_types.py`: `index_alpha_equivalent`,
  `substitute_index`, `instantiate_pi`).
- [ ] Decide and document the policy for unprovable constraints (reject vs runtime
  assert); add a runtime "shape mismatch" error path.

### Phase 2 — CPU (dense) lowering for dynamic shapes
Widen `lowering/` to emit dynamic-extent MLIR. MLIR supports it natively, so this
is largely "replace baked constants with dim SSA values and `?` in types."

- [ ] Emit dynamic tensor/memref types (`tensor<?x…>`, `memref<?x…, strided<…>>`)
  driven by the dimension environment.
- [ ] Dynamic allocation: `tensor.empty(%d0, %d1, …)` / dynamic `bufferization`;
  compute output buffer sizes from runtime dims.
- [ ] Dynamic control flow: `scf.for` with runtime bounds; dynamic
  `tensor.extract_slice` / `insert_slice` sizes (generalize the recently-fixed
  loop-map in `tensor_ops.py`).
- [ ] Runtime/ABI (`runtime.py`, `abi.py`, `remora_rt.c`): the compiled `.so`
  reads input sizes from descriptors; the runtime allocates outputs from the
  computed runtime dims; thread explicit dim args if used.
- [ ] **Vertical slice (CPU):** dynamic rank-1 element-wise map compiles once,
  runs at multiple sizes, matches the interpreter.
- [ ] Widen: rank-1 fold, rank-2 element-wise map, then nested map/fold.

### Phase 3 — GPU lowering for dynamic shapes
Two sub-paths with different difficulty.

**3a. Descriptor-ABI map/reduce kernels (easier — plumbing exists):**
- [ ] Replace static `total_size` / `frame_size` / multi-index "plane" constants
  in `build_descriptor_abi_general_map_gpu_module` with the already-loaded
  `%inN_size{axis}` SSA values (`_descriptor_load_lines`, `_multi_index_lines`,
  `_linear_index_lines`).
- [ ] Compute grid/block at launch time from runtime input sizes
  (`executor.py` / `runtime.py`), instead of from `KernelMeta.output_shape`
  constants. `KernelMeta` grows a notion of dynamic dims.
- [ ] Output descriptor: allocate device output from runtime dims; D2H copy uses
  runtime sizes.

**3b. The general-expr emitter (`_gpu_expr_lowering.py`) — the hard part:**
The current model **statically unrolls** cells (a rank-1 cell of size 3 → 3
`GpuArrayExpr` components; a fold over `dim*K` materialized components). You
**cannot unroll a dynamic-length cell**.
- [ ] Introduce a **loop-based emission** for axes whose extent is dynamic: emit
  an in-kernel `scf.for` over the runtime dimension instead of unrolling into N
  components. Keep static unrolling where the dim is statically known (common and
  faster) → a clean fork: *static cell ⇒ unroll (existing), dynamic cell ⇒ loop
  (new)*.
- [ ] Make `GpuArrayExpr` / `GpuReduce` carry a shape that may be dynamic, and the
  reduce/store paths handle dynamic counts via loops.
- [ ] Reject (loudly) any dynamic-cell construct not yet covered by the loop path.
- [ ] **Vertical slice (GPU):** dynamic rank-1 element-wise map + fold compile
  once, run at multiple sizes, match the oracle.

### Phase 4 — Coverage, parity, and hardening
- [ ] Op sweep for dynamic support, each with "compile-once-run-many-sizes"
  parity on CPU **and** GPU: element-wise map, fold/reduce, scan, matmul, views
  (reverse/rotate/drop/take/transpose/reshape), iota.
- [ ] Element-type sweep (f32 **and** i32, bool where relevant) at dynamic sizes.
- [ ] Loud-rejection tests for dynamic constructs still unsupported
  (rejected-not-silent).
- [ ] Update `AGENTS.md` testing notes with the dynamic-size parity pattern.
- [ ] Convert at least one real example (heat equation) to a single dynamic-shape
  definition runnable at any grid size.

### Phase 1 definition of done
A function with a `Π`-bound dimension (e.g. `(define/pi ((n Dim)) (scale [xs
(Array Float n)] (Array Float n)) (map (* 2.0) xs))`) compiles to **one** artifact
and runs correctly at several runtime sizes on the interpreter, the compiled CPU
backend, and GPU, with numeric parity tests committed.

---

## Part 2 — Boxed arrays (`Σ` / ragged nested data)

Depends on Part 1: `unbox` recovers runtime dimensions, which only have meaning
once dynamic shapes exist. The front-end is already plumbed (`SigmaType`,
`box`/`unbox` syntax, `BoxExpr`/`UnboxExpr`, `HIRBox`/`HIRUnbox`) but `box`/`unbox`
are currently **type-erased (no runtime effect)** — sound only because every
shape is a known constant today.

### Phase B0 — Box runtime representation & ABI
- [ ] Design the **box ABI**: a box must carry, at runtime, its existentially
  hidden dimension witnesses **plus** the value (a descriptor, or inline
  dims + descriptor). `box` packs; `unbox` reads dims into the dimension
  environment (Part 1) and exposes the inner value's descriptor.
- [ ] Design the **array-of-boxes** layout for heterogeneous nesting. Options to
  evaluate: (a) array of box-descriptors (pointers + per-element dims), or
  (b) CSR-style flattened values + offsets array. Pick one; document the
  trade-offs (random access vs compactness vs GPU-friendliness).

### Phase B1 — Front-end → HIR (make box/unbox real)
- [ ] Replace the type-erasure `HIRBox`/`HIRUnbox` with nodes that materialize /
  recover runtime dimensions.
- [ ] Typechecker: ensure `unbox` binds the hidden dimension variables into the
  runtime dimension environment for its body; verify `Σ` formation/elimination
  rules against the reference semantics (`docs/remora-reference/`).

### Phase B2 — Lowering box / unbox
- [ ] **CPU first:** lower `box` (store dims + value per the ABI) and `unbox`
  (load dims as runtime `DimValue`s, then lower the body with dynamic shapes).
- [ ] Array-of-boxes construction and indexing/`unbox` of an element.
- [ ] `map` over an array of boxes where each iteration unboxes a different shape
  (the headline ragged operation) — interpreter + CPU first.
- [ ] **GPU:** host-orchestrated or restricted forms initially (per-thread
  divergent shapes are hard); reject unsupported forms loudly.

### Phase B3 — Coverage & tests
- [ ] Parity tests for `box`/`unbox`, array-of-boxes build/index, and ragged
  `map`/`fold`, against the interpreter oracle.
- [ ] A realistic ragged example under `examples/` (e.g. variable-length
  sequences / jagged rows).

### Boxes definition of done
A program that builds an array of differently-shaped boxed arrays, maps a
rank-polymorphic function over it (unboxing each), and reduces the results runs
correctly on the interpreter and compiled CPU, with parity tests committed.

---

## Cross-cutting concerns
- [ ] **AD interaction:** gradients of dynamic-shape functions (the AD source
  transform and tape) must thread runtime dims; add dynamic-size AD parity.
- [ ] **Caching:** a dynamic artifact is shape-independent — cache key drops the
  shape signature; ensure `cache.py` keys correctly so one compile serves all
  sizes.
- [ ] **Diagnostics:** clear messages for unprovable shape constraints and for
  unsupported dynamic constructs (ties into the `FUTURE_WORK.md` "better error
  messages" item).
- [ ] **Performance note:** document that dynamic kernels may be slower than
  specialized ones (no unrolling / size-specific optimization); the JIT
  specialization item remains the complement for hot shapes.

## Key risks
- **GPU static-unroll model** (`_gpu_expr_lowering.py`) is pervasive; the
  static-vs-dynamic emission fork (Phase 3b) is the largest single risk and
  should be prototyped on the narrow vertical slice before broad rollout.
- **Constant-site sprawl:** the `int(d.value)` pattern is everywhere; the
  `DimValue` refactor (Phase 0) must be done centrally or regressions will leak.
  Lean on golden-MLIR tests (static dims must remain byte-identical) to catch
  drift.
- **Shape-constraint soundness:** wrong static discharge of a dimension equality
  is a silent miscompile; prefer a runtime check when not provably equal.

## Reference map (where the work lands)
| Area | Files |
|------|-------|
| Dependent types / index vars | `remora/index.py`, `remora/dependent_types.py`, `remora/types.py` |
| Specialization gate | `remora/compiler.py` (`specialize_top_level_function`, `free_type_index_vars`) |
| Typechecker | `remora/typechecker.py` |
| CPU/dense lowering | `remora/lowering/tensor_ops.py`, `module.py`, `_builder_ops.py`, `view_ops.py` |
| GPU lowering | `remora/gpu_lowering.py`, `remora/codegen.py`, `remora/_gpu_expr_lowering.py` |
| Descriptor ABI / runtime | `remora/abi.py`, `remora/runtime.py`, `remora/remora_rt.c`, `remora/executor.py` |
| Box/unbox front-end | `remora/ast_nodes.py`, `remora/lisp_reader.py`, `remora/hir.py` |
| Reference semantics | `docs/remora-reference/` |
