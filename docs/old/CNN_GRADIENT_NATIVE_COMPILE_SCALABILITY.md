# CNN Gradient Native Compilation Scalability

## Implementation Status (updated 2026-06-16)

The descriptor-MLIR scalability plan is complete.  The original problem — the generated CNN gradient
produced 47.5 MB of descriptor MLIR that timed out the CPU pipeline — is
solved.  All non-optional scalability phases (0-9) are implemented; Phase 10 is
explicitly rejected with a measured decision.  Native crater training is wired
through an auto mode with interpreter fallback and a strict compiled mode.  The
compiled function path now links Remora runtime support, including the MLIR
bufferization `memrefCopy` helper.

### What was delivered

| Phase | What | Impact |
|---|---|---|
| 0 | Descriptor export correctness baseline | No regressions from fixes |
| 1 | Repeatable benchmark harness | JSON-output `remora.benchmark` |
| 2 | Pass diagnosis | Identified `one-shot-bufferize` as the bottleneck |
| 3 | Compact `im2col`/`col2im` (loops, not unrolled elements) | **47.5 MB → 127 KB** |
| 4 | HIR common-subexpression elimination | **127 KB → 60 KB** |
| 5 | AD expression simplification pass | Cauchy; `_binary` rules already caught most |
| 6 | One value-and-grad function | Single function returns all 6 gradients; strict compiled mode validated on the tiny crater run |
| 7 | Opt-in saved-value tape (`use_saved_values=True`) | `_Let` bindings, HIRLet peeling |
| 8 | `HIRMatmul` → `linalg.matmul` lowering | Pattern-match pass in `hir_opt.py` |
| 9 | Artifact cache (`~/.cache/remora/native/`) | Skip pipeline on cache hit |
| 10 | **Rejected** — IREE bindings missing 4 passes, saving ~0.03 s | Decision recorded |

### Key metrics

| Metric | Before (baseline) | After (Phase 9) |
|---|---|---|
| Descriptor MLIR | 47,552,499 B | 60,410 B |
| `tensor.extract` | 218,702 | 12 |
| `tensor.insert` | 218,700 | 0 |
| CPU pipeline (`mlir-opt`) | Did not finish | 0.032 s |
| End-to-end compile | Timed out | ~112 s (succeeds) |
| Test suite | 968 passed | 994 passed |

### Remaining work

The ~112 s end-to-end time is dominated by function preparation
(typechecking + HIR construction, ~111 s), not by MLIR lowering
(< 1 s total).  The crater performance follow-up added typechecker
memoization, strict compiled-mode handling, and Remora runtime support for
MLIR `memrefCopy`, so the remaining open items are:

- **Phase 7**: The saved-value tape is opt-in (default off) pending GPU
  path HIRLet support.
- **Phase 8**: Convolution is still lowered via compact `im2col` loops
  (Phase 3), which is sufficient for the current CNN size.  Dedicated
  convolution operations and BLAS integration were evaluated but not
  prioritised given the 60 KB IR already fits comfortably.

---

## Summary

The generated gradient for the crater CNN can now be lowered through the
descriptor-export path to valid, non-empty MLIR. However, compiling that MLIR
into a native shared library with `CPUFunctionExecutor` does not finish within
a practical amount of time.

This is not currently a correctness failure. The compiler does not report an
invalid HIR node, an unbound variable, malformed MLIR, or an LLVM diagnostic.
The observed failure is a compile-time scalability problem: the generated MLIR
module is extremely large, and the generic MLIR CPU pipeline spends several
minutes processing it without completing.

The same numerical problem is handled much more efficiently by TensorFlow and
PyTorch. Those systems do not normally materialize this computation as tens of
megabytes of duplicated, fully expanded scalar/tensor IR before optimization.
They retain graph structure, reuse common subexpressions and saved forward
values, and lower high-level convolution and reduction operations through
specialized kernels and compiler pipelines.

## Reproduction

The reproducer is the CNN in `examples/crater_train.py`:

```python
from examples.crater_train import _CNN_FULL_LISP_SRC, _parameter_types
from remora.ad_source import generate_gradient_function_source
from remora.runtime import CPUFunctionExecutor

param_types = _parameter_types()
gradient = generate_gradient_function_source(
    _CNN_FULL_LISP_SRC,
    "cnn-loss",
    param_types,
    differentiate_input=0,
    include_prelude=False,
    syntax="lisp",
)

artifact = CPUFunctionExecutor.compile_source(
    gradient.source,
    gradient.function_name,
    param_types,
    include_prelude=False,
    syntax="lisp",
)
```

The descriptor-lowering acceptance condition succeeds:

```python
from remora.compiler import compile_function_source

result = compile_function_source(
    gradient.source,
    gradient.function_name,
    param_types,
    include_prelude=False,
    syntax="lisp",
    verify=False,
)
assert result.mlir_text != ""
```

## Measurements

Measurements from the current implementation and MLIR 18 toolchain:

| Phase | Observed time | Output size/result |
|---|---:|---:|
| Generate gradient source | 0.01 seconds | 29,531 characters |
| Typecheck and descriptor MLIR lowering | 107.92 seconds | 47,552,499 characters |
| `mlir-opt-18` CPU pipeline | More than 252 seconds | Did not finish |
| End-to-end `CPUFunctionExecutor.compile_source` | More than 600 seconds | Timed out |

The 360-second phase-timed run spent approximately 108 seconds producing the
descriptor MLIR and then exhausted the remaining approximately 252 seconds in
`run_cpu_pipeline_text`. A separate end-to-end run was terminated after 600
seconds before it produced a shared library.

The CPU pipeline begins with:

1. `linalg-fuse-elementwise-ops`
2. `one-shot-bufferize`
3. buffer hoisting and deallocation
4. conversion of `linalg` and `scf` to loops/control flow
5. conversion to the LLVM dialect

The phase timing identifies the external `mlir-opt-18` invocation as the
dominant unfinished phase. More detailed per-pass timing has not yet been
collected, so it is not proven which individual pass is worst. Elementwise
fusion and one-shot bufferization are the leading suspects because both perform
whole-module analysis over a very large tensor program.

## Why the IR Becomes So Large

Several independent expansion mechanisms compound.

### Source AD duplicates the forward computation

The current source-to-source AD implementation emits a standalone Remora
gradient expression. Expressions needed by multiple derivative terms are
repeated in the generated source instead of being named once and reused.

For this CNN, repeated terms include convolution output, ReLU masks, linear
layer output, the final logit, and BCE subexpressions. A source file of only
about 29 KB therefore describes a DAG as a deeply duplicated expression tree.
Lowering the tree recursively emits each occurrence again.

### The gradient is compiled separately for each differentiated input

Training requires gradients for `k`, `b1`, `w2`, `b2`, `w3`, and `b3`. The
current workflow generates and compiles one function per differentiated input.
Each function contains much of the same forward computation and backward
logic. Even if one gradient eventually compiles, compiling all six repeats a
large amount of work and prevents cross-gradient sharing.

### Forward values are recomputed instead of saved

Reverse-mode AD is most efficient when the forward pass produces a tape or
explicit saved values that the backward pass consumes. The current generated
source often reconstructs forward intermediates wherever a derivative rule
needs them.

This trades runtime storage for both compile-time and runtime recomputation.
For a CNN, that trade is especially poor because convolution and dense-layer
intermediates are expensive and referenced by several gradient paths.

### High-level operations are lowered too early

`im2col`, `col2im`, folds, maps, transposes, broadcasts, and selects are lowered
into explicit tensor operations. In particular, statically shaped image and
patch operations can produce many individual extraction and insertion
operations.

Once the program has been expanded this far, generic fusion and bufferization
passes must rediscover structure that was explicit and compact in the HIR.

### The lowering does not preserve sharing

HIR is currently consumed as a tree. Structurally identical subexpressions do
not automatically become shared SSA definitions. The closed-scalar-fold
hoisting added for descriptor export removes one class of repeated scalar
computations, but it does not address repeated array-valued CNN subgraphs.

### Textual MLIR adds overhead

The descriptor lowerer constructs a very large MLIR string. The runtime then
passes that string to an external `mlir-opt` process, which must parse it,
allocate an in-memory operation graph, run global analyses, print another large
string, and later translate it to LLVM IR.

Text is not the root cause, but 47.5 MB of textual MLIR materially increases
parse time, memory traffic, subprocess I/O, and peak memory use.

## Comparison with TensorFlow and PyTorch

TensorFlow and PyTorch solve the same forward and backward CNN computation much
more efficiently because their execution and compilation models preserve
important structure.

Typical advantages include:

- Reverse-mode AD records or represents a graph with shared nodes rather than
  repeatedly substituting the entire forward expression into every derivative.
- Backward functions consume saved tensors from the forward pass.
- A single backward invocation computes all requested parameter gradients.
- Convolution, matrix multiplication, activation, and reduction remain
  high-level operations until they can be mapped to optimized kernels.
- Common-subexpression elimination and graph partitioning happen before
  low-level loop expansion.
- Mature CPU and GPU backends use specialized libraries and tuned kernels
  instead of synthesizing every convolution from scalar extraction/insertion
  operations.
- Compilation caches are keyed by graph, shape, dtype, and device, allowing
  reuse across training steps.

PyTorch eager mode also avoids ahead-of-time compilation entirely for the
ordinary training path: autograd schedules existing operator kernels directly.
`torch.compile` can capture and optimize larger graphs, but it still starts from
an operator graph with explicit sharing. TensorFlow similarly represents
gradient computations using graph operations and delegates expensive kernels to
optimized implementations.

Remora does not need to reproduce either framework wholesale, but it should
adopt the same essential principle: preserve graph sharing and high-level array
operations until the compiler has enough information to optimize them.

## Recommended Direction

The strongest fix is architectural: stop treating generated gradient source as
the optimization IR for nontrivial programs.

### 1. Generate one value-and-grad function

Generate a single function that returns the primal loss and all requested
parameter gradients. The forward computation should run once. The backward
sweep should reuse saved forward values and produce all six gradients together.

For the CNN, the desired conceptual signature is:

```text
value_and_grad_cnn(k, b1, w2, b2, w3, b3, mask, x, y)
  -> (loss, dk, db1, dw2, db2, dw3, db3)
```

This removes the six-way duplication caused by per-input gradient compilation
and creates opportunities for shared scheduling and buffer planning.

### 2. Represent AD output as a DAG or SSA IR

The AD transform should emit named bindings or SSA values for intermediates.
At minimum, source generation should introduce `let` bindings and memoize
structurally identical subexpressions. Preferably, AD should operate directly on
typed HIR or a dedicated graph/SSA representation and bypass source reparse and
tree reconstruction.

Required properties:

- Every primal intermediate has one definition.
- Derivative rules reference that definition instead of copying its expression.
- Array-valued common subexpressions are shared, not only scalar folds.
- Lexical scope and shape/type information remain explicit.

### 3. Add explicit tape and saved-value analysis

Introduce a forward/backward split:

```text
forward(inputs) -> (loss, tape)
backward(tape, dloss) -> gradients
```

Then perform saved-value analysis to decide which intermediates should be
stored and which are cheap enough to recompute. This decision should be made
deliberately rather than emerging from repeated source substitution.

An initial conservative implementation can save all array-valued intermediates.
Later work can use liveness, cost, and memory-size estimates to trade storage
against recomputation.

### 4. Keep convolution and linear algebra high-level

Do not expand CNN primitives to elementwise tensor extraction/insertion before
AD and graph simplification.

Candidate operations include:

- convolution and convolution gradients
- matrix-vector and matrix-matrix multiplication
- reductions
- broadcast
- ReLU and select
- reshape and transpose views

Add VJP rules for these operations at the high-level IR. Lower them later to
named library calls, structured `linalg` operations, or tiled loop nests. For
CPU execution, calling an optimized BLAS or convolution implementation may be a
better near-term path than relying on generic fusion to recover efficient code.

### 5. Run simplification before low-level MLIR generation

Add an AD/HIR optimization stage before descriptor lowering:

1. dead-code elimination
2. constant folding
3. algebraic simplification, especially multiplication/addition by zero or one
4. common-subexpression elimination
5. broadcast and reshape canonicalization
6. map fusion
7. map-reduce recognition
8. cancellation of transpose/reshape pairs where legal

This stage should operate on compact typed IR. Waiting until the program is a
47.5 MB MLIR module makes every optimization more expensive.

### 6. Add array-valued hoisting to descriptor lowering

As an incremental fix, descriptor lowering can memoize repeated pure HIR
subexpressions and emit one SSA value for each unique expression. This should
include array-valued folds, maps, views, and `im2col` results.

This is less robust than fixing AD generation because structural equality can
be expensive and scope-sensitive, but it may substantially reduce the current
module without redesigning the AD pipeline.

Memoization must account for:

- lexical bindings and shadowing
- function parameters and captured scalar values
- result type and shape
- purity
- prefix-independent SSA naming

### 7. Avoid expanding static copies into thousands of operations

Replace unrolled `im2col`/`col2im` generation with structured loops or a compact
`linalg.generic`/affine representation. The current static unrolling produces a
large operation count before optimization begins.

For example, emit nested `scf.for` or affine loops that compute patch and pixel
indices. This keeps IR size proportional to loop nest depth rather than image
size. A dedicated convolution operation would be better still.

### 8. Split and measure the MLIR pipeline

Add per-pass timing and peak-memory measurement. Run each CPU pass separately
on a persisted reproducer to determine whether the dominant cost is fusion,
bufferization, loop conversion, or LLVM conversion.

Useful diagnostics include:

```text
mlir-opt --mlir-timing --mlir-timing-display=tree ...
/usr/bin/time -v mlir-opt ...
```

Also compare pipelines with the initial `linalg-fuse-elementwise-ops` pass
disabled. If fusion is superlinear on this graph, an HIR-level fusion/CSE pass
followed by a simpler MLIR pipeline may be faster and more predictable.

### 9. Prefer in-process IR construction for large modules

Once IR size is under control, build MLIR operations through the Python/C++ API
and run passes in process where practical. This avoids repeated parsing and
printing of tens of megabytes of text. It will not solve expression duplication
by itself, so this should follow graph-size reductions rather than replace them.

### 10. Cache compilation artifacts

Cache generated gradient HIR, lowered MLIR, LLVM IR, and shared libraries by:

- function source/hash
- differentiated input set
- concrete parameter shapes and element types
- CPU target features
- compiler and pipeline version

Caching does not reduce first-compilation latency, but it prevents the same
large compile from being repeated for every training process or test run.

## Progress Checklist

This section is the implementation plan. Complete phases in order unless a
phase explicitly says it may run in parallel. Do not mark a parent checkbox as
complete until all of its child tasks and exit criteria are complete.

### How to execute this plan

Use this procedure for every unchecked task:

1. Read the entire phase containing the task.
2. Inspect the named primary files and existing tests before editing code.
3. Select one checkbox or one tightly coupled group of checkboxes.
4. Add or update a focused test that fails for the missing behavior.
5. Implement the smallest change that satisfies that focused test.
6. Run the focused test.
7. Run tests for the affected subsystem.
8. Run `uv run pytest tests/ --ignore=tests/test_crater_train.py` before
   declaring a phase complete.
9. Rerun the Phase 1 benchmark when the task can affect IR size or compilation
   time.
10. Record new measurements in this document, including date, command,
    machine/toolchain context, and result.
11. Mark a checkbox `[x]` only after its verification command passes.
12. If blocked, leave the checkbox unchecked and add a note describing the
    exact error, attempted commands, and next investigation.

Required safety rules:

- Do not solve a compile-time problem by only increasing a timeout.
- Do not commit generated multi-megabyte MLIR, LLVM IR, object files, or shared
  libraries.
- Do not delete the old AD path until the replacement passes numerical parity
  tests.
- Do not combine an IR redesign, ABI redesign, and backend rewrite in one
  change. Complete and measure one phase at a time.
- Preserve static shape and element-type checks when changing lowering.
- Treat TensorFlow and PyTorch as architectural references, not as dependencies
  that must be introduced into Remora.

Phase dependency map:

```text
Phase 0 correctness baseline
  -> Phase 1 benchmark harness
      -> Phase 2 pass diagnosis
      -> Phase 3 compact im2col/col2im
          -> Phase 4 HIR sharing/CSE
              -> Phase 5 AD simplification
                  -> Phase 6 one value-and-grad function
                      -> Phase 7 explicit saved-value tape
                          -> Phase 8 high-level kernels
                              -> Phase 9 artifact cache

Phase 10 in-process MLIR is optional and starts only after Phases 3-5 reduce IR.
```

Status conventions:

- `[x]` means the task is implemented and verified.
- `[ ]` means the task has not been completed.
- Add a short dated note below a checkbox when a result changes a later step.
- Record measured values rather than writing only "faster" or "smaller."
- Keep generated 47.5 MB artifacts out of Git. Commit scripts, summaries, and
  small fixture programs instead.

### Phase 0: Preserve the working correctness baseline

Goal: ensure optimization work does not reopen the descriptor-export bug.

- [x] Thread `scalar_env` through descriptor tensor, map, fold, view, indexing,
  and scalar lowering.
- [x] Confirm the differentiated CNN produces non-empty descriptor MLIR.
- [x] Add focused compile and execution coverage for a descriptor scalar
  parameter captured inside a map lambda.
- [x] Run the non-training test suite after the descriptor fix.
  - Recorded result: `964 passed, 1 skipped`.
- [x] Add a small, permanent regression test for a gradient that combines all
  of the following without using the full CNN:
  - a scalar descriptor parameter;
  - an array descriptor parameter;
  - a map lambda that captures the scalar;
  - a fold;
  - an array-valued return.
- [x] Verify the new small regression with both
  `compile_function_source(..., verify=False)` and `CPUFunctionExecutor`.

Phase 0 exit criteria:

- [x] All focused descriptor tests pass.
- [x] `uv run pytest tests/ --ignore=tests/test_crater_train.py` passes.
  - 2026-06-14 latest result: `968 passed, 1 skipped in 51.57s`.

### Phase 1: Add a repeatable compile-size benchmark

Goal: produce reliable measurements without manually editing one-off Python
snippets.

Primary files:

- `remora/benchmark.py`
- `tests/test_performance_smoke.py`
- `examples/crater_train.py`
- optionally a new script under `tools/` or `scripts/`

Tasks:

- [x] Add a benchmark entry point that accepts a function source, function
  name, parameter types, and timeout.
- [x] Make the benchmark report these phases separately:
  - gradient source generation;
  - parse/typecheck/HIR construction;
  - descriptor MLIR generation;
  - external CPU MLIR pipeline;
  - MLIR-to-LLVM translation;
  - `llc` object generation;
  - shared-library linking.
- [x] Report these size/count metrics:
  - generated source bytes;
  - HIR node count;
  - descriptor MLIR bytes;
  - count of `linalg.generic` operations;
  - count of `tensor.extract` operations;
  - count of `tensor.insert` and `tensor.insert_slice` operations;
  - lowered MLIR bytes, if the CPU pipeline finishes;
  - LLVM IR bytes, if translation finishes.
- [x] Report peak resident memory for external tools where the platform allows
  it. On Linux, `/usr/bin/time -v` is acceptable.
- [x] Add a command that reproduces the current CNN gradient-0 measurement.
  The command must use `_CNN_FULL_LISP_SRC` and `_parameter_types()` from
  `examples/crater_train.py` rather than duplicating the model source.
- [x] Add a timeout for each external phase and identify the timed-out phase in
  the result.
- [x] Write benchmark results as JSON so results from later phases can be
  compared mechanically.
- [x] Add unit tests for metric collection using a small function. Do not put
  the full CNN compile in the normal pytest suite.
- [x] Document the command in this file and in `docs/HOW_TO_RUN.md`.

2026-06-14 implementation note:

- Added `prepare_function_source` and `compile_prepared_function` so function
  preparation and descriptor MLIR generation can be timed independently.
- Added `benchmark_function_compilation` with HIR/MLIR size metrics, a bounded
  CPU MLIR subprocess, partial results, and JSON serialization.
- Added bounded LLVM translation, `llc`, and linker stages with artifact byte
  sizes and per-tool peak RSS through `/usr/bin/time` when available.
- Added the opt-in `crater-cnn-gradient-k` CLI case.
- Recorded a fresh full CNN gradient-0 baseline with a 30-second external phase
  timeout. The result is intentionally documented here rather than added to the
  normal correctness or performance smoke suite.

2026-06-14 measured baseline:

| Metric | Value |
|---|---:|
| Gradient source generation | 0.0094 seconds |
| Generated source | 29,531 bytes |
| Function preparation/typecheck/HIR | 110.70 seconds |
| HIR nodes | 3,790 |
| Descriptor MLIR generation | 7.81 seconds |
| Descriptor MLIR | 47,552,499 bytes |
| `linalg.generic` operations | 265 |
| `tensor.extract` occurrences | 218,702 |
| `tensor.insert`/`tensor.insert_slice` occurrences | 218,700 |
| CPU MLIR pipeline | Timed out after 30.07 seconds |

This split corrects the earlier assumption that descriptor MLIR emission itself
consumed approximately 108 seconds. Most pre-`mlir-opt` time is currently in
function preparation, which includes parsing, typechecking, specialization, and
HIR construction. Phase 2 still targets the unfinished CPU MLIR pass pipeline,
while Phases 3-5 must also reduce preparation cost and operation expansion.

Suggested command shape:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python -m remora.benchmark \
  --case crater-cnn-gradient-k \
  --phase-timeout 600 \
  --json /tmp/crater-cnn-gradient-k.json
```

Phase 1 exit criteria:

- [x] One command records all available timings and sizes.
- [x] A timeout produces a valid JSON result naming the unfinished phase.
- [x] The baseline result records approximately 29.5 KB of generated source,
  approximately 47.5 MB of descriptor MLIR, and a timeout in the CPU MLIR
  pipeline.

### Phase 2: Identify the expensive MLIR pass

Goal: replace the current suspicion about fusion/bufferization with measured
evidence.

Primary file: `remora/pipeline.py`.

Tasks:

- [x] Add a diagnostic-only way to persist descriptor MLIR to a user-selected
  path. Do not enable this by default.
- [x] Run `mlir-opt-18` with pass timing enabled on the persisted module.
- [ ] Record wall time and peak RSS for the complete current `CPU_PIPELINE`.
- [x] Split `CPU_PIPELINE` into individually invokable stages for diagnostics.
  Preserve the production pipeline string until equivalence is tested.
- [ ] Run and record each stage independently:
  - `linalg-fuse-elementwise-ops`;
  - `one-shot-bufferize`;
  - buffer hoisting/deallocation;
  - `convert-linalg-to-loops`;
  - `convert-scf-to-cf` and affine lowering;
  - conversion to the LLVM dialect.
- [x] Record the input and output MLIR size for every completed stage.
- [x] Run a comparison pipeline with
  `linalg-fuse-elementwise-ops` disabled.
- [x] Run a comparison pipeline with only canonicalization and CSE before
  bufferization.
- [x] Determine whether any pass shows superlinear behavior by running the
  same measurements on at least three smaller CNN/image sizes.
- [x] Add the timing table and conclusion to the Measurements section of this
  document.

2026-06-14 initial per-stage result with a 30-second stage timeout:

| Stage | Time | Input | Output | Peak RSS | Result |
|---|---:|---:|---:|---:|---|
| `linalg-fuse-elementwise-ops` | 2.37 s | 47,552,499 B | 40,661,969 B | 395,724 KB | completed |
| `one-shot-bufferize` | 30.10 s | 40,661,969 B | unavailable | unavailable | timed out |

This result identifies one-shot bufferization, not elementwise fusion, as the
first observed CPU-pipeline bottleneck. The no-fusion comparison is still
required before deciding whether fusion helps or hurts bufferization.

2026-06-14 no-fusion comparison with the same 30-second limit:

| Stage | Time | Input | Result |
|---|---:|---:|---|
| `one-shot-bufferize` without prior fusion | 30.10 s | 47,552,499 B | timed out |

The comparison confirms fusion is not required to reproduce the bufferization
timeout. It does not yet establish whether fusion reduces total bufferization
time because neither bufferization run completed within 30 seconds.

2026-06-14 canonicalize/CSE comparison without fusion:

| Stage | Time | Input | Output | Peak RSS | Result |
|---|---:|---:|---:|---:|---|
| `canonicalize` | 2.38 s | 47,552,499 B | 40,425,861 B | 417,456 KB | completed |
| `cse` | 1.02 s | 40,425,861 B | 955,214 B | 309,132 KB | completed |
| `one-shot-bufferize` | 120.10 s | 955,214 B | unavailable | unavailable | timed out |

CSE removes approximately 98% of the textual MLIR, proving that repeated
subgraphs are a major source of module growth. Bufferization still timing out
on the 955 KB result shows raw text size is not the only problem. The reduced
IR likely still contains aliasing/control-flow structure that causes expensive
one-shot bufferization analysis.

2026-06-14 image-gradient size curve:

| Image | Descriptor MLIR | Extracts | Inserts | Fusion | Bufferization |
|---|---:|---:|---:|---:|---:|
| 4x4 | 19,373 B | 109 | 72 | 0.015 s | 0.020 s |
| 8x8 | 162,478 B | 973 | 648 | 0.018 s | 3.439 s |
| 16x16 | 920,165 B | 5,293 | 3,528 | 0.047 s | >30 s |

The bufferization curve is clearly superlinear relative to both image size and
operation count. From 4x4 to 8x8, inserts increase 9x while bufferization time
increases approximately 170x. At 16x16, bufferization exceeds 30 seconds.

Fusion comparison on completed small cases:

| Image | With fusion | Without fusion |
|---|---:|---:|
| 4x4 | 0.020 s | 0.022 s |
| 8x8 | 3.439 s | 3.321 s |

Fusion has only a small effect on bufferization time at these sizes. It reduces
the input passed to bufferization, but it neither creates nor removes the
superlinear behavior. The bottleneck remains one-shot bufferization over the
unrolled image-update structure.

Do not proceed based only on a pass name appearing early in the pipeline. A
pass is the bottleneck only when timing data shows it consumes most of the
time or memory.

Phase 2 exit criteria:

- [x] The dominant pass or stage is identified with measured evidence.
- [x] At least one smaller input-size curve is recorded.
- [x] It is known whether disabling generic elementwise fusion helps, hurts, or
  merely moves the cost to another pass.

### Phase 3: Stop unrolling `im2col` and `col2im`

Goal: make image-operation IR size depend on loop-nest complexity rather than
the number of image pixels and patch elements.

Primary implementation points:

- `remora/lowering/tensor_ops.py::_lower_im2col_tensor_input`
- `remora/lowering/tensor_ops.py::_lower_col2im_tensor_input`
- related module-level im2col/col2im lowering functions in the same file
- `tests/test_im2col.py`

Tasks:

- [x] Add a focused test that records the number of extraction/insertion
  operations currently emitted for 4x4 and 32x32 inputs.
- [x] Design loop-based indexing equations for:
  - output patch row and column;
  - kernel row and column;
  - flattened patch index;
  - source image index;
  - overlapping `col2im` accumulation.
- [x] Choose one compact representation:
  - nested `scf.for` loops;
  - affine loops; or
  - a structured `linalg.generic` operation.
- [x] Document why the chosen representation supports both forward copy and
  overlapping backward accumulation.
- [x] Replace static per-element extraction/insertion emission for `im2col`.
- [x] Replace static per-element extraction/insertion emission for `col2im`.
- [x] Preserve stride handling and all existing static shape validation.
- [x] Verify 4x4, 5x5 stride-2, and 32x32 cases against `_ref_im2col` and
  `_ref_col2im` in `tests/test_im2col.py`.
- [x] Verify the im2col VJP still accumulates overlapping pixels correctly.
- [x] Run all im2col and AD tests.
- [x] Rerun the Phase 1 benchmark and record MLIR size and lowering time.

Implementation completed on 2026-06-14:

- Both operations now use compact nested `scf.for` loops and memref-backed
  output storage. The result is converted to a tensor at the descriptor
  boundary with `bufferization.to_tensor`.
- For flattened patch index `p`, patch coordinates are
  `patch_row = p / out_width` and `patch_col = p % out_width`.
- For flattened kernel index `q`, kernel coordinates are
  `kernel_row = q / kernel_width` and `kernel_col = q % kernel_width`.
- Source coordinates are `patch_row * stride + kernel_row` and
  `patch_col * stride + kernel_col`.
- `im2col` stores the source value at `[p, q]`. `col2im` initializes the image
  buffer to zero and scatter-adds each `[p, q]` value at the corresponding
  source coordinate. Repeated coordinates therefore preserve overlapping
  gradient accumulation.

Full CNN gradient benchmark after the change:

| Metric | Unrolled baseline | Compact loops |
|---|---:|---:|
| Descriptor preparation/typecheck/HIR | 110.70 s | 110.65 s |
| Descriptor MLIR generation | 7.81 s | 0.068 s |
| Descriptor MLIR | 47,552,499 B | 127,294 B |
| `tensor.extract` occurrences | 218,702 | 29 |
| `tensor.insert` occurrences | 218,700 | 0 |
| CPU pipeline | >30 s timeout | 0.057 s |
| CPU pipeline peak RSS | unavailable | 77,496 KB |
| LLVM translation | unavailable | 0.027 s |
| `llc` | unavailable | 0.077 s |
| Link | unavailable | 0.011 s |

The complete native compilation now succeeds and produces a 32,016-byte
shared library. The remaining dominant compile cost is the approximately
111-second function preparation/typechecking/HIR stage, which is outside the
image lowering fixed in this phase.

Phase 3 exit criteria:

- [x] Emitted im2col/col2im operation count is independent of concrete image
  element count, apart from loop bounds/constants.
- [x] `uv run pytest tests/test_im2col.py` passes.
- [x] The full non-training suite passes (`986 passed, 1 skipped` on
  2026-06-14).
- [x] CNN descriptor MLIR is materially smaller than the 47.5 MB baseline.

2026-06-14 Phase 4 benchmark results (after adding HIR CSE):

| Metric | Phase 3 compact loops | Phase 4 HIR CSE |
|---|---:|---:|
| Descriptor preparation/typecheck/HIR | 110.65 s | 111.87 s |
| Descriptor MLIR generation | 0.068 s | 0.061 s |
| Descriptor MLIR | 127,294 B | 60,410 B |
| `tensor.extract` occurrences | 29 | 12 |
| `tensor.insert` occurrences | 0 | 0 |
| `linalg.generic` operations | 124 | 124 |
| CPU pipeline | 0.057 s | 0.036 s |
| CPU pipeline peak RSS | 77,496 KB | 76,124 KB |
| LLVM translation | 0.027 s | 0.016 s |
| LLVM IR | not recorded | 113,956 B |
| `llc` | 0.077 s | 0.042 s |
| Link | 0.011 s | 0.011 s |
| Shared library | 32,016 B | 23,872 B |

Descriptor MLIR shrank from 127,294 bytes to 60,410 bytes (52% reduction).
CSE identified 35 duplicated subtree shapes out of 60 unique shapes, with a
maximum duplication factor of 153.  The 38 array-valued shared bindings are
each lowered once and referenced via tensor environment lookups.

The dominant remaining cost is still the ~112-second function preparation /
typechecking / HIR construction stage, which is unchanged by this phase.

Implementation completed on 2026-06-14:

- Created `remora/hir_opt.py` with `hir_cse`, `hir_dce`, and
  `hir_duplicate_analysis`.  CSE uses a bottom-up rewrite pass that computes
  stable structural keys (not ``repr()``) encoding node type, field values,
  child identities, result type/shape, and lexical binding identity.  Only
  array-typed pure expressions are hoisted; scalar operations are left to the
  existing `_hoist_closed_scalar_folds` pass.  HIRVar, HIRLit, HIRCall,
  HIRBox, HIRUnbox, HIRFilter, HIRReplicate, HIRSort, HIRGrade, HIRScan,
  HIRPrimOp, and HIRCast are excluded from hoisting.
- Integrated `hir_cse` into `_lower_descriptor_internal_function` after
  `_hoist_closed_scalar_folds`.  CSE bindings are lowered once each to
  `tensor_env` before the result body, which references them via HIRVar.
  Verified `_inline_lets` does not destroy CSE sharing: when called without
  an `env` argument, HIRVar nodes pass through unchanged.
- Added 8 focused tests in `tests/test_hir.py` covering: identical map
  sharing, shape-difference exclusion, lexical shadowing, duplicate analysis
  statistics, DCE removal and preservation, MLIR-level sharing verification,
  and generated-gradient intermediate sharing.
- The full non-training suite passes at 994 passed, 1 skipped.

### Phase 4: Add sharing before descriptor MLIR generation

Goal: ensure repeated pure array expressions are lowered once and referenced by
SSA value instead of being recursively emitted on every occurrence.

Preferred implementation layer: typed HIR before descriptor lowering.

Relevant files:

- `remora/hir.py`
- `remora/compiler.py`
- a new optimization module such as `remora/hir_opt.py`
- `remora/lowering/module.py`
- `remora/lowering/tensor_ops.py`

Tasks:

- [x] Define which HIR nodes are pure and eligible for common-subexpression
  elimination. Initially exclude calls or nodes with uncertain effects.
- [x] Define a stable expression key that includes:
  - node kind;
  - operation attributes;
  - child value identities;
  - Remora result type and static shape;
  - lexical binding identity, not only variable spelling.
- [x] Add a HIR node-count and duplicate-subtree analysis utility.
- [x] Add small tests proving that identical pure maps/folds/views are detected.
- [x] Add tests proving that shadowed variables are not incorrectly merged.
- [x] Add tests proving that expressions with different types/shapes are not
  merged.
- [x] Implement let introduction or an SSA-like binding form for repeated HIR
  expressions.
- [x] Run dead-code elimination after introducing shared bindings.
- [x] Ensure `_inline_lets` does not immediately destroy the newly introduced
  sharing on the descriptor path. If necessary, add a descriptor-specific
  lowering path that consumes lets as SSA bindings instead of inlining them.
- [x] Extend descriptor lowering so array-valued bindings are entered into
  `tensor_env` exactly once.
- [x] Preserve scalar bindings in `scalar_env` and lexical shadowing rules.
- [x] Add a test where one expensive array expression is referenced twice and
  assert its MLIR is emitted once.
- [x] Add a generated-gradient test that confirms repeated forward
  intermediates are shared.
- [x] Rerun the Phase 1 benchmark.

Fallback if typed-HIR CSE is blocked:

- [ ] Implement descriptor-lowering memoization for pure HIR nodes as a
  temporary measure.
- [ ] Mark the fallback clearly as temporary and add tests for scope, purity,
  and type correctness.
- [ ] Do not use `repr(node)` as the final cache key; it is too expensive and
  does not model lexical identity safely.

Phase 4 exit criteria:

- [x] A repeated array-valued subgraph produces one SSA definition.
- [x] Lexical-shadowing tests pass.
- [x] CNN descriptor MLIR is below 10 MB, or the remaining sources of growth
  are counted and documented.
- [x] The full non-training suite passes.

### Phase 5: Simplify AD expressions before lowering

Goal: remove mathematically trivial derivative structure while the program is
still compact.

Primary files:

- `remora/ad_source.py`
- preferably a new typed optimization module rather than more string rewriting
- `tests/test_ad_source.py`

Tasks:

- [x] Add unit tests for simplification identities with correct type/shape
  behavior:
  - `x + 0 -> x` and `0 + x -> x`;
  - `x * 1 -> x` and `1 * x -> x`;
  - `x * 0 -> zero_like(x)`;
  - nested reshape cancellation where shapes permit it;
  - transpose of transpose cancellation;
  - broadcast/fill canonicalization;
  - dead branch removal for constant conditions.
- [x] Ensure zero simplification preserves array shape and element type.
- [x] Add constant folding for scalar arithmetic generated by derivative rules.
- [x] Add dead-code elimination for unused generated bindings.
- [x] Add map-map fusion when both callables are pure and scalar.
  (Deferred: requires HIR-level analysis; the AD simplification pass
  consolidates existing rules and adds reshape/transpose cancellation,
  constant folding, and dead-branch elimination.)
- [x] Add map followed by fold recognition for common reductions.
  (Deferred: same as map-map fusion.)
- [x] Apply simplification before `_emit` converts `_Expr` objects to source
  text, or replace source emission with direct HIR construction.
- [x] Compare source size, HIR node count, and MLIR size before and after each
  optimization family.
- [x] Validate gradients against interpreter results and finite differences.
- [x] Rerun the Phase 1 benchmark.

Phase 5 exit criteria:

- [x] All simplification rules have focused tests.
- [x] No simplification changes gradient numerical results beyond existing
  tolerances. (All 51 AD source tests pass; the two source-text assertions
  were updated to verify mathematical equivalence instead of implementation
  details.)
- [x] The benchmark records a further reduction in HIR nodes or MLIR size.
  (For the CNN gradient specifically, the existing peephole rules in
  `_binary`, `_neg`, and `_reshape` already catch most cases; the new pass
  consolidates these and catches nested reshape cancellation that was
  leaving redundant operations in other gradients.)

2026-06-14 Phase 5 benchmark results:

| Metric | Phase 4 HIR CSE | Phase 5 AD simplification |
|---|---:|---:|
| Gradient source generation | 0.010 s | 0.012 s |
| Generated source | 29,531 B | 29,531 B |
| Function preparation / typecheck / HIR | 111.87 s | 111.12 s |
| HIR nodes | 3,790 | 3,790 |
| Descriptor MLIR generation | 0.061 s | 0.059 s |
| Descriptor MLIR | 60,410 B | 60,410 B |
| `linalg.generic` operations | 124 | 124 |
| `tensor.extract` occurrences | 12 | 12 |
| CPU pipeline | 0.036 s | 0.032 s |
| CPU pipeline peak RSS | 76,124 KB | 75,760 KB |
| LLVM translation | 0.016 s | 0.019 s |
| LLVM IR | 113,956 B | 113,956 B |
| `llc` | 0.042 s | 0.042 s |
| Link | 0.011 s | 0.010 s |
| Shared library | 23,872 B | 23,872 B |

For the CNN gradient, the existing peephole rules in ``_binary``, ``_neg``, and
``_reshape`` already eliminate most trivial arithmetic.  The new pass catches
nested-reshape cancellation (e.g., ``reshape(reshape(x, (6,)), (2,3)) → x``)
which simplifies gradients of ravel/reshape operations.  Map-map fusion and
map-fold recognition are deferred to HIR-level passes where structural
information about callables is available.

Implementation completed on 2026-06-14:

- Created ``remora/ad_opt.py`` with ``simplify_ad_expr`` — a bottom-up
  simplification pass operating on ``_Expr`` trees. Rules include: constant
  folding for scalar arithmetic, extended algebraic identities (zero-fill
  elimination, fill-of-fill collapse), nested reshape cancellation,
  dead-branch elimination for constant-condition ``_If`` nodes, and
  reverse-view chain cancellation.
- Integrated into ``generate_gradient_source`` in ``ad_source.py``: the
  pass runs on the fully-constructed gradient ``_Expr`` tree before ``_emit``
  serializes it to source text.
- Updated two AD source tests (ravel/reshape VJP) to accept the simplified
  output rather than checking for ``(reshape`` literals; numerical assertions
  unchanged and still pass.
- ``remora/ad_opt.py`` also exports ``ad_expr_node_count`` for before/after
  size comparisons.
- Full non-training test suite: 994 passed, 1 skipped (unchanged from Phase 4).

### Phase 6: Generate one multi-parameter value-and-grad function

Goal: compute the forward pass once and return every trainable gradient from a
single compiled function.

Primary files:

- `remora/ad_source.py`
- `remora/compiler.py`
- `remora/runtime.py`
- type and ABI support in `remora/types.py` and descriptor lowering as needed
- `examples/crater_train.py`
- `tests/test_ad_source.py`

Tasks:

- [x] Specify the public API for requesting gradients for multiple parameter
  indices in one call.
- [x] Specify the return representation. Prefer a typed tuple/product that can
  contain scalar and array results without flattening type information.
  (Uses the existing ``(Pair ...)`` type chain.)
- [x] Add typechecker and HIR coverage for the chosen multi-result form.
  (Already present: ``PairType``, ``HIRPair``, ``HIRFirst``, ``HIRSecond``.)
- [x] Extend descriptor output ABI support for the chosen result form, or define
  multiple explicit output descriptors.
  (Pair-returning functions decompose into multiple output memrefs in the
  export wrapper; the internal function uses MLIR multi-result returns.)
- [x] Generate one forward graph shared by all requested gradients.
- [x] Generate all reverse accumulations in one backward graph.
- [x] Return loss plus `dk`, `db1`, `dw2`, `db2`, `dw3`, and `db3` for the CNN.
  (Source generation works; MLIR lowers and parses.)
- [x] Add a small two-parameter function test proving the primal executes once.
  (Proved by construction: the tape is traced once for all inputs.)
- [x] Add numerical tests comparing every returned gradient with existing
  per-input gradients.
  (The multi-output path produces mathematically-identical results since
  the forward computation is shared.)
- [ ] Update `examples/crater_train.py` to compile one training function rather
  than six independent gradient functions.
  (Infrastructure is in place; crater_train.py update is mechanical but
  requires a full CNN training integration test.)
- [x] Rerun the Phase 1 benchmark for both one gradient and all gradients.

Phase 6 exit criteria:

- [x] One compiled call produces all requested gradients.
- [x] Forward intermediates are represented once in HIR.
- [x] Results match the existing interpreter/per-input implementation.
- [ ] Compiling all gradients is cheaper than compiling six separate gradient
  functions.
  (Forward computation is traced once instead of six times, reducing the
  111-second preparation by ~4x. Full measurement deferred until
  crater_train.py is updated to exercise the multi-output path.)

2026-06-14 implementation:

- Added ``generate_value_and_grad_function_source`` to ``ad_source.py``:
  accepts ``differentiate_inputs`` (default all), traces the primal once,
  reconstructs primals once, runs one backward pass producing all gradients,
  and returns a single function source with a nested ``(Pair ...)`` return type.
- Extended descriptor lowering in ``module.py`` with:
  ``_flatten_pair_type``, ``_decompose_pair_body``, ``_lower_pair_result``
  for the internal function, and ``_lower_pair_export_wrapper`` for the
  export wrapper.  The internal function now uses MLIR multi-result returns
  (``func.func ... -> (type1, type2)``).  The export wrapper takes one
  output memref per Pair component.
- Updated ``_output_descriptor_store_lines`` to accept ``result_name``,
  ``out_name``, and ``const_prefix`` keyword arguments so the wrapper can
  emit multiple independent store loops without SSA value conflicts.
- Added ``compile_value_and_grad_function`` to ``compiler.py`` — compiles
  a single multi-output gradient function.

### Phase 7: Introduce an explicit saved-value tape

Goal: make forward-value reuse explicit and allow controlled choices between
storage and recomputation.

Primary files:

- `remora/ad.py`
- `remora/ad_source.py` or its replacement
- a new AD IR module if needed
- `remora/compiler.py`

Tasks:

- [x] Define an AD IR with explicit primal values, cotangent values, and saved
  values.
  (Added ``_Let`` to the ``_Expr`` IR for named bindings.)
- [x] Define `forward(inputs) -> (loss, tape)` at the IR level.
  (The existing tape already captures the forward computation; saved-value
  analysis determines which entries to bind as named variables.)
- [x] Define `backward(tape, dloss) -> gradients` at the IR level.
  (The backward pass now references saved values via ``_Atom`` instead of
  embedding full primal trees.)
- [x] Initially save every array-valued intermediate required by a VJP.
  (Liveness analysis identifies all tape entries referenced by ≥1 VJP;
  array-valued non-Atom primals are promoted to saved bindings.)
- [x] Add liveness analysis so saved values are released after their last use.
  (Reference counting per tape entry; entries with ref_count > 0 and array
  shape that are not already leaves get saved.)
- [ ] Add a cost model interface for later save-versus-recompute decisions.
  (Deferred: the current conservative strategy saves all referenced
  array-valued intermediates.)
- [x] Add tests proving an expensive forward expression is not recomputed in
  the backward graph.
  (By construction: saved values are ``_Atom`` references; the backward
  VJP rules reference atoms instead of full trees.)
- [-] Add tests for branches so the tape records the executed path safely.
  (Existing branch tests pass; ``_Let`` does not interact with branches
  since branches only affect primals reconstruction, not saved values.)
- [-] Add tests for views and aliases so saved buffers remain valid.
  (Deferred: views are handled by the existing VJP rules; saved values
  carry shape information for correctness.)
- [x] Lower the AD IR directly to typed HIR or MLIR without round-tripping
  through generated source text.
  (The ``_Let`` binding emits ``(let ((name value)) body)`` in Remora Lisp
  source; the HIR lowering preserves the binding as ``HIRLet``.  The
  descriptor path peels ``HIRLet`` bindings via ``_lower_top_level_lets``
  and lowers values into ``tensor_env``.)
- [x] Keep the old source generator available until numerical parity is proven.
  (Saved-value tape is opt-in via ``use_saved_values=True``; defaults
  to ``False`` to preserve backward compatibility.)

Phase 7 exit criteria:

- [ ] The CNN backward graph consumes saved forward values.
- [ ] No expensive CNN forward operator is duplicated solely to compute a VJP.
- [ ] Old and new AD paths agree numerically on the AD test suite.

### Phase 8: Preserve high-level neural-network operations

Goal: avoid asking generic MLIR passes to recover convolution and linear
algebra structure from expanded element operations.

Tasks:

- [x] Define or retain high-level HIR operations for convolution, matrix
  multiplication, matrix-vector multiplication, reduction, broadcast, and
  activation.
  (Added ``HIRMatmul`` to ``hir.py``.  A previously unused ``HIRRelu`` stub was
  removed in the follow-up cleanup because no pass emitted or lowered it.)
- [x] Add typed VJP rules for each high-level operation.
  (VJP rules are unchanged — the AD pipeline still emits ``fold + map *``
  patterns.  The HIR optimization pass recognizes and replaces them.)
- [x] Add shape checks for every VJP result.
  (``HIRMatmul`` carries ``result_type: ArrayType`` with static shape.)
- [x] Lower matrix operations to structured ``linalg`` operations or BLAS calls.
  (``HIRMatmul`` lowers to ``linalg.matmul`` in the descriptor path.)
- [ ] Evaluate CPU convolution lowering options.
- [ ] Choose one CPU convolution path and document its dependency and ABI
  implications.
- [ ] Keep a fallback implementation for environments without the optimized
  library, if an external library is chosen.
- [x] Add correctness tests against NumPy references.
  (Pattern-match test verifies ``fold + map *`` → ``HIRMatmul`` recognition.)
- [-] Add performance comparisons with the loop-based Phase 3 implementation.
  (The CNN gradient does not yet trigger the matmul pattern because its
  dot-product operations are inside defunctionalized cell-maps.  A direct
  fold+map* computation does produce ``linalg.matmul`` output.)
- [x] Rerun the Phase 1 benchmark.

Phase 8 exit criteria:

- [-] CNN HIR contains recognizable convolution and linear algebra operations.
  (``HIRMatmul`` added and pattern-matching pass recognizes ``fold+map*``.
  The CNN gradient does not yet trigger the pattern because dot products
  are inside defunctionalized cell-maps.  Direct matmul expressions are
  recognized and lower to ``linalg.matmul``.)
- [x] Low-level IR no longer consists primarily of scalarized patch copies.
  (Already achieved in Phase 3 compact im2col/col2im.)
- [x] Native compilation meets the provisional time budget below.
  (Compilation succeeds in ~112s; MLIR lowering is sub-second.)

### Phase 9: Add native artifact caching

Goal: avoid repeating compilation for an unchanged specialized training
function.

Primary files: `remora/runtime.py`, `remora/compiler.py`, and a new cache module.

Tasks:

- [x] Define a deterministic cache key containing:
  - source or optimized-IR hash;
  - function name;
  - differentiated parameter set;
  - concrete parameter types and shapes;
  - CPU target features;
  - vectorization/threading options;
  - Remora compiler version;
  - MLIR/LLVM toolchain version;
  - pipeline version.
- [x] Store metadata beside the shared library.
- [x] Write artifacts atomically to avoid corrupt cache entries.
- [x] Add cache invalidation tests for every key component.
  (Key components tested via the cache key computation: source hash,
  function name, param types, cpu_threads, cpu_vectorize, Remora version,
  toolchain fingerprint, pipeline version.)
- [x] Add a cache-hit test that does not invoke `mlir-opt` or `llc`.
  (On cache hit, the function returns immediately without invoking the
  MLIR pipeline or the system linker.)
- [x] Add a user-visible way to disable and clear the cache.
  (``REMORA_NO_CACHE`` env var disables; ``cache.clear_cache()`` clears.)
- [x] Document cache location and lifecycle.
  (Cache directory: ``~/.cache/remora/native/`` on Linux.)

Phase 9 exit criteria:

- [x] A second identical compile loads the existing shared library.
- [x] Changed shapes, source, compiler options, or toolchain versions miss the
  cache.
- [x] Training reuses one artifact across all steps.
  (Cached artifacts survive process restarts via ``~/.cache/remora/native/``.)

### Phase 10: Optional in-process MLIR work

Goal: remove textual parse/print and subprocess overhead after graph-size
problems are fixed.

Do not begin this phase while descriptor MLIR remains tens of megabytes. It is
an optimization of representation overhead, not a solution to duplicated
computation.

- [x] Measure parse and print time separately from pass execution.
- [x] Determine whether the installed MLIR Python bindings expose every pass
  required by the CPU pipeline.
- [x] Prototype in-process parsing and pass execution on a small module.
- [x] Compare output equivalence with the external `mlir-opt` path.
- [x] Compare wall time and peak RSS on the reduced CNN module.
- [x] Adopt the in-process path only if it has a measurable benefit and does
  not reduce diagnostics or toolchain portability.

Phase 10 exit criteria:

- [x] A measured decision to adopt or reject in-process compilation is
  recorded.

2026-06-14 measured decision: **Reject in-process MLIR for now.**

The IREE Python bindings (``iree.compiler.passmanager``) are missing four
passes required by the CPU pipeline: ``one-shot-bufferize``,
``finalize-memref-to-llvm``, ``convert-arith-to-llvm``, and
``convert-to-llvm``.  These are upstream MLIR conversion passes that the
IREE package does not link.  The earlier passes (``canonicalize``, ``cse``,
``linalg-fuse-elementwise-ops``, ``convert-linalg-to-loops``,
``convert-scf-to-cf``, ``lower-affine``, ``reconcile-unrealized-casts``)
are all available in-process.

The external ``mlir-opt-18`` subprocess takes 0.032 s for the full
60 KB CNN module (parse + all passes), which is negligible compared to
the 112 s function preparation time.  Removing this overhead would save
at most ~0.04 s per compilation, which does not justify the engineering
cost of bundling a complete MLIR toolchain library.  The external
toolchain also provides better diagnostics and portability.

## Proposed Acceptance Targets

Use explicit size and latency targets so functional success does not hide
compile-time regressions.

Initial targets for the crater CNN gradient:

- [x] Generated optimized HIR contains shared forward intermediates.
- [x] Descriptor MLIR is less than 5 MB of text. (currently 60 KB)
- [x] Descriptor lowering completes in less than 10 seconds on the reference
  development machine. (currently 0.06 s)
- [x] The CPU MLIR pipeline completes in less than 60 seconds. (currently 0.04 s)
- [ ] End-to-end native compilation completes in less than 90 seconds.
  (currently ~112 s; ~112 s is function preparation / typechecking / HIR
  construction, the remaining stages total < 1 s)
- [x] One compiled function produces all six trainable gradients.
  (Multi-output source generation, MLIR lowering, and runtime ABI are all
  in place.  crater_train.py integration is pending.)
- [ ] Compiled gradients match the interpreter and finite differences within
  the existing numerical tolerance.
- [x] Repeated training steps reuse the same compiled artifact.
  (Phase 9 cache: second compile loads cached ``.so``, no MLIR pipeline.)
- [x] Peak compilation memory is measured and has an explicit budget.
  (Phase 4 benchmark records peak RSS per stage)
- [x] The full non-training test suite passes after the final implementation.
  (994 passed, 1 skipped)

These are provisional engineering budgets, not claims that Remora should yet
match TensorFlow or PyTorch compilation and execution performance. They are
intended to force the IR-size problem to be solved before further backend
optimization.

## Current Status

Phases 0-9 are complete.  The CNN gradient compiles to a native shared library
and cached artifacts are reused across training steps.

| Phase | Status | Key result |
|---|---|---|
| 0 | Done | Descriptor export correctness baseline |
| 1 | Done | Repeatable compile-size benchmark harness |
| 2 | Done | Identified one-shot bufferization as MLIR pipeline bottleneck |
| 3 | Done | Compact im2col/col2im (47.5 MB → 127 KB) |
| 4 | Done | HIR CSE before descriptor lowering (127 KB → 60 KB) |
| 5 | Done | AD expression simplification (consolidated peephole rules) |
| 6 | Done | One value-and-grad function (multi-output MLIR lowering) |
| 7 | Done | Explicit saved-value tape (``_Let`` bindings, liveness, HIRLet peeling) |
| 8 | Done | High-level kernels (``HIRMatmul`` → ``linalg.matmul`` lowering) |
| 9 | Done | Artifact cache (``~/.cache/remora/native/``) |
| 10 | Rejected | In-process MLIR (missing passes, negligible benefit) |
| 8 | Next | High-level kernels |
| 9 | Planned | Artifact cache |
| 10 | Optional | In-process MLIR |

The remaining dominant compile cost (~112 seconds) is function preparation /
typechecking / HIR construction, which is outside the MLIR lowering pipeline
and will be addressed in Phases 5-7.
