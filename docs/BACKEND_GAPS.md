# Backend And Full-Language Gaps

This document tracks remaining gaps between the current RemoraC implementation
and a fuller implementation of the Remora language. It includes language
features that are not parsed or typed yet, semantics that exist only in a
restricted form, and backend support gaps. For future milestones and research
directions, see `docs/ROADMAP.md`. For completed work, see
`docs/IMPLEMENTATION_LOG.md`.

## Major Full-Language Gaps

These are the largest gaps between the current static dense-core compiler and a
fuller Remora implementation.

### True Dynamic Shapes

The current compiled CPU/GPU pipeline is static-shapes-only at the lowering
boundary. The front end has dependent-shape machinery, but compiled functions
specialize dimensions to constants and reject unspecialized index variables.

A full implementation needs:

1. runtime-sized arrays in compiled artifacts;
1. dynamic-shape MLIR such as `tensor<?x...>` and `memref<?x...>`;
1. runtime loop bounds and allocation sizes;
1. GPU kernels that read sizes and strides from descriptors instead of baked
   constants;
1. residual runtime checks for shape constraints not discharged statically.

### Runtime Boxes And Ragged Arrays

The current `box`/`unbox` machinery is effectively type erasure because shapes
are static. Full Remora-style irregular data needs existential packages that
carry runtime dimension witnesses.

Missing:

1. runtime representation for boxed values and hidden dimensions;
1. `box`/`unbox` that carries and recovers dimension witnesses;
1. arrays of boxes with different hidden shapes;
1. ragged/irregular nested arrays;
1. allocation and layout strategy for boxed values on CPU and GPU.

This depends heavily on true dynamic shapes.

### Segmented Reductions And Irregular Parallelism

Segmented reductions from the Remora papers are not implemented. There are no
surface forms, AST/HIR nodes, type rules, interpreter semantics, CPU lowering,
or GPU lowering for them. They are a natural companion to boxes, ragged arrays,
and irregular parallel programs.

### Full Rank-Polymorphic Surface Language

The implementation covers a large dense core, but not every source form from the
Remora papers. Missing paper forms are listed below under
[Missing Surface Syntax From The Papers](#missing-surface-syntax-from-the-papers).

### Full Backend Parity

The interpreter, CPU backend, and GPU backend do not implement identical
surfaces. CPU covers the current dense static core well. GPU covers many dense
numeric patterns but is still narrower for higher-order functions, closure
capture, general recursion, boxes, irregular data, AD, and some operation/type
combinations.

## Language Features Deferred Or Missing

These features are upstream of lowering: the parser or typechecker rejects or
does not yet model them fully.

### Dynamic Higher-Order Functions

Functions passed as arguments work via monomorphization. Call-through-variable
cases such as `let f = inc in f(f 5)` work when statically resolvable.

Still missing: passing a function through a `map` body as a callable, such as
`map f arr` where `f` is a let-bound variable. The map lowering needs to resolve
the callable from the scalar environment.

### Functions In Function Position

The typechecker defers map over function-valued arrays. This blocks the classic
Remora MIMD pattern:

```lisp
(define m [[square sqrt] [add1 sub1]])
(m 9)
```

### Remaining Text-Path Deferral

One CPU text-path site remains intentionally deferred: binary map operator
sections. A unary section callable such as `(* 2)` passed to a binary map is
semantically ambiguous in the current implementation. The typechecker currently
rejects sections in binary callable positions; full Remora would support this
with pair-type output.

### `shape` / `rank` Of Function Values

Querying the shape or rank of a function-typed value is deferred in HIR/runtime.

### Missing Surface Syntax From The Papers

The following paper forms are absent from the current grammar:

1. `(frame [d1 ... dn] expr1 ... exprn)` for explicit frame construction;
1. `(array [d1 ... dn] atom1 ... atomn)` for explicit array-of-atoms
   construction;
1. `all` keyword on parameters for consuming an entire argument as one cell.

### `ComposeExpr` Asymmetry

Function composition is in the ML-syntax grammar and parser but not in the Lisp
reader. Lisp programs cannot use the same composition syntax.

## Type System, Numeric, And AD Gaps

These are not all part of the original Remora language, but they are important
gaps for a mature implementation of this compiler.

### Numeric Type Coverage

Implemented scalar types are currently centered on `Int`, `Float`, `Float64`,
and `Bool`. Missing or incomplete extensions include:

1. `Int64` literals, annotations, promotion, runtime dtype plumbing, and backend
   lowering;
1. more systematic mixed-precision rules;
1. complex numbers and complex arrays;
1. complex arithmetic and casts;
1. comparison policy for complex values;
1. FFT primitives such as `rfft` and `irfft`.

### Bool Representation And Predicate Workflows

Bool is implemented, but future work remains:

1. bit-packed bool arrays;
1. predicate-array fusion for `map predicate -> filter` and related pipelines;
1. higher-level predicate utilities such as `where` and `count-true`.

### Automatic Differentiation Coverage

Reverse-mode AD exists, but it is not a full-language, all-backend facility.

Limitations:

1. primarily scalar-cost floating-point functions;
1. GPU AD is narrower than CPU/interpreter support;
1. no complex AD semantics;
1. no AD over runtime boxes, ragged arrays, segmented reductions, or true
   dynamic-shape programs;
1. limited pair-valued/multi-parameter gradient support in compiled backends.

### Library And Primitive Coverage

Compared with a mature full array language, RemoraC still lacks broad standard
library coverage:

1. FFT and signal-processing primitives;
1. broader linear algebra such as solve, eigensolvers, and norms;
1. random number generation and Monte Carlo utilities;
1. statistical reductions;
1. sparse or segmented data primitives.

## CPU Lowering Follow-Ups

The CPU text lowering path covers the dense-core constructs accepted by the
typechecker, including recursion, higher-order monomorphization, closure
capture, arbitrary-rank `indices-of`, integer arithmetic, multi-array maps, and
compound map bodies with indexing.

Remaining CPU work is mostly beyond the current dense core or performance
engineering:

1. recursive array construction with dependent result sizes;
1. threaded CPU vectorization and scheduling policy;
1. MLIR builder-path retirement or revival as a validation backend;
1. CPU performance benchmark suite;
1. better native cache invalidation using lowering-version fingerprints.

## GPU Lowering Gaps

GPU coverage is high for the static dense subset, but not semantically identical
to CPU/interpreter support. Remaining useful work is mostly about scale limits,
general device-side calls, dynamic/irregular data, and operation/type
completeness.

### Phase 5 Scale Limits

1. Multi-block i32 prefix sum: done.
1. Multi-block scatter-add for `N > 1024`: partial. The guard was relaxed, but
   emitting `llvm.atomicrmw fadd` text for larger inputs remains.
1. Recursive multi-level scan for `N > 1M`: deferred.
1. Sort beyond `N > 1M`: deferred for radix sort; fallback bitonic sort is
   practical for correctness but not performance.

### Phase 6 Structural Nodes

1. `HIRPair` / `HIRFirst` / `HIRSecond`: done in map bodies via the expression
   compiler.
1. `HIRFoldRight` standalone: done for associative `+`/`*`.
1. `HIRCall` in map bodies: partial. Non-recursive helpers inline;
   unsupported calls fail loudly.
1. Recursive device functions: partial. Scalar self-tail-recursive helpers
   inside `map` are supported; general recursion is deferred.

### Recursive Multi-Level Scan

The existing multi-block scan handles up to 1,048,576 elements. For larger
inputs, the block-sum scan itself exceeds 1024 elements and needs another
aggregation level. A future implementation should use recursive or hierarchical
aggregation similar to CUB:

1. per-block Hillis-Steele scan;
1. scan block sums in subgroups;
1. scan subgroup sums;
1. propagate prefixes.

### Sort Beyond 1M

The radix sort handles up to roughly 1,048,576 elements. For larger arrays, the
block count exceeds 1024. Options:

1. partition into segments, sort each, then merge;
1. extend histograms with 2D block grids;
1. use multi-block bitonic sort as a correctness fallback.

### Device-Side Function Calls

Non-recursive helper calls inside GPU map bodies are inlined. Scalar
self-tail-recursive helpers lower to loop blocks. Remaining work:

1. call-through-variable on GPU using specialization or a closed function table;
1. private reusable device functions instead of always inlining large helpers;
1. better diagnostics for unsupported higher-order GPU programs.

PTX device-side function pointers would only be needed for truly dynamic callee
selection and would require a PTX-level function table plus indirect calls.

### General Recursion On GPU

General recursion remains open. Options:

1. non-tail recursion with a per-thread stack and explicit frame layout;
1. mutual tail recursion as a state machine with a program counter;
1. array-returning recursive helpers with scratch buffers or segmented outputs.

This should wait for a concrete workload, such as tree traversal, dynamic
programming, or recursive combinator interpretation.

### Scatter-Add Atomic Text Path

The current parallel scatter-add builder handles `N <= 1024` with a
shared-memory barrier and single-thread add. For larger inputs, shared memory
cannot hold all elements. The remaining work is emitting global
`llvm.atomicrmw fadd` text.

### GPU Features Missing Relative To CPU/Interpreter

GPU support remains narrower for:

1. full closure capture;
1. full higher-order functions;
1. dynamic call-through-variable;
1. reusable device-side functions;
1. non-tail, mutual, and array-returning recursion;
1. runtime boxes and ragged arrays;
1. segmented reductions;
1. full pair support outside supported expression contexts;
1. full AD coverage;
1. complete dtype/op coverage for every operation.

## Backend Support Ambiguities To Reconcile

The docs should be reconciled with executable behavior for:

1. GPU `sort`/`grade` dtype support, where some docs still say f32-only while
   later implementation notes describe i32 support;
1. interpreter "dynamic shapes" wording versus compiled-code dynamic-shape
   support;
1. GPU pair support in map bodies versus top-level/general pair support;
1. im2col/col2im support by backend and context;
1. f64 operation support by backend and operation family.

Long-term, these should be expressed as an executable support matrix used by
docs, diagnostics, and parametrized tests.

## Syntax And Usability Gaps

These are not blockers for semantic completeness, but they are notable gaps for
language usability and research exploration.

### J/K-Like Compact Array Syntax

RemoraC has ML and Lisp syntaxes. It does not have a terse APL/J/K-like syntax
for rank operators, adverbs, trains, or tacit array programming. Such a frontend
could lower to the existing AST/HIR without changing core semantics, but would
increase documentation and testing burden.

### Diagnostics And Support Matrix

Many gaps are currently known through docs, tests, or backend errors. A fuller
implementation should expose:

1. source-located errors for unsupported language/backend features;
1. `--explain-lowering` output showing which lowering path was selected;
1. an executable support matrix shared by docs, tests, and diagnostics.
