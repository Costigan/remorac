# Study Guide For Understanding RemoraC

This guide lists the computer-science topics worth studying to understand the
Remora language, its semantics and type system, and this project's CPU, CPU
vector, and GPU compilation paths.

Start with the current project docs:

1. `docs/PROJECT_OVERVIEW_AND_ARCHITECTURE.md`
1. `docs/DENSE_CORE.md`
1. `docs/USER_GUIDE.md`
1. `docs/ABI.md`
1. `docs/BACKEND_GAPS.md`
1. `docs/ROADMAP.md`
1. `docs/remora-reference/README.md`

Some docs in this directory are archived or historical. Use them to understand
why the project took a path, not as the final support matrix. The current
contracts are the Dense Core scope, descriptor ABI, user guide, backend gaps,
implementation log, and project overview.

## Table Of Contents

- [Start Here](#start-here)
  - [Minimum Viable Path](#minimum-viable-path)
  - [Goal-Based Entry Points](#goal-based-entry-points)
  - [What Changed Since About 1995](#what-changed-since-about-1995)
  - [Modern Python And Tooling Primer](#modern-python-and-tooling-primer)
  - [Pipeline Landmarks](#pipeline-landmarks)
- [Learning Path](#learning-path)
  - [1. Array Programming Languages](#1-array-programming-languages)
  - [2. Rank Polymorphism And Remora Semantics](#2-rank-polymorphism-and-remora-semantics)
  - [3. Static And Dependent Type Systems](#3-static-and-dependent-type-systems)
  - [4. Functional Programming And Language Implementation](#4-functional-programming-and-language-implementation)
  - [5. Parsing, ASTs, And Frontends](#5-parsing-asts-and-frontends)
  - [6. Compiler IRs And Program Transformations](#6-compiler-irs-and-program-transformations)
  - [7. MLIR And Structured Lowering](#7-mlir-and-structured-lowering)
  - [8. CPU Code Generation](#8-cpu-code-generation)
  - [9. CPU Vectorization And Multicore Execution](#9-cpu-vectorization-and-multicore-execution)
  - [10. GPU Programming And CUDA Execution](#10-gpu-programming-and-cuda-execution)
  - [11. Descriptor ABI, Runtime Values, And Views](#11-descriptor-abi-runtime-values-and-views)
  - [12. GPU Execution Plans And Kernel Routing](#12-gpu-execution-plans-and-kernel-routing)
  - [13. Automatic Differentiation](#13-automatic-differentiation)
  - [14. Parallel Algorithms For Array Primitives](#14-parallel-algorithms-for-array-primitives)
  - [15. Testing Compilers And Numeric Backends](#15-testing-compilers-and-numeric-backends)
  - [16. Performance Engineering And Benchmarking](#16-performance-engineering-and-benchmarking)
- [Topic Map By Project Area](#topic-map-by-project-area)
- [Suggested Reading Order](#suggested-reading-order)
- [Minimal Background Checklist](#minimal-background-checklist)
- [Deeper Reading List](#deeper-reading-list)
- [Practical Exercises](#practical-exercises)

## Start Here

This guide is intentionally broad, but most readers should not read it
front-to-back on the first pass. Use the goal-based paths below, skim the
foundation rows in the delta table, and spend close attention on the modern or
RemoraC-specific rows.

### Minimum Viable Path

If you want the shortest useful path through the project:

1. Read `docs/USER_GUIDE.md` and run two examples: one with `--target interp`
   and one with the default compiled CPU target.
1. Read sections 1, 2, 7, 11, and 15 in this guide.
1. Read `docs/PROJECT_OVERVIEW_AND_ARCHITECTURE.md`, `docs/DENSE_CORE.md`, and
   `docs/ABI.md`.
1. Follow one small program through `--emit-ast`, `--emit-typed-ast`,
   `--emit-hir`, and `--emit-mlir`.
1. Run `uv run python -m compileall -q remora` before and after a small edit.

Expected time: one focused day for orientation, two to three days to read the
core docs and trace the pipeline, and one to two weeks to become productive in
a backend or typechecker area.

Prerequisites to check up front:

- You can read Python classes with type annotations and `@dataclass`.
- You are comfortable with basic functional-programming vocabulary: lexical
  scope, closures, higher-order functions, and recursion.
- You know the difference between scalar values, vectors, matrices, and
  higher-rank tensors.
- You can read a simple compiler pipeline diagram and inspect intermediate
  representations without expecting source-level structure to survive intact.

### Goal-Based Entry Points

| Goal                         | Read first                         | Then study                                                                                                              | Skip or defer                                         |
| ---------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Use Remora programs          | `docs/USER_GUIDE.md`, sections 1-2 | `docs/DENSE_CORE.md`; examples in `examples/`                                                                           | MLIR, GPU routing, compiler QA                        |
| Work on the typechecker      | Sections 1-3                       | `remora/typechecker.py`, `remora/types.py`, `remora/constraints.py`, Slepak dissertation                                | CPU vectorization and GPU execution plans             |
| Work on CPU compilation      | Sections 6-8 and 11                | `docs/ABI.md`, `remora/lowering/`, `remora/pipeline.py`, `remora/runtime.py`                                            | Full Remora dynamic shapes                            |
| Work on GPU lowering         | Sections 10-12 and 15              | `docs/BACKEND_GAPS.md`, `remora/codegen.py`, `remora/gpu_lowering.py`, `remora/_gpu_expr_lowering.py`, GPU parity tests | First-principles functional compiler material         |
| Work on AD                   | Section 13 plus sections 1-2 and 6 | `examples/ad_*.lisp`, `remora/ad.py`, HIR optimization, backend support limits                                          | Introductory backprop material if you already know it |
| Plan future full Remora work | Sections 2-3, 11-12, 14-16         | `docs/ROADMAP.md`, `docs/PLAN_TO_IMPLEMENT_FULL_REMORA.md`, records/ragged/dynamic-shape papers                         | Current CPU ABI minutiae until needed                 |

### What Changed Since About 1995

For a CS/AI reader returning after a long interval, the efficient path is a
delta, not a from-scratch course. The top table lists material you likely still
own; skim those sections for RemoraC-specific constraints. The second table is
where most of the new or substantially changed material lives.

| Foundations you likely already own                                                            | Guide section | How to read it                                                                           |
| --------------------------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------- |
| Lambda calculus, function types, type soundness                                               | 3             | Skim the refresher; focus on shape-indexed typing and erasure.                           |
| Closures, lexical scope, free variables, lambda lifting, defunctionalization                  | 4             | Read "Why it matters here" and the RemoraC lowering notes.                               |
| Recursion, mutual recursion, tail calls, call graphs and SCCs                                 | 4             | Standard material; only the CPU/interpreter support rules are project-specific.          |
| CFGs, LALR parsing, parser generators, AST vs parse tree, desugaring                          | 5             | Skim unless changing grammar or reader code.                                             |
| CSE, DCE, inlining, ANF, IR design                                                            | 6             | Classic compiler material; focus on this project's AST -> HIR -> MLIR handoff.           |
| AD fundamentals: forward/reverse mode, tapes, VJPs, gradient descent                          | 13            | As an AI reader, jump to array cotangents and source generation.                         |
| Parallel skeletons: tree reduction, prefix sum, sort, scatter/gather, tiling, matmul blocking | 14            | Skim for how these appear as Remora primitives and backend tests.                        |
| SIMD, cache locality, memory vs compute bound, roofline intuition                             | 9, 16         | The ideas are familiar; study current MLIR/LLVM/vector tooling and hardware constraints. |

| New or materially different                                                       | Guide section | What to study closely                                                                                  |
| --------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------ |
| Rank polymorphism as a typed semantic discipline                                  | 1-2           | Principal frames, type-directed dynamic semantics, and shape soundness.                                |
| Bidirectional typechecking                                                        | 3             | The infer/check split and why it helps higher-rank and shape-indexed programs.                         |
| Restricted dependent/indexed types for shapes                                     | 3             | DML/Xi-style indexed dimensions, shape constraints, and erasure that still preserves runtime metadata. |
| MLIR: dialects, regions, `linalg.generic`, progressive lowering                   | 7             | This did not exist in 1995; it is the central modern compiler substrate here.                          |
| Tensor-to-memref bufferization and `llvm.emit_c_interface`                        | 7-8           | Value-semantic tensors becoming ABI-compatible buffers.                                                |
| GPU/CUDA/SIMT, warps, coalescing, occupancy                                       | 10            | The execution and memory model behind the CUDA target.                                                 |
| PTX, NVVM, and LLVM's NVPTX backend                                               | 10            | The target stack below GPU MLIR lowering.                                                              |
| Multi-kernel execution plans, fusion, device-resident buffers, capability routing | 12            | Modern array-compiler architecture with little direct 1990s analogue.                                  |
| Descriptor ABI: aligned pointer, offset, sizes, strides                           | 11            | The exact contract shared by CPU, GPU, ctypes, and views.                                              |
| AD by source generation into the normal pipeline; array-op VJPs                   | 13            | RemoraC's implementation strategy for gradients over rank-polymorphic array code.                      |
| Silent-miscompile testing: oracle, differential, metamorphic, expected rejection  | 15            | The project-specific correctness discipline, especially for GPU lowering.                              |
| Modern Python tooling: dataclasses, type hints, `uv`, `pytest`                    | This section  | The mechanics needed to read and safely modify this Python compiler.                                   |

### Modern Python And Tooling Primer

RemoraC is a Python compiler, not a Python DSL. Most source modules are ordinary
Python 3.11+ code using dataclasses, type annotations, pytest tests, and `uv`
for environment management.

| Tool or idiom    | What to know here                                                                                                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `@dataclass`     | AST, type, and IR nodes are mostly immutable-looking records with named fields. Read them as algebraic data type variants, even though Python does not enforce that model as tightly as ML or Haskell. |
| Type hints       | Annotations document expected shapes of compiler data structures and help readers; there is no configured type checker in this repo. Do not assume `mypy` has enforced them.                           |
| Pattern matching | Some modules use `isinstance` cascades rather than ML-style pattern matching. Treat each branch as one constructor case.                                                                               |
| `uv`             | `uv sync` installs dependencies; `uv run ...` executes commands inside the managed environment. Prefer the documented `uv run` commands over invoking system Python.                                   |
| `pytest`         | Tests are plain pytest. GPU tests run by default when the environment supports them; `REMORA_TEST_GPU=0` only allows unavailable GPU support to skip.                                                  |
| `compileall`     | `uv run python -m compileall -q remora` is the fast syntax/import sanity check because no linter, formatter, or static type checker is configured.                                                     |

When reading dataclass IR nodes, start from the type definitions, then search
for construction sites, then search for consumers. For example, read a HIR node
in `remora/hir.py`, find where `elaborate.py` or optimizer passes create it,
and then inspect the lowering or interpreter branch that consumes it.

### Pipeline Landmarks

Keep this map nearby while reading the detailed sections:

```text
source (.remora / .lisp)
  -> parser.py / lisp_reader.py
  -> ast_nodes.py
  -> typechecker.py
  -> elaborate.py
  -> hir.py
  -> hir_opt.py / defunc.py
  -> remora/lowering/* for MLIR
  -> pipeline.py and runtime.py for CPU
  -> codegen.py, gpu_lowering.py, _gpu_expr_lowering.py for GPU
```

Two contracts recur throughout the guide:

- Descriptor ABI: the canonical treatment is section 11 and `docs/ABI.md`.
  Other sections mention descriptors only to show where that contract is used.
- Correctness testing: the canonical treatment is section 15. GPU compile-only
  tests are smoke tests; numeric parity against the interpreter is the standard
  for lowering correctness.

## Learning Path

### 1. Array Programming Languages

Remora is an array language in the APL/J/K family. The central idea is that
programs operate on whole arrays, and ordinary function application can imply
iteration over array structure.

#### Overview

Array programming starts from a different default mental model than scalar
imperative programming. In C, Fortran, Java, or ordinary ML, a scalar value is
usually the unit of computation and arrays are containers that loops traverse.
In APL-derived languages, arrays are the ordinary values and scalar values are
just arrays of rank 0. Addition, comparison, selection, reduction, scan,
transpose, reshape, and indexing are operations on whole array values. The
programmer writes at the level of the mathematical object, while the
implementation is responsible for deciding how to traverse memory and where
parallelism is available.

The basic vocabulary is rank, shape, axis, and element. The rank of an array is
the number of dimensions in its shape. A scalar has rank 0 and shape `[]`; a
length-5 vector has rank 1 and shape `[5]`; a 3-by-4 matrix has rank 2 and shape
`[3, 4]`. The axes are the dimensions named by positions in the shape. With
row-major layout, the rightmost index varies fastest in contiguous memory, so
the rows of a matrix are contiguous and the columns are strided. These terms are
simple, but they become important because many array operations are best
understood as shape transformations rather than as loops. A transpose permutes
axes. A reshape changes the shape while preserving the linear sequence of
elements when possible. A ravel collapses all axes into one. A slice restricts
one or more axes. A view may change the apparent shape, offset, or strides
without copying the underlying data.

Array programming also changes how one thinks about iteration. A loop such as
`for i in range(n): c[i] = a[i] + b[i]` is not a primitive idea in these
languages; it is an implementation of elementwise addition. The source-level
program says that two arrays are added. If their shapes agree under the
language's rules, the operation denotes an array of results. Similarly, a
reduction says that an associative or at least binary operation combines cells
along an axis or frame; a scan says that all prefixes are combined; a stencil
says that local neighborhoods are transformed into output cells. The programmer
is expected to reason about whole-array transformations, and the compiler is
expected to recover the loops, vector instructions, kernels, temporary buffers,
and memory accesses.

The APL tradition is compact and sometimes cryptic, but its central technical
insight is not cryptic: many data-parallel computations have regular shape
structure. A scalar operation can often be lifted over every element of an
array. A vector operation can often be lifted over every row of a matrix or
every cell of a higher-rank tensor. A reduction over many independent rows can
run those row reductions in parallel. A transpose or reshape can often be
represented by metadata rather than by physically moving data. When the shape is
known statically, the compiler can generate loops with fixed bounds, specialize
descriptor layouts, and reject shape errors before execution.

Remora inherits this way of thinking but gives it a typed, rank-polymorphic
form. The key distinction is between a cell and a frame. A function is written
as consuming cells of a particular rank. When the function is applied to a
higher-rank argument, the remaining leading dimensions form the frame over which
the function is applied. For example, a function written for scalar cells can be
applied elementwise to a vector, matrix, or rank-3 tensor. A function written
for vector cells can be applied to each row of a matrix, or to each vector cell
inside a rank-3 tensor. This idea is older than Remora, but Remora makes it a
semantic and type-theoretic organizing principle rather than a collection of
special cases.

For a compiler, this has immediate consequences. The source program may not
contain the loops that are eventually executed. The compiler must infer or check
the shapes that determine those loops, elaborate implicit lifting into explicit
operations, and preserve enough shape information through intermediate
representations to lower the program correctly. It is not enough to know that an
expression has type "array of floats"; the compiler must know whether the
function is being applied to scalar cells, vector cells, matrix cells, or some
other suffix of the shape. Otherwise it cannot know which dimensions are
iteration dimensions and which dimensions are part of each operand consumed by
the function body.

This is why array languages expose parallelism naturally. A program written as
`map f xs`, an elementwise expression, or a row-wise reduction already says that
many independent or regularly dependent computations exist. The independence is
not discovered by heroic loop analysis after the fact; it is part of the source
semantics. The hard compiler problems do not disappear, but they move. Instead
of asking whether a scalar loop can be vectorized, the compiler asks how a
known whole-array operation should be represented: as a `linalg.generic`, as an
explicit loop nest, as a GPU kernel, as a runtime library call, as a descriptor
view, or as part of a larger fused computation.

The most important habit to reacquire when reading Remora code is therefore to
ask shape questions first. What is the rank of each value? Which dimensions are
cell dimensions? Which dimensions are frame dimensions? Is this operation
materializing a new array, or merely changing a descriptor? Does this reduction
collapse a dimension, preserve a dimension as a scan, or produce an aggregate
cell? Those questions are the array-language equivalent of asking about binding,
evaluation order, and control flow in a scalar functional language.

Example to keep in mind: a scalar function applied to a shape `[2, 3]` array
has frame `[2, 3]` and scalar cells. A vector function expecting shape `[3]`
cells, applied to that same array, has frame `[2]` and runs once per row. The
same physical data can therefore induce different iteration spaces depending on
the function's cell rank.

Study these concepts:

- Arrays as the default value model: scalars are rank-0 arrays.
- Rank, shape, axes, cells, frames, and row-major layout.
- Whole-array programming instead of explicit scalar loops.
- Lifting scalar or cell functions over higher-rank inputs.
- Reductions, scans, stencils, transposes, reshapes, and views.
- Why array languages expose parallelism naturally.

Why it matters here: RemoraC's parser, typechecker, elaborator, HIR, and
lowering all preserve the distinction between the cell a function consumes and
the frame over which the function is lifted.

Recommended reading:

- `docs/remora-reference/remora-tutorial-draft.txt` for Remora-specific array
  notation and examples.
- `docs/remora-reference/intro-rank-polymorphic-programming-remora.txt` for the
  bridge from APL-style intuition to rank-polymorphic programming.
- Iverson, *A Programming Language*, for the historical source of whole-array
  thinking.
- APL/J/K tutorials for intuition about rank, shape, scan, reduce, and tacit
  array thinking.

### 2. Rank Polymorphism And Remora Semantics

Rank polymorphism is the semantic core. A function is written for cells of some
rank, then application distributes it over frames of higher-rank arguments. With
multiple arguments, Remora computes a principal frame and replicates smaller
frames when valid.

#### Overview

Rank polymorphism is a disciplined account of a familiar array-language
behavior: the same operation can act on scalars, vectors, matrices, and
higher-rank arrays without the programmer writing a new loop for each rank. In
ordinary parametric polymorphism, a function may be generic in the element type.
In rank polymorphism, the function is also generic in the amount of surrounding
array structure over which it is applied. The function itself still has a cell
rank: it expects arguments whose shapes have certain suffixes. Application to
larger arrays is interpreted as iteration over the leading shape dimensions.

The central equation is:

```text
shape = frame ++ cell
```

The cell is the suffix of the shape consumed by the function body. The frame is
the prefix over which the application is repeated. If a scalar negation function
expects a rank-0 cell, applying it to a shape `[100, 20]` array gives frame
`[100, 20]` and cell `[]`; the function runs once for each scalar cell. If a
vector norm expects a rank-1 cell of length `20`, applying it to a shape
`[100, 20]` array gives frame `[100]` and cell `[20]`; the function runs once
for each row. If a matrix operation expects shape `[3, 3]` cells, applying it to
a shape `[10, 50, 3, 3]` array gives frame `[10, 50]` and cell `[3, 3]`.

Multiple arguments require a rule for frame agreement. Suppose a binary scalar
operation is applied to arrays with shapes `[10, 20]` and `[20]`. One possible
semantics is to reject the program, another is NumPy-style broadcasting from the
right, and another is Remora's principal-frame account. In Remora, arguments
are decomposed according to the cell ranks expected by the function, and their
frames must agree through a principal-frame relation. Smaller frames may be
replicated when the relation permits it. The important point is that this is
not an accidental runtime convenience. It is part of the dynamic semantics and
is tracked by the type system so that well-typed programs are shape safe.

Reranking is the explicit mechanism for changing the cell/frame boundary. If a
function is normally scalar, reranking can ask that it be applied to larger
cells. If a function is normally vector-valued, reranking can determine whether
it sees an entire vector, each row of a matrix, or each smaller cell inside a
higher-rank array. Reranking is not merely a performance hint. It changes the
meaning of application by changing which dimensions are consumed in a single
function call and which dimensions drive iteration.

For a semantics-minded reader, rank polymorphism is interesting because
ordinary application carries more information than in the simply typed lambda
calculus. In the lambda calculus, application reduces by substituting an
argument value into a function body. In Remora, application must also determine
how argument shapes are split, what frame is induced, how many body evaluations
occur, and how the results are assembled. The result shape is normally the
principal frame followed by the result cell shape. Thus the operational account
of application depends on type and shape information. This is why Remora's
semantics is often described as type directed: the type tells the evaluator
which rank of cell a function consumes, and the runtime shape tells it how many
such cells are present.

This type-directed character can feel unusual if one last studied programming
language semantics when most textbook examples were small call-by-value lambda
calculi. In those calculi, erasing types usually leaves the same dynamic
behavior, because types are only a static discipline. In Remora, types and
shape annotations help disambiguate array application. A formal presentation
therefore has to say not only what values exist and how expressions step, but
also how typed terms determine cell ranks, frame variables, principal frames,
and replication. Shape soundness then says that the static discipline is strong
enough to prevent a class of runtime shape failures.

In RemoraC, this semantics is implemented by making implicit structure explicit
as early as possible. The parser records the surface expression, but the
typechecker establishes scalar types, ranks, shapes, function types, and
constraints. The elaborator turns rank-polymorphic application into a core form
where frame and cell structure can be represented. Later passes lower that
structure to HIR nodes such as maps, folds, views, and applications whose shape
roles are less implicit. By the time MLIR or GPU code is emitted, the compiler
needs concrete loop bounds, descriptor shapes, and memory indexing formulas, not
an informal idea that an operation "broadcasts somehow."

This distinction is important when debugging or extending the compiler. A bug
in rank semantics often does not look like a syntax error or a simple type
error. It may appear as a result array with the right element type but the wrong
shape, a map body that is lowered as if it consumed scalars instead of vectors,
a view inside a map body that uses the wrong strides, or a GPU kernel that
computes correct values only for a top-level case. Because rank polymorphism
determines where loops are introduced, any misunderstanding of the frame/cell
split can propagate all the way to generated code.

A useful way to read the Remora papers is to separate three layers. First, the
mathematical layer defines arrays, shapes, cells, frames, principal frames, and
application. Second, the type-theoretic layer shows how these ideas can be
checked statically, including shape variables and constraints. Third, the
implementation layer asks how the compiler should elaborate and lower these
ideas without either losing information or hard-coding example-specific cases.
RemoraC lives in the third layer but is constrained by the first two.

Tiny shape example: if a function consumes vector cells of length `n`, then an
argument of shape `[m, n]` is treated as `m` cells, not as one matrix cell. If
the programmer wants the whole `[m, n]` value to be one cell, the expression
must use a function or reranking form whose cell rank is 2. This is a semantic
choice, not an optimization detail.

Study these concepts:

- Frame/cell decomposition: shape = frame prefix + cell suffix.
- Principal-frame agreement across arguments.
- Cell replication and broadcasting-like behavior.
- Reranking: changing which suffix dimensions are treated as cells.
- Type-driven dynamic semantics: type information identifies iteration spaces.
- Shape soundness: well-typed programs do not fail with shape errors at run
  time.

Why it matters here: `frame.py`, `elaborate.py`, `typechecker.py`, and
`hir.py` exist largely to turn implicit rank-polymorphic behavior into explicit
lower-level operations.

Recommended reading:

- `docs/remora-reference/semantics-of-rank-polymorphism.txt` for the formal
  frame/cell semantics.
- `docs/remora-reference/slepak-dissertation.txt` for the full type and
  semantics development.
- `docs/remora-reference/remora-overview.md` for a shorter orientation before
  reading the formal material.

### 3. Static And Dependent Type Systems

Remora's type system tracks scalar type, rank, and shape. The full language uses
dependent indices for dimensions and shapes. RemoraC currently compiles a dense
static core: dimensions are known by lowering time, even though the front end
has machinery pointing toward richer dependent shapes.

#### Overview

A static type system is a compile-time approximation of program behavior. In a
conventional typed functional language, the typechecker may prove that a
function expecting an integer is never applied to a string, or that a branch
expression returns a value of the same type on both arms. For an array language,
that is not enough. Most meaningful mistakes are not merely scalar type
mistakes; they are shape mistakes. Adding two floats is fine, but adding a
shape `[10, 20]` array to a shape `[11, 20]` array may be meaningless. Reducing
over a scalar cell is different from reducing over a vector cell. Reshaping
requires preservation of element count. Indexing requires bounds and rank
information. A useful type system for Remora must therefore track scalar type,
rank, and shape.

The simplest starting point is the simply typed lambda calculus: terms have
base types and function types, and application checks that the argument type
matches the domain type. Parametric polymorphism generalizes this by allowing
types to contain variables. A polymorphic identity function, for example, has a
type like `forall a. a -> a`. Higher-rank polymorphism allows `forall` types to
appear inside function arguments and results rather than only at the outermost
level. This matters for languages where functions can accept polymorphic
functions as values or return functions with polymorphic behavior. RemoraC's
dense CPU path supports higher-order functions and `ForallType`
monomorphization, so these ideas are not just theoretical background.

Dependent types go further by allowing types to mention values, or more often
some restricted compile-time index language. Full dependent type theory is
powerful enough to express arbitrary propositions, but production compilers
usually use restricted dependent or indexed systems because they must be
decidable and predictable. For Remora, the relevant values are dimensions,
shapes, ranks, and constraints over them. A vector type can be indexed by its
length; a matrix type can be indexed by row and column counts; a reshape can be
accepted only if the product of dimensions is preserved. The typechecker is
therefore partly an arithmetic and symbolic-constraint solver over shapes.

This kind of system sits between conventional Hindley-Milner typing and full
interactive theorem proving. It is richer than ordinary ML types because
programs carry dimension information in types. It is more restricted than Coq
or Agda because the compiler needs automatic checking and useful error
messages. The index language must support the expressions that arise from array
programming, but it cannot become an undecidable general-purpose logic. The
engineering question is always: which equalities and inequalities should the
compiler understand automatically, and when should it require annotations,
reject the program, or defer work to a future dynamic-shape representation?

Bidirectional typechecking is a practical method for controlling this
complexity. Instead of asking every expression to synthesize its type from
nothing, the checker alternates between inference and checking. In inference
mode, an expression produces a type. In checking mode, an expression is checked
against an expected type. Lambda expressions, polymorphic functions, and
dependent shapes are often much easier to check when an expected type is known.
This is one reason bidirectional systems are common in modern languages with
higher-rank or dependent features: they reduce annotation burden while keeping
the algorithm understandable.

Constraint generation and solving are the other half of the story. During
typechecking, the compiler may learn that two shape expressions must be equal,
that one frame must be a prefix of another, that a dimension variable must be
instantiated with a particular integer, or that a result shape is obtained by
combining a frame with a cell shape. These facts become constraints. Solving
them may involve unification, normalization of index expressions, substitution
of dimension variables, or specialized reasoning about products and shape
concatenation. In a static dense core, the goal is to know all dimensions by
lowering time. That lets the backend generate fixed-rank descriptors, concrete
loop nests, and type-specialized runtime calls.

Existential types and Sigma types are relevant because array programs sometimes
hide shape information. A function may produce an array whose shape depends on
data, or a box may package a value while hiding the exact type or shape from
some part of the program. Full Remora work points toward richer dynamic shapes,
boxes, and possibly ragged arrays. RemoraC's current dense core does not fully
implement that world at runtime, but the type-system architecture contains
machinery that points in that direction. When you see boxes in the current CPU
path, remember that they are largely type-erasure constructs rather than a full
runtime existential package system.

Type erasure is the process of removing compile-time information after it has
served its purpose. In many typed languages, types are erased before code
generation because the generated code does not need them. In RemoraC, erasure
must be more careful. Some shape information can be erased because it is
already reflected in fixed loop bounds or descriptor fields. Some information
must survive as runtime metadata: a descriptor needs sizes and strides; an
entry point needs an ABI-compatible representation; a view needs offset and
stride changes. The key is to erase logical type detail while preserving the
operational facts needed for execution.

Monomorphization is another bridge from rich types to executable code. A
polymorphic function may be conceptually one source-level definition, but
native code often needs concrete versions specialized to particular scalar
types, ranks, shapes, or function arguments. Higher-order functions complicate
this further because passing a function as an argument may require
specialization of the callee for the actual function value. RemoraC's CPU path
handles `ForallType`, higher-order function monomorphization, and closure
capture. This is one reason the typechecker and compiler pipeline are more like
a real functional-language compiler than a thin array-expression translator.

The practical payoff of the type system is shape soundness. A well-typed dense
RemoraC program should not fail at runtime because two static shapes disagree
or because a backend discovers too late that it does not know a rank. When the
system is extended, each new feature should be judged by whether it preserves
that property. If dynamic shapes are added, then the soundness statement will
have to include runtime checks or richer runtime evidence. If ragged arrays are
added, shape regularity assumptions will need revision. If dynamic higher-order
dispatch is added, monomorphization will no longer be the whole story. The
current static dense core is a deliberately tractable point in this design
space.

Concrete type-level intuition: a reshape from shape `[2, 3]` to shape `[6]`
is valid because both shapes describe six elements. A reshape from `[2, 3]` to
`[5]` should be rejected statically in the dense core. That rejection is not a
backend convenience; it is the type system preventing an impossible runtime
layout interpretation.

Study these concepts:

- Simply typed lambda calculus and function types.
- Parametric polymorphism and higher-rank polymorphism.
- Bidirectional typechecking: checking against an expected type vs inferring a
  type.
- Indexed and dependent types, especially restricted index languages.
- Shape variables, dimension variables, constraints, unification, and
  normalization.
- Existential types/Sigma types for hidden shapes and boxes.
- Type erasure: removing type-level detail while preserving enough runtime
  shape information for execution.
- Monomorphization: cloning polymorphic or higher-order functions for concrete
  argument types.

Why it matters here: the largest source module is `remora/typechecker.py`.
RemoraC's CPU path handles higher-order functions, closure capture,
`ForallType`, recursion, and dense static shapes. Dynamic shapes, runtime boxes,
ragged arrays, and full dynamic higher-order dispatch are future work.

Recommended reading:

- Justin Slepak, *A Typed Programming Language: The Semantics of Rank
  Polymorphism* (`docs/remora-reference/slepak-dissertation.txt`), for Remora's
  dependent shape and rank-polymorphic type theory.
- Dunfield and Krishnaswami, *Complete and Easy Bidirectional Typechecking for
  Higher-Rank Polymorphism*, for the bidirectional checking algorithmic style.
- Xi and Pfenning, *Dependent Types in Practical Programming*, for restricted
  dependent typing as a programming-language design technique.
- Pierce, *Types and Programming Languages*, for the lambda-calculus and type
  soundness foundations.
- Chlipala, *Certified Programming with Dependent Types*, for broader context
  on dependent proofs and programs.

### 4. Functional Programming And Language Implementation

RemoraC is not just an array-kernel generator. The dense CPU/interpreter subset
supports lambdas, closures, higher-order functions, monomorphized function
arguments, recursion, local bindings, and AD-generated programs.

#### Overview

A language with arrays alone could be implemented as an expression compiler:
parse array operations, lower them to loops or kernels, and stop there. RemoraC
has to do more because Remora is also a functional language. Functions are
values, lambdas can appear in expressions, local bindings introduce lexical
scope, functions can be passed to higher-order combinators such as `map` and
`fold`, and recursive definitions are part of the dense CPU/interpreter subset.
Understanding the compiler therefore requires both array-language concepts and
the implementation techniques of functional programming languages.

Lexical scope is the starting point. A variable occurrence refers to the
nearest enclosing binding with the same name, not to a dynamically chosen caller
environment. A lambda expression may mention variables defined outside the
lambda. Those variables are its free variables, and the runtime representation
of the lambda must preserve their values. A closure is the usual representation:
code pointer plus environment. For example, a function that returns `lambda x. x + y` must remember the particular `y` in scope when the lambda was created.
The source program treats the result as a function value; the implementation
must decide how that value is represented, passed, specialized, or eliminated.

Closure conversion is the compiler transformation that makes this explicit. It
rewrites functions so that free variables become fields in an environment or
extra explicit arguments. Lambda lifting is a related transformation that moves
nested functions to top level by adding parameters for the variables they used
to capture. These transformations are standard in functional compilers because
low-level targets such as C, LLVM, MLIR LLVM dialect, or PTX do not directly
implement lexical closures as source-level language values. The compiler must
choose a representation compatible with the target ABI and with optimization
passes.

Defunctionalization takes a different approach to higher-order programs. Rather
than representing arbitrary function values as closures, the compiler replaces
the finite set of possible function values with a first-order data type of
tags, plus an apply function that dispatches on those tags. In an optimizing
compiler, defunctionalization is often combined with specialization so the
dispatch can disappear for known cases. This is especially relevant for array
languages and GPU compilation because arbitrary dynamic function calls are
awkward or impossible in many kernel-generation paths. If a `map` receives a
known function argument, the compiler would rather inline or specialize the map
body than generate a general runtime function-call mechanism inside every
element operation.

Higher-order functions interact with rank polymorphism in a particularly rich
way. A `map` is not just a loop; it is a higher-order operator that receives a
function and applies it across cells. A `fold` receives a combining function and
uses it according to a reduction structure. Reranking can change what cells the
function sees. Polymorphism can make the same higher-order definition apply to
different element types and shapes. The compiler must preserve the functional
meaning while eventually lowering to first-order loops, regions, kernels, or
runtime calls. That is why RemoraC includes both type-driven monomorphization
and defunctionalization machinery.

Recursion adds another layer. Direct self-recursion can sometimes be compiled
as an ordinary function call. Mutual recursion requires call-graph analysis to
group strongly connected components and make sure all functions in the cycle
are available to one another. Tail recursion can be transformed into a loop
because the recursive call is the final action of the function. Non-tail
recursion requires preserving pending work on the call stack or using another
explicit representation. The interpreter uses trampolining for deep tail-style
behavior so that Python's own recursion limit does not define Remora's
semantics. Native compilation must make corresponding decisions in the lowering
pipeline.

Local bindings are semantically simple but important operationally. A `let`
names an intermediate result and scopes it over a body. In a pure language,
this is close to substitution, but compilers rarely implement it by naive
substitution because that duplicates work and obscures sharing. Instead, a
lower-level IR often makes evaluation order and binding explicit. This matters
for array programs because a named intermediate may represent a scalar, a
descriptor view, a materialized array, a closure, or a value that should be
common-subexpression eliminated. The HIR and later lowering stages must know
which of those cases they are handling.

An interpreter is often the clearest executable specification of a language.
For RemoraC, this is not just a convenience for users. The interpreter is the
semantic oracle used by tests, especially for backend numeric parity. If the
compiler and interpreter disagree, the presumption should be that the compiler
or a backend lowering path is wrong unless the interpreter is known to be
outside the implemented contract. This oracle role is especially important for
GPU lowering, where compile-only tests can miss silent miscompiles. The
functional features must therefore be implemented consistently across the
interpreter, elaboration, HIR lowering, defunctionalization, MLIR generation,
and runtime execution.

Automatic differentiation also stresses the functional-language side of the
system. AD-generated programs may contain many local bindings, generated helper
functions, state-like loops, and expressions produced by transformation rather
than written by a human. A compiler that only recognizes friendly surface
examples will fail here. The functional core must be general enough that
generated programs pass through the same semantic machinery as hand-written
programs. This is one reason the project emphasizes no example-specific
lowering code.

For a reader returning to programming-language implementation after several
decades, the main modernization is not that the old concepts disappeared.
Lexical scope, closures, lambda lifting, recursion, and interpreters are still
central. What changed is the composition pressure. These ideas now coexist with
shape-dependent typing, array semantics, MLIR lowering, descriptor ABIs, GPU
kernels, and numeric parity testing. RemoraC is a compact example of that
composition: a functional language compiler whose most important values are
statically shaped arrays and whose backends must exploit the regularity those
types expose.

Study these concepts:

- Lexical scope, environments, closures, and free variables.
- Lambda lifting and closure conversion.
- Defunctionalization: replacing higher-order values with first-order tags or
  specialized functions.
- Recursion, mutual recursion, tail calls, trampolines, and call graphs.
- Let-polymorphism and top-level definition handling.
- Interpreters as executable specifications.

Why it matters here: the interpreter is the semantic oracle. The compiler must
preserve that behavior through elaboration, HIR lowering, defunctionalization,
MLIR generation, and native execution.

Recommended reading:

- Friedman and Wand, *Essentials of Programming Languages*, for interpreters,
  environments, closures, and semantic models.
- Appel, *Modern Compiler Implementation*, for closure conversion, calling
  conventions, and functional-language compilation.
- Futhark defunctionalisation paper in `docs/remora-reference/` for a nearby
  array-language treatment of higher-order elimination.
- `docs/IMPLEMENT_RECURSION.md` for this project's recursion implementation
  notes.
- `docs/IMPLEMENTATION_LOG.md` for historical context on how the functional
  subset evolved.

### 5. Parsing, ASTs, And Frontends

RemoraC has two source syntaxes: ML-like `.remora` and Lisp-like `.lisp`. Both
lower to one AST, which prevents language semantics from splitting across
frontends.

#### Overview

The frontend of a compiler is the part that turns source text into a structured
program representation. That sounds mechanical, but it establishes an important
architectural boundary. The parser should recognize the surface grammar,
recover source locations, report syntax errors, and construct an AST. It should
not be the place where each backend invents its own interpretation of the
language. In RemoraC, this boundary is especially important because the project
has two concrete syntaxes: an ML-like syntax for `.remora` files and a
Lisp-like syntax for `.lisp` files. They must converge quickly, or the language
would effectively split into two languages with subtly different behavior.

A context-free grammar describes how tokens form phrases: expressions,
definitions, literals, type annotations, applications, operator sections, view
forms, and so on. Parser generators such as Lark take such a grammar and build
a parser that can recognize source programs. The ML-like syntax has the usual
concerns of infix operators, precedence, associativity, parentheses, layout-like
readability, and ambiguity between related constructs. A parser must determine
where an expression starts and ends before the typechecker can determine what
it means. For example, function application, indexing, operator sections, and
array literals may all use compact notation; the grammar must make these
distinctions without relying on later semantic guesses.

LALR parsing is a deterministic bottom-up parsing technique used by many parser
generators. The practical issue is conflict management. If the grammar allows
the parser to either shift another token or reduce an existing phrase, the tool
reports a shift/reduce conflict. If two reductions are possible, it reports a
reduce/reduce conflict. Some conflicts can be resolved by precedence
declarations; others indicate that the grammar is ambiguous or too clever. For
RemoraC, a clean grammar matters because the parser is not the right place for
semantic backtracking. The frontend should produce one clear AST or a useful
diagnostic.

The Lisp-like frontend has a different profile. S-expressions are syntactically
simple: a program is a tree of atoms and lists. That simplicity moves work from
parsing into syntax-directed translation. Once the reader has an S-expression,
the frontend must interpret list heads such as `let`, `lambda`, `define`,
`map`, `fold`, or primitive operators and construct the same AST nodes that the
ML parser constructs. This is often easier to extend, but it still requires
careful validation. The reader must reject malformed special forms, wrong
argument counts, obsolete syntax, and ambiguous constructs before the
typechecker sees them.

An abstract syntax tree is not merely a parse tree. A parse tree mirrors the
grammar: it contains nodes for punctuation, precedence productions, and
syntactic conveniences. An AST represents the program in terms useful to later
compiler phases. Parentheses disappear, operator precedence has already been
resolved, and equivalent surface constructs may be represented by the same
node. RemoraC's `ast_nodes.py` is the shared AST vocabulary for both frontends.
That shared vocabulary includes ordinary language forms such as variables,
literals, lets, lambdas, applications, and definitions, plus array-specific
forms such as map, fold, scan, transpose, reshape, ravel, take, drop, sort,
grade, matmul, boxes, pairs, and AD forms.

Desugaring is the process of translating convenient surface notation into a
smaller or more regular core. In some compilers, desugaring is a separate pass.
In smaller systems, parts of it happen during AST construction. The important
principle is that surface syntax should not multiply semantic cases
unnecessarily. Operator sections, shorthand forms, and alternative syntactic
spelling should become explicit AST forms or be translated into ordinary
lambda/application forms before the compiler gets too deep. The later pipeline
should not need to remember whether a construct was written with ML-like
surface sugar or Lisp-style prefix notation.

Source locations are another frontend responsibility that becomes more
important as the type system grows richer. A syntax error can usually point to
a token. A type error involving rank, shape constraints, polymorphic
instantiation, or higher-order functions may originate from a much larger
expression. Good diagnostics require carrying source spans forward into the
AST and sometimes through typed representations. Even if the current
implementation's diagnostics are modest, the architecture should avoid losing
locations too early.

Prelude injection is part of the frontend contract in RemoraC. User programs
are not compiled in isolation; `stdlib/prelude.rem` is automatically prepended.
That means names from the prelude participate in parsing, typechecking,
elaboration, and backend lowering as if they were part of the source program.
The frontend must also infer or honor syntax selection: `.remora` and `.lisp`
files enter through different readers but converge on one AST. When debugging a
program, remember that the visible source may not be the whole source presented
to the compiler.

The main design risk in a multi-syntax frontend is semantic drift. If the ML
parser handles a construct one way and the Lisp reader handles it another way,
tests may pass for one syntax and fail for the other. The shared AST is the
defense against this. The typechecker should not care which concrete syntax
produced a `MapExpr` or `LetExpr`; it should see the same node fields and apply
the same rules. When adding a feature, check both frontends if the feature is a
surface-language construct. When changing semantics, change the shared phases,
not just one parser.

For someone returning to compiler construction after a long interval, the old
frontend lessons still hold, but the tolerance for ad hoc parsing is lower in a
compiler like this. The language has two syntaxes, typed rank polymorphism,
dependent shape machinery, and multiple backends. A small surface ambiguity can
become a typechecker confusion; a missing AST distinction can become a backend
special case; a lost source span can make a shape error incomprehensible. A
boring, explicit frontend is a strength.

Study these concepts:

- Context-free grammars and parser generators.
- LALR parsing and ambiguity management.
- S-expression parsing and syntax-directed AST construction.
- Source locations and diagnostics.
- Desugaring surface forms into a shared core.

Why it matters here: `remora/parser.py`, `remora/grammar.lark`, and
`remora/lisp_reader.py` feed the same AST classes in `remora/ast_nodes.py`.
Prelude injection and syntax inference are part of the frontend contract.

Recommended reading:

- Lark parser documentation for the parser generator and grammar mechanics used
  by the ML-like frontend.
- Aho, Lam, Sethi, and Ullman, *Compilers: Principles, Techniques, and Tools*,
  for grammar, parsing, and frontend background.
- `docs/USER_GUIDE.md` for the concrete surface syntax that the frontends
  accept.

### 6. Compiler IRs And Program Transformations

RemoraC has several internal representations. Each one makes more structure
explicit and removes source-level convenience before lowering to MLIR or GPU
kernels.

Pipeline:

```text
source
  -> AST
  -> Typed AST
  -> elaborated core
  -> erased/backend core
  -> HIR
  -> optimized HIR
  -> MLIR or direct GPU LLVM-dialect text
  -> native CPU artifact or PTX
```

#### Overview

An intermediate representation is a compiler's working language. Source
languages are designed for humans; target languages are constrained by
machines, ABIs, and toolchains. IRs sit between them. They make some facts more
explicit, remove surface conveniences, expose optimization opportunities, and
separate language semantics from backend mechanics. A serious compiler usually
has more than one IR because no single representation is ideal for parsing,
typechecking, optimization, structured array lowering, and native code
generation.

The AST is closest to source syntax. It records what the user wrote in a
language-oriented form: variables, lambdas, applications, array operations,
definitions, and literals. A typed AST decorates or pairs that structure with
the results of typechecking: scalar types, function types, array ranks, shapes,
polymorphic instantiations, and constraints. This is the first point where many
source terms become fully meaningful. In Remora, an application cannot be
compiled correctly until the compiler knows the function's cell expectations
and the argument shapes that determine the frame.

A core language is usually smaller and more explicit than the source language.
Elaboration translates the rich surface or typed language into this core. In a
dependently typed or indexed setting, elaboration may insert implicit
arguments, evidence, coercions, instantiations, or explicit frame/cell
structure that the source did not spell out. The point is not merely to lower
syntax; it is to turn type-directed meaning into explicit program structure.
For RemoraC, elaboration is where rank-polymorphic behavior starts becoming
something the backend can understand rather than an implicit rule attached to
ordinary application.

Erasure then removes information that was needed for typechecking but should
not survive as full logical structure at runtime. This does not mean throwing
away all shape information. It means separating type-level explanation from
operational representation. Static dimensions may become loop bounds;
descriptor fields may carry sizes and strides; some polymorphic abstractions
may be specialized away. The backend core after erasure is closer to executable
array computation than to a formal typing derivation.

HIR is the main compiler boundary before backend lowering in RemoraC. It names
the operations the backends are expected to support: maps, folds, scans, traces,
conditionals, primitive operations, calls, lets, views, sorting, matrix
multiplication, boxes, pairs, and related array constructs. When backend
support is discussed in project docs, it is usually expressed in terms of HIR
nodes because HIR is the shared contract between language-level compilation and
machine-level lowering. If a construct reaches HIR, a backend either needs to
lower it correctly, route it to a supported plan, or reject it loudly.

Program transformations operate on these IRs. Common subexpression elimination
finds repeated computations and shares them. Dead code elimination removes
bindings whose results are unused. Inlining substitutes a known function body at
a call site. Monomorphization clones polymorphic functions for concrete types
or shapes. Defunctionalization removes higher-order values by replacing them
with first-order representations. Call-graph analysis identifies recursion and
strongly connected components. Each transformation is easier or harder
depending on the IR. For example, CSE is much easier when expressions have a
regular representation and side effects are controlled; recursion analysis is
easier when functions and calls are explicit.

Administrative normal form is a useful concept even when a compiler does not
literally convert every program to textbook ANF. The idea is to name
intermediate computations so that evaluation order is explicit. Instead of a
large nested expression, the IR contains a sequence of lets binding simpler
expressions. This matters for code generation because low-level targets often
need named SSA values, explicit blocks, and clear dominance relationships. It
also matters for arrays because an intermediate may correspond to a materialized
buffer, a descriptor view, a scalar temporary, or a candidate for fusion.

Lowering is the process of translating from a higher-level IR to a lower-level
one. It is not necessarily a loss of structure; it is a change in vocabulary.
A `map` may lower to `linalg.generic`, an explicit `scf.for` loop nest, a GPU
kernel, or a runtime call depending on target and context. A `reshape` may
lower to descriptor reinterpretation rather than element movement. A `fold`
may lower to a loop, a tree reduction, or a multi-kernel plan. The essential
requirement is semantic preservation: the lower-level program must compute the
same result for every input covered by the source program's contract.

Testing strategy follows from the IR structure. Golden tests are useful when an
IR or MLIR text output should have a particular shape. They catch accidental
changes in lowering structure and are good for reviewing compiler output.
However, they are not a proof of correctness. A wrong program can have stable
golden output. Numeric and differential tests compare execution results against
an oracle, usually the interpreter in this project. For backend work,
especially GPU work, semantic tests are mandatory because the highest-risk bug
is a silent miscompile: the compiler accepts a program, generates code, and
returns the wrong value.

When reading the RemoraC pipeline, keep asking what each stage knows and what
it is allowed to forget. The parser knows syntax but not types. The typechecker
knows shapes and constraints but should not decide GPU routing. Elaboration
makes rank behavior explicit. HIR records operations in a backend-neutral form.
Optimization can simplify but must preserve shape and value semantics. MLIR and
GPU lowering commit to loops, descriptors, kernels, and ABI details. Many
compiler bugs are boundary bugs: a stage assumes a later stage still has
information that was erased, or a backend assumes an HIR node only appears in a
simple context when the frontend can actually generate it anywhere.

Study these concepts:

- AST vs typed AST vs intermediate representation.
- Core languages and elaboration.
- Administrative normal form and making evaluation order explicit.
- Common subexpression elimination and dead code elimination.
- Call graphs and strongly connected components.
- Lowering high-level array operations into explicit loops or kernels.
- Golden tests vs semantic/numeric tests.

Why it matters here: the HIR is the main compiler boundary before backend
lowering. Backend support is often stated in terms of HIR nodes such as
`HIRMap`, `HIRFold`, `HIRScan`, `HIRSort`, `HIRMatmul`, and view nodes.

Recommended reading:

- Muchnick, *Advanced Compiler Design and Implementation*, for classic
  optimization and IR design.
- Cooper and Torczon, *Engineering a Compiler*, for a modern compiler-pipeline
  view of IRs and transformations.
- `docs/PROJECT_OVERVIEW_AND_ARCHITECTURE.md` for RemoraC's actual stage
  boundaries and file map.
- `docs/WORKSTREAM_0_PLAN.md` for planned architecture and backend-routing
  cleanup.

### 7. MLIR And Structured Lowering

The CPU backend lowers HIR to standard MLIR dialects. MLIR is central to the
project because it represents tensors, loops, scalar arithmetic, memory
references, and LLVM lowering in a composable compiler stack.

#### Overview

MLIR is a compiler infrastructure designed around extensible intermediate
representations. Traditional compiler pipelines often have one fixed IR, such
as LLVM IR, and every frontend must eventually squeeze its program into that
form. LLVM IR is excellent for low-level optimization and code generation, but
it is too low-level to conveniently represent tensors, affine loop nests,
structured control flow, GPU kernels, or domain-specific operations. MLIR
addresses this by allowing many dialects to coexist in one module and by
providing systematic lowering paths from high-level structured dialects to
lower-level dialects and eventually LLVM.

The basic unit in MLIR is an operation. Operations have operands, results,
attributes, types, and optionally regions. A region contains blocks, and blocks
contain operations. This general structure can represent ordinary SSA
expressions, control-flow bodies, loop bodies, function bodies, and nested
computations. The familiar LLVM idea of SSA values is present, but MLIR's
regions make it possible to represent structured constructs directly rather
than immediately flattening everything into branches and phi nodes. This is one
reason MLIR is attractive for a language like Remora: maps, reductions,
conditionals, and loop nests can remain structured through several phases.

Dialects define families of operations and types. The `func` dialect represents
functions and calls. The `arith` and `math` dialects represent scalar integer,
floating-point, comparison, and mathematical operations. The `tensor` dialect
represents value-semantic tensors. The `linalg` dialect represents structured
linear algebra and generic loop-like tensor computations. The `scf` dialect
represents structured control flow such as `for` and `if`. The `memref` dialect
represents buffer-style memory references. The `llvm` dialect represents
operations close to LLVM IR. The `gpu` and `nvvm` dialects represent GPU-level
concepts and NVIDIA-specific operations. A lowering pipeline gradually replaces
higher-level dialect operations with lower-level ones.

`linalg.generic` is one of the key structured operations for array compilers.
It describes a computation over one or more input and output tensors or
buffers, using affine indexing maps to say how loop indices map to operand
indices. The body region describes the scalar computation for each output
element or reduction step. This representation is more informative than a nest
of raw loops because it exposes iteration spaces, parallel dimensions,
reduction dimensions, and indexing structure to compiler passes. For Remora
maps and many elementwise operations, this is a natural target.

Tensor semantics and memref semantics are deliberately different. A tensor is a
value: operations conceptually produce new tensor values. A memref is a view of
memory: operations read and write through references. High-level functional
array languages are usually closer to tensor semantics, while efficient native
code ultimately needs memory buffers. Bufferization is the process of turning
tensor programs into memref programs, deciding where buffers are allocated,
which operations can reuse memory, and how function boundaries pass data. This
is one of the central difficulties in lowering pure array programs to efficient
imperative code.

Function-boundary ABI issues matter because generated code must be callable
from outside the MLIR world. MLIR can represent rich tensors internally, but a
compiled shared library called through Python `ctypes` needs a concrete C ABI.
RemoraC uses a descriptor ABI: pointers, offsets, sizes, and strides. MLIR's
memref descriptors and `llvm.emit_c_interface` are relevant because they define
how structured values become C-callable function arguments. A mismatch at this
boundary can produce code that verifies internally but cannot be called
correctly from the runtime.

Pass pipelines are the operational heart of MLIR use. A pass may canonicalize
operations, run CSE, bufferize tensors, convert linalg to loops, lower
structured control flow, convert memrefs to LLVM-compatible representations, or
emit LLVM dialect. The order matters. Running a pass too early may destroy
structure needed by a later optimization; running it too late may leave illegal
operations for the target. RemoraC's pipeline relies on external tools such as
`mlir-opt` and `mlir-translate`, so the textual MLIR emitted by the compiler
must be accepted by MLIR's verifier before those passes can do useful work.

Textual MLIR generation is pragmatic but demanding. A builder API can protect
against malformed operations, but text generation is sometimes simpler to
debug, easier to inspect, and easier to adapt in a Python project. The cost is
that the compiler author must be precise about SSA names, operation syntax,
types, attributes, regions, block arguments, and indentation-like structure.
The MLIR verifier becomes the first line of defense. Passing verification only
means the IR is structurally valid; it does not mean the generated program
implements Remora semantics correctly.

Structured lowering is the key design idea to preserve. If the compiler lowers
too eagerly to low-level pointer arithmetic, it loses the opportunity for MLIR
to reason about loops, tensors, reductions, and memory effects. If it keeps
programs too high-level for too long, it may reach a target that cannot handle
the remaining operations. The right lowering strategy is staged: represent
array structure explicitly, use MLIR dialects that understand that structure,
then progressively lower to loops, buffers, LLVM, and native code.

For RemoraC, MLIR is not an abstract backend detail. It is the main CPU
lowering substrate. HIR nodes become MLIR operations or regions. Static shapes
become tensor and memref types. Scalar operations become `arith` or `math`
operations. Maps and reductions become structured operations or explicit loop
nests. Views become descriptor or memref transformations where possible. When
something goes wrong in CPU compilation, reading the emitted MLIR is often the
fastest way to determine whether the bug is in language lowering, MLIR typing,
bufferization assumptions, ABI conversion, or later LLVM/codegen tooling.

Study these concepts:

- MLIR modules, operations, regions, blocks, attributes, and types.
- `func`, `arith`, `math`, `tensor`, `linalg`, `scf`, `memref`, `llvm`, `gpu`,
  and `nvvm` dialects.
- `linalg.generic` and affine indexing maps.
- Tensor semantics vs memref/buffer semantics.
- Bufferization and function-boundary ABI issues.
- MLIR pass pipelines, canonicalization, CSE, fusion, and lowering to LLVM.
- Textual MLIR generation and verification.

Why it matters here: CPU lowering is text-based and centered in
`remora/lowering/`. The pipeline uses external tools such as `mlir-opt`,
`mlir-translate`, `llc`, and a C compiler/linker.

Recommended reading:

- MLIR official documentation: LangRef, Dialects, Tutorials, Bufferization, for
  the operation/region/dialect model and lowering pipeline.
- LLVM LangRef for the low-level IR target beneath MLIR.
- `docs/MLIR_IMPLEMENTATION_PLAN.md` for historical rationale behind the
  project lowering strategy.
- `docs/DOCS_TODO.md` for architecture topics that still need documentation.

### 8. CPU Code Generation

The compiled CPU path is the complete compiled backend for the current dense
static core. It lowers array operations into MLIR, then into LLVM/object code,
links a shared library or executable, and calls it through Python runtime code.

#### Overview

CPU code generation is where the compiler stops describing array computation
and starts producing callable machine code. In RemoraC, the source program has
already passed through parsing, typechecking, elaboration, HIR construction,
optimization, and MLIR lowering before the final native artifact appears. The
CPU backend is the complete compiled backend for the current dense static core,
so it carries a high correctness burden. If the typechecker accepts a dense
program and the feature is in the CPU support matrix, the compiled path should
produce the same result as the interpreter.

Ahead-of-time compilation and JIT-like APIs differ mostly in when compilation
happens and how artifacts are managed. A traditional ahead-of-time compiler
takes source and produces an executable or object file before execution. A JIT
compiler compiles during program execution. RemoraC has command-line flows that
produce executables or shared libraries, and API flows that feel JIT-like from
Python because a function can be compiled and then called. Underneath, the
system still builds native artifacts through an MLIR/LLVM/toolchain pipeline
and loads or runs them through runtime code.

The CPU path must cross a language boundary. Remora values live conceptually in
the language semantics. Python values live in NumPy arrays, Python scalars, and
`ctypes` objects. Generated native code follows a C-compatible ABI. MLIR and
LLVM have their own internal function signatures. The compiler and runtime must
make all of these agree. For arrays, the central representation is a descriptor
containing an allocated pointer, an aligned pointer, an offset, sizes, and
strides. This lets native code understand both contiguous arrays and views
without needing Python or NumPy objects directly.

Calling conventions are the rules that determine how arguments and results are
passed: in registers, on the stack, by pointer, by value, or through hidden
result parameters. At the C ABI boundary, small scalar details matter: integer
widths, pointer sizes, alignment, struct layout, bool representation, and
ownership conventions. MLIR's `llvm.emit_c_interface` and memref descriptor
lowering are relevant because they generate wrapper-compatible interfaces
around lower-level functions. The Python runtime then uses `ctypes` definitions
that must match the compiled descriptors exactly.

Most array operations become loop nests or calls to helper routines. An
elementwise map over a shape `[m, n]` array naturally becomes a two-dimensional
iteration space, though the emitted loops may be flattened or transformed by
later passes. A fold has an accumulation variable and a dependency along the
reduced dimension. A scan produces every prefix result and therefore has a
different dependency structure. A view such as transpose, reverse, take, or
drop may not need to copy data; it may only change offset, sizes, and strides.
The backend's job is to choose a correct representation and then emit MLIR that
the pipeline can lower to efficient native code.

Runtime support libraries are used when an operation is better implemented as a
helper than as generated inline MLIR. Sorting, matrix multiplication, certain
descriptor utilities, or other operations may call into C runtime code. This is
a normal compiler engineering tradeoff. Inlining everything into generated IR
can make the compiler complex and produce bloated output. Calling helpers can
centralize tricky code, use existing optimized routines, and simplify testing.
The cost is that the ABI between generated code and runtime helpers becomes
another contract that must be maintained.

The CPU backend also has to decide when to materialize arrays. A pure
source-level expression may conceptually produce intermediate arrays, but an
optimizer may fuse operations, eliminate temporaries, or represent a result as
a view. Conversely, some operations require materialization because later code
needs contiguous data, because a helper expects a certain layout, or because a
view cannot express the desired transformation alone. This is one of the
reasons descriptor semantics and lowering decisions are tightly coupled.

`ctypes` interop is deliberately low-level. Python loads a shared object,
constructs C-compatible argument descriptors, calls a native function, and
interprets the result. There is little dynamic safety at this boundary. A wrong
field type, wrong rank-specialized descriptor, missing lifetime guarantee, or
incorrect output allocation can produce crashes or silent data corruption.
Therefore the ABI document and `remora/abi.py` are as much part of the compiler
contract as the typechecker or HIR definitions.

Correctness and performance pull on different parts of CPU code generation.
Correctness requires semantic agreement with the interpreter across shapes,
dtypes, higher-order functions, recursion, views, and AD-generated programs.
Performance requires good loop structure, predictable memory access, vectorizer
friendly IR, cache locality, avoidance of unnecessary temporaries, and sensible
use of runtime helpers. The first priority is always correctness. A fast
miscompile is worse than a slow correct program, especially in a language where
array operations may produce large numeric results that are not manually
inspected.

When debugging the CPU path, follow the artifact chain. Inspect the HIR to see
whether the language-level operation is represented correctly. Inspect emitted
MLIR to see whether shapes, loops, scalar operations, and descriptors are
correct. Check the MLIR pipeline if verification or lowering fails. Check the C
runtime and `ctypes` ABI if compiled code runs but produces corrupted results
or crashes at the boundary. Finally, compare against the interpreter for
semantic correctness. The CPU backend is complete for the dense core, but that
completeness is an implementation invariant that tests must keep defending.

Study these concepts:

- Ahead-of-time compilation vs JIT-like compile-on-call APIs.
- Calling conventions and C ABI boundaries.
- Memref descriptors and `llvm.emit_c_interface`.
- Loop nests for maps, folds, scans, and views.
- Runtime support libraries for operations such as sort and matmul.
- `ctypes` interop with compiled shared libraries.
- Linking C runtime helpers into generated code.

Why it matters here: CPU correctness is expected for the dense core accepted by
the typechecker. CPU performance work also requires understanding when MLIR
generates scalar loops, when it vectorizes, and when custom runtime helpers or
BLAS-like calls are preferable.

Recommended reading:

- LLVM Kaleidoscope tutorial for lowering and native-code generation basics.
- MLIR Toy tutorial for the staged lowering model used by MLIR-based
  compilers.
- `docs/ABI.md` for the descriptor contract that CPU code must honor.
- `docs/IMPLEMENTATION_NOTES.md` for project-specific backend notes.
- `remora/remora_rt.c` for runtime helper functions linked into compiled CPU
  artifacts.

### 9. CPU Vectorization And Multicore Execution

CPU vector instructions are SIMD: one instruction operates on multiple elements.
Remora's dense, static-shape programs are natural candidates for vectorization,
but performance depends on memory layout, loop structure, alignment, and compiler
passes.

#### Overview

Modern CPUs are not just scalar machines with faster clocks. They contain
vector units, multiple cores, deep cache hierarchies, branch predictors,
hardware prefetchers, and sophisticated out-of-order execution engines. A
compiler for dense array programs has to generate code that lets this hardware
work. The mathematical fact that a Remora `map` applies independently to many
cells is only the starting point. High performance depends on whether the
lowered loops have predictable bounds, contiguous memory access, simple
dependencies, aligned data, and enough work per thread to amortize overhead.

SIMD means single instruction, multiple data. An AVX instruction on x86, for
example, may add several single-precision floats at once. SSE, AVX, AVX-512,
and NEON differ in register width, instruction set, and target architecture,
but the core idea is the same: pack multiple lanes into one vector register and
apply one operation to all lanes. A scalar loop that adds one element per
iteration may become a vector loop that adds 4, 8, 16, or more elements per
iteration depending on dtype and hardware. The compiler may also generate a
cleanup loop for the remaining elements when the iteration count is not a
multiple of the vector width.

Loop vectorization is the compiler analysis and transformation that turns
scalar loops into SIMD loops. It needs to prove that iterations are independent
or that dependencies are compatible with vector execution. Elementwise maps are
usually good candidates because each output element depends on corresponding
input elements and no other output element. Reductions are harder because each
iteration updates an accumulator. Scans are harder still because every output
depends on previous elements. These operations can still be optimized, but they
require specialized transformations such as tree reductions, vector partial
accumulators, prefix algorithms, or target-specific idioms.

Memory layout often dominates arithmetic. A CPU can execute many floating-point
operations per cycle when data is already in registers, but loading data from
main memory is much slower. Caches bridge that gap, but only when access
patterns have locality. Row-major traversal of contiguous arrays is friendly to
hardware prefetching and cache lines. Traversing a column of a row-major matrix
is strided and may waste most of each loaded cache line. Views complicate the
picture because a transpose or slice can preserve array semantics while
changing strides. A compiler that knows descriptor strides can generate correct
code, but the resulting access pattern may or may not be fast.

Alignment also matters. Some vector instructions are fastest when data starts
at an address aligned to the vector width or cache-line boundary. Modern
hardware handles unaligned loads better than older machines did, but alignment
still affects code generation, peel loops, and performance. The descriptor ABI
separates allocated and aligned pointers partly because low-level code needs to
know which address should be used for element access. When integrating with
NumPy arrays, the compiler must respect the actual data pointer, dtype, shape,
and strides rather than assume a fresh ideal allocation.

Multicore execution is different from SIMD. SIMD uses one core's vector lanes;
multicore parallelism uses multiple CPU cores. Parallel loops split an
iteration space into chunks assigned to worker threads. This is attractive for
large maps, independent row operations, and some reductions. The overheads are
thread creation or scheduling, synchronization, cache coherence traffic, and
load imbalance. For small arrays, a parallel loop can be slower than a serial
loop. For large arrays, the limiting factor may become memory bandwidth rather
than core count.

False sharing is a classic multicore performance trap. It occurs when different
threads write to different variables that happen to occupy the same cache line.
The hardware coherence protocol then bounces that cache line between cores even
though the program has no logical sharing. Array reductions can trigger related
problems if many threads update adjacent partial accumulators or a shared
output. Good parallel lowering often gives each thread private temporaries and
combines them later, with attention to padding and memory layout.

The compiler's IR affects whether downstream vectorizers can help. MLIR and
LLVM can recognize many loop patterns, but only if the emitted IR preserves
the right structure. Overly complex index arithmetic, opaque function calls
inside loop bodies, unpredictable aliasing, or unnecessary materialization can
block vectorization. Conversely, structured operations such as `linalg.generic`
can expose parallel and reduction dimensions explicitly. RemoraC's CPU
vectorization work therefore depends both on high-level array semantics and on
the exact MLIR emitted by `remora/lowering/`.

Static shapes are an advantage. Known ranks and dimensions allow fixed loop
nests, simpler bounds, rank-specialized descriptors, and better opportunities
for unrolling or specialization. Static shape knowledge does not guarantee
fast code, but it reduces uncertainty. The backend can know that an operation
iterates over a 1024-element vector, a 64-by-64 matrix, or a rank-3 tensor with
particular strides. This is much more information than a generic dynamic array
library call may expose to a compiler.

Benchmark interpretation requires separating several ceilings. Some kernels
are compute-bound: arithmetic throughput is the limiting factor. Others are
memory-bound: moving bytes dominates. Many simple maps are memory-bound because
each element requires only a few operations but several loads and stores.
Matrix multiplication can be compute-bound when tiled well because each loaded
value participates in many multiply-adds. Scans and reductions have dependency
structures that affect both vectorization and parallelization. Understanding
these distinctions explains why one Remora primitive may be competitive while
another lags a tuned NumPy, BLAS, or compiler-generated implementation.

Study these concepts:

- SIMD vs scalar execution.
- AVX/SSE/NEON-style vector lanes and vector width.
- Loop vectorization, unrolling, and alignment.
- Memory bandwidth vs arithmetic throughput.
- Cache locality, row-major traversal, strides, and views.
- Parallel loops, OpenMP, work splitting, reductions, and false sharing.
- Reductions and scans as special cases: dependencies limit naive
  vectorization.

Why it matters here: RemoraC has `--cpu-vectorize` and `--cpu-threads`, but the
benchmark docs show map/fold CPU throughput still needs tuning. Understanding
vectorization explains why some operations, such as stencil and scan, can be
competitive while others lag tuned NumPy/BLAS.

Recommended reading:

- Intel or Agner Fog optimization manuals for CPU microarchitecture, SIMD, and
  memory-hierarchy details.
- LLVM Loop Vectorizer documentation for what downstream vectorization passes
  can and cannot infer.
- MLIR Vector dialect documentation for explicit vector IR concepts.
- `docs/BENCHMARK_PLAN.md` for intended CPU performance experiments.
- `docs/BENCHMARK_RESULTS.md` for current measured strengths and gaps.

### 10. GPU Programming And CUDA Execution

The GPU backend targets NVIDIA CUDA-style execution through generated PTX and a
descriptor ABI. It is intentionally narrower than the CPU backend but must be
numerically correct for every program it accepts.

#### Overview

A GPU is a throughput machine. It is designed to run many lightweight threads
and hide memory latency by switching among them. This is a different model from
a CPU, which spends substantial hardware on making a small number of threads
run with low latency. For dense array programs, the GPU model can be a good
fit: maps, reductions, scans, sorting passes, and matrix operations often
contain thousands or millions of similar element computations. The challenge is
to express those computations in a way that respects GPU execution and memory
constraints.

In CUDA terminology, a kernel is launched over a grid of thread blocks. Each
block contains threads. Threads are executed by the hardware in warps, commonly
groups of 32 threads on NVIDIA GPUs. A kernel's logical thread id is derived
from block and thread indices, and that id is typically mapped to an array
element, a tile, a reduction lane, or some other piece of work. Occupancy is a
measure of how many warps can be resident on a streaming multiprocessor at
once. Higher occupancy can help hide latency, but it is not the only goal:
register pressure, shared memory usage, memory access patterns, and arithmetic
intensity all matter.

SIMT means single instruction, multiple threads. It resembles SIMD at a high
level, but the programming model exposes individual threads rather than vector
lanes. Threads in a warp execute the same instruction stream when possible. If
they take different branches, the warp experiences divergence: one path runs
for the threads that took it, then the other path runs for the rest, with
inactive threads masked off. Branches are not forbidden, but branch-heavy code
with irregular control flow can lose much of the GPU's throughput advantage.
Array languages help when their operations produce regular control flow, but
conditionals inside map bodies still require attention.

GPU memory spaces have different costs and uses. Global memory is large and
accessible by all threads, but high latency. Shared memory is block-local and
fast when used well, but limited. Registers are fastest but private to a
thread, and using too many reduces occupancy. Local memory is usually a spill
area backed by global memory and should not be confused with cheap stack
storage. Constant and texture memory have specialized uses. A good GPU lowering
strategy decides which values should live in registers, which tiles should be
staged in shared memory, and how global memory should be accessed.

Coalescing is central. When neighboring threads in a warp access neighboring
memory addresses, the hardware can combine those accesses into efficient memory
transactions. If each thread accesses a widely separated strided address, the
warp may issue many transactions and waste bandwidth. Row-major contiguous
arrays are usually friendly for elementwise maps when consecutive thread ids
map to consecutive elements. Views such as transpose, reverse, take, and
subarray can preserve semantics while producing less friendly access patterns.
The descriptor ABI makes those views representable; the lowering must still
generate correct and preferably coalesced address calculations.

Kernel launch overhead and host/device transfers are often more important than
raw arithmetic speed for small workloads. Launching a kernel has a fixed cost.
Copying data between host and device can dominate computation unless arrays are
large or remain device-resident across many operations. This is why GPU
execution systems use memory pools, device arrays, and execution plans that
chain kernels without unnecessary transfers. A compiler that sends every tiny
operation as a separate host/device round trip will lose even if each kernel is
individually well written.

Parallel reductions and scans illustrate the algorithmic difference between
CPU and GPU lowering. A reduction cannot simply have every thread update one
global accumulator without synchronization; that would create races and severe
contention. Instead, reductions are staged: threads compute partial results,
combine within a block, write block results, and possibly launch another stage.
Scans are more complex because every prefix result is needed. Efficient scan
algorithms use tree patterns, shared memory, and sometimes multiple kernels for
large arrays. Sorting and scatter-like operations introduce further issues:
atomics, race freedom, temporary storage, and stable ordering.

PTX is NVIDIA's virtual instruction set. NVVM is NVIDIA's LLVM-based compiler
path, and LLVM's NVPTX backend can emit PTX from suitable LLVM IR. RemoraC's
GPU backend generates LLVM dialect or related textual representations that are
lowered toward PTX. PTX is not the same as final machine code; the NVIDIA
driver or toolchain lowers PTX further for a particular GPU architecture. This
layering gives portability across GPU generations, but it also means compiler
authors need to understand which abstractions survive to PTX and which are
resolved later.

RemoraC's GPU backend is intentionally narrower than the CPU backend. That is
a correctness choice. Arbitrary closures, dynamic boxes, or unsupported
higher-order behavior should not be approximated by kernels that happen to
compile. The project rule is that unsupported GPU features must fail loudly.
This is particularly important because GPU bugs are often silent: a kernel may
launch successfully, write output of the right shape and dtype, and still
compute wrong values because of a stride error, dtype mismatch, race, or
incorrect map-body lowering.

For RemoraC, GPU lowering is not one mechanism. There are fast paths for
certain maps, a general recursive expression compiler for map bodies, support
for folds and selected scans, descriptor-level view handling, execution plans
for multi-kernel operations, and routing logic in `codegen.py`. When working on
the GPU path, always ask which route a program takes. A top-level operation and
the same operation inside a `map` body may exercise different code paths. The
tests therefore need numeric parity across context, dtype, and shape, not just
compile-only checks.

Study these concepts:

- GPU architecture: threads, warps, blocks, grids, occupancy.
- SIMT execution and branch divergence.
- Global, shared, local, and register memory.
- Coalesced memory access and bandwidth limits.
- Kernel launch overhead and host/device transfer cost.
- Parallel reductions, scans, sort, scatter, and atomics.
- Device-resident execution and memory pools.
- PTX, NVVM, and LLVM's NVPTX backend.

Why it matters here: `remora/gpu_lowering.py`, `remora/codegen.py`, and
`remora/_gpu_expr_lowering.py` generate GPU kernels or execution plans for maps,
folds, scans, views, sort/grade, matmul, filter/replicate, state-fold loops, and
selected recursive helpers. Unsupported GPU features must fail loudly, not
silently compute wrong results.

Recommended reading:

- NVIDIA CUDA C Programming Guide for the thread/block/grid execution model and
  memory hierarchy.
- NVIDIA PTX ISA documentation for the virtual instruction set emitted by the
  GPU backend.
- Mark Harris, CUDA reduction and scan articles for concrete parallel primitive
  implementations.
- Merrill and Garland, GPU scan/sort papers or CUB documentation for production
  scan/sort design patterns.
- `docs/BACKEND_GAPS.md` for current GPU support limits.
- `docs/IMPLEMENTATION_LOG.md` for the evolution of GPU lowering support.

### 11. Descriptor ABI, Runtime Values, And Views

The Dense Core ABI is a key design anchor. Array arguments and outputs cross
CPU/GPU boundaries through rank-specialized descriptors containing base pointers,
offsets, sizes, and strides.

#### Overview

An ABI is a contract between separately compiled or separately implemented
pieces of a system. It specifies how values are represented, how functions are
called, how memory is laid out, and which side owns which responsibilities. In
RemoraC, the Dense Core descriptor ABI is the contract that lets Python,
generated CPU code, generated GPU kernels, runtime helpers, and NumPy-like
values agree on what an array is. Without this contract, every backend would
need a different private representation and cross-boundary calls would be
fragile.

The descriptor represents an array by metadata plus pointers. The allocated
pointer identifies the base allocation. The aligned pointer identifies the
address used for element access. The offset shifts the logical first element
relative to that aligned pointer. The sizes array gives the extent of each
dimension. The strides array gives the step, in elements, needed to move by one
index along each dimension. A rank-specialized descriptor has fixed-size sizes
and strides fields for a particular rank. A rank-2 descriptor, for example,
contains two sizes and two strides; a rank-0 descriptor has no size or stride
entries but still participates in the same ABI family.

This representation is what makes views cheap. A contiguous row-major 3-by-4
matrix of floats might have sizes `[3, 4]` and strides `[4, 1]`. Transposing it
can be represented by sizes `[4, 3]` and strides `[1, 4]` with the same data
pointer. Taking a slice can adjust the offset and reduce a size. Reversing an
axis can use a negative stride and an offset that points at the logical first
element of the reversed view. Ravel and reshape may be descriptor
reinterpretations when the layout conditions are satisfied. The operation is
then metadata manipulation, not element copying.

Strides are also where correctness bugs hide. Address calculation for an index
tuple is a dot product of indices and strides plus offset. If the compiler
assumes contiguity where a view is strided, it will read the wrong elements. If
it mishandles negative strides, it may reverse incorrectly or access outside
the allocation. If it confuses byte strides with element strides, dtype size
will corrupt indexing. NumPy exposes byte strides; Remora's ABI documents its
own convention. Any conversion between representations must be explicit.

Rank-specialized descriptors are a pragmatic choice for the dense static core.
The compiler knows ranks by lowering time, so generated functions can have
signatures specialized to rank 0, rank 1, rank 2, and so on. This avoids
dynamic loops over shape metadata in many places and makes ABI declarations
more concrete. A dynamic-rank descriptor would be more flexible but would
require carrying a rank field, dynamically sized shape/stride arrays, and more
runtime checks. Full dynamic-shape or boxed values may eventually need a richer
representation, but the static dense core benefits from specialization.

Scalar rank-0 descriptors deserve attention because Remora treats scalars as
rank-0 arrays. At the language level this is elegant: scalar operations are
array operations on empty shapes. At the ABI level it means the system must
decide how a scalar crosses a function boundary. It may be represented as a
plain scalar in some internal contexts, but the public ABI needs consistent
rules for scalar arguments and results. Confusion between scalar-by-value and
rank-0 descriptor representations can produce subtle boundary bugs.

Boolean layout is another small but important contract. A public bool array
needs a stable representation for Python interop and descriptor storage.
Internal compiler predicates may use MLIR `i1`, LLVM `i1`, or target-specific
condition values. Those are not necessarily the same as a public one-byte or
integer bool layout. Lowering must convert between predicate representation and
stored bool representation deliberately, especially when arrays of bools cross
the ABI.

C struct layout depends on field order, alignment, padding, pointer width, and
integer width. A descriptor definition in Python `ctypes` must match the struct
layout expected by generated code and runtime helpers. On a 64-bit platform,
pointers and index-sized integers are typically 8 bytes, but the ABI should not
be left to assumption. If a field is signed in generated code and unsigned in
Python, negative strides or offsets can break. If padding differs, every field
after the mismatch may be read incorrectly. This is why `docs/ABI.md` is a
normative document rather than a casual implementation note.

The relationship to NumPy is practical. NumPy arrays already have dtype, shape,
strides, data pointer, base allocation, and view semantics. RemoraC can use
NumPy arrays as host-side data, but it must translate NumPy metadata into the
Remora descriptor contract. Contiguous arrays are the easy case. Slices,
transposes, non-default strides, dtype conversions, and ownership/lifetime
issues are where integration becomes delicate. The generated native code must
not outlive the data it points to, and device transfers must preserve the
logical view semantics or materialize a correct contiguous copy when required.

For compiler work, the descriptor ABI is the meeting point of semantics and
machine representation. A type says an expression has shape `[m, n]`; a
descriptor carries sizes `m` and `n`. A view expression says no element values
change; a descriptor update changes offset and strides. A GPU kernel expects
rank-specialized arguments; the execution layer constructs matching device
descriptors. When a program produces wrong values only for transposed, sliced,
or reversed inputs, suspect descriptor indexing before suspecting arithmetic.

Concrete descriptor example: a contiguous 2-by-3 row-major matrix has sizes
`[2, 3]` and strides `[3, 1]`. Its transpose can use sizes `[3, 2]` and strides
`[1, 3]` with the same data pointer. No element values changed; only the
mapping from logical indices to memory changed.

Study these concepts:

- C struct layout, alignment, pointer-sized fields, and signed integer sizes.
- Strided array views and offset indexing.
- Rank-specialized vs dynamic-rank descriptors.
- Row-major contiguous layout.
- Scalar rank-0 descriptors.
- Public bool layout vs internal predicate representation.
- NumPy array metadata: shape, strides, dtype, base allocation, views.

Why it matters here: view operations such as transpose, slice, reshape, ravel,
reverse, take, and drop depend on descriptor semantics. Dynamic shapes and
runtime boxes, if implemented later, will extend this representation.

Recommended reading:

- `docs/ABI.md` for the normative Dense Core descriptor layout.
- NumPy ndarray internals documentation for shape/stride/view concepts at the
  Python boundary.
- MLIR memref descriptor documentation for the related compiler-side memory
  representation.

### 12. GPU Execution Plans And Kernel Routing

Not every GPU operation is one kernel. Sort, filter, replicate, scan, and some
optimization loops use multi-kernel plans with temporary device buffers and
host-orchestrated steps.

#### Overview

The simplest mental model of GPU compilation is one source operation becomes
one kernel. That model works for some elementwise maps, but it is not general.
Many array primitives require multiple kernels, temporary buffers, prefix sums,
host-side launch sequencing, or runtime decisions. A GPU execution plan is a
structured representation of that work: which kernels run, what buffers they
read and write, what temporaries they allocate, and how outputs flow from one
step to the next.

Kernel fusion and multi-kernel execution are the two competing pressures.
Fusion combines producer and consumer operations into one kernel so that
intermediate arrays are not written to and read from global memory. This can
reduce memory bandwidth and launch overhead. However, not every operation can
be fused conveniently. Reductions, scans, sorts, filters, and operations with
global coordination often need synchronization beyond a single thread block or
need intermediate summaries. Since CUDA has no global synchronization inside an
ordinary kernel across all blocks, multi-stage algorithms often become
multi-kernel pipelines.

Temporary buffers are part of the algorithm, not an implementation accident.
Radix sort may need histograms, prefix sums, ping-pong buffers, or per-pass
output arrays. Filter and replicate often use a predicate or count array, scan
it to compute output positions, then scatter selected or repeated elements.
Large reductions may write block partials and reduce those partials later.
Stateful optimization loops may keep parameter arrays, gradient arrays, and
optimizer state resident on the device across iterations. The execution plan
must make these buffers explicit enough for allocation, reuse, and lifetime
management.

Producer/consumer relationships define liveness. If kernel B reads a buffer
written by kernel A, A must run first and the buffer must remain alive until B
finishes. If no later step reads a temporary, it can be released or returned to
a memory pool. Buffer reuse is important because GPU allocation can be
expensive and because large temporaries can exhaust device memory. A plan that
knows lifetimes can reuse storage for non-overlapping temporaries, while an
unstructured sequence of launches may allocate more than necessary.

Execution DAGs are a natural way to describe these relationships. Nodes are
kernels or host-side steps; edges are data dependencies. In a simple pipeline,
the DAG is a chain. In a more complex plan, independent producers could run in
parallel streams, or several intermediates could feed a later kernel. RemoraC's
current execution plans are primarily about correct launch sequencing and
buffer management, but the same representation points toward future scheduling
and cost-model work.

Prefix sums are a recurring building block. A filter computes a boolean flag
for each element, scans the flags to compute compacted output indices, and then
scatters elements whose flag is true. A replicate computes a count for each
input element, scans counts to compute output ranges, and then writes repeated
values. Stream compaction, partitioning, sparse-like expansion, and some
segmented algorithms all use this pattern. This is why scan performance and
correctness matter beyond the user-visible `scan` primitive.

Kernel routing is the decision process that maps an HIR program to a backend
implementation. A simple f32 elementwise map may use a specialized fast path. A
map body with nested conditionals, views, casts, or helper calls may need the
general GPU expression lowering path. A sort may route to a radix-sort
execution plan. An unsupported closure capture should be rejected. The routing
logic is therefore a capability matrix encoded in code: which HIR nodes, dtypes,
shapes, and contexts are supported by which lowering route.

Explainability matters because silent fallback can be dangerous. If a program
is too complex for a GPU route, the compiler should either choose a known
correct alternative or report that GPU code generation is unavailable. It
should not emit a partial approximation. Capability checks need to distinguish
top-level support from support inside compound contexts. A view operation that
works as a top-level descriptor transformation may require different logic
inside a map body. A fold over f32 may be supported while the same pattern over
i32 or bool needs separate lowering. These distinctions should be visible in
tests and errors.

Cost models are the next layer beyond capability. A route may be correct but
slow. A small array may run faster on CPU because GPU launch overhead dominates.
A large contiguous map may be ideal for GPU. A strided view may be bandwidth
inefficient. A multi-kernel plan may amortize overhead only when enough data is
processed or when data stays resident on the device across several operations.
Choosing CPU vs GPU, choosing fusion vs materialization, and choosing a
specialized route vs a general route are all cost-model questions once
correctness is guaranteed.

RemoraC's current architecture already contains the ingredients for this
discussion: HIR nodes that describe operations, `codegen.py` routing, GPU
lowering modules, execution plans, device buffers, and tests that distinguish
accepted programs from rejected ones. The important engineering discipline is
to keep routing explicit. When adding a GPU feature, document which route
handles it, what dtypes and shapes are covered, what contexts are covered, and
which tests prove numeric parity. A backend capability matrix is not paperwork;
it is how future maintainers avoid assuming support that is not actually
present.

Study these concepts:

- Kernel fusion vs multi-kernel pipelines.
- Producer/consumer buffers, liveness, and buffer reuse.
- Execution DAGs and launch sequencing.
- Prefix-sum based compaction and replication.
- Route selection and backend capability matrices.
- Cost models for choosing CPU vs GPU.

Why it matters here: project docs repeatedly call out a need for explainable
backend routing, capability matrices, and cost-aware scheduling. This is where
compiler architecture meets performance engineering.

Recommended reading:

- `docs/WORKSTREAM_0_PLAN.md` for planned routing and capability-matrix work.
- `docs/PLAN_TO_IMPLEMENT_FULL_REMORA.md` for the broader feature roadmap that
  will stress routing.
- `docs/remorac-vs-futhark.md` for comparison with a mature array compiler.
- Futhark PLDI 2017 paper for whole-program GPU compilation and fusion context.
- Futhark incremental flattening and memory-optimization papers in
  `docs/remora-reference/` for advanced plan/scheduling ideas.

### 13. Automatic Differentiation

RemoraC includes reverse-mode AD support for scalar-cost functions, with source
generation and compiled CPU support. GPU AD is narrower but exists for selected
dense numeric loops.

#### Overview

If you already know backpropagation, VJPs, and optimizer loops, do not spend
time re-learning AD from first principles here. The RemoraC-specific questions
are different: how reverse-mode AD interacts with rank-polymorphic array
semantics, how cotangent shapes are represented for cells and frames, and how
generated gradient programs travel through the same compiler pipeline as
hand-written Remora.

RemoraC focuses on reverse-mode AD for scalar-cost functions. The objective is
the familiar one from optimization and machine learning: compute one scalar
loss and accumulate sensitivities with respect to many dense numeric inputs.
The novelty is that the program being differentiated is not a scalar expression
tree. It may contain elementwise maps, folds, scans, views, reshapes,
transposes, and rank-polymorphic applications whose iteration space is
determined by frame/cell decomposition.

For arrays, a VJP is also a shape transformation. An elementwise map
differentiates elementwise, but the cotangent must preserve the result's frame
and cell structure. A reduction maps many input contributions to fewer output
cotangents, so the backward pass often replicates, broadcasts, or otherwise
distributes the reduced cotangent across the original cells. A scan has prefix
dependencies and may require a reverse scan or a structured adjoint. A view
operation such as reshape or transpose usually has an inverse-view cotangent
rule, while a slice-like operation may need to scatter cotangents into a larger
zero-initialized array. These are array algorithms as much as calculus rules.

RemoraC's implementation strategy is source generation around an AD
expression/tape representation. Instead of introducing a separate derivative IR
that bypasses the normal compiler, it constructs gradient source that can pass
through parsing, typechecking, elaboration, HIR optimization, MLIR lowering,
and backend execution. This is high leverage because AD bugs then exercise the
same shape checker and lowering paths as ordinary user programs. It also means
AD can fail for ordinary compiler reasons: generated source may be too large,
an optimizer may miss a simplification, or a backend may not support the
generated combination of array operations.

GPU AD should be read as an explicit envelope, not as full-language automatic
GPU differentiation. Selected dense numeric loops and optimizer-shaped
programs exist, but arbitrary higher-order AD, dynamic shapes, boxes, and
general recursive differentiated programs are outside the current GPU support
model. When extending this area, make the accepted subset and rejection paths
clear; a silently wrong GPU gradient is worse than a rejected program.

The distinction between primal values and cotangents is semantic and
representational. A primal is an ordinary value computed by the original
program. A cotangent is an accumulated sensitivity flowing backward. For
scalars, cotangents look like numbers. For arrays, they have shapes related to
the primal arrays. For structured values such as pairs or boxes, a full AD
system needs corresponding cotangent structures or restrictions. RemoraC's AD
support is aimed at dense numeric scalar-cost functions, so the implemented
path can avoid some of the full generality that a language-wide AD semantics
would require.

Other AD systems may use operator overloading or IR transformation. Keep those
models in mind for comparison, but read this codebase as a source-generation
system whose generated program must be valid RemoraC input.

AD-generated programs can be much larger and less friendly than hand-written
programs. They may contain many temporaries, repeated subexpressions, generated
helper functions, and state-like loops for optimizers. Simplification and
common subexpression elimination are therefore important. Algebraic identities
such as multiplying by zero, adding zero, or reshaping through inverse views can
remove large amounts of work. CSE can share repeated primal or cotangent
computations. Without these simplifications, a theoretically correct gradient
may be too large or too slow to compile and run.

Gradient descent and related optimizers introduce another compiler stressor:
iteration with state. An optimizer step updates parameters using gradients,
learning rates, and sometimes auxiliary state such as momentum or adaptive
moments. In a pure functional IR, this state is represented as values threaded
through loops or folds rather than mutable variables. The backend still has to
compile it efficiently, ideally keeping arrays resident on the device or in
reusable buffers when possible. The `examples/ad_optimize.lisp` path is
important because it tests AD, generated source, compilation, and runtime loops
together.

Correctness testing for AD should not rely only on typechecking. A gradient can
have the right shape and still be numerically wrong. Finite-difference checks
are useful as an independent sanity test: perturb an input slightly and compare
the observed output change with the computed directional derivative. These
checks are approximate and sensitive to step size and floating-point error, but
they catch many implementation mistakes. Interpreter-vs-compiled parity also
matters because the gradient source must mean the same thing on each backend
that accepts it.

For RemoraC, AD sits at the intersection of language semantics, compiler
transformations, and backend support. The source language must express the
generated gradient. The typechecker must verify its shapes and scalar types.
The HIR optimizer must keep it tractable. The CPU backend must compile the full
dense generated program. GPU support is narrower and should remain explicit
about what AD-shaped loops and lifted gradients it supports. When an AD example
fails, the cause may be calculus-rule generation, source emission, typechecking,
optimization blowup, unsupported backend routing, or a low-level numeric bug.

Minimal AD example: for a scalar expression `x * x`, the primal computation
multiplies `x` by itself, and the reverse pass accumulates `2 * x` into the
cotangent for `x`. For an elementwise square over an array, the same local rule
applies to each element, but the compiler must also preserve the array shape
and any surrounding frame/cell structure.

Study these concepts:

- Forward-mode vs reverse-mode automatic differentiation.
- Computational graphs, tapes, primal values, cotangents, and VJPs.
- Differentiating array operations: map, fold, scan, views, and reductions.
- AD simplification and common subexpression elimination.
- Gradient descent and optimizer state loops.

Why it matters here: AD-generated programs stress the compiler because they can
produce large expressions, many intermediate arrays, stateful-looking loops, and
backend-specific lowering paths.

Recommended reading:

- Baydin et al., *Automatic Differentiation in Machine Learning*, for a broad
  survey of AD modes and terminology.
- Futhark AD SC22 paper in `docs/remora-reference/` for AD in a parallel array
  language.
- `docs/USER_GUIDE.md` for user-facing AD examples.
- `docs/IMPLEMENTATION_LOG.md` for the implementation history of AD support.

### 14. Parallel Algorithms For Array Primitives

To work on backends, study the algorithms behind common array primitives, not
just their APIs.

#### Overview

Array languages present operations such as `map`, `fold`, `scan`, `sort`,
`transpose`, and `matmul` as high-level primitives. A backend implementer must
know the algorithms behind those primitives. The same source operation may
lower to a scalar loop on CPU, vectorized loops, a library call, one GPU
kernel, or a multi-kernel execution plan. Correctness is defined by the
language operation, but performance is determined by the algorithm and memory
behavior chosen by the backend.

Elementwise maps are the easiest parallel primitive. Each output cell is
computed independently from corresponding input cells. This independence
supports SIMD vectorization on CPU, thread-per-element kernels on GPU, and
fusion with producers or consumers. Map fusion is the transformation that
combines consecutive elementwise operations so intermediate arrays are not
materialized. For example, computing `sqrt(x * x + y * y)` should ideally read
`x` and `y`, do the scalar arithmetic, and write one output, rather than write
an intermediate for `x * x`, another for `y * y`, another for the sum, and then
the square root.

Reductions combine many values into fewer values using a binary operation and
an identity. A serial reduction is straightforward, but parallel reduction
requires restructuring. A tree reduction combines pairs, then pairs of pairs,
and so on. This exposes parallelism but may change floating-point rounding
because floating-point addition is not truly associative. The language and
tests must tolerate appropriate numerical differences when parallel reduction
order differs from interpreter order. On GPU, reductions also require block
partial results, shared memory or warp-level operations, and often multiple
kernel stages for large inputs.

Scans, or prefix sums, compute all prefix reductions. Inclusive scan includes
the current element in each prefix; exclusive scan reports the prefix before
the current element. Right scans traverse from the other direction. Segmented
scans reset at segment boundaries and are crucial for irregular nested
parallelism. Scan is more complex than reduction because every intermediate
prefix is an output. Efficient parallel scan algorithms, such as Blelloch's
upsweep/downsweep pattern, are foundational for compaction, replication,
partitioning, radix sort, sparse algorithms, and many data-parallel workflows.

Sorting and grading are more than convenience functions. `sort` returns values
in order; `grade` or argsort returns the permutation of indices that would sort
the values. Comparison-based sorts such as bitonic sort have regular parallel
structure but may do more comparisons than asymptotically optimal sequential
sorts. Radix sort exploits fixed-width integer or bitwise representations and
often wins on GPUs for numeric keys. Floating-point radix sort requires careful
key transformation to respect numeric ordering, especially around signs, zeros,
NaNs, or supported-subset decisions. A compiler backend may support only
specific dtypes because each dtype needs a correct algorithm.

Scatter and gather are dual memory access patterns. Gather reads from
positions specified by an index array. Scatter writes to positions specified by
an index array. Scatter-add or indexed accumulation combines values that may
target the same output position. Race freedom is the central issue: if two GPU
threads write the same location without coordination, the result is undefined
or nondeterministic. Atomics can make such updates safe but may be slow and may
have dtype limitations. Some algorithms avoid atomics by sorting or grouping
updates first, trading extra work for deterministic structure.

Stencils and convolutions compute each output from a local neighborhood of
input values. A one-dimensional heat-flow update is a typical stencil: each
point depends on its neighbors. In images and neural networks, convolution
slides a kernel over spatial dimensions. `im2col` is a transformation that
turns sliding windows into columns so convolution can be expressed as matrix
multiplication. This can reuse optimized GEMM implementations, but it may
increase memory use substantially. Tiling is often required to get good cache
or shared-memory behavior.

Matrix multiplication is a performance world of its own. The naive triple loop
has a simple definition, but high performance depends on tiling, register
blocking, cache reuse, vectorization, and on GPU, shared memory and warp-level
matrix instructions. Arithmetic intensity is high when each loaded element is
used many times, which is why matmul can reach much higher fractions of peak
compute throughput than simple maps. For a research compiler, calling BLAS or a
specialized GPU library may be the right engineering choice for some cases,
while generated code may be acceptable for small or specialized shapes.

Tridiagonal solvers illustrate that not all array-relevant algorithms are
embarrassingly parallel. The Thomas algorithm is efficient sequentially for
tridiagonal systems but has dependencies that limit direct parallelism.
Parallel cyclic reduction exposes more parallelism by recursively eliminating
variables, but it has different constants, memory patterns, and numerical
properties. Heat-flow and PDE-like examples often lead to these algorithms, so
understanding them helps connect array primitives to real scientific workloads.

For RemoraC backend work, algorithm knowledge prevents two classes of mistakes.
The first is semantic: using an algorithm that computes a subtly different
operation, mishandles shapes, or violates dtype behavior. The second is
architectural: emitting a correct but structurally poor implementation that
cannot vectorize, coalesce, reuse memory, or scale. The HIR may say `HIRScan`
or `HIRSort`, but the backend must choose a concrete algorithm with explicit
memory, parallelism, and numerical behavior.

Study these concepts:

- Elementwise maps and map fusion.
- Tree reductions and reduction identities.
- Inclusive/exclusive scans, right scans, and segmented scans.
- Radix sort, bitonic sort, and grade/argsort.
- Scatter/gather, scatter-add, atomics, and race freedom.
- Stencils, convolution, `im2col`, and tiling.
- Matrix multiplication, shared-memory tiling, BLAS, and arithmetic intensity.
- Tridiagonal solvers such as Thomas algorithm and parallel cyclic reduction.

Why it matters here: the compiler lowers high-level Remora operations to these
algorithms on CPU and GPU. The benchmark and heat-flow docs show that real
applications are built from these primitives.

Recommended reading:

- Blelloch, *Prefix Sums and Their Applications*, for scan as a foundational
  parallel primitive.
- Cormen et al., *Introduction to Algorithms*, for baseline sequential and
  parallel algorithmic vocabulary.
- NVIDIA CUDA samples for concrete reductions, scans, matrix multiplication,
  and sorting kernels.
- `docs/HEAT1D_PLAN.md` for a RemoraC application built from array primitives.
- `docs/heat_flow_notes.txt` for numerical-method context around heat-flow
  examples.

### 15. Testing Compilers And Numeric Backends

The project treats compile-only backend tests as insufficient. The highest-risk
failure mode is a silent miscompile: code compiles, runs, and returns wrong
values.

#### Overview

Compiler testing is different from ordinary application testing because a
compiler has two behaviors: it must accept and reject the right programs, and
for accepted programs it must preserve semantics through many transformations.
A parser bug may reject valid syntax. A typechecker bug may accept an ill-shaped
program. A lowering bug may generate invalid MLIR. The worst backend bug is a
silent miscompile: the compiler accepts a valid program, generated code runs,
and the output is wrong. Numeric backends make this worse because outputs may
be large arrays whose errors are not obvious by inspection.

Differential testing compares multiple implementations of the same semantics.
For RemoraC, the interpreter is the primary oracle because it implements the
language directly and supports constructs beyond some compiled backends. A test
can evaluate source with the interpreter, compile the same source for CPU or
GPU, run it, and compare results. NumPy, JAX, Futhark, or hand-written Python
can also serve as oracles for specific algorithms, but the interpreter is the
most project-specific reference because it shares Remora's rank-polymorphic
semantics.

Golden tests serve a different purpose. A golden MLIR test says that a given
input should lower to a particular textual structure. This is useful for
locking down code generation shape, catching accidental pass changes, and
making reviews concrete. But a golden file can preserve a bug. If the expected
MLIR computes the wrong index expression, the test will happily enforce the
wrong behavior. Golden tests should therefore complement semantic tests, not
replace them. They are strongest for structural contracts and weakest for
numeric correctness.

Floating-point comparisons require tolerances. Parallel reductions may combine
values in a different order from the interpreter, and floating-point addition
is not associative. Transcendental functions such as `exp`, `log`, and `sqrt`
may differ slightly across libraries or targets. Tests should use absolute and
relative tolerances appropriate to dtype and operation. At the same time,
tolerances should not be so loose that real errors disappear. Integer and bool
operations should generally use exact comparisons unless the semantics says
otherwise.

Metamorphic testing checks properties rather than only expected outputs. For
array languages, useful properties include shape laws, identity laws, inverse
view laws, and relationships among primitives. Reshaping and then raveling
under suitable layout assumptions should preserve the element sequence.
Transposing twice with inverse permutations should recover the original shape
and values. Mapping the identity function should return the same values. A
scan's last element should match a corresponding reduction for associative
operations. These tests can reveal bugs even when no hand-written expected
array is provided.

Property-based testing goes further by generating many programs or inputs.
For a dependently typed array language, random generation must respect types
and shapes or most generated cases will be meaningless rejections. A useful
generator may produce well-typed small programs, shapes within implementation
limits, dtypes across supported sets, and combinations of maps, folds, views,
and conditionals. Shrinking is valuable: when a generated program fails, the
test framework should reduce it to a smaller counterexample. Even without a
full property-testing framework, parameterized tests over dtype, shape, and
context provide much of this value.

Expected-rejection tests are as important as acceptance tests. A backend should
fail loudly when it cannot guarantee correctness. This is particularly true for
GPU lowering. If closure capture, a dtype, a view context, or a dynamic feature
is unsupported, the compiler should raise a clear error such as
`GPUScaffoldError` or `CodegenUnavailable` rather than emit a kernel that
partially implements the construct. Tests should lock in these rejections so a
future change does not accidentally turn "unsupported" into "silently wrong."

Coverage must include context, not just operation names. A view operation at
top level may use descriptor rewriting. The same view inside a map body may go
through a recursive expression lowering path. A fold over scalar values may
use a different route from a fold over array-valued cells. A GPU fast path may
handle f32 but not i32 or bool. Therefore tests should sweep dtypes, shapes,
and compound contexts. The project guidance that every GPU op needs numeric
parity tests across context exists because real bugs have hidden in these
distinctions.

CI limitations shape testing policy. If CI lacks CUDA hardware and sets
`REMORA_TEST_GPU=0`, GPU tests may skip there even though local development
runs them by default on a GPU machine. That means GPU correctness cannot be
outsourced entirely to CI. Changes to GPU lowering require local parity runs on
appropriate hardware, and PR notes should say so. This is not process
formalism; it is a response to a concrete risk profile where CI cannot execute
the highest-risk backend.

For RemoraC, a good test suite forms a ladder. Parser tests check syntax.
Typechecker tests check static acceptance and rejection. HIR and MLIR tests
check structural lowering. Interpreter tests check semantics. CPU compiled
tests check native execution against the interpreter. GPU parity tests check
accepted GPU routes against the interpreter across shapes, dtypes, and
contexts. Acceptance tests check command-line behavior and user-visible
contracts. When diagnosing a failure, locate which rung first disagrees with
the previous one.

Example testing pattern: do not only check that a GPU map produces PTX. Run the
generated kernel for f32 and i32 inputs, compare against `evaluate_source(...)`,
and include at least one case where the same operation appears inside a compound
context such as a map or fold body.

Study these concepts:

- Differential testing: compare interpreter, CPU, GPU, NumPy, JAX, or Futhark.
- Golden tests for structural output such as MLIR.
- Numeric tolerances for floating-point comparisons.
- Metamorphic testing for algebraic and shape laws.
- Property-based generation of well-typed programs.
- Backend support matrices and expected-rejection tests.
- CI gaps when GPU hardware is unavailable.

Why it matters here: the interpreter is the oracle. GPU changes need numeric
parity tests across dtype, shape, and context, especially inside map/fold bodies
where different lowering paths are used.

Recommended reading:

- `docs/CODEX_PROJECT_REVIEW.md` for known correctness and coverage concerns.
- `docs/UPDATE_CI.md` for the CI/GPU testing context.
- Existing tests under `tests/test_gpu_*`, `tests/test_execution.py`, and
  `tests/acceptance/` for concrete examples of parity, execution, and
  acceptance testing.

### 16. Performance Engineering And Benchmarking

RemoraC is a research compiler, but the docs set concrete performance goals.
Understanding performance requires measuring execution separately from compile
time and separating algorithmic problems from implementation overhead.

#### Overview

Performance engineering begins with measurement discipline. A compiler can
make a program fast in several different senses: it can reduce compile time,
reduce first-call latency, reduce steady-state execution time, reduce memory
traffic, reduce allocation, or improve scaling across cores and devices. These
goals are related but not identical. A benchmark that includes compilation time
answers a different question from one that measures repeated execution of an
already compiled function. A GPU benchmark that includes host/device transfer
answers a different question from one that keeps arrays resident on the device.

Throughput and latency are different metrics. Latency is the time for one
operation or request. Throughput is work per unit time, often elements per
second or bytes per second. Small arrays are often latency-bound: overheads
such as Python calls, kernel launches, dynamic loading, or allocation dominate.
Large arrays are often throughput-bound: the system spends most of its time
moving data or doing arithmetic. Warmup also matters. JIT-like APIs, dynamic
library loading, GPU context initialization, and first-use compilation can make
the first run much slower than later runs.

Roofline thinking is a useful mental model. Each kernel has an arithmetic
intensity: operations per byte of memory traffic. Hardware has a peak compute
rate and a peak memory bandwidth. A low-intensity kernel such as a simple
elementwise map may be memory-bound because it performs little arithmetic per
loaded and stored element. A high-intensity kernel such as well-tiled matrix
multiplication may be compute-bound because each loaded value is reused many
times. The roofline model helps explain why optimizing arithmetic instructions
does little for a memory-bound kernel and why improving locality can matter
more than reducing a few scalar operations.

Allocation and materialization are frequent hidden costs. A source-level array
expression may create several conceptual intermediates. If the compiler
materializes all of them, performance can be dominated by allocation and memory
traffic. Fusion avoids intermediates by combining operations into one loop or
kernel. View operations avoid copying by changing descriptors. Memory pools
reduce allocation overhead by reusing buffers. Buffer liveness analysis enables
safe reuse of temporaries. These optimizations are particularly important on
GPU, where allocation and transfer overheads can dominate small or chained
operations.

Device-resident execution is a key distinction for GPU benchmarking. If every
operation copies inputs from host to device, launches a kernel, and copies
outputs back, the benchmark measures PCIe or interconnect traffic as much as
kernel performance. Many real workloads transfer data once, run many kernels,
and transfer results at the end. RemoraC's device arrays, execution plans, and
memory pools point toward that model. Benchmarks should state which model they
measure: end-to-end including transfers, or device-resident kernel execution.

Fair comparison is difficult. NumPy may call optimized C loops or BLAS, but it
also may allocate temporaries for chained expressions. JAX may include tracing
and compilation unless warmed up correctly. Futhark may generate highly
optimized fused GPU code but has its own compilation model. BLAS and CUB are
specialized libraries written by experts for important primitives. A research
compiler should compare against these systems honestly, but the benchmark must
make clear whether it is comparing language expressiveness, compile time,
steady-state speed, memory use, or a specific primitive implementation.

Algorithmic differences must be separated from implementation overhead. If
RemoraC uses an asymptotically worse sort algorithm than CUB, no amount of
minor code generation tuning will close the gap. If both systems use similar
algorithms but RemoraC launches many more kernels or materializes unnecessary
intermediates, the issue is plan composition or fusion. If generated CPU loops
are correct but fail to vectorize, the issue may be MLIR structure, aliasing,
strides, or compiler flags. Benchmark results should lead to hypotheses at the
right level.

Shape and dtype sweeps are more informative than one-off timings. Small,
medium, and large arrays expose different overhead regimes. Contiguous and
strided views expose memory-layout effects. f32, f64, i32, and bool may take
different lowering paths or use different hardware units. Reductions over short
rows differ from reductions over long vectors. GPU kernels may need enough
elements to fill the device. A benchmark that only tests one friendly shape can
misrepresent both strengths and weaknesses.

Correctness remains a prerequisite. Performance work can easily introduce
miscompiles by changing loop orders, fusing operations with hidden dependencies,
assuming contiguity, or using unsafe fast math. Every performance optimization
should have semantic tests that cover the transformed cases. For floating-point
programs, the acceptable numerical differences should be explicit. For integer,
bool, shape, and view behavior, exactness is usually expected. The fastest
benchmark result is useless if it computes the wrong array.

For RemoraC, performance engineering connects directly to the roadmap. CPU
map/fold throughput depends on vectorization, loop structure, and memory
layout. GPU small-problem performance depends on launch overhead and
host/device transfer amortization. Large GPU sort and fold performance depends
on algorithm choice, memory coalescing, and execution plans. AD optimization
examples depend on generated-code size, simplification, and state-loop
lowering. Benchmark documents should therefore be read not just as scorecards,
but as diagnostic reports that identify which compiler layer needs work.

Study these concepts:

- Throughput, latency, warmup, compile time, and execution time.
- Roofline thinking: memory bandwidth vs compute throughput.
- Kernel launch overhead and data-transfer amortization.
- Device-resident execution vs host-chained execution.
- Allocation cost, memory pools, and buffer reuse.
- Fusion and avoiding intermediate materialization.
- Fair comparisons against NumPy, JAX, Futhark, BLAS, and CUB.

Why it matters here: benchmark docs show RemoraC is competitive on some scans,
stencils, GPU folds, and large GPU sorts, while CPU map/fold vectorization,
small-GPU launch overhead, fusion, and plan composition remain active areas.

Recommended reading:

- `docs/BENCHMARK_PLAN.md` for intended benchmark methodology and targets.
- `docs/BENCHMARK_RESULTS.md` for current measured performance behavior.
- `docs/thoughts.txt` for informal performance and roadmap observations.
- Williams et al., *Roofline: An Insightful Visual Performance Model*, for the
  memory-bandwidth vs compute-throughput framework.
- Futhark performance and memory papers in `docs/remora-reference/` for mature
  array-compiler optimization strategies.

## Topic Map By Project Area

### To Understand The Language

Learn:

- Array programming
- Rank, shape, frame, cell, lifting, reranking
- Remora surface syntax
- Higher-order functional programming
- Static shape typing

Read:

- `docs/USER_GUIDE.md`
- `docs/DENSE_CORE.md`
- `docs/remora-reference/remora-tutorial-draft.txt`
- `docs/remora-reference/semantics-of-rank-polymorphism.txt`

### To Understand The Typechecker

Learn:

- Bidirectional typechecking
- Dependent/indexed types
- Shape constraints
- Higher-rank polymorphism
- Type erasure and monomorphization

Read:

- `docs/remora-reference/slepak-dissertation.txt`
- `docs/remora-reference/semantics-of-rank-polymorphism.txt`
- `remora/typechecker.py`
- `remora/types.py`
- `remora/constraints.py`
- `remora/dependent_types.py`
- `remora/index.py`

### To Understand CPU Compilation

Learn:

- MLIR structured ops
- Tensor-to-memref lowering
- LLVM lowering and linking
- C ABI and ctypes
- Loop optimization and vectorization

Read:

- `docs/PROJECT_OVERVIEW_AND_ARCHITECTURE.md`
- `docs/ABI.md`
- `docs/IMPLEMENTATION_NOTES.md`
- `remora/lowering/`
- `remora/pipeline.py`
- `remora/runtime.py`

### To Understand CPU Vector Instructions

Learn:

- SIMD execution
- Loop vectorization
- Cache locality and strides
- Reductions and scans under dependencies
- OpenMP-style threading

Read:

- LLVM loop/vectorizer docs
- MLIR vector dialect docs
- `docs/BENCHMARK_RESULTS.md`
- CPU pipeline code in `remora/pipeline.py`

### To Understand GPU Compilation

Learn:

- CUDA execution model
- PTX/NVVM/LLVM lowering
- Descriptor ABI kernel arguments
- Parallel reductions, scans, sorting, views, and memory coalescing
- Multi-kernel execution plans

Read:

- `docs/ABI.md`
- `docs/BACKEND_GAPS.md`
- `docs/IMPLEMENTATION_LOG.md`
- `remora/codegen.py`
- `remora/gpu_lowering.py`
- `remora/_gpu_expr_lowering.py`
- `remora/execution_plan.py`
- `remora/executor.py`

### To Understand Future Full Remora Work

Learn:

- Dynamic shapes
- Runtime existential packages/boxes
- Ragged arrays
- Segmented reductions
- Records and arrays of records
- Dynamic higher-order dispatch
- Cost models and scheduling

Read:

- `docs/PLAN_TO_IMPLEMENT_FULL_REMORA.md`
- `docs/ROADMAP.md`
- `docs/BACKEND_GAPS.md`
- `docs/WORKSTREAM_0_PLAN.md`
- `docs/remora-reference/Records with Rank Polymorphism.txt`

## Suggested Reading Order

### Phase 1: User-Level Remora

1. `docs/USER_GUIDE.md`
1. `docs/DENSE_CORE.md`
1. `docs/remora-reference/remora-tutorial-draft.txt`
1. Run examples with `uv run remorac --target interp ...` and
   `uv run remorac ...`

Goal: understand how rank-polymorphic Remora programs are written.

### Phase 2: Formal Semantics And Types

1. `docs/remora-reference/semantics-of-rank-polymorphism.txt`
1. `docs/remora-reference/slepak-dissertation.txt`
1. Dunfield and Krishnaswami on bidirectional typing
1. `remora/typechecker.py` with tests open beside it

Goal: understand how Remora's types determine shape-safe execution.

### Phase 3: Compiler Architecture

1. `docs/PROJECT_OVERVIEW_AND_ARCHITECTURE.md`
1. `docs/IMPLEMENTATION_NOTES.md`
1. `docs/IMPLEMENTATION_LOG.md`
1. Inspect output from `--emit-ast`, `--emit-typed-ast`, `--emit-hir`, and
   `--emit-mlir`

Goal: understand how the source program becomes HIR and MLIR.

### Phase 4: CPU Backend

1. MLIR Toy tutorial and dialect docs
1. `docs/ABI.md`
1. `remora/lowering/`
1. `remora/pipeline.py`
1. `remora/runtime.py`

Goal: understand descriptor-based CPU compilation and execution.

### Phase 5: GPU Backend

1. CUDA programming guide
1. PTX ISA overview
1. `docs/BACKEND_GAPS.md`
1. `remora/codegen.py`
1. `remora/gpu_lowering.py`
1. `remora/_gpu_expr_lowering.py`
1. GPU parity tests

Goal: understand which programs are accepted on GPU, which lowering route they
take, and how correctness is tested.

### Phase 6: Performance And Research Directions

1. `docs/BENCHMARK_PLAN.md`
1. `docs/BENCHMARK_RESULTS.md`
1. `docs/remorac-vs-futhark.md`
1. Futhark PLDI 2017, incremental flattening, and memory papers
1. `docs/ROADMAP.md`
1. `docs/PLAN_TO_IMPLEMENT_FULL_REMORA.md`

Goal: understand where RemoraC is strong, where it is behind mature systems,
and what full-language work requires.

## Minimal Background Checklist

Before working on core compiler changes, you should be comfortable with:

- Reading simple Remora ML and Lisp programs.
- Explaining rank, shape, frame, cell, lifting, and reranking.
- Explaining why static shapes make this compiler tractable today.
- Reading Python dataclass-style AST and IR definitions.
- Following a program through AST, typed AST, HIR, and MLIR.
- Reading simple MLIR with `func`, `arith`, `tensor`, `linalg`, `scf`, and
  `memref` operations.
- Understanding the descriptor ABI fields: `allocated`, `aligned`, `offset`,
  `sizes`, and `strides`.
- Explaining why GPU compile-only tests are not enough.
- Explaining why CPU vectorization and GPU execution are both memory-layout
  problems as much as code-generation problems.

## Deeper Reading List

Language and semantics:

- Shivers, Slepak, Manolios, *An Introduction to Rank-polymorphic Programming
  in Remora*
- Slepak, Shivers, Manolios, *The Semantics of Rank Polymorphism*
- Slepak, *A Typed Programming Language: The Semantics of Rank Polymorphism*
- Iverson, *A Programming Language*
- Backus, *Can Programming Be Liberated from the von Neumann Style?*

Type systems:

- Pierce, *Types and Programming Languages*
- Dunfield and Krishnaswami, *Complete and Easy Bidirectional Typechecking for
  Higher-Rank Polymorphism*
- Xi and Pfenning, *Dependent Types in Practical Programming*
- Eisenberg, *Dependent Types in Haskell: Theory and Practice* for modern
  dependent-type intuition

Compilers:

- Appel, *Modern Compiler Implementation*
- Cooper and Torczon, *Engineering a Compiler*
- Muchnick, *Advanced Compiler Design and Implementation*
- MLIR documentation: LangRef, Toy tutorial, Dialects, Bufferization
- LLVM LangRef

Array and parallel compilers:

- Henriksen et al., *Futhark: Purely Functional GPU-programming with Nested
  Parallelism and In-place Array Updates*
- Futhark incremental flattening paper
- Futhark memory optimization paper
- Futhark defunctionalisation paper
- AUTOMAP rank-polymorphism paper in `docs/remora-reference/`

CPU/GPU performance:

- NVIDIA CUDA C Programming Guide
- NVIDIA PTX ISA
- Blelloch, *Prefix Sums and Their Applications*
- Williams et al., *Roofline: An Insightful Visual Performance Model*
- Agner Fog optimization manuals
- LLVM Loop Vectorizer documentation

Automatic differentiation:

- Baydin et al., *Automatic Differentiation in Machine Learning*
- Futhark AD SC22 paper in `docs/remora-reference/`

## Practical Exercises

1. Run a small Remora program and inspect each compiler stage:

   ```bash
   uv run remorac --emit-ast examples/prelude_sum.remora
   uv run remorac --emit-typed-ast examples/prelude_sum.remora
   uv run remorac --emit-hir examples/prelude_sum.remora
   uv run remorac --emit-mlir examples/prelude_sum.remora
   ```

1. Write a scalar function and apply it to a vector, matrix, and rank-3 tensor.
   Predict the frame/cell split before running it.

1. Write a row-wise reduction using `rerank` or a function expecting vector
   cells. Compare interpreter and CPU output.

1. Pick a view operation such as `transpose`, `reshape`, `ravel`, or `take`.
   Trace how its shape changes at the type level and how its descriptor offset,
   sizes, and strides should change.

1. For a GPU-supported map, find the numeric parity test that proves it computes
   the same result as the interpreter. Add a dtype or shape variation.

1. Run a benchmark from `docs/BENCHMARK_PLAN.md` and classify the bottleneck:
   compile time, launch overhead, memory bandwidth, missing fusion, missing
   vectorization, or algorithm choice.
