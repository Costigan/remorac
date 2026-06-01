# Remora Dense Core Implementation Notes

This file records implementation decisions made while building the prototype.
It is intentionally practical: normative contracts stay in `docs/ABI.md` and
the phase plan stays in `docs/MLIR_IMPLEMENTATION_PLAN.md`.

## Current Scope

The implementation is currently limited to the Phase 0 foundation through a
CPU-first Phase 7/8 usability slice:

- Python package skeleton and dependency metadata.
- Rank-0 through rank-3 external ABI descriptor structs.
- Parser and AST for the Dense Core surface subset.
- Static type representation and a small typechecker for literals, `iota`,
  `let`, `if`, primitive operators, `map`, and `fold`.
- HIR definitions and typed-AST-to-HIR lowering for the accepted typed subset.
- Defunctionalization for inline non-capturing lambdas used by `map`/`fold`.
- Textual MLIR lowering for `iota`, primitive scalar section maps directly over
  `iota`, and simple lifted scalar lambda maps directly over `iota`, validated
  by parsing through `iree.compiler.ir`.
- Textual MLIR lowering for scalar `fold` over direct `iota` and over a direct
  scalar map of `iota`.
- Simple `HIRLet` inlining before MLIR emission, enough for local and top-level
  value aliases such as `let xs = iota 10 in map (* 2.0) xs`.
- Textual MLIR lowering for static array literals using `tensor.from_elements`,
  including rank-1, rank-2, and rank-3 examples.
- Scalar elementwise maps now lower for rank-0 scalar inputs, rank-1 maps over
  `iota`, and rank-1 through rank-3 maps over static array literals. Ranked
  maps use identity affine maps with one parallel iterator per dimension.
- Binary scalar elementwise maps now lower for rank-0 through rank-3 inputs.
  Ranked binary maps use multi-input `linalg.generic`; rank-0 binary maps lower
  as scalar function application.
- Nested scalar maps lower for the current direct tensor input subset, including
  map chains over `iota` and static array literals.
- Standalone scalar literals and primitive scalar expressions lower to
  parse-validated MLIR, including integer and float arithmetic, `/`, numeric
  comparisons, boolean `&&`/`||`, and explicit `int` to `float` casts.
- Phase 6 pipeline helpers can run an in-process validation pipeline, verify
  emitted MLIR through `mlir-opt` or `iree-opt` when available, and use
  `iree-compile` to produce CUDA PTX for current lowered modules.
- `remora.compiler` exposes public source-to-MLIR and source-to-PTX helpers.
- `remora.runtime` contains both a typed-AST interpreter and a compiled CPU
  executor. The compiled path lowers through standalone MLIR/LLVM tools, emits
  an object with `llc-18`, links a temporary shared library with `gcc`/`cc`, and
  calls `main` through `ctypes`.
- `remorac` is registered as a console script with compiled CPU execution by
  default, an explicit `--target interp` reference evaluator, MLIR/PTX
  inspection targets, and AST/typed-AST/HIR/MLIR/PTX emit flags.
- `remora` is registered as a CPU-only REPL console script with persistent
  value definitions, `:type`, `:mlir`, `:load`, `:reset`, `:target`, and
  `:help`.

Full descriptor-ABI code generation, CUDA launch path, dynamic shapes, dynamic
rank, or automatic differentiation has not been implemented.

## Backend Planning Decisions

- The production backend is standard LLVM/MLIR, not IREE. The current
  `iree-compiler` dependency remains useful for parser/verifier scaffolding and
  temporary PTX inspection, but IREE HAL PTX does not satisfy the Remora runtime
  ABI milestones.
- The standalone CPU/fusion toolchain is pinned to LLVM/MLIR 18.1.3 and
  recorded in `docs/MLIR_TOOLCHAIN.md`. Production NVIDIA lowering remains
  deferred until Remora emits direct `gpu.module` kernels using the descriptor
  ABI.
- CPU compiled execution comes before CUDA execution. The typed-AST evaluator
  is now an explicit `--target interp` reference path; `remorac --target cpu`
  runs the compiled CPU executor for the lowered Dense Core subset.
- Binary scalar-cell maps now lower to MLIR and unblock prelude `dot` as binary
  map plus fold.
- Fusion, kernel-count, and smoke timing tests are now part of the vertical
  slice. They should land before broadening the language beyond Dense Core.

## Project and Tooling

- The project uses `pyproject.toml` with Python `>=3.11`.
- Runtime/prototype dependencies are `lark`, `numpy`, `cuda-python`, and
  `iree-compiler`.
- Tests use `pytest`.
- `uv` is available and has been used to create/update `uv.lock`.
- In the sandbox, test commands use `UV_CACHE_DIR=/tmp/uv-cache` to avoid
  writes to the default user cache.

## ABI Decisions

- `remora.abi` defines the exact rank-specialized ctypes structs from
  `docs/ABI.md`:
  - `RemoraMemRef0`
  - `RemoraMemRef1`
  - `RemoraMemRef2`
  - `RemoraMemRef3`
- Descriptor fields are literal ctypes fields, not packed arrays:
  `allocated`, `aligned`, `offset`, followed by rank-specific `sizeN` and
  `strideN` fields.
- Sizes, strides, and offsets use signed 64-bit integers.
- Strides are stored in elements, not bytes.
- `make_memref_descriptor` accepts a pointer value, shape, strides, dtype, and
  offset. The dtype is validated for caller clarity but is not stored in the ABI
  descriptor.
- `make_numpy_memref_descriptor` uses the base numpy allocation for
  `allocated == aligned` and represents view displacement with `offset`.
- Numpy view support is already covered for transposed and sliced arrays. This
  follows `docs/ABI.md`: view offsets are not hidden by changing `aligned`.

Deferred ABI/runtime work:

- CPU `ExecutionEngine` ABI round trips.
- CUDA ABI round trips.
- Kernel metadata describing descriptor element types.
- Adapter kernels if MLIR lowered memrefs do not match the external ABI.

## Parser Decisions

- The parser uses Lark LALR with separate starts for:
  - `program`
  - `definition`
  - `expr`
- Public parser entry points are:
  - `parse_program`
  - `parse_definition`
  - `parse_expr`
  - `parse_file`
  - `parse_repl_input`
- `parse_repl_input` tries a definition first, then an expression.
- Infix operators are parsed into `AppExpr(VarExpr(op), [left, right])`.
  This keeps primitive operation handling in the typechecker instead of adding
  many operator-specific AST nodes.
- Operator sections have explicit AST nodes:
  - `OperatorFuncExpr`
  - `LeftSectionExpr`
  - `RightSectionExpr`
- Newlines are significant only at the top-level program boundary so a
  definition body does not accidentally consume the following final expression.
- A newline is allowed immediately after `in`, `then`, and `else` so checked-in
  examples can use readable multi-line `let` and conditional forms.
- Prelude injection strips leading blank/comment-only lines from user source
  before prepending definitions. This avoids creating a blank top-level
  separator between the injected prelude definitions and a commented example
  body.
- Source locations currently store filename plus placeholder line/column `0`.
  Precise source spans are deferred.

Known parser limitation:

- An array literal immediately following another atom is ambiguous with index
  syntax. Tests use `let` bindings for array operands in `map`/`fold` cases
  where needed. This should be revisited when indexing syntax is finalized.

## Typechecker Decisions

- Dense Core types are:
  - `ScalarType`
  - `ArrayType`
  - `FuncType`
  - `StaticDim`
- Only static non-negative integer dimensions are accepted.
- `eval_static_dim` currently accepts integer literals only. Broader constant
  folding is deferred.
- Rank is limited to 0 through 3. Rank-4 results raise a Dense Core rank-limit
  error.
- Array literals recursively enforce consistent element type and nested shape.
- Empty array literals are rejected until explicit type annotations exist.
- `iota n` has type `int[n]`.
- `shape expr` and `rank expr` are static metadata operations in Dense Core.
  Function operands are rejected as deferred. For scalar operands, `rank`
  returns `0` and `shape` has type `int[0]`.
- Array indexing is typed for static rank-1 through rank-3 arrays. Each index
  must be `int`; full-rank indexing returns a scalar, and partial indexing
  drops the indexed outer dimensions and returns the remaining array cell.
- Primitive numeric behavior:
  - `int op int -> int` for `+`, `-`, `*`
  - mixed `int`/`float` promotes to `float`
  - `/` returns `float`
  - comparisons return `bool`
  - `&&` and `||` require `bool`
- Numeric promotions are explicit in the typed tree with `TypedCast`.
- Lambdas and operator sections are accepted only when checked against an
  expected function type, currently through `map` or `fold`.
- There is one narrow local lambda exception for CPU examples:
  `let f = \x -> body in f arg` is inferred from the direct application
  argument type. General standalone lambda inference remains deferred.
- `map` tries scalar cells first, then progressively larger suffix cell shapes.
  This supports scalar maps over rank-1/2/3 arrays and vector-cell maps such as
  row reductions.
- Binary `map` is supported for scalar cells over two scalar values or two
  arrays with identical static shapes. This is the compiler-shaped CPU path used
  by the starter `dot` prelude helper. Mixed array/scalar binary maps and
  array-valued binary map cells are deferred.
- `fold` supports scalar accumulator folds over rank-1 arrays and array-cell
  folds over rank-2/rank-3 arrays for the primitive fold-callable subset.
- Top-level value definitions are supported.
- Top-level function definitions are supported when used as statically known
  direct call targets or unary `map` callables. The typechecker specializes the
  function body at the call site from the concrete argument types and represents
  the callee as a typed static lambda. Recursive top-level functions are
  rejected as deferred.
- Typed array literals and top-level value definitions preserve their typed
  children so later HIR lowering does not need to re-run type inference.
- Division operator functions and sections require numeric operands just like
  ordinary division expressions. Regression tests cover bool operands for both
  `map` sections and `fold (/)`.

Deferred typechecker work:

- Function annotations and a general top-level function type inference story
  beyond call-site specialization.
- Type variables or a real bidirectional annotation story for standalone
  lambdas beyond the direct local application pattern.
- Compile-time constant folding for shape expressions.
- Static bounds checking for literal indices.
- Composition typing.
- Generalized array-cell folds.
- Better diagnostic locations and source spans.

## HIR Decisions

- `remora.hir` defines a small functional HIR for the typed subset currently
  accepted by the typechecker.
- `lower_to_hir` lowers a `TypedProgram` into an `HIRProgram`.
- Top-level value definitions are lowered by wrapping the main expression in
  nested `HIRLet` nodes. No top-level storage model exists yet.
- Programs with top-level value definitions but no body are rejected by HIR
  lowering instead of silently dropping the definitions.
- User-authored top-level function definitions lower for the current static
  subset by specializing them at direct use sites as typed static lambdas. Direct
  scalar calls inline through `HIRLet`; unary `map` callables go through the
  existing lambda-lifting path. Full top-level `HIRFunction` generation remains
  deferred.
- `HIRMap` carries the frame shape and cell shape resolved by the typechecker.
  This is the key metadata the later linalg lowering will need.
- `HIRFold` carries the outer reduction dimension resolved from the typed array.
- `shape` and `rank` lower to constants from type metadata. `rank` becomes an
  `HIRLit`; `shape` becomes an `HIRArrayLit` containing static dimensions.
  Scalar shape lowers to an empty `int[0]` HIR array.
- `HIRIndex` carries the lowered array expression, lowered index expressions,
  and the type after dropping indexed outer dimensions.
- Primitive scalar operations lower to `HIRPrimOp` with typed operation names
  like `+f`, `*i`, and comparison/bool suffixes.
- Numeric promotions lower to explicit `HIRCast` nodes.
- Lambdas lower to `HIRLambda` and are still present in HIR. They are not yet
  lambda-lifted or defunctionalized.
- Operator functions and sections lower to `HIRPrimCallable`. Sections retain
  the bound left or right operand as an HIR expression.

Deferred HIR work:

- Top-level `HIRFunction` generation from checked function definitions.
- HIR lowering for composition and generalized conditionals.
- A richer primitive operation naming scheme may be needed before MLIR lowering
  for comparisons and bool operations.

## Defunctionalization Decisions

- `remora.defunc` provides `defunctionalize(HIRProgram) -> HIRProgram`.
- Inline `HIRLambda` callables at `HIRMap` and `HIRFold` sites are lifted into
  generated top-level `HIRFunction`s named `__lambda_N`.
- The original HOF site is rewritten to an `HIRVar` pointing at the generated
  function and carrying the lambda's `FuncType`.
- Primitive operator callables and operator sections remain `HIRPrimCallable`.
  They do not need generated functions at this stage.
- Existing named function references represented as `HIRVar` are already static
  and pass through unchanged.
- Lambdas that capture outer variables are rejected with an explicit deferred
  closure-conversion diagnostic.
- A bare `HIRLambda` in expression position after defunctionalization is also
  rejected as a deferred dynamic higher-order function.

Deferred defunctionalization work:

- Closure conversion or lambda lifting with explicit captured scalar arguments.
- Monomorphization for higher-order top-level function parameters.
- Static analysis for top-level function values once function definitions have
  real type checking.
- Dynamic dispatch tags/closure structs remain out of Dense Core scope.

## MLIR Lowering Decisions

- `remora.lowering` provides the first Phase 5 lowering spike.
- The installed `iree-compiler` package exposes `iree.compiler.ir`, core type
  parsing, module parsing, and several dialect modules.
- A top-level `mlir` Python package is not installed in this environment.
- Importing `iree.compiler.dialects.linalg` currently fails because PyYAML is
  not installed. The project has not added PyYAML just to use generated Python
  builders.
- Because of that API shape, the first lowering slice emits textual MLIR and
  immediately validates it with `iree.compiler.ir.Module.parse`.
- The current lowering supports `HIRIota` as the program body and scalar
  `HIRMap` over a direct `HIRIota` array when the callable is a primitive
  operator section with a literal bound operand or a lifted unary `HIRFunction`
  from defunctionalization.
- `HIRFold` lowers for rank-1 scalar reductions and rank-2/rank-3 array-cell
  reductions over the outermost dimension when the fold callable is a primitive
  operator function. Rank-2/rank-3 folds use one reduction iterator followed by
  parallel iterators for the remaining result dimensions.
- Scalar `HIRMap` lowers over rank-0 scalar inputs, direct `HIRIota`, direct
  static `HIRArrayLit`, and nested scalar `HIRMap` inputs for scalar-cell maps
  only.
- Binary scalar-cell `HIRMap` lowers for rank-0 through rank-3 inputs. Ranked
  binary maps emit a multi-input `linalg.generic` with one identity indexing map
  for each input and one identity map for the output. Rank-0 binary maps lower
  as scalar function application in `main`.
- Cell `HIRMap` lowers for the current rank-1-cell reduction pattern, e.g.
  `map (\row -> fold (+) 0 row) xs`, producing row/cell reductions over rank-2
  and rank-3 inputs.
- `_lower_let` lowers scalar `HIRLet` nodes through a small SSA environment.
  Tensor lets still use the simple HIR inlining path until the lowerer has a
  general tensor SSA value environment.
- Scalar `HIRFunction` and `HIRCall` lower to `func.func private` and
  `func.call` for manually constructed/static HIR functions. User-authored
  top-level function definitions also reach MLIR for the current direct-call and
  unary-map subset via call-site static lambda specialization, not by emitting
  general top-level HIR/MLIR functions.
- `type_to_mlir` covers scalar types, static ranked tensor types, and function
  type spelling for tests.
- `MLIRLowering.lower_type` parses the textual type spelling into a real MLIR
  type object.
- `MLIRLowering.lower_program` emits a `func.func @main` containing
  `tensor.empty`, `linalg.generic`, `linalg.index`, `arith.index_cast`, and
  `linalg.yield` for `iota`.
- For `map (* 2.0) (iota 10)`, lowering emits two `linalg.generic` operations:
  one for `iota`, then one scalar elementwise map using explicit `arith.sitofp`,
  `arith.constant`, and `arith.mulf`.
- For `map (\x -> x * 2.0) (iota 10)` and `map (\x -> x * x) (iota 10)`,
  defunctionalized lifted functions are currently inlined into the map
  `linalg.generic` body. Separate MLIR `func.func` emission for lifted functions
  is deferred.
- For `fold (+) 0.0 (map (* 2.0) (iota 10))`, lowering emits three
  `linalg.generic` operations: iota, scalar map, and scalar reduction. The fold
  uses `tensor.from_elements` for the scalar initial accumulator and
  `tensor.extract` to return the rank-0 tensor result as a scalar.
- Prelude `dot` now lowers through MLIR as binary map plus fold. For
  let-bound rank-1 vectors, the parse-validated MLIR contains one parallel
  multi-input `linalg.generic` for multiplication and one reduction
  `linalg.generic` for summation.
- For array-cell folds such as `fold (+) [0, 0] [[1, 2], [3, 4]]`, lowering
  emits a `linalg.generic` with an input identity map, an output map that drops
  the outer reduction dimension, and `iterator_types = ["reduction", ...]`.
- For cell maps whose body is a fold over each rank-1 cell, lowering initializes
  the output with `linalg.fill` and emits a `linalg.generic` with parallel frame
  iterators and one reduction iterator for the cell dimension.
- Top-level value definition programs like `def xs = iota 10` followed by
  `map (* 2.0) xs` lower through the same let-inlining path.
- Static array literals lower by flattening nested `HIRArrayLit` elements in
  row-major order and emitting scalar constants followed by `tensor.from_elements`.
- Static `shape` and `rank` expressions are already lowered by HIR to constants.
  Non-empty shapes use `tensor.from_elements`; scalar `shape` uses
  `tensor.empty() : tensor<0xi32>`.
- Full-rank `HIRIndex` with literal integer indices lowers to `tensor.extract`
  for tensor-producing expressions such as `iota` and static array literals.
- Standalone `HIRLit`, `HIRCast`, and `HIRPrimOp` expressions lower through a
  small scalar-region emitter. The same emitter is used for simple lifted
  lambda bodies inside scalar maps.
- Rank-0 `HIRMap` lowers as scalar function application inside `main`, not as
  `linalg.generic`, because there is no frame iteration to materialize.
- `_lower_prim_op` support currently covers all scalar primitive operations
  accepted by the typechecker: integer/float arithmetic, floating division,
  numeric comparisons, and boolean `and`/`or`.
- Boolean constants are emitted in a form the parser accepts and canonicalizes
  back to `arith.constant true`/`false` in printed MLIR.
- The current textual MLIR output is locked down with checked-in golden
  fixtures under `tests/golden_mlir/` for the implemented `iota`, scalar map,
  rank-2 literal map, and map-then-fold slices. These fixtures validate the
  current parse-checked textual lowering path, not generated Python builder
  APIs.

Deferred MLIR lowering work:

- Switch to dialect builders if/when the required generated bindings and their
  dependencies are stable in the project environment.
- Lower generalized non-direct tensor values beyond the current nested
  scalar-map subset, generalized array-cell fold callables beyond primitive
  operators, and generalized cell maps beyond rank-1-cell fold bodies.
- Lower partial indexing to `tensor.extract_slice` or another array-cell
  representation. Dynamic index expression lowering is also deferred.
- Replace tensor let inlining with real SSA environment lowering when lowering
  grows beyond direct iota/map/fold slices.
- Run `mlir-opt --verify-diagnostics` checks beyond parse validation once
  `mlir-opt` is available in the development environment.
- Runtime-dependent `shape` lowering with `tensor.dim` remains deferred until
  dynamic dimensions are introduced.

## Pipeline and Codegen Decisions

- `remora.pipeline` owns Phase 6 toolchain detection and pass-manager plumbing.
  It checks `PATH` first and then the active Python environment's script
  directory, so `.venv/bin/iree-opt` and `.venv/bin/iree-compile` are detected
  even when the virtualenv is not activated in the shell.
- Standalone LLVM/MLIR tools are detected with versioned executable fallbacks:
  `/usr/bin/mlir-opt-18`, `/usr/bin/mlir-translate-18`, and `/usr/bin/llc-18`.
- The installed `iree-compiler` package exposes `iree.compiler.passmanager`,
  and the validation pipeline `builtin.module(canonicalize,cse)` runs against
  current lowered modules.
- `verify_module_text` uses standalone `mlir-opt` when present and otherwise
  uses `iree-opt --verify-diagnostics -`.
- `CPU_PIPELINE` is validated through standalone `mlir-opt-18`; current Dense
  Core modules lower to LLVM dialect and translate to LLVM IR through
  `mlir-translate-18`.
- `FUSION_PIPELINE` is validated through standalone `mlir-opt-18`; map chains
  and the current dot lowering fuse down to one `linalg.generic`. The
  map-then-fold milestone shape currently lowers from three `linalg.generic`
  ops to two after fusion, so full map/reduce fusion remains tracked.
- `remora.codegen.generate_ptx` uses `iree-compile` with the CUDA HAL backend
  and `--iree-hal-dump-executable-files-to` to obtain emitted `.ptx` files.
  This proves the current lowered MLIR can reach PTX with the installed IREE
  toolchain.
- The PTX produced today is an IREE HAL dispatch kernel. Its launch ABI is not
  the final Remora external memref-descriptor ABI from `docs/ABI.md`; direct
  CUDA launching remains a Phase 7/runtime integration task.
- `KernelMeta` extraction is intentionally minimal and reflects only stable
  facts in the generated PTX today: entry name, PTX parameter count, and
  `.maxntid` block size. Input/output element types are left empty until the
  final Remora kernel ABI is generated.
- The in-process IREE pass registry still does not recognize the standalone CPU
  lowering pipeline. That path raises `PipelineUnavailable`; the validated
  production-style path is the external standalone `mlir-opt-18` runner.

Deferred pipeline/codegen work:

- Install `ptxas` for standalone PTX assembly checks.
- Lower Remora modules to explicit `gpu.module` / `gpu.func` kernels and
  validate a production NVIDIA NVVM pipeline against the descriptor ABI.
- Replace the IREE HAL dispatch PTX path with a final direct-launch Remora ABI
  path or add an adapter layer that makes the ABI boundary explicit.
- Replace the temporary shared-library CPU executor with a direct MLIR
  `ExecutionEngine` binding if/when compatible Python bindings are available.
- Move compiled CPU execution to the final descriptor-output ABI instead of
  returning MLIR heap-allocated memref structs from `main`.

## Compiler Facade and CPU Runtime Decisions

- `remora.compiler.compile_source` is the public source-to-compiler-artifact
  path. It parses, typechecks, lowers to HIR, defunctionalizes, lowers to MLIR,
  runs the validation pipeline, and verifies textual MLIR when an external
  verifier is available.
- `compile_source_to_mlir` and `compile_source_to_ptx` provide small public
  helpers for examples, CLI plumbing, and future tests.
- The compiled CPU runtime lowers MLIR to LLVM IR, emits a temporary object with
  `llc-18`, links a temporary shared library with `gcc`/`cc`, and calls `main`
  with `ctypes`.
- The typed-AST evaluator remains available as `--target interp` and as a test
  oracle for cases that have not been lowered to compiled MLIR yet.
- CPU execution returns Python scalars or numpy arrays plus the checked Remora
  type. Arrays use numpy dtypes matching the Dense Core scalar policy:
  `int32`, `float32`, and `bool`.
- `remora.display.format_result` is the shared result formatter for `remorac`
  and the REPL. It prints booleans as `true`/`false`, preserves a decimal point
  for float scalars and float arrays, and supports vectors, matrices, and
  rank-3 arrays through numpy rendering with Remora scalar formatting.
- The interpreter covers the checked-in examples: scalar arithmetic, conditionals,
  top-level value definitions, direct top-level function calls, top-level
  functions used as unary `map` callables, `iota`, `map`, `fold`, nested maps,
  row reductions, rank-2/rank-3 literals, operator sections, and the narrow
  direct local lambda application pattern.
- The compiled CPU executor covers the Dense Core acceptance subset and
  additional tests for scalar values, vectors, matrices, rank-3 arrays, vector
  sum, dot product, static `shape`/`rank`, and booleans.
- `stdlib/prelude.rem` now contains the supported starter subset: `add`, `sub`,
  `mul`, `div`, `sum`, `product`, `scale`, and `dot`. These are loaded
  automatically by the compiler facade and CPU evaluator; the REPL initializes
  and resets its session definitions with the same prelude definitions.
- The `remorac` console script defaults to `--target cpu`, printing the
  compiled CPU result. It also supports `--target interp`, `--emit-ast`,
  `--emit-typed-ast`, `--emit-hir`, `--emit-mlir`, `--emit-ptx`, plus
  `--target mlir` and `--target ptx` aliases for artifact inspection.

Deferred CPU/runtime work:

- Switch compiled CPU execution to explicit output descriptors matching
  `docs/ABI.md`; the current path reads MLIR-returned memref structs.
- Replace the subprocess `llc`/`gcc` shared-library path with in-process
  execution if a stable MLIR/LLVM execution binding is added.
- Add CUDA driver module-load and launch tests only after the Remora ABI kernel
  boundary is explicit.

## REPL Decisions

- `remora.repl` implements the first interactive shell as a thin CPU-only layer
  over the current parser, typechecker, compiler facade, and interim evaluator.
- Session state is stored as accumulated top-level value-definition source
  strings. Each expression is evaluated by building a full temporary source
  program from those definitions plus the current expression.
- Top-level function definitions persist in the REPL and can be used by later
  direct calls or as unary `map` callables. They are still specialized at use
  sites, not generalized as first-class runtime function values.
- `:type` typechecks the expression in the current session context without
  evaluating it.
- `:mlir` lowers the expression in the current session context through the
  compiler facade and prints validated MLIR when the current lowering subset
  supports it.
- `:prelude` prints the starter prelude definitions currently injected into new
  sessions. `:defs` prints only user-added definitions after the prelude.
- `:load` loads top-level value/function definitions from a file and evaluates
  the file body if present. This is intentionally simple and line-oriented for
  current one-line `def` examples.
- `:target` reports `cpu`; non-CPU targets are rejected until the descriptor ABI
  and runtime launch path are ready.

Deferred REPL work:

- Support annotated/generalized top-level function definitions after a real
  function type story exists.
- Replace source-string session accumulation with typed environment/HIR state
  if definitions become multi-line or more complex.
- Add GPU target support only after the final Remora ABI execution path exists.

## Test Coverage So Far

Current tests cover:

- Package/dependency imports.
- ctypes ABI field order and struct sizes.
- Descriptor construction from contiguous numpy arrays.
- Descriptor construction from sliced/transposed numpy views.
- Parser coverage for literals, arrays, lambdas, lets, `map`, `fold`, `iota`,
  application, definitions, nesting, infix precedence, conditionals, REPL input,
  and malformed syntax.
- Typechecker coverage for scalar literals, rank-1/2/3 array literals, `iota`,
  scalar maps, row-reduction maps, vector folds, numeric casts, rank-4
  rejection, the M2 milestone expression, direct top-level function calls,
  top-level functions as map callables, static `shape`/`rank`, array indexing,
  and recursive-function deferral.
- HIR coverage for `iota`, array literals, casts, scalar maps, vector-cell map
  shape metadata, folds, operator sections, top-level value definitions, and the
  M2 milestone expression. Static `shape`/`rank` and array indexing HIR lowering
  are also covered.
- Regression coverage for division callable operand validation, right operator
  sections, negative-stride numpy views, the current array-literal/index parse
  behavior, and definition-only HIR rejection.
- Defunctionalization coverage for inline lambda lifting, primitive callables,
  named static function references, operator sections, and rejection of captured
  lambdas.
- Initial MLIR lowering coverage for type spelling/parsing, `iota` textual MLIR
  parse validation, primitive scalar section maps over direct `iota`, and
  simple lifted scalar lambda maps over direct `iota`, plus explicit deferral of
  unsupported lowering cases.
- Scalar MLIR lowering coverage for standalone literals, arithmetic, numeric
  comparisons, boolean operations, division, and explicit `int` to `float`
  casts.
- Rank-0 scalar map coverage for primitive operator sections, lifted lambdas,
  and let-bound scalar inputs.
- Comparison-valued scalar maps over `iota` are covered to exercise bool tensor
  results from lifted lambdas.
- Nested scalar map coverage for map chains over `iota` and static array
  literals.
- Binary scalar map coverage for rank-0 through rank-3 inputs, including
  primitive binary maps, lifted binary lambda maps, and prelude `dot` lowering
  as binary map plus fold.
- Fold lowering coverage for direct `iota` and the Phase 5 milestone-shaped
  `fold (+) 0.0 (map (* 2.0) (iota 10))` program.
- Rank-2/rank-3 array-cell fold coverage over static literals.
- Rank-2/rank-3 rank-1-cell map coverage for lifted row/cell reduction
  lambdas.
- Let/top-level value lowering coverage for iota aliases used by maps and folds.
- Scalar let, scalar HIR function emission, and scalar HIR call coverage.
- Static tensor literal coverage for rank-1 through rank-3 and scalar
  elementwise map coverage over rank-2/rank-3 literals.
- Golden MLIR fixture coverage for the current parse-validated lowering output
  of `iota`, scalar map over `iota`, scalar map over a rank-2 literal, and
  scalar fold over a mapped `iota`.
- MLIR lowering coverage for static top-level function direct calls and
  top-level functions used as unary `map` callables. Static `shape`/`rank` and
  full-rank literal indexing to `tensor.extract` are covered.
- Pipeline/codegen coverage for toolchain detection, validation-pipeline
  pass-manager execution, direct `run_pipeline`, external verifier execution
  when available, unavailable-pass diagnostics, standalone CPU lowering to LLVM
  dialect/LLVM IR, checked-in pipeline artifact consistency, and CUDA PTX
  inspection generation through `iree-compile` when available. PTX tests cover
  both a simple map and the Phase 6 milestone expression
  `fold (+) 0.0 (map (* 2.0) (iota 1000))`.
- Fusion/performance-smoke coverage for map-chain fusion, dot fusion,
  map-then-fold materialization status, fused operation counts for vector scale,
  map-chain, vector sum, and dot, plus CPU pipeline compile-time thresholds.
- CPU runtime and CLI coverage for scalar evaluation, direct top-level function
  calls, top-level functions as map callables, `iota`/`map`/`fold`,
  row-reduction maps, static `shape`/`rank`, array indexing, prelude `sum`,
  `product`, `scale`, and `dot`, every checked-in example file, compiler facade
  MLIR/PTX helpers, `remorac` CPU output over every checked-in example, CLI emit
  flags, MLIR/PTX target aliases, MLIR/PTX output for top-level function maps,
  missing files, invalid sources, and recursive function diagnostics.
- Display coverage for int/float/bool scalars, vectors, matrices, rank-3 arrays,
  and CLI boolean output.
- REPL coverage for expression evaluation, persistent value/function
  definitions, definitions referencing earlier definitions, top-level functions
  used in direct calls and maps, recursive-function diagnostics, `:type`,
  `:mlir`, `:prelude`, `:defs`, `:load`, `:reset`, prelude availability across
  reset, target diagnostics, error recovery, `:quit`, and the
  `remora --target cpu` entry point.
- Acceptance coverage under `tests/acceptance/` for CPU-facing pass/fail cases:
  scalar arithmetic, top-level function calls, top-level functions used in
  maps, row reductions, rank-3 maps, static `shape`, indexing, prelude `sum`,
  dot product, recursive-function diagnostics, and rank-4 rejection. Deferred examples are
  checked into `tests/acceptance/deferred/` but intentionally excluded from the
  manifest.

The latest full local test command was:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest
```

with all tests passing.
