# Remora Dense Core Implementation Notes

This file records implementation decisions made while building the prototype.
It is intentionally practical: normative contracts stay in `docs/ABI.md` and
the phase plan stays in `docs/MLIR_IMPLEMENTATION_PLAN.md`.

## Current Scope

The implementation is currently limited to the Phase 0 foundation through an
initial Phase 5 MLIR lowering spike:

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
- Scalar elementwise maps over static array literals now lower for rank-1
  through rank-3, using identity affine maps with one parallel iterator per
  dimension.
- Standalone scalar literals and primitive scalar expressions lower to
  parse-validated MLIR, including integer and float arithmetic, `/`, numeric
  comparisons, boolean `&&`/`||`, and explicit `int` to `float` casts.

Full MLIR lowering, execution engine, CUDA launch path, dynamic shapes, dynamic
rank, or automatic differentiation has not been implemented.

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
- Primitive numeric behavior:
  - `int op int -> int` for `+`, `-`, `*`
  - mixed `int`/`float` promotes to `float`
  - `/` returns `float`
  - comparisons return `bool`
  - `&&` and `||` require `bool`
- Numeric promotions are explicit in the typed tree with `TypedCast`.
- Lambdas and operator sections are accepted only when checked against an
  expected function type, currently through `map` or `fold`.
- `map` tries scalar cells first, then progressively larger suffix cell shapes.
  This supports scalar maps over rank-1/2/3 arrays and vector-cell maps such as
  row reductions.
- `fold` currently supports scalar accumulator folds over rank-1 arrays. Array
  cell folds are explicitly deferred.
- Top-level value definitions are supported.
- General top-level function definition inference is deferred because the
  language does not yet have annotations or monomorphization.
- Typed array literals and top-level value definitions preserve their typed
  children so later HIR lowering does not need to re-run type inference.
- Division operator functions and sections require numeric operands just like
  ordinary division expressions. Regression tests cover bool operands for both
  `map` sections and `fold (/)`.

Deferred typechecker work:

- Function annotations and top-level function type inference/checking.
- Type variables or a real bidirectional annotation story for standalone
  lambdas.
- Compile-time constant folding for shape expressions.
- `shape` and `rank` typing.
- Index expression typing.
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
- Top-level function definitions remain deferred, so `HIRProgram.functions` is
  currently empty in successful programs.
- `HIRMap` carries the frame shape and cell shape resolved by the typechecker.
  This is the key metadata the later linalg lowering will need.
- `HIRFold` carries the outer reduction dimension resolved from the typed array.
- Primitive scalar operations lower to `HIRPrimOp` with typed operation names
  like `+f`, `*i`, and comparison/bool suffixes.
- Numeric promotions lower to explicit `HIRCast` nodes.
- Lambdas lower to `HIRLambda` and are still present in HIR. They are not yet
  lambda-lifted or defunctionalized.
- Operator functions and sections lower to `HIRPrimCallable`. Sections retain
  the bound left or right operand as an HIR expression.

Deferred HIR work:

- Top-level `HIRFunction` generation from checked function definitions.
- HIR lowering for `shape`, `rank`, composition, indexing, and generalized
  conditionals.
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
- Scalar `HIRFold` lowers when its input is a direct `HIRIota` or a direct
  scalar `HIRMap` over `HIRIota`, and when the fold callable is a primitive
  operator function.
- Scalar `HIRMap` lowers over direct `HIRIota` or direct static `HIRArrayLit`
  inputs for scalar-cell maps only.
- `HIRLet` is not lowered as an MLIR SSA binding yet. The current lowerer first
  inlines simple HIR let bindings and then lowers the resulting expression. This
  supports local `let` and top-level value definitions whose values are in the
  current lowering subset.
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
- Top-level value definition programs like `def xs = iota 10` followed by
  `map (* 2.0) xs` lower through the same let-inlining path.
- Static array literals lower by flattening nested `HIRArrayLit` elements in
  row-major order and emitting scalar constants followed by `tensor.from_elements`.
- Standalone `HIRLit`, `HIRCast`, and `HIRPrimOp` expressions lower through a
  small scalar-region emitter. The same emitter is used for simple lifted
  lambda bodies inside scalar maps.
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
- Lower maps over non-direct values, generalized folds, cell maps, and
  standalone defunctionalized function calls.
- Replace let inlining with real SSA environment lowering when lowering grows
  beyond direct iota/map/fold slices.
- Run MLIR verifier/pass manager checks beyond parse validation.

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
  rejection, the M2 milestone expression, and deferred function definitions.
- HIR coverage for `iota`, array literals, casts, scalar maps, vector-cell map
  shape metadata, folds, operator sections, top-level value definitions, and the
  M2 milestone expression.
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
- Comparison-valued scalar maps over `iota` are covered to exercise bool tensor
  results from lifted lambdas.
- Fold lowering coverage for direct `iota` and the Phase 5 milestone-shaped
  `fold (+) 0.0 (map (* 2.0) (iota 10))` program.
- Let/top-level value lowering coverage for iota aliases used by maps and folds.
- Static tensor literal coverage for rank-1 through rank-3 and scalar
  elementwise map coverage over rank-2/rank-3 literals.
- Golden MLIR fixture coverage for the current parse-validated lowering output
  of `iota`, scalar map over `iota`, scalar map over a rank-2 literal, and
  scalar fold over a mapped `iota`.

The latest full local test command was:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

with all tests passing.
