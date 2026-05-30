# Remora Dense Core Implementation Notes

This file records implementation decisions made while building the prototype.
It is intentionally practical: normative contracts stay in `docs/ABI.md` and
the phase plan stays in `docs/MLIR_IMPLEMENTATION_PLAN.md`.

## Current Scope

The implementation is currently limited to the Phase 0 through Phase 4
foundation:

- Python package skeleton and dependency metadata.
- Rank-0 through rank-3 external ABI descriptor structs.
- Parser and AST for the Dense Core surface subset.
- Static type representation and a small typechecker for literals, `iota`,
  `let`, `if`, primitive operators, `map`, and `fold`.
- HIR definitions and typed-AST-to-HIR lowering for the accepted typed subset.
- Defunctionalization for inline non-capturing lambdas used by `map`/`fold`.

No MLIR lowering, execution engine, CUDA launch path, dynamic shapes, dynamic
rank, or automatic differentiation has been implemented.

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
- Defunctionalization coverage for inline lambda lifting, primitive callables,
  named static function references, operator sections, and rejection of captured
  lambdas.

The latest full local test command was:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

with all tests passing.
