# Remora Dense Core

Dense Core is the current implementation target for Remora. It is the
performance-oriented static subset that should become solid on multicore CPU
and NVIDIA GPU before broader Remora features are added.

## Scope

Dense Core supports dense rectangular arrays with:

- static rank
- static non-negative dimensions
- rank 0 through rank 10
- scalar element types `int`, `float`, and `bool`
- descriptor-compatible row-major contiguous arrays and view-capable metadata

Rank 11 and above are rejected. Dynamic rank, dynamic dimensions, ragged arrays,
boxed arrays, hidden-shape values, arrays of functions, and dynamic higher-order
dispatch are outside Dense Core.

## Surface Forms

Implemented forms:

- integer, float, and boolean literals
- array literals with consistent rectangular shape
- `let`
- `if` over scalar booleans
- primitive arithmetic, comparison, and boolean operators
- explicit numeric promotion from `int` to `float`
- `iota` with a compile-time integer dimension
- `shape` and `rank` as static metadata operations
- `reverse` over statically shaped arrays, reversing the outermost axis
- full-rank and static partial indexing
- `map` over statically known callables
- scalar `fold` over statically known accumulator callables
- array-cell `fold` over primitive accumulator callables
- top-level value definitions
- top-level function definitions when statically specialized at direct use sites
- starter prelude functions: `add`, `sub`, `mul`, `div`, `sum`, `product`,
  `scale`, and `dot`

Rejected or deferred forms must produce stable diagnostics. They should not be
accepted by the parser/typechecker and then fail later with generic backend
errors.

## Type Rules

Scalar policy:

- `int` lowers to `i32`
- `float` lowers to `f32`
- `bool` is represented as a logical boolean in the language
- `int + int`, `int - int`, and `int * int` return `int`
- mixed `int`/`float` arithmetic promotes the `int` operand to `float`
- `/` returns `float`
- comparisons return `bool`
- `&&` and `||` require `bool`

Array policy:

- Array literals must be rectangular and non-empty.
- Empty array literals are deferred until explicit type annotations exist.
- `iota n` has type `int[n]` and requires `n` to be a compile-time integer
  constant.
- `shape x` returns `int[rank(x)]`.
- `rank x` returns an `int` constant.
- Index expressions require `int` indices.
- Literal indices are checked against static extents during typechecking.
- Full-rank indexing returns a scalar.
- Partial indexing drops the indexed outer dimensions.

Callable policy:

- Lambdas, named functions, operator functions, and operator sections are
  accepted only when the concrete callable is statically known.
- The typechecker specializes top-level functions at direct call sites.
- Dynamic function values are deferred: returning closures, storing functions in
  arrays, selecting functions through runtime conditionals, and device-side
  indirect calls are not Dense Core.

## ABI and Bool Layout

Dense Core kernels use the descriptor ABI in `docs/ABI.md`.

Element type is not stored in the descriptor. Kernel metadata records the dtype
for each descriptor argument.

Public `bool` arrays use one byte per element at descriptor boundaries. CPU and
GPU code may compute predicates internally as `i1`, but any public descriptor
load/store of a boolean array must use byte-backed storage with normalized
values:

- `0` means `false`
- `1` means `true`
- nonzero external bool input values should be treated as `true` only if an
  explicit importer chooses to normalize them

This policy matches `numpy.bool_` storage and avoids exposing LLVM `i1` memory
layout as part of the Remora ABI.

## Backend Requirements

CPU:

- The compiled CPU path is the default execution target.
- The typed-AST interpreter remains a correctness oracle via `--target interp`.
- Multicore CPU execution must preserve Dense Core semantics and descriptor ABI
  compatibility.

GPU:

- NVIDIA execution must use descriptor-ABI kernels, not IREE HAL PTX.
- Unsupported GPU programs must report target diagnostics instead of silently
  falling back to CPU in user-facing commands.
- GPU bool arrays must follow the byte-backed bool policy above.

## Deferred Full-Language Features

Deferred until after Dense Core CPU/GPU performance gates:

- dynamic dimensions and dynamic rank
- boxes and hidden-shape arrays
- rank-polymorphic annotations beyond current direct specialization
- generalized first-class functions
- arrays of functions
- ragged arrays
- broad slicing/transposition surface syntax
- scans and richer collective operators
- standard library expansion beyond the starter prelude
- AMD/ROCm backend
