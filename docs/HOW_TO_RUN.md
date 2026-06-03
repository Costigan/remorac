# How To Run

This project targets Python 3.11+. Use `uv` from the repository root when
available.

## Tests

Run the full test suite:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest
```

Run focused suites:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_execution.py
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_runtime.py
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_lowering.py
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest tests/test_acceptance.py
```

Compile-check the package:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python -m compileall -q remora
```

## Examples

Run examples on the compiled CPU path:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac examples/prelude_sum.remora
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac examples/dot_product.remora
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac examples/rank4_map.remora
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac examples/rank10_map.remora
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac examples/rank10_shape.remora
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac examples/rank10_rank.remora
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac examples/rank10_indexing.remora
```

Dense Core's current static-rank target is rank 0 through rank 10. The compiled
CPU path has representative rank-4 and rank-10 coverage for scalar-cell maps,
static `shape`/`rank`, and full-rank indexing.
The direct Remora ABI PTX slice is still narrower and currently supports
rank-1 through rank-3 `float32` maps only.

Some examples intentionally exercise syntax that is not lowered to compiled MLIR
yet, such as scalar conditionals. Use the interpreter target for those:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac --target interp examples/conditional.remora
```

Run every checked-in example through the interpreter:

```bash
for f in examples/*.remora; do
  env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac --target interp "$f"
done
```

Inspect compiler artifacts:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac --emit-ast examples/prelude_sum.remora
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac --emit-typed-ast examples/prelude_sum.remora
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac --emit-hir examples/prelude_sum.remora
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac --emit-mlir examples/prelude_sum.remora
```

PTX emission is available for the current IREE-backed inspection path:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remorac --emit-ptx examples/lift_map.remora
```

## REPL

Start the compiled CPU REPL:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remora
```

Start the reference-interpreter REPL:

```bash
env UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run remora --target interp
```

Useful REPL commands:

```text
:help       Show commands
:type EXPR  Show the inferred type
:mlir EXPR  Print MLIR for supported lowering cases
:target     Show or switch between cpu and interp
:prelude    Show starter prelude definitions
:defs       Show user definitions in this session
:reset      Clear user definitions and reload the prelude
:quit       Exit
```

Example session:

```text
remora> sum (iota 10)
45.0
remora> def xs = iota 4
Defined: xs : int[4]
remora> scale 2.0 xs
[0.0, 2.0, 4.0, 6.0]
remora> :type shape [[1, 2], [3, 4]]
shape [[1, 2], [3, 4]] : int[2]
```
