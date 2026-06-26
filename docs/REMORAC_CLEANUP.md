# Remorac CLI Cleanup Plan

Shift `remorac` from "script interpreter with a compiler backend" to a proper
compiler toolchain: feel like a combination of a C compiler and a runtime.

## Phase 1 — CLI reshape

Surface-level changes. No new compiler passes or lowering stages.

### 1.1 Unify `remorac` and `remora`

One binary: `remorac`. The REPL is accessed via `remorac --repl`.

- Merge `remora/cli.py` and `remora/repl.py` into a single entry point.
- Remove the `remora` console_script entry point from `pyproject.toml`.
- `remorac --repl` starts the interactive REPL.
- `--repl` loads any source files provided, then drops into the REPL with those
  definitions in the environment.

### 1.2 Accept multiple source files

```
remorac a.remora b.lisp c.remora
```

Source files are concatenated in the order given, with a newline separator between
each file. Later definitions overwrite earlier ones (same-named `def` shadows).
This is simple concatenation — no module system or separate compilation.

- Change `parser.add_argument("file", ...)` to `nargs="+"`.
- The path into the compiler pipeline receives the concatenated source.
- If files disagree on syntax (mix `.remora` and `.lisp`), error out explicitly.

### 1.3 Remove redundant targets

`--target mlir` and `--target ptx` are removed. They are identical to `--emit-mlir`
and `--emit-ptx` respectively. Users who want just the IR text use:

```
remorac --emit-mlir file.remora
remorac --emit-ptx file.remora
```

### 1.4 Rename `gpu-nvidia` to `cuda`

Everywhere: CLI flag, REPL target, variable names, function parameters, doc
strings. The concept is the same — compile to GPU via IREE/CUDA.

Affected files at minimum: `cli.py`, `repl.py`, `compiler.py`, `runtime.py`,
`codegen.py`, `gpu_lowering.py`, test files, `AGENTS.md`, `USER_GUIDE.md`,
`PROJECT_OVERVIEW_AND_ARCHITECTURE.md`.

### 1.5 Infer syntax from file extension

- `.remora` → ML syntax
- `.lisp` → Lisp syntax
- `--syntax` remains available as an override for cases where extension inference
  is wrong (e.g., piped stdin, unusual filenames).
- If no file is given and no `--syntax` flag, default to ML.
- When `--syntax` is explicitly provided, it overrides all file extensions.

### 1.6 `--repl` flag

```
remorac --repl lib.remora          # load lib, start REPL
remorac --repl                     # start REPL with just the prelude
remorac --target cuda --repl       # REPL targeting CUDA
```

- Loads all source files into the REPL environment (definitions are available).
- If `main` is defined in a loaded file, it is ignored for execution purposes;
  the REPL takes over as the interactive mode.
- `--compile-only --repl` is an error. REPL always requires the host toolchain
  at runtime and cannot be baked into a static `.so`.

Implementation: the existing ReplSession already accumulates definitions into
`definition_sources`. Add a method to pre-populate from the concatenated source
of all loaded files (type-check and add definitions without executing a body).

### 1.7 `--compile-only` flag

```
remorac --compile-only file.remora           # produces a.so, a.json
remorac --compile-only -o libfoo.so file.remora  # produces libfoo.so, libfoo.json
```

- Does not run the program.
- Produces a `.so` shared library in the current working directory.
- Also produces a metadata `.json` file (same basename, `.json` extension) with
  the hash of all input sources, function signatures, etc. Used for incremental
  rebuild skipping (see §1.9).

### 1.8 `-o` flag

Specify output path for the compiled `.so`.

```
remorac --compile-only -o libfoo.so file.remora
```

- The metadata file uses the same basename: `libfoo.json`.
- Defaults to `a.so` / `a.json` when `-o` is not given.
- When `--compile-only` is not set but `-o` is given, produce the named `.so`
  AND run it, then leave it on disk (same as now but the file is named instead
  of a temp path).

### 1.9 Metadata `.json` and incremental rebuild

Each compilation writes a metadata file alongside the `.so`:

```json
{
  "key": "<sha256>",
  "sources": ["<sha256>", "<sha256>"],
  "remora_version": "<commit-hash>",
  "toolchain_fingerprint": "<sha256>",
  "cpu_threads": 1,
  "cpu_vectorize": false
}
```

On the next compilation with the same output path:
- Compute a new key from the current inputs.
- If the metadata file exists and the key matches, skip the compilation pipeline
  (`.so` is already up-to-date).
- If the key differs, recompile and overwrite the metadata.

This replaces the `~/.cache/remora/native/` directory entirely.

### 1.10 Remove cache directory

- Delete `remora/cache.py` or repurpose it for the new `.json` metadata
  approach.
- Remove `~/.cache/remora/native/` references throughout the codebase.
- Existing cache directories on disk are NOT cleaned up automatically (the user
  can delete `~/.cache/remora/` manually).

### 1.11 `--cleanup` / `--rm` flag

```
remorac --cleanup file.remora       # run, then delete a.so + a.json
remorac --rm file.remora            # alias
```

After successful execution, removes the `.so` and `.json` files that were
produced. If `-o` was given, removes the specified `.so` and its `.json`.
Without `-o`, removes `a.so` / `a.json`.

When combined with `--compile-only`, `--cleanup` is an error: producing an
artifact and immediately deleting it makes no sense.

## Phase 2 — Program arguments and entry point

### 2.1 `--args` flag

```
remorac prog.remora --args 1 2.0 true
remorac prog.remora --args '[1, 2, 3]' '[4.0, 5.0]'
```

- Everything after `--args` up to the next flag or end-of-argv is collected.
- Individual tokens are joined with spaces and parsed as Remora expression
  literals in the source's syntax (ML or Lisp, inferred from extension).
- The collected expressions become arguments to the entry function.
- Shell quoting (`'...'`) handles tokens that contain spaces.

### 2.2 Main entry point resolution

1. If a function named `main` exists, it is the entry point. `--args` tokens are
   passed as arguments to `main`.
2. If exactly one function is defined (regardless of name), it is the entry
   point.
3. If the program has only a top-level body expression (no function named
   `main`, and 0 or 2+ functions), the body is executed directly and `--args`
   is an error.
4. A top-level body expression with exactly one function defined: the body
   expression is the entry (existing behavior). The function is added to the
   environment but the body is what runs.

### 2.3 Auto-wrap top-level expressions as `main`

When the user explicitly passes `--args` and the program has a top-level body
expression (no named functions), auto-wrap:

```
-- source --
x + 1
-- becomes --
def main x = x + 1
main <arg1>
```

This allows simple scripts to accept arguments without boilerplate.

### 2.4 `--compile-only` with `--args`

When `--compile-only` is combined with `--args`, the arguments are used to
infer the parameter types for the compiled function. For example:

```
remorac --compile-only prog.remora --args 5 3.0 true
```

Infers `main` has signature `(Int, Float, Bool) -> <return type>`. The `.so`
contains the compiled `main` function with those parameter types.

Without `--args`, `--compile-only` produces an `.so` with the program's main
body or `main()` with no parameters.

## Phase 3 — Future (separate plans)

### 3.1 Mixed CPU/GPU execution

When targeting CUDA, automatically partition the program: operations that can
run on GPU go there; the rest runs on CPU. Requires data transfer between
device and host. **Deferred — needs its own design document.**

### 3.2 Module system

Proper namespaces, imports, and separate compilation. **Deferred — needs its
own design document.**

## Implementation notes

### Error handling

- `--compile-only --repl`: error, "REPL cannot be compiled into a standalone .so"
- Mixed `.remora` and `.lisp` files: error, "cannot mix ML and Lisp syntax source files"
- `--args` with a body-only program and no wrapping possible: error
- `--cleanup` with `--compile-only`: error, "cannot use --cleanup with --compile-only"

### Backward compatibility

- `remora` command (REPL entry point) is removed. Users must use `remorac --repl`.
- `--target gpu-nvidia` removed. Users must use `--target cuda`.
- `--target mlir` and `--target ptx` removed.
- `--syntax` remains but is no longer required when filenames have extensions.
- `~/.cache/remora/` is no longer used; new artifacts go to CWD.

### Files to modify

| File | Change |
|------|--------|
| `remora/cli.py` | Major rewrite: multiple files, `--repl`, `--compile-only`, `-o`, `--args`, `--cleanup`, entry point resolution |
| `remora/repl.py` | Pre-populate from loaded source files; integrate with CLI |
| `remora/compiler.py` | `--compile-only` mode (produce .so, don't execute); `--args` type inference |
| `remora/runtime.py` | `--compile-only` path; remove or separate the execution step |
| `remora/cache.py` | Replace with metadata.json approach or remove |
| `remora/pipeline.py` | `gpu-nvidia` → `cuda` rename |
| `remora/codegen.py` | `gpu-nvidia` → `cuda` rename |
| `remora/gpu_lowering.py` | `gpu-nvidia` → `cuda` rename |
| `pyproject.toml` | Remove `remora` console_script |
| `AGENTS.md` | Update commands section |
| `docs/USER_GUIDE.md` | Update CLI section |
| `docs/PROJECT_OVERVIEW_AND_ARCHITECTURE.md` | Update file descriptions |
| Test files | `gpu-nvidia` → `cuda`, updated CLI invocations |

### Testing

- Acceptance tests for each new flag combination.
- Refactoring validation: all existing tests pass after rename.
- New tests for entry point resolution logic.
- New tests for `--args` parsing and type inference.
- New tests for incremental rebuild skip via metadata.json.
