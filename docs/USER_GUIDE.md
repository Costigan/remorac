# Remora Dense Core User Guide

## Installation

```bash
# Clone and set up
git clone <repo>
cd remorac
uv sync
```

## Quick Start

### REPL
```bash
uv run remora
```
```
Remora REPL [target: cpu]
> 1 + 2
3
> map (* 2) [1, 2, 3]
[2, 4, 6]
> :target interp
> fold (+) 0 (iota 5)
10
```

### Compile and run a file
```bash
uv run remorac --target cpu program.remora
```

### GPU validation (requires iree-compile)
```bash
uv run remorac --target gpu-nvidia program.remora
```

## Language Reference (Dense Core subset)

### Scalar Types
`int` (i32), `float` (f32), `bool`

### Operators
`+` `-` `*` `/` `<` `<=` `==` `!=` `&&` `||`

### Expressions
```remora
-- literals
42
3.14
true
[1, 2, 3]
[[1, 2], [3, 4]]

-- arithmetic
1 + 2 * 3
(1.0 + 2.0) / 3.0

-- let bindings
let x = 5 in x + 1

-- if (scalar or tensor condition)
if true then 1 else 2
if [true, false] then [1, 2] else [10, 20]

-- iota (integer range)
iota 5          -- [0, 1, 2, 3, 4]

-- map (element-wise)
map (* 2) [1, 2, 3]       -- unary
map (+) [1, 2] [10, 20]    -- binary
map (\x -> x + 1) [1, 2]   -- lambda

-- fold (reduction)
fold (+) 0 [1, 2, 3]              -- scalar result
fold (+) [0, 0] [[1, 2], [3, 4]]  -- array-cell

-- indexing
xs[0]
xs[1 + 2]

-- views
reverse [1, 2, 3]          -- [3, 2, 1]
transpose [[1, 2], [3, 4]] -- [[1, 3], [2, 4]]
reshape [1, 2, 3, 4] [2, 2]
ravel [[1, 2], [3, 4]]     -- [1, 2, 3, 4]
take 2 [1, 2, 3, 4]        -- [1, 2]
drop 2 [1, 2, 3, 4]        -- [3, 4]

-- shape and rank
shape [1, 2, 3]            -- [3]
rank [1, 2, 3]             -- 1
```

### Cell Maps
```remora
-- reduce each row (cell = rank-1 sub-array)
map (\row -> fold (+) 0 row) [[1, 2], [3, 4]]

-- access cell elements by index
map (\row -> row[0] + row[1]) [[1, 2], [3, 4], [5, 6]]
```

### Top-level definitions
```remora
def add x y = x + y
def double x = map (* 2) x
def sum xs = fold (+) 0.0 xs
```

### Prelude functions
`add` `sub` `mul` `div` `neg` `id` `const`
`sum` `product` `scale` `dot`
`max` `min` `abs`
`any` `all`

## CLI Options

```
remorac --target {cpu,interp,mlir,ptx,gpu-nvidia} file.remora
remorac --cpu-threads N file.remora       # multicore execution
remorac --cpu-vectorize file.remora       # vectorized lowering
remorac --emit-mlir file.remora           # print MLIR
remorac --emit-ptx file.remora            # print PTX (GPU)
```

## REPL Commands

```
:target [cpu|interp|gpu-nvidia]   show or set target
:type <expr>                      infer type
:mlir <expr>                      print MLIR
:load <file>                      load definitions
:defs                             show user definitions
:prelude                          show built-in definitions
:reset                            clear definitions
:help                             show commands
:quit                             exit
```

## Limits (Dense Core)

- Arrays: rank 0–10, static dimensions, rectangular
- Rank 11+ rejected at typecheck
- Dynamic rank/dimensions deferred
- Higher-order functions (compose, flip) deferred
- GPU: maps, reductions supported; views and indexing have diagnostic errors

## Performance

```bash
# Benchmark a program
uv run remora-bench program.remora

# Threaded execution
uv run remorac --cpu-threads 4 program.remora

# Vectorized execution
uv run remorac --cpu-vectorize program.remora
```

## Jupyter

```python
%load_ext remora.jupyter

%%remora --target cpu
map (* 2) [1, 2, 3]
```

## GPU (NVIDIA)

```bash
# Validate GPU compilation
uv run remorac --target gpu-nvidia program.remora

# Supported GPU operations:
# - Element-wise maps (f32/i32/bool, rank 1-10)
# - Scalar reductions (f32, rank 1)
# - Unsupported: views, indexing, dynamic shapes

# In REPL:
uv run remora --target gpu-nvidia
> map (* 2.0) (iota 10)
```
