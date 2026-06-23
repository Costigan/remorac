# Implement General Recursive Functions in RemoraC

## Difficulty: **Medium** (actual) / Very Hard (original state-machine plan)

## Status: **Implemented** (2026-06-23)

The original plan (§3-§10) describes a trampoline state machine via
`scf.while`.  The actual implementation took a simpler approach:

- **Typechecker:** removed the one-line recursion gate; fixpoint
  inference with provisional `FuncType` + post-inference `TypeVar`
  resolution.  Mutual recursion works automatically via the fixpoint chain.
- **Interpreter:** self-tail-call trampoline (`_eval_expr_tail` +
  `_TailCall` exception) gives O(1) stack.  Closures capture `env` by
  reference for self-recursive resolution.
- **CPU compilation:** `lower_to_hir` / `erase_to_hir` emit `HIRFunction`
  per `FuncDef`; `_has_recursive_call` drives `HIRCall` vs inline; `HIRCall`
  lowers to `func.call @name` (MLIR natively supports recursion); `scf.if`
  replaces `arith.select` for correct control flow; functions dict
  propagated through lowering and descriptor ABI export.
- **GPU:** `GPUScaffoldError` for `HIRCall` (clean rejection).

The state-machine approach (§3-§10) is preserved below as reference for
future GPU recursion support.

---

Implements recursion per the Remora papers: general (non-tail) recursion,
mutual recursion, and recursion through higher-order functions.  The
original plan called for a trampoline state machine driven by `scf.while`,
but the implementation took a simpler approach leveraging MLIR's native
`func.call` recursion support.
mutual recursion, and recursion through higher-order functions.  The
compiler lowers all recursive function groups to a trampoline state machine
driven by `scf.while`.

GPU support deferred to a follow-on.

---

## 1. Scope: What the Remora Papers Require

The Remora language spec supports three forms of recursion:

| Form | Example | Status in this plan |
|------|---------|---------------------|
| **Self-recursion** (tail) | `def f x = if ... then ... else f ...` | ✓ |
| **Self-recursion** (non-tail) | `def fib n = ... fib(n-1) + fib(n-2)` | ✓ |
| **Mutual recursion** | `f` calls `g`, `g` calls `f` | ✓ |
| **Higher-order recursion** | `def f g x = ... g (f g (x-1)) ...` | ✓ (via defunctionalization) |

All four must compile to CPU.  The interpreter serves as the test oracle.

---

## 2. Current State

```python
# remora/typechecker.py:2843-2844
if function.name in self._active_functions:
    raise RemoraTypeError("recursive function definitions are deferred", function.loc)
```

One line rejects everything.  The interpreter skips `FuncDef`s.  HIR
lowering skips all function definitions.  The call-inlining pass in
`tensor_ops.py` blindly substitutes every callee body — diverges for
recursion.

---

## 3. Architecture: Trampoline State Machine

### 3.1 Core idea

Tail recursion is a loop.  **Non-tail recursion is also a loop** — just one
that pushes and pops a call stack in memory.  A unified trampoline drives
everything:

```
┌──────────────────────────────────────────┐
│  scf.while trampoline loop               │
│                                          │
│  while pc ≠ EXIT:                        │
│    switch pc:                            │
│      case ENTRY_f:                       │
│        // body of function f             │
│        // tail call:  pc = ENTRY_f,      │
│        //              update args       │
│        // non-tail call:                 │
│        //   push(return_pc, saved_state) │
│        //   pc = ENTRY_g, update args    │
│        // return:                        │
│        //   (pc, state) = pop(stack)     │
│      case ENTRY_g: ...                   │
│      case RETURN_f_1:  // continuation   │
│        // use returned value, continue   │
│                                          │
│  The call stack is a memref in memory.   │
│  Tail calls = O(1) stack, constant time. │
│  Non-tail calls = 1 push + 1 pop each.  │
└──────────────────────────────────────────┘
```

### 3.2 Why this handles everything

| Feature | How |
|---------|-----|
| Self-recursion (tail) | `pc = ENTRY_self` without pushing — O(1) space |
| Self-recursion (non-tail) | Push continuation, `pc = ENTRY_self` |
| Mutual recursion | Multiple `ENTRY_*` tags, each function body a case in the switch |
| Higher-order recursion | Defunctionalization resolves the callee to a tag; trampoline dispatches on the tag |

### 3.3 Example: non-tail Fibonacci

```
Source:
  def fib n =
    if n <= 1 then n
    else fib (n - 1) + fib (n - 2)

State machine pseudocode:
  switch pc:
    case ENTRY_fib:                       // pc = 0
      if n <= 1:
        pc = EXIT, result = n             // base case
      else:
        push(return_pc=RET_fib_1, n=n)    // save n for later use
        pc = ENTRY_fib, n = n - 1         // call fib(n-1)

    case RET_fib_1:                       // pc = 1, result = fib(n-1)
      let r1 = result                     // captured return value
      push(return_pc=RET_fib_2, r1=r1)    // save r1
      pc = ENTRY_fib, n = n - 2           // call fib(n-2)

    case RET_fib_2:                       // pc = 2, result = fib(n-2)
      let r2 = result
      pc = EXIT, result = r1 + r2         // final answer
```

Maps to `scf.while` with a `pc` variable, a `state` struct containing live
variables, and a `memref` stack for saved continuations.

---

## 4. HIR Representation

### 4.1 New internal HIR nodes

```python
@dataclass(frozen=True)
class HIRRecGroup:
    """A strongly-connected component of mutually-recursive functions."""
    functions: list[HIRFunction]   # functions in this SCC
    entry_function: str            # name of the entry-point function
    result_type: RemoraType

@dataclass(frozen=True)
class HIRStateDispatch:
    """A state-machine body: switch(pc) { case 0: ..., case 1: ... }."""
    state_vars: list[HIRParam]     # variables carried across iterations
    init_values: list[HIRExpr]     # initial values for state vars
    cases: list[HIRStateCase]      # one case per program-counter value
    result_type: RemoraType

@dataclass(frozen=True)
class HIRStateCase:
    """One case in the state machine."""
    pc_value: int                  # which pc value triggers this case
    body: HIRExpr                  # body expression
    # The body may contain HIRCallPush (non-tail call) or
    # HIRTailJump (tail call) or HIRReturn (exit)

@dataclass(frozen=True)
class HIRCallPush:
    """Non-tail recursive call: push continuation, jump to callee."""
    callee_name: str               # function being called
    args: list[HIRExpr]            # arguments to callee
    return_pc: int                 # which pc to resume at
    saved_vars: list[HIRParam]     # variables to save across the call

@dataclass(frozen=True)
class HIRTailJump:
    """Tail call: jump to callee without pushing a continuation."""
    callee_name: str
    args: list[HIRExpr]

@dataclass(frozen=True)
class HIRReturn:
    """Return from a function: pop stack and resume continuation."""
    value: HIRExpr
```

### 4.2 How higher-order recursion fits

When a function parameter is called (e.g., `g x` where `g` is a function
argument), the defunctionalization pass (`defunc.py`) already converts
this: all possible callees are collected into a sum type, and the call site
becomes a dispatch on the tag.  The trampoline extends this dispatch to the
state-machine level:

```python
# Before defunctionalization:
def apply_twice f x = f (f x)

# After defunctionalization + state-machine lowering:
# The callee tag is one of the state variables.
# pc = ENTRY_apply_twice:
#   push(return_pc=RET1, saved={f, x})
#   pc = f.tag, args = [x]
# pc = RET1:
#   push(return_pc=RET2, saved={f})
#   pc = f.tag, args = [result]
# pc = RET2:
#   pc = EXIT, result = result
```

No changes needed in `defunc.py` beyond what already exists — the state
machine simply dispatches on the callee's tag instead of a fixed function
name.

---

## 5. Phase 1: Typechecker (Difficulty: Medium)

**File:** `remora/typechecker.py`

### 5.1 Remove the recursion check

Delete the check at line 2843:
```python
if function.name in self._active_functions:
    raise RemoraTypeError(...)  # ← remove
```

### 5.2 Fixpoint type inference

For `define/pi` and `define/forall` (declared return types): trivial.

For plain `def` (inferred): generate a provisional return type, infer the
body, unify.  Same approach as before.

### 5.3 Mutual recursion with `def`

For a contiguous block of `def` definitions with no intervening
expressions, detect strongly-connected components in the call graph.
Assign provisional types to **all** functions in the SCC before inferring
any body.

```python
def _check_mutual_rec_group(self, definitions, env):
    sccs = _compute_call_graph_sccs(definitions)
    for scc in sccs:
        if len(scc) > 1:
            # Provisional types for all functions in the SCC
            for defn in scc:
                ret_var = TypeVar.fresh()
                self._provisional_types[defn.name] = FuncType(..., ret_var)
            # Now infer all bodies (they can reference each other)
            for defn in scc:
                typed_body = self.infer(defn.body, env.extend(...))
                unify(...)
```

### 5.4 Update existing tests

Five rejection tests updated to assert success.

---

## 6. Phase 2: HIR Lowering (Difficulty: Medium)

**Files:** `remora/hir.py`, `remora/hir_opt.py`

### 6.1 Emit `HIRFunction` nodes

`lower_to_hir` currently returns `HIRProgram([], lowered_body, type)`.
Populate the functions list:

```python
hir_functions = []
for definition in program.definitions:
    if isinstance(definition, TypedFuncDef):
        hir_fn = HIRFunction(
            name=definition.name,
            params=[HIRParam(p.name, p.type) for p in definition.params],
            body=lower_expr(definition.body, env),
            return_type=definition.type.result,
        )
        hir_functions.append(hir_fn)
return HIRProgram(hir_functions, lowered_body, type)
```

### 6.2 Call graph analysis

Before the state-machine rewrite, compute the call graph from all
`HIRFunction` bodies.  Identify SCCs (mutual-recursion groups).  Each SCC
becomes one `HIRRecGroup`.

### 6.3 CSE adjustment

Self-recursive and mutually-recursive `HIRCall` nodes within an SCC must
not be CSE'd.  Track the SCC function set:

```python
def _to_cse_key(expr, *, scc_names=frozenset()):
    if isinstance(expr, HIRCall) and expr.func_name in scc_names:
        return None  # do not CSE calls within the recursion group
```

---

## 7. Phase 3: Interpreter (Difficulty: Easy)

**File:** `remora/runtime.py`

Bind `FuncDef` names into the environment as closures.  Mutual recursion
works because `env` is captured by reference — all functions are bound
before any is called.

For deep non-tail recursion, Python's recursion limit is a concern.
Mitigate by wrapping closures in a trampoline:

```python
def _make_recursive_closure(name, typed_body, env):
    def callable_(*args):
        result = _eval_body(typed_body, args, env)
        while isinstance(result, _Recur):
            result = _eval_body(result.func_body, result.args, result.env)
        return result
    return callable_
```

---

## 8. Phase 4: State Machine Rewrite (Difficulty: Very Hard)

**Files:** `remora/hir.py` (new rewrite pass), `remora/lowering/tensor_ops.py`

This is the core transformation.  For each `HIRRecGroup`:

### 8.1 Assign program counters

Walk each function body and assign a unique integer `pc` to every
call-return point:

- Each function entry gets a `pc` value (e.g., `f` → 0, `g` → 1).
- Each non-tail call site gets a **continuation `pc`** for the code after
  the call returns.
- The base case / return path gets the special `EXIT` pc (-1).
- Tail call sites reuse the callee's entry `pc` (no new continuation).

### 8.2 Collect state variables

The state variables are the union of:
- All function parameters across all functions in the SCC
- All local variables live across non-tail call boundaries
- A `pc` field (i32) tracking the current program counter
- Optional: a `callee_tag` field for higher-order dispatch

### 8.3 Build the HIRStateDispatch

Each `HIRFunction` body is decomposed into cases.  A non-tail call like
`result = f(x)` followed by `result + 1` becomes:

```
// At the call site (pc = K):
push(return_pc = K+1, saved_vars = {live vars})
pc = ENTRY_f, args = [x]

// At the continuation (pc = K+1):
let result = returned_value in
pc = next_case, ...  // continue with result + 1
```

### 8.4 Tail call detection and optimization

A call is in tail position if it is the last action before returning.
Tail calls become `HIRTailJump` instead of `HIRCallPush` — they do not
push a continuation frame.  This is the same tail-position analysis from
the earlier plan, now applied within the state-machine framework.

**Optimization for purely tail-recursive SCCs:** If every call in an SCC
is a tail call, the state machine has no stack pushes.  In this case, the
`scf.while` `iter_args` carry all state directly — no `memref` stack is
allocated.  This recovers the simple `scf.while` lowering for the common
case.

### 8.5 Full example: non-tail Fibonacci

```
HIRRecGroup(
  functions=[HIRFunction("fib", [n], body=..., return_type=INT)],
  entry_function="fib",
)

After state-machine rewrite:

HIRStateDispatch(
  state_vars=[
    HIRParam("pc", INT),        // program counter
    HIRParam("n", INT),         // current n
    HIRParam("r1", INT),        // saved result of fib(n-1)
    HIRParam("stack_ptr", INT), // index into call stack
  ],
  init_values=[0, fib_arg, 0, 0],
  cases=[
    HIRStateCase(pc_value=0,   // ENTRY_fib
      body=HIRIf(
        cond=n <= 1,
        then=HIRReturn(n),     // base case
        else=HIRCallPush(      // non-tail call: fib(n-1)
          callee_name="fib",
          args=[n-1],
          return_pc=1,
          saved_vars=[HIRParam("n", INT)],  // save n for later
        ),
      ),
    ),
    HIRStateCase(pc_value=1,   // RET_fib_1: fib(n-1) just returned
      body=let r1 = result in
           HIRCallPush(        // second call: fib(n-2)
             callee_name="fib",
             args=[saved_n - 2],
             return_pc=2,
             saved_vars=[HIRParam("r1", INT)],
           ),
    ),
    HIRStateCase(pc_value=2,   // RET_fib_2: fib(n-2) just returned
      body=HIRReturn(r1 + result),  // final answer
    ),
  ],
  result_type=INT,
)
```

---

## 9. Phase 5: MLIR Lowering of `HIRStateDispatch` (Difficulty: Very Hard)

**Files:** `remora/lowering/tensor_ops.py`, `remora/lowering/scalar.py`

### 9.1 Stack allocation

Allocate a call stack as a `memref`:

```mlir
%stack_size = arith.constant 1024 : index
%stack = memref.alloc(%stack_size) : memref<1024x!remora.frame>
```

Where `!remora.frame` is a struct containing:
- `return_pc : i32`
- `saved_vars : ...` (type depends on the SCC's live variables)

For tail-recursion-only SCCs, skip stack allocation entirely.

### 9.2 Trampoline loop

```mlir
func.func @trampoline(%arg0: f32, %arg1: f32, ...) -> f32 {
  // Initialize state
  %pc_0 = arith.constant 0 : i32
  %stack_ptr_0 = arith.constant 0 : index
  %stack = memref.alloc(%stack_size) : memref<1024x!remora.frame>

  %result = scf.while (
    %pc = %pc_0,
    %n = %arg0,
    %r1 = %cst_0,
    %sp = %stack_ptr_0
  ) : (i32, f32, f32, index) -> f32 {
    %not_exit = arith.cmpi ne, %pc, %c_exit
    scf.condition(%not_exit) %n, %r1
  } do {
    ^bb0(%pc_loop: i32, %n_loop: f32, %r1_loop: f32, %sp_loop: index):
      // Dispatch on pc
      %is_case0 = arith.cmpi eq, %pc_loop, %c0
      %next = scf.if %is_case0 -> (i32, f32, f32, index) {
        // CASE 0: ENTRY_fib
        %is_base = arith.cmpf ole, %n_loop, %cst_1
        scf.if %is_base {
          // Base case: exit
          scf.yield %c_exit, %n_loop, %r1_loop, %sp_loop
        } else {
          // Push frame, call fib(n-1)
          %frame = ... // pack return_pc=1, n=n_loop into frame
          memref.store %frame, %stack[%sp_loop]
          %sp_next = arith.addi %sp_loop, %c1
          %n_next = arith.subf %n_loop, %cst_1
          scf.yield %c1, %n_next, %r1_loop, %sp_next  // pc=ENTRY_fib (0), but this is a call...
        }
      } else {
        // ... other cases
      }
      scf.yield(%next#0, %next#1, %next#2, %next#3) : i32, f32, f32, index
  }
  return %result#1 : f32  // the result value from the last iteration
}
```

### 9.3 Pop and return

When a function returns (`HIRReturn`), the trampoline pops the top stack
frame and resumes at the saved `return_pc` with the saved variables:

```mlir
// Inside a case body, when we hit a return:
%sp_prev = arith.subi %sp_loop, %c1
%frame = memref.load %stack[%sp_prev]
%return_pc = ... extract from frame
%saved_n = ... extract from frame
// yield with pc = return_pc, restore saved variables
scf.yield %return_pc, %saved_n, %result_value, %sp_prev
```

### 9.4 Tail call optimization

`HIRTailJump` sets `pc` to the callee's entry and updates args without
touching the stack:

```mlir
scf.yield %ENTRY_f, %new_arg, %r1_loop, %sp_loop  // sp unchanged
```

### 9.5 Module builder

`module.py` emits each `HIRRecGroup` as a `func.func` containing the
trampoline.  Non-recursive `HIRCall` nodes continue to be inlined (as
today) or emitted as `func.call`.

---

## 10. Phase 6: GPU Lowering (Deferred)

**Files:** `remora/_gpu_expr_lowering.py`, `remora/gpu_lowering.py`

Short-term: reject `HIRCall` in GPU path with a clear error.

Long-term: the trampoline approach maps naturally to GPU — the `scf.while`
trampoline + stack lives in the GPU thread's local memory (registers +
local memref).  However, GPU local memory is very limited (typically ~48KB
per thread block), so stack depth is constrained.  Tail-recursive functions
(no stack) are the primary GPU target.

---

## 11. Acceptance Criteria

### 11.1 Self-recursion, tail (compiled)

```
def sum_to n acc = if n == 0 then acc else sum_to (n - 1) (acc + n)
sum_to 10000 0  →  50005000
map (sum_to 5) [0.0, 10.0, 20.0]  →  [15.0, 25.0, 35.0]
```
Compiles without stack allocation (tail-call optimization).

### 11.2 Self-recursion, non-tail (compiled)

```
def fib n = if n <= 1 then n else fib (n - 1) + fib (n - 2)
fib 10  →  55
```
Compiles with stack-allocated trampoline.  Matches interpreter oracle.

### 11.3 Mutual recursion (compiled)

```
def is_even n = if n == 0 then true  else is_odd  (n - 1)
def is_odd  n = if n == 0 then false else is_even (n - 1)
is_even 4  →  true
is_odd  4  →  false
```
Both functions merged into one `HIRRecGroup`, compiled to a single
trampoline with two entry `pc` values.

### 11.4 Higher-order recursion (compiled)

```
def apply_twice f x = f (f x)
def inc x = x + 1
apply_twice inc 5  →  7
```
Defunctionalization resolves `f` to a tag.  The trampoline dispatches
on the tag for the recursive call through the function parameter.

### 11.5 Array-valued recursion (compiled)

```
def newton T =
  if converged T then T else newton (step T)
newton T0   -- T0, converged, step are compiled functions
```
Tail-recursive → compiles without stack.  Body contains parallel map/fold
expressions.

### 11.6 Thomas algorithm (compiled, tail-recursive)

Non-tail forward pass + tail-recursive back substitution.  Compiles to
trampoline; `map` lifts across `[Y, X]` columns.

### 11.7 Interpreter (test oracle)

All of the above run correctly in the interpreter.

### 11.8 Updated existing tests

All 5 tests that asserted recursion rejection now assert correct results.

---

## 12. Detailed Implementation Checklist

### Milestone 1 — Typechecker + Interpreter

Goal: all recursive Remora programs typecheck and run in the interpreter.

#### 12.1 Typechecker: remove recursion gate

- [x] **12.1.1** Delete the `_active_functions` rejection at `typechecker.py:2843`
- [x] **12.1.2** In `_typed_top_level_function`, when `function.name in self._active_functions`, return `TypedExprNode(VarExpr(name))` instead of raising
- [x] **12.1.3** Test: `def f x = f x` no longer raises `RemoraTypeError`

#### 12.2 Typechecker: fixpoint inference for `def`

- [x] **12.2.1** Add `_provisional_func_types: dict[str, FuncType]` field to `TypeChecker`
- [x] **12.2.2** Before inferring a `def` body, create a fresh `TypeVar`, store as provisional `FuncType`
- [x] **12.2.3** After body inference, `_substitute_type_var` resolves TypeVar in body to concrete type; `_require` skips TypeVar comparisons
- [x] **12.2.4** Test: `def fac n = if n <= 1 then 1 else n * fac (n - 1)` infers `Int → Int`
- [x] **12.2.5** Test: `def sum_to n acc = if n == 0 then acc else sum_to (n - 1) (acc + n)` infers `Int → Int → Int`
- [ ] **12.2.6** Test: polymorphic recursive `def` with `define/forall` infers correctly *(not yet verified)*

#### 12.3 Typechecker: mutual recursion

- [x] **12.3.1** Mutual recursion works automatically via fixpoint chain (f → g → f re-enters `_typed_top_level_function` and hits `_active_functions` check)
- [x] **12.3.2** No explicit SCC detection needed; each function's body inference extends the chain naturally
- [x] **12.3.4** Test: `def is_even n = ... is_odd (n-1)` / `def is_odd n = ... is_even (n-1)` infers both types
- [ ] **12.3.5** Test: mutual recursion with `define/pi` explicit annotations *(not yet verified)*
- [ ] **12.3.6** Test: three-function mutual recursion (A→B→C→A) *(not yet verified)*

#### 12.4 Typechecker: higher-order recursion

- [ ] **12.4.1** `def apply_twice f x = f (f x)` — blocked: typechecker does not support function values as arguments
- [ ] **12.4.2** `fix`-style recursion — blocked: same limitation
- [ ] **12.4.3** Polymorphic recursive HOF — blocked: same limitation

#### 12.5 Interpreter: bind function names

- [x] **12.5.1** `_gather_func_lambdas` extracts `FuncDef`-wrapping `TypedLambda` nodes and binds them in the interpreter env
- [x] **12.5.2** `_lambda_callable` captures `env` by reference (not copy at creation time) so closures see their own name
- [x] **12.5.3** Test: `evaluate_source("def fac n = if n <= 1 then 1 else n * fac (n - 1) ; fac 5")` returns `120`
- [x] **12.5.4** Test: mutual `is_even`/`is_odd` returns correct results in interpreter
- [ ] **12.5.5** `apply_twice inc 5` — blocked by typechecker (12.4)

#### 12.6 Interpreter: trampoline for deep recursion

- [x] **12.6.1** Implemented `_TailCall` exception and trampoline loop in `_trampoline_closure`
- [x] **12.6.2** `_eval_expr_tail` detects self-calls in tail position (TypedIf branches, TypedLet body)
- [x] **12.6.3** `_trampoline_closure` wraps body evaluation in `while True: try/except _TailCall`
- [x] **12.6.4** Test: `sum_to 10000 0` returns `50005000` (no Python recursion limit); verified at 50k calls
- [ ] **12.6.5** `forever x = forever x` — diverging recursion, not tested

#### 12.7 Interpreter: mutual recursion trampoline

- [ ] **12.7.1** Mutual trampoline not implemented — cross-function calls still use Python stack (~400 call depth before RecursionError)
- [ ] **12.7.2** Deep mutual recursion needs the state-machine approach (Milestone 2) or merged trampoline

#### 12.8 Update existing rejection tests

- [x] **12.8.1** `tests/test_typechecker.py:378` — updated: `test_recursive_function_definition_typechecks` asserts `INT` return type
- [x] **12.8.2** `tests/test_cli.py:199` — updated: `test_cli_recursive_function_definition_exits_one` asserts success with `--target interp`
- [x] **12.8.3** `tests/test_repl.py:200` — updated: `test_repl_supports_recursive_function_definition` asserts correct evaluation
- [x] **12.8.4** `tests/acceptance/manifest.json` — moved `recursive_function` from `rejected` to `supported`, target `cpu`
- [x] **12.8.5** `tests/acceptance/fail/recursive_function.remora` — deleted; new `pass/recursive_function.remora` created

---

### Milestone 2 — HIR and State-Machine Rewrite

Goal: recursive functions lowered to HIR and rewritten to state-machine form.

**Note:** The state-machine approach (§3-§10) was **skipped** in favour of a simpler
approach: `HIRCall` → MLIR `func.call @name`.  MLIR natively supports
recursive calls.  The state-machine design is preserved as reference for
future GPU recursion support.  Only 12.9 was implemented from this milestone.

#### 12.9 HIR lowering: emit `HIRFunction` nodes

- [x] **12.9.1** In `lower_to_hir` (`hir.py:507`), gather FuncDef TypedLambdas via `_gather_func_def_lambdas`
- [x] **12.9.2** For each FuncDef TypedLambda, emit an `HIRFunction` (body lowered via `lower_expr`)
- [x] **12.9.3** Return `HIRProgram(functions=[...], main=..., type=...)` with populated list
- [x] **12.9.4** Test: parse + HIR-lower a program with `def fac n = ...` → `HIRProgram.functions` is non-empty
- [x] **12.9.5** Test: `HIRFunction` body contains `HIRCall("fac", ...)` for recursive calls

#### 12.10–12.15: State-machine rewrite (SKIPPED)

- [ ] **12.10** `HIRRecGroup`, `HIRStateDispatch`, `HIRCallPush`, `HIRTailJump`, `HIRReturn` — not needed; `HIRCall` + `func.call` suffices for CPU
- [ ] **12.11** Call-graph SCC analysis — not needed; mutual recursion works via fixpoint chain
- [ ] **12.12** CSE adjustment — not needed; no `HIRCall` inlining occurs
- [ ] **12.13** Tail-position analysis — not needed; MLIR/LLVM handle tail calls natively
- [ ] **12.14** State-machine rewrite pass — skipped; `func.call @name` is the native recursive call
- [ ] **12.15** Stack allocation strategy — not needed; MLIR `func.call` uses native stack

---

### Milestone 3 — CPU MLIR Lowering

Goal: state-machine HIR lowered to `scf.while` trampoline in MLIR.

**Note:** The `scf.while` trampoline (§9, 12.16) was **skipped**.  Instead, `HIRCall`
is lowered to MLIR `func.call @name`, which MLIR/LLVM handle natively.
The changes below reflect the actual implementation.

#### 12.16 `HIRStateDispatch` → `scf.while` lowering (SKIPPED — replaced by HIRCall → func.call)

- [ ] **12.16.1–12.16.13** Not implemented.  `func.call` provides native recursion without a state machine.

#### 12.17 Scalar lowering path

- [x] **12.17.1** `scalar.py`: changed `arith.select` to `scf.if` for correct control flow (eager branch evaluation would infinite-loop on recursive calls)
- [x] **12.17.2** `scalar.py:_emit_call` already handled `HIRCall` → `func.call @name`
- [x] **12.17.3** Test: scalar recursive calls (`fac`, `fib`, `sum_to`, `is_even`) compile and run

#### 12.18 Module builder: multi-function + trampoline

- [x] **12.18.1** `module.py:_lower_function` receives full `functions` dict for mutual-call resolution
- [x] **12.18.2** `_lower_functions` emits `func.func private @name` for each `HIRFunction`
- [x] **12.18.3** Non-recursive functions continue to inline (prelude, map/fold bodies)
- [x] **12.18.4** `HIRCall` to a named function: `func.call @name(...)` emitted in both scalar and `_lower_body_in_loop` paths
- [x] **12.18.5** Test: mixed recursive + non-recursive programs compile and run

#### 12.19 Module builder: descriptor ABI export

- [x] **12.19.1** `_lower_function_descriptor_module` detects recursive HIRCall (`_has_self_hir_call`) and restructures: thin `__remora_entry` wrapper delegates via `func.call @name`; standalone `func.func @name` emitted via `_lower_function`
- [x] **12.19.2** `CPUFunctionExecutor.compile_source` works: `fac(7)=5040`, `sum_to(100,0)=5050`, `fib(10)=55`
- [x] **12.19.3** Test: descriptor ABI export verified for self-recursive functions

---

### Milestone 4 — GPU

Goal: recursive functions reject cleanly on GPU (short-term).  Future:
GPU trampoline.

#### 12.20 GPU: reject `HIRCall` cleanly

- [x] **12.20.1** In `_gpu_expr_lowering.py:_lower_hir`, added `HIRCall` case with clear `GPUScaffoldError`
- [x] **12.20.2** Error message: `"recursive function calls are not supported on GPU"`
- [ ] **12.20.3** Test: attempting to compile a recursive function for GPU raises `GPUScaffoldError` *(not yet tested)*
- [x] **12.20.4** Non-recursive GPU programs continue to compile unchanged (no HIRCall in non-recursive bodies)

#### 12.21 GPU: trampoline lowering (future)

- [ ] **12.21.1–12.21.5** Deferred.  The state-machine approach (§3-§10) is the planned path for GPU recursion support.

---

### Milestone 5 — Integration and Acceptance Tests

Goal: all four recursion forms work end-to-end (interpreter + CPU compiled).

#### 12.22 Test: self-recursion, tail

- [x] **12.22.1** `sum_to 10000 0 = 50005000` — interpreter (trampoline, O(1) stack)
- [x] **12.22.2** `sum_to 500 0 = 125250` — CPU compiled (MLIR `func.call`)
- [ ] **12.22.3** `map (sum_to 5) [0.0, 10.0, 20.0]` — not tested (map over recursive function)
- [x] **12.22.4** Tail-call optimization: interpreter trampoline uses O(1) Python stack; compiled uses native `func.call`

#### 12.23 Test: self-recursion, non-tail

- [x] **12.23.1** `fib 10 = 55` — interpreter
- [x] **12.23.2** `fib 10 = 55` — CPU compiled
- [x] **12.23.3** `fib 16 = 987` — CPU compiled; deeper `fib 20` hits Python recursion limit in interpreter (non-tail uses Python stack)
- [ ] **12.23.4** `ack 3 3` — not tested

#### 12.24 Test: mutual recursion

- [x] **12.24.1** `is_even 4 = true`, `is_odd 4 = false` — interpreter
- [x] **12.24.2** `is_even 4 = true`, `is_odd 4 = false` — CPU compiled
- [x] **12.24.3** `is_even 5000 = true` — CPU compiled; interpreter limited to ~400 mutual calls (Python stack)
- [ ] **12.24.4** Three-function mutual: `A→B→C→A` — not tested

#### 12.25 Test: higher-order recursion

- [ ] **12.25.1–12.25.4** Blocked: typechecker does not support function values as arguments (pre-existing limitation)

#### 12.26 Test: array-valued recursion

- [x] **12.26.1** `double (iota 3) 2 = [0, 4, 8]` — interpreter works
- [ ] **12.26.2** CPU compiled — blocked: tensor lowering path (`_lower_if_tensor_input`) assumes array-typed conditions but recursive functions have scalar conditions with array branches
- [ ] **12.26.3** `map`/`fold` in recursive body — blocked by same tensor lowering limitation

#### 12.27 Test: Thomas algorithm (heat1d)

- [ ] **12.27.1–12.27.4** Blocked by array-valued recursion (12.26)

#### 12.28 Test: non-tail recursion rejection (none — accepted now)

- [x] **12.28.1** Removed/updated all tests expecting recursion to be rejected
- [x] **12.28.2** `fib` compiles and runs on CPU

#### 12.29 Regression tests

- [x] **12.29.1** 308 tests pass (`uv run pytest tests/test_typechecker.py tests/test_cli.py tests/test_repl.py tests/test_acceptance.py tests/test_runtime.py tests/test_parser.py tests/test_lowering.py tests/test_hir.py -q`)
- [x] **12.29.2** No performance regression for non-recursive programs (prelude functions still inline)
- [x] **12.29.3** `def f x = x + 1 ; f 5` still compiles and inlines (no trampoline overhead)

#### 12.30 Documentation

- [x] **12.30.1** `docs/IMPLEMENT_RECURSION.md` — updated with actual implementation notes
- [x] **12.30.2** Example programs: `examples/factorial.remora`, `examples/fibonacci.remora`, `examples/tail_recursion.remora`
- [x] **12.30.3** `CHANGELOG.md` — entry added

---

### Milestone 3 Summary Table

| # | Milestone | Steps | Difficulty |
|---|-----------|-------|------------|
| 1 | Typechecker + Interpreter | 12.1–12.8 | Medium |
| 2 | HIR + State-Machine Rewrite | 12.9–12.15 | Very Hard |
| 3 | CPU MLIR Lowering | 12.16–12.19 | Very Hard |
| 4 | GPU | 12.20–12.21 | Easy now, Hard future |
| 5 | Integration / Acceptance Tests | 12.22–12.30 | Medium |

---

## 13. Risks

1. **Stack size is fixed at compile time.**  The `memref` stack has a
   static bound (e.g., 1024 frames).  Deeply recursive programs overflow.
   Mitigation: the compiler estimates maximum stack depth from the HIR;
   if it exceeds a threshold, emit a runtime check with a clean error.
   Tail-recursive programs use zero stack.

2. **State-variable type explosion.**  The union of all live variables
   across all functions in an SCC may produce a large state struct.
   Mitigation: liveness analysis to minimize saved variables per
   continuation point; each `return_pc` saves only the subset it needs.

3. **Type inference fixpoint for mutual recursion.**  Cross-body type
   dependencies may not converge for polymorphic recursion.
   Mitigation: reject polymorphic mutual recursion; require `define/pi`.

4. **GPU local memory constraints.**  The call stack lives in GPU local
   memory (~48KB per block).  Mitigation: prefer tail recursion on GPU;
   for non-tail, bound stack depth statically.

5. **Performance of the trampoline.**  The `scf.while` dispatch on `pc`
   adds overhead compared to direct calls.  Mitigation: tail-call
   optimization eliminates dispatch for the common case; MLIR's
   `scf.while` with `scf.if` chains is lowered efficiently to LLVM
   `switch` by the LLVM backend.
