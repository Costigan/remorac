# Implement General Recursive Functions in RemoraC

## Difficulty: **Very Hard**

Implements recursion per the Remora papers: general (non-tail) recursion,
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

- [ ] **12.1.1** Delete the `_active_functions` rejection at `typechecker.py:2843`
- [ ] **12.1.2** In `_typed_top_level_function`, when `function.name in self._active_functions`, return the provisional type instead of raising
- [ ] **12.1.3** Test: `def f x = f x` no longer raises `RemoraTypeError`

#### 12.2 Typechecker: fixpoint inference for `def`

- [ ] **12.2.1** Add `_provisional_types: dict[str, FuncType]` field to `TypeChecker`
- [ ] **12.2.2** Before inferring a `def` body, create a fresh return-type variable, bind it as provisional type
- [ ] **12.2.3** After body inference, unify the body type with the provisional return type
- [ ] **12.2.4** Test: `def fac n = if n <= 1 then 1 else n * fac (n - 1)` infers `Int → Int`
- [ ] **12.2.5** Test: `def sum_to n acc = if n == 0 then acc else sum_to (n - 1) (acc + n)` infers `Int → Int → Int`
- [ ] **12.2.6** Test: polymorphic recursive `def` with `define/forall` infers correctly

#### 12.3 Typechecker: mutual recursion

- [ ] **12.3.1** Detect contiguous `def` blocks with no intervening expressions
- [ ] **12.3.2** Build call graph for the block, compute SCCs (Tarjan or Kosaraju)
- [ ] **12.3.3** For each SCC with >1 function, assign provisional types to all members before inferring any body
- [ ] **12.3.4** Test: `def is_even n = ... is_odd (n-1)` / `def is_odd n = ... is_even (n-1)` as a contiguous block infers both types
- [ ] **12.3.5** Test: mutual recursion with `define/pi` explicit annotations
- [ ] **12.3.6** Test: three-function mutual recursion (A→B→C→A)

#### 12.4 Typechecker: higher-order recursion

- [ ] **12.4.1** Verify that `def apply_twice f x = f (f x)` already typechecks (no self-reference to `apply_twice` in the body — this is non-recursive HOF use)
- [ ] **12.4.2** Test: `def fix f x = f (fix f) x` (recursive HOF) infers correctly
- [ ] **12.4.3** Test: polymorphic recursive HOF with `define/forall`

#### 12.5 Interpreter: bind function names

- [ ] **12.5.1** In `_bind_definition` (`runtime.py:1175`), replace `return` on `FuncDef` with binding a Python closure
- [ ] **12.5.2** The closure captures `env` by reference so mutual recursion resolves
- [ ] **12.5.3** Test: `evaluate_source("def fac n = if n <= 1 then 1 else n * fac (n - 1) ; fac 5")` returns `120`
- [ ] **12.5.4** Test: mutual `is_even`/`is_odd` returns correct results in interpreter
- [ ] **12.5.5** Test: `def apply_twice f x = f (f x) ; def inc x = x + 1 ; apply_twice inc 5` returns `7`

#### 12.6 Interpreter: trampoline for deep recursion

- [ ] **12.6.1** Implement `_TailCall` marker class and trampoline loop in `_make_recursive_closure`
- [ ] **12.6.2** Detect self-calls in tail position within the interpreter closure
- [ ] **12.6.3** Wrap the body evaluation in a `while isinstance(result, _TailCall)` loop
- [ ] **12.6.4** Test: `def sum_to n acc = if n == 0 then acc else sum_to (n - 1) (acc + n) ; sum_to 10000 0` returns `50005000` (no Python recursion limit)
- [ ] **12.6.5** Test: `def forever x = forever x ; forever 1` → does not overflow stack (loops forever or until CPU limit)

#### 12.7 Interpreter: mutual recursion trampoline

- [ ] **12.7.1** Extend trampoline to handle calls to other functions in the same SCC
- [ ] **12.7.2** Test: deep mutual recursion (10,000 alternating calls) without stack overflow

#### 12.8 Update existing rejection tests

- [ ] **12.8.1** `tests/test_typechecker.py:378` — `test_recursive_function_definition_is_deferred` → `test_recursive_function_typechecks`
- [ ] **12.8.2** `tests/test_cli.py:199` — `test_cli_recursive_function_definition_exits_one` → asserts exit 0
- [ ] **12.8.3** `tests/test_repl.py:200` — `test_repl_reports_deferred_recursive_function_definition` → asserts no error
- [ ] **12.8.4** `tests/acceptance/manifest.json` — move `recursive_function` from `rejected` to `supported`
- [ ] **12.8.5** `tests/acceptance/fail/recursive_function.remora` — update to a passing example or delete

---

### Milestone 2 — HIR and State-Machine Rewrite

Goal: recursive functions lowered to HIR and rewritten to state-machine form.

#### 12.9 HIR lowering: emit `HIRFunction` nodes

- [ ] **12.9.1** In `lower_to_hir` (`hir.py:507`), stop skipping `FuncDef`s
- [ ] **12.9.2** For each `TypedFuncDef`, emit an `HIRFunction` (body lowered, params preserved)
- [ ] **12.9.3** Return `HIRProgram(functions=[...], main=..., type=...)` with populated list
- [ ] **12.9.4** Test: parse + HIR-lower a program with `def fac n = ...` → `HIRProgram.functions` is non-empty
- [ ] **12.9.5** Test: `HIRFunction` body contains `HIRCall("fac", ...)` for recursive calls

#### 12.10 HIR: new internal nodes

- [ ] **12.10.1** Define `HIRRecGroup` dataclass in `hir.py`
- [ ] **12.10.2** Define `HIRStateDispatch` dataclass
- [ ] **12.10.3** Define `HIRStateCase` dataclass
- [ ] **12.10.4** Define `HIRCallPush` dataclass (non-tail call)
- [ ] **12.10.5** Define `HIRTailJump` dataclass (tail call)
- [ ] **12.10.6** Define `HIRReturn` dataclass (exit)
- [ ] **12.10.7** Register all new nodes in HIR visitor/dispatch tables
- [ ] **12.10.8** Test: all nodes construct correctly with valid types

#### 12.11 Call-graph SCC analysis

- [ ] **12.11.1** Walk `HIRFunction` bodies to collect all `HIRCall.func_name` references
- [ ] **12.11.2** Build directed call graph (function name → set of called function names)
- [ ] **12.11.3** Compute SCCs using Tarjan's algorithm
- [ ] **12.11.4** For each SCC, create an `HIRRecGroup` containing its functions
- [ ] **12.11.5** Non-recursive functions (SCC of size 1 with no self-call) are excluded from groups
- [ ] **12.11.6** Test: `[f→g, g→f]` produces one SCC of size 2
- [ ] **12.11.7** Test: `[f→f]` produces one SCC of size 1
- [ ] **12.11.8** Test: `[f→g, g→h]` (no cycle) produces three SCCs of size 1

#### 12.12 CSE adjustment

- [ ] **12.12.1** Add `scc_names` parameter to CSE pass context
- [ ] **12.12.2** In `_to_cse_key`, return `None` for `HIRCall` whose `func_name` is in `scc_names`
- [ ] **12.12.3** Test: self-recursive call not CSE'd; non-recursive call still CSE'd

#### 12.13 Tail-position analysis

- [ ] **12.13.1** Implement `_is_in_tail_position(expr, func_names, in_tail=True)` walker
- [ ] **12.13.2** Tail position: function body, let body, if branches
- [ ] **12.13.3** Non-tail: argument to any operator, map/fold body, let binding value
- [ ] **12.13.4** For `HIRCall`, return `(is_tail, callee_name)` tuple
- [ ] **12.13.5** Test: `if x then f (x-1) else 0` → `f(x-1)` is in tail position
- [ ] **12.13.6** Test: `f (x-1) + f (x-2)` → neither call is in tail position
- [ ] **12.13.7** Test: `let y = f x in g y` → `f x` not tail, `g y` is tail

#### 12.14 State-machine rewrite pass

- [ ] **12.14.1** Implement `_rewrite_scc_to_state_machine(scc: HIRRecGroup) → HIRStateDispatch`
- [ ] **12.14.2** Assign unique integer `pc` values: one per function entry, one per non-tail call continuation
- [ ] **12.14.3** Walk each function body, partitioning into cases at call boundaries
- [ ] **12.14.4** Collect live variables at each call site (union of all variables used after the call returns)
- [ ] **12.14.5** Build the unified state-variable list (all params + all live variables across all functions)
- [ ] **12.14.6** Convert non-tail calls to `HIRCallPush(callee_name, args, return_pc, saved_vars)`
- [ ] **12.14.7** Convert tail calls to `HIRTailJump(callee_name, args)`
- [ ] **12.14.8** Convert returns to `HIRReturn(value)`
- [ ] **12.14.9** Produce `HIRStateDispatch(state_vars, init_values, cases, result_type)`
- [ ] **12.14.10** Test: `def f x = if x == 0 then 0 else f (x - 1)` → single function, tail call → no `HIRCallPush`, only `HIRTailJump`
- [ ] **12.14.11** Test: `def fib n = if n <= 1 then n else fib (n-1) + fib (n-2)` → non-tail calls → `HIRCallPush` nodes with `return_pc` values
- [ ] **12.14.12** Test: mutual `is_even`/`is_odd` → two-entry SCC → `HIRCallPush` with cross-function callee names
- [ ] **12.14.13** Test: state-variable count is minimized (dead variables not saved)

#### 12.15 Stack allocation strategy

- [ ] **12.15.1** Analyze `HIRStateDispatch` for presence of `HIRCallPush` nodes
- [ ] **12.15.2** If no `HIRCallPush` (tail-recursion-only): skip stack allocation
- [ ] **12.15.3** If `HIRCallPush` present: compute max stack depth from SCC's longest call chain
- [ ] **12.15.4** Define the stack frame struct type (return_pc + saved variables for the deepest saving site)
- [ ] **12.15.5** Emit stack depth check at trampoline entry; raise error if exceeded at runtime
- [ ] **12.15.6** Test: tail-only SCC produces no stack allocation in lowering

---

### Milestone 3 — CPU MLIR Lowering

Goal: state-machine HIR lowered to `scf.while` trampoline in MLIR.

#### 12.16 `HIRStateDispatch` → `scf.while` lowering

- [ ] **12.16.1** In `tensor_ops.py`, add lowering dispatch for `HIRStateDispatch`
- [ ] **12.16.2** Emit `scf.while` with `iter_args` for all state variables + stack pointer
- [ ] **12.16.3** Emit stack `memref` allocation before the `scf.while` (if non-tail calls exist)
- [ ] **12.16.4** The `before` region: check `pc == EXIT`, emit `scf.condition(%is_exit)`
- [ ] **12.16.5** The `do` region: chain of `scf.if` for each `pc` value
- [ ] **12.16.6** Within each case body, lower the HIR expression using the existing expression compiler
- [ ] **12.16.7** `HIRCallPush` → store frame on stack, update pc/sp/args, `scf.yield`
- [ ] **12.16.8** `HIRTailJump` → update pc/args only, `scf.yield` (sp unchanged)
- [ ] **12.16.9** `HIRReturn` → pop stack, restore pc/saved_vars from frame, `scf.yield`
- [ ] **12.16.10** Test: tail-recursive `sum_to` → `scf.while` with no stack alloc, 2 cases
- [ ] **12.16.11** Test: non-tail `fib` → `scf.while` with stack, 3 cases
- [ ] **12.16.12** Test: mutual `is_even`/`is_odd` → single `scf.while`, 2 entry cases + continuations
- [ ] **12.16.13** Verify MLIR output is valid (passes `mlir-opt --verify-diagnostics`)

#### 12.17 Scalar lowering path

- [ ] **12.17.1** In `scalar.py`, add lowering dispatch for the state-machine pattern
- [ ] **12.17.2** Same `scf.while` structure, but operating on scalars (not tensors)
- [ ] **12.17.3** Test: a program whose main expression is a scalar recursive call compiles

#### 12.18 Module builder: multi-function + trampoline

- [ ] **12.18.1** `module.py` accepts `HIRProgram` with populated `functions` list
- [ ] **12.18.2** For each `HIRRecGroup`, emit one `func.func` containing the trampoline
- [ ] **12.18.3** For non-recursive functions, continue current lowering (inline or `func.call`)
- [ ] **12.18.4** `HIRCall` to a non-recursive function: emit `func.call @name(...)`
- [ ] **12.18.5** Test: program with mixed recursive + non-recursive functions compiles and runs

#### 12.19 Module builder: descriptor ABI export

- [ ] **12.19.1** The trampoline function is exported via the descriptor ABI (same as current `remora_call`)
- [ ] **12.19.2** Recursive function can be called from Python via `CPUFunctionExecutor`
- [ ] **12.19.3** Test: `CPUFunctionExecutor.compile_source("def fac n = ... ; fac 10", "fac", ...)` executes correctly

---

### Milestone 4 — GPU

Goal: recursive functions reject cleanly on GPU (short-term).  Future:
GPU trampoline.

#### 12.20 GPU: reject `HIRCall` cleanly

- [ ] **12.20.1** In `_gpu_expr_lowering.py:_lower_hir`, add `HIRCall` case with clear `GPUScaffoldError`
- [ ] **12.20.2** Error message names the unsupported function and suggests applying the call at Python level
- [ ] **12.20.3** Test: attempting to compile a recursive function for GPU raises `GPUScaffoldError` (not a crash)
- [ ] **12.20.4** Test: non-recursive GPU programs continue to compile unchanged

#### 12.21 GPU: trampoline lowering (future)

- [ ] **12.21.1** Add `GpuStateDispatch` node to GPU expression IR
- [ ] **12.21.2** Lower to `llvm.br` back-edges (same pattern as fold/scan/radix-sort)
- [ ] **12.21.3** Local memory stack for non-tail calls
- [ ] **12.21.4** Test: tail-recursive GPU kernel produces correct results
- [ ] **12.21.5** Test: `map` over tail-recursive function compiles and runs on GPU

---

### Milestone 5 — Integration and Acceptance Tests

Goal: all four recursion forms work end-to-end (interpreter + CPU compiled).

#### 12.22 Test: self-recursion, tail

- [ ] **12.22.1** `sum_to 10000 0 = 50005000` — interpreter (deep recursion, no stack overflow)
- [ ] **12.22.2** `sum_to 10000 0 = 50005000` — CPU compiled
- [ ] **12.22.3** `map (sum_to 5) [0.0, 10.0, 20.0] = [15.0, 25.0, 35.0]` — CPU compiled
- [ ] **12.22.4** Tail-call optimization verified: no stack allocation in MLIR output

#### 12.23 Test: self-recursion, non-tail

- [ ] **12.23.1** `fib 10 = 55` — interpreter
- [ ] **12.23.2** `fib 10 = 55` — CPU compiled (stack-allocated trampoline)
- [ ] **12.23.3** `fib 15 = 610` — CPU compiled (exercises deeper stack)
- [ ] **12.23.4** `def ack m n = if m == 0 then n+1 else if n == 0 then ack (m-1) 1 else ack (m-1) (ack m (n-1)) ; ack 3 3` — interpreter (classic deeply recursive)

#### 12.24 Test: mutual recursion

- [ ] **12.24.1** `is_even 4 = true`, `is_odd 4 = false` — interpreter
- [ ] **12.24.2** `is_even 4 = true`, `is_odd 4 = false` — CPU compiled (single trampoline)
- [ ] **12.24.3** `is_even 1000 = true` — CPU compiled (deep mutual calls)
- [ ] **12.24.4** Three-function mutual: `A→B→C→A` — interpreter + CPU

#### 12.25 Test: higher-order recursion

- [ ] **12.25.1** `apply_twice inc 5 = 7` — interpreter
- [ ] **12.25.2** `apply_twice inc 5 = 7` — CPU compiled
- [ ] **12.25.3** `def fix f x = f (fix f) x ; fix (\self n -> if n == 0 then 1 else n * self (n-1)) 5 = 120` — interpreter
- [ ] **12.25.4** `fix`-style recursion — CPU compiled (defunctionalization resolves the callee tag)

#### 12.26 Test: array-valued recursion

- [ ] **12.26.1** Define a pre-compiled `converged` and `step` function; `newton T0` converges — interpreter
- [ ] **12.26.2** Same `newton` compiles and converges — CPU
- [ ] **12.26.3** The compiled function's body contains `map`/`fold` expressions that lower correctly within the `scf.while`

#### 12.27 Test: Thomas algorithm (heat1d)

- [ ] **12.27.1** Thomas forward pass with non-tail recursion compiles on CPU
- [ ] **12.27.2** Thomas back-substitution with tail recursion compiles on CPU
- [ ] **12.27.3** Combined solve matches `np.linalg.solve` for a known tridiagonal system
- [ ] **12.27.4** `map`-lifted across `[Y, X]` columns: each column gets correct result

#### 12.28 Test: non-tail recursion rejection (none — accepted now)

- [ ] **12.28.1** Remove or update any test expecting non-tail recursion to be rejected
- [ ] **12.28.2** `fib` compiles and runs (previously rejected in the old plan, now supported)

#### 12.29 Regression tests

- [ ] **12.29.1** Full existing test suite passes (`uv run pytest tests/ -x -q`)
- [ ] **12.29.2** No performance regression for non-recursive programs
- [ ] **12.29.3** `def f x = x + 1 ; f 5` still compiles and inlines (no trampoline overhead)

#### 12.30 Documentation

- [ ] **12.30.1** Update `docs/COMPILER_MATURITY_EXAMPLES.md` — mark recursion as supported on CPU
- [ ] **12.30.2** Add recursive function examples to `examples/`
- [ ] **12.30.3** Update `FUTURE_WORK.md` — move recursion from undocumented gap to completed

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
