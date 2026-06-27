"""Desugar ``letrec`` into top-level (mutually) recursive functions.

``letrec`` introduces local recursive function bindings.  Rather than teach the
type checker, elaborator, HIR and every backend about a new binding form, we
lambda-lift each ``letrec`` group to ordinary *untyped* top-level ``FuncDef``s
before type checking.  The existing top-level recursion support (provisional
type inference in the type checker, and the tail-recursion SCC lowering on the
interpreter / CPU / GPU backends) then handles them with no further changes.

Lifting is capture-aware: free variables of the group are threaded as leading
parameters of every lifted function and forwarded at every call site.  Because
the captured parameters keep their original names, references to them inside
the lifted bodies resolve correctly and no substitution of captured variables
is required.

The pass runs inside :func:`remora.lisp_reader.parse_lisp`, so every downstream
consumer (interpreter, CPU compile, GPU compile) sees the already-lifted
program.
"""

from __future__ import annotations

import typing
from dataclasses import fields, replace

from remora.ast_nodes import (
    AppExpr,
    Expr,
    FuncDef,
    LambdaExpr,
    LetExpr,
    LetRecExpr,
    Program,
    UnboxExpr,
    ValDef,
    VarExpr,
)
from remora.errors import RemoraError
from remora.operators import ALL_PRIMITIVE_OPS

_EXPR_CLASSES: tuple[type, ...] = tuple(
    t for t in typing.get_args(Expr) if isinstance(t, type)
)


def _iter_subexprs(expr: object) -> list[Expr]:
    """All immediate sub-expressions of ``expr`` (no binding awareness)."""
    out: list[Expr] = []
    for f in fields(expr):
        value = getattr(expr, f.name)
        if isinstance(value, _EXPR_CLASSES):
            out.append(value)
        elif isinstance(value, (list, tuple)):
            out.extend(item for item in value if isinstance(item, _EXPR_CLASSES))
    return out


def _map_subexprs(expr, fn):
    """Rebuild ``expr`` with ``fn`` applied to every immediate sub-expression."""
    changes: dict[str, object] = {}
    for f in fields(expr):
        value = getattr(expr, f.name)
        if isinstance(value, _EXPR_CLASSES):
            changes[f.name] = fn(value)
        elif isinstance(value, list):
            changes[f.name] = [
                fn(item) if isinstance(item, _EXPR_CLASSES) else item for item in value
            ]
        elif isinstance(value, tuple):
            changes[f.name] = tuple(
                fn(item) if isinstance(item, _EXPR_CLASSES) else item for item in value
            )
    return replace(expr, **changes) if changes else expr


def _free_vars(expr: Expr) -> set[str]:
    """Free value-variable names of ``expr`` (binding-aware)."""
    if isinstance(expr, VarExpr):
        return {expr.name}
    if isinstance(expr, LambdaExpr):
        return _free_vars(expr.body) - set(expr.params)
    if isinstance(expr, LetExpr):
        return _free_vars(expr.value) | (_free_vars(expr.body) - {expr.name})
    if isinstance(expr, LetRecExpr):
        bound = {name for name, _ in expr.bindings}
        free: set[str] = set()
        for _, lam in expr.bindings:
            free |= _free_vars(lam.body) - set(lam.params)
        free |= _free_vars(expr.body)
        return free - bound
    if isinstance(expr, UnboxExpr):
        return _free_vars(expr.box_expr) | (
            _free_vars(expr.body) - {expr.value_name} - set(expr.hidden_names)
        )
    free = set()
    for sub in _iter_subexprs(expr):
        free |= _free_vars(sub)
    return free


class _Lifter:
    def __init__(self, global_names: typing.Iterable[str]) -> None:
        self._global: set[str] = set(global_names) | set(ALL_PRIMITIVE_OPS)
        self.lifted: list[FuncDef] = []
        self._counter = 0

    def transform(self, expr: Expr | None) -> Expr | None:
        if expr is None:
            return None
        if isinstance(expr, LetRecExpr):
            return self._lift(expr)
        return _map_subexprs(expr, self.transform)

    def _lift(self, letrec: LetRecExpr) -> Expr:
        names = [name for name, _ in letrec.bindings]
        name_set = set(names)

        for name, lam in letrec.bindings:
            if not isinstance(lam, LambdaExpr):
                raise RemoraError(
                    f"letrec binding '{name}' must be a (lambda ...); "
                    "letrec binds recursive functions"
                )

        # 1. Lift any nested letrecs inside the binding bodies and the body
        #    first, so what remains references only this group's names.
        lams: list[tuple[str, LambdaExpr]] = []
        for name, lam in letrec.bindings:
            lams.append((name, replace(lam, body=self.transform(lam.body))))
        body = self.transform(letrec.body)

        # 2. Compute the free variables this group must capture.  Only the
        #    *bound lambda bodies* are lifted to top level; the letrec body
        #    stays in its original scope, so its free variables resolve there
        #    and must not be threaded as captured parameters.
        free: set[str] = set()
        for _, lam in lams:
            free |= _free_vars(lam.body) - set(lam.params)
        free -= name_set
        free -= self._global
        captured = sorted(free)

        # 3. Fresh, collision-free top-level names for each binding.
        gid = self._counter
        self._counter += 1
        rename = {name: f"__letrec{gid}_{name}" for name in names}

        # 4. A captured name that also names a parameter cannot be threaded
        #    without alpha-renaming; reject loudly rather than miscompile.
        for name, lam in lams:
            clash = set(captured) & set(lam.params)
            if clash:
                raise RemoraError(
                    f"letrec function '{name}' has parameter(s) "
                    f"{sorted(clash)} that shadow captured variable(s); "
                    "rename the parameter(s) to use letrec here"
                )

        # 5. Emit one top-level FuncDef per binding.
        for name, lam in lams:
            lifted_body = self._rewrite_calls(
                lam.body, rename, captured, set(lam.params)
            )
            self.lifted.append(
                FuncDef(
                    name=rename[name],
                    params=list(captured) + list(lam.params),
                    body=lifted_body,
                    loc=lam.loc,
                )
            )

        # 6. Replace the letrec with its (call-rewritten) body.
        return self._rewrite_calls(body, rename, captured, set())

    def _rewrite_calls(
        self,
        expr: Expr,
        rename: dict[str, str],
        captured: list[str],
        shadowed: set[str],
    ) -> Expr:
        if isinstance(expr, VarExpr):
            if expr.name in rename and expr.name not in shadowed:
                raise RemoraError(
                    f"letrec-bound function '{expr.name}' is used as a value; "
                    "only direct calls of letrec functions are supported"
                )
            return expr
        if (
            isinstance(expr, AppExpr)
            and isinstance(expr.func, VarExpr)
            and expr.func.name in rename
            and expr.func.name not in shadowed
        ):
            name = expr.func.name
            new_args = [
                self._rewrite_calls(arg, rename, captured, shadowed)
                for arg in expr.args
            ]
            prefix = [VarExpr(cap, expr.loc) for cap in captured]
            return AppExpr(
                VarExpr(rename[name], expr.func.loc), prefix + new_args, expr.loc
            )
        if isinstance(expr, LambdaExpr):
            inner = shadowed | set(expr.params)
            return replace(
                expr, body=self._rewrite_calls(expr.body, rename, captured, inner)
            )
        if isinstance(expr, LetExpr):
            new_value = self._rewrite_calls(expr.value, rename, captured, shadowed)
            inner = shadowed | {expr.name}
            return replace(
                expr,
                value=new_value,
                body=self._rewrite_calls(expr.body, rename, captured, inner),
            )
        if isinstance(expr, UnboxExpr):
            new_box = self._rewrite_calls(expr.box_expr, rename, captured, shadowed)
            inner = shadowed | {expr.value_name} | set(expr.hidden_names)
            return replace(
                expr,
                box_expr=new_box,
                body=self._rewrite_calls(expr.body, rename, captured, inner),
            )
        if isinstance(expr, LetRecExpr):
            inner = shadowed | {name for name, _ in expr.bindings}
            return _map_subexprs(
                expr, lambda e: self._rewrite_calls(e, rename, captured, inner)
            )
        return _map_subexprs(
            expr, lambda e: self._rewrite_calls(e, rename, captured, shadowed)
        )


def desugar_letrec(program: Program) -> Program:
    """Lambda-lift every ``letrec`` in ``program`` to top-level functions."""
    lifter = _Lifter(d.name for d in program.definitions)
    new_definitions: list[object] = []
    for definition in program.definitions:
        if isinstance(definition, FuncDef):
            new_definitions.append(
                replace(definition, body=lifter.transform(definition.body))
            )
        elif isinstance(definition, ValDef):
            new_definitions.append(
                replace(definition, value=lifter.transform(definition.value))
            )
        else:  # pragma: no cover - defensive
            new_definitions.append(definition)
    new_body = lifter.transform(program.body) if program.body is not None else None
    if not lifter.lifted:
        return program
    new_definitions.extend(lifter.lifted)
    return replace(program, definitions=new_definitions, body=new_body)
