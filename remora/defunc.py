"""Defunctionalization for the current Dense Core HIR subset."""

from __future__ import annotations

from dataclasses import replace

from remora.errors import RemoraError
from remora.hir import (
    HIRArrayLit,
    HIRCall,
    HIRCallable,
    HIRCast,
    HIRExpr,
    HIRFold,
    HIRFunction,
    HIRIota,
    HIRLambda,
    HIRLet,
    HIRLit,
    HIRMap,
    HIRParam,
    HIRPrimCallable,
    HIRPrimOp,
    HIRProgram,
    HIRVar,
)


class RemoraDefuncError(RemoraError):
    """Raised when Dense Core cannot statically resolve a function value."""


def defunctionalize(program: HIRProgram) -> HIRProgram:
    """Lift inline lambdas used by HOF sites into named HIR functions."""
    pass_ = _Defunctionalizer()
    return pass_.run(program)


class _Defunctionalizer:
    def __init__(self) -> None:
        self._counter = 0
        self._lifted: list[HIRFunction] = []

    def run(self, program: HIRProgram) -> HIRProgram:
        functions = [self._rewrite_function(function) for function in program.functions]
        main = self._rewrite_expr(program.main) if program.main is not None else None
        return HIRProgram(functions + self._lifted, main, program.return_type)

    def _rewrite_function(self, function: HIRFunction) -> HIRFunction:
        return replace(function, body=self._rewrite_expr(function.body))

    def _rewrite_expr(self, expr: HIRExpr) -> HIRExpr:
        if isinstance(expr, HIRMap):
            return HIRMap(
                expr.frame_shape,
                expr.cell_shape,
                self._rewrite_callable(expr.func),
                self._rewrite_expr(expr.array),
                expr.result_type,
            )
        if isinstance(expr, HIRFold):
            return HIRFold(
                expr.reduction_dim,
                self._rewrite_callable(expr.func),
                self._rewrite_expr(expr.init),
                self._rewrite_expr(expr.array),
                expr.result_type,
            )
        if isinstance(expr, HIRLet):
            return HIRLet(
                expr.name,
                expr.value_type,
                self._rewrite_expr(expr.value),
                self._rewrite_expr(expr.body),
                expr.result_type,
            )
        if isinstance(expr, HIRCall):
            return HIRCall(
                expr.func_name,
                [self._rewrite_expr(arg) for arg in expr.args],
                expr.result_type,
            )
        if isinstance(expr, HIRLambda):
            raise RemoraDefuncError("dynamic higher-order functions are deferred")
        if isinstance(expr, HIRPrimOp):
            return HIRPrimOp(
                expr.op,
                [self._rewrite_expr(arg) for arg in expr.args],
                expr.result_type,
            )
        if isinstance(expr, HIRCast):
            return HIRCast(
                self._rewrite_expr(expr.value),
                expr.from_type,
                expr.to_type,
                expr.result_type,
            )
        if isinstance(expr, HIRArrayLit):
            return HIRArrayLit(
                [self._rewrite_expr(element) for element in expr.elements],
                expr.result_type,
            )
        if isinstance(expr, (HIRIota, HIRVar, HIRLit)):
            return expr
        raise AssertionError(f"unknown HIR expression {type(expr).__name__}")

    def _rewrite_callable(self, callable_: HIRCallable) -> HIRCallable:
        if isinstance(callable_, HIRLambda):
            return self._lift_lambda(callable_)
        if isinstance(callable_, HIRPrimCallable):
            return HIRPrimCallable(
                callable_.op,
                callable_.params,
                callable_.result_type,
                left_arg=(
                    self._rewrite_expr(callable_.left_arg)
                    if callable_.left_arg is not None
                    else None
                ),
                right_arg=(
                    self._rewrite_expr(callable_.right_arg)
                    if callable_.right_arg is not None
                    else None
                ),
            )
        if isinstance(callable_, HIRVar):
            return callable_
        raise AssertionError(f"unknown HIR callable {type(callable_).__name__}")

    def _lift_lambda(self, lambda_: HIRLambda) -> HIRVar:
        param_names = {param.name for param in lambda_.params}
        free = _free_vars(lambda_.body) - param_names
        if free:
            names = ", ".join(sorted(free))
            raise RemoraDefuncError(
                f"lambda captures outer variables ({names}); closure conversion is deferred"
            )

        name = f"__lambda_{self._counter}"
        self._counter += 1
        body = self._rewrite_expr(lambda_.body)
        self._lifted.append(
            HIRFunction(name, lambda_.params, body, lambda_.result_type.result)
        )
        return HIRVar(name, lambda_.result_type)


def _free_vars(expr: HIRExpr) -> set[str]:
    if isinstance(expr, HIRVar):
        return {expr.name}
    if isinstance(expr, HIRLet):
        return _free_vars(expr.value) | (_free_vars(expr.body) - {expr.name})
    if isinstance(expr, HIRMap):
        return _free_vars_callable(expr.func) | _free_vars(expr.array)
    if isinstance(expr, HIRFold):
        return (
            _free_vars_callable(expr.func)
            | _free_vars(expr.init)
            | _free_vars(expr.array)
        )
    if isinstance(expr, HIRCall):
        return set().union(*(_free_vars(arg) for arg in expr.args))
    if isinstance(expr, HIRLambda):
        params = {param.name for param in expr.params}
        return _free_vars(expr.body) - params
    if isinstance(expr, HIRPrimOp):
        return set().union(*(_free_vars(arg) for arg in expr.args))
    if isinstance(expr, HIRCast):
        return _free_vars(expr.value)
    if isinstance(expr, HIRArrayLit):
        return set().union(*(_free_vars(element) for element in expr.elements))
    if isinstance(expr, (HIRIota, HIRLit)):
        return set()
    raise AssertionError(f"unknown HIR expression {type(expr).__name__}")


def _free_vars_callable(callable_: HIRCallable) -> set[str]:
    if isinstance(callable_, HIRLambda):
        params = {param.name for param in callable_.params}
        return _free_vars(callable_.body) - params
    if isinstance(callable_, HIRPrimCallable):
        free: set[str] = set()
        if callable_.left_arg is not None:
            free |= _free_vars(callable_.left_arg)
        if callable_.right_arg is not None:
            free |= _free_vars(callable_.right_arg)
        return free
    if isinstance(callable_, HIRVar):
        return {callable_.name}
    raise AssertionError(f"unknown HIR callable {type(callable_).__name__}")
