"""Defunctionalization for the current Dense Core HIR subset."""

from __future__ import annotations

from dataclasses import replace

from remora.errors import RemoraError
from remora.hir import (
    HIRApply,
    HIRAppend,
    HIRArrayLit,
    HIRBox,
    HIRCall,
    HIRCallable,
    HIRCast,
    HIRDrop,
    HIRExpr,
    HIRFilter,
    HIRFold,
    HIRFoldRight,
    HIRFunction,
    HIRGrade,
    HIRIf,
    HIRIndex,
    HIRIndicesOf,
    HIRIota,
    HIRLambda,
    HIRLet,
    HIRLit,
    HIRMap,
    HIRParam,
    HIRPrimCallable,
    HIRPrimOp,
    HIRProgram,
    HIRRavel,
    HIRReduce,
    HIRReplicate,
    HIRReshape,
    HIRReverse,
    HIRRotate,
    HIRScatterAdd,
    HIRPair,
    HIRFirst,
    HIRSecond,
    HIRScan,
    HIRSlice,
    HIRSort,
    HIRSubarray,
    HIRTake,
    HIRTranspose,
    HIRUnbox,
    HIRVar,
    HIRWithShape,
)
from remora.hir_dispatch import hir_dispatch
from remora.types import ScalarType


class RemoraDefuncError(RemoraError):
    """Raised when Dense Core cannot statically resolve a function value."""


def defunctionalize(program: HIRProgram) -> HIRProgram:
    """Lift inline lambdas used by HOF sites into named HIR functions."""
    pass_ = _Defunctionalizer()
    return pass_.run(program)


def _rewrite_let(
    defunc: _Defunctionalizer,
    expr: HIRLet,
    scalar_env: dict[str, HIRExpr],
) -> HIRLet:
    value = defunc._rewrite_expr(expr.value, scalar_env)
    body_env = scalar_env
    if isinstance(expr.value_type, ScalarType):
        body_env = {**scalar_env, expr.name: value}
    return HIRLet(
        expr.name, expr.value_type, value,
        defunc._rewrite_expr(expr.body, body_env),
        expr.result_type,
    )


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

    def _rewrite_expr(
        self,
        expr: HIRExpr,
        scalar_env: dict[str, HIRExpr] | None = None,
    ) -> HIRExpr:
        scalar_env = scalar_env or {}
        return hir_dispatch(expr, {
            HIRMap: lambda e: HIRMap(
                e.frame_shape, e.cell_shape,
                self._rewrite_callable(e.func, scalar_env),
                [self._rewrite_expr(a, scalar_env) for a in e.arrays],
                e.result_type,
            ),
            HIRApply: lambda e: HIRApply(
                e.frame_shape, e.cell_shape,
                self._rewrite_callable(e.func, scalar_env),
                [self._rewrite_expr(a, scalar_env) for a in e.arrays],
                e.result_type,
            ),
            HIRFold: lambda e: HIRFold(
                e.reduction_dim,
                self._rewrite_callable(e.func, scalar_env),
                self._rewrite_expr(e.init, scalar_env),
                self._rewrite_expr(e.array, scalar_env),
                e.result_type,
            ),
            HIRReduce: lambda e: HIRReduce(
                e.reduction_dim,
                self._rewrite_callable(e.func, scalar_env),
                self._rewrite_expr(e.init, scalar_env),
                self._rewrite_expr(e.array, scalar_env),
                e.result_type,
            ),
            HIRFoldRight: lambda e: HIRFoldRight(
                e.reduction_dim,
                self._rewrite_callable(e.func, scalar_env),
                self._rewrite_expr(e.init, scalar_env),
                self._rewrite_expr(e.array, scalar_env),
                e.result_type,
            ),
            HIRScan: lambda e: HIRScan(
                e.reduction_dim,
                self._rewrite_callable(e.func, scalar_env),
                self._rewrite_expr(e.init, scalar_env),
                self._rewrite_expr(e.array, scalar_env),
                e.exclusive,
                e.right,
                e.result_type,
            ),
            HIRRotate: lambda e: HIRRotate(
                self._rewrite_expr(e.array, scalar_env),
                e.shift,
                e.result_type,
            ),
            HIRSubarray: lambda e: HIRSubarray(
                self._rewrite_expr(e.array, scalar_env),
                e.offsets,
                e.sizes,
                e.result_type,
            ),
            HIRIndicesOf: lambda e: HIRIndicesOf(
                self._rewrite_expr(e.array, scalar_env),
                e.result_type,
            ),
            HIRWithShape: lambda e: HIRWithShape(
                self._rewrite_expr(e.source, scalar_env),
                e.result_type,
            ),
            HIRBox: lambda e: HIRBox(
                self._rewrite_expr(e.value, scalar_env),
                e.result_type,
            ),
            HIRUnbox: lambda e: HIRUnbox(
                self._rewrite_expr(e.box_value, scalar_env),
                e.hidden_names,
                e.value_name,
                self._rewrite_expr(e.body, scalar_env),
                e.result_type,
            ),
            HIRAppend: lambda e: HIRAppend(
                self._rewrite_expr(e.left, scalar_env),
                self._rewrite_expr(e.right, scalar_env),
                e.result_type,
            ),
            HIRScatterAdd: lambda e: HIRScatterAdd(
                self._rewrite_expr(e.target, scalar_env),
                self._rewrite_expr(e.index, scalar_env),
                self._rewrite_expr(e.update, scalar_env),
                e.result_type,
            ),
            HIRPair: lambda e: HIRPair(
                self._rewrite_expr(e.left, scalar_env),
                self._rewrite_expr(e.right, scalar_env),
                e.result_type,
            ),
            HIRFirst: lambda e: HIRFirst(
                self._rewrite_expr(e.pair, scalar_env),
                e.result_type,
            ),
            HIRSecond: lambda e: HIRSecond(
                self._rewrite_expr(e.pair, scalar_env),
                e.result_type,
            ),
            HIRFilter: lambda e: HIRFilter(
                self._rewrite_callable(e.predicate, scalar_env),
                self._rewrite_expr(e.array, scalar_env),
                e.result_type,
            ),
            HIRReplicate: lambda e: HIRReplicate(
                self._rewrite_expr(e.counts, scalar_env),
                self._rewrite_expr(e.array, scalar_env),
                e.result_type,
            ),
            HIRSort: lambda e: HIRSort(
                self._rewrite_expr(e.array, scalar_env),
                e.result_type,
            ),
            HIRGrade: lambda e: HIRGrade(
                self._rewrite_expr(e.array, scalar_env),
                e.result_type,
            ),
            HIRLet: lambda e: _rewrite_let(self, e, scalar_env),
            HIRCall: lambda e: HIRCall(
                e.func_name,
                [self._rewrite_expr(a, scalar_env) for a in e.args],
                e.result_type,
            ),
            HIRLambda: lambda e: (_ for _ in ()).throw(
                RemoraDefuncError("dynamic higher-order functions are deferred")
            ),
            HIRPrimOp: lambda e: HIRPrimOp(
                e.op,
                [self._rewrite_expr(a, scalar_env) for a in e.args],
                e.result_type,
            ),
            HIRIf: lambda e: HIRIf(
                self._rewrite_expr(e.condition, scalar_env),
                self._rewrite_expr(e.then_branch, scalar_env),
                self._rewrite_expr(e.else_branch, scalar_env),
                e.result_type,
            ),
            HIRCast: lambda e: HIRCast(
                self._rewrite_expr(e.value, scalar_env),
                e.from_type, e.to_type, e.result_type,
            ),
            HIRIndex: lambda e: HIRIndex(
                self._rewrite_expr(e.array, scalar_env),
                [self._rewrite_expr(i, scalar_env) for i in e.indices],
                e.result_type,
            ),
            HIRSlice: lambda e: e,
            HIRTranspose: lambda e: HIRTranspose(
                self._rewrite_expr(e.array, scalar_env), e.result_type,
            ),
            HIRReshape: lambda e: HIRReshape(
                self._rewrite_expr(e.array, scalar_env), e.result_type,
            ),
            HIRRavel: lambda e: HIRRavel(
                self._rewrite_expr(e.array, scalar_env), e.result_type,
            ),
            HIRReverse: lambda e: HIRReverse(
                self._rewrite_expr(e.array, scalar_env), e.result_type,
            ),
            HIRTake: lambda e: HIRTake(
                e.count, self._rewrite_expr(e.array, scalar_env), e.result_type,
            ),
            HIRDrop: lambda e: HIRDrop(
                e.count, self._rewrite_expr(e.array, scalar_env), e.result_type,
            ),
            HIRArrayLit: lambda e: HIRArrayLit(
                [self._rewrite_expr(el, scalar_env) for el in e.elements],
                e.result_type,
            ),
            HIRVar: lambda e: scalar_env.get(e.name, e),
        }, default=lambda e: e)

    def _rewrite_callable(
        self,
        callable_: HIRCallable,
        scalar_env: dict[str, HIRExpr] | None = None,
    ) -> HIRCallable:
        scalar_env = scalar_env or {}
        if isinstance(callable_, HIRLambda):
            return self._lift_lambda(callable_, scalar_env)
        if isinstance(callable_, HIRPrimCallable):
            return HIRPrimCallable(
                callable_.op,
                callable_.params,
                callable_.result_type,
                left_arg=(
                    self._rewrite_expr(callable_.left_arg, scalar_env)
                    if callable_.left_arg is not None
                    else None
                ),
                right_arg=(
                    self._rewrite_expr(callable_.right_arg, scalar_env)
                    if callable_.right_arg is not None
                    else None
                ),
            )
        if isinstance(callable_, HIRVar):
            return callable_
        raise AssertionError(f"unknown HIR callable {type(callable_).__name__}")

    def _lift_lambda(
        self,
        lambda_: HIRLambda,
        scalar_env: dict[str, HIRExpr] | None = None,
    ) -> HIRVar:
        scalar_env = scalar_env or {}
        body = self._rewrite_expr(lambda_.body, scalar_env)
        param_names = {param.name for param in lambda_.params}
        free = _free_vars(body) - param_names
        if free:
            names = ", ".join(sorted(free))
            raise RemoraDefuncError(
                f"lambda captures outer variables ({names}); closure conversion is deferred"
            )

        name = f"__lambda_{self._counter}"
        self._counter += 1
        self._lifted.append(
            HIRFunction(name, lambda_.params, body, lambda_.result_type.result)
        )
        return HIRVar(name, lambda_.result_type)


def _free_vars(expr: HIRExpr) -> set[str]:
    """Return the set of free variable names in *expr*."""
    return hir_dispatch(expr, {
        HIRVar: lambda e: {e.name},
        HIRLet: lambda e: _free_vars(e.value) | (_free_vars(e.body) - {e.name}),
        HIRMap: lambda e: _free_vars_callable(e.func) | set().union(
            *(_free_vars(a) for a in e.arrays)
        ),
        HIRApply: lambda e: _free_vars_callable(e.func) | set().union(
            *(_free_vars(a) for a in e.arrays)
        ),
        HIRFold: lambda e: (
            _free_vars_callable(e.func)
            | _free_vars(e.init) | _free_vars(e.array)
        ),
        HIRReduce: lambda e: (
            _free_vars_callable(e.func)
            | _free_vars(e.init) | _free_vars(e.array)
        ),
        HIRFoldRight: lambda e: (
            _free_vars_callable(e.func)
            | _free_vars(e.init) | _free_vars(e.array)
        ),
        HIRScan: lambda e: (
            _free_vars_callable(e.func)
            | _free_vars(e.init) | _free_vars(e.array)
        ),
        HIRReverse: lambda e: _free_vars(e.array),
        HIRCall: lambda e: set().union(*(_free_vars(a) for a in e.args)),
        HIRLambda: lambda e: _free_vars(e.body) - {p.name for p in e.params},
        HIRPrimOp: lambda e: set().union(*(_free_vars(a) for a in e.args)),
        HIRIf: lambda e: _free_vars(e.condition) | _free_vars(e.then_branch) | _free_vars(e.else_branch),
        HIRCast: lambda e: _free_vars(e.value),
        HIRIndex: lambda e: _free_vars(e.array) | set().union(
            *(_free_vars(i) for i in e.indices)
        ),
        HIRSlice: lambda e: set(),
        HIRTranspose: lambda e: _free_vars(e.array),
        HIRReshape: lambda e: _free_vars(e.array),
        HIRRavel: lambda e: _free_vars(e.array),
        HIRTake: lambda e: _free_vars(e.array),
        HIRDrop: lambda e: _free_vars(e.array),
        HIRArrayLit: lambda e: set().union(*(_free_vars(el) for el in e.elements)),
    }, default=lambda e: set())


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
