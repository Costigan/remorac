"""Shared support for the current narrow GPU map slice."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from remora.hir import (
    HIRExpr,
    HIRFunction,
    HIRIf,
    HIRLambda,
    HIRLit,
    HIRMap,
    HIRPrimCallable,
    HIRPrimOp,
    HIRVar,
)
from remora.types import BOOL, FLOAT, INT, ArrayType, ScalarType, static_dim


@dataclass(frozen=True)
class F32MapOperation:
    op: str
    constant: float | None = None
    constant_side: str | None = None


@dataclass(frozen=True)
class F32InputExpr:
    index: int


@dataclass(frozen=True)
class F32ScalarParamExpr:
    index: int


@dataclass(frozen=True)
class F32ConstantExpr:
    value: float


@dataclass(frozen=True)
class F32BinaryExpr:
    op: str
    left: "F32Expr"
    right: "F32Expr"


@dataclass(frozen=True)
class F32SelectExpr:
    condition: "F32Expr"
    then_expr: "F32Expr"
    else_expr: "F32Expr"


@dataclass(frozen=True)
class F32CmpExpr:
    op: str
    left: "F32Expr"
    right: "F32Expr"


F32Expr = F32InputExpr | F32ScalarParamExpr | F32ConstantExpr | F32BinaryExpr | F32SelectExpr | F32CmpExpr


@dataclass(frozen=True)
class F32MapKernel:
    shape: tuple[int, ...]
    operation: F32MapOperation
    num_inputs: int
    expression: F32Expr | None = None
    scalar_count: int = 0
    input_kinds: tuple[str, ...] = ()
    subarray_offsets: tuple[tuple[int, int] | None, ...] | None = None


@dataclass(frozen=True)
class I32MapOperation:
    op: str
    constant: int | None = None
    constant_side: str | None = None


@dataclass(frozen=True)
class I32MapKernel:
    shape: tuple[int, ...]
    operation: I32MapOperation
    num_inputs: int


def analyze_supported_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
    element_type: ScalarType,
) -> F32MapKernel | I32MapKernel:
    type_name = "float" if element_type == FLOAT else "int" if element_type == INT else "bool"

    if len(function.params) not in (1, 2):
        raise on_unsupported(f"{context} currently supports one or two input parameters")

    input_types: list[ArrayType] = []
    for param in function.params:
        if not (
            isinstance(param.type, ArrayType)
            and param.type.element == element_type
            and 1 <= param.type.rank <= 10
        ):
            raise on_unsupported(f"{context} currently supports rank-1 through rank-10 {type_name} inputs only")
        input_types.append(param.type)

    if not (
        isinstance(function.return_type, ArrayType)
        and function.return_type.element == element_type
        and 1 <= function.return_type.rank <= 10
    ):
        raise on_unsupported(f"{context} currently supports rank-1 through rank-10 {type_name} outputs only")

    if any(input_type.shape != function.return_type.shape for input_type in input_types):
        raise on_unsupported(f"{context} input and output shapes must match")

    if not (
        isinstance(function.body, HIRMap)
        and len(function.body.arrays) == len(function.params)
        and all(isinstance(array, HIRVar) for array in function.body.arrays)
        and [array.name for array in function.body.arrays] == [param.name for param in function.params]
        and isinstance(function.body.func, HIRPrimCallable)
    ):
        raise on_unsupported(f"{context} currently supports primitive maps over function parameters only")

    callable_ = function.body.func
    if element_type == FLOAT:
        allowed_ops = {"+", "-", "*", "/"}
    elif element_type == INT:
        allowed_ops = {"+", "-", "*", "/"}
    else: # BOOL
        allowed_ops = {"&&", "||", "==", "!="}
        
    if callable_.op not in allowed_ops:
         raise on_unsupported(f"{context} does not support operator {callable_.op} for {type_name}")

    if len(function.params) == 1:
        if isinstance(callable_.left_arg, HIRLit) and callable_.left_arg.type == element_type:
            operation: F32MapOperation | I32MapOperation = F32MapOperation(callable_.op, callable_.left_arg.value, "left") if element_type == FLOAT else I32MapOperation(callable_.op, int(callable_.left_arg.value), "left")
        elif isinstance(callable_.right_arg, HIRLit) and callable_.right_arg.type == element_type:
            operation = F32MapOperation(callable_.op, callable_.right_arg.value, "right") if element_type == FLOAT else I32MapOperation(callable_.op, int(callable_.right_arg.value), "right")
        else:
            raise on_unsupported(f"{context} unary map requires a literal {type_name} section")
    elif callable_.left_arg is None and callable_.right_arg is None:
        operation = F32MapOperation(callable_.op) if element_type == FLOAT else I32MapOperation(callable_.op)
    else:
        raise on_unsupported(f"{context} binary map does not support operator sections")

    KernelClass = F32MapKernel if element_type == FLOAT else I32MapKernel
    return KernelClass(
        tuple(dim.value for dim in function.return_type.shape),
        operation,
        len(function.params),
    )


def analyze_supported_f32_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> F32MapKernel:
    try:
        simple = analyze_supported_map_function(
            function,
            on_unsupported=on_unsupported,
            context=context,
            element_type=FLOAT,
        )
        assert isinstance(simple, F32MapKernel)
        return simple
    except Exception as simple_error:
        try:
            return _analyze_fused_f32_map(function, on_unsupported, context)
        except Exception:
            raise simple_error


def _simple_f32_expression(kernel: F32MapKernel) -> F32Expr:
    operation = kernel.operation
    if kernel.num_inputs == 2:
        return F32BinaryExpr(operation.op, F32InputExpr(0), F32InputExpr(1))
    assert operation.constant is not None
    constant = F32ConstantExpr(float(operation.constant))
    value = F32InputExpr(0)
    if operation.constant_side == "left":
        return F32BinaryExpr(operation.op, constant, value)
    return F32BinaryExpr(operation.op, value, constant)


def _analyze_fused_f32_map(
    function: HIRFunction,
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> F32MapKernel:
    from remora.hir import HIRApply, HIRSubarray, HIRVar

    array_param_indices: dict[str, int] = {}
    scalar_param_indices: dict[str, int] = {}
    input_types: list[ArrayType] = []
    input_kinds: list[str] = []
    subarray_offsets: list[tuple[int, int] | None] = []

    # ── Phase 1: register subarray views as logical inputs ──
    # Each unique (param, offsets) pair becomes a separate input slot.
    subarray_map: dict[tuple[str, int, int], int] = {}

    def _collect_subarrays(expr: HIRExpr) -> None:
        if isinstance(expr, HIRSubarray) and isinstance(expr.array, HIRVar):
            base = expr.array.name
            ro = static_dim(expr.offsets[0])
            co = static_dim(expr.offsets[1])
            key = (base, ro, co)
            if key not in subarray_map:
                subarray_map[key] = len(input_types)
                input_types.append(expr.result_type)
                input_kinds.append("array")
                subarray_offsets.append((ro, co))
        # recursively collect from children
        for fld in ("arrays", "args", "condition", "then_branch", "else_branch",
                     "body", "value", "init", "array"):
            ch = getattr(expr, fld, None)
            if ch is None:
                continue
            if isinstance(ch, (list, tuple)):
                for c in ch:
                    _collect_subarrays(c)
            elif isinstance(ch, HIRExpr):
                _collect_subarrays(ch)

    _collect_subarrays(function.body)

    # ── Phase 2: register function parameters ──
    def _has_direct_ref(body, name):
        if isinstance(body, HIRVar) and body.name == name:
            return True
        if isinstance(body, HIRSubarray):
            return False
        for fld in ("arrays", "args", "condition", "then_branch", "else_branch",
                     "body", "value", "init", "array"):
            ch = getattr(body, fld, None)
            if ch is None: continue
            for c in (ch if isinstance(ch, (list, tuple)) else [ch]):
                if isinstance(c, HIRExpr) and _has_direct_ref(c, name):
                    return True
        return False

    for param in function.params:
        if isinstance(param.type, ArrayType) and param.type.element == FLOAT and 1 <= param.type.rank <= 10:
            if subarray_map and not _has_direct_ref(function.body, param.name):
                key = (param.name, 0, 0)
                if key in subarray_map:
                    array_param_indices[param.name] = subarray_map[key]
                continue
            key = (param.name, 0, 0)
            if key not in subarray_map:
                subarray_map[key] = len(input_types)
                input_types.append(param.type)
                input_kinds.append("array")
                subarray_offsets.append(None)
            array_param_indices[param.name] = subarray_map[key]
        elif param.type == FLOAT:
            scalar_param_indices[param.name] = len(scalar_param_indices)
            input_kinds.append("scalar")
        else:
            raise on_unsupported(
                f"{context} currently supports rank-1 through rank-10 float array inputs and float scalar inputs only"
            )
    if len(input_types) < 1 or len(input_types) > 10:
        raise on_unsupported(f"{context} currently supports 1–10 array inputs, got {len(input_types)}")
    if not isinstance(function.return_type, ArrayType) or function.return_type.element != FLOAT:
        raise on_unsupported(f"{context} currently supports float array outputs only")
    if any(
        input_type.shape != function.return_type.shape and (subarray_offsets[i] is None)
        for i, input_type in enumerate(input_types)
    ):
        raise on_unsupported(f"{context} input and output shapes must match")
    scalar_env = {
        name: F32ScalarParamExpr(index) for name, index in scalar_param_indices.items()
    }
    # build combined param_indices for _f32_expr_from_array — includes
    # both direct params and subarray views
    all_indices: dict[str, int] = {}
    for (base, ro, co), idx in subarray_map.items():
        all_indices[f"{base}__{ro}_{co}"] = idx
    # also map the base param name for direct references
    for param in function.params:
        if isinstance(param.type, ArrayType):
            all_indices[param.name] = array_param_indices.get(param.name, 0)
    expression = _f32_expr_from_array(
        function.body,
        all_indices,
        scalar_env,
        on_unsupported,
        context,
    )
    root = expression if isinstance(expression, (F32BinaryExpr, F32SelectExpr, F32CmpExpr)) else None
    if root is None:
        raise on_unsupported(f"{context} fused map result must be arithmetic")
    root_op = root.op if isinstance(root, F32BinaryExpr) else "select"
    return F32MapKernel(
        tuple(dim.value for dim in function.return_type.shape),
        F32MapOperation(root_op),
        len(input_types),
        expression,
        len(scalar_param_indices),
        tuple(input_kinds),
        subarray_offsets=tuple(subarray_offsets) if any(o is not None for o in subarray_offsets) else None,
    )


def _f32_expr_from_array(
    expr: HIRExpr,
    param_indices: dict[str, int],
    scalar_env: dict[str, F32Expr],
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> F32Expr:
    from remora.hir import HIRApply, HIRLit, HIRSubarray, HIRVar
    if isinstance(expr, HIRVar) and expr.name in param_indices:
        return F32InputExpr(param_indices[expr.name])
    if isinstance(expr, HIRVar) and expr.name in scalar_env:
        return scalar_env[expr.name]
    if isinstance(expr, HIRLit):
        return F32ConstantExpr(float(expr.value))
    if isinstance(expr, HIRSubarray) and isinstance(expr.array, HIRVar):
        ro = static_dim(expr.offsets[0])
        co = static_dim(expr.offsets[1])
        key = f"{expr.array.name}__{ro}_{co}"
        if key in param_indices:
            return F32InputExpr(param_indices[key])
        raise on_unsupported(f"{context} subarray of '{expr.array.name}' at ({ro},{co}) is not registered")
    if isinstance(expr, HIRIf):
        return F32SelectExpr(
            _f32_expr_from_array(expr.condition, param_indices, scalar_env, on_unsupported, context),
            _f32_expr_from_array(expr.then_branch, param_indices, scalar_env, on_unsupported, context),
            _f32_expr_from_array(expr.else_branch, param_indices, scalar_env, on_unsupported, context),
        )
    if not isinstance(expr, (HIRMap, HIRApply)):
        raise on_unsupported(f"{context} fused maps require parameter or map operands")
    array_exprs = [
        _f32_expr_from_array(array, param_indices, scalar_env, on_unsupported, context)
        for array in expr.arrays
    ]
    if isinstance(expr.func, HIRPrimCallable):
        op = expr.func.op
        if op not in {"+", "-", "*", "/"}:
            raise on_unsupported(f"{context} does not support fused operator {op}")
        if len(array_exprs) == 2:
            if expr.func.left_arg is not None or expr.func.right_arg is not None:
                raise on_unsupported(
                    f"{context} binary map does not support operator sections"
                )
            return F32BinaryExpr(op, array_exprs[0], array_exprs[1])
        if len(array_exprs) == 1:
            if isinstance(expr.func.left_arg, HIRLit):
                return F32BinaryExpr(op, F32ConstantExpr(float(expr.func.left_arg.value)), array_exprs[0])
            if isinstance(expr.func.right_arg, HIRLit):
                return F32BinaryExpr(op, array_exprs[0], F32ConstantExpr(float(expr.func.right_arg.value)))
    if isinstance(expr.func, HIRLambda) and len(expr.func.params) == len(array_exprs):
        env = dict(scalar_env)
        env.update({param.name: value for param, value in zip(expr.func.params, array_exprs)})
        return _f32_expr_from_scalar(expr.func.body, env, on_unsupported, context)
    raise on_unsupported(f"{context} fused map callable is not supported")


def _f32_expr_from_scalar(
    expr: HIRExpr,
    env: dict[str, F32Expr],
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> F32Expr:
    if isinstance(expr, HIRVar) and expr.name in env:
        return env[expr.name]
    if isinstance(expr, HIRLit) and expr.type == FLOAT:
        return F32ConstantExpr(float(expr.value))
    if isinstance(expr, HIRPrimOp) and len(expr.args) == 2:
        op = expr.op
        for suffix in ("f", "b", "i"):
            if op.endswith(suffix):
                op = op[:-1]
                break
        if op in {"+", "-", "*", "/"}:
            return F32BinaryExpr(
                op,
                _f32_expr_from_scalar(expr.args[0], env, on_unsupported, context),
                _f32_expr_from_scalar(expr.args[1], env, on_unsupported, context),
            )
        if op in {">", "<", ">=", "<=", "==", "!="}:
            return F32CmpExpr(
                op,
                _f32_expr_from_scalar(expr.args[0], env, on_unsupported, context),
                _f32_expr_from_scalar(expr.args[1], env, on_unsupported, context),
            )
        raise on_unsupported(f"{context} does not support fused scalar operator {expr.op}")
    if isinstance(expr, HIRIf):
        return F32SelectExpr(
            _f32_expr_from_scalar(expr.condition, env, on_unsupported, context),
            _f32_expr_from_scalar(expr.then_branch, env, on_unsupported, context),
            _f32_expr_from_scalar(expr.else_branch, env, on_unsupported, context),
        )
    raise on_unsupported(f"{context} fused scalar expression is not supported")


def analyze_supported_i32_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> I32MapKernel:
    return analyze_supported_map_function(
        function,
        on_unsupported=on_unsupported,
        context=context,
        element_type=INT,
    ) # type: ignore


def analyze_supported_bool_map_function(
    function: HIRFunction,
    *,
    on_unsupported: Callable[[str], Exception],
    context: str,
) -> I32MapKernel:
    return analyze_supported_map_function(
        function,
        on_unsupported=on_unsupported,
        context=context,
        element_type=BOOL,
    ) # type: ignore


def _require_scalar_array_params(
    function: HIRFunction,
    element_type: ScalarType,
    context: str,
    on_unsupported: Callable[[str], Exception],
) -> list[ArrayType]:
    input_types: list[ArrayType] = []
    type_name = "float" if element_type == FLOAT else "int" if element_type == INT else "bool"
    for param in function.params:
        if not (
            isinstance(param.type, ArrayType)
            and param.type.element == element_type
            and 1 <= param.type.rank <= 10
        ):
            raise on_unsupported(
                f"{context} currently supports rank-1 through rank-10 {type_name} inputs only"
            )
        input_types.append(param.type)
    return input_types
