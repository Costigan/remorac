"""Module-level MLIR lowering: builder, main lowering, functions, descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from remora.hir import (
    HIRApply,
    HIRAppend,
    HIRArrayLit,
    HIRBox,
    HIRCall,
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
    HIRLet,
    HIRLit,
    HIRMap,
    HIRPrimCallable,
    HIRPrimOp,
    HIRProgram,
    HIRRavel,
    HIRReduce,
    HIRReplicate,
    HIRReshape,
    HIRReverse,
    HIRRotate,
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
from remora.types import ArrayType, FuncType, RemoraType, ScalarType, SigmaType

from remora.lowering.indexing import (
    _lower_index_module,
    _lower_index_result,
    _lower_scalar_index_expr,
)
from remora.lowering.scalar import (
    _Operand,
    _RegionEmitter,
    _cast_if_needed,
    _literal_value,
    _load_iree_ir,
    _lower_scalar_module,
    _lower_scalar_value_for_fold_init,
)
from remora.lowering.tensor_ops import (
    _flatten_array_literal,
    _identity_affine_map,
    _lower_append_module,
    _lower_array_fold_module,
    _lower_array_fold_result,
    _lower_array_literal_module,
    _lower_binary_map_module,
    _lower_binary_map_result,
    _lower_fold_callable_body,
    _lower_fold_input,
    _lower_fold_module,
    _lower_fold_result,
    _lower_grade_module,
    _lower_indices_of_module,
    _lower_iota_module,
    _lower_iota_scalar_map_module,
    _lower_iota_scalar_map_result,
    _lower_map_callable_body,
    _lower_map_callable_result,
    _lower_map_cell_module,
    _lower_map_cell_result,
    _lower_map_cell_fold_result,
    _lower_rotate_module,
    _lower_scalar_fold_module,
    _lower_scalar_fold_result,
    _lower_scalar_map_binary_module,
    _lower_scalar_map_module,
    _lower_scan_module,
    _lower_sort_module,
    _lower_filter_module,
    _lower_replicate_module,
    _lower_subarray_module,
    _lower_tensor_input,
    _lower_transpose_input,
    _lower_with_shape_module,
    _parallel_iterators,
)
from remora.lowering.types import (
    RemoraLoweringError,
    TensorEnv,
    _TensorValue,
    _expr_result_type,
    _is_scalar_type,
    _join_prefix,
    type_to_mlir,
)
from remora.lowering.view_ops import (
    _lower_transpose_module,
    _lower_transpose_result,
    _lower_view_input,
    _lower_view_module,
    _lower_view_result,
)


def _lower_via_builder(
    program: HIRProgram, functions: dict[str, HIRFunction]
) -> tuple[str, Any]:
    """Lower *program* via the MLIR builder API path (Stream E7)."""
    from remora.lowering._builder_ops import lower_program_via_builder

    return lower_program_via_builder(program)


@dataclass(frozen=True)
class LoweredModule:
    text: str
    module: Any


class _MLIRMainModuleBuilder:
    """Small textual MLIR builder for deterministic single-main modules."""

    def __init__(
        self,
        result_type: str,
        *,
        functions: dict[str, HIRFunction] | None = None,
    ) -> None:
        self.result_type = result_type
        self.functions = functions or {}
        self.blocks: list[str] = []
        self.externs: list[str] = []

    def add_extern(self, decl: str) -> None:
        self.externs.append(decl)

    def add_block(self, block: str) -> None:
        if block.strip():
            self.blocks.append(block.rstrip())

    def render(self, result_value: str) -> str:
        function_text = _lower_functions(self.functions)
        function_prefix = f"\n{function_text}\n" if function_text else ""
        extern_text = "\n".join(self.externs)
        extern_prefix = f"{extern_text}\n" if extern_text else ""
        body = "\n".join(self.blocks)
        return f"""module {{
{extern_prefix}\
{function_prefix}\
  func.func @main() -> {self.result_type} {{
{body}
    return {result_value} : {self.result_type}
  }}
}}"""


class MLIRLowering:
    """Minimal MLIR lowering context for the Phase 5 spike."""

    def __init__(self) -> None:
        self.ir = _load_iree_ir()
        self.context = self.ir.Context()
        self.context.allow_unregistered_dialects = True

    def lower_type(self, value_type: RemoraType) -> Any:
        """Lower a Remora type to an MLIR type object."""
        with self.context, self.ir.Location.unknown(self.context):
            return self.ir.Type.parse(type_to_mlir(value_type), self.context)

    def lower_program(
        self,
        program: HIRProgram,
        *,
        export_output_descriptor: bool = False,
    ) -> LoweredModule:
        functions = {
            function.name: function for function in program.functions
        }
        main = _prepare_main_expr(program.main)
        if not isinstance(
            main,
            (
                HIRLit,
                HIRCast,
                HIRPrimOp,
                HIRLet,
                HIRCall,
                HIRIf,
                HIRIndex,
                HIRSlice,
                HIRTranspose,
                HIRReshape,
                HIRRavel,
                HIRReverse,
                HIRRotate,
                HIRSubarray,
                HIRTake,
                HIRDrop,
                HIRIota,
                HIRIndicesOf,
                HIRWithShape,
                HIRArrayLit,
                HIRAppend,
                HIRBox,
                HIRMap,
                HIRApply,
                HIRFold,
                HIRReduce,
                HIRFoldRight,
                HIRScan,
                HIRUnbox,
                HIRSort,
                HIRGrade,
                HIRFilter,
                HIRReplicate,
            ),
        ):
            raise RemoraLoweringError(
                "only scalar expressions, scalar lets/calls, full-rank indexing, "
                "view operations, iota, array literals, scalar maps, scalar folds, "
                "box/unbox, sort, grade, and reverse lower to MLIR so far"
            )

        # Prefer builder API path; fall back to text-based if unsupported node types
        try:
            text, module_obj = _lower_via_builder(program, functions)
            if export_output_descriptor and program.return_type is not None:
                text = _add_output_descriptor_export(text, program.return_type)
                with self.context, self.ir.Location.unknown(self.context):
                    module_obj = self.ir.Module.parse(text)
                text = str(module_obj)
            return LoweredModule(text, module_obj)
        except Exception:
            pass

        text = _lower_main_module(main, functions)
        if export_output_descriptor:
            if program.return_type is None:
                raise RemoraLoweringError(
                    "output descriptor export requires a result type"
                )
            text = _add_output_descriptor_export(
                text, program.return_type
            )
        with self.context, self.ir.Location.unknown(self.context):
            module = self.ir.Module.parse(text)
        return LoweredModule(str(module), module)

    def lower_function_descriptor_export(
        self,
        function: HIRFunction,
        *,
        export_name: str = "remora_call",
    ) -> LoweredModule:
        """Lower one typed HIR function to descriptor-in/descriptor-out MLIR."""
        text = _lower_function_descriptor_module(function, export_name)
        with self.context, self.ir.Location.unknown(self.context):
            module = self.ir.Module.parse(text)
        return LoweredModule(str(module), module)


# ---------------------------------------------------------------------------
# Expression preparation and inlining
# ---------------------------------------------------------------------------


def _prepare_main_expr(expr: HIRExpr | None) -> HIRExpr | None:
    if _can_lower_as_scalar_expr(expr):
        return expr
    if isinstance(expr, HIRLet) and isinstance(expr.value_type, ArrayType):
        return expr
    return _inline_lets(expr) if expr is not None else None


def _can_lower_as_scalar_expr(expr: HIRExpr | None) -> bool:
    if expr is None:
        return False
    if isinstance(
        expr, (HIRLit, HIRCast, HIRPrimOp, HIRCall, HIRIf)
    ):
        return _is_scalar_type(
            expr.result_type
            if not isinstance(expr, HIRLit)
            else expr.type
        )
    if isinstance(expr, HIRLet):
        return (
            _is_scalar_type(expr.value_type)
            and _is_scalar_type(expr.result_type)
            and _can_lower_as_scalar_expr(expr.value)
            and _can_lower_as_scalar_expr(expr.body)
        )
    return False


def _inline_lets(
    expr: HIRExpr | None, env: dict[str, HIRExpr] | None = None
) -> HIRExpr | None:
    if expr is None:
        return None
    env = dict(env or {})
    if isinstance(expr, HIRLet):
        value = _inline_lets(expr.value, env)
        if value is None:
            raise RemoraLoweringError("let value cannot be empty")
        return _inline_lets(
            expr.body, {**env, expr.name: value}
        )
    if isinstance(expr, HIRVar):
        return env.get(expr.name, expr)
    if isinstance(expr, HIRBox):
        return _inline_lets(expr.value, env)
    if isinstance(expr, HIRUnbox):
        box_value = _inline_lets(expr.box_value, env)
        if box_value is None:
            raise RemoraLoweringError("unbox value cannot be empty")
        return _inline_lets(
            expr.body, {**env, expr.value_name: box_value}
        )
    if isinstance(expr, (HIRMap, HIRApply)):
        arrays = [
            _inline_lets(array, env) for array in expr.arrays
        ]
        if any(array is None for array in arrays):
            raise RemoraLoweringError("map array cannot be empty")
        return type(expr)(
            expr.frame_shape,
            expr.cell_shape,
            _inline_callable(expr.func, env),
            arrays,
            expr.result_type,
        )  # type: ignore[arg-type]
    if isinstance(expr, (HIRFold, HIRReduce)):
        init = _inline_lets(expr.init, env)
        array = _inline_lets(expr.array, env)
        if init is None or array is None:
            raise RemoraLoweringError("fold operands cannot be empty")
        return type(expr)(
            expr.reduction_dim,
            _inline_callable(expr.func, env),
            init,
            array,
            expr.result_type,
        )
    if isinstance(expr, HIRPrimOp):
        args = [
            _inline_lets(arg, env) for arg in expr.args
        ]
        if any(arg is None for arg in args):
            raise RemoraLoweringError("primitive operands cannot be empty")
        return HIRPrimOp(
            expr.op, args, expr.result_type
        )  # type: ignore[arg-type]
    if isinstance(expr, HIRIf):
        condition = _inline_lets(expr.condition, env)
        then_branch = _inline_lets(expr.then_branch, env)
        else_branch = _inline_lets(expr.else_branch, env)
        if condition is None or then_branch is None or else_branch is None:
            raise RemoraLoweringError(
                "conditional operands cannot be empty"
            )
        return HIRIf(condition, then_branch, else_branch, expr.result_type)
    if isinstance(expr, HIRIndex):
        array = _inline_lets(expr.array, env)
        indices = [
            _inline_lets(index, env) for index in expr.indices
        ]
        if array is None or any(index is None for index in indices):
            raise RemoraLoweringError("index operands cannot be empty")
        return HIRIndex(
            array, indices, expr.result_type  # type: ignore[arg-type]
        )
    if isinstance(expr, HIRCast):
        value = _inline_lets(expr.value, env)
        if value is None:
            raise RemoraLoweringError("cast value cannot be empty")
        return HIRCast(value, expr.from_type, expr.to_type, expr.result_type)
    if isinstance(expr, HIRArrayLit):
        elements = [
            _inline_lets(element, env)
            for element in expr.elements
        ]
        if any(element is None for element in elements):
            raise RemoraLoweringError("array elements cannot be empty")
        return HIRArrayLit(
            elements, expr.result_type  # type: ignore[arg-type]
        )
    if isinstance(expr, HIRCall):
        args = [
            _inline_lets(arg, env) for arg in expr.args
        ]
        if any(arg is None for arg in args):
            raise RemoraLoweringError("call arguments cannot be empty")
        return HIRCall(
            expr.func_name, args, expr.result_type  # type: ignore[arg-type]
        )
    return expr


def _inline_callable(
    callable_: object, env: dict[str, HIRExpr]
) -> object:
    if isinstance(callable_, HIRPrimCallable):
        left = (
            _inline_lets(callable_.left_arg, env)
            if callable_.left_arg is not None
            else None
        )
        right = (
            _inline_lets(callable_.right_arg, env)
            if callable_.right_arg is not None
            else None
        )
        return HIRPrimCallable(
            callable_.op,
            callable_.params,
            callable_.result_type,
            left_arg=left,
            right_arg=right,
        )
    if isinstance(callable_, HIRVar):
        inlined = env.get(callable_.name)
        return inlined if inlined is not None else callable_
    return callable_


# ---------------------------------------------------------------------------
# Main module dispatch
# ---------------------------------------------------------------------------


def _lower_main_module(
    node: HIRLit
    | HIRCast
    | HIRPrimOp
    | HIRLet
    | HIRCall
    | HIRIf
    | HIRIndex
    | HIRSlice
    | HIRTranspose
    | HIRReshape
    | HIRRavel
    | HIRReverse
    | HIRTake
    | HIRDrop
    | HIRIota
    | HIRArrayLit
    | HIRMap
    | HIRApply
    | HIRFold
    | HIRReduce
    | HIRFoldRight
    | HIRScan
    | HIRRotate
    | HIRSubarray
    | HIRIndicesOf
    | HIRWithShape
    | HIRBox
    | HIRUnbox
    | HIRFilter
    | HIRReplicate
    | HIRSort
    | HIRGrade
    | HIRAppend,
    functions: dict[str, HIRFunction],
) -> str:
    # Box/Unbox are type-erased at runtime
    if isinstance(node, HIRBox):
        return _lower_main_module(node.value, functions)
    if isinstance(node, HIRUnbox):
        return _lower_main_module(node.body, functions)
    if isinstance(node, HIRAppend):
        return _lower_append_module(node, functions)
    if isinstance(node, HIRLet) and isinstance(
        node.value_type, ArrayType
    ):
        return _lower_tensor_let_module(node, functions)
    if isinstance(node, HIRIf) and isinstance(
        _expr_result_type(node.condition), ArrayType
    ):
        return _lower_tensor_if_module(node)
    if isinstance(
        node,
        (HIRLit, HIRCast, HIRPrimOp, HIRLet, HIRCall, HIRIf),
    ):
        return _lower_scalar_module(node, functions)
    if isinstance(
        node,
        (
            HIRIndex,
            HIRSlice,
            HIRTranspose,
            HIRReshape,
            HIRRavel,
            HIRReverse,
            HIRTake,
            HIRDrop,
        ),
    ):
        return _lower_view_module(node, functions)
    if isinstance(node, HIRRotate):
        return _lower_rotate_module(node, functions)
    if isinstance(node, HIRSubarray):
        return _lower_subarray_module(node, functions)
    if isinstance(node, HIRIndicesOf):
        return _lower_indices_of_module(node, functions)
    if isinstance(node, HIRSort):
        return _lower_sort_module(node, functions)
    if isinstance(node, HIRGrade):
        return _lower_grade_module(node, functions)
    if isinstance(node, HIRFilter):
        return _lower_filter_module(node, functions)
    if isinstance(node, HIRReplicate):
        return _lower_replicate_module(node, functions)
    if isinstance(node, HIRWithShape):
        return _lower_with_shape_module(node, functions)
    if isinstance(node, HIRIota):
        return _lower_iota_module(node)
    if isinstance(node, HIRArrayLit):
        return _lower_array_literal_module(node)
    if isinstance(node, (HIRFold, HIRReduce)):
        return _lower_fold_module(node, functions)
    if isinstance(node, HIRFoldRight):
        return _lower_fold_module(node, functions)
    if isinstance(node, HIRScan):
        return _lower_scan_module(node, functions)
    if not isinstance(node, (HIRMap, HIRApply)):
        raise RemoraLoweringError(
            f"unexpected map/apply node type {type(node).__name__}"
        )
    if len(node.arrays) == 2:
        if not node.cell_shape and not isinstance(
            node.result_type, ArrayType
        ):
            return _lower_scalar_map_binary_module(
                node, functions
            )
        return _lower_binary_map_module(node, functions)
    if len(node.arrays) != 1:
        raise RemoraLoweringError(
            "only unary and binary map MLIR lowering is supported"
        )
    if not isinstance(node.result_type, ArrayType):
        return _lower_scalar_map_module(node, functions)
    return _lower_iota_scalar_map_module(node, functions)


def _lower_tensor_let_module(
    node: HIRLet, functions: dict[str, HIRFunction]
) -> str:
    value_blocks: list[str] = []
    tensor_env: TensorEnv = {}
    scalar_env: dict[str, HIRExpr] = {}
    body: HIRExpr = node
    ordinal = 0
    while isinstance(body, HIRLet) and isinstance(
        body.value_type, ArrayType
    ):
        code, value_name, value_type, element_type = _lower_tensor_input(
            body.value,
            f"let_{ordinal}_{body.name}",
            functions,
            tensor_env,
        )
        value_blocks.append(code)
        tensor_env[body.name] = _TensorValue(
            value_name, value_type, element_type
        )
        body = body.body
        ordinal += 1

    while isinstance(body, HIRLet) and _is_scalar_type(
        body.value_type
    ):
        scalar_env[body.name] = body.value
        body = body.body
    body = _inline_lets(body, scalar_env)

    body_code, result_value, result_type = (
        _lower_main_result_with_tensor_env(
            body,
            functions,
            tensor_env,
        )
    )
    builder = _MLIRMainModuleBuilder(result_type)
    for block in value_blocks:
        builder.add_block(block)
    builder.add_block(body_code)
    return builder.render(result_value)


def _lower_main_result_with_tensor_env(
    node: HIRExpr,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv,
) -> tuple[str, str, str]:
    if isinstance(node, HIRVar):
        code, value_name, value_type, _element_type = _lower_tensor_input(
            node,
            "result",
            functions,
            tensor_env,
        )
        return code, value_name, value_type
    if isinstance(node, (HIRMap, HIRApply)):
        if len(node.arrays) == 2:
            if not node.cell_shape and not isinstance(
                node.result_type, ArrayType
            ):
                raise RemoraLoweringError(
                    "tensor let body cannot be a scalar binary map yet"
                )
            return _lower_binary_map_result(
                node, functions, tensor_env
            )
        if len(node.arrays) != 1:
            raise RemoraLoweringError(
                "only unary and binary map MLIR lowering is supported"
            )
        if not isinstance(node.result_type, ArrayType):
            raise RemoraLoweringError(
                "tensor let body cannot be a scalar map yet"
            )
        return _lower_iota_scalar_map_result(
            node, functions, tensor_env
        )
    if isinstance(node, (HIRFold, HIRReduce)):
        return _lower_fold_result(node, functions, tensor_env)
    if isinstance(
        node,
        (
            HIRIndex,
            HIRSlice,
            HIRTranspose,
            HIRReshape,
            HIRRavel,
            HIRReverse,
            HIRTake,
            HIRDrop,
        ),
    ):
        return _lower_view_result(
            node, functions, tensor_env
        )
    if isinstance(node, (HIRIota, HIRArrayLit, HIRWithShape)):
        code, value_name, value_type, _element_type = (
            _lower_tensor_input(
                node,
                "result",
                functions,
                tensor_env,
            )
        )
        return code, value_name, value_type
    if isinstance(node, HIRIf) and isinstance(
        _expr_result_type(node.condition), ArrayType
    ):
        return _lower_tensor_if_result(
            node, functions, tensor_env
        )
    raise RemoraLoweringError(
        "unsupported tensor let body for MLIR lowering"
    )


def _lower_tensor_if_module(node: HIRIf) -> str:
    body, result_value, result_type = _lower_tensor_if_result(
        node, {}, None
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_tensor_if_result(
    node: HIRIf,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None,
) -> tuple[str, str, str]:
    cond_code, cond_name, cond_type, cond_elem = _lower_tensor_input(
        node.condition, "cond", functions, tensor_env
    )
    then_code, then_name, then_type, then_elem = _lower_tensor_input(
        node.then_branch, "then_val", functions, tensor_env
    )
    else_code, else_name, else_type, else_elem = _lower_tensor_input(
        node.else_branch, "else_val", functions, tensor_env
    )
    result_type = type_to_mlir(node.result_type)
    if then_type != result_type or else_type != result_type:
        raise RemoraLoweringError(
            "tensor if branches must have matching types"
        )
    rank = (
        node.result_type.rank
        if isinstance(node.result_type, ArrayType)
        else 1
    )
    identity = _identity_affine_map(rank)
    iterators = _parallel_iterators(rank)
    result_name = "%if_result"
    body = f"""{cond_code}
{then_code}
{else_code}
    %if_empty = tensor.empty() : {result_type}
    {result_name} = linalg.generic {{
      indexing_maps = [{identity}, {identity}, {identity}, {identity}],
      iterator_types = {iterators}
    }} ins({cond_name}, {then_name}, {else_name} : {cond_type}, {then_type}, {else_type}) outs(%if_empty : {result_type}) {{
    ^bb0(%c: {cond_elem}, %t: {then_elem}, %e: {else_elem}, %out: {then_elem}):
      %selected = arith.select %c, %t, %e : {then_elem}
      linalg.yield %selected : {then_elem}
    }} -> {result_type}
"""
    return body.rstrip(), result_name, result_type


# ---------------------------------------------------------------------------
# Function lowering
# ---------------------------------------------------------------------------


def _lower_functions(functions: dict[str, HIRFunction]) -> str:
    lowered = [
        _lower_function(function) for function in functions.values()
    ]
    return "\n\n".join(lowered)


def _lower_function(function: HIRFunction) -> str:
    has_array_params = any(
        isinstance(param.type, ArrayType)
        for param in function.params
    )
    has_array_return = isinstance(function.return_type, ArrayType)

    if has_array_params or has_array_return:
        return _lower_function_with_tensor(function)

    result_type = type_to_mlir(function.return_type)
    args = [
        f"%arg{index}: {type_to_mlir(param.type)}"
        for index, param in enumerate(function.params)
    ]
    env = {
        param.name: _Operand(
            f"%arg{index}",
            [],
            type_to_mlir(param.type),
        )
        for index, param in enumerate(function.params)
    }
    emitter = _RegionEmitter(
        input_name="",
        input_type="",
        functions={function.name: function},
    )
    value = emitter.emit_expr(function.body, env)
    lines = [
        *emitter.lines,
        *_cast_if_needed(
            value.value, value.type, result_type, "%result_cast"
        ),
    ]
    result_value = (
        "%result_cast" if value.type != result_type else value.value
    )
    body = "\n".join(lines)
    return f"""  func.func private @{function.name}({", ".join(args)}) -> {result_type} {{
{body}
    return {result_value} : {result_type}
  }}"""


def _lower_function_with_tensor(function: HIRFunction) -> str:
    """Lower a HIR function with array params or array return type."""
    result_type = type_to_mlir(function.return_type)
    args = [
        f"%arg{index}: {type_to_mlir(param.type)}"
        for index, param in enumerate(function.params)
    ]
    tensor_env: TensorEnv = {}
    scalar_env: dict[str, _Operand] = {}
    for index, param in enumerate(function.params):
        param_type = type_to_mlir(param.type)
        if isinstance(param.type, ScalarType):
            scalar_env[param.name] = _Operand(
                f"%arg{index}", [], param_type
            )
        elif isinstance(param.type, ArrayType):
            tensor_env[param.name] = _TensorValue(
                f"%arg{index}",
                param_type,
                type_to_mlir(param.type.element),
            )
        else:
            raise RemoraLoweringError(
                "function params must be scalar or array types"
            )

    if isinstance(function.return_type, ArrayType):
        code, result_name, lowered_result_type, _element_type = (
            _lower_tensor_input(
                function.body,
                "result",
                {function.name: function},
                tensor_env,
            )
        )
        if lowered_result_type != result_type:
            raise RemoraLoweringError(
                "lowered function result type mismatch"
            )
        body = code
        result_value = result_name
    elif isinstance(function.return_type, ScalarType):
        body_expr = _inline_lets(function.body)
        if isinstance(body_expr, (HIRMap, HIRApply)):
            code, result_name, lowered_result_type, _element_type = (
                _lower_tensor_input(
                    body_expr,
                    "result",
                    {function.name: function},
                    tensor_env,
                )
            )
            body = code
            result_value = result_name
        elif isinstance(body_expr, (HIRFold, HIRReduce)):
            return (
                _lower_descriptor_scalar_result_body(
                    body_expr,
                    result_type,
                    scalar_env,
                    tensor_env,
                )[0]
                if len(function.params) > 0
                else ""
            )
        else:
            raise RemoraLoweringError(
                "scalar-returning function with array params must have a map or fold body"
            )
        if not body.startswith("    "):
            body = "    " + body.replace("\n", "\n    ")
    else:
        raise RemoraLoweringError(
            "function return type must be scalar or array"
        )

    return f"""  func.func private @{function.name}({", ".join(args)}) -> {result_type} {{
{body}
    return {result_value} : {result_type}
  }}"""


# ---------------------------------------------------------------------------
# Descriptor export (output)
# ---------------------------------------------------------------------------


def _add_output_descriptor_export(
    mlir_text: str, return_type: RemoraType
) -> str:
    wrapper = _output_descriptor_export_function(return_type)
    stripped = mlir_text.rstrip()
    if not stripped.endswith("}"):
        raise RemoraLoweringError(
            "expected lowered MLIR module to end with '}'"
        )
    return f"{stripped[:-1]}{wrapper}\n}}"


def _output_descriptor_export_function(
    return_type: RemoraType,
) -> str:
    result_type = type_to_mlir(return_type)
    memref_type = _output_memref_type(return_type)
    lines = [
        "",
        f"  func.func @remora_main_out(%out: {memref_type}) attributes {{ llvm.emit_c_interface }} {{",
        f"    %result = call @main() : () -> {result_type}",
    ]
    if isinstance(return_type, ScalarType):
        lines.append(
            f"    memref.store %result, %out[] : {memref_type}"
        )
    elif isinstance(return_type, (ArrayType, SigmaType)):
        if isinstance(return_type, SigmaType):
            body_t = return_type.body
            elem = type_to_mlir(body_t.element if isinstance(body_t, ArrayType) else body_t)
            lines.append(
                f"    %result_mem = bufferization.to_memref %result : memref<?x{elem}>"
            )
            lines.append(
                f'    "memref.copy"(%result_mem, %out) : (memref<?x{elem}>, {memref_type}) -> ()'
            )
        else:
            lines.extend(
                _output_descriptor_store_lines(
                    return_type, result_type, memref_type
                )
            )
    else:
        raise RemoraLoweringError(
            f"cannot export output descriptor for type {return_type}"
        )
    lines.extend(
        [
            "    return",
            "  }",
        ]
    )
    return "\n".join(lines)


def _output_memref_type(return_type: RemoraType) -> str:
    if isinstance(return_type, ScalarType):
        return f"memref<{type_to_mlir(return_type)}>"
    if isinstance(return_type, SigmaType):
        elem = type_to_mlir(return_type.body.element if isinstance(return_type.body, ArrayType) else return_type.body)
        return f"memref<?x{elem}>"
    if isinstance(return_type, ArrayType):
        dim_parts: list[str] = []
        for dim in return_type.shape:
            value = getattr(dim, "value", None)
            if value is None:
                raise RemoraLoweringError(
                    f"cannot lower ABI type {return_type}: shape contains "
                    f"non-concrete dimension {dim}"
                )
            dim_parts.append(str(value))
        dims = "x".join(dim_parts)
        element = type_to_mlir(return_type.element)
        if dims:
            strides = ", ".join(
                "?" for _axis in range(return_type.rank)
            )
            return f"memref<{dims}x{element}, strided<[{strides}], offset: ?>>"
        return f"memref<{element}>"
    raise RemoraLoweringError(
        f"cannot lower output descriptor type {return_type}"
    )


def _output_descriptor_store_lines(
    return_type: ArrayType,
    result_type: str,
    memref_type: str,
) -> list[str]:
    if return_type.rank == 0:
        return [
            f"    %value = tensor.extract %result[] : {result_type}",
            f"    memref.store %value, %out[] : {memref_type}",
        ]

    lines = [
        "    %c0 = arith.constant 0 : index",
        "    %c1 = arith.constant 1 : index",
    ]
    for axis, dim in enumerate(return_type.shape):
        lines.append(
            f"    %c{axis}_ub = arith.constant {dim.value} : index"
        )
    for axis in range(return_type.rank):
        indent = "    " + "  " * axis
        lines.append(
            f"{indent}scf.for %i{axis} = %c0 to %c{axis}_ub step %c1 {{"
        )

    indices = ", ".join(
        f"%i{axis}" for axis in range(return_type.rank)
    )
    body_indent = "    " + "  " * return_type.rank
    lines.extend(
        [
            f"{body_indent}%value = tensor.extract %result[{indices}] : {result_type}",
            f"{body_indent}memref.store %value, %out[{indices}] : {memref_type}",
        ]
    )
    for axis in reversed(range(return_type.rank)):
        indent = "    " + "  " * axis
        lines.append(f"{indent}}}")
    return lines


# ---------------------------------------------------------------------------
# Descriptor export (functions)
# ---------------------------------------------------------------------------


def _lower_function_descriptor_module(
    function: HIRFunction, export_name: str
) -> str:
    internal_name = "__remora_entry"
    internal = _lower_descriptor_internal_function(
        function, internal_name
    )
    wrapper = _lower_descriptor_export_wrapper(
        function, internal_name, export_name
    )
    return f"""module {{
{internal}

{wrapper}
}}"""


def _lower_descriptor_internal_function(
    function: HIRFunction, name: str
) -> str:
    arg_decls = [
        f"%arg{index}: {type_to_mlir(param.type)}"
        for index, param in enumerate(function.params)
    ]
    scalar_env: dict[str, _Operand] = {}
    tensor_env: TensorEnv = {}
    for index, param in enumerate(function.params):
        param_type = type_to_mlir(param.type)
        if isinstance(param.type, ScalarType):
            scalar_env[param.name] = _Operand(
                f"%arg{index}", [], param_type
            )
        elif isinstance(param.type, ArrayType):
            tensor_env[param.name] = _TensorValue(
                f"%arg{index}",
                param_type,
                type_to_mlir(param.type.element),
            )
        else:
            raise RemoraLoweringError(
                "descriptor-exported functions require scalar or array parameters"
            )

    result_type = type_to_mlir(function.return_type)
    if isinstance(function.return_type, ArrayType):
        code, result_value, lowered_result_type, _element_type = (
            _lower_tensor_input(
                function.body,
                "result",
                {},
                tensor_env,
            )
        )
        if lowered_result_type != result_type:
            raise RemoraLoweringError(
                "lowered function result type mismatch"
            )
        body = code
    elif isinstance(function.return_type, ScalarType):
        body, result_value = (
            _lower_descriptor_scalar_result_body(
                function.body,
                result_type,
                scalar_env,
                tensor_env,
            )
        )
    else:
        raise RemoraLoweringError(
            "descriptor-exported functions require scalar or array results"
        )

    return f"""  func.func private @{name}({", ".join(arg_decls)}) -> {result_type} {{
{body}
    return {result_value} : {result_type}
    }}"""


def _lower_descriptor_scalar_result_body(
    expr: HIRExpr,
    result_type: str,
    scalar_env: dict[str, _Operand],
    tensor_env: TensorEnv,
) -> tuple[str, str]:
    expr = _inline_lets(expr)
    if isinstance(expr, (HIRFold, HIRReduce)):
        input_code, input_name, input_type, input_element_type = (
            _lower_fold_input(
                expr.array,
                {},
                tensor_env=tensor_env,
            )
        )
        init_code, init_value = _lower_scalar_value_for_fold_init(
            expr.init,
            result_type,
            functions={},
            env=scalar_env,
            result_prefix="init_scalar",
        )
        fold_body = _lower_fold_callable_body(
            expr.func,
            {},
            input_name="%in",
            input_type=input_element_type,
            acc_name="%acc",
            acc_type=result_type,
            result_type=result_type,
        )
        body = f"""{input_code}
{init_code}
    %init = tensor.from_elements {init_value} : tensor<{result_type}>
    %folded = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = [\"reduction\"]
    }} ins({input_name} : {input_type}) outs(%init : tensor<{result_type}>) {{
    ^bb0(%in: {input_element_type}, %acc: {result_type}):
{fold_body}
    }} -> tensor<{result_type}>
    %extracted = tensor.extract %folded[] : tensor<{result_type}>"""
        return body, "%extracted"

    emitter = _RegionEmitter(input_name="", input_type="")
    value = emitter.emit_expr(expr, scalar_env)
    lines = [
        *emitter.lines,
        *_cast_if_needed(
            value.value, value.type, result_type, "%result_cast"
        ),
    ]
    result_value = (
        "%result_cast" if value.type != result_type else value.value
    )
    return "\n".join(lines), result_value


def _lower_descriptor_export_wrapper(
    function: HIRFunction,
    internal_name: str,
    export_name: str,
) -> str:
    input_memrefs = [
        _output_memref_type(param.type)
        for param in function.params
    ]
    output_memref = _output_memref_type(function.return_type)
    args = [
        *(
            f"%arg{index}: {memref}"
            for index, memref in enumerate(input_memrefs)
        ),
        f"%out: {output_memref}",
    ]
    lines = [
        f"  func.func @{export_name}({', '.join(args)}) attributes {{ llvm.emit_c_interface }} {{",
    ]
    call_args: list[str] = []
    call_types: list[str] = []
    for index, param in enumerate(function.params):
        param_type = type_to_mlir(param.type)
        memref_type = input_memrefs[index]
        value_name = f"%in{index}"
        if isinstance(param.type, ScalarType):
            lines.append(
                f"    {value_name} = memref.load %arg{index}[] : {memref_type}"
            )
        elif isinstance(param.type, ArrayType):
            lines.append(
                f"    {value_name} = bufferization.to_tensor %arg{index} restrict : {memref_type}"
            )
        else:
            raise RemoraLoweringError(
                "descriptor-exported functions require scalar or array parameters"
            )
        call_args.append(value_name)
        call_types.append(param_type)

    result_type = type_to_mlir(function.return_type)
    lines.append(
        f"    %result = call @{internal_name}({', '.join(call_args)}) : "
        f"({', '.join(call_types)}) -> {result_type}"
    )
    if isinstance(function.return_type, ScalarType):
        lines.append(
            f"    memref.store %result, %out[] : {output_memref}"
        )
    elif isinstance(function.return_type, ArrayType):
        lines.extend(
            _output_descriptor_store_lines(
                function.return_type,
                result_type,
                output_memref,
            )
        )
    else:
        raise RemoraLoweringError(
            "descriptor-exported functions require scalar or array results"
        )
    lines.extend(["    return", "  }"])
    return "\n".join(lines)
