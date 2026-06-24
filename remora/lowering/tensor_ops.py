"""Tensor operation lowering for MLIR: maps, folds, iota, array literals."""

from __future__ import annotations

from typing import Any

from remora.hir import (
    HIRApply,
    HIRAppend,
    HIRArrayLit,
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
    HIRIota,
    HIRLambda,
    HIRLet,
    HIRLit,
    HIRMap,
    HIRMatmul,
    HIRPrimCallable,
    HIRPrimOp,
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
    HIRScatterAdd,
    HIRIm2col,
    HIRCol2im,
    HIRTake,
    HIRTranspose,
    HIRVar,
    HIRWithShape,
    HIRIndicesOf,
)
from remora.types import ArrayType, BOOL, FLOAT, INT, ScalarType, SigmaType, StaticDim

from remora.lowering.scalar import (
    _Operand,
    _RegionEmitter,
    _arith_op,
    _cast_if_needed,
    _literal_value,
    _lower_callable_operand,
    _lower_scalar_value_for_fold_init,
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


# ---------------------------------------------------------------------------
# Affine map helpers
# ---------------------------------------------------------------------------


def _identity_affine_map(rank: int) -> str:
    dims = ", ".join(f"d{axis}" for axis in range(rank))
    results = ", ".join(f"d{axis}" for axis in range(rank))
    return f"affine_map<({dims}) -> ({results})>"


def _constant_affine_map(rank: int) -> str:
    """Affine map projecting from *rank* dimensions to a scalar (no results)."""
    dims = ", ".join(f"d{axis}" for axis in range(rank))
    return f"affine_map<({dims}) -> ()>"


def _parallel_iterators(rank: int) -> str:
    return "[" + ", ".join('"parallel"' for _axis in range(rank)) + "]"


def _tensor_rank_from_mlir_type(mlir_type: str) -> int:
    """Extract the rank from a MLIR tensor type string like 'tensor<2x3xi32>'."""
    if not mlir_type.startswith("tensor<"):
        return 0  # scalar type like 'i32', 'f32', etc.
    inner = mlir_type[len("tensor<") : -1]  # remove 'tensor<' and trailing '>'
    # Find the element type by looking for the last 'x' followed by non-digit
    # Simple: count dimensions by splitting on 'x' and checking if parts are digits
    parts = inner.split("x")
    dim_count = 0
    for p in parts:
        if p and p[0].isdigit():
            dim_count += 1
        else:
            break
    return dim_count


# ---------------------------------------------------------------------------
# Memref-interface call lowering (manual bufferization for recursion)
# ---------------------------------------------------------------------------


def _lower_mref_call(
    node: HIRCall,
    func: HIRFunction,
    prefix: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str, str]:
    """Lower a HIRCall to a memref-interface function (``__mref`` suffix).

    Tensor-typed args are wrapped in ``memref.alloc`` + copy before the
    call and the result is read back via ``bufferization.to_tensor``
    afterwards.  This keeps the call boundary purely memref→memref so the
    bufferize-function-boundaries pipeline pass sees no tensor loopback.
    """
    from remora.lowering.module import (
        _plain_memref_type,
        _output_descriptor_store_lines,
    )
    from remora.lowering.scalar import (
        _RegionEmitter,
        _Operand as _ScOperand,
    )
    from remora.types import ScalarType as _ScalarType

    result_type = type_to_mlir(node.result_type)
    result_elem = (
        type_to_mlir(node.result_type.element)
        if isinstance(node.result_type, ArrayType)
        else result_type
    )

    arg_lines: list[str] = []
    call_args: list[str] = []   # SSA names passed to func.call
    call_types: list[str] = []  # MLIR types for call signature

    # Output memref (allocated first; callee stores result into it)
    mref_out_type = _plain_memref_type(node.result_type)
    out_memref = f"%{prefix}_out"
    arg_lines.append(f"    {out_memref} = memref.alloc() : {mref_out_type}")
    call_args.append(out_memref)
    call_types.append(mref_out_type)

    for i, arg in enumerate(node.args):
        arg_remora = _expr_result_type(arg)
        if isinstance(arg_remora, ArrayType):
            arg_code, arg_name, arg_type, arg_elem = _lower_tensor_input(
                arg,
                _join_prefix(prefix, f"a{i}"),
                functions,
                tensor_env,
                scalar_env,
            )
            if arg_code:
                arg_lines.append(arg_code)

            # Wrap tensor in a fresh memref
            in_mref_type = _plain_memref_type(arg_remora)
            in_memref = f"%{prefix}_in{i}"
            arg_lines.append(
                f"    {in_memref} = memref.alloc() : {in_mref_type}"
            )
            copy_lines = _output_descriptor_store_lines(
                arg_remora,
                arg_type,
                in_mref_type,
                result_name=arg_name,
                out_name=in_memref,
                const_prefix=f"_c{prefix}_{i}",
            )
            arg_lines.extend(copy_lines)
            call_args.append(in_memref)
            call_types.append(in_mref_type)
        else:
            arg_prefix = _join_prefix(prefix, f"a{i}")
            sc_em = _RegionEmitter(
                input_name=arg_prefix,
                input_type="",
                functions=functions,
                prefix=arg_prefix,
            )
            sc_op = sc_em.emit_expr(arg, scalar_env or {})
            arg_lines.extend(sc_em.lines)
            call_args.append(sc_op.value)
            call_types.append(sc_op.type)

    call_name = f"%{prefix}"
    arg_list = ", ".join(call_args)
    type_list = ", ".join(call_types)

    # The memref function returns void — result is written into %out_memref
    arg_lines.append(
        f"    func.call @{func.name}({arg_list})"
        f" : ({type_list}) -> ()"
    )
    # Read result tensor from the output memref
    arg_lines.append(
        f"    {call_name} = bufferization.to_tensor {out_memref}"
        f" restrict writable : {mref_out_type}"
    )

    code = "\n".join(arg_lines)
    return code, call_name, result_type, result_elem


def _fold_iterators(rank: int) -> str:
    if rank < 1:
        raise RemoraLoweringError("fold rank must be at least 1")
    iterators = ['"reduction"', *('"parallel"' for _axis in range(rank - 1))]
    return "[" + ", ".join(iterators) + "]"


def _drop_first_affine_map(rank: int) -> str:
    if rank < 2:
        raise RemoraLoweringError("array-cell fold rank must be at least 2")
    dims = ", ".join(f"d{axis}" for axis in range(rank))
    results = ", ".join(f"d{axis}" for axis in range(1, rank))
    return f"affine_map<({dims}) -> ({results})>"


def _take_first_affine_map(rank: int, count: int) -> str:
    if count < 1 or count > rank:
        raise RemoraLoweringError("invalid affine map result rank")
    dims = ", ".join(f"d{axis}" for axis in range(rank))
    results = ", ".join(f"d{axis}" for axis in range(count))
    return f"affine_map<({dims}) -> ({results})>"


def _take_last_affine_map(rank: int, count: int) -> str:
    """Affine map projecting from *rank* dimensions to the last *count*.

    Used to broadcast a cell-shaped free array (trailing dims of the input)
    across the leading frame dimensions of a cell-fold reduction.
    """
    if count < 1 or count > rank:
        raise RemoraLoweringError("invalid affine map result rank")
    dims = ", ".join(f"d{axis}" for axis in range(rank))
    results = ", ".join(f"d{axis}" for axis in range(rank - count, rank))
    return f"affine_map<({dims}) -> ({results})>"


def _reverse_first_axis_affine_map(array_type: ArrayType) -> str:
    if array_type.rank < 1:
        raise RemoraLoweringError("reverse expects an array of rank at least 1")
    if not isinstance(array_type.shape[0], StaticDim):
        raise RemoraLoweringError("reverse requires a static leading dimension")
    dims = ", ".join(f"d{axis}" for axis in range(array_type.rank))
    results = [f"d{axis}" for axis in range(array_type.rank)]
    results[0] = f"{array_type.shape[0].value - 1} - d0"
    return f"affine_map<({dims}) -> ({', '.join(results)})>"


def _map_cell_iterators(frame_rank: int, cell_rank: int) -> str:
    if cell_rank < 1:
        raise RemoraLoweringError(
            "cell maps require at least one cell dimension"
        )
    iterators = [
        *('"parallel"' for _axis in range(frame_rank)),
        *('"reduction"' for _axis in range(cell_rank)),
    ]
    return "[" + ", ".join(iterators) + "]"


def _cell_element_affine_map(frame_rank: int, position: int) -> str:
    frame_dims = ", ".join(f"d{axis}" for axis in range(frame_rank))
    results = ", ".join(
        [f"d{axis}" for axis in range(frame_rank)] + [str(position)]
    )
    return f"affine_map<({frame_dims}) -> ({results})>"


# ---------------------------------------------------------------------------
# Iota / array literal lowering
# ---------------------------------------------------------------------------


def _lower_iota_module(node: HIRIota) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    result_type = type_to_mlir(node.result_type)
    element_type = type_to_mlir(node.result_type.element)
    if element_type != "i32":
        raise RemoraLoweringError(
            "iota lowering currently supports i32 results only"
        )

    body = f"""    %empty = tensor.empty() : {result_type}
    %result = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = [\"parallel\"]
    }} outs(%empty : {result_type}) {{
    ^bb0(%out: {element_type}):
      %idx = linalg.index 0 : index
      %cast = arith.index_cast %idx : index to {element_type}
      linalg.yield %cast : {element_type}
    }} -> {result_type}
"""
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%result")


def _lower_array_literal_module(node: HIRArrayLit) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    code, name, result_type, _element_type = _lower_tensor_input(
        node, "literal", {}
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(code)
    return builder.render(name)


# ---------------------------------------------------------------------------
# Tensor input lowering (entry point for turning HIR exprs into SSA values)
# ---------------------------------------------------------------------------


def _lower_matmul_tensor_input(
    node,
    prefix: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str, str]:
    """Lower ``HIRMatmul`` to a BLAS runtime call (f32) or ``linalg.matmul``."""
    left = node.left
    right = node.right
    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)

    # Lower left and right operands
    left_code, left_name, left_type, left_elem = _lower_tensor_input(
        left, f"{prefix}_left", functions, tensor_env, scalar_env
    )
    right_code, right_name, right_type, right_elem = _lower_tensor_input(
        right, f"{prefix}_right", functions, tensor_env, scalar_env
    )

    left_rtype = _expr_result_type(left)
    right_rtype = _expr_result_type(right)
    if (
        result_elem == "f32"
        and isinstance(left_rtype, ArrayType)
        and isinstance(right_rtype, ArrayType)
        and left_rtype.rank == 2
        and right_rtype.rank == 2
    ):
        M = left_rtype.shape[0].value
        K = left_rtype.shape[1].value
        N = right_rtype.shape[1].value
        p = prefix
        a_mr = f"memref<{M}x{K}xf32>"
        b_mr = f"memref<{K}x{N}xf32>"
        c_mr = f"memref<{M}x{N}xf32>"
        result_name = f"%{p}"
        code = f"""{left_code}
{right_code}
    %{p}_c0 = arith.constant 0 : index
    %{p}_c1 = arith.constant 1 : index
    %{p}_cM = arith.constant {M} : index
    %{p}_cK = arith.constant {K} : index
    %{p}_cN = arith.constant {N} : index
    %{p}_abuf = memref.alloc() : {a_mr}
    scf.for %{p}_ai = %{p}_c0 to %{p}_cM step %{p}_c1 {{
      scf.for %{p}_aj = %{p}_c0 to %{p}_cK step %{p}_c1 {{
        %{p}_av = tensor.extract {left_name}[%{p}_ai, %{p}_aj] : {left_type}
        memref.store %{p}_av, %{p}_abuf[%{p}_ai, %{p}_aj] : {a_mr}
      }}
    }}
    %{p}_bbuf = memref.alloc() : {b_mr}
    scf.for %{p}_bi = %{p}_c0 to %{p}_cK step %{p}_c1 {{
      scf.for %{p}_bj = %{p}_c0 to %{p}_cN step %{p}_c1 {{
        %{p}_bv = tensor.extract {right_name}[%{p}_bi, %{p}_bj] : {right_type}
        memref.store %{p}_bv, %{p}_bbuf[%{p}_bi, %{p}_bj] : {b_mr}
      }}
    }}
    %{p}_cbuf = memref.alloc() : {c_mr}
    func.call @remora_matmul_f32(%{p}_abuf, %{p}_bbuf, %{p}_cbuf) : ({a_mr}, {b_mr}, {c_mr}) -> ()
    {result_name} = bufferization.to_tensor %{p}_cbuf restrict writable : {c_mr}"""
        return code, result_name, result_type, result_elem

    # Create an empty output tensor for linalg.matmul
    empty_name = f"%{prefix}_empty"
    zero_name = f"%{prefix}_zero"
    zero_val = f"%{prefix}_zv"
    result_name = f"%{prefix}"

    code = f"""{left_code}
{right_code}
    {empty_name} = tensor.empty() : {result_type}
    {zero_val} = arith.constant 0.0 : {result_elem}
    {zero_name} = linalg.fill ins({zero_val} : {result_elem}) outs({empty_name} : {result_type}) -> {result_type}
    {result_name} = linalg.matmul
      ins({left_name}, {right_name} : {left_type}, {right_type})
      outs({zero_name} : {result_type}) -> {result_type}"""

    return code, result_name, result_type, result_elem


def _lower_scan_tensor_input(
    node: HIRScan,
    prefix: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str, str]:
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("scan tensor input requires array result type")
    if node.result_type.rank != 1:
        raise RemoraLoweringError("scan tensor input only supports rank 1")

    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)
    init_value_str = _literal_value(node.init, result_elem)
    op_name = _arith_op(node.func.op, result_elem)
    N = node.reduction_dim.value
    p = prefix

    input_code, input_name, input_type, _input_elem = _lower_tensor_input(
        node.array, f"{p}_in", functions, tensor_env, scalar_env
    )

    if node.exclusive:
        loop = f"""\
    %{p}_stored = tensor.insert %{p}_c into %{p}_acc[%{p}_i] : {result_type}
      %{p}_elem = tensor.extract {input_name}[%{p}_i] : {input_type}
      %{p}_next = {op_name} %{p}_c, %{p}_elem : {result_elem}"""
    elif node.right:
        loop = f"""\
    %{p}_rev = arith.subi %{p}_cNm1, %{p}_i : index
      %{p}_elem = tensor.extract {input_name}[%{p}_rev] : {input_type}
      %{p}_next = {op_name} %{p}_c, %{p}_elem : {result_elem}
      %{p}_stored = tensor.insert %{p}_next into %{p}_acc[%{p}_rev] : {result_type}"""
    else:
        loop = f"""\
    %{p}_elem = tensor.extract {input_name}[%{p}_i] : {input_type}
      %{p}_next = {op_name} %{p}_c, %{p}_elem : {result_elem}
      %{p}_stored = tensor.insert %{p}_next into %{p}_acc[%{p}_i] : {result_type}"""

    extra_consts = ""
    if node.right:
        extra_consts = f"\n    %{p}_cNm1 = arith.constant {N - 1} : index"

    code = f"""{input_code}
    %{p}_init = arith.constant {init_value_str} : {result_elem}
    %{p}_c0 = arith.constant 0 : index
    %{p}_c1 = arith.constant 1 : index
    %{p}_cN = arith.constant {N} : index{extra_consts}
    %{p}_empty = tensor.empty() : {result_type}
    %{p}_filled = linalg.fill ins(%{p}_init : {result_elem}) outs(%{p}_empty : {result_type}) -> {result_type}
    %{p}, %{p}_carry = "scf.for"(%{p}_c0, %{p}_cN, %{p}_c1, %{p}_filled, %{p}_init) ({{
    ^bb0(%{p}_i: index, %{p}_acc: {result_type}, %{p}_c: {result_elem}):
      {loop}
      "scf.yield"(%{p}_stored, %{p}_next) : ({result_type}, {result_elem}) -> ()
    }}) : (index, index, index, {result_type}, {result_elem}) -> ({result_type}, {result_elem})"""

    return code, f"%{p}", result_type, result_elem


def _lower_sort_tensor_input(
    node: HIRSort,
    prefix: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str, str]:
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("sort tensor input requires array result type")
    if node.result_type.rank != 1:
        raise RemoraLoweringError("sort tensor input only supports rank 1")

    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)
    N = node.result_type.shape[0].value
    rt_func = _sort_runtime_func(result_elem)
    p = prefix

    input_code, input_name, input_type, _input_elem = _lower_tensor_input(
        node.array, f"{p}_in", functions, tensor_env, scalar_env
    )

    code = f"""{input_code}
    %{p}_c0 = arith.constant 0 : index
    %{p}_c1 = arith.constant 1 : index
    %{p}_cN = arith.constant {N} : index
    %{p}_buf = memref.alloc() : memref<{N}x{result_elem}>
    scf.for %{p}_i = %{p}_c0 to %{p}_cN step %{p}_c1 {{
      %{p}_val = tensor.extract {input_name}[%{p}_i] : {input_type}
      memref.store %{p}_val, %{p}_buf[%{p}_i] : memref<{N}x{result_elem}>
    }}
    func.call @{rt_func}(%{p}_buf) : (memref<{N}x{result_elem}>) -> ()
    %{p} = bufferization.to_tensor %{p}_buf restrict writable : memref<{N}x{result_elem}>"""

    return code, f"%{p}", result_type, result_elem


def _lower_tensor_input(
    node: HIRExpr,
    prefix: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str, str]:
    if isinstance(node, HIRVar):
        if tensor_env is None or node.name not in tensor_env:
            raise RemoraLoweringError(
                "only tensor literals, iota values, and descriptor inputs lower as tensor inputs so far"
            )
        value = tensor_env[node.name]
        return "", value.name, value.type, value.element_type

    if isinstance(node, HIRIota):
        result_type = type_to_mlir(node.result_type)
        element_type = type_to_mlir(node.result_type.element)
        code = f"""    %{prefix}_empty = tensor.empty() : {result_type}
    %{prefix} = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = [\"parallel\"]
    }} outs(%{prefix}_empty : {result_type}) {{
    ^bb0(%{prefix}_out: {element_type}):
      %{prefix}_idx = linalg.index 0 : index
      %{prefix}_cast = arith.index_cast %{prefix}_idx : index to {element_type}
      linalg.yield %{prefix}_cast : {element_type}
    }} -> {result_type}"""
        return code, f"%{prefix}", result_type, element_type

    if isinstance(node, HIRArrayLit):
        result_type = type_to_mlir(node.result_type)
        element_type = type_to_mlir(node.result_type.element)
        flat = _flatten_array_literal(node)
        if not flat:
            return (
                f"    %{prefix} = tensor.empty() : {result_type}",
                f"%{prefix}",
                result_type,
                element_type,
            )
        lines = []
        names = []
        for index, literal in enumerate(flat):
            name = f"%{prefix}_c{index}"
            names.append(name)
            lines.append(
                f"    {name} = arith.constant {_literal_value(literal, element_type)} : {element_type}"
            )
        values = ", ".join(names)
        lines.append(
            f"    %{prefix} = tensor.from_elements {values} : {result_type}"
        )
        return "\n".join(lines), f"%{prefix}", result_type, element_type

    if isinstance(node, HIRAppend):
        return _lower_append_input(
            node,
            prefix,
            functions,
            tensor_env=tensor_env,
            scalar_env=scalar_env,
        )

    if isinstance(node, (HIRFold, HIRReduce)) and isinstance(
        node.result_type, ArrayType
    ):
        code, name, result_type = _lower_fold_result(
            node, functions, tensor_env, prefix=prefix, scalar_env=scalar_env
        )
        return code, name, result_type, type_to_mlir(node.result_type.element)

    if isinstance(node, HIRScatterAdd):
        from remora.lowering.scalar import _lower_scalar_module

        target_code, target_name, target_type, target_elem = _lower_tensor_input(
            node.target, _join_prefix(prefix, "target"), functions, tensor_env, scalar_env
        )
        # Lower the index: literal → constant, non-literal → scalar module
        if isinstance(node.index, HIRLit) and node.index.type == INT:
            idx_val = int(node.index.value)
            idx_code = f"    %{_join_prefix(prefix, 'idx')} = arith.constant {idx_val} : index"
            idx_name = f"%{_join_prefix(prefix, 'idx')}"
        else:
            raise RemoraLoweringError(
                "scatter-add fold input only supports literal index values"
            )
        # Lower update
        if isinstance(node.update, HIRLit):
            lit_val = _literal_value(node.update, target_elem)
            update_name = f"%{_join_prefix(prefix, 'update')}"
            update_code = f"    {update_name} = arith.constant {lit_val} : {target_elem}"
        elif isinstance(node.update, HIRIndex) and len(node.update.indices) == 1:
            idx_item = node.update.indices[0]
            if isinstance(idx_item, HIRLit):
                idx_val = int(idx_item.value)
                arr_code, arr_name, arr_type, arr_elem = _lower_tensor_input(
                    node.update.array, _join_prefix(prefix, "idx_arr"), functions, tensor_env, scalar_env
                )
                update_name = f"%{_join_prefix(prefix, 'update')}"
                update_code = f"""{arr_code}
    {update_name}_pos = arith.constant {idx_val} : index
    {update_name} = tensor.extract {arr_name}[{update_name}_pos] : {arr_type}"""
            else:
                raise RemoraLoweringError(
                    "scatter-add cannot lower non-literal index in fold input"
                )
        else:
            raise RemoraLoweringError(
                f"scatter-add cannot lower update of type {type(node.update).__name__}"
            )
        result_type = type_to_mlir(node.result_type)
        result_name = f"%{prefix}"
        code = f"""{target_code}
{idx_code}
{update_code}
    {result_name}_extracted = tensor.extract {target_name}[{idx_name}] : {target_type}
    {result_name}_added = arith.addf {result_name}_extracted, {update_name} : {target_elem}
    {result_name} = tensor.insert {result_name}_added into {target_name}[{idx_name}] : {target_type}"""
        return code, result_name, result_type, target_elem

    if isinstance(node, (HIRMap, HIRApply)):
        if node.cell_shape:
            code, name, result_type = _lower_map_cell_result(
                node, functions, tensor_env, scalar_env=scalar_env,
                prefix=prefix,
            )
            if not isinstance(node.result_type, ArrayType):
                raise RemoraLoweringError("cell-map tensor input must be an array")
            return code, name, result_type, type_to_mlir(node.result_type.element)
        return _lower_fold_input(
            node, functions, prefix, tensor_env=tensor_env, scalar_env=scalar_env
        )

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
            HIRSubarray,
            HIRRotate,
            HIRScatterAdd,
        ),
    ):
        from remora.lowering.view_ops import _lower_view_input

        return _lower_view_input(
            node, functions, prefix, tensor_env=tensor_env, scalar_env=scalar_env
        )

    if isinstance(node, HIRWithShape):
        result_type = type_to_mlir(node.result_type)
        result_elem = type_to_mlir(node.result_type.element)
        rank = node.result_type.rank

        if isinstance(node.source, HIRLit):
            lit_val = _literal_value(node.source, result_elem)
            identity = _identity_affine_map(rank)
            iterators = _parallel_iterators(rank)
            val_name = f"%{prefix}_val"
            empty_name = f"%{prefix}_empty"
            target_name = f"%{prefix}"
            code = f"""    {val_name} = arith.constant {lit_val} : {result_elem}
    {empty_name} = tensor.empty() : {result_type}
    {target_name} = linalg.generic {{
      indexing_maps = [{identity}],
      iterator_types = {iterators}
    }} outs({empty_name} : {result_type}) {{
    ^bb0(%out: {result_elem}):
      linalg.yield {val_name} : {result_elem}
    }} -> {result_type}"""
            return code, target_name, result_type, result_elem

        # Non-literal source: recursively lower, then broadcast
        source_remora = _expr_result_type(node.source)
        if isinstance(source_remora, ArrayType):
            src_code, src_name, src_type, src_elem = _lower_tensor_input(
                node.source, f"{prefix}_src", functions, tensor_env, scalar_env
            )
            source_rank = source_remora.rank
            # Broadcast: source maps to last source_rank dims of target
            all_dims = ", ".join(f"d{a}" for a in range(rank))
            src_dims = ", ".join(f"d{a}" for a in range(rank - source_rank, rank))
            src_map = f"affine_map<({all_dims}) -> ({src_dims})>"
            tgt_map = _identity_affine_map(rank)
            iterators = _parallel_iterators(rank)
            empty_name = f"%{prefix}_empty"
            target_name = f"%{prefix}"
            code = f"""{src_code}
    {empty_name} = tensor.empty() : {result_type}
    {target_name} = linalg.generic {{
      indexing_maps = [{src_map}, {tgt_map}],
      iterator_types = {iterators}
    }} ins({src_name} : {src_type}) outs({empty_name} : {result_type}) {{
    ^bb0(%in: {src_elem}, %out: {result_elem}):
      linalg.yield %in : {result_elem}
    }} -> {result_type}"""
            return code, target_name, result_type, result_elem

        raise RemoraLoweringError(
            "only scalar-literal or array-source with-shape lowers as tensor input so far"
        )

    if isinstance(node, HIRIm2col):
        return _lower_im2col_tensor_input(
            node, functions, prefix, tensor_env, scalar_env
        )

    if isinstance(node, HIRCol2im):
        return _lower_col2im_tensor_input(
            node, functions, prefix, tensor_env, scalar_env
        )

    if isinstance(node, HIRIf) and isinstance(node.result_type, ArrayType):
        return _lower_if_tensor_input(
            node, functions, prefix, tensor_env, scalar_env
        )

    if isinstance(node, HIRMatmul):
        return _lower_matmul_tensor_input(node, prefix, functions, tensor_env, scalar_env)

    if isinstance(node, HIRScan) and isinstance(node.result_type, ArrayType):
        return _lower_scan_tensor_input(node, prefix, functions, tensor_env, scalar_env)

    if isinstance(node, HIRSort) and isinstance(node.result_type, ArrayType):
        return _lower_sort_tensor_input(node, prefix, functions, tensor_env, scalar_env)

    if isinstance(node, HIRCall):
        func = functions.get(node.func_name)
        if func is not None:
            # ── memref-interface call (manual bufferization wrapper) ──
            if func.name.startswith("__") and func.name.endswith("_mref"):
                return _lower_mref_call(
                    node, func, prefix, functions, tensor_env, scalar_env
                )

            from remora.lowering.scalar import _RegionEmitter, _Operand as _ScOperand
            from remora.types import ScalarType as _ScalarType
            arg_lines: list[str] = []
            arg_names: list[str] = []
            arg_types: list[str] = []
            for i, arg in enumerate(node.args):
                arg_remora = _expr_result_type(arg)
                if isinstance(arg_remora, _ScalarType):
                    arg_prefix = _join_prefix(prefix, f"a{i}")
                    sc_em = _RegionEmitter(
                        input_name=arg_prefix, input_type="",
                        functions=functions,
                        prefix=arg_prefix,
                    )
                    sc_op = sc_em.emit_expr(arg, scalar_env or {})
                    arg_lines.extend(sc_em.lines)
                    arg_names.append(sc_op.value)
                    arg_types.append(sc_op.type)
                else:
                    arg_code, arg_name, arg_type, arg_elem = _lower_tensor_input(
                        arg, _join_prefix(prefix, f"a{i}"), functions, tensor_env, scalar_env
                    )
                    if arg_code:
                        arg_lines.append(arg_code)
                    arg_names.append(arg_name)
                    arg_types.append(arg_type)
            result_type = type_to_mlir(node.result_type)
            result_elem = type_to_mlir(node.result_type.element) if isinstance(node.result_type, ArrayType) else result_type
            call_name = f"%{prefix}"
            arg_list = ", ".join(arg_names)
            type_list = ", ".join(arg_types)
            call_line = (
                f"    {call_name} = func.call @{func.name}({arg_list})"
                f" : ({type_list}) -> {result_type}"
            )
            code = "\n".join(arg_lines + [call_line]) if arg_lines else call_line
            return code, call_name, result_type, result_elem

    raise RemoraLoweringError(
        "only tensor literals and iota values lower as tensor inputs so far"
    )


def _lower_if_tensor_input(node, functions, prefix, tensor_env, scalar_env=None):
    """Lower an array-typed HIRIf as a tensor input."""
    from remora.types import ScalarType as _ScalarType

    cond_type = (
        node.condition.result_type
        if hasattr(node.condition, "result_type")
        else node.condition.type
    )
    if isinstance(cond_type, _ScalarType):
        return _lower_if_tensor_input_scalar_cond(
            node, functions, prefix, tensor_env, scalar_env
        )

    cond_code, cond_name, cond_type, cond_elem = _lower_tensor_input(
        node.condition, f"{prefix}_cond", functions, tensor_env, scalar_env
    )
    then_code, then_name, then_type, then_elem = _lower_tensor_input(
        node.then_branch, f"{prefix}_then", functions, tensor_env, scalar_env
    )
    else_code, else_name, else_type, else_elem = _lower_tensor_input(
        node.else_branch, f"{prefix}_else", functions, tensor_env, scalar_env
    )
    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)
    rank = node.result_type.rank
    
    identity = ", ".join(f"d{a}" for a in range(rank))
    identity_map = f"affine_map<({identity}) -> ({identity})>"
    iterators = "[" + ", ".join('"parallel"' for _ in range(rank)) + "]"
    
    empty_name = f"%{prefix}_empty"
    result_name = f"%{prefix}"
    code = f"""{cond_code}
{then_code}
{else_code}
    {empty_name} = tensor.empty() : {result_type}
    {result_name} = linalg.generic {{
      indexing_maps = [{identity_map}, {identity_map}, {identity_map}, {identity_map}],
      iterator_types = {iterators}
    }} ins({cond_name}, {then_name}, {else_name} : {cond_type}, {then_type}, {else_type})
      outs({empty_name} : {result_type}) {{
    ^bb0(%c: {cond_elem}, %t: {result_elem}, %e: {result_elem}, %o: {result_elem}):
      %sel = arith.select %c, %t, %e : {result_elem}
      linalg.yield %sel : {result_elem}
    }} -> {result_type}"""
    return code, result_name, result_type, result_elem


def _lower_if_tensor_input_scalar_cond(
    node, functions, prefix, tensor_env, scalar_env=None
):
    """Lower an array-typed HIRIf with a scalar condition via scf.if."""
    from remora.lowering.scalar import _RegionEmitter as _ScRegionEmitter

    cond_prefix = _join_prefix(prefix, "cond")
    sc_emitter = _ScRegionEmitter(
        input_name=cond_prefix, input_type="", functions=functions,
        prefix=cond_prefix,
    )
    cond_op = sc_emitter.emit_expr(node.condition, scalar_env or {})
    cond_lines = "\n".join(sc_emitter.lines)
    cond_val = cond_op.value

    then_code, then_name, then_type, then_elem = _lower_tensor_input(
        node.then_branch, f"{prefix}_then", functions, tensor_env, scalar_env
    )
    else_code, else_name, else_type, else_elem = _lower_tensor_input(
        node.else_branch, f"{prefix}_else", functions, tensor_env, scalar_env
    )
    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)
    result_name = f"%{prefix}"

    # Put branch computations *inside* the scf.if regions so that
    # recursive / side-effecting operations are guarded by the
    # condition (control-dependent), not executed unconditionally.
    def _indent_block(text: str, spaces: int = 6) -> str:
        prefix = " " * spaces
        return "\n".join(
            prefix + line.lstrip()
            for line in text.split("\n")
            if line.strip()
        )

    _inthen = _indent_block(then_code)
    _inelse = _indent_block(else_code)

    code = f"""{cond_lines}
    {result_name} = scf.if {cond_val} -> ({result_type}) {{
{_inthen}
      scf.yield {then_name} : {result_type}
    }} else {{
{_inelse}
      scf.yield {else_name} : {result_type}
    }}"""
    return code, result_name, result_type, result_elem


def _lower_transpose_input(
    node: HIRTranspose,
    functions: dict[str, HIRFunction],
    prefix: str,
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str, str]:
    from remora.lowering.view_ops import _lower_transpose_result

    code, result_value, result_type = _lower_transpose_result(
        node, functions, tensor_env, scalar_env=scalar_env
    )
    element_type = type_to_mlir(node.result_type.element)
    return code, result_value, result_type, element_type


# ---------------------------------------------------------------------------
# Scalar map / binary map modules (top-level entry points)
# ---------------------------------------------------------------------------


def _lower_scalar_map_module(
    node: HIRMap | HIRApply, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if node.frame_shape or node.cell_shape:
        raise RemoraLoweringError(
            "only rank-0 scalar maps lower as scalar MLIR so far"
        )

    result_type = type_to_mlir(node.result_type)
    emitter = _RegionEmitter(input_name="", input_type="")
    input_value = emitter.emit_expr(node.array, {})
    callable_lines, result_value = _lower_map_callable_result(
        node.func,
        functions,
        input_name=input_value.value,
        input_type=input_value.type,
        result_type=result_type,
        next_temp=emitter.next_temp,
    )
    builder = _MLIRMainModuleBuilder(result_type, functions=functions)
    builder.add_block("\n".join([*emitter.lines, *callable_lines]))
    return builder.render(result_value)


def _lower_scalar_map_binary_module(
    node: HIRMap | HIRApply, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if node.frame_shape or node.cell_shape:
        raise RemoraLoweringError("only rank-0 binary maps lower as scalar MLIR")
    if len(node.arrays) != 2:
        raise RemoraLoweringError(
            "binary scalar map requires exactly two inputs"
        )

    result_type = type_to_mlir(node.result_type)
    emitter = _RegionEmitter(
        input_name="", input_type="", functions=functions
    )
    left = emitter.emit_expr(node.arrays[0], {})
    right = emitter.emit_expr(node.arrays[1], {})
    callable_lines, result_value = _lower_map_binary_callable_result(
        node.func,
        functions,
        left_name=left.value,
        left_type=left.type,
        right_name=right.value,
        right_type=right.type,
        result_type=result_type,
        next_temp=emitter.next_temp,
    )
    builder = _MLIRMainModuleBuilder(result_type, functions=functions)
    builder.add_block("\n".join([*emitter.lines, *callable_lines]))
    return builder.render(result_value)


# ---------------------------------------------------------------------------
# Tensor map lowering
# ---------------------------------------------------------------------------


def _lower_iota_scalar_map_module(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_value, result_type = _lower_iota_scalar_map_result(
        node,
        functions,
        tensor_env,
        scalar_env=scalar_env,
    )
    builder = _MLIRMainModuleBuilder(result_type, functions=functions)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_iota_scalar_map_result(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    *,
    scalar_env: dict[str, _Operand] | None = None,
    prefix: str = "",
) -> tuple[str, str, str]:
    if node.cell_shape:
        return _lower_map_cell_result(
            node, functions, tensor_env, scalar_env=scalar_env, prefix=prefix
        )
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("map lowering requires an array result")

    input_code, input_name, input_type, input_element_type = (
        _lower_tensor_input(
            node.array,
            _join_prefix(prefix, "input"),
            functions,
            tensor_env,
            scalar_env,
        )
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    rank = node.result_type.rank
    identity = _identity_affine_map(rank)
    iterators = _parallel_iterators(rank)
    op_lines = _lower_map_callable_body(
        node.func,
        functions,
        input_name="%in",
        input_type=input_element_type,
        result_type=result_element_type,
        scalar_env=scalar_env,
    )

    body = f"""{input_code}
    %map_empty = tensor.empty() : {result_type}
    %mapped = linalg.generic {{
      indexing_maps = [{identity}, {identity}],
      iterator_types = {iterators}
    }} ins({input_name} : {input_type}) outs(%map_empty : {result_type}) {{
    ^bb0(%in: {input_element_type}, %out: {result_element_type}):
{op_lines}
    }} -> {result_type}
"""
    return body.rstrip(), "%mapped", result_type


def _lower_binary_map_module(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_value, result_type = _lower_binary_map_result(
        node, functions, tensor_env, scalar_env=scalar_env
    )
    builder = _MLIRMainModuleBuilder(result_type, functions=functions)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_binary_map_result(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    *,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str]:
    if len(node.arrays) != 2:
        raise RemoraLoweringError("binary map requires exactly two inputs")
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError(
            "ranked binary map lowering requires an array result"
        )

    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)

    def _lower_input(arg: HIRExpr, prefix: str):
        """Lower a single map input, promoting scalars to tensors."""
        arg_remora_type = _expr_result_type(arg)
        if _is_scalar_type(arg_remora_type):
            rank = node.result_type.rank
            splat_identity = _identity_affine_map(rank)
            splat_iterators = _parallel_iterators(rank)
            if isinstance(arg, HIRLit):
                scalar_code = (
                    f"    %{prefix}_scalar = arith.constant "
                    f"{_literal_value(arg, result_element_type)} : {result_element_type}"
                )
                scalar_value = f"%{prefix}_scalar"
            else:
                # Scalar variable / descriptor input: reuse its lowered value.
                scalar_code, scalar_value, _st, _se = _lower_tensor_input(
                    arg, prefix, functions, tensor_env, scalar_env
                )
            splat_code = f"""    %{prefix}_empty = tensor.empty() : {result_type}
    %{prefix} = linalg.generic {{
      indexing_maps = [{splat_identity}],
      iterator_types = {splat_iterators}
    }} outs(%{prefix}_empty : {result_type}) {{
    ^bb0(%{prefix}_out: {result_element_type}):
      linalg.yield {scalar_value} : {result_element_type}
    }} -> {result_type}"""
            code = f"{scalar_code}\n{splat_code}" if scalar_code else splat_code
            return code, f"%{prefix}", result_type, result_element_type
        return _lower_tensor_input(
            arg, prefix, functions, tensor_env, scalar_env
        )

    left_code, left_name, left_type, left_element_type = _lower_input(
        node.arrays[0], "left"
    )
    right_code, right_name, right_type, right_element_type = _lower_input(
        node.arrays[1], "right"
    )

    result_rank = node.result_type.rank
    left_rank = _tensor_rank_from_mlir_type(left_type)
    right_rank = _tensor_rank_from_mlir_type(right_type)

    # Broadcasting indexing maps: each input projects from the principal
    # (result) rank down to its own rank, keeping the first k dimensions.
    def _broadcast_map(input_rank: int) -> str:
        if input_rank == result_rank:
            return _identity_affine_map(result_rank)
        if input_rank == 0:
            return _constant_affine_map(result_rank)
        dims = ", ".join(f"d{i}" for i in range(result_rank))
        kept = ", ".join(f"d{i}" for i in range(input_rank))
        return f"affine_map<({dims}) -> ({kept})>"

    left_map = _broadcast_map(left_rank)
    right_map = _broadcast_map(right_rank)
    identity = _identity_affine_map(result_rank)
    iterators = _parallel_iterators(result_rank)
    op_lines = _lower_map_binary_callable_body(
        node.func,
        functions,
        left_name="%left_map_in",
        left_type=left_element_type,
        right_name="%right_map_in",
        right_type=right_element_type,
        result_type=result_element_type,
        scalar_env=scalar_env,
    )

    body = f"""{left_code}
{right_code}
    %map_empty = tensor.empty() : {result_type}
    %mapped = linalg.generic {{
      indexing_maps = [{left_map}, {right_map}, {identity}],
      iterator_types = {iterators}
    }} ins({left_name}, {right_name} : {left_type}, {right_type}) outs(%map_empty : {result_type}) {{
    ^bb0(%left_map_in: {left_element_type}, %right_map_in: {right_element_type}, %out: {result_element_type}):
{op_lines}
    }} -> {result_type}
"""
    return body.rstrip(), "%mapped", result_type


# ---------------------------------------------------------------------------
# Cell map lowering
# ---------------------------------------------------------------------------


def _lower_map_cell_module(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_value, result_type = _lower_map_cell_result(
        node, functions, tensor_env, scalar_env=scalar_env
    )
    builder = _MLIRMainModuleBuilder(result_type, functions=functions)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_map_cell_result(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    *,
    scalar_env: dict[str, _Operand] | None = None,
    prefix: str = "",
) -> tuple[str, str, str]:
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("cell-map lowering requires an array result")
    if len(node.cell_shape) != 1:
        raise RemoraLoweringError("only rank-1 cell maps lower to MLIR so far")
    if node.result_type.rank != len(node.frame_shape):
        raise RemoraLoweringError(
            "only scalar-result cell maps lower to MLIR so far"
        )
    if isinstance(node.func, HIRVar):
        function = functions.get(node.func.name)
        if function is None:
            raise RemoraLoweringError(
                f"unknown cell-map function {node.func.name}"
            )
    elif isinstance(node.func, HIRLambda):
        # Inline let-bound helper arguments (e.g. a named ``dot-patch`` call
        # expands to a HIRLet chain wrapping the fold) so the fold/index
        # routing sees the actual reduction body.
        from remora.lowering.module import _inline_lets

        inlined_body = _inline_lets(node.func.body)
        if inlined_body is None:
            raise RemoraLoweringError("cell-map lambda body cannot be empty")
        function = HIRFunction(
            name="__cell_lambda",
            params=list(node.func.params),
            body=inlined_body,
            return_type=node.func.result_type.result,
        )
    else:
        raise RemoraLoweringError(
            "only lifted lambda or inline lambda cell maps lower to MLIR so far"
        )
    if len(function.params) != 1:
        raise RemoraLoweringError(
            "only unary cell-map functions lower to MLIR so far"
        )

    param_name = function.params[0].name

    if isinstance(function.body, (HIRFold, HIRReduce)):
        return _lower_map_cell_fold_result(
            node, function, param_name, functions, tensor_env,
            scalar_env=scalar_env, prefix=prefix,
        )

    # Cell body = scalar prim-op combining a fold-over-the-cell with a literal
    # (e.g. average pooling: ``(/ (fold + 0.0 row) 4.0)``).  Lower the inner
    # fold to a per-cell scalar, then apply the op elementwise over the frame.
    if (
        isinstance(function.body, HIRPrimOp)
        and len(function.body.args) == 2
        and isinstance(node.result_type, ArrayType)
    ):
        a0, a1 = function.body.args
        if (isinstance(a0, (HIRFold, HIRReduce)) and isinstance(a1, HIRLit)
                and _reduces_param(a0.array, param_name)):
            return _lower_map_cell_fold_scalar_op_result(
                node, function, a0, a1, function.body.op, True,
                param_name, functions, tensor_env, scalar_env=scalar_env, prefix=prefix,
            )
        if (isinstance(a1, (HIRFold, HIRReduce)) and isinstance(a0, HIRLit)
                and _reduces_param(a1.array, param_name)):
            return _lower_map_cell_fold_scalar_op_result(
                node, function, a1, a0, function.body.op, False,
                param_name, functions, tensor_env, scalar_env=scalar_env, prefix=prefix,
            )

    return _lower_map_cell_index_result(
        node, function, param_name, functions, tensor_env,
        scalar_env=scalar_env, prefix=prefix,
    )


def _lower_map_cell_fold_result(
    node: HIRMap | HIRApply,
    function: HIRFunction,
    param_name: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    *,
    scalar_env: dict[str, _Operand] | None = None,
    prefix: str = "",
) -> tuple[str, str, str]:
    body_fold = function.body
    if not isinstance(body_fold.init, HIRLit):
        raise RemoraLoweringError(
            "only literal cell-fold initial values lower to MLIR so far"
        )
    # The cell-map parameter must be reduced somewhere. For nested folds like
    # (fold + 0 (fold + init m)), allow the inner fold to reduce the parameter.
    if not _reduces_param(body_fold.array, param_name):
        raise RemoraLoweringError(
            "cell-map fold must reduce the cell-map parameter (directly or via nested fold)"
        )
    input_remora_type = _expr_result_type(node.array)
    if not isinstance(input_remora_type, ArrayType):
        raise RemoraLoweringError("cell-map input must be an array")
    input_rank = input_remora_type.rank
    frame_rank = len(node.frame_shape)
    if input_rank != frame_rank + len(node.cell_shape):
        raise RemoraLoweringError(
            "cell-map frame and cell ranks do not match input rank"
        )

    input_code, input_name, input_type, input_element_type = (
        _lower_tensor_input(
            node.array,
            _join_prefix(prefix, "input"),
            functions,
            tensor_env,
            scalar_env,
        )
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    if input_element_type != result_element_type:
        raise RemoraLoweringError(
            "cell-map fold element type must match result element type"
        )
    init_value = _literal_value(body_fold.init, result_element_type)
    if not isinstance(body_fold.func, HIRPrimCallable):
        raise RemoraLoweringError(
            "only primitive cell-fold callables lower to MLIR so far"
        )
    producer = body_fold.array
    if isinstance(producer, (HIRMap, HIRApply)):
        # fold + init (map OP p free...) -- a per-element producer map whose
        # result the fold reduces (e.g. dot-product ``fold + 0 (map * p k)``).
        return _lower_cell_fold_producer_result(
            node,
            body_fold,
            producer,
            param_name,
            functions,
            tensor_env,
            input_code,
            input_name,
            input_type,
            input_element_type,
            result_type,
            result_element_type,
            init_value,
            scalar_env=scalar_env,
            prefix=prefix,
        )
    fold_body = _lower_fold_callable_body(
        body_fold.func,
        functions,
        input_name="%in",
        input_type=input_element_type,
        acc_name="%acc",
        acc_type=result_element_type,
        result_type=result_element_type,
        scalar_env=scalar_env,
    )
    map_empty = f"%{_join_prefix(prefix, 'map_empty')}"
    init_name = f"%{_join_prefix(prefix, 'init')}"
    filled = f"%{_join_prefix(prefix, 'filled')}"
    mapped = f"%{_join_prefix(prefix, 'mapped')}"
    body = f"""{input_code}
    {map_empty} = tensor.empty() : {result_type}
    {init_name} = arith.constant {init_value} : {result_element_type}
    {filled} = linalg.fill ins({init_name} : {result_element_type}) outs({map_empty} : {result_type}) -> {result_type}
    {mapped} = linalg.generic {{
      indexing_maps = [{_identity_affine_map(input_rank)}, {_take_first_affine_map(input_rank, frame_rank)}],
      iterator_types = {_map_cell_iterators(frame_rank, len(node.cell_shape))}
    }} ins({input_name} : {input_type}) outs({filled} : {result_type}) {{
    ^bb0(%in: {input_element_type}, %acc: {result_element_type}):
{fold_body}
    }} -> {result_type}
"""
    return body.rstrip(), mapped, result_type


def _lower_map_cell_fold_scalar_op_result(
    node: HIRMap | HIRApply,
    cell_function: HIRFunction,
    fold_arg: HIRFold | HIRReduce,
    lit_arg: HIRLit,
    op: str,
    fold_on_left: bool,
    param_name: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    *,
    scalar_env: dict[str, _Operand] | None = None,
    prefix: str = "",
) -> tuple[str, str, str]:
    """Lower a cell-map whose body is ``op(fold-over-cell, literal)``.

    Lowers the inner fold-over-the-cell to a per-cell scalar tensor via the
    existing cell-fold path, then applies the scalar ``op`` (with the literal)
    elementwise over the frame.  Used by e.g. average pooling
    ``(map (lambda (row) (/ (fold + 0.0 row) C)) grid)``.
    """
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("cell-map fold-scalar-op result must be an array")
    fold_fn = HIRFunction(
        name="__cell_fold",
        params=list(cell_function.params),
        body=fold_arg,
        return_type=node.result_type.element,
    )
    fold_code, fold_name, rt = _lower_map_cell_fold_result(
        node, fold_fn, param_name, functions, tensor_env,
        scalar_env=scalar_env, prefix=_join_prefix(prefix, "cf"),
    )
    elem = type_to_mlir(node.result_type.element)
    arith = _arith_op(op[0], elem)
    litval = _literal_value(lit_arg, elem)
    frame_rank = len(node.frame_shape)
    idmap = _identity_affine_map(frame_rank)
    iters = _parallel_iterators(frame_rank)
    p = prefix
    lit_name = f"%{_join_prefix(p, 'polit')}"
    empty = f"%{_join_prefix(p, 'poempty')}"
    out = f"%{_join_prefix(p, 'po')}"
    v = f"%{_join_prefix(p, 'pov')}"
    r = f"%{_join_prefix(p, 'por')}"
    o = f"%{_join_prefix(p, 'poo')}"
    operands = f"{v}, {lit_name}" if fold_on_left else f"{lit_name}, {v}"
    code = f"""{fold_code}
    {lit_name} = arith.constant {litval} : {elem}
    {empty} = tensor.empty() : {rt}
    {out} = linalg.generic {{
      indexing_maps = [{idmap}, {idmap}],
      iterator_types = {iters}
    }} ins({fold_name} : {rt}) outs({empty} : {rt}) {{
    ^bb0({v}: {elem}, {o}: {elem}):
      {r} = {arith} {operands} : {elem}
      linalg.yield {r} : {elem}
    }} -> {rt}"""
    return code, out, rt


def _lower_cell_fold_producer_result(
    node: HIRMap | HIRApply,
    body_fold: HIRFold | HIRReduce,
    producer: HIRMap | HIRApply,
    param_name: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None,
    input_code: str,
    input_name: str,
    input_type: str,
    input_element_type: str,
    result_type: str,
    result_element_type: str,
    init_value: str,
    *,
    scalar_env: dict[str, _Operand] | None = None,
    prefix: str = "",
) -> tuple[str, str, str]:
    """Lower a cell-fold whose per-element value is a producer map.

    Handles ``fold <f> init (map OP p free...)`` where the inner map is a
    scalar (elementwise) map combining the cell parameter ``p`` with one or
    more cell-shaped free arrays (e.g. the dot-product
    ``fold + 0.0 (map * p k)``).  The cell dimension is the reduction
    dimension; the free arrays are broadcast across the frame.
    """
    resolved_callable = producer.func
    if isinstance(producer.func, HIRVar):
        # Defunctionalization may lift the inline lambda to a named
        # function.  If the function body is a simple HIRPrimOp (like
        # `*` applied to its params), resolve it to a HIRPrimCallable
        # so the cell-fold lowering can process it.
        func = functions.get(producer.func.name)
        if func is not None and isinstance(func.body, HIRPrimOp):
            param_names = {p.name for p in func.params}
            if all(isinstance(a, HIRVar) and a.name in param_names for a in func.body.args):
                # HIRPrimOp uses typed operator names like `*f`, `+i`;
                # HIRPrimCallable uses bare names like `*`, `+`.
                raw_op = func.body.op
                if len(raw_op) > 1 and raw_op[-1] in ("f", "i"):
                    raw_op = raw_op[:-1]
                resolved_callable = HIRPrimCallable(
                    raw_op,
                    tuple(p.type for p in func.params),
                    func.return_type,
                )
    if not isinstance(resolved_callable, HIRPrimCallable):
        raise RemoraLoweringError(
            "only primitive cell-fold producer maps lower to MLIR so far"
        )
    if producer.cell_shape:
        raise RemoraLoweringError(
            "only scalar (elementwise) cell-fold producer maps lower to MLIR so far"
        )

    cell_rank = len(node.cell_shape)
    frame_rank = len(node.frame_shape)
    input_rank = frame_rank + cell_rank

    operands = list(producer.arrays)
    has_section = resolved_callable.left_arg is not None or resolved_callable.right_arg is not None
    if has_section and len(operands) != 1:
        raise RemoraLoweringError(
            "cell-fold producer map sections require exactly one map operand"
        )
    param_positions = [
        i
        for i, a in enumerate(operands)
        if isinstance(a, HIRVar) and a.name == param_name
    ]
    if len(param_positions) != 1:
        raise RemoraLoweringError(
            "cell-fold producer map must reference the cell parameter exactly once"
        )
    param_pos = param_positions[0]

    free_operands = [a for i, a in enumerate(operands) if i != param_pos]
    free_codes: list[str] = []
    free_inputs: list[tuple[str, str, str]] = []  # (name, mlir_type, element_type)
    for fi, operand in enumerate(free_operands):
        op_type = _expr_result_type(operand)
        if not isinstance(op_type, ArrayType) or op_type.rank != cell_rank:
            raise RemoraLoweringError(
                "cell-fold producer free operands must be cell-shaped arrays"
            )
        if type_to_mlir(op_type.element) != result_element_type:
            raise RemoraLoweringError(
                "cell-fold producer element type must match result element type"
            )
        code, name, mtype, etype = _lower_tensor_input(
            operand, _join_prefix(prefix, f"free{fi}"), functions, tensor_env, scalar_env
        )
        free_codes.append(code)
        free_inputs.append((name, mtype, etype))

    if has_section:
        section_lines, section_value = _lower_primitive_callable_result(
            resolved_callable,
            input_name="%in",
            input_type=input_element_type,
            result_type=result_element_type,
            scalar_env=scalar_env,
        )
        elem_line = "\n".join(section_lines)
        elem_value = section_value
    else:
        op = _arith_op(resolved_callable.op, result_element_type)
        sep = ", " if "cmp" in op else " "
        operand_values: list[str] = []
        free_idx = 0
        for i in range(len(operands)):
            if i == param_pos:
                operand_values.append("%in")
            else:
                operand_values.append(f"%in_free{free_idx}")
                free_idx += 1
        # When the callable has fewer operands than the operation needs
        # (e.g. `(lambda (x) (* x x))` — 1 map operand, binary `*`),
        # replicate the available operands.
        if isinstance(resolved_callable, HIRPrimCallable) and len(operand_values) == 1:
            arity = 1
            if resolved_callable.op in {"+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!="}:
                arity = 2
            if arity > len(operand_values):
                operand_values = operand_values * arity
        elem_line = (
            f"      %elem = {op}{sep}" + ", ".join(operand_values) + f" : {result_element_type}"
        )
        elem_value = "%elem"

    fold_body = _lower_fold_callable_body(
        body_fold.func,
        functions,
        input_name=elem_value,
        input_type=result_element_type,
        acc_name="%acc",
        acc_type=result_element_type,
        result_type=result_element_type,
        scalar_env=scalar_env,
    )

    ins_names = [input_name] + [n for n, _, _ in free_inputs]
    ins_types = [input_type] + [t for _, t, _ in free_inputs]
    maps = [_identity_affine_map(input_rank)]
    for _ in free_inputs:
        maps.append(_take_last_affine_map(input_rank, cell_rank))
    maps.append(_take_first_affine_map(input_rank, frame_rank))

    bb_args = [f"%in: {input_element_type}"]
    for fi, (_, _, etype) in enumerate(free_inputs):
        bb_args.append(f"%in_free{fi}: {etype}")
    bb_args.append(f"%acc: {result_element_type}")

    free_code_block = "\n".join(c for c in free_codes if c)
    map_empty = f"%{_join_prefix(prefix, 'map_empty')}"
    init_name = f"%{_join_prefix(prefix, 'init')}"
    filled = f"%{_join_prefix(prefix, 'filled')}"
    mapped = f"%{_join_prefix(prefix, 'mapped')}"
    body = f"""{input_code}
{free_code_block}
    {map_empty} = tensor.empty() : {result_type}
    {init_name} = arith.constant {init_value} : {result_element_type}
    {filled} = linalg.fill ins({init_name} : {result_element_type}) outs({map_empty} : {result_type}) -> {result_type}
    {mapped} = linalg.generic {{
      indexing_maps = [{", ".join(maps)}],
      iterator_types = {_map_cell_iterators(frame_rank, cell_rank)}
    }} ins({", ".join(ins_names)} : {", ".join(ins_types)}) outs({filled} : {result_type}) {{
    ^bb0({", ".join(bb_args)}):
{elem_line}
{fold_body}
    }} -> {result_type}
"""
    return body.rstrip(), mapped, result_type


def _reduces_param(expr: HIRExpr, param_name: str) -> bool:
    """Check if *expr* directly or indirectly reduces *param_name*."""
    if isinstance(expr, HIRVar) and expr.name == param_name:
        return True
    if isinstance(expr, (HIRFold, HIRReduce)):
        return _reduces_param(expr.array, param_name)
    if isinstance(expr, (HIRMap, HIRApply)):
        # A fold whose array is a per-element map (e.g. dot-product
        # ``fold + 0 (map * p k)``) reduces the param when the map references it.
        return any(_reduces_param(a, param_name) for a in expr.arrays)
    return False


def _collect_cell_indices(expr: HIRExpr, param_name: str) -> set[int]:
    """Collect all literal index values used on *param_name* in *expr*."""
    indices: set[int] = set()
    if (
        isinstance(expr, HIRIndex)
        and isinstance(expr.array, HIRVar)
        and expr.array.name == param_name
    ):
        for idx in expr.indices:
            if isinstance(idx, HIRLit):
                indices.add(int(idx.value))
        return indices
    for field_name in (
        "condition",
        "then_branch",
        "else_branch",
        "value",
        "body",
        "init",
        "array",
    ):
        child = getattr(expr, field_name, None)
        if isinstance(child, HIRExpr):
            indices |= _collect_cell_indices(child, param_name)
    if isinstance(expr, (HIRPrimOp, HIRMap, HIRApply)):
        for child_expr in getattr(expr, "args", []):
            if isinstance(child_expr, HIRExpr):
                indices |= _collect_cell_indices(child_expr, param_name)
        for child_expr in getattr(expr, "arrays", []):
            if isinstance(child_expr, HIRExpr):
                indices |= _collect_cell_indices(child_expr, param_name)
    if isinstance(expr, (HIRFold, HIRReduce)):
        if isinstance(expr.init, HIRExpr):
            indices |= _collect_cell_indices(expr.init, param_name)
        if isinstance(expr.array, HIRExpr):
            indices |= _collect_cell_indices(expr.array, param_name)
        if isinstance(expr.func, HIRVar):
            pass
    if isinstance(expr, HIRCall):
        for arg in expr.args:
            if isinstance(arg, HIRExpr):
                indices |= _collect_cell_indices(arg, param_name)
    if isinstance(expr, HIRLet):
        indices |= _collect_cell_indices(expr.value, param_name)
        indices |= _collect_cell_indices(expr.body, param_name)
    if isinstance(expr, HIRIndex):
        if isinstance(expr.array, HIRExpr):
            indices |= _collect_cell_indices(expr.array, param_name)
        for idx in expr.indices:
            if isinstance(idx, HIRExpr):
                indices |= _collect_cell_indices(idx, param_name)
    return indices


def _lower_map_cell_index_result(
    node: HIRMap | HIRApply,
    function: HIRFunction,
    param_name: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    *,
    scalar_env: dict[str, _Operand] | None = None,
    prefix: str = "",
) -> tuple[str, str, str]:
    cell_indices = sorted(
        _collect_cell_indices(function.body, param_name)
    )
    if not cell_indices:
        raise RemoraLoweringError(
            "cell-map body must reference the cell parameter via indexing or fold"
        )
    cell_size = cell_indices[-1] + 1
    input_remora_type = _expr_result_type(node.array)
    if not isinstance(input_remora_type, ArrayType):
        raise RemoraLoweringError("cell-map input must be an array")
    input_rank = input_remora_type.rank
    frame_rank = len(node.frame_shape)
    if input_rank != frame_rank + 1:
        raise RemoraLoweringError("cell-map requires rank-1 cells")
    if cell_size > input_remora_type.shape[frame_rank].value:
        raise RemoraLoweringError("cell index out of bounds for cell size")

    input_code, input_name, input_type, input_element_type = (
        _lower_tensor_input(
            node.array,
            _join_prefix(prefix, "input"),
            functions,
            tensor_env,
            scalar_env,
        )
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)

    cell_maps = [
        _cell_element_affine_map(frame_rank, pos)
        for pos in range(cell_size)
    ]
    output_map = _identity_affine_map(frame_rank)
    ins_types = f"{input_type}"
    ins_names = input_name
    if cell_size > 1:
        ins_types = ", ".join([input_type] * cell_size)
        ins_names = ", ".join([input_name] * cell_size)
    map_str = ", ".join([*cell_maps, output_map])
    iterators = _parallel_iterators(frame_rank)
    cell_param_names = [f"%cell_{pos}" for pos in range(cell_size)]

    emitter = _RegionEmitter(
        input_name="", input_type="", functions=functions
    )
    env: dict[str, _Operand] = dict(scalar_env or {})
    for pos in range(cell_size):
        env[f"{param_name}_{pos}"] = _Operand(
            cell_param_names[pos], [], input_element_type
        )

    rewritten_body = _rewrite_cell_indices(function.body, param_name, env)
    value = emitter.emit_expr(rewritten_body, env)
    cast_lines = _cast_if_needed(
        value.value, value.type, result_element_type, "%cell_result_cast"
    )
    result_value_name = (
        "%cell_result_cast" if cast_lines else value.value
    )
    region_body = "\n".join(
        [
            *emitter.lines,
            *cast_lines,
            f"      linalg.yield {result_value_name} : {result_element_type}",
        ]
    )

    map_empty = f"%{_join_prefix(prefix, 'map_empty')}"
    mapped = f"%{_join_prefix(prefix, 'mapped')}"
    body = f"""{input_code}
    {map_empty} = tensor.empty() : {result_type}
    {mapped} = linalg.generic {{
      indexing_maps = [{map_str}],
      iterator_types = {iterators}
    }} ins({ins_names} : {ins_types}) outs({map_empty} : {result_type}) {{
    ^bb0({', '.join(f'{name}: {input_element_type}' for name in cell_param_names)}, %out: {result_element_type}):
{region_body}
    }} -> {result_type}
"""
    return body.rstrip(), mapped, result_type


def _rewrite_cell_indices(
    expr: HIRExpr,
    param_name: str,
    env: dict[str, _Operand],
) -> HIRExpr:
    """Replace HIRIndex(cell_param, [HIRLit(pos)]) with HIRVar(param_name_pos)."""
    if (
        isinstance(expr, HIRIndex)
        and isinstance(expr.array, HIRVar)
        and expr.array.name == param_name
    ):
        if len(expr.indices) == 1 and isinstance(expr.indices[0], HIRLit):
            pos = int(expr.indices[0].value)
            var_name = f"{param_name}_{pos}"
            if var_name in env:
                return HIRVar(var_name, expr.result_type)
    if isinstance(expr, (HIRPrimOp, HIRMap, HIRApply)):
        return HIRPrimOp(
            expr.op,
            [
                _rewrite_cell_indices(arg, param_name, env)
                for arg in expr.args
            ],
            expr.result_type,
        )
    if isinstance(expr, HIRLet):
        return HIRLet(
            expr.name,
            expr.value_type,
            _rewrite_cell_indices(expr.value, param_name, env),
            _rewrite_cell_indices(expr.body, param_name, env),
            expr.result_type,
        )
    if isinstance(expr, HIRIf):
        return HIRIf(
            _rewrite_cell_indices(expr.condition, param_name, env),
            _rewrite_cell_indices(expr.then_branch, param_name, env),
            _rewrite_cell_indices(expr.else_branch, param_name, env),
            expr.result_type,
        )
    if isinstance(expr, HIRCast):
        return HIRCast(
            _rewrite_cell_indices(expr.value, param_name, env),
            expr.from_type,
            expr.to_type,
            expr.result_type,
        )
    return expr


# ---------------------------------------------------------------------------
# Fold lowering
# ---------------------------------------------------------------------------


def _lower_fold_module(
    node: HIRFold | HIRReduce | HIRFoldRight,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_value, result_type = _lower_fold_result(
        node, functions, tensor_env, scalar_env=scalar_env
    )
    called = _collect_called_functions(node, functions)
    builder = _MLIRMainModuleBuilder(result_type, functions=called)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_fold_result(
    node: HIRFold | HIRReduce,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    *,
    prefix: str = "",
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str]:
    if isinstance(node.result_type, ArrayType):
        input_type = _expr_result_type(node.array)
        if isinstance(input_type, ArrayType) and input_type.rank >= 2:
            return _lower_array_fold_result(
                node, functions, tensor_env, prefix=prefix, scalar_env=scalar_env
            )
        return _lower_state_fold_result(
            node, functions, tensor_env, prefix=prefix, scalar_env=scalar_env
        )
    return _lower_scalar_fold_result(
        node, functions, tensor_env, scalar_env=scalar_env
    )


def _lower_scalar_fold_module(
    node: HIRFold | HIRReduce,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_value, result_type = _lower_scalar_fold_result(
        node, functions, tensor_env, scalar_env=scalar_env
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_scalar_fold_result(
    node: HIRFold | HIRReduce,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    *,
    scalar_env: dict[str, _Operand] | None = None,
    prefix: str = "",
) -> tuple[str, str, str]:
    input_code, input_name, input_type, input_element_type = _lower_fold_input(
        node.array,
        functions,
        _join_prefix(prefix, "fold_input"),
        tensor_env=tensor_env,
        scalar_env=scalar_env,
    )
    result_type = type_to_mlir(node.result_type)
    # Promote i32 accumulator to i64 for primitive fold callables
    # to avoid overflow on large reductions. HIRLambda bodies retain
    # their original types since the lambda operates in the Remora type system.
    acc_type = result_type
    truncate = False
    if result_type == "i32" and isinstance(node.func, HIRPrimCallable):
        acc_type = "i64"
        truncate = True

    init_code, init_value = _lower_scalar_value_for_fold_init(
        node.init,
        acc_type,
        functions=functions,
        env=scalar_env or {},
        result_prefix=_join_prefix(prefix, "init"),
        ssa_prefix=prefix,
    )
    fold_body = _lower_fold_callable_body(
        node.func,
        functions,
        input_name="%in",
        input_type=input_element_type,
        acc_name="%acc",
        acc_type=acc_type,
        result_type=acc_type,
        scalar_env=scalar_env,
    )

    trunc_block = ""
    if truncate:
        body = f"""{input_code}
{init_code}
    %init = tensor.from_elements {init_value} : tensor<{acc_type}>
    %folded = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = [\"reduction\"]
    }} ins({input_name} : {input_type}) outs(%init : tensor<{acc_type}>) {{
    ^bb0(%in: {input_element_type}, %acc: {acc_type}):
{fold_body}
    }} -> tensor<{acc_type}>
    %wide = tensor.extract %folded[] : tensor<{acc_type}>
    %result = arith.trunci %wide : {acc_type} to {result_type}
"""
    else:
        body = f"""{input_code}
{init_code}
    %init = tensor.from_elements {init_value} : tensor<{acc_type}>
    %folded = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = [\"reduction\"]
    }} ins({input_name} : {input_type}) outs(%init : tensor<{acc_type}>) {{
    ^bb0(%in: {input_element_type}, %acc: {acc_type}):
{fold_body}
    }} -> tensor<{acc_type}>
    %result = tensor.extract %folded[] : tensor<{acc_type}>
"""
    return body.rstrip(), "%result", result_type


def _collect_called_functions(node, functions: dict) -> dict:
    """Collect HIR functions that are directly called (via HIRCall) from the node."""
    names: set[str] = set()
    def _walk(expr):
        if isinstance(expr, HIRCall):
            names.add(expr.func_name)
        for attr in ('func', 'body', 'value', 'array', 'init', 'condition',
                     'then_branch', 'else_branch', 'left', 'right'):
            child = getattr(expr, attr, None)
            if child is not None and hasattr(child, '__class__') and child.__class__.__module__.startswith('remora'):
                _walk(child)
        for attr in ('args', 'arrays', 'elements'):
            children = getattr(expr, attr, None)
            if isinstance(children, (list, tuple)):
                for c in children:
                    if hasattr(c, '__class__') and c.__class__.__module__.startswith('remora'):
                        _walk(c)
    _walk(node)
    if isinstance(node, (HIRFold, HIRReduce)) and isinstance(node.func, HIRVar):
        func = functions.get(node.func.name)
        if func is not None:
            _walk(func.body)
    return {n: functions[n] for n in names if n in functions}


def _inline_hir_calls(expr, functions: dict):
    """Inline HIRCall nodes and simplify trivial let bindings."""
    if isinstance(expr, HIRCall):
        func = functions.get(expr.func_name)
        if func is not None:
            body = func.body
            for param, arg in reversed(list(zip(func.params, expr.args))):
                body = HIRLet(param.name, param.type, _inline_hir_calls(arg, functions), body, expr.result_type)
            return _inline_hir_calls(body, functions)
        return expr
    if isinstance(expr, HIRLet):
        value = _inline_hir_calls(expr.value, functions)
        body = _inline_hir_calls(expr.body, functions)
        if isinstance(value, HIRVar):
            return _subst_var(body, expr.name, value)
        return HIRLet(expr.name, expr.value_type, value, body, expr.result_type)
    if isinstance(expr, HIRApply):
        return HIRApply(expr.frame_shape, expr.cell_shape, expr.func,
                        [_inline_hir_calls(a, functions) for a in expr.arrays], expr.result_type)
    if isinstance(expr, HIRMap):
        return HIRMap(expr.frame_shape, expr.cell_shape, expr.func,
                      [_inline_hir_calls(a, functions) for a in expr.arrays], expr.result_type)
    if isinstance(expr, HIRPrimOp):
        return HIRPrimOp(expr.op, [_inline_hir_calls(a, functions) for a in expr.args], expr.result_type)
    return expr


def _subst_var(expr, name: str, replacement):
    """Substitute all occurrences of HIRVar(name) with replacement."""
    if isinstance(expr, HIRVar) and expr.name == name:
        return replacement
    if isinstance(expr, HIRApply):
        return HIRApply(expr.frame_shape, expr.cell_shape, expr.func,
                        [_subst_var(a, name, replacement) for a in expr.arrays], expr.result_type)
    if isinstance(expr, HIRMap):
        func = expr.func
        if isinstance(func, HIRLambda):
            if name not in [p.name for p in func.params]:
                func = HIRLambda(func.params, _subst_var(func.body, name, replacement), func.result_type)
        return HIRMap(expr.frame_shape, expr.cell_shape, func,
                      [_subst_var(a, name, replacement) for a in expr.arrays], expr.result_type)
    if isinstance(expr, HIRLet):
        value = _subst_var(expr.value, name, replacement)
        body = expr.body if expr.name == name else _subst_var(expr.body, name, replacement)
        return HIRLet(expr.name, expr.value_type, value, body, expr.result_type)
    if isinstance(expr, HIRPrimOp):
        return HIRPrimOp(expr.op, [_subst_var(a, name, replacement) for a in expr.args], expr.result_type)
    return expr


def _lower_state_fold_result(
    node: HIRFold | HIRReduce,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    *,
    prefix: str = "",
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str]:
    """Lower a fold with array-valued accumulator over a rank-1 input.

    Uses ``scf.for`` with ``iter_args`` carrying the accumulator tensor.
    Each iteration applies the fold body to (carry, element) and yields
    the new carry.
    """
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("state fold requires an array result type")

    result_mlir = type_to_mlir(node.result_type)
    reduction_dim = node.reduction_dim
    N_val = int(reduction_dim.value)

    init_code, init_name, init_type, init_elem = _lower_tensor_input(
        node.init, _join_prefix(prefix, "sf_init"), functions, tensor_env, scalar_env,
    )

    lam_params: list[HIRParam] = []
    lam_body: HIRExpr | None = None
    callable_expr = node.func

    if isinstance(callable_expr, HIRLambda):
        lam_params = callable_expr.params
        lam_body = callable_expr.body
    elif isinstance(callable_expr, HIRVar):
        func = functions.get(callable_expr.name)
        if func is not None:
            lam_params = func.params
            lam_body = func.body
    if lam_body is None:
        raise RemoraLoweringError(
            f"state fold callable must be a lambda or named function "
            f"(got {type(callable_expr).__name__})"
        )

    p = _join_prefix(prefix, "sf")
    cN = f"%{_join_prefix(p, 'N')}"
    c0 = f"%{_join_prefix(p, 'c0')}"
    c1 = f"%{_join_prefix(p, 'c1')}"
    idx = f"%{_join_prefix(p, 'i')}"
    result = f"%{_join_prefix(p, 'result')}"

    if not isinstance(node.result_type, ArrayType) or node.result_type.rank != 1:
        raise RemoraLoweringError("state fold scalar decomposition supports rank-1 results only")
    K = int(node.result_type.shape[0].value)

    init_scalars: list[str] = []
    init_extract_code = ""
    for k in range(K):
        sc_name = f"%{_join_prefix(p, f'init_s{k}')}"
        ck_name = f"%{_join_prefix(p, f'ck{k}')}"
        init_extract_code += f"    {ck_name} = arith.constant {k} : index\n"
        init_extract_code += f"    {sc_name} = tensor.extract {init_name}[{ck_name}] : {init_type}\n"
        init_scalars.append(sc_name)

    carry_scalars = [f"%{_join_prefix(p, f'carry_s{k}')}" for k in range(K)]
    iter_arg_str = ", ".join(f"{cs} = {ins}" for cs, ins in zip(carry_scalars, init_scalars))
    type_str = ", ".join(init_elem for _ in range(K))

    carry_tensor = f"%{_join_prefix(p, 'carry_tensor')}"
    carry_elems_str = ", ".join(carry_scalars)
    reconstruct_code = f"      {carry_tensor} = tensor.from_elements {carry_elems_str} : {result_mlir}"

    loop_tenv = dict(tensor_env or {})
    loop_senv = dict(scalar_env or {})

    if len(lam_params) >= 1:
        acc_param = lam_params[0].name
        loop_tenv[acc_param] = _TensorValue(carry_tensor, result_mlir, init_elem)

    cast_var = f"%{_join_prefix(p, 'i32')}"
    loop_body_prefix = f"      {cast_var} = arith.index_cast {idx} : index to i32"
    if len(lam_params) >= 2:
        elem_param = lam_params[1].name
        loop_senv[elem_param] = _Operand(cast_var, [], "i32")

    lines: list[str] = []
    try:
        body_code, new_carry_val, new_carry_type, _ = _lower_tensor_input(
            lam_body, _join_prefix(p, "body"), functions,
            loop_tenv, loop_senv,
        )
        lines.append(body_code)
    except RemoraLoweringError:
        new_carry_val, new_carry_type = _lower_body_in_loop(
            lam_body, lines, _join_prefix(p, "body"), functions,
            loop_tenv, loop_senv, _next_uid=0,
        )

    new_scalars: list[str] = []
    extract_code = ""
    for k in range(K):
        ns = f"%{_join_prefix(p, f'new_s{k}')}"
        ek = f"%{_join_prefix(p, f'ek{k}')}"
        extract_code += f"      {ek} = arith.constant {k} : index\n"
        extract_code += f"      {ns} = tensor.extract {new_carry_val}[{ek}] : {new_carry_type}\n"
        new_scalars.append(ns)

    new_scalars_str = ", ".join(new_scalars)
    loop_body = loop_body_prefix + "\n" + reconstruct_code + "\n" + "\n".join(lines) + "\n" + extract_code

    result_scalars = [f"%{_join_prefix(p, f'result_s{k}')}" for k in range(K)]
    result_scalars_str = ", ".join(result_scalars)

    code = f"""{init_code}
{init_extract_code}\
    {cN} = arith.constant {N_val} : index
    {c0} = arith.constant 0 : index
    {c1} = arith.constant 1 : index
    {result_scalars_str} = scf.for {idx} = {c0} to {cN} step {c1} iter_args({iter_arg_str}) -> ({type_str}) {{
{loop_body}\
      scf.yield {new_scalars_str} : {type_str}
    }}
    {result} = tensor.from_elements {result_scalars_str} : {result_mlir}"""
    return code.rstrip(), result, result_mlir


def _lower_array_fold_module(
    node: HIRFold | HIRReduce,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_value, result_type = _lower_array_fold_result(
        node, functions, tensor_env, scalar_env=scalar_env
    )
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render(result_value)


def _lower_array_fold_result(
    node: HIRFold | HIRReduce,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    *,
    prefix: str = "",
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str]:
    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError(
            "array fold lowering requires an array result"
        )

    input_remora_type = _expr_result_type(node.array)
    if (
        not isinstance(input_remora_type, ArrayType)
        or input_remora_type.rank < 2
    ):
        raise RemoraLoweringError(
            "array-cell fold lowering requires rank-2 or rank-3 input"
        )

    input_code, input_name, input_type, input_element_type = _lower_fold_input(
        node.array,
        functions,
        _join_prefix(prefix, "fold_input"),
        tensor_env=tensor_env,
        scalar_env=scalar_env,
    )
    init_code, init_name, init_type, init_element_type = _lower_tensor_input(
        node.init,
        _join_prefix(prefix, "fold_init"),
        functions,
        tensor_env,
        scalar_env,
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    if init_type != result_type:
        raise RemoraLoweringError(
            "array-cell fold init type must match result type"
        )
    if (
        input_element_type != result_element_type
        or init_element_type != result_element_type
    ):
        raise RemoraLoweringError(
            "array-cell fold element types must match"
        )

    rank = input_remora_type.rank
    fold_body = _lower_fold_callable_body(
        node.func,
        functions,
        input_name="%in",
        input_type=input_element_type,
        acc_name="%acc",
        acc_type=result_element_type,
        result_type=result_element_type,
        scalar_env=scalar_env,
    )

    folded = f"%{_join_prefix(prefix, 'folded')}"
    body = f"""{input_code}
{init_code}
    {folded} = linalg.generic {{
      indexing_maps = [{_identity_affine_map(rank)}, {_drop_first_affine_map(rank)}],
      iterator_types = {_fold_iterators(rank)}
    }} ins({input_name} : {input_type}) outs({init_name} : {result_type}) {{
    ^bb0(%in: {input_element_type}, %acc: {result_element_type}):
{fold_body}
    }} -> {result_type}
"""
    return body.rstrip(), folded, result_type


def _lower_fold_input(
    node: HIRExpr,
    functions: dict[str, HIRFunction],
    prefix: str = "",
    *,
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str, str]:
    if isinstance(node, (HIRIota, HIRArrayLit, HIRWithShape)):
        return _lower_tensor_input(
            node, _join_prefix(prefix, "input"), functions, tensor_env, scalar_env
        )
    if isinstance(node, HIRVar):
        return _lower_tensor_input(
            node, _join_prefix(prefix, "input"), functions, tensor_env, scalar_env
        )

    if isinstance(
        node,
        (
            HIRIndex,
            HIRSlice,
            HIRTranspose,
            HIRReshape,
            HIRRavel,
            HIRTake,
            HIRDrop,
            HIRSubarray,
            HIRRotate,
        ),
    ):
        return _lower_tensor_input(
            node, _join_prefix(prefix, "input"), functions, tensor_env, scalar_env
        )

    if isinstance(node, HIRAppend):
        return _lower_tensor_input(
            node, _join_prefix(prefix, "input"), functions, tensor_env, scalar_env
        )

    if isinstance(node, (HIRMap, HIRApply)):
        if node.cell_shape:
            code, name, result_type = _lower_map_cell_result(
                node, functions, tensor_env, scalar_env=scalar_env,
                prefix=prefix,
            )
            if not isinstance(node.result_type, ArrayType):
                raise RemoraLoweringError("cell-map fold input must be an array")
            return code, name, result_type, type_to_mlir(node.result_type.element)
        if not isinstance(node.result_type, ArrayType):
            raise RemoraLoweringError(
                "map fold input must have array type"
            )
        if len(node.arrays) == 2:
            return _lower_binary_map_fold_input(
                node, functions, prefix, tensor_env, scalar_env
            )
        if len(node.arrays) != 1:
            raise RemoraLoweringError(
                "only unary and binary scalar maps lower to fold MLIR so far"
            )

        # ---- compound-body detection ----
        # When a map callable body contains tensor-level operations (fold,
        # index, nested map) the scalar emitter running inside a
        # linalg.generic block cannot lower them.  We switch to a
        # scf.for-based path that allows full tensor operations.
        if isinstance(node.func, HIRLambda):
            if _body_needs_tensor_lowering(node.func.body):
                return _lower_map_body_with_loops(
                    node, functions, prefix,
                    tensor_env=tensor_env, scalar_env=scalar_env,
                )

        input_code, input_name, input_type, input_element_type = (
            _lower_fold_input(
                node.array,
                functions,
                _join_prefix(prefix, "input"),
                tensor_env=tensor_env,
                scalar_env=scalar_env,
            )
        )
        map_type = type_to_mlir(node.result_type)
        map_element_type = type_to_mlir(node.result_type.element)
        rank = node.result_type.rank
        identity = _identity_affine_map(rank)
        iterators = _parallel_iterators(rank)
        map_empty = f"%{_join_prefix(prefix, 'map_empty')}"
        mapped = f"%{_join_prefix(prefix, 'mapped')}"
        map_in = f"%{_join_prefix(prefix, 'map_in')}"
        map_out = f"%{_join_prefix(prefix, 'map_out')}"
        map_body = _lower_map_callable_body(
            node.func,
            functions,
            input_name=map_in,
            input_type=input_element_type,
            result_type=map_element_type,
            scalar_env=scalar_env,
        )
        code = f"""{input_code}
    {map_empty} = tensor.empty() : {map_type}
    {mapped} = linalg.generic {{
      indexing_maps = [{identity}, {identity}],
      iterator_types = {iterators}
    }} ins({input_name} : {input_type}) outs({map_empty} : {map_type}) {{
    ^bb0({map_in}: {input_element_type}, {map_out}: {map_element_type}):
{map_body}
    }} -> {map_type}"""
        return code, mapped, map_type, map_element_type

    if isinstance(node, (HIRFold, HIRReduce)) and isinstance(
        node.result_type, ArrayType
    ):
        code, name, result_type = _lower_fold_result(
            node, functions, tensor_env, prefix=prefix, scalar_env=scalar_env
        )
        return code, name, result_type, type_to_mlir(node.result_type.element)

    # Fallback: lower as a general tensor input
    return _lower_tensor_input(
        node, _join_prefix(prefix, "fold_input"), functions, tensor_env, scalar_env
    )


def _body_needs_tensor_lowering(body: HIRExpr) -> bool:
    """Return True if *body* contains operations the scalar emitter cannot emit
    inside a ``linalg.generic`` region body.

    Intrinsic compound nodes: ``HIRFold``, ``HIRReduce``, ``HIRIndex``.
    ``HIRMap`` / ``HIRApply`` are only compound when their *callable body*
    or their array arguments are compound.
    ``HIRLet`` chains are traversed so that intermediate bindings hoist
    complex sub-expressions out of the final callable body.
    """
    if isinstance(body, (HIRFold, HIRReduce, HIRIndex)):
        return True
    if isinstance(body, (HIRVar, HIRLit, HIRCast)):
        return False
    if isinstance(body, HIRLet):
        return _body_needs_tensor_lowering(body.body) or _body_needs_tensor_lowering(
            body.value
        )
    if isinstance(body, HIRIf):
        return any(
            _body_needs_tensor_lowering(c)
            for c in (body.condition, body.then_branch, body.else_branch)
        )
    if isinstance(body, (HIRMap, HIRApply)):
        if isinstance(body.func, HIRLambda):
            if _body_needs_tensor_lowering(body.func.body):
                return True
        return any(_body_needs_tensor_lowering(a) for a in body.arrays)
    if isinstance(body, HIRPrimOp):
        return any(_body_needs_tensor_lowering(a) for a in body.args)
    if isinstance(body, HIRCall):
        return False
    return False


def _lower_map_body_with_loops(
    node: HIRMap,
    functions: dict[str, HIRFunction],
    prefix: str,
    *,
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str, str]:
    """Lower a scalar HIRMap whose body needs tensor access.

    Uses ``scf.for`` loops so sub-expressions can freely emit
    ``tensor.extract``, ``tensor.extract_slice``, and ``linalg.generic``
    reductions — operations that are illegal inside a ``linalg.generic``
    block body.
    """
    result_mlir = type_to_mlir(node.result_type)
    elem_mlir = type_to_mlir(node.result_type.element)
    frame = node.frame_shape

    if len(frame) != 1:
        raise RemoraLoweringError(
            "loop-based map lowering supports only rank-1 frames"
        )
    N_val = frame[0].value
    outer = node.result_type.shape[1].value if node.result_type.rank > 1 else 1

    # Lower the array being mapped
    arr_code, arr_name, arr_type, arr_elem = _lower_tensor_input(
        node.array, _join_prefix(prefix, "lp_arr"), functions,
        tensor_env, scalar_env,
    )

    lam = node.func
    if not isinstance(lam, HIRLambda):
        raise RemoraLoweringError("loop-based map needs a HIRLambda callable")
    param = lam.params[0].name
    body = lam.body

    # Build scf.for
    cN = f"%{_join_prefix(prefix, 'lp_N')}"
    c0 = f"%{_join_prefix(prefix, 'lp_c0')}"
    c1 = f"%{_join_prefix(prefix, 'lp_c1')}"
    empty = f"%{_join_prefix(prefix, 'lp_empty')}"
    zero = f"%{_join_prefix(prefix, 'lp_zero')}"
    filled = f"%{_join_prefix(prefix, 'lp_fill')}"
    result = f"%{_join_prefix(prefix, 'lp_result')}"
    idx = f"%{_join_prefix(prefix, 'lp_i')}"
    acc = f"%{_join_prefix(prefix, 'lp_acc')}"

    # Scalar environment for the loop body.  scf.for induction variables
    # are of type ``index``; we cast to ``i32`` because that is the Remora
    # integer type used throughout the lowering.
    loop_env = dict(scalar_env or {})
    cast_var = f"%{_join_prefix(prefix, 'lp_i32')}"

    loop_body_prefix = f"      {cast_var} = arith.index_cast {idx} : index to {arr_elem}"

    loop_env[param] = _Operand(cast_var, [], arr_elem)

    lines: list[str] = []
    body_val, body_mlir = _lower_body_in_loop(
        body, lines, prefix, functions, tensor_env, loop_env,
        _next_uid=0,
    )

    updated = f"%{_join_prefix(prefix, 'lp_updated')}"
    rank = node.result_type.rank
    if rank > 1:
        # Write the rank-(rank-1) cell produced for row {idx} into the output.
        # Offsets/sizes/strides must match the full result rank, not just 2.
        cell_dims = [str(d.value) for d in node.result_type.shape[1:]]
        offsets = ", ".join([idx] + ["0"] * (rank - 1))
        sizes = ", ".join(["1"] + cell_dims)
        strides = ", ".join(["1"] * rank)
        insert = (
            f"      {updated} = tensor.insert_slice {body_val}"
            f" into {acc}[{offsets}] [{sizes}] [{strides}]"
            f" : {body_mlir} into {result_mlir}"
        )
    else:
        insert = (
            f"      {updated} = tensor.insert {body_val}"
            f" into {acc}[{idx}] : {result_mlir}"
        )

    loop_body = loop_body_prefix + "\n" + "\n".join(lines) + "\n" + insert

    # The fill value initialises the output accumulator; its literal must match
    # the element type (a float literal is invalid for an integer/bool type).
    zero_lit = "0.000000e+00" if elem_mlir.startswith("f") else "0"
    code = f"""{arr_code}
    {cN} = arith.constant {N_val} : index
    {c0} = arith.constant 0 : index
    {c1} = arith.constant 1 : index
    {zero} = arith.constant {zero_lit} : {elem_mlir}
    {empty} = tensor.empty() : {result_mlir}
    {filled} = linalg.fill ins({zero} : {elem_mlir}) outs({empty} : {result_mlir}) -> {result_mlir}
    {result} = scf.for {idx} = {c0} to {cN} step {c1} iter_args({acc} = {filled}) -> {result_mlir} {{
{loop_body}
      scf.yield {updated} : {result_mlir}
    }}"""
    return code, result, result_mlir, elem_mlir


def _lower_body_in_loop(
    expr: HIRExpr,
    lines: list[str],
    prefix: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv,
    scalar_env: dict[str, _Operand],
    _next_uid: int = 0,
) -> tuple[str, str]:
    """Recursively lower *expr* inside a ``scf.for`` loop body.

    * HIRLet bindings lower the value first, register it in ``tensor_env`` /
      ``scalar_env``, then lower the body.
    * Scalar HIRFold / HIRReduce delegates to ``_lower_scalar_fold_result``.
    * Everything else goes through ``_lower_tensor_input`` (which has full
      access to tensor operations inside the loop).  If that fails, the
      scalar emitter is tried as a fallback for simple scalar expressions.
    """
    from remora.types import ArrayType, ScalarType

    if isinstance(expr, HIRLet):
        val_name, val_mlir = _lower_body_in_loop(
            expr.value, lines, _join_prefix(prefix, f"lv{_next_uid}"), functions,
            tensor_env, scalar_env, _next_uid + 1,
        )
        if _is_scalar_type(expr.value_type) or (
            isinstance(expr.value_type, ArrayType) and expr.value_type.rank == 0
        ):
            scalar_env[expr.name] = _Operand(val_name, [], val_mlir)
        # Compute correct element type for tensor_env registration
        if isinstance(expr.value_type, ArrayType) and expr.value_type.rank > 0:
            elem_mlir = type_to_mlir(expr.value_type.element)
        else:
            elem_mlir = val_mlir
        tensor_env[expr.name] = _TensorValue(val_name, val_mlir, elem_mlir)
        return _lower_body_in_loop(
            expr.body, lines, _join_prefix(prefix, f"lb{_next_uid}"), functions,
            tensor_env, scalar_env, _next_uid + 1,
        )

    if isinstance(expr, (HIRFold, HIRReduce)) and _is_scalar_type(expr.result_type):
        fold_prefix = _join_prefix(prefix, f"sf{_next_uid}")
        fold_code, fold_val, fold_type = _lower_scalar_fold_result(
            expr, functions, tensor_env, scalar_env=scalar_env,
            prefix=fold_prefix,
        )
        lines.append(fold_code)
        return fold_val, fold_type

    if isinstance(expr, HIRCall):
        func = functions.get(expr.func_name)
        if func is not None:
            call_prefix = _join_prefix(prefix, f"cl{_next_uid}")
            arg_vals: list[str] = []
            arg_types: list[str] = []
            for i, arg in enumerate(expr.args):
                av, at = _lower_body_in_loop(
                    arg, lines, _join_prefix(call_prefix, f"a{i}"), functions,
                    tensor_env, scalar_env, _next_uid + i + 1,
                )
                arg_vals.append(av)
                arg_types.append(at)
            result_mlir = type_to_mlir(expr.result_type)
            call_result = f"%{_join_prefix(call_prefix, 'ret')}"
            arg_list = ", ".join(arg_vals)
            type_list = ", ".join(arg_types)
            lines.append(
                f"    {call_result} = func.call @{func.name}({arg_list})"
                f" : ({type_list}) -> {result_mlir}"
            )
            return call_result, result_mlir

    if isinstance(expr, HIRScatterAdd):
        from remora.lowering.scalar import _RegionEmitter as _ScatterEmitter
        sp = _join_prefix(prefix, f"sa{_next_uid}")

    if isinstance(expr, HIRIndex) and isinstance(expr.result_type, ScalarType):
        ip = _join_prefix(prefix, f"ix{_next_uid}")
        arr_val, arr_type = _lower_body_in_loop(
            expr.array, lines, _join_prefix(ip, "arr"), functions,
            tensor_env, scalar_env, _next_uid + 1,
        )
        idx_strs: list[str] = []
        for i, idx_expr in enumerate(expr.indices):
            if isinstance(idx_expr, HIRLit) and idx_expr.type == INT:
                c_name = f"%{_join_prefix(ip, f'c{i}')}"
                lines.append(f"    {c_name} = arith.constant {int(idx_expr.value)} : index")
                idx_strs.append(c_name)
            else:
                iv, _ = _lower_body_in_loop(
                    idx_expr, lines, _join_prefix(ip, f"i{i}"), functions,
                    tensor_env, scalar_env, _next_uid + i + 2,
                )
                cast_name = f"%{_join_prefix(ip, f'ic{i}')}"
                lines.append(f"    {cast_name} = arith.index_cast {iv} : i32 to index")
                idx_strs.append(cast_name)
        result_mlir = type_to_mlir(expr.result_type)
        result_name = f"%{_join_prefix(ip, 'val')}"
        idx_list = ", ".join(idx_strs)
        lines.append(f"    {result_name} = tensor.extract {arr_val}[{idx_list}] : {arr_type}")
        return result_name, result_mlir

    if isinstance(expr, HIRPrimOp) and _is_scalar_type(expr.result_type):
        pp = _join_prefix(prefix, f"po{_next_uid}")
        if tensor_env:
            arg_vals: list[str] = []
            arg_vals: list[str] = []
            for i, arg in enumerate(expr.args):
                av, _ = _lower_body_in_loop(
                    arg, lines, _join_prefix(pp, f"a{i}"), functions,
                    tensor_env, scalar_env, _next_uid + i + 1,
                )
                arg_vals.append(av)
            result_mlir = type_to_mlir(expr.result_type)
            result_name = f"%{_join_prefix(pp, 'r')}"
            base_op = expr.op
            for sfx in ("f", "i", "b"):
                if base_op.endswith(sfx):
                    base_op = base_op[:-1]
                    break
            from remora.operators import arith_op as _aop
            mlir_op = _aop(base_op, result_mlir)
            if len(arg_vals) == 2:
                lines.append(f"    {result_name} = {mlir_op} {arg_vals[0]}, {arg_vals[1]} : {result_mlir}")
            elif len(arg_vals) == 1:
                lines.append(f"    {result_name} = {mlir_op} {arg_vals[0]} : {result_mlir}")
            else:
                raise RemoraLoweringError(f"HIRPrimOp with {len(arg_vals)} args not supported in loop body")
            return result_name, result_mlir

    if isinstance(expr, HIRScatterAdd):
        sp2 = _join_prefix(prefix, f"sa2_{_next_uid}")
        target_val, target_type = _lower_body_in_loop(
            expr.target, lines, _join_prefix(sp2, "tgt"), functions,
            tensor_env, scalar_env, _next_uid + 1,
        )
        idx_val, _ = _lower_body_in_loop(
            expr.index, lines, _join_prefix(sp2, "idxv"), functions,
            tensor_env, scalar_env, _next_uid + 2,
        )
        idx_ssa = f"%{_join_prefix(sp2, 'idx')}"
        lines.append(f"    {idx_ssa} = arith.index_cast {idx_val} : i32 to index")
        upd_val, _ = _lower_body_in_loop(
            expr.update, lines, _join_prefix(sp2, "upd"), functions,
            tensor_env, scalar_env, _next_uid + 3,
        )
        elem_type = type_to_mlir(expr.result_type.element)
        old_val = f"%{_join_prefix(sp2, 'old')}"
        new_val = f"%{_join_prefix(sp2, 'new')}"
        result_val = f"%{_join_prefix(sp2, 'result')}"
        lines.append(f"    {old_val} = tensor.extract {target_val}[{idx_ssa}] : {target_type}")
        lines.append(f"    {new_val} = arith.addf {old_val}, {upd_val} : {elem_type}")
        lines.append(f"    {result_val} = tensor.insert {new_val} into {target_val}[{idx_ssa}] : {target_type}")
        return result_val, target_type

    if isinstance(expr, (HIRMap, HIRApply)) and isinstance(getattr(expr, 'func', None), HIRPrimCallable):
        if isinstance(expr.result_type, ArrayType) and expr.result_type.rank >= 1:
            ep = _join_prefix(prefix, f"ew{_next_uid}")
            op_vals: list[str] = []
            op_types: list[str] = []
            for i, arr in enumerate(expr.arrays):
                ov, ot = _lower_body_in_loop(
                    arr, lines, _join_prefix(ep, f"op{i}"), functions,
                    tensor_env, scalar_env, _next_uid + i + 1,
                )
                op_vals.append(ov)
                op_types.append(ot)
            result_mlir = type_to_mlir(expr.result_type)
            elem_mlir = type_to_mlir(expr.result_type.element)
            rank = expr.result_type.rank
            identity = _identity_affine_map(rank)
            iterators = _parallel_iterators(rank)
            prim = expr.func
            op_name = prim.op
            from remora.operators import arith_op as _arith_op
            if len(op_vals) == 1:
                const_val = None
                if prim.right_arg is not None and isinstance(prim.right_arg, HIRLit):
                    const_val = prim.right_arg.value
                    const_side = "right"
                elif prim.left_arg is not None and isinstance(prim.left_arg, HIRLit):
                    const_val = prim.left_arg.value
                    const_side = "left"
                if const_val is not None:
                    empty = f"%{_join_prefix(ep, 'empty')}"
                    result = f"%{_join_prefix(ep, 'result')}"
                    mlir_op = _arith_op(op_name, elem_mlir)
                    left_arg = f"%{_join_prefix(ep, 'in0')}" if const_side == "right" else f"%{_join_prefix(ep, 'const')}"
                    right_arg = f"%{_join_prefix(ep, 'const')}" if const_side == "right" else f"%{_join_prefix(ep, 'in0')}"
                    const_str = f"{float(const_val):.6e}" if elem_mlir == "f32" else str(int(const_val))
                    lines.append(f"""\
    %{_join_prefix(ep, 'const')} = arith.constant {const_str} : {elem_mlir}
    {empty} = tensor.empty() : {result_mlir}
    {result} = linalg.generic {{
      indexing_maps = [{identity}, {identity}],
      iterator_types = {iterators}
    }} ins({op_vals[0]} : {op_types[0]}) outs({empty} : {result_mlir}) {{
    ^bb0(%{_join_prefix(ep, 'in0')}: {elem_mlir}, %{_join_prefix(ep, 'out')}: {elem_mlir}):
      %{_join_prefix(ep, 'r')} = {mlir_op} {left_arg}, {right_arg} : {elem_mlir}
      linalg.yield %{_join_prefix(ep, 'r')} : {elem_mlir}
    }} -> {result_mlir}""")
                    return result, result_mlir
            elif len(op_vals) == 2:
                empty = f"%{_join_prefix(ep, 'empty')}"
                result = f"%{_join_prefix(ep, 'result')}"
                mlir_op = _arith_op(op_name, elem_mlir)
                is_t0 = "tensor" in op_types[0]
                is_t1 = "tensor" in op_types[1]
                if is_t0 and is_t1:
                    lines.append(f"""\
    {empty} = tensor.empty() : {result_mlir}
    {result} = linalg.generic {{
      indexing_maps = [{identity}, {identity}, {identity}],
      iterator_types = {iterators}
    }} ins({op_vals[0]}, {op_vals[1]} : {op_types[0]}, {op_types[1]}) outs({empty} : {result_mlir}) {{
    ^bb0(%{_join_prefix(ep, 'in0')}: {elem_mlir}, %{_join_prefix(ep, 'in1')}: {elem_mlir}, %{_join_prefix(ep, 'out')}: {elem_mlir}):
      %{_join_prefix(ep, 'r')} = {mlir_op} %{_join_prefix(ep, 'in0')}, %{_join_prefix(ep, 'in1')} : {elem_mlir}
      linalg.yield %{_join_prefix(ep, 'r')} : {elem_mlir}
    }} -> {result_mlir}""")
                elif is_t0 and not is_t1:
                    lines.append(f"""\
    {empty} = tensor.empty() : {result_mlir}
    {result} = linalg.generic {{
      indexing_maps = [{identity}, {identity}],
      iterator_types = {iterators}
    }} ins({op_vals[0]} : {op_types[0]}) outs({empty} : {result_mlir}) {{
    ^bb0(%{_join_prefix(ep, 'in0')}: {elem_mlir}, %{_join_prefix(ep, 'out')}: {elem_mlir}):
      %{_join_prefix(ep, 'r')} = {mlir_op} %{_join_prefix(ep, 'in0')}, {op_vals[1]} : {elem_mlir}
      linalg.yield %{_join_prefix(ep, 'r')} : {elem_mlir}
    }} -> {result_mlir}""")
                elif not is_t0 and is_t1:
                    lines.append(f"""\
    {empty} = tensor.empty() : {result_mlir}
    {result} = linalg.generic {{
      indexing_maps = [{identity}, {identity}],
      iterator_types = {iterators}
    }} ins({op_vals[1]} : {op_types[1]}) outs({empty} : {result_mlir}) {{
    ^bb0(%{_join_prefix(ep, 'in0')}: {elem_mlir}, %{_join_prefix(ep, 'out')}: {elem_mlir}):
      %{_join_prefix(ep, 'r')} = {mlir_op} {op_vals[0]}, %{_join_prefix(ep, 'in0')} : {elem_mlir}
      linalg.yield %{_join_prefix(ep, 'r')} : {elem_mlir}
    }} -> {result_mlir}""")
                else:
                    result_name2 = f"%{_join_prefix(ep, 'sr')}"
                    lines.append(f"    {result_name2} = {mlir_op} {op_vals[0]}, {op_vals[1]} : {elem_mlir}")
                    return result_name2, elem_mlir
                return result, result_mlir

    if isinstance(expr, HIRLit):
        lp = _join_prefix(prefix, f"lt{_next_uid}")
        result_mlir = type_to_mlir(expr.type)
        result_name = f"%{_join_prefix(lp, 'c')}"
        if isinstance(expr.value, float):
            val_str = f"{expr.value:.6e}"
        elif isinstance(expr.value, bool):
            val_str = "1" if expr.value else "0"
        else:
            val_str = str(int(expr.value))
        lines.append(f"    {result_name} = arith.constant {val_str} : {result_mlir}")
        return result_name, result_mlir

    if isinstance(expr, HIRVar):
        if expr.name in (scalar_env or {}):
            op = scalar_env[expr.name]
            return op.value, op.type
        if expr.name in (tensor_env or {}):
            tv = tensor_env[expr.name]
            return tv.name, tv.type

    # Try full tensor lowering first
    try:
        code, val, mlir_type, _elem = _lower_tensor_input(
            expr, _join_prefix(prefix, f"tl{_next_uid}"), functions, tensor_env, scalar_env,
        )
        lines.append(code)
        return val, mlir_type
    except RemoraLoweringError:
        pass

    # Scalar-emitter fallback for simple expressions (HIRPrimOp, HIRVar, …)
    emitter = _RegionEmitter(
        input_name="", input_type="", functions=functions,
    )
    value = emitter.emit_expr(expr, scalar_env)
    lines.extend(emitter.lines)
    return value.value, value.type


def _lower_binary_map_fold_input(
    node: HIRMap | HIRApply,
    functions: dict[str, HIRFunction],
    prefix: str,
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str, str]:
    map_type = type_to_mlir(node.result_type)
    map_element_type = type_to_mlir(node.result_type.element)
    rank = node.result_type.rank
    splat_identity = _identity_affine_map(rank)
    splat_iterators = _parallel_iterators(rank)

    def _lower_operand(arg: HIRExpr, side: str):
        """Lower one binary-map operand, splatting scalars to the result rank."""
        op_prefix = _join_prefix(prefix, side)
        if _is_scalar_type(_expr_result_type(arg)):
            if isinstance(arg, HIRLit):
                scalar_code = (
                    f"    %{op_prefix}_scalar = arith.constant "
                    f"{_literal_value(arg, map_element_type)} : {map_element_type}"
                )
                scalar_value = f"%{op_prefix}_scalar"
            else:
                scalar_code, scalar_value, _st, _se = _lower_fold_input(
                    arg, functions, op_prefix,
                    tensor_env=tensor_env, scalar_env=scalar_env,
                )
            splat_code = f"""    %{op_prefix}_empty = tensor.empty() : {map_type}
    %{op_prefix} = linalg.generic {{
      indexing_maps = [{splat_identity}],
      iterator_types = {splat_iterators}
    }} outs(%{op_prefix}_empty : {map_type}) {{
    ^bb0(%{op_prefix}_out: {map_element_type}):
      linalg.yield {scalar_value} : {map_element_type}
    }} -> {map_type}"""
            code = f"{scalar_code}\n{splat_code}" if scalar_code else splat_code
            return code, f"%{op_prefix}", map_type, map_element_type
        return _lower_fold_input(
            arg, functions, op_prefix,
            tensor_env=tensor_env, scalar_env=scalar_env,
        )

    left_code, left_name, left_type, left_element_type = _lower_operand(
        node.arrays[0], "left"
    )
    right_code, right_name, right_type, right_element_type = _lower_operand(
        node.arrays[1], "right"
    )
    identity = _identity_affine_map(rank)
    iterators = _parallel_iterators(rank)
    map_empty = f"%{_join_prefix(prefix, 'map_empty')}"
    mapped = f"%{_join_prefix(prefix, 'mapped')}"
    map_left = f"%{_join_prefix(prefix, 'map_left')}"
    map_right = f"%{_join_prefix(prefix, 'map_right')}"
    map_out = f"%{_join_prefix(prefix, 'map_out')}"
    map_body = _lower_map_binary_callable_body(
        node.func,
        functions,
        left_name=map_left,
        left_type=left_element_type,
        right_name=map_right,
        right_type=right_element_type,
        result_type=map_element_type,
        scalar_env=scalar_env,
    )
    code = f"""{left_code}
{right_code}
    {map_empty} = tensor.empty() : {map_type}
    {mapped} = linalg.generic {{
      indexing_maps = [{identity}, {identity}, {identity}],
      iterator_types = {iterators}
    }} ins({left_name}, {right_name} : {left_type}, {right_type}) outs({map_empty} : {map_type}) {{
    ^bb0({map_left}: {left_element_type}, {map_right}: {right_element_type}, {map_out}: {map_element_type}):
{map_body}
    }} -> {map_type}"""
    return code, mapped, map_type, map_element_type


# ---------------------------------------------------------------------------
# Callable body lowering (map / fold)
# ---------------------------------------------------------------------------


def _lower_map_callable_body(
    callable_: object,
    functions: dict[str, HIRFunction],
    input_name: str,
    input_type: str,
    result_type: str,
    scalar_env: dict[str, _Operand] | None = None,
) -> str:
    lines, result_value = _lower_map_callable_result(
        callable_,
        functions,
        input_name=input_name,
        input_type=input_type,
        result_type=result_type,
        scalar_env=scalar_env,
    )
    lines.append(f"      linalg.yield {result_value} : {result_type}")
    return "\n".join(lines)


def _lower_map_callable_result(
    callable_: object,
    functions: dict[str, HIRFunction],
    input_name: str,
    input_type: str,
    result_type: str,
    next_temp: int = 0,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[list[str], str]:
    if isinstance(callable_, HIRPrimCallable):
        return _lower_primitive_callable_result(
            callable_,
            input_name=input_name,
            input_type=input_type,
            result_type=result_type,
            scalar_env=scalar_env,
        )
    if isinstance(callable_, HIRVar):
        function = functions.get(callable_.name)
        if function is None:
            raise RemoraLoweringError(
                f"unknown map function {callable_.name}"
            )
        if len(function.params) != 1:
            raise RemoraLoweringError(
                "only unary map functions lower to MLIR so far"
            )
        emitter = _RegionEmitter(
            input_name=input_name,
            input_type=input_type,
            next_temp=next_temp,
            functions=functions,
        )
        value = emitter.emit_expr(
            function.body,
            {
                **(scalar_env or {}),
                function.params[0].name: _Operand(input_name, []),
            },
        )
        lines = [
            *emitter.lines,
            *_cast_if_needed(
                value.value, value.type, result_type, "%result_cast"
            ),
        ]
        result_value = (
            "%result_cast" if value.type != result_type else value.value
        )
        return lines, result_value
    if isinstance(callable_, HIRLambda):
        if len(callable_.params) != 1:
            raise RemoraLoweringError(
                "only unary lambda map functions lower to MLIR so far"
            )
        emitter = _RegionEmitter(
            input_name=input_name,
            input_type=input_type,
            next_temp=next_temp,
            functions=functions,
        )
        value = emitter.emit_expr(
            callable_.body,
            {
                **(scalar_env or {}),
                callable_.params[0].name: _Operand(input_name, [], input_type),
            },
        )
        lines = [
            *emitter.lines,
            *_cast_if_needed(
                value.value, value.type, result_type, "%result_cast"
            ),
        ]
        result_value = (
            "%result_cast" if value.type != result_type else value.value
        )
        return lines, result_value
    raise RemoraLoweringError(
        "only primitive and lifted function map callables lower to MLIR so far"
    )


def _lower_map_binary_callable_body(
    callable_: object,
    functions: dict[str, HIRFunction],
    left_name: str,
    left_type: str,
    right_name: str,
    right_type: str,
    result_type: str,
    scalar_env: dict[str, _Operand] | None = None,
) -> str:
    lines, result_value = _lower_map_binary_callable_result(
        callable_,
        functions,
        left_name=left_name,
        left_type=left_type,
        right_name=right_name,
        right_type=right_type,
        result_type=result_type,
        scalar_env=scalar_env,
    )
    lines.append(f"      linalg.yield {result_value} : {result_type}")
    return "\n".join(lines)


def _lower_map_binary_callable_result(
    callable_: object,
    functions: dict[str, HIRFunction],
    left_name: str,
    left_type: str,
    right_name: str,
    right_type: str,
    result_type: str,
    next_temp: int = 0,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[list[str], str]:
    if isinstance(callable_, HIRPrimCallable):
        if callable_.left_arg is not None or callable_.right_arg is not None:
            raise RemoraLoweringError(
                "binary map operator sections are deferred"
            )
        return _lower_binary_primitive_callable_result(
            callable_,
            left_name=left_name,
            left_type=left_type,
            right_name=right_name,
            right_type=right_type,
            result_type=result_type,
        )
    if isinstance(callable_, HIRVar):
        function = functions.get(callable_.name)
        if function is None:
            raise RemoraLoweringError(
                f"unknown map function {callable_.name}"
            )
        if len(function.params) != 2:
            raise RemoraLoweringError(
                "binary map functions must take two parameters"
            )
        emitter = _RegionEmitter(
            input_name="",
            input_type="",
            next_temp=next_temp,
            functions=functions,
        )
        value = emitter.emit_expr(
            function.body,
            {
                **(scalar_env or {}),
                function.params[0].name: _Operand(
                    left_name, [], left_type
                ),
                function.params[1].name: _Operand(
                    right_name, [], right_type
                ),
            },
        )
        lines = [
            *emitter.lines,
            *_cast_if_needed(
                value.value, value.type, result_type, "%result_cast"
            ),
        ]
        result_value = (
            "%result_cast" if value.type != result_type else value.value
        )
        return lines, result_value
    if isinstance(callable_, HIRLambda):
        if len(callable_.params) != 2:
            raise RemoraLoweringError(
                "binary map lambda functions must take two parameters"
            )
        emitter = _RegionEmitter(
            input_name="",
            input_type="",
            next_temp=next_temp,
            functions=functions,
        )
        value = emitter.emit_expr(
            callable_.body,
            {
                **(scalar_env or {}),
                callable_.params[0].name: _Operand(
                    left_name, [], left_type
                ),
                callable_.params[1].name: _Operand(
                    right_name, [], right_type
                ),
            },
        )
        lines = [
            *emitter.lines,
            *_cast_if_needed(
                value.value, value.type, result_type, "%result_cast"
            ),
        ]
        result_value = (
            "%result_cast" if value.type != result_type else value.value
        )
        return lines, result_value
    raise RemoraLoweringError(
        "only primitive and lifted function binary map callables lower to MLIR so far"
    )


def _lower_binary_primitive_callable_result(
    callable_: HIRPrimCallable,
    left_name: str,
    left_type: str,
    right_name: str,
    right_type: str,
    result_type: str,
) -> tuple[list[str], str]:
    left_lines = _cast_if_needed(
        left_name, left_type, result_type, "%left_cast"
    )
    right_lines = _cast_if_needed(
        right_name, right_type, result_type, "%right_cast"
    )
    left_value = "%left_cast" if left_lines else left_name
    right_value = "%right_cast" if right_lines else right_name
    op = _arith_op(callable_.op, result_type)
    lines = [
        *left_lines,
        *right_lines,
        f"      %result = {op} {left_value}, {right_value} : {result_type}",
    ]
    return lines, "%result"


def _lower_primitive_callable_body(
    callable_: HIRPrimCallable,
    input_name: str,
    input_type: str,
    result_type: str,
    scalar_env: dict[str, _Operand] | None = None,
) -> str:
    lines, result_value = _lower_primitive_callable_result(
        callable_,
        input_name=input_name,
        input_type=input_type,
        result_type=result_type,
        scalar_env=scalar_env,
    )
    return "\n".join(
        [*lines, f"      linalg.yield {result_value} : {result_type}"]
    )


def _lower_primitive_callable_result(
    callable_: HIRPrimCallable,
    input_name: str,
    input_type: str,
    result_type: str,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[list[str], str]:
    if callable_.op in {"exp", "log"}:
        input_lines = _cast_if_needed(
            input_name, input_type, "f32", "%input_cast"
        )
        input_value = "%input_cast" if input_lines else input_name
        mlir_op = "math.exp" if callable_.op == "exp" else "math.log"
        return [
            *input_lines,
            f"      %result = {mlir_op} {input_value} : f32",
        ], "%result"

    op_type = result_type
    if callable_.op in {"==", "!=", "<", "<="}:
        op_type = input_type

    left = _lower_callable_operand(callable_.left_arg, "%left", op_type, scalar_env)
    right = _lower_callable_operand(callable_.right_arg, "%right", op_type, scalar_env)
    if callable_.left_arg is None:
        left.value = input_name
        left.lines = _cast_if_needed(
            input_name, input_type, op_type, "%left_cast"
        )
        if left.lines:
            left.value = "%left_cast"
    if callable_.right_arg is None:
        right.value = input_name
        right.lines = _cast_if_needed(
            input_name, input_type, op_type, "%right_cast"
        )
        if right.lines:
            right.value = "%right_cast"

    op = _arith_op(callable_.op, op_type)

    sep = ", " if "cmp" in op else " "

    lines = [
        *left.lines,
        *right.lines,
        f"      %result = {op}{sep}{left.value}, {right.value} : {op_type}",
    ]
    return lines, "%result"


def _lower_fold_callable_body(
    callable_: object,
    functions: dict[str, HIRFunction],
    input_name: str,
    input_type: str,
    acc_name: str,
    acc_type: str,
    result_type: str,
    scalar_env: dict[str, _Operand] | None = None,
) -> str:
    if isinstance(callable_, HIRPrimCallable):
        if callable_.left_arg is not None or callable_.right_arg is not None:
            # Fold with operator section: resolve the section and emit the
            # binary operation using the section value on the fixed side and
            # the input element on the variable side.
            section_lines, section_value = _lower_primitive_callable_result(
                callable_,
                input_name=input_name,
                input_type=input_type,
                result_type=result_type,
                scalar_env=scalar_env,
            )
            return "\n".join([
                *section_lines,
                f"      linalg.yield {section_value} : {result_type}",
            ])
        left_lines = _cast_if_needed(
            acc_name, acc_type, result_type, "%fold_left"
        )
        right_lines = _cast_if_needed(
            input_name, input_type, result_type, "%fold_right"
        )
        left_value = "%fold_left" if left_lines else acc_name
        right_value = "%fold_right" if right_lines else input_name
        op = _arith_op(callable_.op, result_type)
        sep = ", " if "cmp" in op else " "
        lines = [
            *left_lines,
            *right_lines,
            f"      %fold_result = {op}{sep}{left_value}, {right_value} : {result_type}",
            f"      linalg.yield %fold_result : {result_type}",
        ]
        return "\n".join(lines)

    if isinstance(callable_, HIRVar):
        function = functions.get(callable_.name)
        if function is None:
            raise RemoraLoweringError(
                f"unknown fold function {callable_.name}"
            )
        if len(function.params) != 2:
            raise RemoraLoweringError(
                "fold functions must take two parameters"
            )
        param_names = {function.params[0].name, function.params[1].name}
        if isinstance(function.body, HIRPrimOp) and all(
            isinstance(arg, HIRVar) and arg.name in param_names
            for arg in function.body.args
        ):
            left_lines = _cast_if_needed(
                acc_name, acc_type, result_type, "%fold_left"
            )
            right_lines = _cast_if_needed(
                input_name, input_type, result_type, "%fold_right"
            )
            left_value = "%fold_left" if left_lines else acc_name
            right_value = "%fold_right" if right_lines else input_name
            op = _arith_op(function.body.op[:-1], result_type)
            sep = ", " if "cmp" in op else " "
            lines = [
                *left_lines,
                *right_lines,
                f"      %fold_result = {op}{sep}{left_value}, {right_value} : {result_type}",
                f"      linalg.yield %fold_result : {result_type}",
            ]
            return "\n".join(lines)
        if isinstance(function.body, (HIRMap, HIRApply)) and not function.body.cell_shape:
            if (
                isinstance(function.body.func, HIRPrimCallable)
                and function.body.func.left_arg is None
                and function.body.func.right_arg is None
            ):
                op = _arith_op(function.body.func.op, result_type)
                sep = ", " if "cmp" in op else " "
                left_lines = _cast_if_needed(
                    acc_name, acc_type, result_type, "%fold_left"
                )
                right_lines = _cast_if_needed(
                    input_name, input_type, result_type, "%fold_right"
                )
                left_value = "%fold_left" if left_lines else acc_name
                right_value = (
                    "%fold_right" if right_lines else input_name
                )
                lines = [
                    *left_lines,
                    *right_lines,
                    f"      %fold_result = {op}{sep}{left_value}, {right_value} : {result_type}",
                    f"      linalg.yield %fold_result : {result_type}",
                ]
                return "\n".join(lines)
        emitter = _RegionEmitter(
            input_name="", input_type="", functions=functions
        )
        value = emitter.emit_expr(
            function.body,
            {
                **(scalar_env or {}),
                function.params[0].name: _Operand(
                    acc_name, [], acc_type
                ),
                function.params[1].name: _Operand(
                    input_name, [], input_type
                ),
            },
        )
        cast_lines = _cast_if_needed(
            value.value, value.type, result_type, "%fold_result_cast"
        )
        result_value = (
            "%fold_result_cast" if cast_lines else value.value
        )
        lines = [
            *emitter.lines,
            *cast_lines,
            f"      linalg.yield {result_value} : {result_type}",
        ]
        return "\n".join(lines)

    if isinstance(callable_, HIRLambda):
        if len(callable_.params) != 2:
            raise RemoraLoweringError(
                "fold lambda functions must take two parameters"
            )
        emitter = _RegionEmitter(
            input_name="", input_type="", functions=functions
        )
        value = emitter.emit_expr(
            callable_.body,
            {
                **(scalar_env or {}),
                callable_.params[0].name: _Operand(
                    acc_name, [], acc_type
                ),
                callable_.params[1].name: _Operand(
                    input_name, [], input_type
                ),
            },
        )
        cast_lines = _cast_if_needed(
            value.value, value.type, result_type, "%fold_result_cast"
        )
        result_value = (
            "%fold_result_cast" if cast_lines else value.value
        )
        lines = [
            *emitter.lines,
            *cast_lines,
            f"      linalg.yield {result_value} : {result_type}",
        ]
        return "\n".join(lines)

    raise RemoraLoweringError(
        "only primitive and lifted scalar fold callables lower to MLIR so far"
    )


# ---------------------------------------------------------------------------
# Array literal flattening
# ---------------------------------------------------------------------------


def _flatten_array_literal(node: HIRArrayLit) -> list[HIRLit]:
    flat: list[HIRLit] = []
    for element in node.elements:
        if isinstance(element, HIRLit):
            flat.append(element)
        elif isinstance(element, HIRArrayLit):
            flat.extend(_flatten_array_literal(element))
        else:
            raise RemoraLoweringError(
                "only scalar literal elements lower in tensor literals so far"
            )
    return flat


# ---------------------------------------------------------------------------
# Scan lowering
# ---------------------------------------------------------------------------


def _lower_rotate_module(
    node: HIRRotate, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("rotate lowering requires an array result")
    result_rank = node.result_type.rank
    if result_rank < 1:
        raise RemoraLoweringError("rotate lowering requires at least rank-1 array")

    input_code, input_name, input_type, input_element_type = _lower_tensor_input(
        node.array, "input", functions
    )
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    shift = node.shift.value
    N = node.result_type.shape[0].value

    # Build N-D affine map and iterator types
    dims = ", ".join(f"d{i}" for i in range(result_rank))
    affine_map = f"affine_map<({dims}) -> ({dims})>"
    iterator_types = ", ".join(['"parallel"'] * result_rank)

    # Build trailing dimension indices and extract indices
    trailing_indices = ""
    extract_indices = "%wrapped"
    if result_rank > 1:
        trailing_defs = "\n".join(
            f"      %d{i} = linalg.index {i} : index" for i in range(1, result_rank)
        )
        trailing_indices = "\n" + trailing_defs
        extract_indices = "%wrapped, " + ", ".join(f"%d{i}" for i in range(1, result_rank))

    body = f"""{input_code}
    %rot_zero = arith.constant 0 : index
    %rot_N = arith.constant {N} : index
    %rot_shift = arith.constant {shift} : index
    %empty = tensor.empty() : {result_type}
    %rotated = linalg.generic {{
      indexing_maps = [{affine_map}],
      iterator_types = [{iterator_types}]
    }} outs(%empty : {result_type}) {{
    ^bb0(%out: {result_element_type}):
      %idx = linalg.index 0 : index
      %shifted = arith.addi %idx, %rot_shift : index
      %wrapped = arith.remsi %shifted, %rot_N : index{trailing_indices}
      %elem = tensor.extract {input_name}[{extract_indices}] : {input_type}
      linalg.yield %elem : {result_element_type}
    }} -> {result_type}"""

    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%rotated")


# ---------------------------------------------------------------------------
# Subarray lowering
# ---------------------------------------------------------------------------


def _lower_subarray_module(
    node: HIRSubarray, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    input_code, input_name, input_type, _input_elem = _lower_tensor_input(
        node.array, "input", functions
    )
    result_type_mlir = type_to_mlir(node.result_type)
    input_type_mlir = type_to_mlir(_expr_result_type(node.array))

    offsets = ", ".join(str(o.value) for o in node.offsets)
    sizes = ", ".join(str(s.value) for s in node.sizes)
    strides = ", ".join("1" for _ in node.offsets)

    body = f"""{input_code}
    %extracted = tensor.extract_slice {input_name}[{offsets}] [{sizes}] [{strides}] : {input_type_mlir} to {result_type_mlir}"""

    builder = _MLIRMainModuleBuilder(result_type_mlir)
    builder.add_block(body)
    return builder.render("%extracted")


# ---------------------------------------------------------------------------
# Indices-of lowering
# ---------------------------------------------------------------------------


def _lower_indices_of_module(
    node: HIRIndicesOf, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)
    rank = node.result_type.rank
    input_rank = node.result_type.rank - 1 if rank > 1 else 0

    identity = _identity_affine_map(rank)
    iterators = _parallel_iterators(rank)

    # Generate the conditional chain: for each coordinate dim k, yield linalg.index (k+1)
    yield_val = f"%idx{input_rank}" if input_rank >= 1 else "%c0_i32"
    if rank == 2:  # rank-1 input → rank-2 result [1, N]
        body = f"""    %c0_i32 = arith.constant 0 : i32
    %empty = tensor.empty() : {result_type}
    %indices = linalg.generic {{
      indexing_maps = [{identity}],
      iterator_types = {iterators}
    }} outs(%empty : {result_type}) {{
    ^bb0(%out: {result_elem}):
      %d1 = linalg.index 1 : index
      %cast1 = arith.index_cast %d1 : index to {result_elem}
      linalg.yield %cast1 : {result_elem}
    }} -> {result_type}"""
    elif rank == 3:  # rank-2 input → rank-3 result [2, R, C]
        body = f"""    %empty = tensor.empty() : {result_type}
    %indices = linalg.generic {{
      indexing_maps = [{identity}],
      iterator_types = {iterators}
    }} outs(%empty : {result_type}) {{
    ^bb0(%out: {result_elem}):
      %d0 = linalg.index 0 : index
      %d1 = linalg.index 1 : index
      %d2 = linalg.index 2 : index
      %c0_idx = arith.constant 0 : index
      %is_row = arith.cmpi eq, %d0, %c0_idx : index
      %row_val = arith.index_cast %d1 : index to {result_elem}
      %col_val = arith.index_cast %d2 : index to {result_elem}
      %val = arith.select %is_row, %row_val, %col_val : {result_elem}
      linalg.yield %val : {result_elem}
    }} -> {result_type}"""
    else:
        raise RemoraLoweringError(f"unsupported rank {rank} for indices-of lowering")

    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%indices")


# ---------------------------------------------------------------------------
# With-shape lowering
# ---------------------------------------------------------------------------


def _lower_with_shape_module(
    node: HIRWithShape, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)
    rank = node.result_type.rank

    identity = _identity_affine_map(rank)
    iterators = _parallel_iterators(rank)

    # For scalar→tensor: splat the value
    source_remora = _expr_result_type(node.source)
    if isinstance(source_remora, ScalarType):
        if not isinstance(node.source, HIRLit):
            raise RemoraLoweringError(
                "only scalar-literal with-shape lowers as top-level module"
            )
        lit_val = _literal_value(node.source, result_elem)
        body = f"""    %val = arith.constant {lit_val} : {result_elem}
    %empty = tensor.empty() : {result_type}
    %result = linalg.generic {{
      indexing_maps = [{identity}],
      iterator_types = {iterators}
    }} outs(%empty : {result_type}) {{
    ^bb0(%out: {result_elem}):
      linalg.yield %val : {result_elem}
    }} -> {result_type}"""
        builder = _MLIRMainModuleBuilder(result_type)
        builder.add_block(body)
        return builder.render("%result")

    # Array→tensor: broadcast source tensor to target shape
    if isinstance(source_remora, ArrayType):
        source_rank = source_remora.rank
        src_code, src_name, src_type, src_elem = _lower_tensor_input(
            node.source, "src", functions, tensor_env=None
        )
        all_dims = ", ".join(f"d{a}" for a in range(rank))
        src_dims = ", ".join(f"d{a}" for a in range(rank - source_rank, rank))
        src_map = f"affine_map<({all_dims}) -> ({src_dims})>"
        tgt_map = _identity_affine_map(rank)
        body = f"""{src_code}
    %empty = tensor.empty() : {result_type}
    %result = linalg.generic {{
      indexing_maps = [{src_map}, {tgt_map}],
      iterator_types = {iterators}
    }} ins({src_name} : {src_type}) outs(%empty : {result_type}) {{
    ^bb0(%in: {src_elem}, %out: {result_elem}):
      linalg.yield %in : {result_elem}
    }} -> {result_type}"""
        builder = _MLIRMainModuleBuilder(result_type)
        builder.add_block(body)
        return builder.render("%result")

    raise RemoraLoweringError("only scalar→tensor with-shape lowers to MLIR so far")


def _lower_scatter_add_module(
    node: HIRScatterAdd, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    target_code, target_name, target_type, target_elem = _lower_tensor_input(
        node.target, "target", functions
    )
    from remora.lowering.scalar import _lower_scalar_module

    if isinstance(node.update, HIRLit):
        lit_val = _literal_value(node.update, target_elem)
        update_code = f"    %update = arith.constant {lit_val} : {target_elem}"
        update_name = "%update"
    else:
        update_code = _lower_scalar_module(node.update, functions)
        update_name = "%result"

    if isinstance(node.index, HIRLit) and node.index.type == INT:
        idx_code = f"    %idx = arith.constant {int(node.index.value)} : index"
        idx_name = "%idx"
    else:
        idx_code = _lower_scalar_module(node.index, functions)
        idx_name = "%result"

    result_type = type_to_mlir(node.result_type)
    add_op = "arith.addf" if target_elem == "f32" else "arith.addi"
    body = f"""{target_code}
{update_code}
{idx_code}
    %extracted = tensor.extract {target_name}[{idx_name}] : {target_type}
    %added = {add_op} %extracted, {update_name} : {target_elem}
    %result = tensor.insert %added into {target_name}[{idx_name}] : {target_type}"""
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%result")


# ---------------------------------------------------------------------------
# Append lowering
# ---------------------------------------------------------------------------


def _lower_append_input(
    node: HIRAppend,
    prefix: str,
    functions: dict[str, HIRFunction],
    tensor_env: TensorEnv | None = None,
    scalar_env: dict[str, _Operand] | None = None,
) -> tuple[str, str, str, str]:
    left_code, left_name, left_type, _left_elem = _lower_tensor_input(
        node.left,
        f"{prefix}_left",
        functions,
        tensor_env=tensor_env,
        scalar_env=scalar_env,
    )
    right_code, right_name, right_type, _right_elem = _lower_tensor_input(
        node.right,
        f"{prefix}_right",
        functions,
        tensor_env=tensor_env,
        scalar_env=scalar_env,
    )
    result_type_mlir = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    result_rank = node.result_type.rank

    left_remora = _expr_result_type(node.left)
    right_remora = _expr_result_type(node.right)
    left_shape = left_remora.shape if isinstance(left_remora, ArrayType) else ()
    right_shape = right_remora.shape if isinstance(right_remora, ArrayType) else ()

    left_dim = left_shape[0].value

    zero_offsets = ", ".join(["0"] * result_rank)
    left_sizes = ", ".join(str(d.value) for d in left_shape)
    right_sizes = ", ".join(str(d.value) for d in right_shape)
    strides = ", ".join(["1"] * result_rank)
    right_offsets = f"{left_dim}" + (
        ", 0" * (result_rank - 1) if result_rank > 1 else ""
    )

    empty_name = f"%{prefix}_empty"
    tmp_name = f"%{prefix}_tmp"
    result_name = f"%{prefix}"
    code = f"""{left_code}
{right_code}
    {empty_name} = tensor.empty() : {result_type_mlir}
    {tmp_name} = tensor.insert_slice {left_name} into {empty_name}[{zero_offsets}] [{left_sizes}] [{strides}] : {left_type} into {result_type_mlir}
    {result_name} = tensor.insert_slice {right_name} into {tmp_name}[{right_offsets}] [{right_sizes}] [{strides}] : {right_type} into {result_type_mlir}"""
    return code, result_name, result_type_mlir, result_element_type


def _lower_append_module(
    node: HIRAppend, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_name, result_type_mlir, _result_element_type = (
        _lower_append_input(node, "result", functions)
    )
    builder = _MLIRMainModuleBuilder(result_type_mlir)
    builder.add_block(body)
    return builder.render(result_name)


# ---------------------------------------------------------------------------
# Scan lowering
# ---------------------------------------------------------------------------


def _lower_scan_module(
    node: HIRScan, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("scan lowering requires an array result")
    result_rank = node.result_type.rank
    if result_rank < 1:
        raise RemoraLoweringError("scan lowering requires at least rank 1")

    if result_rank == 1:
        return _lower_scan_rank1(node, functions)

    return _lower_scan_multirank(node, functions, result_rank)


def _lower_scan_rank1(
    node: HIRScan, functions: dict[str, HIRFunction]
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)
    init_value_str = _literal_value(node.init, result_element_type)
    op_name = _arith_op(node.func.op, result_element_type)
    N = node.reduction_dim.value

    input_code, input_name, input_type, input_element_type = _lower_tensor_input(
        node.array, "input", functions
    )

    if node.right:
        body = f"""{input_code}
    %init = arith.constant {init_value_str} : {result_element_type}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant {N} : index
    %cNminus1 = arith.constant {N - 1} : index
    %empty = tensor.empty() : {result_type}
    %filled = linalg.fill ins(%init : {result_element_type}) outs(%empty : {result_type}) -> {result_type}
    %scanned, %_carry = \"scf.for\"(%c0, %cN, %c1, %filled, %init) ({{
    ^bb0(%i: index, %acc_tensor: {result_type}, %carry: {result_element_type}):
      %rev_idx = arith.subi %cNminus1, %i : index
      %elem = tensor.extract {input_name}[%rev_idx] : {input_type}
      %next_carry = {op_name} %carry, %elem : {result_element_type}
      %stored = tensor.insert %next_carry into %acc_tensor[%rev_idx] : {result_type}
      \"scf.yield\"(%stored, %next_carry) : ({result_type}, {result_element_type}) -> ()
    }}) : (index, index, index, {result_type}, {result_element_type}) -> ({result_type}, {result_element_type})"""
    elif node.exclusive:
        body = f"""{input_code}
    %init = arith.constant {init_value_str} : {result_element_type}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant {N} : index
    %empty = tensor.empty() : {result_type}
    %filled = linalg.fill ins(%init : {result_element_type}) outs(%empty : {result_type}) -> {result_type}
    %scanned, %_carry = \"scf.for\"(%c0, %cN, %c1, %filled, %init) ({{
    ^bb0(%i: index, %acc_tensor: {result_type}, %carry: {result_element_type}):
      %stored = tensor.insert %carry into %acc_tensor[%i] : {result_type}
      %elem = tensor.extract {input_name}[%i] : {input_type}
      %next_carry = {op_name} %carry, %elem : {result_element_type}
      \"scf.yield\"(%stored, %next_carry) : ({result_type}, {result_element_type}) -> ()
    }}) : (index, index, index, {result_type}, {result_element_type}) -> ({result_type}, {result_element_type})"""
    else:
        body = f"""{input_code}
    %init = arith.constant {init_value_str} : {result_element_type}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant {N} : index
    %empty = tensor.empty() : {result_type}
    %filled = linalg.fill ins(%init : {result_element_type}) outs(%empty : {result_type}) -> {result_type}
    %scanned, %_carry = \"scf.for\"(%c0, %cN, %c1, %filled, %init) ({{
    ^bb0(%i: index, %acc_tensor: {result_type}, %carry: {result_element_type}):
      %elem = tensor.extract {input_name}[%i] : {input_type}
      %next_carry = {op_name} %carry, %elem : {result_element_type}
      %stored = tensor.insert %next_carry into %acc_tensor[%i] : {result_type}
      \"scf.yield\"(%stored, %next_carry) : ({result_type}, {result_element_type}) -> ()
    }}) : (index, index, index, {result_type}, {result_element_type}) -> ({result_type}, {result_element_type})"""

    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%scanned")


def _lower_scan_multirank(
    node: HIRScan, functions: dict[str, HIRFunction], rank: int
) -> str:
    """Lower a scan over rank >= 2 using nested loops.

    Outer loop over the leading dimension; inner loops over all trailing
    dimensions do element-wise carry updates via flat indexing.
    """
    from remora.lowering.module import _MLIRMainModuleBuilder

    N = node.reduction_dim.value
    result_type = type_to_mlir(node.result_type)
    result_element_type = type_to_mlir(node.result_type.element)

    if node.exclusive or node.right:
        result_type_mlir = result_type
        result_element_type_mlir = result_element_type
        # Trailing dimensions
        trailing_dims = [d.value for d in node.result_type.shape[1:]]
        trailing_total = 1
        for d in trailing_dims:
            trailing_total *= d
        trailing_type = _tensor_type_mlir(trailing_dims, result_element_type_mlir)

        init_code, init_name, _init_type, _ielem = _lower_tensor_input(
            node.init, "scan_init", functions
        )
        input_code, input_name, input_type, _ielem2 = _lower_tensor_input(
            node.array, "input", functions
        )
        op_name = _arith_op(node.func.op, result_element_type_mlir)

        dim_consts = "".join(
            f"    %cD{di} = arith.constant {d} : index\n"
            for di, d in enumerate(trailing_dims)
        )
        suffix_products = []
        for i in range(1, len(trailing_dims)):
            prod = 1
            for d in trailing_dims[i:]:
                prod *= d
            suffix_products.append(prod)

        row_offsets = "%i" + ", %c0" * (rank - 1)
        row_sizes = "1, " + ", ".join(str(d) for d in trailing_dims)
        row_strides = ", ".join(["1"] * rank)

        if len(trailing_dims) == 1:
            multi_idx = "%k"
            multi_idx_compute = ""
        else:
            parts = []
            compute = ""
            remaining = "%k"
            for di in range(len(trailing_dims)):
                if di == len(trailing_dims) - 1:
                    idx = remaining
                else:
                    suffix_var = f"%s{di}"
                    div_op = f"{suffix_var} = arith.divui {remaining}, %cS{di} : index"
                    rem_op = f"%r{di} = arith.remui {remaining}, %cS{di} : index"
                    compute += f"        {div_op}\n        {rem_op}\n"
                    idx = suffix_var
                    remaining = f"%r{di}"
                parts.append(idx)
            multi_idx = ", ".join(parts)
            multi_idx_compute = compute

        suffix_consts = "".join(
            f"    %cS{si} = arith.constant {s} : index\n"
            for si, s in enumerate(suffix_products)
        )

        return _lower_scan_multirank_exclusive_right(
            node, functions, rank,
            result_type_mlir, result_element_type_mlir,
            input_code, input_name, input_type,
            init_code, init_name,
            trailing_dims, trailing_total, trailing_type,
            dim_consts, suffix_products, suffix_consts,
            row_offsets, row_sizes, row_strides,
            multi_idx, multi_idx_compute,
        )

    # Trailing dimensions
    trailing_dims = [d.value for d in node.result_type.shape[1:]]
    trailing_total = 1
    for d in trailing_dims:
        trailing_total *= d
    trailing_type = _tensor_type_mlir(trailing_dims, result_element_type)

    init_code, init_name, _init_type, _ielem = _lower_tensor_input(
        node.init, "scan_init", functions
    )
    input_code, input_name, input_type, _ielem2 = _lower_tensor_input(
        node.array, "input", functions
    )
    op_name = _arith_op(node.func.op, result_element_type)

    # Build constant definitions for all dimensions
    dim_consts = "".join(
        f"    %cD{di} = arith.constant {d} : index\n"
        for di, d in enumerate(trailing_dims)
    )

    # Build product-of-suffixes for flat-index decomposition
    # For trailing dims [d0, d1, d2], products: d1*d2, d2, 1
    suffix_products = []
    for i in range(1, len(trailing_dims)):
        prod = 1
        for d in trailing_dims[i:]:
            prod *= d
        suffix_products.append(prod)

    # Build offset index list for row extraction: [%i, 0, 0, ...]
    row_offsets = "%i" + ", %c0" * (rank - 1)
    # Row sizes: [1, d0, d1, ...]
    row_sizes = "1, " + ", ".join(str(d) for d in trailing_dims)
    # Row strides: all 1s
    row_strides = ", ".join(["1"] * rank)

    # Build flat-index decomposition into multi-index
    if len(trailing_dims) == 1:
        # Single trailing dim: just use %k directly
        multi_idx = "%k"
        multi_idx_compute = ""
    else:
        # Multiple trailing dims: decompose flat index into (j0, j1, ...)
        parts = []
        compute = ""
        remaining = "%k"
        for di in range(len(trailing_dims)):
            if di == len(trailing_dims) - 1:
                idx = remaining
            else:
                suffix_var = f"%s{di}"
                div_op = f"{suffix_var} = arith.divui {remaining}, %cS{di} : index"
                rem_op = f"%r{di} = arith.remui {remaining}, %cS{di} : index"
                compute += f"        {div_op}\n        {rem_op}\n"
                idx = suffix_var
                remaining = f"%r{di}"
            parts.append(idx)
        multi_idx = ", ".join(parts)
        multi_idx_compute = compute

    # Define suffix product constants
    suffix_consts = "".join(
        f"    %cS{si} = arith.constant {s} : index\n"
        for si, s in enumerate(suffix_products)
    )

    # The carry update loop: for k in 0..trailing_total:
    #   extract carry[k] and input[i, k0, k1, ...]
    #   apply op, insert into carry[k]
    body = f"""{init_code}
{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant {N} : index
    %cTotal = arith.constant {trailing_total} : index
{dim_consts}{suffix_consts}
    %empty = tensor.empty() : {result_type}
    %scanned, %_carry = "scf.for"(%c0, %cN, %c1, %empty, {init_name}) ({{
    ^bb0(%i: index, %acc_tensor: {result_type}, %carry: {trailing_type}):
      %new_carry = "scf.for"(%c0, %cTotal, %c1, %carry) ({{
      ^bb1(%k: index, %c: {trailing_type}):
{multi_idx_compute}        %c_elem = tensor.extract %c[{multi_idx}] : {trailing_type}
        %in_elem = tensor.extract {input_name}[%i, {multi_idx}] : {input_type}
        %added = {op_name} %c_elem, %in_elem : {result_element_type}
        %c_next = tensor.insert %added into %c[{multi_idx}] : {trailing_type}
        "scf.yield"(%c_next) : ({trailing_type}) -> ()
      }}) : (index, index, index, {trailing_type}) -> {trailing_type}
      %acc_next = tensor.insert_slice %new_carry into %acc_tensor[{row_offsets}] [{row_sizes}] [{row_strides}] : {trailing_type} into {result_type}
      "scf.yield"(%acc_next, %new_carry) : ({result_type}, {trailing_type}) -> ()
    }}) : (index, index, index, {result_type}, {trailing_type}) -> ({result_type}, {trailing_type})"""

    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%scanned")


def _lower_scan_multirank_exclusive_right(
    node: HIRScan,
    functions: dict[str, HIRFunction],
    rank: int,
    result_type: str,
    result_element_type: str,
    input_code: str,
    input_name: str,
    input_type: str,
    init_code: str,
    init_name: str,
    trailing_dims: list[int],
    trailing_total: int,
    trailing_type: str,
    dim_consts: str,
    suffix_products: list[int],
    suffix_consts: str,
    row_offsets: str,
    row_sizes: str,
    row_strides: str,
    multi_idx: str,
    multi_idx_compute: str,
) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    N = node.reduction_dim.value
    op_name = _arith_op(node.func.op, result_element_type)
    exclusive = node.exclusive
    right = node.right

    if right:
        cNm1 = N - 1
        cm1 = -1
        loop_start = "%cNm1"
        loop_limit = "%cm1"
        loop_step = "%cm1"
        init_consts = f"    %cNm1 = arith.constant {cNm1} : index\n    %cm1 = arith.constant {cm1} : index\n"
    else:
        loop_start = "%c0"
        loop_limit = "%cN"
        loop_step = "%c1"
        init_consts = f"    %cN = arith.constant {N} : index\n" if exclusive else ""

    if exclusive:
        inner_body = f"""{multi_idx_compute}        %c_elem = tensor.extract %c[{multi_idx}] : {trailing_type}
        %in_elem = tensor.extract {input_name}[%i, {multi_idx}] : {input_type}
        %added = {op_name} %c_elem, %in_elem : {result_element_type}
        %c_next = tensor.insert %added into %c[{multi_idx}] : {trailing_type}
        "scf.yield"(%c_next) : ({trailing_type}) -> ()"""
        outer_body = f"""      %acc_w_old_carry = tensor.insert_slice %carry into %acc_tensor[{row_offsets}] [{row_sizes}] [{row_strides}] : {trailing_type} into {result_type}
      %new_carry = "scf.for"(%c0, %cTotal, %c1, %carry) ({{
      ^bb1(%k: index, %c: {trailing_type}):
{inner_body}
      }}) : (index, index, index, {trailing_type}) -> {trailing_type}
      "scf.yield"(%acc_w_old_carry, %new_carry) : ({result_type}, {trailing_type}) -> ()"""
    else:
        inner_body = f"""{multi_idx_compute}        %c_elem = tensor.extract %c[{multi_idx}] : {trailing_type}
        %in_elem = tensor.extract {input_name}[%i, {multi_idx}] : {input_type}
        %added = {op_name} %c_elem, %in_elem : {result_element_type}
        %c_next = tensor.insert %added into %c[{multi_idx}] : {trailing_type}
        "scf.yield"(%c_next) : ({trailing_type}) -> ()"""
        outer_body = f"""      %new_carry = "scf.for"(%c0, %cTotal, %c1, %carry) ({{
      ^bb1(%k: index, %c: {trailing_type}):
{inner_body}
      }}) : (index, index, index, {trailing_type}) -> {trailing_type}
      %acc_next = tensor.insert_slice %new_carry into %acc_tensor[{row_offsets}] [{row_sizes}] [{row_strides}] : {trailing_type} into {result_type}
      "scf.yield"(%acc_next, %new_carry) : ({result_type}, {trailing_type}) -> ()"""

    body = f"""{init_code}
{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
{init_consts}    %cTotal = arith.constant {trailing_total} : index
{dim_consts}{suffix_consts}
    %empty = tensor.empty() : {result_type}
    %scanned, %_carry = "scf.for"({loop_start}, {loop_limit}, {loop_step}, %empty, {init_name}) ({{
    ^bb0(%i: index, %acc_tensor: {result_type}, %carry: {trailing_type}):
{outer_body}
    }}) : (index, index, index, {result_type}, {trailing_type}) -> ({result_type}, {trailing_type})"""

    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render("%scanned")


def _tensor_type_mlir(dims: list[int], elem: str) -> str:
    if not dims:
        return elem
    ds = "x".join(str(d) for d in dims)
    return f"tensor<{ds}x{elem}>"


# ---------------------------------------------------------------------------
# Sort / Grade lowering (C runtime qsort)
# ---------------------------------------------------------------------------


def _sort_runtime_func(result_elem: str) -> str:
    if result_elem == "i32":
        return "remora_sort_i32"
    if result_elem == "f32":
        return "remora_sort_f32"
    raise RemoraLoweringError(f"sort not supported for type {result_elem}")


def _lower_sort_module(node: HIRSort, functions: dict[str, HIRFunction]) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("sort lowering requires array result type")
    rank = node.result_type.rank
    if rank < 1 or rank > 2:
        raise RemoraLoweringError("sort lowering supports ranks 1 and 2")

    input_code, input_name, input_type, input_element_type = _lower_tensor_input(
        node.array, "sort_input", functions, tensor_env=None
    )
    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)
    rt_func = _sort_runtime_func(result_elem)

    if rank == 1:
        n = node.result_type.shape[0].value
        sort_body = f"""{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN{n} = arith.constant {n} : index
    %buf = memref.alloc() : memref<{n}x{result_elem}>
    scf.for %i = %c0 to %cN{n} step %c1 {{
      %val = tensor.extract {input_name}[%i] : {result_type}
      memref.store %val, %buf[%i] : memref<{n}x{result_elem}>
    }}
    func.call @{rt_func}(%buf) : (memref<{n}x{result_elem}>) -> ()
    %sorted = bufferization.to_tensor %buf restrict writable : memref<{n}x{result_elem}>"""
        builder = _MLIRMainModuleBuilder(result_type)
        builder.add_extern(f"  func.func private @{rt_func}(memref<{n}x{result_elem}>)")
        builder.add_block(sort_body)
        return builder.render("%sorted")

    # Rank 2: per-row sort using memref operations
    R = node.result_type.shape[0].value
    C = node.result_type.shape[1].value
    sort_body = f"""{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cR = arith.constant {R} : index
    %cC = arith.constant {C} : index
    %in_mem = memref.alloc() : memref<{R}x{C}x{result_elem}>
    scf.for %i = %c0 to %cR step %c1 {{
      scf.for %j = %c0 to %cC step %c1 {{
        %v = tensor.extract {input_name}[%i, %j] : {result_type}
        memref.store %v, %in_mem[%i, %j] : memref<{R}x{C}x{result_elem}>
      }}
    }}
    scf.for %r = %c0 to %cR step %c1 {{
      %row_buf = memref.alloc() : memref<{C}x{result_elem}>
      scf.for %j = %c0 to %cC step %c1 {{
        %v = memref.load %in_mem[%r, %j] : memref<{R}x{C}x{result_elem}>
        memref.store %v, %row_buf[%j] : memref<{C}x{result_elem}>
      }}
      func.call @remora_sort_1d_{result_elem}(%row_buf) : (memref<{C}x{result_elem}>) -> ()
      scf.for %j = %c0 to %cC step %c1 {{
        %v = memref.load %row_buf[%j] : memref<{C}x{result_elem}>
        memref.store %v, %in_mem[%r, %j] : memref<{R}x{C}x{result_elem}>
      }}
    }}
    %sorted = bufferization.to_tensor %in_mem restrict writable : memref<{R}x{C}x{result_elem}>"""
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_extern(f"  func.func private @remora_sort_1d_{result_elem}(memref<{C}x{result_elem}>)")
    builder.add_block(sort_body)
    return builder.render("%sorted")


def _lower_grade_module(node: HIRGrade, functions: dict[str, HIRFunction]) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, ArrayType):
        raise RemoraLoweringError("grade result must be array type")
    rank = node.result_type.rank
    if rank < 1 or rank > 2:
        raise RemoraLoweringError("grade lowering supports ranks 1 and 2")

    input_code, input_name, input_type, input_element_type = _lower_tensor_input(
        node.array, "grade_input", functions, tensor_env=None
    )
    del input_type
    n = node.result_type.shape[0].value
    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)

    if rank == 1:
        rt_func = "remora_grade_i32" if input_element_type == "i32" else "remora_grade_f32"
        grade_body = f"""{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN{n} = arith.constant {n} : index
    %buf_in = memref.alloc() : memref<{n}x{input_element_type}>
    %buf_out = memref.alloc() : memref<{n}x{result_elem}>
    scf.for %i = %c0 to %cN{n} step %c1 {{
      %val = tensor.extract {input_name}[%i] : tensor<{n}x{input_element_type}>
      memref.store %val, %buf_in[%i] : memref<{n}x{input_element_type}>
    }}
    func.call @{rt_func}(%buf_in, %buf_out) : (memref<{n}x{input_element_type}>, memref<{n}x{result_elem}>) -> ()
    %sorted_indices = bufferization.to_tensor %buf_out restrict writable : memref<{n}x{result_elem}>
    memref.dealloc %buf_in : memref<{n}x{input_element_type}>"""
        builder = _MLIRMainModuleBuilder(result_type)
        builder.add_extern(
            f"  func.func private @{rt_func}(memref<{n}x{input_element_type}>, memref<{n}x{result_elem}>)"
        )
        builder.add_block(grade_body)
        return builder.render("%sorted_indices")

    # Rank 2: per-row grade
    R = node.result_type.shape[0].value
    C = node.result_type.shape[1].value
    rt_1d = f"remora_grade_1d_{input_element_type}"
    grade_body = f"""{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cR = arith.constant {R} : index
    %cC = arith.constant {C} : index
    %in_mem = memref.alloc() : memref<{R}x{C}x{input_element_type}>
    %out_mem = memref.alloc() : memref<{R}x{C}x{result_elem}>
    scf.for %i = %c0 to %cR step %c1 {{
      scf.for %j = %c0 to %cC step %c1 {{
        %v = tensor.extract {input_name}[%i, %j] : tensor<{R}x{C}x{input_element_type}>
        memref.store %v, %in_mem[%i, %j] : memref<{R}x{C}x{input_element_type}>
      }}
    }}
    scf.for %r = %c0 to %cR step %c1 {{
      %row_in = memref.alloc() : memref<{C}x{input_element_type}>
      %row_out = memref.alloc() : memref<{C}x{result_elem}>
      scf.for %j = %c0 to %cC step %c1 {{
        %v = memref.load %in_mem[%r, %j] : memref<{R}x{C}x{input_element_type}>
        memref.store %v, %row_in[%j] : memref<{C}x{input_element_type}>
      }}
      func.call @{rt_1d}(%row_in, %row_out) : (memref<{C}x{input_element_type}>, memref<{C}x{result_elem}>) -> ()
      scf.for %j = %c0 to %cC step %c1 {{
        %v = memref.load %row_out[%j] : memref<{C}x{result_elem}>
        memref.store %v, %out_mem[%r, %j] : memref<{R}x{C}x{result_elem}>
      }}
    }}
    %sorted_indices = bufferization.to_tensor %out_mem restrict writable : memref<{R}x{C}x{result_elem}>"""
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_extern(
        f"  func.func private @{rt_1d}(memref<{C}x{input_element_type}>, memref<{C}x{result_elem}>)"
    )
    builder.add_block(grade_body)
    return builder.render("%sorted_indices")


# ---------------------------------------------------------------------------
# Filter / Replicate lowering (C runtime with dynamic sizing)
# ---------------------------------------------------------------------------


def _cmp_op_to_mlir(op: str, elem_type: str) -> str:
    """Map Remora comparison operator to MLIR arith.cmpi/cmpf predicate."""
    int_preds = {">": "sgt", "<": "slt", ">=": "sge", "<=": "sle", "==": "eq", "!=": "ne"}
    flt_preds = {">": "ogt", "<": "olt", ">=": "oge", "<=": "ole", "==": "oeq", "!=": "one"}
    preds = int_preds if elem_type == "i32" else flt_preds
    op_base = op[:-1] if op.endswith("b") else op
    return preds.get(op_base, "sgt")


def _lower_filter_module(node: HIRFilter, functions: dict[str, HIRFunction]) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, SigmaType):
        raise RemoraLoweringError("filter result must be SigmaType")
    body_type = node.result_type.body
    if not isinstance(body_type, ArrayType) or body_type.rank != 1:
        raise RemoraLoweringError("filter only supports rank-1 arrays")

    input_code, input_name, input_type, input_elem = _lower_tensor_input(
        node.array, "flt_in", functions, tensor_env=None
    )
    n = body_type.shape[0].value
    result_elem = type_to_mlir(body_type.element)

    # Generate mask via linalg.generic
    pred = node.predicate
    if not isinstance(pred, HIRPrimCallable):
        raise RemoraLoweringError("filter predicate must be a primitive operator")
    if pred.right_arg is None or not isinstance(pred.right_arg, HIRLit):
        raise RemoraLoweringError("filter predicate must be a left section with literal")
    rhs_val = _literal_value(pred.right_arg, result_elem)
    cmp_op = _cmp_op_to_mlir(pred.op, input_elem)
    cmp_kind = "arith.cmpi" if input_elem == "i32" else "arith.cmpf"

    rt = "remora_filter_i32" if input_elem == "i32" else "remora_filter_f32"

    filter_body = f"""{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN{n} = arith.constant {n} : index
    %cBig = arith.constant {n+2} : index
    %mask_empty = tensor.empty() : tensor<{n}xi32>
    %mask = linalg.generic {{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }} ins({input_name} : {input_type}) outs(%mask_empty : tensor<{n}xi32>) {{
    ^bb0(%in: {input_elem}, %out: i32):
      %rhs = arith.constant {rhs_val} : {input_elem}
      %cond = {cmp_kind} {cmp_op}, %in, %rhs : {input_elem}
      %intv = arith.extui %cond : i1 to i32
      linalg.yield %intv : i32
    }} -> tensor<{n}xi32>
    %buf_src = memref.alloc() : memref<{n}x{input_elem}>
    %buf_mask = memref.alloc() : memref<{n}xi32>
    %buf_dst = memref.alloc() : memref<{n+2}x{result_elem}>
    scf.for %i = %c0 to %cN{n} step %c1 {{
      %v = tensor.extract {input_name}[%i] : {input_type}
      memref.store %v, %buf_src[%i] : memref<{n}x{input_elem}>
      %m = tensor.extract %mask[%i] : tensor<{n}xi32>
      memref.store %m, %buf_mask[%i] : memref<{n}xi32>
    }}
    %count = func.call @{rt}(%buf_src, %buf_mask, %buf_dst) : (memref<{n}x{input_elem}>, memref<{n}xi32>, memref<{n+2}x{result_elem}>) -> i64
    %count_idx = arith.index_cast %count : i64 to index
    scf.for %k = %c0 to %count_idx step %c1 {{
      %offset = arith.subi %count_idx, %k : index
      %src_minus_1 = arith.subi %offset, %c1 : index
      %val = memref.load %buf_dst[%src_minus_1] : memref<{n+2}x{result_elem}>
      memref.store %val, %buf_dst[%offset] : memref<{n+2}x{result_elem}>
    }}
    %count_i32 = arith.trunci %count : i64 to i32
    memref.store %count_i32, %buf_dst[%c0] : memref<{n+2}x{result_elem}>
    %count_p1 = arith.addi %count_idx, %c1 : index
    %view = memref.subview %buf_dst[0] [%count_p1] [1] : memref<{n+2}x{result_elem}> to memref<?x{result_elem}, strided<[1]>>
    memref.dealloc %buf_src : memref<{n}x{input_elem}>
    memref.dealloc %buf_mask : memref<{n}xi32>
    %result = bufferization.to_tensor %view restrict writable : memref<?x{result_elem}, strided<[1]>>"""
    result_type_str = f"tensor<?x{result_elem}>"
    builder = _MLIRMainModuleBuilder(result_type_str)
    builder.add_extern(
        f"  func.func private @{rt}(memref<{n}x{input_elem}>, memref<{n}xi32>, memref<{n+2}x{result_elem}>) -> i64"
    )
    builder.add_block(filter_body)
    return builder.render("%result")


def _lower_replicate_module(node: HIRReplicate, functions: dict[str, HIRFunction]) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    if not isinstance(node.result_type, SigmaType):
        raise RemoraLoweringError("replicate result must be SigmaType")
    body_type = node.result_type.body
    if not isinstance(body_type, ArrayType) or body_type.rank != 1:
        raise RemoraLoweringError("replicate only supports rank-1 arrays")

    arr_code, arr_name, arr_type, arr_elem = _lower_tensor_input(
        node.array, "rep_arr", functions, tensor_env=None
    )
    cnt_code, cnt_name, cnt_type, _cnt_elem = _lower_tensor_input(
        node.counts, "rep_cnt", functions, tensor_env=None
    )
    n = body_type.shape[0].value
    big_n = n * 100
    result_elem = type_to_mlir(body_type.element)

    rt = "remora_replicate_i32" if arr_elem == "i32" else "remora_replicate_f32"

    replicate_body = f"""{arr_code}
{cnt_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN{n} = arith.constant {n} : index
    %buf_src = memref.alloc() : memref<{n}x{arr_elem}>
    %buf_cnt = memref.alloc() : memref<{n}xi32>
    scf.for %i = %c0 to %cN{n} step %c1 {{
      %v = tensor.extract {arr_name}[%i] : {arr_type}
      memref.store %v, %buf_src[%i] : memref<{n}x{arr_elem}>
      %c = tensor.extract {cnt_name}[%i] : {cnt_type}
      memref.store %c, %buf_cnt[%i] : memref<{n}xi32>
    }}
    %count = func.call @{rt}_count(%buf_src, %buf_cnt) : (memref<{n}x{arr_elem}>, memref<{n}xi32>) -> i64
    %count_idx = arith.index_cast %count : i64 to index
    %count_p1 = arith.addi %count_idx, %c1 : index
    %c{big_n} = arith.constant {big_n} : index
    %buf_dst = memref.alloc() : memref<{big_n}x{result_elem}>
    func.call @{rt}_fill(%buf_src, %buf_cnt, %buf_dst) : (memref<{n}x{arr_elem}>, memref<{n}xi32>, memref<{big_n}x{result_elem}>) -> ()
    scf.for %k = %c0 to %count_idx step %c1 {{
      %offset = arith.subi %count_idx, %k : index
      %src_minus_1 = arith.subi %offset, %c1 : index
      %val = memref.load %buf_dst[%src_minus_1] : memref<{big_n}x{result_elem}>
      memref.store %val, %buf_dst[%offset] : memref<{big_n}x{result_elem}>
    }}
    %count_i32 = arith.trunci %count : i64 to i32
    memref.store %count_i32, %buf_dst[%c0] : memref<{big_n}x{result_elem}>
    %view = memref.subview %buf_dst[0] [%count_p1] [1] : memref<{big_n}x{result_elem}> to memref<?x{result_elem}, strided<[1]>>
    %result = bufferization.to_tensor %view restrict writable : memref<?x{result_elem}, strided<[1]>>
    memref.dealloc %buf_src : memref<{n}x{arr_elem}>
    memref.dealloc %buf_cnt : memref<{n}xi32>"""
    result_type_str = f"tensor<?x{result_elem}>"
    builder = _MLIRMainModuleBuilder(result_type_str)
    builder.add_extern(
        f"  func.func private @{rt}_count(memref<{n}x{arr_elem}>, memref<{n}xi32>) -> i64"
    )
    builder.add_extern(
        f"  func.func private @{rt}_fill(memref<{n}x{arr_elem}>, memref<{n}xi32>, memref<{big_n}x{result_elem}>)"
    )
    builder.add_block(replicate_body)
    return builder.render("%result")


def _lower_rank2_c_unary(node, functions, c_base_name):
    """Per-row rank-2 lowering for unary array→array ops via C _1d wrappers."""
    from remora.lowering.module import _MLIRMainModuleBuilder

    R = node.result_type.shape[0].value
    C = node.result_type.shape[1].value
    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)

    input_code, input_name, input_type, input_elem = _lower_tensor_input(
        node.array, "input", functions, tensor_env=None
    )

    rt = f"{c_base_name}_{input_elem}_1d"

    body = f"""{input_code}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cR = arith.constant {R} : index
    %cC = arith.constant {C} : index
    %in_mem = memref.alloc() : memref<{R}x{C}x{input_elem}>
    %out_mem = memref.alloc() : memref<{R}x{C}x{result_elem}>
    scf.for %i = %c0 to %cR step %c1 {{
      scf.for %j = %c0 to %cC step %c1 {{
        %v = tensor.extract {input_name}[%i, %j] : {input_type}
        memref.store %v, %in_mem[%i, %j] : memref<{R}x{C}x{input_elem}>
      }}
    }}
    scf.for %r = %c0 to %cR step %c1 {{
      %row_in = memref.alloc() : memref<{C}x{input_elem}>
      %row_out = memref.alloc() : memref<{C}x{result_elem}>
      scf.for %j = %c0 to %cC step %c1 {{
        %v = memref.load %in_mem[%r, %j] : memref<{R}x{C}x{input_elem}>
        memref.store %v, %row_in[%j] : memref<{C}x{input_elem}>
      }}
      func.call @{rt}(%row_in, %row_out) : (memref<{C}x{input_elem}>, memref<{C}x{result_elem}>) -> ()
      scf.for %j = %c0 to %cC step %c1 {{
        %v = memref.load %row_out[%j] : memref<{C}x{result_elem}>
        memref.store %v, %out_mem[%r, %j] : memref<{R}x{C}x{result_elem}>
      }}
    }}
    %result = bufferization.to_tensor %out_mem restrict writable : memref<{R}x{C}x{result_elem}>"""
    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_extern(f"  func.func private @{rt}(memref<{C}x{input_elem}>, memref<{C}x{result_elem}>)")
    builder.add_block(body)
    return builder.render("%result")


def _lower_im2col_module(node, functions: dict[str, HIRFunction]) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_name, result_type, _ = _lower_im2col_tensor_input(
        node, functions, prefix="im2col"
    )

    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render(result_name)


def _lower_col2im_module(node, functions: dict[str, HIRFunction]) -> str:
    from remora.lowering.module import _MLIRMainModuleBuilder

    body, result_name, result_type, _ = _lower_col2im_tensor_input(
        node, functions, prefix="col2im"
    )

    builder = _MLIRMainModuleBuilder(result_type)
    builder.add_block(body)
    return builder.render(result_name)


def _lower_im2col_tensor_input(
    node, functions, prefix="", tensor_env=None, scalar_env=None,
):
    image_code, image_name, image_type, image_elem = _lower_tensor_input(
        node.image,
        _join_prefix(prefix, "image"),
        functions,
        tensor_env,
        scalar_env,
    )
    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)
    kh, kw = node.kernel_shape
    stride = node.stride
    n_patches = int(node.result_type.shape[0].value)
    patch_size = int(node.result_type.shape[1].value)
    image_remora_type = _expr_result_type(node.image)
    w = int(image_remora_type.shape[1].value)
    out_w = (w - kw) // stride + 1

    lines: list[str] = []
    lines.append(image_code)
    names = {
        key: f"%{_join_prefix(prefix, key)}"
        for key in (
            "c0", "c1", "c_n_patches", "c_patch_size", "c_out_w", "c_kw",
            "c_stride", "buffer", "patch_row", "patch_col", "kernel_row",
            "kernel_col", "image_row_base", "image_col_base", "image_row",
            "image_col", "pixel", "result",
        )
    }
    lines.extend([
        f"    {names['c0']} = arith.constant 0 : index",
        f"    {names['c1']} = arith.constant 1 : index",
        f"    {names['c_n_patches']} = arith.constant {n_patches} : index",
        f"    {names['c_patch_size']} = arith.constant {patch_size} : index",
        f"    {names['c_out_w']} = arith.constant {out_w} : index",
        f"    {names['c_kw']} = arith.constant {kw} : index",
        f"    {names['c_stride']} = arith.constant {stride} : index",
        f"    {names['buffer']} = memref.alloc() : memref<{n_patches}x{patch_size}x{result_elem}>",
        f"    scf.for %patch = {names['c0']} to {names['c_n_patches']} step {names['c1']} {{",
        f"      {names['patch_row']} = arith.divui %patch, {names['c_out_w']} : index",
        f"      {names['patch_col']} = arith.remui %patch, {names['c_out_w']} : index",
        f"      scf.for %pixel_index = {names['c0']} to {names['c_patch_size']} step {names['c1']} {{",
        f"        {names['kernel_row']} = arith.divui %pixel_index, {names['c_kw']} : index",
        f"        {names['kernel_col']} = arith.remui %pixel_index, {names['c_kw']} : index",
        f"        {names['image_row_base']} = arith.muli {names['patch_row']}, {names['c_stride']} : index",
        f"        {names['image_col_base']} = arith.muli {names['patch_col']}, {names['c_stride']} : index",
        f"        {names['image_row']} = arith.addi {names['image_row_base']}, {names['kernel_row']} : index",
        f"        {names['image_col']} = arith.addi {names['image_col_base']}, {names['kernel_col']} : index",
        f"        {names['pixel']} = tensor.extract {image_name}[{names['image_row']}, {names['image_col']}] : {image_type}",
        f"        memref.store {names['pixel']}, {names['buffer']}[%patch, %pixel_index] : memref<{n_patches}x{patch_size}x{result_elem}>",
        "      }",
        "    }",
        f"    {names['result']} = bufferization.to_tensor {names['buffer']} restrict writable : memref<{n_patches}x{patch_size}x{result_elem}>",
    ])

    return "\n".join(lines), names["result"], result_type, result_elem


def _lower_col2im_tensor_input(
    node, functions, prefix="", tensor_env=None, scalar_env=None,
):
    columns_code, columns_name, columns_type, columns_elem = _lower_tensor_input(
        node.columns,
        _join_prefix(prefix, "columns"),
        functions,
        tensor_env,
        scalar_env,
    )
    result_type = type_to_mlir(node.result_type)
    result_elem = type_to_mlir(node.result_type.element)
    h, w = node.image_shape
    kh, kw = node.kernel_shape
    stride = node.stride
    out_w = (w - kw) // stride + 1
    columns_remora = _expr_result_type(node.columns)
    n_patches = int(columns_remora.shape[0].value)
    patch_size = int(columns_remora.shape[1].value)

    lines: list[str] = []
    lines.append(columns_code)
    names = {
        key: f"%{_join_prefix(prefix, key)}"
        for key in (
            "c0", "c1", "c_h", "c_w", "c_n_patches", "c_patch_size",
            "c_out_w", "c_kw", "c_stride", "zero", "buffer", "patch_row",
            "patch_col", "kernel_row", "kernel_col", "image_row_base",
            "image_col_base", "image_row", "image_col", "column_pixel",
            "image_pixel", "added", "result",
        )
    }
    lines.extend([
        f"    {names['c0']} = arith.constant 0 : index",
        f"    {names['c1']} = arith.constant 1 : index",
        f"    {names['c_h']} = arith.constant {h} : index",
        f"    {names['c_w']} = arith.constant {w} : index",
        f"    {names['c_n_patches']} = arith.constant {n_patches} : index",
        f"    {names['c_patch_size']} = arith.constant {patch_size} : index",
        f"    {names['c_out_w']} = arith.constant {out_w} : index",
        f"    {names['c_kw']} = arith.constant {kw} : index",
        f"    {names['c_stride']} = arith.constant {stride} : index",
        f"    {names['zero']} = arith.constant 0.0 : {columns_elem}",
        f"    {names['buffer']} = memref.alloc() : memref<{h}x{w}x{result_elem}>",
        f"    scf.for %image_row_init = {names['c0']} to {names['c_h']} step {names['c1']} {{",
        f"      scf.for %image_col_init = {names['c0']} to {names['c_w']} step {names['c1']} {{",
        f"        memref.store {names['zero']}, {names['buffer']}[%image_row_init, %image_col_init] : memref<{h}x{w}x{result_elem}>",
        "      }",
        "    }",
        f"    scf.for %patch = {names['c0']} to {names['c_n_patches']} step {names['c1']} {{",
        f"      {names['patch_row']} = arith.divui %patch, {names['c_out_w']} : index",
        f"      {names['patch_col']} = arith.remui %patch, {names['c_out_w']} : index",
        f"      scf.for %pixel_index = {names['c0']} to {names['c_patch_size']} step {names['c1']} {{",
        f"        {names['kernel_row']} = arith.divui %pixel_index, {names['c_kw']} : index",
        f"        {names['kernel_col']} = arith.remui %pixel_index, {names['c_kw']} : index",
        f"        {names['image_row_base']} = arith.muli {names['patch_row']}, {names['c_stride']} : index",
        f"        {names['image_col_base']} = arith.muli {names['patch_col']}, {names['c_stride']} : index",
        f"        {names['image_row']} = arith.addi {names['image_row_base']}, {names['kernel_row']} : index",
        f"        {names['image_col']} = arith.addi {names['image_col_base']}, {names['kernel_col']} : index",
        f"        {names['column_pixel']} = tensor.extract {columns_name}[%patch, %pixel_index] : {columns_type}",
        f"        {names['image_pixel']} = memref.load {names['buffer']}[{names['image_row']}, {names['image_col']}] : memref<{h}x{w}x{result_elem}>",
        f"        {names['added']} = arith.addf {names['column_pixel']}, {names['image_pixel']} : {columns_elem}",
        f"        memref.store {names['added']}, {names['buffer']}[{names['image_row']}, {names['image_col']}] : memref<{h}x{w}x{result_elem}>",
        "      }",
        "    }",
        f"    {names['result']} = bufferization.to_tensor {names['buffer']} restrict writable : memref<{h}x{w}x{result_elem}>",
    ])

    return "\n".join(lines), names["result"], result_type, result_elem
