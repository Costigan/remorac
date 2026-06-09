"""Generate reusable Remora gradient source from an evaluation tape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from remora.ad import EvalTape
from remora.ast_nodes import FuncDef
from remora.types import ArrayType, FLOAT, RemoraType, ScalarType, StaticDim


Shape = tuple[int, ...]


@dataclass(frozen=True)
class _Atom:
    text: str
    shape: Shape


@dataclass(frozen=True)
class _Op:
    op: str
    left: "_Expr"
    right: "_Expr | None"
    shape: Shape


@dataclass(frozen=True)
class _Fill:
    value: "_Expr"
    like: "_Expr"
    shape: Shape


@dataclass(frozen=True)
class _Reshape:
    value: "_Expr"
    shape: Shape


@dataclass(frozen=True)
class _Transpose:
    value: "_Expr"
    shape: Shape


@dataclass(frozen=True)
class _View:
    kind: str
    value: "_Expr"
    count: int | None
    shape: Shape


@dataclass(frozen=True)
class _Append:
    left: "_Expr"
    right: "_Expr"
    shape: Shape


@dataclass(frozen=True)
class _SubarrayView:
    value: "_Expr"
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    shape: Shape


@dataclass(frozen=True)
class _Rotate:
    value: "_Expr"
    shift: int
    shape: Shape


@dataclass(frozen=True)
class _Index:
    value: "_Expr"
    idx: int
    shape: Shape


@dataclass(frozen=True)
class _ScatterAdd:
    target: "_Expr"
    index: "_Expr | int"
    update: "_Expr"
    shape: Shape


@dataclass(frozen=True)
class _If:
    condition: "_Expr"
    then_expr: "_Expr"
    else_expr: "_Expr"
    shape: Shape


@dataclass(frozen=True)
class _Pair:
    left: "_Expr"
    right: "_Expr"
    shape: Shape


_Expr: TypeAlias = _Atom | _Op | _Fill | _Reshape | _Transpose | _View | _Append | _SubarrayView | _Rotate | _Index | _ScatterAdd | _If | _Pair


@dataclass(frozen=True)
class GradientSourceArtifact:
    """A specialized tape and the reusable gradient source generated from it."""

    source: str
    function_name: str
    param_types: tuple[RemoraType, ...]
    tape: EvalTape
    input_index: int


def generate_gradient_source(
    tape: EvalTape,
    param_specs: list[tuple[str, tuple[int, ...]]],
    *,
    differentiate_input: int = 0,
    function_name: str = "grad-f",
) -> str:
    """Return a gradient function for the traced tape graph with one or more params.

    *param_specs* is a list of (name, shape) for each tape input in order.
    *differentiate_input* selects which input's gradient to return (0-based).
    """
    if not tape.entries:
        raise ValueError("cannot generate a gradient from an empty tape")
    if tape.has_data_dependent_control_flow:
        has_select = any(e.kind == "select" for e in tape.entries)
        if not has_select:
            raise NotImplementedError(
                "gradient source does not yet preserve data-dependent conditionals"
            )
    if len(tape.input_indices) < 1:
        raise ValueError("gradient source requires at least one tape input")
    if len(tape.input_indices) != len(param_specs):
        raise ValueError(
            f"param_specs length {len(param_specs)} does not match "
            f"tape input count {len(tape.input_indices)}"
        )

    # Validate shapes
    for (pname, pshape), idx in zip(param_specs, tape.input_indices):
        shape = tuple(int(dim) for dim in pshape)
        if tuple(np.asarray(tape.values[idx]).shape) != shape:
            raise ValueError(
                f"parameter {pname!r} shape does not match the traced tape input"
            )

    primals = _reconstruct_primals_multi(tape, {idx: name for name, _, idx in _enumerate_params(param_specs, tape)})
    adjs: list[_Expr | None] = [None] * len(tape.entries)
    output_shape = _shape_of(tape.values[-1])
    adjs[-1] = _constant(1.0, output_shape)

    for index in reversed(range(len(tape.entries))):
        adj = adjs[index]
        if adj is None:
            continue
        entry = tape.entries[index]
        if entry.kind == "add":
            _accumulate(adjs, entry.inputs[0], _unbroadcast(adj, primals[entry.inputs[0]]))
            _accumulate(adjs, entry.inputs[1], _unbroadcast(adj, primals[entry.inputs[1]]))
        elif entry.kind == "sub":
            _accumulate(adjs, entry.inputs[0], _unbroadcast(adj, primals[entry.inputs[0]]))
            _accumulate(adjs, entry.inputs[1], _unbroadcast(_neg(adj), primals[entry.inputs[1]]))
        elif entry.kind == "mul":
            left, right = entry.inputs
            _accumulate(adjs, left, _unbroadcast(_binary("*", adj, primals[right]), primals[left]))
            _accumulate(adjs, right, _unbroadcast(_binary("*", adj, primals[left]), primals[right]))
        elif entry.kind == "div":
            left, right = entry.inputs
            _accumulate(adjs, left, _unbroadcast(_binary("/", adj, primals[right]), primals[left]))
            numerator = _binary("*", adj, primals[left])
            denominator = _binary("*", primals[right], primals[right])
            _accumulate(adjs, right, _unbroadcast(_neg(_binary("/", numerator, denominator)), primals[right]))
        elif entry.kind == "fold":
            operand = entry.inputs[0]
            _accumulate(adjs, operand, _fill(adj, primals[operand]))
        elif entry.kind == "neg":
            _accumulate(adjs, entry.inputs[0], _neg(adj))
        elif entry.kind in {"reshape", "ravel"}:
            operand = entry.inputs[0]
            _accumulate(adjs, operand, _reshape(adj, primals[operand].shape))
        elif entry.kind == "transpose":
            _accumulate(adjs, entry.inputs[0], _transpose(adj))
        elif entry.kind == "reverse":
            _accumulate(adjs, entry.inputs[0], _view("reverse", adj))
        elif entry.kind == "take":
            operand = primals[entry.inputs[0]]
            count = int(entry.saved[1])
            zero_tail = _binary("*", _constant(0.0), _view("drop", operand, count))
            _accumulate(adjs, entry.inputs[0], _append(adj, zero_tail))
        elif entry.kind == "drop":
            operand = primals[entry.inputs[0]]
            count = int(entry.saved[1])
            zero_head = _binary("*", _constant(0.0), _view("take", operand, count))
            _accumulate(adjs, entry.inputs[0], _append(zero_head, adj))
        elif entry.kind == "append":
            left_count = int(entry.saved[1])
            _accumulate(adjs, entry.inputs[0], _view("take", adj, left_count))
            _accumulate(adjs, entry.inputs[1], _view("drop", adj, left_count))
        elif entry.kind == "subarray":
            operand = primals[entry.inputs[0]]
            offsets = tuple(int(o) for o in entry.saved[1])
            sizes = tuple(int(s) for s in entry.saved[2])
            _accumulate(adjs, entry.inputs[0], _pad_subarray(operand, adj, offsets, sizes))
        elif entry.kind == "rotate":
            shift = int(entry.saved[0])
            n = int(entry.saved[1])
            reverse_shift = (n - shift) % n
            _accumulate(adjs, entry.inputs[0], _rotate(adj, reverse_shift))
        elif entry.kind == "index":
            operand = primals[entry.inputs[0]]
            index_vals = tuple(int(v) for v in entry.saved[1])
            _accumulate(adjs, entry.inputs[0], _pad_index(operand, adj, index_vals))
        elif entry.kind == "select":
            cond = primals[entry.inputs[0]]
            _accumulate(adjs, entry.inputs[1], _If(cond, adj, _fill(_constant(0.0), primals[entry.inputs[1]]), adj.shape))
            _accumulate(adjs, entry.inputs[2], _If(cond, _fill(_constant(0.0), primals[entry.inputs[2]]), adj, adj.shape))
        elif entry.kind in {"const", "var"}:
            continue
        else:
            raise NotImplementedError(f"gradient source VJP: {entry.kind}")

    diff_idx = tape.input_indices[differentiate_input]
    gradient = adjs[diff_idx]
    if gradient is None:
        raise RuntimeError("AD source: input not found on tape")
    param_parts = " ".join(
        f"{name} {_source_type(shape)}" for name, shape in param_specs
    )
    if param_parts:
        param_parts = f"[{param_parts}]"
    body = _emit(gradient)
    return (
        f"(define/pi () ({function_name} {param_parts} {_source_type(param_specs[differentiate_input][1])}) "
        f"{body})"
    )


def _enumerate_params(param_specs, tape):
    return [(name, shape, idx) for (name, shape), idx in zip(param_specs, tape.input_indices)]


def _reconstruct_primals_multi(tape: EvalTape, name_map: dict[int, str]) -> list[_Expr]:
    primals: list[_Expr] = []
    for index, entry in enumerate(tape.entries):
        shape = _shape_of(tape.values[index])
        if entry.kind == "var":
            name = name_map.get(index, f"_unknown_{index}")
            expr = _Atom(name, shape)
        elif entry.kind == "const":
            expr = _constant_value(entry.saved[0], shape)
        elif entry.kind in {"add", "sub", "mul", "div"}:
            op = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[entry.kind]
            expr = _binary(op, primals[entry.inputs[0]], primals[entry.inputs[1]])
        elif entry.kind == "fold":
            operand = primals[entry.inputs[0]]
            expr = _Op("fold", operand, None, shape)
        elif entry.kind == "neg":
            expr = _neg(primals[entry.inputs[0]])
        elif entry.kind in {"reshape", "ravel"}:
            expr = _reshape(primals[entry.inputs[0]], shape)
        elif entry.kind == "transpose":
            expr = _transpose(primals[entry.inputs[0]])
        elif entry.kind == "reverse":
            expr = _view("reverse", primals[entry.inputs[0]])
        elif entry.kind in {"take", "drop"}:
            expr = _view(entry.kind, primals[entry.inputs[0]], int(entry.saved[1]))
        elif entry.kind == "append":
            expr = _append(primals[entry.inputs[0]], primals[entry.inputs[1]])
        elif entry.kind == "subarray":
            offsets = tuple(int(o) for o in entry.saved[1])
            sizes = tuple(int(s) for s in entry.saved[2])
            expr = _subarray_view(primals[entry.inputs[0]], offsets, sizes)
        elif entry.kind == "rotate":
            shift = int(entry.saved[0])
            expr = _rotate(primals[entry.inputs[0]], shift)
        elif entry.kind == "index":
            i = int(entry.saved[1][0])
            expr = _index(primals[entry.inputs[0]], i)
        elif entry.kind == "select":
            expr = _If(primals[entry.inputs[0]], primals[entry.inputs[1]], primals[entry.inputs[2]], shape)
        elif entry.kind == "inactive":
            inp_count = len(entry.inputs)
            if inp_count == 2 and entry.saved:
                expr = _inactive_binary(primals[entry.inputs[0]], primals[entry.inputs[1]], shape, str(entry.saved[0]))
            else:
                expr = _Atom("true", shape)
        else:
            raise NotImplementedError(f"gradient source primal: {entry.kind}")
        primals.append(expr)
    return primals
    if len(tape.input_indices) != 1:
        raise ValueError("gradient source currently requires exactly one tape input")

    input_idx = tape.input_indices[0]
    shape = tuple(int(dim) for dim in param_shape)
    if tuple(np.asarray(tape.values[input_idx]).shape) != shape:
        raise ValueError("parameter shape does not match the traced tape input")

    primals = _reconstruct_primals(tape, input_idx, param_name)
    adjs: list[_Expr | None] = [None] * len(tape.entries)
    output_shape = _shape_of(tape.values[-1])
    adjs[-1] = _constant(1.0, output_shape)

    for index in reversed(range(len(tape.entries))):
        adj = adjs[index]
        if adj is None:
            continue
        entry = tape.entries[index]
        if entry.kind == "add":
            _accumulate(
                adjs, entry.inputs[0], _unbroadcast(adj, primals[entry.inputs[0]])
            )
            _accumulate(
                adjs, entry.inputs[1], _unbroadcast(adj, primals[entry.inputs[1]])
            )
        elif entry.kind == "sub":
            _accumulate(
                adjs, entry.inputs[0], _unbroadcast(adj, primals[entry.inputs[0]])
            )
            _accumulate(
                adjs,
                entry.inputs[1],
                _unbroadcast(_neg(adj), primals[entry.inputs[1]]),
            )
        elif entry.kind == "mul":
            left, right = entry.inputs
            _accumulate(
                adjs,
                left,
                _unbroadcast(_binary("*", adj, primals[right]), primals[left]),
            )
            _accumulate(
                adjs,
                right,
                _unbroadcast(_binary("*", adj, primals[left]), primals[right]),
            )
        elif entry.kind == "div":
            left, right = entry.inputs
            _accumulate(
                adjs,
                left,
                _unbroadcast(_binary("/", adj, primals[right]), primals[left]),
            )
            numerator = _binary("*", adj, primals[left])
            denominator = _binary("*", primals[right], primals[right])
            _accumulate(
                adjs,
                right,
                _unbroadcast(_neg(_binary("/", numerator, denominator)), primals[right]),
            )
        elif entry.kind == "fold":
            operand = entry.inputs[0]
            _accumulate(adjs, operand, _fill(adj, primals[operand]))
        elif entry.kind == "neg":
            _accumulate(adjs, entry.inputs[0], _neg(adj))
        elif entry.kind in {"reshape", "ravel"}:
            operand = entry.inputs[0]
            _accumulate(adjs, operand, _reshape(adj, primals[operand].shape))
        elif entry.kind == "transpose":
            _accumulate(adjs, entry.inputs[0], _transpose(adj))
        elif entry.kind == "reverse":
            _accumulate(adjs, entry.inputs[0], _view("reverse", adj))
        elif entry.kind == "take":
            operand = primals[entry.inputs[0]]
            count = int(entry.saved[1])
            zero_tail = _binary("*", _constant(0.0), _view("drop", operand, count))
            _accumulate(adjs, entry.inputs[0], _append(adj, zero_tail))
        elif entry.kind == "drop":
            operand = primals[entry.inputs[0]]
            count = int(entry.saved[1])
            zero_head = _binary("*", _constant(0.0), _view("take", operand, count))
            _accumulate(adjs, entry.inputs[0], _append(zero_head, adj))
        elif entry.kind == "append":
            left_count = int(entry.saved[1])
            _accumulate(adjs, entry.inputs[0], _view("take", adj, left_count))
            _accumulate(adjs, entry.inputs[1], _view("drop", adj, left_count))
        elif entry.kind == "subarray":
            operand = primals[entry.inputs[0]]
            offsets = tuple(int(o) for o in entry.saved[1])
            sizes = tuple(int(s) for s in entry.saved[2])
            _accumulate(adjs, entry.inputs[0], _pad_subarray(operand, adj, offsets, sizes))
        elif entry.kind == "rotate":
            shift = int(entry.saved[0])
            n = int(entry.saved[1])
            reverse_shift = (n - shift) % n
            _accumulate(adjs, entry.inputs[0], _rotate(adj, reverse_shift))
        elif entry.kind == "index":
            operand = primals[entry.inputs[0]]
            index_vals = tuple(int(v) for v in entry.saved[1])
            _accumulate(adjs, entry.inputs[0], _pad_index(operand, adj, index_vals))
        elif entry.kind == "select":
            cond = primals[entry.inputs[0]]
            _accumulate(adjs, entry.inputs[1], _If(cond, adj, _fill(_constant(0.0), primals[entry.inputs[1]]), adj.shape))
            _accumulate(adjs, entry.inputs[2], _If(cond, _fill(_constant(0.0), primals[entry.inputs[2]]), adj, adj.shape))
        elif entry.kind in {"const", "var"}:
            continue
        else:
            raise NotImplementedError(f"gradient source VJP: {entry.kind}")

    gradient = adjs[input_idx]
    if gradient is None:
        raise RuntimeError("AD source: input not found on tape")
    param_type = _source_type(shape)
    body = _emit(gradient)
    return (
        f"(define/pi () ({function_name} [{param_name} {param_type}] {param_type}) "
        f"{body})"
    )


def generate_gradient_function_source(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    example_input: np.ndarray | None = None,
    *,
    gradient_name: str | None = None,
    include_prelude: bool = True,
    syntax: str = "ml",
) -> GradientSourceArtifact:
    """Specialize, trace, and generate source for a named unary or multi-param function."""
    from remora.lisp_reader import parse_lisp
    from remora.parser import parse_program
    from remora.prelude import with_prelude
    from remora.typechecker import TypeChecker, TypeEnv

    if len(param_types) < 1:
        raise ValueError("gradient source requires at least one parameter type")

    if example_input is None:
        trace_inputs = [_placeholder_input(pt) for pt in param_types]
    elif isinstance(example_input, np.ndarray) and len(param_types) == 1:
        trace_inputs = [np.asarray(example_input, dtype=np.float64)]
    else:
        raise ValueError("example input requires shape validation; use param_types alone")

    for pt, ti in zip(param_types, trace_inputs):
        _validate_example_input(pt, ti)

    program_source = (
        with_prelude(source) if include_prelude and syntax == "ml" else source
    )
    program = (
        parse_lisp(program_source)
        if syntax == "lisp"
        else parse_program(program_source)
    )
    checker = TypeChecker()
    env = TypeEnv()
    function: FuncDef | None = None
    for definition in program.definitions:
        _, env = checker.check_definition(definition, env)
        if isinstance(definition, FuncDef) and definition.name == function_name:
            function = definition
    if function is None:
        raise ValueError(f"function {function_name!r} is not defined")

    specialized = checker.specialize_top_level_function(function, param_types, env)
    if specialized.type.result != FLOAT:
        raise ValueError("gradient source requires a function with Float result")
    param_specs = [(name, tuple(ti.shape)) for (name, _pt), ti in zip(specialized.params, trace_inputs)]
    tape, _input_indices = _trace_function_body_multi(
        specialized.body, trace_inputs, [name for name, _pt in specialized.params]
    )
    generated_name = gradient_name or f"grad_{function_name.replace('-', '_')}"
    generated_source = generate_gradient_source(
        tape,
        param_specs,
        differentiate_input=0,
        function_name=generated_name,
    )
    return GradientSourceArtifact(
        generated_source,
        generated_name,
        param_types,
        tape,
        tape.input_indices[0] if tape.input_indices else 0,
    )


def _trace_function_body(body, param_name: str, example_input: np.ndarray):
    from remora.ad import trace_via_tape

    return trace_via_tape(body, param_name, example_input)


def _trace_function_body_multi(body, trace_inputs: list[np.ndarray], param_names: list[str]):
    from remora.ad import trace_via_tape_multi

    return trace_via_tape_multi(body, trace_inputs, param_names)


def _validate_example_input(param_type: RemoraType, example_input: np.ndarray) -> None:
    shape = tuple(np.asarray(example_input).shape)
    if isinstance(param_type, ScalarType):
        if param_type != FLOAT or shape:
            raise ValueError("example input does not match scalar Float parameter")
        return
    if isinstance(param_type, ArrayType) and param_type.element == FLOAT:
        if not all(isinstance(dim, StaticDim) for dim in param_type.shape):
            raise ValueError("gradient source requires a concrete parameter shape")
        expected = tuple(int(dim.value) for dim in param_type.shape)
        if shape != expected:
            raise ValueError(
                f"example input shape {shape} does not match parameter shape {expected}"
            )
        return
    raise ValueError("gradient source requires a scalar or array Float parameter")


def _placeholder_input(param_type: RemoraType) -> np.ndarray:
    """Build a deterministic nonzero trace input from a concrete Float type."""
    if isinstance(param_type, ScalarType):
        if param_type != FLOAT:
            raise ValueError("gradient source requires a Float parameter")
        return np.asarray(1.0, dtype=np.float64)
    if isinstance(param_type, ArrayType) and param_type.element == FLOAT:
        if not all(isinstance(dim, StaticDim) for dim in param_type.shape):
            raise ValueError("gradient source requires a concrete parameter shape")
        shape = tuple(int(dim.value) for dim in param_type.shape)
        return np.ones(shape, dtype=np.float64)
    raise ValueError("gradient source requires a scalar or array Float parameter")


def _reconstruct_primals(tape: EvalTape, input_idx: int, param_name: str) -> list[_Expr]:
    primals: list[_Expr] = []
    for index, entry in enumerate(tape.entries):
        shape = _shape_of(tape.values[index])
        if entry.kind == "var":
            if index != input_idx:
                raise ValueError("gradient source encountered an unsupported extra input")
            expr = _Atom(param_name, shape)
        elif entry.kind == "const":
            expr = _constant_value(entry.saved[0], shape)
        elif entry.kind in {"add", "sub", "mul", "div"}:
            op = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[entry.kind]
            expr = _binary(op, primals[entry.inputs[0]], primals[entry.inputs[1]])
        elif entry.kind == "fold":
            operand = primals[entry.inputs[0]]
            expr = _Op("fold", operand, None, shape)
        elif entry.kind == "neg":
            expr = _neg(primals[entry.inputs[0]])
        elif entry.kind in {"reshape", "ravel"}:
            expr = _reshape(primals[entry.inputs[0]], shape)
        elif entry.kind == "transpose":
            expr = _transpose(primals[entry.inputs[0]])
        elif entry.kind == "reverse":
            expr = _view("reverse", primals[entry.inputs[0]])
        elif entry.kind in {"take", "drop"}:
            expr = _view(entry.kind, primals[entry.inputs[0]], int(entry.saved[1]))
        elif entry.kind == "append":
            expr = _append(primals[entry.inputs[0]], primals[entry.inputs[1]])
        elif entry.kind == "subarray":
            offsets = tuple(int(o) for o in entry.saved[1])
            sizes = tuple(int(s) for s in entry.saved[2])
            expr = _subarray_view(primals[entry.inputs[0]], offsets, sizes)
        elif entry.kind == "rotate":
            shift = int(entry.saved[0])
            expr = _rotate(primals[entry.inputs[0]], shift)
        elif entry.kind == "index":
            i = int(entry.saved[1][0])
            expr = _index(primals[entry.inputs[0]], i)
        elif entry.kind == "select":
            expr = _If(
                primals[entry.inputs[0]],
                primals[entry.inputs[1]],
                primals[entry.inputs[2]],
                shape,
            )
        elif entry.kind == "inactive":
            inp_count = len(entry.inputs)
            if inp_count == 2 and entry.saved:
                expr = _inactive_binary(primals[entry.inputs[0]], primals[entry.inputs[1]], shape, str(entry.saved[0]))
            else:
                expr = _Atom("true", shape)
        else:
            raise NotImplementedError(f"gradient source primal: {entry.kind}")
        primals.append(expr)
    return primals


def _shape_of(value: object) -> Shape:
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _constant(value: float, shape: Shape = ()) -> _Atom:
    return _Atom(_format_float(value), shape)


def _constant_value(value: object, shape: Shape) -> _Atom:
    array = np.asarray(value)
    if array.ndim != 0:
        raise NotImplementedError("array constants are not supported in gradient source")
    return _constant(float(array), shape)


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("gradient source cannot serialize non-finite constants")
    text = repr(float(value))
    return text if "." in text or "e" in text.lower() else f"{text}.0"


def _source_type(shape: Shape) -> str:
    if not shape:
        return "Float"
    return f"(Array Float {' '.join(str(dim) for dim in shape)})"


def _binary(op: str, left: _Expr, right: _Expr) -> _Expr:
    result_shape = left.shape if len(left.shape) >= len(right.shape) else right.shape
    if op == "+":
        if _is_zero(left):
            return right
        if _is_zero(right):
            return left
        if left == right:
            return _binary("*", _constant(2.0), left)
    if op == "-":
        if _is_zero(right):
            return left
        if left == right and not result_shape:
            return _constant(0.0, result_shape)
    if op == "*":
        if not result_shape and (_is_zero(left) or _is_zero(right)):
            return _constant(0.0, result_shape)
        if _is_one(left):
            return right
        if _is_one(right):
            return left
        if isinstance(left, _Fill) and _is_one(left.value) and left.shape == right.shape:
            return right
        if isinstance(right, _Fill) and _is_one(right.value) and right.shape == left.shape:
            return left
    if op == "/":
        if not result_shape and _is_zero(left):
            return _constant(0.0, result_shape)
        if _is_one(right):
            return left
    return _Op(op, left, right, result_shape)


def _neg(expr: _Expr) -> _Expr:
    if _is_zero(expr):
        return expr
    if isinstance(expr, _Op) and expr.op == "neg":
        assert expr.right is None
        return expr.left
    return _binary("-", _constant(0.0), expr)


def _fill(value: _Expr, like: _Expr) -> _Expr:
    if value.shape == like.shape:
        return value
    return _Fill(value, like, like.shape)


def _reshape(value: _Expr, shape: Shape) -> _Expr:
    if value.shape == shape:
        return value
    return _Reshape(value, shape)


def _transpose(value: _Expr) -> _Expr:
    if len(value.shape) < 2:
        raise ValueError("transpose gradient requires rank at least two")
    shape = (value.shape[1], value.shape[0], *value.shape[2:])
    return _Transpose(value, shape)


def _view(kind: str, value: _Expr, count: int | None = None) -> _Expr:
    if kind == "reverse":
        return _View(kind, value, None, value.shape)
    if not value.shape or count is None:
        raise ValueError(f"{kind} gradient requires a non-scalar value and count")
    leading = count if kind == "take" else value.shape[0] - count
    return _View(kind, value, count, (leading, *value.shape[1:]))


def _append(left: _Expr, right: _Expr) -> _Expr:
    if not left.shape or not right.shape or left.shape[1:] != right.shape[1:]:
        raise ValueError("append gradient operands must have compatible array shapes")
    return _Append(left, right, (left.shape[0] + right.shape[0], *left.shape[1:]))


def _subarray_view(value: _Expr, offsets: tuple[int, ...], sizes: tuple[int, ...]) -> _Expr:
    shape = tuple(
        s if i == 0 else min(s, value.shape[i])  for i, s in enumerate(sizes)
    )
    return _SubarrayView(value, offsets, sizes, shape)


def _pad_subarray(operand: _Expr, adj: _Expr, offsets: tuple[int, ...], sizes: tuple[int, ...]) -> _Expr:
    if len(offsets) != 1:
        raise NotImplementedError(
            "subarray VJP: only rank-1 leading-dimension subarrays are supported"
        )
    n = operand.shape[0]
    start = offsets[0]
    size = sizes[0]
    tail_count = n - start - size
    result = adj
    if tail_count > 0:
        zero_tail = _binary("*", _constant(0.0), _view("drop", operand, start + size))
        result = _append(result, zero_tail)
    if start > 0:
        zero_head = _binary("*", _constant(0.0), _view("take", operand, start))
        result = _append(zero_head, result)
    return result


def _rotate(value: _Expr, shift: int) -> _Expr:
    return _Rotate(value, shift, value.shape)


def _index(value: _Expr, idx: int) -> _Expr:
    return _Index(value, idx, ())


def _pad_index(operand: _Expr, adj: _Expr, index_vals: tuple[int, ...]) -> _Expr:
    if len(index_vals) != 1:
        raise NotImplementedError(
            "index VJP: only rank-1 single-index is supported"
        )
    n = operand.shape[0]
    i = index_vals[0]
    zero_array = _binary("*", _constant(0.0), operand)
    return _ScatterAdd(zero_array, i, adj, operand.shape)


def _inactive_binary(left: _Expr, right: _Expr, shape: Shape, op: str) -> _Op:
    return _Op(op, left, right, shape)


def _unbroadcast(expr: _Expr, target: _Expr) -> _Expr:
    if expr.shape == target.shape:
        return expr
    if target.shape == ():
        return _Op("fold", expr, None, ())
    raise NotImplementedError(
        f"gradient source cannot unbroadcast {expr.shape} to {target.shape}"
    )


def _pair_type_string(param_specs: list[tuple[str, tuple[int, ...]]]) -> str:
    """Build nested Pair type string: (Pair Float (Pair (Array Float 3) Float))."""
    types = [_source_type(shape) for _, shape in param_specs]
    result = types[-1]
    for t in reversed(types[:-1]):
        result = f"(Pair {t} {result})"
    return result


def _accumulate(adjs: list[_Expr | None], index: int, contribution: _Expr) -> None:
    current = adjs[index]
    adjs[index] = contribution if current is None else _binary("+", current, contribution)


def _is_constant(expr: _Expr, value: float) -> bool:
    if not isinstance(expr, _Atom) or expr.shape:
        return False
    try:
        return float(expr.text) == value
    except ValueError:
        return False


def _is_zero(expr: _Expr) -> bool:
    return _is_constant(expr, 0.0)


def _is_one(expr: _Expr) -> bool:
    return _is_constant(expr, 1.0)


def _emit(expr: _Expr) -> str:
    if isinstance(expr, _Atom):
        return expr.text
    if isinstance(expr, _Fill):
        return _emit(_binary("+", _binary("*", _constant(0.0), expr.like), expr.value))
    if isinstance(expr, _Reshape):
        shape = " ".join(str(dim) for dim in expr.shape)
        return f"(reshape {_emit(expr.value)} [{shape}])"
    if isinstance(expr, _Transpose):
        return f"(transpose {_emit(expr.value)})"
    if isinstance(expr, _View):
        if expr.kind == "reverse":
            return f"(reverse {_emit(expr.value)})"
        return f"({expr.kind} {expr.count} {_emit(expr.value)})"
    if isinstance(expr, _Append):
        return f"(append {_emit(expr.left)} {_emit(expr.right)})"
    if isinstance(expr, _SubarrayView):
        off = " ".join(str(o) for o in expr.offsets)
        sz = " ".join(str(s) for s in expr.sizes)
        if len(expr.offsets) == 1:
            off = str(expr.offsets[0])
            sz = str(expr.sizes[0])
        return f"(subarray {_emit(expr.value)} [{off}] [{sz}])"
    if isinstance(expr, _Rotate):
        return f"(rotate {_emit(expr.value)} {expr.shift})"
    if isinstance(expr, _Index):
        return f"(index {_emit(expr.value)} {expr.idx})"
    if isinstance(expr, _ScatterAdd):
        if isinstance(expr.index, int):
            return f"(scatter-add {_emit(expr.target)} {expr.index} {_emit(expr.update)})"
        return f"(scatter-add {_emit(expr.target)} {_emit(expr.index)} {_emit(expr.update)})"
    if isinstance(expr, _If):
        return f"(if {_emit(expr.condition)} {_emit(expr.then_expr)} {_emit(expr.else_expr)})"
    if isinstance(expr, _Pair):
        return f"(pair {_emit(expr.left)} {_emit(expr.right)})"
    if expr.op == "fold":
        return f"(fold + 0.0 {_emit(expr.left)})"
    if expr.op == "neg":
        return f"(- {_emit(expr.left)})"
    assert expr.right is not None
    left = _emit(expr.left)
    right = _emit(expr.right)
    if expr.left.shape == () and expr.right.shape:
        if expr.op in {"+", "*"}:
            return f"(map ({expr.op} {left}) {right})"
        return f"(map (lambda (v) ({expr.op} {left} v)) {right})"
    if expr.left.shape and expr.right.shape == ():
        return f"(map (lambda (v) ({expr.op} v {right})) {left})"
    if expr.left.shape or expr.right.shape:
        return f"(map {expr.op} {left} {right})"
    return f"({expr.op} {left} {right})"
