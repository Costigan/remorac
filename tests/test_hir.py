import pytest

from remora.hir import (
    HIRArrayLit,
    HIRCast,
    HIRFold,
    HIRIndex,
    HIRLoweringError,
    HIRIota,
    HIRLambda,
    HIRLet,
    HIRLit,
    HIRMap,
    HIRPrimCallable,
    HIRPrimOp,
    HIRProgram,
    HIRVar,
    lower_to_hir,
)
from remora.parser import parse_program
from remora.typechecker import TypeChecker
from remora.types import FLOAT, INT, ArrayType, StaticDim


def lower_program_source(source: str):
    typed = TypeChecker().check_program(parse_program(source))
    return lower_to_hir(typed)


def test_lowers_iota():
    program = lower_program_source("iota 10")

    assert isinstance(program, HIRProgram)
    assert isinstance(program.main, HIRIota)
    assert program.main.size == StaticDim(10)
    assert program.return_type == ArrayType(INT, (StaticDim(10),))


def test_lowers_shape_and_rank_to_static_constants():
    shape_program = lower_program_source("shape [[1, 2], [3, 4]]")
    rank_program = lower_program_source("rank [[1, 2], [3, 4]]")
    scalar_shape_program = lower_program_source("shape 42")

    assert isinstance(shape_program.main, HIRArrayLit)
    assert [
        element.value
        for element in shape_program.main.elements
        if isinstance(element, HIRLit)
    ] == [2, 2]
    assert shape_program.return_type == ArrayType(INT, (StaticDim(2),))

    assert isinstance(rank_program.main, HIRLit)
    assert rank_program.main.value == 2
    assert rank_program.return_type == INT

    assert isinstance(scalar_shape_program.main, HIRArrayLit)
    assert scalar_shape_program.main.elements == []
    assert scalar_shape_program.return_type == ArrayType(INT, (StaticDim(0),))


def test_lowers_index_expression():
    program = lower_program_source("[[1, 2], [3, 4]][1, 0]")

    assert isinstance(program.main, HIRIndex)
    assert isinstance(program.main.array, HIRArrayLit)
    assert [
        index.value
        for index in program.main.indices
        if isinstance(index, HIRLit)
    ] == [1, 0]
    assert program.return_type == INT


def test_lowers_array_literal_with_typed_elements():
    program = lower_program_source("[1, 2, 3]")

    assert isinstance(program.main, HIRArrayLit)
    assert [element.value for element in program.main.elements if isinstance(element, HIRLit)] == [
        1,
        2,
        3,
    ]


def test_lowers_numeric_casts_explicitly():
    program = lower_program_source("1 + 2.0")

    assert isinstance(program.main, HIRPrimOp)
    assert program.main.op == "+f"
    assert isinstance(program.main.args[0], HIRCast)
    assert program.main.args[0].from_type == INT
    assert program.main.args[0].to_type == FLOAT


def test_lowers_scalar_map_with_lambda():
    program = lower_program_source("let xs = [1.0, 2.0] in map (\\x -> x * 2.0) xs")

    assert isinstance(program.main, HIRLet)
    assert isinstance(program.main.body, HIRMap)
    map_node = program.main.body
    assert map_node.frame_shape == (StaticDim(2),)
    assert map_node.cell_shape == ()
    assert isinstance(map_node.func, HIRLambda)
    assert isinstance(map_node.func.body, HIRPrimOp)
    assert map_node.func.body.op == "*f"


def test_lowers_binary_map_shape_metadata():
    program = lower_program_source(
        "let xs = [1, 2] in let ys = [3, 4] in map (*) xs ys"
    )

    assert isinstance(program.main, HIRLet)
    inner = program.main.body
    assert isinstance(inner, HIRLet)
    assert isinstance(inner.body, HIRMap)
    map_node = inner.body
    assert len(map_node.arrays) == 2
    assert map_node.frame_shape == (StaticDim(2),)
    assert map_node.cell_shape == ()
    assert map_node.result_type == ArrayType(INT, (StaticDim(2),))


def test_lowers_vector_cell_map_shape_metadata():
    program = lower_program_source(
        "let xs = [[1.0, 2.0], [3.0, 4.0]] in map (\\row -> fold (+) 0.0 row) xs"
    )

    assert isinstance(program.main, HIRLet)
    assert isinstance(program.main.body, HIRMap)
    map_node = program.main.body
    assert map_node.frame_shape == (StaticDim(2),)
    assert map_node.cell_shape == (StaticDim(2),)
    assert map_node.result_type == ArrayType(FLOAT, (StaticDim(2),))


def test_lowers_fold_with_primitive_callable():
    program = lower_program_source("let xs = [1.0, 2.0, 3.0] in fold (+) 0.0 xs")

    assert isinstance(program.main, HIRLet)
    assert isinstance(program.main.body, HIRFold)
    fold = program.main.body
    assert fold.reduction_dim == StaticDim(3)
    assert isinstance(fold.func, HIRPrimCallable)
    assert fold.func.op == "+"
    assert fold.result_type == FLOAT


def test_lowers_array_cell_fold_with_primitive_callable():
    program = lower_program_source(
        "let init = [0, 0] in let xs = [[1, 2], [3, 4]] in fold (+) init xs"
    )

    assert isinstance(program.main, HIRLet)
    inner = program.main.body
    assert isinstance(inner, HIRLet)
    assert isinstance(inner.body, HIRFold)
    fold = inner.body
    assert fold.reduction_dim == StaticDim(2)
    assert isinstance(fold.func, HIRPrimCallable)
    assert fold.func.op == "+"
    assert fold.result_type == ArrayType(INT, (StaticDim(2),))


def test_lowers_operator_section_bound_argument():
    program = lower_program_source("map (* 2.0) (iota 10)")

    assert isinstance(program.main, HIRMap)
    assert isinstance(program.main.func, HIRPrimCallable)
    assert program.main.func.op == "*"
    assert isinstance(program.main.func.left_arg, HIRLit)
    assert program.main.func.left_arg.value == 2.0
    assert program.main.result_type == ArrayType(FLOAT, (StaticDim(10),))


def test_lowers_top_level_value_definitions_as_lets():
    program = lower_program_source("def xs = iota 4\nmap (* 2.0) xs")

    assert isinstance(program.main, HIRLet)
    assert program.main.name == "xs"
    assert isinstance(program.main.value, HIRIota)
    assert isinstance(program.main.body, HIRMap)
    assert isinstance(program.main.body.array, HIRVar)
    assert program.main.body.array.name == "xs"


def test_definition_only_program_is_rejected_by_hir_lowering():
    typed = TypeChecker().check_program(parse_program("def xs = iota 4"))

    with pytest.raises(HIRLoweringError, match="definition-only"):
        lower_to_hir(typed)


def test_lowers_m2_milestone_expression():
    program = lower_program_source("fold (+) 0.0 (map (\\x -> x * x) (iota 10))")

    assert isinstance(program.main, HIRFold)
    assert program.main.result_type == FLOAT
    assert isinstance(program.main.array, HIRMap)
    assert program.main.array.result_type == ArrayType(INT, (StaticDim(10),))
