import pytest

from remora.parser import parse_expr, parse_program
from remora.typechecker import TypeChecker, TypedApp, TypedCast, TypedFold, TypedLet, TypedMap
from remora.types import BOOL, FLOAT, INT, ArrayType, RemoraTypeError, StaticDim


def infer(source: str):
    return TypeChecker().infer(parse_expr(source))


def shape_values(array_type: ArrayType) -> tuple[int, ...]:
    return tuple(dim.value for dim in array_type.shape)


def let_body(typed: TypedLet):
    assert isinstance(typed, TypedLet)
    return typed.body


def test_scalar_literal_typing():
    assert infer("1").type == INT
    assert infer("1.0").type == FLOAT
    assert infer("true").type == BOOL


def test_rank_1_array_literal_typing():
    typed = infer("[1, 2, 3]")

    assert typed.type == ArrayType(INT, (StaticDim(3),))


def test_rank_2_and_rank_3_array_literal_typing():
    rank2 = infer("[[1.0, 2.0], [3.0, 4.0]]")
    rank3 = infer("[[[1], [2]], [[3], [4]]]")

    assert rank2.type == ArrayType(FLOAT, (StaticDim(2), StaticDim(2)))
    assert rank3.type == ArrayType(INT, (StaticDim(2), StaticDim(2), StaticDim(1)))


def test_mismatched_array_element_types_error():
    with pytest.raises(RemoraTypeError, match="expected"):
        infer("[1, 2.0]")


def test_mismatched_nested_array_shapes_error():
    with pytest.raises(RemoraTypeError, match="expected"):
        infer("[[1, 2], [3]]")


def test_iota_has_static_rank_1_int_array_type():
    typed = infer("iota 10")

    assert typed.type == ArrayType(INT, (StaticDim(10),))


def test_map_scalar_lambda_over_rank_1_array_promotes_result():
    typed = let_body(infer("let xs = [1.0, 2.0, 3.0] in map (\\x -> x + 1.0) xs"))

    assert isinstance(typed, TypedMap)
    assert typed.type == ArrayType(FLOAT, (StaticDim(3),))
    assert shape_values(ArrayType(FLOAT, typed.frame_shape)) == (3,)


def test_map_scalar_lambda_over_rank_2_and_rank_3_arrays():
    rank2 = let_body(
        infer("let xs = [[1.0, 2.0], [3.0, 4.0]] in map (\\x -> x * 2.0) xs")
    )
    rank3 = let_body(infer("let xs = [[[1], [2]], [[3], [4]]] in map (\\x -> x + 1) xs"))

    assert isinstance(rank2, TypedMap)
    assert rank2.type == ArrayType(FLOAT, (StaticDim(2), StaticDim(2)))
    assert isinstance(rank3, TypedMap)
    assert rank3.type == ArrayType(INT, (StaticDim(2), StaticDim(2), StaticDim(1)))


def test_map_vector_lambda_using_fold_returns_frame_type():
    typed = let_body(
        infer(
            "let xs = [[1.0, 2.0], [3.0, 4.0]] in map (\\row -> fold (+) 0.0 row) xs"
        )
    )

    assert isinstance(typed, TypedMap)
    assert typed.cell_shape == (StaticDim(2),)
    assert typed.type == ArrayType(FLOAT, (StaticDim(2),))


def test_fold_on_vector_returns_scalar():
    typed = let_body(infer("let xs = [1.0, 2.0, 3.0] in fold (+) 0.0 xs"))

    assert isinstance(typed, TypedFold)
    assert typed.reduction_dim == StaticDim(3)
    assert typed.type == FLOAT


def test_map_operator_section_over_iota_promotes_to_float_array():
    typed = infer("map (* 2.0) (iota 10)")

    assert isinstance(typed, TypedMap)
    assert typed.type == ArrayType(FLOAT, (StaticDim(10),))


def test_numeric_promotion_inserts_typed_cast():
    typed = infer("1 + 2.0")

    assert isinstance(typed, TypedApp)
    assert typed.type == FLOAT
    assert isinstance(typed.args[0], TypedCast)
    assert typed.args[0].from_type == INT
    assert typed.args[0].to_type == FLOAT


def test_rank_4_result_is_rejected():
    with pytest.raises(RemoraTypeError, match="rank limit"):
        infer("[[[[1]]]]")


def test_milestone_m2_expression_typechecks():
    typed = infer("fold (+) 0.0 (map (\\x -> x * x) (iota 10))")

    assert isinstance(typed, TypedFold)
    assert typed.type == FLOAT


def test_program_with_value_definition_typechecks():
    program = parse_program("def xs = iota 4\nmap (* 2.0) xs")
    typed = TypeChecker().check_program(program)

    assert typed.type == ArrayType(FLOAT, (StaticDim(4),))


def test_function_definition_inference_is_deferred():
    program = parse_program("def f x = x\nf 1")

    with pytest.raises(RemoraTypeError, match="function definition"):
        TypeChecker().check_program(program)
