import pytest

from remora.parser import parse_expr, parse_program
from remora.typechecker import (
    TypeChecker,
    TypedApp,
    TypedCast,
    TypedFold,
    TypedIf,
    TypedIndex,
    TypedLet,
    TypedMap,
    TypedRank,
    TypedRightSection,
    TypedShape,
)
from remora.limits import MAX_DENSE_RANK
from remora.types import BOOL, FLOAT, INT, ArrayType, RemoraTypeError, StaticDim


def infer(source: str):
    return TypeChecker().infer(parse_expr(source))


def shape_values(array_type: ArrayType) -> tuple[int, ...]:
    return tuple(dim.value for dim in array_type.shape)


def let_body(typed: TypedLet):
    assert isinstance(typed, TypedLet)
    return typed.body


def nested_scalar_literal(rank: int, value: str = "1") -> str:
    return "[" * rank + value + "]" * rank


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


def test_type_errors_include_source_location():
    program = parse_program("iota -1", "bad.remora")

    with pytest.raises(RemoraTypeError, match=r"bad\.remora:1:6"):
        TypeChecker().check_program(program)


def test_shape_expression_typechecks_from_static_array_type():
    typed = infer("shape [[1, 2], [3, 4]]")

    assert isinstance(typed, TypedShape)
    assert typed.type == ArrayType(INT, (StaticDim(2),))


def test_shape_of_scalar_typechecks_as_empty_int_vector():
    typed = infer("shape 42")

    assert isinstance(typed, TypedShape)
    assert typed.type == ArrayType(INT, (StaticDim(0),))


def test_rank_expression_typechecks_as_int():
    typed = infer("rank [[1, 2], [3, 4]]")

    assert isinstance(typed, TypedRank)
    assert typed.type == INT


def test_index_expression_typechecks_for_full_and_partial_indices():
    scalar = infer("[[1, 2], [3, 4]][1, 0]")
    row = infer("[[1, 2], [3, 4]][1]")

    assert isinstance(scalar, TypedIndex)
    assert scalar.type == INT
    assert isinstance(row, TypedIndex)
    assert row.type == ArrayType(INT, (StaticDim(2),))


def test_index_expression_rejects_non_int_indices():
    with pytest.raises(RemoraTypeError, match="expected int"):
        infer("[[1, 2], [3, 4]][1.0]")


def test_index_expression_rejects_too_many_indices():
    with pytest.raises(RemoraTypeError, match="too many indices"):
        infer("[1, 2][0, 0]")


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


def test_binary_map_operator_over_matching_vectors():
    typed = let_body(let_body(
        infer("let xs = [1, 2] in let ys = [3, 4] in map (*) xs ys")
    ))

    assert isinstance(typed, TypedMap)
    assert typed.type == ArrayType(INT, (StaticDim(2),))
    assert len(typed.arrays) == 2


def test_binary_map_lambda_over_matching_vectors():
    typed = let_body(let_body(
        infer("let xs = [1, 2] in let ys = [3, 4] in map (\\x y -> x + y) xs ys")
    ))

    assert isinstance(typed, TypedMap)
    assert typed.type == ArrayType(INT, (StaticDim(2),))


def test_binary_map_rejects_mismatched_shapes():
    with pytest.raises(RemoraTypeError, match="matching shapes"):
        infer("let xs = [1, 2] in let ys = [3, 4, 5] in map (*) xs ys")


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


def test_fold_on_matrix_returns_row_array():
    typed = let_body(let_body(
        infer("let init = [0, 0] in let xs = [[1, 2], [3, 4]] in fold (+) init xs")
    ))
    assert isinstance(typed, TypedFold)
    assert typed.reduction_dim == StaticDim(2)
    assert typed.type == ArrayType(INT, (StaticDim(2),))


def test_fold_on_rank_3_returns_rank_2_array():
    typed = let_body(let_body(
        infer(
            "let init = [[0], [0]] in "
            "let xs = [[[1], [2]], [[3], [4]]] in "
            "fold (+) init xs"
        )
    ))
    assert isinstance(typed, TypedFold)
    assert typed.reduction_dim == StaticDim(2)
    assert typed.type == ArrayType(INT, (StaticDim(2), StaticDim(1)))


def test_map_operator_section_over_iota_promotes_to_float_array():
    typed = infer("map (* 2.0) (iota 10)")

    assert isinstance(typed, TypedMap)
    assert typed.type == ArrayType(FLOAT, (StaticDim(10),))


def test_map_right_operator_section_typechecks():
    typed = infer("map (2.0 *) (iota 10)")

    assert isinstance(typed, TypedMap)
    assert isinstance(typed.func, TypedRightSection)
    assert typed.type == ArrayType(FLOAT, (StaticDim(10),))


def test_division_operator_section_rejects_bool_operand():
    with pytest.raises(RemoraTypeError, match="numeric"):
        infer("let xs = [true, false] in map (/ true) xs")


def test_division_operator_func_rejects_bool_fold_operand():
    with pytest.raises(RemoraTypeError, match="numeric"):
        infer("let xs = [true, false] in fold (/) 0.0 xs")


def test_numeric_promotion_inserts_typed_cast():
    typed = infer("1 + 2.0")

    assert isinstance(typed, TypedApp)
    assert typed.type == FLOAT
    assert isinstance(typed.args[0], TypedCast)
    assert typed.args[0].from_type == INT
    assert typed.args[0].to_type == FLOAT


def test_if_preserves_typed_branches():
    typed = infer("if true then 1 else 2")

    assert isinstance(typed, TypedIf)
    assert typed.condition.type == BOOL
    assert typed.then_branch.type == INT
    assert typed.else_branch.type == INT


def test_direct_local_lambda_application_typechecks():
    typed = infer("let add1 = \\x -> x + 1 in add1 41")

    assert isinstance(typed, TypedLet)
    assert typed.value.type.result == INT
    assert typed.body.type == INT


def test_rank_4_array_literal_typing_is_in_dense_core_scope():
    typed = infer(nested_scalar_literal(4))

    assert typed.type == ArrayType(
        INT,
        (StaticDim(1), StaticDim(1), StaticDim(1), StaticDim(1)),
    )


def test_rank_10_array_literal_typing_is_in_dense_core_scope():
    typed = infer(nested_scalar_literal(MAX_DENSE_RANK))

    assert typed.type == ArrayType(
        INT,
        tuple(StaticDim(1) for _axis in range(MAX_DENSE_RANK)),
    )


def test_rank_above_dense_core_limit_is_rejected():
    with pytest.raises(RemoraTypeError, match="rank limit"):
        infer(nested_scalar_literal(MAX_DENSE_RANK + 1))


def test_milestone_m2_expression_typechecks():
    typed = infer("fold (+) 0.0 (map (\\x -> x * x) (iota 10))")

    assert isinstance(typed, TypedFold)
    assert typed.type == FLOAT


def test_program_with_value_definition_typechecks():
    program = parse_program("def xs = iota 4\nmap (* 2.0) xs")
    typed = TypeChecker().check_program(program)

    assert typed.type == ArrayType(FLOAT, (StaticDim(4),))


def test_top_level_function_definition_typechecks_at_direct_call_site():
    program = parse_program("def f x = x\nf 1")
    typed = TypeChecker().check_program(program)

    assert typed.type == INT
    assert isinstance(typed.body, TypedApp)
    assert typed.definitions[0].type is None


def test_top_level_function_definition_can_be_used_as_map_callable():
    program = parse_program("def double x = x * 2\nmap double (iota 4)")
    typed = TypeChecker().check_program(program)

    assert typed.type == ArrayType(INT, (StaticDim(4),))


def test_recursive_function_definition_is_deferred():
    program = parse_program("def f x = f x\nf 1")

    with pytest.raises(RemoraTypeError, match="recursive"):
        TypeChecker().check_program(program)
