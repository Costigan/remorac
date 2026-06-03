import pytest
from lark import LarkError

from remora.ast_nodes import (
    AppExpr,
    ArrayLit,
    FloatLit,
    FoldExpr,
    FuncDef,
    IfExpr,
    IntLit,
    IotaExpr,
    IndexExpr,
    LambdaExpr,
    LeftSectionExpr,
    LetExpr,
    MapExpr,
    OperatorFuncExpr,
    Program,
    ValDef,
    VarExpr,
)
from remora.parser import parse_definition, parse_expr, parse_program, parse_repl_input


def test_integer_and_float_literals():
    assert parse_expr("123").value == 123
    assert parse_expr("1.5").value == 1.5


def test_array_literals():
    expr = parse_expr("[1, 2, 3]")

    assert isinstance(expr, ArrayLit)
    assert [element.value for element in expr.elements] == [1, 2, 3]


def test_lambda_with_multiple_parameters():
    expr = parse_expr("\\x y -> x + y")

    assert isinstance(expr, LambdaExpr)
    assert expr.params == ["x", "y"]
    assert isinstance(expr.body, AppExpr)
    assert expr.body.func.name == "+"


def test_let_binding():
    expr = parse_expr("let x = 1 in x")

    assert isinstance(expr, LetExpr)
    assert expr.name == "x"
    assert isinstance(expr.value, IntLit)
    assert isinstance(expr.body, VarExpr)


def test_map_expression_with_operator_section():
    expr = parse_expr("map (* 2.0) xs")

    assert isinstance(expr, MapExpr)
    assert isinstance(expr.func, LeftSectionExpr)
    assert expr.func.op == "*"
    assert isinstance(expr.array, VarExpr)


def test_map_accepts_sliced_operand():
    expr = parse_expr("map (* 2) xs[1:3]")

    assert isinstance(expr, MapExpr)
    assert isinstance(expr.array, IndexExpr)


def test_binary_map_expression():
    expr = parse_expr("map (*) xs ys")

    assert isinstance(expr, MapExpr)
    assert isinstance(expr.func, OperatorFuncExpr)
    assert [array.name for array in expr.arrays if isinstance(array, VarExpr)] == ["xs", "ys"]


def test_fold_expression():
    expr = parse_expr("fold (+) 0.0 xs")

    assert isinstance(expr, FoldExpr)
    assert isinstance(expr.func, OperatorFuncExpr)
    assert expr.func.op == "+"
    assert isinstance(expr.init, FloatLit)


def test_iota_expression():
    expr = parse_expr("iota 10")

    assert isinstance(expr, IotaExpr)
    assert expr.size.value == 10


def test_function_application_single_and_curried():
    single = parse_expr("f x")
    curried = parse_expr("f x y")

    assert isinstance(single, AppExpr)
    assert single.func.name == "f"
    assert len(single.args) == 1
    assert isinstance(curried, AppExpr)
    assert curried.func.name == "f"
    assert [arg.name for arg in curried.args] == ["x", "y"]


def test_top_level_definitions_and_program_body():
    program = parse_program("def scale x = x * 2.0\nscale 3.0")

    assert isinstance(program, Program)
    assert len(program.definitions) == 1
    assert isinstance(program.definitions[0], FuncDef)
    assert isinstance(program.body, AppExpr)


def test_value_definition():
    definition = parse_definition("def x = 1")

    assert isinstance(definition, ValDef)
    assert definition.name == "x"


def test_nested_expressions():
    expr = parse_expr("let x = map (\\a -> a + 1.0) (iota 10) in fold (+) 0.0 x")

    assert isinstance(expr, LetExpr)
    assert isinstance(expr.value, MapExpr)
    assert isinstance(expr.body, FoldExpr)


def test_infix_operator_precedence():
    expr = parse_expr("1 + 2 * 3")

    assert isinstance(expr, AppExpr)
    assert expr.func.name == "+"
    assert isinstance(expr.args[1], AppExpr)
    assert expr.args[1].func.name == "*"


def test_if_then_else():
    expr = parse_expr("if true then 1 else 2")

    assert isinstance(expr, IfExpr)
    assert expr.condition.value is True
    assert expr.then_branch.value == 1
    assert expr.else_branch.value == 2


def test_repl_input_prefers_definition_then_expression():
    definition = parse_repl_input("def x = 1")
    expression = parse_repl_input("x")

    assert isinstance(definition, ValDef)
    assert isinstance(expression, VarExpr)


def test_array_literal_after_atom_currently_parses_as_index():
    expr = parse_expr("xs [1]")

    assert isinstance(expr, IndexExpr)
    assert isinstance(expr.array, VarExpr)
    assert expr.array.name == "xs"
    assert [index.value for index in expr.indices] == [1]


def test_parser_records_source_locations():
    program = parse_program("-- comment\nlet x = 1 in\nx + 2", "sample.remora")

    assert program.body is not None
    assert program.body.loc.file == "sample.remora"
    assert program.body.loc.line == 2
    assert program.body.loc.col == 5


def test_malformed_syntax_errors():
    with pytest.raises(LarkError):
        parse_expr("let x = in 1")
