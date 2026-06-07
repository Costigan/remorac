"""Tests for the Lisp s-expression reader (remora/lisp_reader.py)."""

import pytest
from lark import LarkError

from remora.ast_nodes import (
    AppExpr,
    ArrayLit,
    BoolLit,
    FoldExpr,
    FuncDef,
    IfExpr,
    IndexExpr,
    IntLit,
    IotaExpr,
    LambdaExpr,
    LeftSectionExpr,
    LetExpr,
    MapExpr,
    OperatorFuncExpr,
    Program,
    RankExpr,
    RavelExpr,
    ReshapeExpr,
    ReverseExpr,
    RightSectionExpr,
    ShapeExpr,
    TakeExpr,
    TransposeExpr,
    DropExpr,
    ValDef,
    VarExpr,
)
from remora.lisp_reader import parse_lisp


# ── Literals ────────────────────────────────────────────────────────────────

def test_integer_literal():
    p = parse_lisp("42")
    assert isinstance(p.body, IntLit)
    assert p.body.value == 42


def test_negative_integer_literal():
    p = parse_lisp("-5")
    assert isinstance(p.body, IntLit)
    assert p.body.value == -5


def test_float_literal():
    p = parse_lisp("3.14")
    from remora.ast_nodes import FloatLit
    assert isinstance(p.body, FloatLit)
    assert p.body.value == 3.14


def test_boolean_literals():
    p = parse_lisp("#t")
    assert isinstance(p.body, BoolLit)
    assert p.body.value is True
    p = parse_lisp("#f")
    assert isinstance(p.body, BoolLit)
    assert p.body.value is False


def test_array_literal():
    p = parse_lisp("[1 2 3]")
    assert isinstance(p.body, ArrayLit)
    assert [e.value for e in p.body.elements] == [1, 2, 3]


def test_nested_array_literal():
    p = parse_lisp("[[1 2] [3 4]]")
    assert isinstance(p.body, ArrayLit)
    assert len(p.body.elements) == 2
    assert isinstance(p.body.elements[0], ArrayLit)
    assert [e.value for e in p.body.elements[0].elements] == [1, 2]


def test_empty_array_literal():
    p = parse_lisp("[]")
    assert isinstance(p.body, ArrayLit)
    assert p.body.elements == []


# ── Parenthesized expression ────────────────────────────────────────────────

def test_parenthesized_single_element():
    p = parse_lisp("(42)")
    assert isinstance(p.body, IntLit)
    assert p.body.value == 42


# ── Let and If ──────────────────────────────────────────────────────────────

def test_let_expression():
    p = parse_lisp("(:: x 5 (+ x 1))")
    assert isinstance(p.body, LetExpr)
    assert p.body.name == "x"
    assert isinstance(p.body.value, IntLit)
    assert p.body.value.value == 5
    assert isinstance(p.body.body, AppExpr)


def test_if_expression():
    p = parse_lisp("(if (< 1 2) 10 20)")
    assert isinstance(p.body, IfExpr)
    assert isinstance(p.body.condition, AppExpr)
    assert isinstance(p.body.then_branch, IntLit)
    assert p.body.then_branch.value == 10
    assert isinstance(p.body.else_branch, IntLit)
    assert p.body.else_branch.value == 20


# ── Arithmetic and comparison ───────────────────────────────────────────────

def test_binary_addition():
    p = parse_lisp("(+ 1 2)")
    assert isinstance(p.body, AppExpr)
    assert isinstance(p.body.func, VarExpr)
    assert p.body.func.name == "+"
    assert len(p.body.args) == 2
    assert p.body.args[0].value == 1
    assert p.body.args[1].value == 2


def test_comparison():
    p = parse_lisp("(< x 5)")
    assert isinstance(p.body, AppExpr)
    assert p.body.func.name == "<"
    assert isinstance(p.body.args[0], VarExpr)
    assert p.body.args[0].name == "x"
    assert p.body.args[1].value == 5


def test_boolean_and():
    p = parse_lisp("(&& a b)")
    assert isinstance(p.body, AppExpr)
    assert p.body.func.name == "&&"


def test_boolean_or():
    p = parse_lisp("(|| a b)")
    assert isinstance(p.body, AppExpr)
    assert p.body.func.name == "||"


def test_operator_chain_left_associative():
    p = parse_lisp("(+ 1 2 3)")
    assert isinstance(p.body, AppExpr)
    assert p.body.func.name == "+"
    assert isinstance(p.body.args[0], AppExpr)
    assert p.body.args[0].args[0].value == 1
    assert p.body.args[0].args[1].value == 2
    assert p.body.args[1].value == 3


# ── Operator sections ───────────────────────────────────────────────────────

def test_left_section():
    p = parse_lisp("(+ 2)")
    assert isinstance(p.body, LeftSectionExpr)
    assert p.body.op == "+"
    assert isinstance(p.body.arg, IntLit)
    assert p.body.arg.value == 2


def test_right_section():
    p = parse_lisp("(2 +)")
    assert isinstance(p.body, RightSectionExpr)
    assert p.body.op == "+"
    assert isinstance(p.body.arg, IntLit)
    assert p.body.arg.value == 2


def test_left_section_comparison():
    p = parse_lisp("(< 5)")
    assert isinstance(p.body, LeftSectionExpr)
    assert p.body.op == "<"


def test_left_section_boolean():
    p = parse_lisp("(&& true)")
    from remora.ast_nodes import VarExpr
    assert isinstance(p.body, LeftSectionExpr)
    assert p.body.op == "&&"


def test_right_section_boolean():
    p = parse_lisp("(a &&)")
    assert isinstance(p.body, RightSectionExpr)
    assert p.body.op == "&&"


# ── Map and Fold ────────────────────────────────────────────────────────────

def test_map_with_left_section():
    p = parse_lisp("(map (+ 2) xs)")
    assert isinstance(p.body, MapExpr)
    assert isinstance(p.body.func, LeftSectionExpr)
    assert p.body.func.op == "+"
    assert isinstance(p.body.arrays[0], VarExpr)
    assert p.body.arrays[0].name == "xs"


def test_map_with_lambda():
    p = parse_lisp("(map (lambda (x) (* x 2)) xs)")
    assert isinstance(p.body, MapExpr)
    assert isinstance(p.body.func, LambdaExpr)
    assert p.body.func.params == ["x"]


def test_map_with_bare_operator():
    p = parse_lisp("(map + xs ys)")
    assert isinstance(p.body, MapExpr)
    assert isinstance(p.body.func, OperatorFuncExpr)
    assert p.body.func.op == "+"
    assert len(p.body.arrays) == 2


def test_fold_with_bare_operator():
    p = parse_lisp("(fold + 0 xs)")
    assert isinstance(p.body, FoldExpr)
    assert isinstance(p.body.func, OperatorFuncExpr)
    assert p.body.func.op == "+"
    assert isinstance(p.body.init, IntLit)
    assert p.body.init.value == 0
    assert isinstance(p.body.array, VarExpr)
    assert p.body.array.name == "xs"


def test_fold_with_lambda():
    p = parse_lisp("(fold (lambda (acc x) (+ acc x)) 0 xs)")
    assert isinstance(p.body, FoldExpr)
    assert isinstance(p.body.func, LambdaExpr)
    assert p.body.func.params == ["acc", "x"]


# ── Iota and Views ──────────────────────────────────────────────────────────

def test_iota():
    p = parse_lisp("(iota 5)")
    assert isinstance(p.body, IotaExpr)
    assert isinstance(p.body.size, IntLit)
    assert p.body.size.value == 5


def test_reverse():
    p = parse_lisp("(reverse xs)")
    assert isinstance(p.body, ReverseExpr)
    assert isinstance(p.body.array, VarExpr)
    assert p.body.array.name == "xs"


def test_transpose():
    p = parse_lisp("(transpose m)")
    assert isinstance(p.body, TransposeExpr)
    assert isinstance(p.body.array, VarExpr)
    assert p.body.array.name == "m"


def test_reshape():
    p = parse_lisp("(reshape xs [2 2])")
    assert isinstance(p.body, ReshapeExpr)
    assert isinstance(p.body.shape, ArrayLit)


def test_ravel():
    p = parse_lisp("(ravel m)")
    assert isinstance(p.body, RavelExpr)


def test_take():
    p = parse_lisp("(take 2 xs)")
    assert isinstance(p.body, TakeExpr)
    assert isinstance(p.body.count, IntLit)
    assert p.body.count.value == 2


def test_drop():
    p = parse_lisp("(drop 2 xs)")
    assert isinstance(p.body, DropExpr)
    assert isinstance(p.body.count, IntLit)
    assert p.body.count.value == 2


# ── Indexing ────────────────────────────────────────────────────────────────

def test_index_single():
    p = parse_lisp("(index xs 0)")
    assert isinstance(p.body, IndexExpr)
    assert isinstance(p.body.array, VarExpr)
    assert p.body.array.name == "xs"
    assert len(p.body.indices) == 1
    assert p.body.indices[0].value == 0


def test_index_multi():
    p = parse_lisp("(index xs 0 1)")
    assert isinstance(p.body, IndexExpr)
    assert len(p.body.indices) == 2
    assert p.body.indices[0].value == 0
    assert p.body.indices[1].value == 1


# ── Shape / Rank ────────────────────────────────────────────────────────────

def test_shape():
    p = parse_lisp("(shape xs)")
    assert isinstance(p.body, ShapeExpr)
    assert isinstance(p.body.array, VarExpr)


def test_rank():
    p = parse_lisp("(rank xs)")
    assert isinstance(p.body, RankExpr)
    assert isinstance(p.body.array, VarExpr)


# ── Function definitions ────────────────────────────────────────────────────

def test_simple_function_definition():
    p = parse_lisp("(define (double [x]) (* x 2))")
    assert len(p.definitions) == 1
    d = p.definitions[0]
    assert isinstance(d, FuncDef)
    assert d.name == "double"
    assert d.params == ["x"]


def test_function_definition_two_params():
    p = parse_lisp("(define (add [x y]) (+ x y))")
    assert p.definitions[0].name == "add"
    assert p.definitions[0].params == ["x", "y"]


def test_function_definition_with_rank_annotation():
    p = parse_lisp("(define (f [x 0]) body)")
    assert p.definitions[0].name == "f"
    assert p.definitions[0].params == ["x"]


def test_function_definition_multi_ranked():
    p = parse_lisp("(define (f [x 0 y 1]) (+ x y))")
    assert p.definitions[0].params == ["x", "y"]


def test_value_definition():
    p = parse_lisp("(define xs [1 2 3])")
    assert len(p.definitions) == 1
    d = p.definitions[0]
    assert isinstance(d, ValDef)
    assert d.name == "xs"
    assert isinstance(d.value, ArrayLit)


def test_multi_definition_program():
    src = "(define (double [x]) (* x 2)) (define triple (lambda (x) (* x 3)))"
    p = parse_lisp(src)
    assert len(p.definitions) == 2
    assert p.definitions[0].name == "double"
    assert p.definitions[1].name == "triple"


def test_program_with_body():
    p = parse_lisp("(define xs [1 2 3]) (+ xs xs)")
    assert len(p.definitions) == 1
    assert isinstance(p.body, AppExpr)


# ── Lambda ──────────────────────────────────────────────────────────────────

def test_lambda_single_param():
    p = parse_lisp("(lambda (x) body)")
    assert isinstance(p.body, LambdaExpr)
    assert p.body.params == ["x"]


def test_lambda_multi_param():
    p = parse_lisp("(lambda (x y) (+ x y))")
    assert isinstance(p.body, LambdaExpr)
    assert p.body.params == ["x", "y"]


def test_unicode_lambda():
    p = parse_lisp("(\u03bb (x) (* x 2))")
    assert isinstance(p.body, LambdaExpr)
    assert p.body.params == ["x"]


# ── Nested expressions ──────────────────────────────────────────────────────

def test_nested_fold_with_let():
    p = parse_lisp("(fold + (:: x 5 (+ x 1)) (iota 10))")
    assert isinstance(p.body, FoldExpr)
    assert isinstance(p.body.init, LetExpr)


def test_nested_let():
    p = parse_lisp("(:: x 1 (:: y 2 (+ x y)))")
    assert isinstance(p.body, LetExpr)
    assert isinstance(p.body.body, LetExpr)


def test_nested_if_in_operator():
    p = parse_lisp("(+ (if (< x 0) 1 2) 3)")
    assert isinstance(p.body, AppExpr)
    assert isinstance(p.body.args[0], IfExpr)


# ── Error cases ─────────────────────────────────────────────────────────────

def test_mismatched_parens():
    with pytest.raises(LarkError):
        parse_lisp("(+ 1 2")


def test_empty_list():
    with pytest.raises((ValueError, LarkError)):
        parse_lisp("()")


def test_define_without_arguments():
    with pytest.raises(LarkError):
        parse_lisp("(define)")
