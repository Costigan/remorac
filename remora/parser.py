"""Parser entry points for Remora Dense Core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer, Tree

from remora.ast_nodes import (
    AppExpr,
    ArrayLit,
    BoolLit,
    ComposeExpr,
    Definition,
    Expr,
    FloatLit,
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
    RightSectionExpr,
    ShapeExpr,
    SourceLoc,
    ValDef,
    VarExpr,
)


_GRAMMAR = (Path(__file__).parent / "grammar.lark").read_text(encoding="utf-8")

_PARSER = Lark(
    _GRAMMAR,
    parser="lalr",
    start=["program", "definition", "expr"],
)


class ASTBuilder(Transformer):
    """Transform Lark parse trees into Remora AST nodes."""

    def __init__(self, filename: str):
        super().__init__(visit_tokens=True)
        self.filename = filename

    def program(self, items: list[Any]) -> Program:
        if len(items) == 1 and isinstance(items[0], list):
            items = items[0]
        definitions: list[Definition] = []
        body: Expr | None = None
        for item in items:
            if isinstance(item, (FuncDef, ValDef)):
                definitions.append(item)
            else:
                body = item
        return Program(definitions, body, self._loc())

    def top_level(self, items: list[Any]) -> list[Any]:
        return items

    def func_def(self, items: list[Any]) -> FuncDef:
        name = str(items[0])
        params = [str(param) for param in items[1]]
        body = items[2]
        return FuncDef(name, params, body, self._loc())

    def val_def(self, items: list[Any]) -> ValDef:
        return ValDef(str(items[0]), items[1], self._loc())

    def params(self, items: list[Any]) -> list[str]:
        return [str(item) for item in items]

    def let_expr(self, items: list[Any]) -> LetExpr:
        return LetExpr(str(items[0]), items[1], items[2], self._loc())

    def if_expr(self, items: list[Any]) -> IfExpr:
        return IfExpr(items[0], items[1], items[2], self._loc())

    def lambda_expr(self, items: list[Any]) -> LambdaExpr:
        params = [str(item) for item in items[1:-2]]
        body = items[-1]
        return LambdaExpr(params, body, self._loc())

    def compose_expr(self, items: list[Any]) -> ComposeExpr:
        return ComposeExpr(items[0], items[2], self._loc())

    def binary_expr(self, items: list[Any]) -> AppExpr:
        left, op, right = items
        return AppExpr(VarExpr(str(op), self._loc()), [left, right], self._loc())

    def application(self, items: list[Any]) -> AppExpr:
        func = items[0]
        args = list(items[1:])
        if isinstance(func, AppExpr):
            return AppExpr(func.func, func.args + args, func.loc)
        return AppExpr(func, args, self._loc())

    def map_expr(self, items: list[Any]) -> MapExpr:
        return MapExpr(items[0], list(items[1:]), self._loc())

    def fold_expr(self, items: list[Any]) -> FoldExpr:
        return FoldExpr(items[0], items[1], items[2], self._loc())

    def iota_expr(self, items: list[Any]) -> IotaExpr:
        return IotaExpr(items[0], self._loc())

    def shape_expr(self, items: list[Any]) -> ShapeExpr:
        return ShapeExpr(items[0], self._loc())

    def rank_expr(self, items: list[Any]) -> RankExpr:
        return RankExpr(items[0], self._loc())

    def operator_func(self, items: list[Any]) -> OperatorFuncExpr:
        return OperatorFuncExpr(str(items[0]), self._loc())

    def left_section(self, items: list[Any]) -> LeftSectionExpr:
        return LeftSectionExpr(str(items[0]), items[1], self._loc())

    def right_section(self, items: list[Any]) -> RightSectionExpr:
        return RightSectionExpr(items[0], str(items[1]), self._loc())

    def paren(self, items: list[Any]) -> Expr:
        return items[0]

    def array_lit(self, items: list[Any]) -> ArrayLit:
        return ArrayLit(list(items), self._loc())

    def atom(self, items: list[Any]) -> Expr:
        expr = items[0]
        for suffix in items[1:]:
            expr = IndexExpr(expr, suffix, self._loc())
        return expr

    def index_suffix(self, items: list[Any]) -> list[Expr]:
        return list(items)

    def float_lit(self, items: list[Any]) -> FloatLit:
        return FloatLit(float(items[0]), self._loc())

    def int_lit(self, items: list[Any]) -> IntLit:
        return IntLit(int(items[0]), self._loc())

    def bool_lit(self, items: list[Any]) -> BoolLit:
        return BoolLit(str(items[0]) == "true", self._loc())

    def var(self, items: list[Any]) -> VarExpr:
        return VarExpr(str(items[0]), self._loc())

    def _loc(self) -> SourceLoc:
        return SourceLoc(self.filename, 0, 0)


def parse_program(source: str, filename: str = "<input>") -> Program:
    tree = _PARSER.parse(source, start="program")
    return _transform(tree, filename)


def parse_definition(source: str, filename: str = "<input>") -> Definition:
    tree = _PARSER.parse(source, start="definition")
    return _transform(tree, filename)


def parse_expr(source: str, filename: str = "<input>") -> Expr:
    tree = _PARSER.parse(source, start="expr")
    return _transform(tree, filename)


def parse_file(path: str | Path) -> Program:
    file_path = Path(path)
    return parse_program(file_path.read_text(encoding="utf-8"), str(file_path))


def parse_repl_input(text: str, filename: str = "<repl>") -> Definition | Expr:
    try:
        return parse_definition(text, filename)
    except Exception as definition_error:
        try:
            return parse_expr(text, filename)
        except Exception:
            raise definition_error


def parse(source: str, filename: str = "<input>") -> Program:
    return parse_program(source, filename)


def _transform(tree: Tree[Token], filename: str) -> Any:
    return ASTBuilder(filename).transform(tree)
