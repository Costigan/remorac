"""Lisp s-expression reader for Remora Dense Core.

Parses Remora's Lisp-style s-expression syntax and produces the same
AST as the existing ML-like parser in remora/parser.py.

Syntax mapping:
    (:: x 5 (+ x 1))        → let x = 5 in x + 1
    (if (< 1 2) 10 20)      → if 1 < 2 then 10 else 20
    (+ 1 2)                 → 1 + 2
    (+ 1 2 3)               → (1 + 2) + 3
    (&& a b)                → a && b
    (+ 2)                   → left section (+ 2)
    (2 +)                   → right section (2 +)
    (define (f [x]) body)   → def f x = body
    (define xs [1 2 3])     → def xs = [1, 2, 3]
    (lambda (x) body)       → \\x -> body
    (λ (x) body)            → \\x -> body
    (map (+ 2) xs)          → map (+ 2) xs
    (fold + 0 xs)           → fold (+) 0 xs
    (reverse xs)            → reverse xs
    (transpose m)           → transpose m
    (reshape xs [2 2])      → reshape xs [2, 2]
    (ravel m)               → ravel m
    (take 2 xs)             → take 2 xs
    (drop 2 xs)             → drop 2 xs
    (index xs 0)            → xs[0]
    (index xs 0 1)          → xs[0, 1]
    (shape xs)              → shape xs
    (rank xs)               → rank xs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer

from remora.ast_nodes import (
    AppExpr,
    AppendExpr,
    ArrayLit,
    BoolLit,
    BoxExpr,
    Definition,
    Expr,
    FloatLit,
    FoldExpr,
    FoldRightExpr,
    FuncDef,
    FilterExpr,
    GradeExpr,
    IfExpr,
    IndexExpr,
    IndicesOfExpr,
    IntLit,
    Iota1Expr,
    IotaExpr,
    LambdaExpr,
    LeftSectionExpr,
    LengthExpr,
    LetExpr,
    MapExpr,
    OperatorFuncExpr,
    Program,
    RankExpr,
    RavelExpr,
    ReduceExpr,
    ReplicateExpr,
    RerankExpr,
    ReshapeExpr,
    ReverseExpr,
    RightSectionExpr,
    RotateExpr,
    ScanExpr,
    SelectExpr,
    ShapeExpr,
    SortExpr,
    SourceLoc,
    SubarrayExpr,
    TakeExpr,
    TraceExpr,
    TransposeExpr,
    DropExpr,
    UnboxExpr,
    ValDef,
    VarExpr,
    WithShapeExpr,
)

_GRAMMAR = r"""
program: sexpr*

?sexpr: "(" list_body ")"
      | rerank_form
      | array_lit
      | BOOL -> bool_lit
      | FLOAT -> float_lit
      | INT -> int_lit
      | NAME -> var
      | MINUS -> var_minus

array_lit: "[" sexpr* "]"

?list_body: define_form
           | let_form
           | if_form
           | select_form
           | lambda_form
           | map_form
           | fold_form
           | reduce_form
           | reduce_zero_form
           | reduce_one_form
           | fold_right_form
           | scan_form
           | scan_one_form
           | escan_form
           | trace_form
           | append_form
           | rotate_form
           | subarray_form
           | indices_of_form
           | with_shape_form
           | box_form
           | unbox_form
           | iota_form
           | iota1_form
           | filter_form
           | replicate_form
           | sort_form
           | grade_form
           | shape_form
           | length_form
           | rank_form
           | transpose_form
           | reverse_form
           | reshape_form
           | ravel_form
           | take_form
           | drop_form
            | index_form
            | index_item_form
            | application

?name_token: NAME | MINUS

define_form: "define" "(" name_token "[" param_spec* "]" ")" sexpr  -> func_def_raw
           | "define" name_token sexpr  -> val_def

param_spec: name_token        -> param_simple
          | name_token INT    -> param_ranked

let_form: "::" name_token sexpr sexpr -> let_expr
if_form: "if" sexpr sexpr sexpr -> if_expr
select_form: "select" sexpr sexpr sexpr -> select_expr
lambda_form: ("lambda" | "λ") "(" name_token* ")" sexpr -> lambda_expr
map_form: "map" sexpr sexpr+ -> map_expr
fold_form: "fold" sexpr sexpr sexpr -> fold_expr
reduce_form: "reduce" sexpr sexpr sexpr -> reduce_expr
reduce_zero_form: "reduce/zero" sexpr sexpr sexpr -> reduce_expr
reduce_one_form: "reduce/1" sexpr sexpr sexpr -> reduce_one_expr
fold_right_form: "fold-right" sexpr sexpr sexpr -> fold_right_expr
scan_form: ("scan" | "iscan" | "iscan/zero" | "scan/zero") sexpr sexpr sexpr -> scan_expr
scan_one_form: ("iscan/1" | "scan/1") sexpr sexpr sexpr -> scan_one_expr
escan_form: ("escan" | "escan/zero") sexpr sexpr sexpr -> escan_expr
trace_form: "trace" sexpr sexpr sexpr -> trace_expr
           | "trace-right" sexpr sexpr sexpr -> trace_right_expr
append_form: "append" sexpr sexpr -> append_expr
rotate_form: "rotate" sexpr sexpr -> rotate_expr
subarray_form: "subarray" sexpr sexpr sexpr -> subarray_expr
indices_of_form: "indices-of" sexpr -> indices_of_expr
with_shape_form: "with-shape" sexpr sexpr -> with_shape_expr
box_form: "box" sexpr -> box_expr
unbox_form: "unbox" sexpr "(" name_token* name_token ")" sexpr -> unbox_expr
iota_form: "iota" sexpr -> iota_expr
iota1_form: "iota1" sexpr -> iota1_expr
filter_form: "filter" sexpr sexpr -> filter_expr
replicate_form: "replicate" sexpr sexpr -> replicate_expr
sort_form: "sort" sexpr sexpr -> sort_expr
grade_form: "grade" sexpr sexpr -> grade_expr
shape_form: "shape" sexpr -> shape_expr
length_form: "length" sexpr -> length_expr
rank_form: "rank" sexpr -> rank_expr
transpose_form: "transpose" sexpr -> transpose_expr
reverse_form: "reverse" sexpr -> reverse_expr
reshape_form: "reshape" sexpr sexpr -> reshape_expr
ravel_form: "ravel" sexpr -> ravel_expr
take_form: "take" sexpr sexpr -> take_expr
drop_form: "drop" sexpr sexpr -> drop_expr
index_form: "index" sexpr sexpr+ -> index_expr
index_item_form: "index-item" sexpr sexpr -> index_expr

application: sexpr sexpr* -> app

BOOL: "#t" | "#f"
FLOAT: /-?([0-9]+\.[0-9]*|[0-9]*\.[0-9]+)/
INT: /-?[0-9]+/
NAME: /[a-zA-Z_+\/*<=>!&|?][a-zA-Z0-9_+\-*\/\<=>!&|?']*/
MINUS: "-"

rerank_form: "~(" INT+ ")" sexpr -> rerank

%ignore /[ \t\f\r\n]+/
%ignore /;[^\n]*/
"""

_INFIX_OPS = frozenset({"+", "-", "*", "/", "<=", "<", "==", "!=", "||", "&&"})

_VIEW_FORMS = frozenset({
    "iota", "shape", "rank", "transpose", "reverse",
    "reshape", "ravel", "take", "drop", "index",
})

_PARSER = Lark(
    _GRAMMAR,
    parser="lalr",
    start="program",
    maybe_placeholders=False,
)


class LispASTBuilder(Transformer):
    """Transform Lark parse trees into Remora AST nodes."""

    def __init__(self, filename: str):
        super().__init__(visit_tokens=True)
        self.filename = filename

    # ── program ──────────────────────────────────────────────────────────

    def program(self, items: list[Any]) -> Program:
        definitions: list[Definition] = []
        body: Expr | None = None
        for item in items:
            if isinstance(item, (FuncDef, ValDef)):
                definitions.append(item)
            elif body is None:
                body = item
        return Program(definitions, body, self._loc_from(items))

    # ── define ───────────────────────────────────────────────────────────

    def func_def_raw(self, items: list[Any]) -> FuncDef:
        name = str(items[0])
        param_specs: list[tuple[str, int | None]] = items[1:-1]
        body: Expr = items[-1]
        param_names = [p[0] for p in param_specs]
        param_ranks = [p[1] for p in param_specs]
        all_ranks_none = all(r is None for r in param_ranks)
        return FuncDef(
            name, param_names, body, self._loc_from(items),
            param_ranks=None if all_ranks_none else param_ranks,
        )

    def val_def(self, items: list[Any]) -> ValDef:
        return ValDef(str(items[0]), items[1], self._loc_from(items))

    def param_simple(self, items: list[Any]) -> tuple[str, None]:
        return (str(items[0]), None)

    def param_ranked(self, items: list[Any]) -> tuple[str, int]:
        return (str(items[0]), int(items[1]))

    # ── let / if / lambda ────────────────────────────────────────────────

    def let_expr(self, items: list[Any]) -> LetExpr:
        return LetExpr(str(items[0]), items[1], items[2], self._loc_from(items))

    def if_expr(self, items: list[Any]) -> IfExpr:
        return IfExpr(items[0], items[1], items[2], self._loc_from(items))

    def select_expr(self, items: list[Any]) -> SelectExpr:
        return SelectExpr(items[0], items[1], items[2], self._loc_from(items))

    def lambda_expr(self, items: list[Any]) -> LambdaExpr:
        # items: [param_names*, body]
        # When there are no params, param list is empty
        params: list[str] = [str(item) for item in items[:-1]]
        body: Expr = items[-1]
        return LambdaExpr(params, body, self._loc_from(items))

    # ── map / fold ───────────────────────────────────────────────────────

    def map_expr(self, items: list[Any]) -> MapExpr:
        func = self._as_callable(items[0])
        arrays = list(items[1:])
        return MapExpr(func, arrays, self._loc_from(items))

    def fold_expr(self, items: list[Any]) -> FoldExpr:
        func = self._as_callable(items[0])
        init = items[1]
        array = items[2]
        return FoldExpr(func, init, array, self._loc_from(items))

    def reduce_expr(self, items: list[Any]) -> ReduceExpr:
        func = self._as_callable(items[0])
        init = items[1]
        array = items[2]
        return ReduceExpr(func, init, array, self._loc_from(items))

    def reduce_one_expr(self, items: list[Any]) -> ReduceExpr:
        func = self._as_callable(items[0])
        init = items[1]
        array = items[2]
        return ReduceExpr(func, init, array, self._loc_from(items), require_nonempty=True)

    def fold_right_expr(self, items: list[Any]) -> FoldRightExpr:
        func = self._as_callable(items[0])
        init = items[1]
        array = items[2]
        return FoldRightExpr(func, init, array, self._loc_from(items))

    def scan_expr(self, items: list[Any]) -> ScanExpr:
        func = self._as_callable(items[0])
        init = items[1]
        array = items[2]
        return ScanExpr(func, init, array, self._loc_from(items), exclusive=False)

    def scan_one_expr(self, items: list[Any]) -> ScanExpr:
        func = self._as_callable(items[0])
        init = items[1]
        array = items[2]
        return ScanExpr(func, init, array, self._loc_from(items), exclusive=False, require_nonempty=True)

    def escan_expr(self, items: list[Any]) -> ScanExpr:
        func = self._as_callable(items[0])
        init = items[1]
        array = items[2]
        return ScanExpr(func, init, array, self._loc_from(items), exclusive=True)

    def trace_expr(self, items: list[Any]) -> TraceExpr:
        func = self._as_callable(items[0])
        init = items[1]
        array = items[2]
        return TraceExpr(func, init, array, self._loc_from(items))

    def trace_right_expr(self, items: list[Any]) -> TraceExpr:
        func = self._as_callable(items[0])
        init = items[1]
        array = items[2]
        return TraceExpr(func, init, array, self._loc_from(items), right=True)

    def append_expr(self, items: list[Any]) -> AppendExpr:
        return AppendExpr(items[0], items[1], self._loc_from(items))

    def rotate_expr(self, items: list[Any]) -> RotateExpr:
        return RotateExpr(items[0], items[1], self._loc_from(items))

    def rerank(self, items: list[Any]) -> RerankExpr:
        ranks = [int(item) for item in items[:-1]]
        func = items[-1]
        return RerankExpr(ranks, func, self._loc_from(items))

    def subarray_expr(self, items: list[Any]) -> SubarrayExpr:
        array = items[0]
        offsets = (
            list(items[1].elements) if isinstance(items[1], ArrayLit) else [items[1]]
        )
        shape = (
            list(items[2].elements) if isinstance(items[2], ArrayLit) else [items[2]]
        )
        return SubarrayExpr(array, offsets, shape, self._loc_from(items))

    def indices_of_expr(self, items: list[Any]) -> IndicesOfExpr:
        return IndicesOfExpr(items[0], self._loc_from(items))

    def with_shape_expr(self, items: list[Any]) -> WithShapeExpr:
        return WithShapeExpr(items[0], items[1], self._loc_from(items))

    def box_expr(self, items: list[Any]) -> BoxExpr:
        return BoxExpr(items[0], self._loc_from(items))

    def unbox_expr(self, items: list[Any]) -> UnboxExpr:
        all_names = [str(item) for item in items[1:-1]]
        if len(all_names) < 1:
            raise ValueError("unbox requires at least one hidden name and a value name")
        hidden_names = all_names[:-1]
        value_name = all_names[-1]
        body = items[-1]
        return UnboxExpr(items[0], hidden_names, value_name, body, self._loc_from(items))

    def iota1_expr(self, items: list[Any]) -> Iota1Expr:
        return Iota1Expr(items[0], self._loc_from(items))

    def filter_expr(self, items: list[Any]) -> FilterExpr:
        return FilterExpr(items[0], items[1], self._loc_from(items))

    def replicate_expr(self, items: list[Any]) -> ReplicateExpr:
        return ReplicateExpr(items[0], items[1], self._loc_from(items))

    def sort_expr(self, items: list[Any]) -> SortExpr:
        return SortExpr(items[0], items[1], self._loc_from(items))

    def grade_expr(self, items: list[Any]) -> GradeExpr:
        return GradeExpr(items[0], items[1], self._loc_from(items))

    # ── iota / shape / rank / views ──────────────────────────────────────

    def iota_expr(self, items: list[Any]) -> IotaExpr:
        return IotaExpr(items[0], self._loc_from(items))

    def shape_expr(self, items: list[Any]) -> ShapeExpr:
        return ShapeExpr(items[0], self._loc_from(items))

    def length_expr(self, items: list[Any]) -> LengthExpr:
        return LengthExpr(items[0], self._loc_from(items))

    def rank_expr(self, items: list[Any]) -> RankExpr:
        return RankExpr(items[0], self._loc_from(items))

    def transpose_expr(self, items: list[Any]) -> TransposeExpr:
        return TransposeExpr(items[0], self._loc_from(items))

    def reverse_expr(self, items: list[Any]) -> ReverseExpr:
        return ReverseExpr(items[0], self._loc_from(items))

    def reshape_expr(self, items: list[Any]) -> ReshapeExpr:
        return ReshapeExpr(items[1], items[0], self._loc_from(items))

    def ravel_expr(self, items: list[Any]) -> RavelExpr:
        return RavelExpr(items[0], self._loc_from(items))

    def take_expr(self, items: list[Any]) -> TakeExpr:
        return TakeExpr(items[0], items[1], self._loc_from(items))

    def drop_expr(self, items: list[Any]) -> DropExpr:
        return DropExpr(items[0], items[1], self._loc_from(items))

    def index_expr(self, items: list[Any]) -> IndexExpr:
        array = items[0]
        indices = list(items[1:])
        return IndexExpr(array, indices, self._loc_from(items))

    # ── application / operators ──────────────────────────────────────────

    def app(self, items: list[Any]) -> Expr:
        if len(items) == 1:
            return items[0]

        first = items[0]
        rest = items[1:]

        if isinstance(first, VarExpr) and first.name in _INFIX_OPS:
            if len(rest) == 1:
                return LeftSectionExpr(first.name, rest[0], self._loc_from(items))
            return self._make_op_chain(first.name, rest)

        if len(rest) == 1 and isinstance(rest[0], VarExpr) and rest[0].name in _INFIX_OPS:
            return RightSectionExpr(first, rest[0].name, self._loc_from(items))

        return AppExpr(first, list(rest), self._loc_from(items))

    # ── atoms ────────────────────────────────────────────────────────────

    def var(self, items: list[Any]) -> VarExpr:
        return VarExpr(str(items[0]), self._loc_from(items))

    def var_minus(self, items: list[Any]) -> VarExpr:
        return VarExpr("-", self._loc_from(items))

    def int_lit(self, items: list[Any]) -> IntLit:
        return IntLit(int(items[0]), self._loc_from(items))

    def float_lit(self, items: list[Any]) -> FloatLit:
        return FloatLit(float(items[0]), self._loc_from(items))

    def bool_lit(self, items: list[Any]) -> BoolLit:
        val = str(items[0]) == "#t"
        return BoolLit(val, self._loc_from(items))

    def array_lit(self, items: list[Any]) -> ArrayLit:
        return ArrayLit(list(items), self._loc_from(items))

    # ── helpers ──────────────────────────────────────────────────────────

    def _as_callable(self, expr: Expr) -> Expr:
        if isinstance(expr, VarExpr) and expr.name in _INFIX_OPS:
            return OperatorFuncExpr(expr.name, expr.loc)
        return expr

    def _make_op_chain(self, op: str, args: list[Expr]) -> Expr:
        result = self._make_binary(op, args[0], args[1])
        for arg in args[2:]:
            result = self._make_binary(op, result, arg)
        return result

    def _make_binary(self, op: str, left: Expr, right: Expr) -> AppExpr:
        op_var = VarExpr(op, self._loc())
        return AppExpr(op_var, [left, right], self._loc_from([left, right]))

    def _loc(self, line: int = 0, col: int = 0) -> SourceLoc:
        return SourceLoc(self.filename, line, col)

    def _loc_from(self, items: list[Any]) -> SourceLoc:
        for item in items:
            if isinstance(item, Token):
                return SourceLoc(self.filename, item.line, item.column)
            loc = getattr(item, "loc", None)
            if isinstance(loc, SourceLoc):
                return loc
            if isinstance(item, list):
                loc = self._loc_from(item)
                if loc.line or loc.col:
                    return loc
        return SourceLoc(self.filename, 0, 0)


# ── public API ────────────────────────────────────────────────────────────

def parse_lisp(source: str, filename: str = "<input>") -> Program:
    """Parse a Remora Lisp-syntax program and return a Program AST."""
    tree = _PARSER.parse(source, start="program")
    return LispASTBuilder(filename).transform(tree)


def parse_lisp_file(path: str | Path) -> Program:
    file_path = Path(path)
    return parse_lisp(file_path.read_text(encoding="utf-8"), str(file_path))
