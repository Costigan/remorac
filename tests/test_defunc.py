import pytest

from remora.defunc import RemoraDefuncError, defunctionalize
from remora.hir import (
    HIRFunction,
    HIRMap,
    HIRPrimCallable,
    HIRProgram,
    HIRVar,
    lower_to_hir,
)
from remora.parser import parse_program
from remora.typechecker import TypeChecker
from remora.types import FLOAT, INT, ArrayType, FuncType, StaticDim


def lower_program_source(source: str):
    typed = TypeChecker().check_program(parse_program(source))
    return lower_to_hir(typed)


def test_inline_lambda_in_map_is_lifted_to_named_function():
    program = lower_program_source("map (\\x -> x * 2.0) (iota 10)")
    lowered = defunctionalize(program)

    assert len(lowered.functions) == 1
    lifted = lowered.functions[0]
    assert lifted.name == "__lambda_0"
    assert [param.name for param in lifted.params] == ["x"]
    assert lifted.return_type == FLOAT
    assert isinstance(lowered.main, HIRMap)
    assert isinstance(lowered.main.func, HIRVar)
    assert lowered.main.func.name == "__lambda_0"
    assert lowered.main.func.type == FuncType((INT,), FLOAT)


def test_milestone_map_lambda_is_static_after_defunc():
    program = lower_program_source("fold (+) 0.0 (map (\\x -> x * x) (iota 10))")
    lowered = defunctionalize(program)

    assert len(lowered.functions) == 1
    assert lowered.functions[0].name == "__lambda_0"


def test_primitive_callable_in_fold_does_not_need_lifting():
    program = lower_program_source("let xs = [1.0, 2.0, 3.0] in fold (+) 0.0 xs")
    lowered = defunctionalize(program)

    assert lowered.functions == []
    assert lowered.main is not None


def test_operator_section_in_map_remains_primitive_callable():
    program = lower_program_source("map (* 2.0) (iota 10)")
    lowered = defunctionalize(program)

    assert lowered.functions == []
    assert isinstance(lowered.main, HIRMap)
    assert isinstance(lowered.main.func, HIRPrimCallable)
    assert lowered.main.func.op == "*"


def test_named_function_reference_in_map_is_already_static():
    func_type = FuncType((FLOAT,), FLOAT)
    function = HIRFunction("__double", [], HIRVar("body", FLOAT), FLOAT)
    program = HIRProgram(
        [function],
        HIRMap(
            frame_shape=(StaticDim(4),),
            cell_shape=(),
            func=HIRVar("__double", func_type),
            arrays=[HIRVar("xs", ArrayType(FLOAT, (StaticDim(4),)))],
            result_type=ArrayType(FLOAT, (StaticDim(4),)),
        ),
        ArrayType(FLOAT, (StaticDim(4),)),
    )

    lowered = defunctionalize(program)

    assert lowered.functions == [function]
    assert isinstance(lowered.main, HIRMap)
    assert isinstance(lowered.main.func, HIRVar)
    assert lowered.main.func.name == "__double"


def test_lambda_capturing_outer_variable_is_deferred():
    program = lower_program_source(
        "let scale = 2.0 in let xs = [1.0, 2.0] in map (\\x -> x * scale) xs"
    )

    with pytest.raises(RemoraDefuncError, match="captures outer variables"):
        defunctionalize(program)
