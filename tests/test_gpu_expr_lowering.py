from remora._gpu_expr_lowering import _GpuTailStateMachine, gpu_expr_from_hir
from remora.hir import HIRCall, HIRFunction, HIRIf, HIRLit, HIRParam, HIRPrimOp, HIRVar
from remora.types import BOOL, FLOAT


def _is_zero(name: str):
    return HIRPrimOp("==f", [HIRVar(name, FLOAT), HIRLit(0.0, FLOAT)], BOOL)


def _dec(name: str):
    return HIRPrimOp("-", [HIRVar(name, FLOAT), HIRLit(1.0, FLOAT)], FLOAT)


def test_mutual_scalar_tail_recursion_lowers_to_gpu_state_machine():
    even = HIRFunction(
        name="even",
        params=[HIRParam("n", FLOAT)],
        body=HIRIf(
            _is_zero("n"),
            HIRLit(1.0, FLOAT),
            HIRCall("odd", [_dec("n")], FLOAT),
            FLOAT,
        ),
        return_type=FLOAT,
    )
    odd = HIRFunction(
        name="odd",
        params=[HIRParam("n", FLOAT)],
        body=HIRIf(
            _is_zero("n"),
            HIRLit(0.0, FLOAT),
            HIRCall("even", [_dec("n")], FLOAT),
            FLOAT,
        ),
        return_type=FLOAT,
    )

    expr = gpu_expr_from_hir(
        HIRCall("even", [HIRVar("x", FLOAT)], FLOAT),
        input_map={},
        scalar_env={"x": 0},
        coords=[],
        functions={"even": even, "odd": odd},
    )

    assert isinstance(expr, _GpuTailStateMachine)
    assert expr.init_target == "even"
    assert {step.name for step in expr.steps} == {"even", "odd"}
