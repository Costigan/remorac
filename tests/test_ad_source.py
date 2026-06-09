"""Source-to-source reverse-mode AD tests."""

import numpy as np
import pytest

from remora.ad import EvalTape, TapeEntry, grad_via_tape, trace_via_tape
from remora.ad_source import generate_gradient_source
from remora.compiler import (
    compile_gradient_function_source,
    compile_gradient_function_source_to_supported_gpu_artifacts,
    compile_function_source,
    compile_function_source_to_supported_gpu_artifacts,
)
from remora.executor import RemoraExecutor
from remora.lisp_reader import parse_lisp
from remora.pipeline import PipelineUnavailable
from remora.runtime import CUDAError, cuda_available, evaluate_source
from remora.typechecker import TypeChecker
from remora.types import ArrayType, FLOAT, FuncType, StaticDim


_SQ_LOSS = (
    "(define/pi ([n Dim]) "
    "  (sq-loss [x (Array Float n)] Float) "
    "  (fold + 0.0 (* x x)))"
)


def _sq_loss_body(n: int):
    checker = TypeChecker()
    checker.check_program(parse_lisp(_SQ_LOSS))
    dim = StaticDim(n)
    specialized = checker._typed_top_level_function(
        checker._functions["sq-loss"],
        FuncType((ArrayType(FLOAT, (dim,)),), FLOAT),
        checker._build_prelude_env(),
        index_args=(dim,),
    )
    return specialized.body


def test_generated_square_gradient_is_reusable_source():
    body = _sq_loss_body(5)
    traced_at = np.arange(1.0, 6.0)
    tape, _ = trace_via_tape(body, "x", traced_at)

    source = generate_gradient_source(tape, "x", traced_at.shape)

    assert "(map (* 2.0) x)" in source
    result = evaluate_source(
        source + " (grad-f [5.0 4.0 3.0 2.0 1.0])",
        include_prelude=False,
        syntax="lisp",
    )
    np.testing.assert_array_equal(result.value, [10.0, 8.0, 6.0, 4.0, 2.0])


def test_source_level_grad_executes_for_concrete_function():
    result = evaluate_source(
        "(define/pi () (sq [x Float] Float) (* x x)) ((grad sq) 3.0)",
        include_prelude=False,
        syntax="lisp",
    )
    assert result.value == pytest.approx(6.0)

    renamed = evaluate_source(
        "(define/pi () (sq [value Float] Float) (* value value)) "
        "((grad sq) 4.0)",
        include_prelude=False,
        syntax="lisp",
    )
    assert renamed.value == pytest.approx(8.0)


def test_generated_source_handles_division_negation_and_fold_fill():
    polynomial = EvalTape()
    px = polynomial.push_input(np.asarray(3.0))
    one = polynomial.push_const(1.0)
    plus = polynomial.push(TapeEntry("add", (px, one), ()), np.asarray(4.0))
    two_const = polynomial.push_const(2.0)
    minus = polynomial.push(TapeEntry("sub", (px, two_const), ()), np.asarray(1.0))
    polynomial.push(
        TapeEntry("mul", (plus, minus), (np.asarray(1.0), np.asarray(4.0))),
        np.asarray(4.0),
    )
    polynomial_source = generate_gradient_source(polynomial, "x", ())
    polynomial_result = evaluate_source(
        polynomial_source + " (grad-f 7.0)",
        include_prelude=False,
        syntax="lisp",
    )
    assert polynomial_result.value == pytest.approx(13.0)

    tape = EvalTape()
    x = tape.push_input(np.asarray(8.0))
    two = tape.push_const(2.0)
    quotient = tape.push(
        TapeEntry("div", (x, two), (np.asarray(2.0), np.asarray(8.0))),
        np.asarray(4.0),
    )
    tape.push(TapeEntry("neg", (quotient,), ()), np.asarray(-4.0))
    source = generate_gradient_source(tape, "x", ())
    result = evaluate_source(
        source + " (grad-f 13.0)", include_prelude=False, syntax="lisp"
    )
    assert result.value == pytest.approx(-0.5)

    sum_tape = EvalTape()
    values = np.asarray([2.0, 4.0, 6.0])
    array = sum_tape.push_input(values)
    sum_tape.push(TapeEntry("fold", (array,), (values,)), np.asarray(12.0))
    sum_source = generate_gradient_source(sum_tape, "x", values.shape)
    sum_result = evaluate_source(
        sum_source + " (grad-f [9.0 8.0 7.0])",
        include_prelude=False,
        syntax="lisp",
    )
    np.testing.assert_array_equal(sum_result.value, np.ones(3))


def test_generated_gradient_compiles_for_cpu_and_gpu_artifacts():
    n = 8
    dim = StaticDim(n)
    param_type = ArrayType(FLOAT, (dim,))
    body = _sq_loss_body(n)
    x = np.linspace(-2.0, 2.0, n)
    tape, _ = trace_via_tape(body, "x", x)
    source = generate_gradient_source(
        tape, "x", x.shape, function_name="grad_f"
    )

    cpu_artifact = compile_function_source(
        source,
        "grad_f",
        (param_type,),
        verify=False,
        include_prelude=False,
        syntax="lisp",
    )
    assert cpu_artifact.mlir_text

    try:
        gpu_artifact = compile_function_source_to_supported_gpu_artifacts(
            source,
            "grad_f",
            (param_type,),
            include_prelude=False,
            kernel_name="remora_grad_f",
            syntax="lisp",
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"GPU pipeline not available: {exc}")

    assert gpu_artifact.ptx_text
    assert gpu_artifact.kernels
    np.testing.assert_allclose(grad_via_tape(body, "x", x), 2.0 * x)


def test_named_function_gradient_compiler_workflow():
    n = 6
    param_type = ArrayType(FLOAT, (StaticDim(n),))
    x = np.linspace(-1.0, 1.0, n)

    cpu = compile_gradient_function_source(
        _SQ_LOSS,
        "sq-loss",
        (param_type,),
        x,
        include_prelude=False,
        syntax="lisp",
        verify=False,
    )
    assert cpu.gradient_source.function_name == "grad_sq_loss"
    assert "(map (* 2.0) x)" in cpu.gradient_source.source
    assert cpu.compiler.mlir_text

    gpu = compile_gradient_function_source_to_supported_gpu_artifacts(
        _SQ_LOSS,
        "sq-loss",
        (param_type,),
        x,
        include_prelude=False,
        syntax="lisp",
    )
    assert gpu.gpu.ptx_text
    assert gpu.gpu.kernels[0].name == "remora_grad_sq_loss"


def test_gradient_source_rejects_branch_specialized_tape():
    tape = EvalTape(has_data_dependent_control_flow=True)
    tape.push_input(np.asarray(2.0))
    with pytest.raises(NotImplementedError, match="conditionals"):
        generate_gradient_source(tape, "x", ())


@pytest.mark.skipif(not cuda_available(), reason="live CUDA driver is not available")
def test_generated_gradient_executes_on_gpu():
    n = 8
    dim = StaticDim(n)
    param_type = ArrayType(FLOAT, (dim,))
    body = _sq_loss_body(n)
    x = np.linspace(-2.0, 2.0, n, dtype=np.float32)
    tape, _ = trace_via_tape(body, "x", x)
    source = generate_gradient_source(
        tape, "x", x.shape, function_name="grad_f"
    )
    artifact = compile_function_source_to_supported_gpu_artifacts(
        source,
        "grad_f",
        (param_type,),
        include_prelude=False,
        kernel_name="remora_grad_f",
        syntax="lisp",
    )

    try:
        executor = RemoraExecutor(artifact.ptx_text, artifact.kernels)
    except CUDAError as exc:
        pytest.skip(f"live CUDA device is not available: {exc}")
    with executor:
        gpu_gradient = executor.execute_main([x])

    np.testing.assert_allclose(gpu_gradient, grad_via_tape(body, "x", x), rtol=1e-5)
