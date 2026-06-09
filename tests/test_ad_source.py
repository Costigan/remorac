"""Source-to-source reverse-mode AD tests."""

import numpy as np
import pytest

from remora.ad import EvalTape, TapeEntry, grad_via_tape, trace_via_tape
from remora.ad_source import generate_gradient_source
from remora.compiler import (
    compile_gradient_function_source,
    compile_gradient_function_source_to_supported_gpu_artifacts,
    compile_source_gradient_function,
    compile_source_gradient_function_to_supported_gpu_artifacts,
    compile_source,
    compile_function_source,
    compile_function_source_to_supported_gpu_artifacts,
)
from remora.executor import RemoraExecutor
from remora.lisp_reader import parse_lisp
from remora.pipeline import PipelineUnavailable
from remora.runtime import (
    CUDAError,
    cuda_available,
    evaluate_source,
    evaluate_source_compiled,
)
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

    specialized = evaluate_source(
        _SQ_LOSS + " ((grad (iapp sq-loss 3)) [1.0 2.0 3.0])",
        include_prelude=False,
        syntax="lisp",
    )
    np.testing.assert_array_equal(specialized.value, [2.0, 4.0, 6.0])


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

    cpu = compile_gradient_function_source(
        _SQ_LOSS,
        "sq-loss",
        (param_type,),
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
        include_prelude=False,
        syntax="lisp",
    )
    assert gpu.gpu.ptx_text
    assert gpu.gpu.kernels[0].name == "remora_grad_sq_loss"


def test_shape_driven_gradient_generation_validates_optional_example():
    n = 4
    param_type = ArrayType(FLOAT, (StaticDim(n),))
    generated = compile_gradient_function_source(
        _SQ_LOSS,
        "sq-loss",
        (param_type,),
        include_prelude=False,
        syntax="lisp",
        verify=False,
    )
    assert "(Array Float 4)" in generated.gradient_source.source

    with pytest.raises(ValueError, match="does not match parameter shape"):
        compile_gradient_function_source(
            _SQ_LOSS,
            "sq-loss",
            (param_type,),
            np.ones(3),
            include_prelude=False,
            syntax="lisp",
            verify=False,
        )


def test_shape_driven_scalar_gradient_compiles_without_example():
    source = "(define/pi () (poly [value Float] Float) (* value value))"
    compiled = compile_gradient_function_source(
        source,
        "poly",
        (FLOAT,),
        include_prelude=False,
        syntax="lisp",
        verify=False,
    )
    assert compiled.gradient_source.source == (
        "(define/pi () (grad_poly [value Float] Float) (* 2.0 value))"
    )
    assert compiled.compiler.mlir_text


def test_public_gradient_workflow_rejects_conditionals():
    source = (
        "(define/pi () (piecewise [value Float] Float) "
        "(if (> value 0.0) (* value value) (- 0.0 value)))"
    )
    with pytest.raises(NotImplementedError, match="conditionals"):
        compile_gradient_function_source(
            source,
            "piecewise",
            (FLOAT,),
            include_prelude=False,
            syntax="lisp",
            verify=False,
        )


def test_source_level_gradient_request_compiles_automatically():
    request = _SQ_LOSS + " (grad (iapp sq-loss 5))"
    cpu = compile_source_gradient_function(
        request,
        include_prelude=False,
        syntax="lisp",
        verify=False,
    )
    assert cpu.gradient_source.function_name == "grad_sq_loss"
    assert "(Array Float 5)" in cpu.gradient_source.source

    applied_request = _SQ_LOSS + " ((grad (iapp sq-loss 5)) [1.0 2.0 3.0 4.0 5.0])"
    gpu = compile_source_gradient_function_to_supported_gpu_artifacts(
        applied_request,
        include_prelude=False,
        syntax="lisp",
    )
    assert gpu.gpu.kernels[0].name == "remora_grad_sq_loss"

    scalar_request = (
        "(define/pi () (sq [value Float] Float) (* value value)) (grad sq)"
    )
    scalar = compile_source_gradient_function(
        scalar_request,
        include_prelude=False,
        syntax="lisp",
        verify=False,
    )
    assert scalar.gradient_source.source.endswith("(* 2.0 value))")

    with pytest.raises(ValueError, match="specialized"):
        compile_source_gradient_function(
            _SQ_LOSS + " (grad sq-loss)",
            include_prelude=False,
            syntax="lisp",
            verify=False,
        )


def test_ordinary_compile_source_rewrites_applied_gradient():
    source = _SQ_LOSS + " ((grad (iapp sq-loss 5)) [1.0 2.0 3.0 4.0 5.0])"
    artifact = compile_source(
        source,
        verify=False,
        include_prelude=False,
        syntax="lisp",
    )
    assert artifact.return_type == ArrayType(FLOAT, (StaticDim(5),))
    assert "arith.mulf" in artifact.mlir_text
    assert "2.000000e+00" in artifact.mlir_text

    scalar = compile_source(
        "(define/pi () (sq [value Float] Float) (* value value)) "
        "((grad sq) 4.0)",
        verify=False,
        include_prelude=False,
        syntax="lisp",
    )
    assert scalar.return_type == FLOAT
    assert "arith.mulf" in scalar.mlir_text

    executed = evaluate_source_compiled(
        source,
        include_prelude=False,
        syntax="lisp",
    )
    np.testing.assert_array_equal(executed.value, [2.0, 4.0, 6.0, 8.0, 10.0])


def test_ordinary_gradient_rewrite_avoids_function_name_collision():
    source = (
        "(define/pi () (__remora_grad_sq [x Float] Float) x) "
        "(define/pi () (sq [x Float] Float) (* x x)) "
        "((grad sq) 3.0)"
    )
    artifact = compile_source(
        source,
        verify=False,
        include_prelude=False,
        syntax="lisp",
    )
    assert artifact.return_type == FLOAT
    assert "__remora_grad_sq_2" in str(artifact.typed)


def test_ordinary_compile_source_rejects_bare_gradient_value():
    with pytest.raises(ValueError, match="function value"):
        compile_source(
            _SQ_LOSS + " (grad (iapp sq-loss 5))",
            verify=False,
            include_prelude=False,
            syntax="lisp",
        )


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


@pytest.mark.parametrize(
    "source,expected_ops",
    [
        (
            "(define/pi ([n Dim]) (loss [x (Array Float n)] Float) "
            "(fold + 0.0 (* (+ x 1.0) (- x 2.0))))",
            ("arith.addf", "arith.subf"),
        ),
        (
            "(define/pi ([n Dim]) (loss [x (Array Float n)] Float) "
            "(fold + 0.0 (* (* x x) x)))",
            ("arith.mulf", "arith.addf"),
        ),
        (
            "(define/pi ([n Dim]) (loss [x (Array Float n)] Float) "
            "(fold + 0.0 (/ x (+ x 1.0))))",
            ("arith.divf", "arith.addf"),
        ),
    ],
)
def test_fused_nested_gradient_compiles_to_one_gpu_kernel(source, expected_ops):
    param_type = ArrayType(FLOAT, (StaticDim(8),))
    artifact = compile_gradient_function_source_to_supported_gpu_artifacts(
        source,
        "loss",
        (param_type,),
        include_prelude=False,
        syntax="lisp",
    )
    assert len(artifact.gpu.kernels) == 1
    assert artifact.gpu.kernels[0].num_inputs == 1
    for operation in expected_ops:
        assert operation in artifact.gpu.scaffold.text


@pytest.mark.skipif(not cuda_available(), reason="live CUDA driver is not available")
@pytest.mark.parametrize("case", ["polynomial", "cubic", "division"])
def test_fused_nested_gradient_executes_on_gpu(case):
    sources = {
        "polynomial": (
            "(define/pi ([n Dim]) (loss [x (Array Float n)] Float) "
            "(fold + 0.0 (* (+ x 1.0) (- x 2.0))))"
        ),
        "cubic": (
            "(define/pi ([n Dim]) (loss [x (Array Float n)] Float) "
            "(fold + 0.0 (* (* x x) x)))"
        ),
        "division": (
            "(define/pi ([n Dim]) (loss [x (Array Float n)] Float) "
            "(fold + 0.0 (/ x (+ x 1.0))))"
        ),
    }
    x = np.linspace(0.25, 2.0, 8, dtype=np.float32)
    param_type = ArrayType(FLOAT, (StaticDim(len(x)),))
    artifact = compile_gradient_function_source_to_supported_gpu_artifacts(
        sources[case],
        "loss",
        (param_type,),
        include_prelude=False,
        syntax="lisp",
    )
    expected = {
        "polynomial": 2.0 * x - 1.0,
        "cubic": 3.0 * x * x,
        "division": 1.0 / ((x + 1.0) ** 2),
    }[case]
    try:
        executor = RemoraExecutor(artifact.gpu.ptx_text, artifact.gpu.kernels)
    except CUDAError as exc:
        pytest.skip(f"live CUDA device is not available: {exc}")
    with executor:
        result = executor.execute_main([x])
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_ravel_vjp_restores_matrix_shape():
    source = (
        "(define/pi () "
        "(loss [x (Array Float 2 3)] Float) "
        "(fold + 0.0 (* (ravel x) (ravel x))))"
    )
    param_type = ArrayType(FLOAT, (StaticDim(2), StaticDim(3)))
    generated = compile_gradient_function_source(
        source,
        "loss",
        (param_type,),
        include_prelude=False,
        syntax="lisp",
        verify=False,
    )
    assert "(reshape" in generated.gradient_source.source
    assert "[2 3]" in generated.gradient_source.source

    request = source + " ((grad loss) [[1.0 2.0 3.0] [4.0 5.0 6.0]])"
    interpreted = evaluate_source(request, include_prelude=False, syntax="lisp")
    compiled = evaluate_source_compiled(
        request, include_prelude=False, syntax="lisp"
    )
    expected = 2.0 * np.arange(1.0, 7.0).reshape(2, 3)
    np.testing.assert_array_equal(interpreted.value, expected)
    np.testing.assert_array_equal(compiled.value, expected)


def test_reshape_vjp_restores_vector_shape():
    source = (
        "(define/pi () (loss [x (Array Float 6)] Float) "
        "(fold + 0.0 (* (ravel (reshape x [2 3])) "
        "(ravel (reshape x [2 3])))))"
    )
    param_type = ArrayType(FLOAT, (StaticDim(6),))
    generated = compile_gradient_function_source(
        source,
        "loss",
        (param_type,),
        include_prelude=False,
        syntax="lisp",
        verify=False,
    )
    assert "[6]" in generated.gradient_source.source

    request = source + " ((grad loss) [1.0 2.0 3.0 4.0 5.0 6.0])"
    compiled = evaluate_source_compiled(
        request, include_prelude=False, syntax="lisp"
    )
    np.testing.assert_array_equal(compiled.value, 2.0 * np.arange(1.0, 7.0))


def test_transpose_vjp_swaps_cotangent_axes_back():
    source = (
        "(define/pi () (loss [x (Array Float 2 3)] Float) "
        "(fold + 0.0 (* (ravel (transpose x)) (ravel (transpose x)))))"
    )
    request = source + " ((grad loss) [[1.0 2.0 3.0] [4.0 5.0 6.0]])"
    interpreted = evaluate_source(request, include_prelude=False, syntax="lisp")
    compiled = evaluate_source_compiled(
        request, include_prelude=False, syntax="lisp"
    )
    expected = 2.0 * np.arange(1.0, 7.0).reshape(2, 3)
    np.testing.assert_array_equal(interpreted.value, expected)
    np.testing.assert_array_equal(compiled.value, expected)


def test_reverse_vjp_reverses_cotangent_back():
    source = (
        "(define/pi () (loss [x (Array Float 5)] Float) "
        "(fold + 0.0 (* (reverse x) (reverse x))))"
    )
    request = source + " ((grad loss) [1.0 2.0 3.0 4.0 5.0])"
    compiled = evaluate_source_compiled(
        request, include_prelude=False, syntax="lisp"
    )
    np.testing.assert_array_equal(compiled.value, [2.0, 4.0, 6.0, 8.0, 10.0])


@pytest.mark.parametrize(
    "view,expected",
    [
        ("(take 3 x)", [2.0, 4.0, 6.0, 0.0, 0.0]),
        ("(drop 2 x)", [0.0, 0.0, 6.0, 8.0, 10.0]),
    ],
)
def test_take_and_drop_vjps_zero_pad_cotangent(view, expected):
    source = (
        "(define/pi () (loss [x (Array Float 5)] Float) "
        f"(fold + 0.0 (* {view} {view})))"
    )
    generated = compile_gradient_function_source(
        source,
        "loss",
        (ArrayType(FLOAT, (StaticDim(5),)),),
        include_prelude=False,
        syntax="lisp",
        verify=False,
    )
    assert "(append" in generated.gradient_source.source
    request = source + " ((grad loss) [1.0 2.0 3.0 4.0 5.0])"
    interpreted = evaluate_source(request, include_prelude=False, syntax="lisp")
    compiled = evaluate_source_compiled(
        request, include_prelude=False, syntax="lisp"
    )
    np.testing.assert_array_equal(interpreted.value, expected)
    np.testing.assert_array_equal(compiled.value, expected)


# ── Append VJP ──────────────────────────────────────────────────────────


def test_append_gradient_source_generation():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 3)] Float) "
        "  (fold + 0.0 (* (append x x) (append x x))))"
    )
    param_type = ArrayType(FLOAT, (StaticDim(3),))
    generated = compile_gradient_function_source(
        source,
        "loss",
        (param_type,),
        include_prelude=False,
        syntax="lisp",
        verify=False,
    )
    assert "(take" in generated.gradient_source.source
    assert "(drop" in generated.gradient_source.source
    assert "(append" in generated.gradient_source.source


_APPEND_LOSS = (
    "(define/pi () "
    "  (loss [x (Array Float 4)] Float) "
    "  (fold + 0.0 (* (append x x) (append x x))))"
)


def test_compiled_append_gradient_cpu():
    param_type = ArrayType(FLOAT, (StaticDim(4),))
    generated = compile_gradient_function_source(
        _APPEND_LOSS,
        "loss",
        (param_type,),
        include_prelude=False,
        syntax="lisp",
        verify=False,
    )
    assert "(append" in generated.gradient_source.source
    assert "(take" in generated.gradient_source.source
    assert "(drop" in generated.gradient_source.source

    request = _APPEND_LOSS + " ((grad loss) [1.0 2.0 3.0 4.0])"
    interpreted = evaluate_source(request, include_prelude=False, syntax="lisp")
    compiled = evaluate_source_compiled(
        request, include_prelude=False, syntax="lisp"
    )
    expected = 4.0 * np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(interpreted.value, expected)
    np.testing.assert_array_equal(compiled.value, expected)


def test_append_rank2_gradient_compiled():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 4 2)] Float) "
        "  (fold + 0.0 (ravel (* (append x x) (append x x)))))"
    )
    request = (
        source
        + " ((grad loss) [[1.0 2.0] [3.0 4.0] [5.0 6.0] [7.0 8.0]])"
    )
    interpreted = evaluate_source(request, include_prelude=False, syntax="lisp")
    compiled = evaluate_source_compiled(
        request, include_prelude=False, syntax="lisp"
    )
    x = np.arange(1.0, 9.0).reshape(4, 2)
    expected = 4.0 * x
    np.testing.assert_array_equal(interpreted.value, expected)
    np.testing.assert_array_equal(compiled.value, expected)


# ── Subarray VJP ─────────────────────────────────────────────────────────


def test_subarray_gradient_source_generation():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 5)] Float) "
        "  (fold + 0.0 (* (subarray x [2] [3]) (subarray x [2] [3]))))"
    )
    param_type = ArrayType(FLOAT, (StaticDim(5),))
    generated = compile_gradient_function_source(
        source,
        "loss",
        (param_type,),
        include_prelude=False,
        syntax="lisp",
        verify=False,
    )
    assert "(take" in generated.gradient_source.source
    assert "(append" in generated.gradient_source.source


def test_compiled_subarray_gradient_cpu():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 6)] Float) "
        "  (fold + 0.0 (* (subarray x [1] [4]) (subarray x [1] [4]))))"
    )
    request = source + " ((grad loss) [1.0 2.0 3.0 4.0 5.0 6.0])"
    interpreted = evaluate_source(request, include_prelude=False, syntax="lisp")
    compiled = evaluate_source_compiled(
        request, include_prelude=False, syntax="lisp"
    )
    expected = np.array([0.0, 4.0, 6.0, 8.0, 10.0, 0.0])
    np.testing.assert_array_equal(interpreted.value, expected)
    np.testing.assert_array_equal(compiled.value, expected)


def test_subarray_full_gradient_source():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 4)] Float) "
        "  (fold + 0.0 (* (subarray x [0] [4]) (subarray x [0] [4]))))"
    )
    request = source + " ((grad loss) [1.0 2.0 3.0 4.0])"
    interpreted = evaluate_source(request, include_prelude=False, syntax="lisp")
    compiled = evaluate_source_compiled(
        request, include_prelude=False, syntax="lisp"
    )
    expected = 2.0 * np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(interpreted.value, expected)
    np.testing.assert_array_equal(compiled.value, expected)


# ── Rotate VJP ───────────────────────────────────────────────────────────


def test_rotate_gradient_interpreter():
    source = (
        "(define/pi () "
        "  (loss [x (Array Float 4)] Float) "
        "  (fold + 0.0 (* (rotate x 1) (rotate x 1))))"
    )
    request = source + " ((grad loss) [1.0 2.0 3.0 4.0])"
    interpreted = evaluate_source(request, include_prelude=False, syntax="lisp")
    compiled = evaluate_source_compiled(
        request, include_prelude=False, syntax="lisp"
    )
    expected = 2.0 * np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(interpreted.value, expected)
    np.testing.assert_array_equal(compiled.value, expected)
