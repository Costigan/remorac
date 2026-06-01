import importlib.util

import numpy as np
import pytest

from remora.runtime import CPUExecutor, CPUFunctionExecutor, evaluate_source, evaluate_source_compiled
from remora.runtime import EvaluationError
from remora.types import FLOAT, INT, ArrayType, StaticDim


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("iree") is None,
    reason="IREE compiler MLIR bindings are not installed",
)


def assert_compiled_matches_interpreter(source: str) -> None:
    compiled = evaluate_source_compiled(source)
    interpreted = evaluate_source(source)

    assert compiled.type == interpreted.type
    if isinstance(interpreted.value, np.ndarray):
        np.testing.assert_array_equal(compiled.value, interpreted.value)
    elif isinstance(interpreted.value, float):
        assert compiled.value == pytest.approx(interpreted.value)
    else:
        assert compiled.value == interpreted.value


def test_compiled_cpu_executes_scalar_expression():
    assert_compiled_matches_interpreter("1 + 2.0")


def test_compiled_cpu_executes_vector_map():
    result = evaluate_source_compiled("map (* 2.0) (iota 5)")

    np.testing.assert_array_equal(result.value, np.array([0, 2, 4, 6, 8], dtype=np.float32))


def test_compiled_cpu_executes_matrix_and_rank3_maps():
    matrix = evaluate_source_compiled("let xs = [[1.0, 2.0], [3.0, 4.0]] in map (* 2.0) xs")
    tensor3 = evaluate_source_compiled("let xs = [[[1], [2]], [[3], [4]]] in map (\\x -> x + 1) xs")

    np.testing.assert_array_equal(
        matrix.value,
        np.array([[2, 4], [6, 8]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        tensor3.value,
        np.array([[[2], [3]], [[4], [5]]], dtype=np.int32),
    )


def test_compiled_cpu_executes_vector_sum_and_dot():
    summed = evaluate_source_compiled("fold (+) 0.0 (iota 10)")
    dot = evaluate_source_compiled(
        "let xs = [1.0, 2.0, 3.0] in "
        "let ys = [4.0, 5.0, 6.0] in "
        "dot xs ys"
    )

    assert summed.value == pytest.approx(45.0)
    assert dot.value == pytest.approx(32.0)


def test_compiled_cpu_executes_static_shape():
    result = evaluate_source_compiled("shape [[1, 2], [3, 4]]")

    np.testing.assert_array_equal(result.value, np.array([2, 2], dtype=np.int32))


def test_compiled_cpu_executes_bool_rank_and_empty_shape():
    boolean = evaluate_source_compiled("(1 < 2) && (2 < 3)")
    rank = evaluate_source_compiled("rank [[1, 2], [3, 4]]")
    scalar_shape = evaluate_source_compiled("shape 42")

    assert boolean.value is True
    assert rank.value == 2
    np.testing.assert_array_equal(scalar_shape.value, np.array([], dtype=np.int32))


def test_cpu_executor_compile_source_keeps_artifact_until_closed():
    artifact = CPUExecutor.compile_source("map (* 2) (iota 4)")
    try:
        result = CPUExecutor(artifact).execute_main()
    finally:
        artifact.close()

    np.testing.assert_array_equal(result.value, np.array([0, 2, 4, 6], dtype=np.int32))


def test_cpu_executor_writes_into_explicit_output_descriptor():
    artifact = CPUExecutor.compile_source("map (* 2.0) (iota 5)")
    output = np.empty((5,), dtype=np.float32)
    try:
        CPUExecutor(artifact).execute_main_into(output)
    finally:
        artifact.close()

    np.testing.assert_array_equal(output, np.array([0, 2, 4, 6, 8], dtype=np.float32))


def test_cpu_executor_writes_into_strided_output_descriptor_view():
    artifact = CPUExecutor.compile_source("map (* 2.0) (iota 5)")
    backing = np.full((5, 2), -1.0, dtype=np.float32)
    output = backing[:, 0]
    try:
        CPUExecutor(artifact).execute_main_into(output)
    finally:
        artifact.close()

    np.testing.assert_array_equal(output, np.array([0, 2, 4, 6, 8], dtype=np.float32))
    np.testing.assert_array_equal(backing[:, 1], np.full((5,), -1.0, dtype=np.float32))


def test_cpu_executor_writes_scalar_matrix_and_rank3_output_descriptors():
    scalar_artifact = CPUExecutor.compile_source("1 + 2.0")
    matrix_artifact = CPUExecutor.compile_source(
        "let xs = [[1.0, 2.0], [3.0, 4.0]] in map (* 2.0) xs"
    )
    rank3_artifact = CPUExecutor.compile_source(
        "let xs = [[[1], [2]], [[3], [4]]] in map (\\x -> x + 1) xs"
    )
    scalar_output = np.empty((), dtype=np.float32)
    matrix_output = np.empty((2, 2), dtype=np.float32)
    rank3_output = np.empty((2, 2, 1), dtype=np.int32)

    try:
        CPUExecutor(scalar_artifact).execute_main_into(scalar_output)
        CPUExecutor(matrix_artifact).execute_main_into(matrix_output)
        CPUExecutor(rank3_artifact).execute_main_into(rank3_output)
    finally:
        scalar_artifact.close()
        matrix_artifact.close()
        rank3_artifact.close()

    assert scalar_output.item() == pytest.approx(3.0)
    np.testing.assert_array_equal(
        matrix_output,
        np.array([[2, 4], [6, 8]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        rank3_output,
        np.array([[[2], [3]], [[4], [5]]], dtype=np.int32),
    )


def test_cpu_executor_rejects_output_descriptor_shape_and_dtype_mismatches():
    artifact = CPUExecutor.compile_source("map (* 2.0) (iota 5)")
    executor = CPUExecutor(artifact)
    try:
        with pytest.raises(EvaluationError, match="shape mismatch"):
            executor.execute_main_into(np.empty((4,), dtype=np.float32))
        with pytest.raises(EvaluationError, match="dtype mismatch"):
            executor.execute_main_into(np.empty((5,), dtype=np.int32))
    finally:
        artifact.close()


def test_cpu_function_executor_runs_descriptor_input_vector_map():
    artifact = CPUFunctionExecutor.compile_source(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(5),)),),
    )
    try:
        result = CPUFunctionExecutor(artifact).execute(np.arange(5, dtype=np.float32))
    finally:
        artifact.close()

    np.testing.assert_array_equal(result.value, np.array([0, 2, 4, 6, 8], dtype=np.float32))


def test_cpu_function_executor_runs_rank0_descriptor_input():
    artifact = CPUFunctionExecutor.compile_source(
        "def bump x = x + 1.0",
        "bump",
        (FLOAT,),
    )
    try:
        result = CPUFunctionExecutor(artifact).execute(np.asarray(2.0, dtype=np.float32))
    finally:
        artifact.close()

    assert result.value == pytest.approx(3.0)


def test_cpu_function_executor_honors_strided_descriptor_inputs_and_outputs():
    artifact = CPUFunctionExecutor.compile_source(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(5),)),),
    )
    backing_input = np.arange(10, dtype=np.float32).reshape(5, 2)
    input_view = backing_input[:, 0]
    backing_output = np.full((5, 2), -1.0, dtype=np.float32)
    output_view = backing_output[:, 1]
    try:
        CPUFunctionExecutor(artifact).execute_into(output_view, input_view)
    finally:
        artifact.close()

    np.testing.assert_array_equal(output_view, np.array([0, 4, 8, 12, 16], dtype=np.float32))
    np.testing.assert_array_equal(backing_output[:, 0], np.full((5,), -1.0, dtype=np.float32))


def test_cpu_function_executor_runs_rank2_and_rank3_descriptor_input_maps():
    matrix_artifact = CPUFunctionExecutor.compile_source(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(2), StaticDim(2))),),
    )
    rank3_artifact = CPUFunctionExecutor.compile_source(
        "def inc xs = map (\\x -> x + 1) xs",
        "inc",
        (ArrayType(INT, (StaticDim(2), StaticDim(2), StaticDim(1))),),
    )
    try:
        matrix = CPUFunctionExecutor(matrix_artifact).execute(
            np.array([[1, 2], [3, 4]], dtype=np.float32)
        )
        tensor3 = CPUFunctionExecutor(rank3_artifact).execute(
            np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
        )
    finally:
        matrix_artifact.close()
        rank3_artifact.close()

    np.testing.assert_array_equal(
        matrix.value,
        np.array([[2, 4], [6, 8]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        tensor3.value,
        np.array([[[2], [3]], [[4], [5]]], dtype=np.int32),
    )


def test_cpu_function_executor_runs_binary_descriptor_input_map():
    artifact = CPUFunctionExecutor.compile_source(
        "def add xs ys = map (+) xs ys",
        "add",
        (
            ArrayType(FLOAT, (StaticDim(4),)),
            ArrayType(FLOAT, (StaticDim(4),)),
        ),
    )
    try:
        result = CPUFunctionExecutor(artifact).execute(
            np.array([1, 2, 3, 4], dtype=np.float32),
            np.array([10, 20, 30, 40], dtype=np.float32),
        )
    finally:
        artifact.close()

    np.testing.assert_array_equal(
        result.value,
        np.array([11, 22, 33, 44], dtype=np.float32),
    )


def test_cpu_function_executor_runs_fold_and_dot_over_descriptor_inputs():
    sum_artifact = CPUFunctionExecutor.compile_source(
        "def sumit xs = fold (+) 0.0 xs",
        "sumit",
        (ArrayType(FLOAT, (StaticDim(4),)),),
    )
    dot_artifact = CPUFunctionExecutor.compile_source(
        "def dotit xs ys = dot xs ys",
        "dotit",
        (
            ArrayType(FLOAT, (StaticDim(3),)),
            ArrayType(FLOAT, (StaticDim(3),)),
        ),
    )
    try:
        summed = CPUFunctionExecutor(sum_artifact).execute(
            np.array([1, 2, 3, 4], dtype=np.float32)
        )
        dotted = CPUFunctionExecutor(dot_artifact).execute(
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32),
        )
    finally:
        sum_artifact.close()
        dot_artifact.close()

    assert summed.value == pytest.approx(10.0)
    assert dotted.value == pytest.approx(32.0)


def test_cpu_function_executor_rejects_input_and_output_mismatches():
    artifact = CPUFunctionExecutor.compile_source(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(5),)),),
    )
    executor = CPUFunctionExecutor(artifact)
    try:
        with pytest.raises(EvaluationError, match="expects 1 inputs"):
            executor.execute()
        with pytest.raises(EvaluationError, match="input 0 shape mismatch"):
            executor.execute(np.empty((4,), dtype=np.float32))
        with pytest.raises(EvaluationError, match="input 0 dtype mismatch"):
            executor.execute(np.empty((5,), dtype=np.int32))
        with pytest.raises(EvaluationError, match="output shape mismatch"):
            executor.execute_into(
                np.empty((4,), dtype=np.float32),
                np.empty((5,), dtype=np.float32),
            )
        with pytest.raises(EvaluationError, match="output dtype mismatch"):
            executor.execute_into(
                np.empty((5,), dtype=np.int32),
                np.empty((5,), dtype=np.float32),
            )
    finally:
        artifact.close()
