from pathlib import Path

import numpy as np

from remora.cli import main
from remora.compiler import compile_source_to_mlir, compile_source_to_ptx
from remora.display import format_result
from remora.runtime import evaluate_source, format_value
from remora.types import BOOL


def test_cpu_evaluates_scalar_expression():
    result = evaluate_source("1 + 2.0")

    assert result.type.name == "float"
    assert result.value == 3.0


def test_cpu_evaluates_iota_map_and_fold():
    mapped = evaluate_source("map (* 2.0) (iota 5)")
    folded = evaluate_source("fold (+) 0.0 (iota 10)")

    np.testing.assert_array_equal(mapped.value, np.array([0, 2, 4, 6, 8], dtype=np.float32))
    assert folded.value == 45.0


def test_cpu_evaluates_prelude_functions():
    summed = evaluate_source("sum (iota 10)")
    product = evaluate_source("let xs = [2.0, 3.0, 4.0] in product xs")
    scaled = evaluate_source("scale 2.0 (iota 4)")
    dot = evaluate_source(
        "let xs = [1.0, 2.0, 3.0] in "
        "let ys = [4.0, 5.0, 6.0] in "
        "dot xs ys"
    )

    assert summed.value == 45.0
    assert product.value == 24.0
    np.testing.assert_array_equal(scaled.value, np.array([0, 2, 4, 6], dtype=np.float32))
    assert dot.value == 32.0


def test_cpu_evaluates_binary_map():
    result = evaluate_source(
        "let xs = [1, 2, 3] in let ys = [4, 5, 6] in map (*) xs ys"
    )
    lambda_result = evaluate_source(
        "let xs = [1, 2] in let ys = [3, 4] in map (\\x y -> x + y) xs ys"
    )

    np.testing.assert_array_equal(result.value, np.array([4, 10, 18], dtype=np.int32))
    np.testing.assert_array_equal(lambda_result.value, np.array([4, 6], dtype=np.int32))


def test_cpu_evaluates_static_shape_and_rank():
    shape = evaluate_source("shape [[1, 2], [3, 4]]")
    rank = evaluate_source("rank [[1, 2], [3, 4]]")
    scalar_shape = evaluate_source("shape 42")

    np.testing.assert_array_equal(shape.value, np.array([2, 2], dtype=np.int32))
    assert rank.value == 2
    np.testing.assert_array_equal(scalar_shape.value, np.array([], dtype=np.int32))


def test_cpu_evaluates_indexing():
    scalar = evaluate_source("[[1, 2], [3, 4]][1, 0]")
    row = evaluate_source("[[1, 2], [3, 4]][0]")
    iota_item = evaluate_source("(iota 10)[3]")
    let_item = evaluate_source("let xs = [[1, 2], [3, 4]] in xs[0, 1]")

    assert scalar.value == 3
    np.testing.assert_array_equal(row.value, np.array([1, 2], dtype=np.int32))
    assert iota_item.value == 3
    assert let_item.value == 2


def test_cpu_evaluates_reverse():
    result = evaluate_source("reverse [1, 2, 3]")
    matrix = evaluate_source("reverse [[1, 2], [3, 4]]")

    np.testing.assert_array_equal(result.value, np.array([3, 2, 1], dtype=np.int32))
    np.testing.assert_array_equal(matrix.value, np.array([[3, 4], [1, 2]], dtype=np.int32))


def test_cpu_evaluates_row_reduction_map():
    source = "let xs = [[1.0, 2.0], [3.0, 4.0]] in map (\\row -> fold (+) 0.0 row) xs"

    result = evaluate_source(source)

    np.testing.assert_array_equal(result.value, np.array([3, 7], dtype=np.float32))


def test_cpu_evaluates_top_level_function_call():
    result = evaluate_source("def add1 x = x + 1\nadd1 41")

    assert result.value == 42


def test_cpu_evaluates_top_level_function_in_map():
    result = evaluate_source("def double x = x * 2\nmap double (iota 4)")

    np.testing.assert_array_equal(result.value, np.array([0, 2, 4, 6], dtype=np.int32))


def test_cpu_evaluates_all_checked_in_examples():
    for path in sorted(
        p for p in Path("examples").glob("*.remora") if not p.name.startswith(".")
    ):
        result = evaluate_source(path.read_text(encoding="utf-8"))
        assert result.type is not None, path


def test_compiler_facade_emits_mlir():
    mlir = compile_source_to_mlir("map (* 2) (iota 4)")

    assert "func.func @main() -> tensor<4xi32>" in mlir


def test_compiler_facade_emits_mlir_for_prelude_sum():
    mlir = compile_source_to_mlir("sum (iota 10)")

    assert "func.func @main() -> f32" in mlir
    assert "linalg.generic" in mlir
    assert "arith.addf" in mlir


def test_compiler_facade_emits_ptx():
    artifact = compile_source_to_ptx("map (* 2) (iota 4)")

    assert ".visible .entry" in artifact.ptx_text
    assert artifact.kernels


def test_cli_cpu_target_prints_result(tmp_path, capsys):
    source_file = tmp_path / "prog.remora"
    source_file.write_text("fold (+) 0.0 (iota 10)", encoding="utf-8")

    exit_code = main([str(source_file)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "45.0"


def test_cli_cpu_target_prints_shape(tmp_path, capsys):
    source_file = tmp_path / "prog.remora"
    source_file.write_text("shape [[1, 2], [3, 4]]", encoding="utf-8")

    exit_code = main([str(source_file)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "[2, 2]"


def test_format_value_for_arrays():
    assert format_value(np.array([1, 2], dtype=np.int32)) == "[1, 2]"


def test_cli_cpu_target_prints_remora_bool(tmp_path, capsys):
    source_file = tmp_path / "prog.remora"
    source_file.write_text("(1 < 2) && (2 < 3)", encoding="utf-8")

    exit_code = main([str(source_file)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "true"


def test_runtime_result_can_use_shared_display():
    result = evaluate_source("true")

    assert format_result(result.value, BOOL) == "true"
