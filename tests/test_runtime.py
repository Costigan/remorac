from pathlib import Path

import numpy as np

from remora.cli import main
from remora.compiler import compile_source_to_mlir, compile_source_to_ptx
from remora.runtime import evaluate_source, format_value


def test_cpu_evaluates_scalar_expression():
    result = evaluate_source("1 + 2.0")

    assert result.type.name == "float"
    assert result.value == 3.0


def test_cpu_evaluates_iota_map_and_fold():
    mapped = evaluate_source("map (* 2.0) (iota 5)")
    folded = evaluate_source("fold (+) 0.0 (iota 10)")

    np.testing.assert_array_equal(mapped.value, np.array([0, 2, 4, 6, 8], dtype=np.float32))
    assert folded.value == 45.0


def test_cpu_evaluates_row_reduction_map():
    source = "let xs = [[1.0, 2.0], [3.0, 4.0]] in map (\\row -> fold (+) 0.0 row) xs"

    result = evaluate_source(source)

    np.testing.assert_array_equal(result.value, np.array([3, 7], dtype=np.float32))


def test_cpu_evaluates_all_checked_in_examples():
    for path in sorted(Path("examples").glob("*.remora")):
        result = evaluate_source(path.read_text(encoding="utf-8"))
        assert result.type is not None, path


def test_compiler_facade_emits_mlir():
    mlir = compile_source_to_mlir("map (* 2) (iota 4)")

    assert "func.func @main() -> tensor<4xi32>" in mlir


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


def test_format_value_for_arrays():
    assert format_value(np.array([1, 2], dtype=np.int32)) == "[1, 2]"
