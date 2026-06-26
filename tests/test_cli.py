from pathlib import Path

import pytest

from remora.cli import main
from remora.runtime import has_openmp_runtime


def write_source(tmp_path, text: str):
    path = tmp_path / "prog.remora"
    path.write_text(text, encoding="utf-8")
    return path


def so_path(tmp_path) -> str:
    """Return a unique .so path under tmp_path to avoid CDLL caching in-process."""
    return str(tmp_path / "out.so")


def test_cli_emit_ast(tmp_path, capsys):
    source = write_source(tmp_path, "map (* 2) (iota 4)")

    assert main(["--emit-ast", str(source)]) == 0
    output = capsys.readouterr().out
    assert "Program(" in output
    assert "MapExpr(" in output


def test_cli_emit_typed_ast(tmp_path, capsys):
    source = write_source(tmp_path, "map (* 2) (iota 4)")

    assert main(["--emit-typed-ast", str(source)]) == 0
    output = capsys.readouterr().out
    assert "TypedProgram(" in output
    assert "ArrayType(" in output


def test_cli_emit_hir(tmp_path, capsys):
    source = write_source(tmp_path, "map (* 2) (iota 4)")

    assert main(["--emit-hir", str(source)]) == 0
    output = capsys.readouterr().out
    assert "HIRProgram(" in output
    assert "HIRMap(" in output


def test_cli_emit_mlir(tmp_path, capsys):
    source = write_source(tmp_path, "map (* 2) (iota 4)")

    assert main(["--emit-mlir", str(source)]) == 0
    output = capsys.readouterr().out
    assert "func.func @main() -> tensor<4xi32>" in output


def test_cli_emit_ptx(tmp_path, capsys):
    source = write_source(tmp_path, "map (* 2) (iota 4)")

    assert main(["--emit-ptx", str(source)]) == 0
    output = capsys.readouterr().out
    assert ".visible .entry" in output


def test_cli_loads_prelude_for_cpu_target(tmp_path, capsys):
    source = write_source(tmp_path, "sum (iota 10)")

    assert main(["--shared", "-o", so_path(tmp_path), str(source)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "45.0"
    assert captured.err == ""


def test_cli_accepts_cpu_threads_option(tmp_path, capsys):
    source = write_source(tmp_path, "1 + 2")

    assert main(["--shared", "-o", so_path(tmp_path), "--cpu-threads", "1", str(source)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "3\n"
    assert captured.err == ""


def test_cli_accepts_cpu_vectorize_option(tmp_path, capsys):
    source = write_source(tmp_path, "map (* 2.0) (iota 4)")

    assert main(["--shared", "-o", so_path(tmp_path), "--cpu-vectorize", str(source)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "[0.0, 2.0, 4.0, 6.0]\n"
    assert captured.err == ""


def test_cli_accepts_no_cpu_vectorize_option(tmp_path, capsys):
    source = write_source(tmp_path, "1 + 2")

    assert main(["--shared", "-o", so_path(tmp_path), "--no-cpu-vectorize", str(source)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "3\n"
    assert captured.err == ""


def test_cli_runs_threaded_cpu_when_openmp_available(tmp_path, capsys):
    if not has_openmp_runtime():
        pytest.skip("OpenMP runtime is unavailable")

    source = write_source(tmp_path, "map (* 2) (iota 4)")

    assert main(["--shared", "-o", so_path(tmp_path), "--cpu-threads", "4", str(source)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "[0, 2, 4, 6]\n"
    assert captured.err == ""


def test_cli_emit_mlir_for_top_level_function_direct_call(tmp_path, capsys):
    source = write_source(tmp_path, "def add1 x = x + 1\nadd1 41")

    assert main(["--emit-mlir", str(source)]) == 0
    output = capsys.readouterr().out
    assert "func.func @main() -> i32" in output


def test_cli_emit_mlir_for_top_level_function_map(tmp_path, capsys):
    source = write_source(tmp_path, "def double x = x * 2\nmap double (iota 4)")

    assert main(["--emit-mlir", str(source)]) == 0
    output = capsys.readouterr().out
    assert "func.func @main() -> tensor<4xi32>" in output
    assert "arith.muli" in output


def test_cli_emit_ptx_for_top_level_function_map(tmp_path, capsys):
    source = write_source(tmp_path, "def double x = x * 2\nmap double (iota 4)")

    assert main(["--emit-ptx", str(source)]) == 0
    output = capsys.readouterr().out
    assert ".visible .entry" in output


def test_cli_cpu_runs_checked_in_examples(tmp_path, capsys):
    compiled_examples = {
        "bool_logic.remora",
        "chained_maps.remora",
        "dot_product.remora",
        "indexing.remora",
        "lift_map.remora",
        "matrix_row_reduce.remora",
        "nested_let.remora",
        "prelude_scale.remora",
        "prelude_sum.remora",
        "rank10_indexing.remora",
        "rank4_map.remora",
        "rank10_map.remora",
        "rank10_rank.remora",
        "rank10_shape.remora",
        "rank3_map.remora",
        "reduce_iota.remora",
        "scalar_arithmetic.remora",
        "shape_rank.remora",
        "three_dimensional_transform.remora",
        "threshold_mask.remora",
        "top_level_value.remora",
    }
    for path in sorted(
        p for p in Path("examples").glob("*.remora") if not p.name.startswith(".")
    ):
        args = ["--shared", "-o", str(tmp_path / f"{path.stem}.so"), str(path)] if path.name in compiled_examples else ["--target", "interp", str(path)]
        assert main(args) == 0, path
        captured = capsys.readouterr()
        assert captured.out.strip(), path
        assert captured.err == "", path


def test_cli_invalid_source_exits_one(tmp_path, capsys):
    source = write_source(tmp_path, "missing")

    assert main(["--shared", "-o", so_path(tmp_path), str(source)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "remorac: unbound variable 'missing'" in captured.err


def test_cli_top_level_function_definition_runs_on_cpu(tmp_path, capsys):
    source = write_source(tmp_path, "def f x = x\nf 1")

    assert main(["--shared", "-o", so_path(tmp_path), str(source)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "1"
    assert captured.err == ""


def test_cli_recursive_function_definition_exits_one(tmp_path, capsys):
    source = write_source(tmp_path, "def fac n = if n <= 1 then 1 else n * fac (n - 1)\nfac 5")

    assert main(["--target", "interp", str(source)]) == 0


def test_cli_missing_file_exits_one(capsys):
    assert main(["does-not-exist.remora"]) == 1
    assert "remorac:" in capsys.readouterr().err
