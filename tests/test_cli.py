from pathlib import Path

from remora.cli import main


def write_source(tmp_path, text: str):
    path = tmp_path / "prog.remora"
    path.write_text(text, encoding="utf-8")
    return path


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


def test_cli_target_mlir_alias(tmp_path, capsys):
    source = write_source(tmp_path, "map (* 2) (iota 4)")

    assert main(["--target", "mlir", str(source)]) == 0
    output = capsys.readouterr().out
    assert "func.func @main() -> tensor<4xi32>" in output


def test_cli_target_ptx_alias(tmp_path, capsys):
    source = write_source(tmp_path, "map (* 2) (iota 4)")

    assert main(["--target", "ptx", str(source)]) == 0
    output = capsys.readouterr().out
    assert ".visible .entry" in output


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


def test_cli_cpu_runs_checked_in_examples(capsys):
    for path in sorted(Path("examples").glob("*.remora")):
        assert main([str(path)]) == 0, path
        captured = capsys.readouterr()
        assert captured.out.strip(), path
        assert captured.err == "", path


def test_cli_invalid_source_exits_one(tmp_path, capsys):
    source = write_source(tmp_path, "missing")

    assert main([str(source)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "remorac: unbound variable 'missing'" in captured.err


def test_cli_top_level_function_definition_runs_on_cpu(tmp_path, capsys):
    source = write_source(tmp_path, "def f x = x\nf 1")

    assert main([str(source)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "1"
    assert captured.err == ""


def test_cli_recursive_function_definition_exits_one(tmp_path, capsys):
    source = write_source(tmp_path, "def f x = f x\nf 1")

    assert main([str(source)]) == 1
    assert "recursive function definitions are deferred" in capsys.readouterr().err


def test_cli_missing_file_exits_one(capsys):
    assert main(["does-not-exist.remora"]) == 1
    assert "remorac:" in capsys.readouterr().err
