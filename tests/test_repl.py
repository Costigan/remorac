import pytest

from remora.repl import ReplSession, main


def test_repl_evaluates_expression():
    session = ReplSession()

    assert session.eval_input("1 + 2.0") == "3.0"


def test_repl_persists_value_definition():
    session = ReplSession()

    assert session.eval_input("def xs = iota 4") == "Defined: xs : int[4]"
    assert session.eval_input("map (* 2.0) xs") == "[0.0, 2.0, 4.0, 6.0]"


def test_repl_definition_can_reference_previous_definition():
    session = ReplSession()

    assert session.eval_input("def x = 41") == "Defined: x : int"
    assert session.eval_input("def y = x + 1") == "Defined: y : int"
    assert session.eval_input("y") == "42"


def test_repl_persists_function_definition():
    session = ReplSession()

    assert session.eval_input("def add1 x = x + 1") == "Defined: add1 : <function>"
    assert session.eval_input("add1 41") == "42"


def test_repl_function_definition_can_be_used_in_map():
    session = ReplSession()

    assert session.eval_input("def double x = x * 2") == "Defined: double : <function>"
    assert session.eval_input("map double (iota 4)") == "[0, 2, 4, 6]"


def test_repl_type_command_uses_session_definitions():
    session = ReplSession()
    session.eval_input("def xs = iota 4")

    assert session.eval_input(":type map (* 2.0) xs") == "map (* 2.0) xs : float[4]"


def test_repl_evaluates_shape_and_rank():
    session = ReplSession()

    assert (
        session.eval_input(":type shape [[1, 2], [3, 4]]")
        == "shape [[1, 2], [3, 4]] : int[2]"
    )
    assert session.eval_input("shape [[1, 2], [3, 4]]") == "[2, 2]"
    assert session.eval_input("rank [[1, 2], [3, 4]]") == "2"


def test_repl_evaluates_indexing():
    session = ReplSession()

    assert session.eval_input("[[1, 2], [3, 4]][1, 0]") == "3"
    assert session.eval_input("[[1, 2], [3, 4]][0]") == "[1, 2]"


def test_repl_accepts_multiline_expression():
    session = ReplSession()

    assert session.eval_input("let x = 1 in\nx + 1") == "2"


def test_repl_collects_multiline_input(monkeypatch):
    session = ReplSession()
    monkeypatch.setattr("builtins.input", lambda _prompt: "x + 1)")

    assert session._collect_full_input("(let x = 1 in") == "(let x = 1 in\nx + 1)"


def test_repl_mlir_command_uses_session_definitions():
    session = ReplSession()
    session.eval_input("def xs = iota 4")

    mlir = session.eval_input(":mlir map (* 2) xs")

    assert mlir is not None
    assert "func.func @main() -> tensor<4xi32>" in mlir


def test_repl_reset_clears_definitions():
    session = ReplSession()
    session.eval_input("def x = 1")

    assert session.eval_input(":reset") == "State reset."
    assert session.eval_input("x").startswith("Error: unbound variable")


def test_repl_load_file(tmp_path):
    source = tmp_path / "prog.remora"
    source.write_text("def xs = iota 4\nmap (* 2.0) xs", encoding="utf-8")
    session = ReplSession()

    assert session.eval_input(f":load {source}") == "Defined: xs : int[4]\n[0.0, 2.0, 4.0, 6.0]"
    assert session.eval_input("fold (+) 0.0 xs") == "6.0"


def test_repl_reports_deferred_recursive_function_definition():
    session = ReplSession()

    assert session.eval_input("def f x = f x") == "Defined: f : <function>"
    assert "recursive function definitions are deferred" in session.eval_input("f 1")


def test_repl_error_recovery():
    session = ReplSession()

    assert session.eval_input("missing").startswith("Error: unbound variable")
    assert session.eval_input("1 + 1") == "2"


def test_repl_target_command():
    session = ReplSession()

    assert session.eval_input(":target") == "Current target: cpu"
    assert session.eval_input(":target gpu-nvidia") == "Error: only the cpu REPL target is currently available"


def test_repl_quit_command_raises_system_exit():
    session = ReplSession()

    with pytest.raises(SystemExit):
        session.eval_input(":quit")


def test_repl_main_accepts_cpu_target(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError))

    assert main(["--target", "cpu"]) == 0
    assert "Remora REPL" in capsys.readouterr().out
