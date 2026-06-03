import pytest

from remora.repl import ReplSession, main


def test_repl_evaluates_expression():
    session = ReplSession()

    assert session.eval_input("1 + 2.0") == "3.0"


def test_repl_cpu_target_uses_compiled_execution(monkeypatch):
    calls = []

    def fake_evaluate_source_compiled(source, *, include_prelude=True):
        calls.append((source, include_prelude))
        from remora.runtime import EvaluationResult
        from remora.types import INT

        return EvaluationResult(7, INT)

    monkeypatch.setattr("remora.repl.evaluate_source_compiled", fake_evaluate_source_compiled)

    session = ReplSession()

    assert session.eval_input("1 + 2") == "7"
    assert len(calls) == 1
    source, include_prelude = calls[0]
    assert include_prelude is False
    assert source.endswith("\n1 + 2")
    assert "def sum xs = fold (+) 0.0 xs" in source


def test_repl_interp_target_uses_reference_evaluator(monkeypatch):
    compiled_calls = []
    interp_calls = []

    def fake_evaluate_source_compiled(source, *, include_prelude=True):
        compiled_calls.append((source, include_prelude))
        from remora.runtime import EvaluationResult
        from remora.types import INT

        return EvaluationResult(1, INT)

    def fake_evaluate_source(source, *, include_prelude=True):
        interp_calls.append((source, include_prelude))
        from remora.runtime import EvaluationResult
        from remora.types import INT

        return EvaluationResult(2, INT)

    monkeypatch.setattr("remora.repl.evaluate_source_compiled", fake_evaluate_source_compiled)
    monkeypatch.setattr("remora.repl.evaluate_source", fake_evaluate_source)

    session = ReplSession(target="interp")

    assert session.eval_input("1 + 1") == "2"
    assert compiled_calls == []
    assert len(interp_calls) == 1


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


def test_repl_loads_prelude_functions():
    session = ReplSession()

    assert session.eval_input("sum (iota 10)") == "45.0"
    assert session.eval_input(":type scale 2.0 (iota 4)") == "scale 2.0 (iota 4) : float[4]"


def test_repl_shows_prelude_and_user_definitions():
    session = ReplSession()

    prelude = session.eval_input(":prelude")
    assert prelude is not None
    assert "def sum xs = fold (+) 0.0 xs" in prelude
    assert "def scale s xs = map (* s) xs" in prelude

    assert session.eval_input(":defs") == "No user definitions."
    assert session.eval_input("def xs = iota 4") == "Defined: xs : int[4]"
    assert session.eval_input(":defs") == "def xs = iota 4"


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
    assert (
        session.eval_input("shape [[[[[[[[[[1]]]]]]]]]]")
        == "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]"
    )
    assert session.eval_input("rank [[[[[[[[[[1]]]]]]]]]]") == "10"


def test_repl_evaluates_indexing():
    session = ReplSession()

    assert session.eval_input("[[1, 2], [3, 4]][1, 0]") == "3"
    assert session.eval_input("[[1, 2], [3, 4]][0]") == "[1, 2]"
    assert (
        session.eval_input(
            "let xs = [[[[[[[[[[1]]]]]]]]]] in xs[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
        )
        == "1"
    )


def test_repl_displays_rank4_and_rank10_arrays():
    session = ReplSession()

    assert session.eval_input("let xs = [[[[1]]]] in map (\\x -> x + 1) xs") == "[[[[2]]]]"
    assert (
        session.eval_input(
            "let xs = [[[[[[[[[[1]]]]]]]]]] in map (\\x -> x + 1) xs"
        )
        == "[[[[[[[[[[2]]]]]]]]]]"
    )


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
    assert session.eval_input("sum (iota 4)") == "6.0"


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


def test_repl_reports_expression_parse_error_for_expression_like_input():
    session = ReplSession()

    message = session.eval_input("map \\x -> x [1,2,3]")

    assert message is not None
    assert message.startswith("Parse error:")
    assert "Expected one of" in message
    assert "* DEF" not in message
    assert "Hint: parenthesize lambda callables" in message


def test_repl_hints_parenthesized_array_literal_after_map_callable():
    session = ReplSession()

    message = session.eval_input("map (\\x -> x) [1,2,3]")

    assert message is not None
    assert message.startswith("Parse error:")
    assert "Hint: parenthesize array literal arguments" in message
    assert session.eval_input("map (\\x -> x) ([1,2,3])") == "[1, 2, 3]"


def test_repl_target_command():
    session = ReplSession()

    assert session.eval_input(":target") == "Current target: cpu"
    assert session.eval_input(":target interp") == "Target: interp"
    assert session.eval_input(":target") == "Current target: interp"
    assert session.eval_input(":target cpu") == "Target: cpu"
    assert session.eval_input(":target gpu-nvidia") == "Error: available REPL targets: cpu, interp"


def test_repl_quit_command_raises_system_exit():
    session = ReplSession()

    with pytest.raises(SystemExit):
        session.eval_input(":quit")


def test_repl_main_accepts_cpu_target(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError))

    assert main(["--target", "cpu"]) == 0
    assert "Remora REPL" in capsys.readouterr().out


def test_repl_main_accepts_interp_target(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError))

    assert main(["--target", "interp"]) == 0
    assert "Remora REPL" in capsys.readouterr().out
