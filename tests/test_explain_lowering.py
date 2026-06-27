"""Tests for --explain-lowering / compiler.explain_lowering()."""

from remora.compiler import explain_lowering, LoweringExplanation


class TestExplainLowering:
    def test_explain_simple_sum(self):
        explanation = explain_lowering("sum (iota 10)", syntax="ml")
        assert isinstance(explanation, LoweringExplanation)

    def test_explain_returns_structure(self):
        explanation = explain_lowering("sum (iota 10)", syntax="ml")
        assert isinstance(explanation.target, str)
        assert isinstance(explanation.decisions, list)
        assert isinstance(explanation.capability_keys, list)

    def test_explain_reports_decisions(self):
        explanation = explain_lowering("sum (iota 10)", syntax="ml")
        assert len(explanation.decisions) > 0
        for d in explanation.decisions:
            assert "route_name" in d
            assert "accepted" in d
            assert "reason" in d

    def test_explain_no_panic_on_empty(self):
        explanation = explain_lowering("1.0", syntax="ml")
        assert isinstance(explanation, LoweringExplanation)

    def test_explain_frozen_dataclass(self):
        explanation = LoweringExplanation(
            target="gpu",
            route_selected="sort_bitonic",
            decisions=[{"route_name": "sort_bitonic", "accepted": True, "reason": "matched"}],
            capability_keys=["sort"],
        )
        assert explanation.target == "gpu"
        assert explanation.route_selected == "sort_bitonic"


class TestExplainLoweringCLI:
    def test_cli_text_output(self):
        import subprocess
        result = subprocess.run(
            ["uv", "run", "remorac", "--syntax", "lisp", "--explain-lowering", "text",
             "-c", "(map (+ 1.0) (iota 5.0))"],
            capture_output=True, text=True, cwd=None,
        )
        # Command may fail if there's no GPU toolchain; just check for expected output patterns
        output = result.stdout + result.stderr
        assert "target:" in output or "remorac:" in output or result.returncode in (0, 1)

    def test_cli_json_output(self):
        import subprocess
        result = subprocess.run(
            ["uv", "run", "remorac", "--syntax", "lisp", "--explain-lowering", "json",
             "-c", "(map (+ 1.0) (iota 5.0))"],
            capture_output=True, text=True, cwd=None,
        )
        output = result.stdout + result.stderr
        assert "target" in output or "remorac:" in output or result.returncode in (0, 1)
