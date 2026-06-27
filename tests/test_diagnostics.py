"""Tests for source-located diagnostics (Workstream 0 / Track E)."""

from remora.errors import RemoraError, SourceSpan
from remora.types import RemoraTypeError


class TestSourceSpan:
    def test_format_with_file(self):
        span = SourceSpan(file="test.remora", line=5, col=12)
        assert span.format() == "test.remora:5:12"

    def test_format_with_end(self):
        span = SourceSpan(file="test.lisp", line=3, col=1, end_line=3, end_col=10)
        assert "3:1" in span.format()
        assert "3:10" in span.format()

    def test_format_no_file(self):
        span = SourceSpan(file=None, line=10, col=0)
        assert span.format() == "10:0"

    def test_frozen(self):
        span = SourceSpan(file="f", line=1, col=1)
        try:
            span.line = 2  # type: ignore
            assert False, "should have raised"
        except Exception:
            pass


class TestRemoraErrorLocated:
    def test_located_attaches_span(self):
        error = RemoraError("something went wrong")
        span = SourceSpan(file="test.remora", line=42, col=8)
        error.located(span)
        assert error.span is not None
        assert error.span.file == "test.remora"

    def test_str_with_span(self):
        error = RemoraError("bad type")
        span = SourceSpan(file="prog.remora", line=3, col=7)
        error.located(span)
        msg = str(error)
        assert "prog.remora:3:7" in msg
        assert "bad type" in msg

    def test_str_without_span(self):
        error = RemoraError("plain error")
        assert str(error) == "plain error"


class TestDiagnosticsAreStable:
    def test_type_error_has_span_when_loc_provided(self):
        from remora.types import RemoraTypeError
        from remora.errors import SourceSpan
        e = RemoraTypeError("test error")
        e.located(SourceSpan(file="test.remora", line=5, col=1))
        assert e.span is not None
        assert "test.remora:5:1" in str(e)

    def test_type_error_without_loc_degrads_gracefully(self):
        from remora.types import RemoraTypeError
        e = RemoraTypeError("no loc")
        assert str(e) == "no loc"

    def test_type_error_with_actual_loc(self):
        from remora.ast_nodes import SourceLoc
        from remora.types import RemoraTypeError
        e = RemoraTypeError("with loc", loc=SourceLoc(file="prog.remora", line=3, col=7))
        assert e.span is not None
        assert "prog.remora:3:7" in str(e)

    def test_codegen_unavailable_is_remora_error(self):
        from remora.codegen import CodegenUnavailable
        assert issubclass(CodegenUnavailable, RemoraError)

    def test_gpu_scaffold_error_is_remora_error(self):
        from remora.gpu_lowering import GPUScaffoldError
        assert issubclass(GPUScaffoldError, RemoraError)

    def test_hir_lowering_error_is_remora_error(self):
        from remora.hir import HIRLoweringError
        assert issubclass(HIRLoweringError, RemoraError)

    def test_remora_error_subclasses_dont_panic_on_no_span(self):
        from remora.codegen import CodegenUnavailable
        e = CodegenUnavailable("test")
        msg = str(e)
        assert "test" in msg
