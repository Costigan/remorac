"""Tests that docs-visible support claims match capabilities.py."""

from remora.capabilities import (
    Backend,
    CAPABILITIES,
    as_rows,
    lookup,
    supported_ops,
)


class TestDocsMatchCapabilities:
    def test_gpu_sort_f32_only(self):
        gpu_sort_f32 = lookup("sort", Backend.GPU, dtype="f32")
        assert gpu_sort_f32 is not None
        assert gpu_sort_f32.status in ("supported", "limited")

        gpu_sort_i32 = lookup("sort", Backend.GPU, dtype="i32")
        assert gpu_sort_i32 is not None
        assert gpu_sort_i32.status == "unsupported"

    def test_gpu_grade_f32_only(self):
        gpu_grade_f32 = lookup("grade", Backend.GPU, dtype="f32")
        assert gpu_grade_f32 is not None
        assert gpu_grade_f32.status in ("supported", "limited")

        gpu_grade_i32 = lookup("grade", Backend.GPU, dtype="i32")
        assert gpu_grade_i32 is not None
        assert gpu_grade_i32.status == "unsupported"

    def test_gpu_matmul_f32_only(self):
        gpu_mm_f32 = lookup("matmul", Backend.GPU, dtype="f32")
        assert gpu_mm_f32 is not None
        assert gpu_mm_f32.status in ("supported", "limited")

        gpu_mm_f64 = lookup("matmul", Backend.GPU, dtype="f64")
        assert gpu_mm_f64 is not None
        assert gpu_mm_f64.status == "unsupported"

    def test_gpu_pair_unsupported(self):
        for op in ("pair", "first", "second"):
            cap = lookup(op, Backend.GPU)
            assert cap is not None
            assert cap.status == "unsupported", f"{op} should be unsupported on GPU"

    def test_gpu_im2col_unsupported(self):
        cap = lookup("im2col", Backend.GPU)
        assert cap is not None
        assert cap.status == "unsupported"

    def test_gpu_box_unsupported(self):
        cap = lookup("box", Backend.GPU)
        assert cap is not None
        assert cap.status == "unsupported"

    def test_cpu_supports_core_ops(self):
        ops = supported_ops(Backend.CPU)
        for op in ("map", "fold", "reduce", "scan", "sort", "let", "if",
                    "lambda", "reverse", "transpose", "reshape", "take", "drop"):
            assert op in ops, f"{op} should be supported on CPU"

    def test_interp_supports_full_subset(self):
        ops = supported_ops(Backend.INTERP)
        for op in ("pair", "box", "unbox", "im2col", "col2im", "compose"):
            assert op in ops, f"{op} should be supported in interpreter"

    def test_as_rows_is_valid_markdown_table_source(self):
        rows = as_rows()
        assert len(rows) > 0
        for row in rows:
            assert "op" in row
            assert "backend" in row
            assert "status" in row

    def test_all_gpu_unsupported_have_reason(self):
        for cap in CAPABILITIES:
            if cap.backend == Backend.GPU and cap.status == "unsupported":
                assert cap.unsupported_reason, (
                    f"{cap.op} GPU unsupported but no reason given"
                )
