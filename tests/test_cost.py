"""Unit tests for remora/cost.py — cost-annotation data structures."""

from remora.cost import CostShape, ScheduleCandidate


class TestCostShape:
    def test_instantiate(self):
        cs = CostShape(
            elem_count=1000,
            bytes_read=4000,
            bytes_written=4000,
            flops=500,
            temporary_bytes=0,
            irregularity="regular",
        )
        assert cs.elem_count == 1000
        assert cs.irregularity == "regular"

    def test_frozen(self):
        cs = CostShape(
            elem_count=1, bytes_read=4, bytes_written=4,
            flops=0, temporary_bytes=0, irregularity="regular",
        )
        try:
            cs.elem_count = 2  # type: ignore
            assert False, "dataclass should be frozen"
        except Exception:
            pass

    def test_symbolic_dim(self):
        cs = CostShape(
            elem_count="N",
            bytes_read="4*N",
            bytes_written="4*N",
            flops="N",
            temporary_bytes=0,
            irregularity="regular",
        )
        assert cs.elem_count == "N"


class TestScheduleCandidate:
    def test_instantiate(self):
        sc = ScheduleCandidate(
            backend="cpu",
            plan_kind="standalone_map",
            estimated_cost=None,
            requirements=["map", "f32"],
            fallback_reason=None,
        )
        assert sc.backend == "cpu"
        assert sc.plan_kind == "standalone_map"

    def test_frozen(self):
        sc = ScheduleCandidate(
            backend="gpu",
            plan_kind="fused_map_reduce",
            estimated_cost=None,
            requirements=["map", "f32"],
            fallback_reason=None,
        )
        try:
            sc.backend = "cpu"  # type: ignore
            assert False, "dataclass should be frozen"
        except Exception:
            pass

    def test_fallback_reason(self):
        sc = ScheduleCandidate(
            backend="gpu",
            plan_kind="dynamic_loop",
            estimated_cost=None,
            requirements=["map", "f32"],
            fallback_reason="GPU boxes unsupported",
        )
        assert sc.fallback_reason == "GPU boxes unsupported"
