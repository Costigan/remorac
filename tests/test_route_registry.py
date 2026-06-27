"""Tests for remora/route_registry.py — backend route selection."""

from remora.route_registry import (
    Route,
    RouteContext,
    RouteDecision,
    RouteResult,
    build_route_registry_test_routes,
    select_route,
)
from remora.capabilities import CAPABILITIES, Capability


class TestRouteRegistryIntegrity:
    def test_routes_exist(self):
        routes = build_route_registry_test_routes()
        assert len(routes) > 0

    def test_priorities_are_total(self):
        routes = build_route_registry_test_routes()
        priorities = [r.priority for r in routes]
        assert priorities == sorted(priorities), "routes must be sorted by priority"
        assert len(priorities) == len(set(priorities)), "priorities must be unique"

    def test_every_route_has_name_and_build(self):
        routes = build_route_registry_test_routes()
        for r in routes:
            assert r.name
            assert r.build is not None, f"route {r.name} missing build function"

    def test_capability_keys_exist_in_registry(self):
        routes = build_route_registry_test_routes()
        all_cap_keys = {(c.op, c.backend.value) for c in CAPABILITIES}
        for route in routes:
            for key in route.capability_keys:
                found = any(
                    c.op == key and c.backend.value == "gpu"
                    for c in CAPABILITIES
                )
                assert found, (
                    f"route {route.name!r} references capability key {key!r} "
                    f"which has no GPU entry in capabilities.py"
                )

    def test_general_dispatch_is_last(self):
        routes = build_route_registry_test_routes()
        last = routes[-1]
        assert last.name == "general_dispatch", (
            f"last route should be general_dispatch, got {last.name}"
        )


class TestSelectRoute:
    def test_select_route_with_empty_function(self):
        from remora.hir import HIRFunction, HIRParam, HIRLit
        from remora.types import INT
        fn = HIRFunction(
            name="test_fn",
            params=[HIRParam("x", INT)],
            body=HIRLit(42, INT),
            return_type=INT,
        )
        route, decisions = select_route(fn, RouteContext(kernel_name="test", toolchain=None))
        assert route is not None or len(decisions) > 0

    def test_all_decisions_record_reason(self):
        from remora.hir import HIRFunction, HIRParam, HIRLit
        from remora.types import INT
        fn = HIRFunction(
            name="test_fn",
            params=[HIRParam("x", INT)],
            body=HIRLit(42, INT),
            return_type=INT,
        )
        route, decisions = select_route(fn, RouteContext(kernel_name="test", toolchain=None))
        for d in decisions:
            assert d.route_name
            assert d.reason
            assert isinstance(d.accepted, bool)


class TestRouteResult:
    def test_route_result_dataclass(self):
        result = RouteResult(ptx="ptx_text", metas=[], plan=None)
        assert result.ptx == "ptx_text"
        assert result.metas == []
        assert result.plan is None


class TestRouteDecision:
    def test_route_decision_dataclass(self):
        d = RouteDecision(route_name="test_route", accepted=True, reason="accepted", capability_keys=("map",))
        assert d.route_name == "test_route"
        assert d.accepted is True
        assert "accepted" in d.reason
