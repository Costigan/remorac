"""Tests for remora/capabilities.py — the executable support matrix."""

from remora.capabilities import (
    CAPABILITIES,
    Backend,
    Capability,
    Context,
    RANK_ANY,
    as_rows,
    lookup,
    supported_ops,
)
from remora.limits import MAX_DENSE_RANK


KNOWN_DTYPES = {"f32", "f64", "i32", "bool"}


def test_registry_non_empty():
    assert len(CAPABILITIES) > 0


def test_no_duplicate_keys():
    keys = [cap.unique_key for cap in CAPABILITIES]
    assert len(keys) == len(set(keys)), f"duplicate unique keys found: {[k for k in keys if keys.count(k) > 1]}"


def test_unsupported_has_reason():
    for cap in CAPABILITIES:
        if cap.status in ("unsupported", "limited"):
            assert cap.unsupported_reason, (
                f"{cap.op}/{cap.backend.value} is {cap.status} but has no unsupported_reason"
            )


def test_dtype_strings_known():
    for cap in CAPABILITIES:
        for d in cap.dtypes:
            assert d in KNOWN_DTYPES, (
                f"unknown dtype {d!r} in {cap.op}/{cap.backend.value}"
            )


def test_ranks_within_max():
    for cap in CAPABILITIES:
        if isinstance(cap.ranks, tuple):
            for r in cap.ranks:
                assert 0 <= r <= MAX_DENSE_RANK, (
                    f"rank {r} in {cap.op}/{cap.backend.value} exceeds MAX_DENSE_RANK={MAX_DENSE_RANK}"
                )


def test_all_dynamic_and_boxed_false():
    for cap in CAPABILITIES:
        assert cap.dynamic_shape is False, (
            f"{cap.op}/{cap.backend.value}: dynamic_shape must be False (current reality)"
        )
        assert cap.boxed_ragged is False, (
            f"{cap.op}/{cap.backend.value}: boxed_ragged must be False (current reality)"
        )


def test_all_static_shape_true():
    for cap in CAPABILITIES:
        assert cap.static_shape is True, (
            f"{cap.op}/{cap.backend.value}: static_shape must be True (current reality)"
        )


class TestLookup:
    def test_lookup_hit(self):
        result = lookup("map", Backend.CPU, dtype="f32", rank=1, context=Context.TOP_LEVEL)
        assert result is not None
        assert result.op == "map"
        assert result.status in ("supported", "limited")

    def test_lookup_miss_dtype(self):
        result = lookup("sort", Backend.GPU, dtype="i32")
        assert result is not None
        assert result.status == "unsupported"
        assert result.unsupported_reason

    def test_lookup_miss_op(self):
        result = lookup("nonexistent_op", Backend.CPU)
        assert result is None

    def test_lookup_returns_best_match(self):
        result = lookup("map", Backend.GPU, dtype="f32", rank=2, context=Context.MAP_BODY)
        assert result is not None
        assert result.backend == Backend.GPU
        assert result.dtypes == frozenset({"f32"})

    def test_lookup_fallback_cpu(self):
        result = lookup("map", Backend.CPU, dtype="i32")
        assert result is not None
        assert result.status in ("supported", "limited")


class TestSupportedOps:
    def test_interp_has_all_core_ops(self):
        ops = supported_ops(Backend.INTERP)
        for op in ("map", "fold", "reduce", "scan", "sort", "let", "if", "lambda"):
            assert op in ops, f"{op} missing from interpreter supported ops"

    def test_cpu_has_all_core_ops(self):
        ops = supported_ops(Backend.CPU)
        for op in ("map", "fold", "reduce", "scan", "sort", "let", "if", "lambda"):
            assert op in ops, f"{op} missing from CPU supported ops"

    def test_gpu_has_limited_core_ops(self):
        ops = supported_ops(Backend.GPU)
        for op in ("map", "fold", "sort", "matmul", "reverse", "transpose"):
            assert op in ops, f"{op} missing from GPU supported ops"


class TestAsRows:
    def test_as_rows_returns_list_of_dicts(self):
        rows = as_rows()
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert isinstance(rows[0], dict)
        for field in ("op", "backend", "dtypes", "ranks", "status"):
            assert field in rows[0]

    def test_as_rows_count_matches_registry(self):
        rows = as_rows()
        assert len(rows) == len(CAPABILITIES)
