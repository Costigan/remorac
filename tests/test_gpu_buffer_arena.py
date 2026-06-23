"""Tests for the device buffer arena (DeviceMemoryPool).

The pool recycles device buffers by size class so iterative Python+Remora loops
stop paying a cuMemAlloc/cuMemFree on every execute() call, and so buffers are
shared across executors that share a runtime.
"""
import numpy as np
import pytest

from remora.compiler import compile_function_source_to_mlir_gpu_ptx
from remora.executor import RemoraExecutor
from remora.runtime import CUDARuntime, DeviceMemoryPool, RuntimeUnavailable
from remora.types import FLOAT, ArrayType, StaticDim
from conftest import gpu_required_or_skip


# ---------------------------------------------------------------------------
# Pure-logic tests (no GPU needed)
# ---------------------------------------------------------------------------


def test_size_class_rounds_to_power_of_two_with_floor():
    sc = DeviceMemoryPool.size_class
    assert sc(0) == 256
    assert sc(1) == 256
    assert sc(256) == 256
    assert sc(257) == 512
    assert sc(1024) == 1024
    assert sc(1025) == 2048
    assert sc(4 * 1024 * 1024) == 4 * 1024 * 1024  # already a power of two


class _FakeRuntime:
    """Minimal stand-in tracking live allocations (no real device)."""

    def __init__(self) -> None:
        self.live: set[int] = set()
        self._next = 1000

    def alloc(self, nbytes: int) -> int:
        ptr = self._next
        self._next += nbytes
        self.live.add(ptr)
        return ptr

    def free(self, ptr: int) -> None:
        self.live.discard(ptr)


def test_pool_recycles_same_size_class_and_drains():
    rt = _FakeRuntime()
    pool = DeviceMemoryPool(rt)

    p = pool.alloc(1000)               # class 1024 -> one real alloc
    assert pool.stats()["device_allocs"] == 1
    pool.free(p, 1000)

    p2 = pool.alloc(900)               # same class 1024 -> reused, no new alloc
    assert p2 == p
    assert pool.stats()["device_allocs"] == 1
    assert pool.stats()["reuses"] == 1

    pool.free(p2, 900)
    assert len(rt.live) == 1           # still held by the pool

    pool.clear()                       # drain -> real frees
    assert len(rt.live) == 0
    assert pool.stats()["pooled_buffers"] == 0


def test_pool_disable_drains_and_bypasses():
    rt = _FakeRuntime()
    pool = DeviceMemoryPool(rt)
    pool.free(pool.alloc(512), 512)
    assert len(rt.live) == 1
    pool.set_enabled(False)            # drains
    assert len(rt.live) == 0
    p = pool.alloc(512)                # straight to runtime
    pool.free(p, 512)                  # straight free, not pooled
    assert len(rt.live) == 0
    assert pool.stats()["pooled_buffers"] == 0


# ---------------------------------------------------------------------------
# GPU end-to-end tests
# ---------------------------------------------------------------------------


def _gpu_runtime() -> CUDARuntime:
    try:
        return CUDARuntime()
    except RuntimeUnavailable as exc:
        gpu_required_or_skip(str(exc))


_SRC = "def f xs = map (* 2.0) xs"


def _compile(n: int):
    return compile_function_source_to_mlir_gpu_ptx(
        _SRC, "f", (ArrayType(FLOAT, (StaticDim(n),)),), include_prelude=False,
    )


def test_pool_eliminates_allocs_across_execute_calls():
    rt = _gpu_runtime()
    try:
        ptx, kernels, _ = _compile(1024)
        x = np.arange(1024, dtype=np.float32)
        ex = RemoraExecutor(ptx, kernels, runtime=rt)
        try:
            ex.execute_main([x])  # warmup populates the pool
            allocs = rt.memory_pool.stats()["device_allocs"]
            for _ in range(25):
                out = ex.execute_main([x])
                np.testing.assert_allclose(out, x * 2.0, rtol=1e-5)
            stats = rt.memory_pool.stats()
            # no new device allocations after warmup; everything recycled
            assert stats["device_allocs"] == allocs
            assert stats["reuses"] > 0
        finally:
            ex.close()
    finally:
        rt.close()


def test_pool_shared_across_executors_on_same_runtime():
    rt = _gpu_runtime()
    try:
        ptx, kernels, _ = _compile(1024)
        x = np.arange(1024, dtype=np.float32)

        ex1 = RemoraExecutor(ptx, kernels, runtime=rt)
        ex1.execute_main([x])
        ex1.close()  # returns buffers to the shared pool (does not drain it)

        allocs = rt.memory_pool.stats()["device_allocs"]
        ex2 = RemoraExecutor(ptx, kernels, runtime=rt)
        try:
            out = ex2.execute_main([x])
            np.testing.assert_allclose(out, x * 2.0, rtol=1e-5)
            # ex2 reused ex1's buffers from the shared pool: no new device allocs
            assert rt.memory_pool.stats()["device_allocs"] == allocs
        finally:
            ex2.close()
    finally:
        rt.close()


def test_runtime_close_drains_the_pool():
    rt = _gpu_runtime()
    ptx, kernels, _ = _compile(256)
    x = np.arange(256, dtype=np.float32)
    ex = RemoraExecutor(ptx, kernels, runtime=rt)
    ex.execute_main([x])
    assert rt.memory_pool.stats()["pooled_buffers"] > 0
    ex.close()
    rt.close()  # must drain without error
    # a fresh pool on a reused-name runtime would start empty; closing twice is safe
    rt.close()
