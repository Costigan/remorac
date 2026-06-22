"""Shared pytest fixtures for the Remora prototype."""

import os
import pytest

# ---------------------------------------------------------------------------
# GPU is a first-class target: locally, the GPU path gets the same emphasis as
# the CPU path.  We therefore default ``REMORA_TEST_GPU`` to "1" when it is
# *unset*, which means:
#   - GPU tests run (they already do whenever a CUDA runtime is present), and
#   - GPU/toolchain *unavailability* is a hard FAILURE, not a silent skip.
# A silent skip is the coverage analogue of a silent miscompile: it quietly
# erases the GPU coverage you think you have.  On a machine that is supposed to
# have a GPU, losing GPU coverage must be loud.
#
# Opt out explicitly when you really do not have/want a GPU run:
#     REMORA_TEST_GPU=0 uv run pytest
# CI sets REMORA_TEST_GPU explicitly (to "0" today, since CI has no GPU), so
# this default never overrides it.
os.environ.setdefault("REMORA_TEST_GPU", "1")


def _gpu_required() -> bool:
    return os.environ.get("REMORA_TEST_GPU") == "1"


def pytest_configure(config: "pytest.Config") -> None:
    """When GPU is required, probe the runtime once so an unavailable GPU
    produces a single clear error instead of a wall of per-test failures."""
    if not _gpu_required():
        return
    try:
        from remora.runtime import CUDARuntime, RuntimeUnavailable
    except Exception:
        # If the runtime module can't even be imported, let the real tests
        # surface the underlying error rather than masking it here.
        return
    try:
        rt = CUDARuntime()
    except RuntimeUnavailable as exc:
        raise pytest.UsageError(
            "REMORA_TEST_GPU is enabled (the local default) but no CUDA "
            f"runtime is available: {exc}\n"
            "Fix the GPU/toolchain, or run CPU-only with "
            "'REMORA_TEST_GPU=0 uv run pytest'."
        )
    else:
        if hasattr(rt, "close"):
            rt.close()


def gpu_required_or_skip(reason: str) -> None:
    """Skip the test unless GPU is available, but fail if REMORA_TEST_GPU=1.

    Use this in GPU-dependent tests after catching a ``RuntimeUnavailable``
    or ``CodegenUnavailable`` exception::

        try:
            rt = CUDARuntime()
        except RuntimeUnavailable as exc:
            gpu_required_or_skip(str(exc))
    """
    if _gpu_required():
        pytest.fail(f"GPU required but not available: {reason}")
    pytest.skip(f"GPU not available: {reason}")
