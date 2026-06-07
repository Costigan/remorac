"""Shared pytest fixtures for the Remora prototype."""

import os
import pytest


def gpu_required_or_skip(reason: str) -> None:
    """Skip the test unless GPU is available, but fail if REMORA_TEST_GPU=1.

    Use this in GPU-dependent tests after catching a ``RuntimeUnavailable``
    or ``CodegenUnavailable`` exception::

        try:
            rt = CUDARuntime()
        except RuntimeUnavailable as exc:
            gpu_required_or_skip(str(exc))
    """
    if os.environ.get("REMORA_TEST_GPU") == "1":
        pytest.fail(f"GPU required but not available: {reason}")
    pytest.skip(f"GPU not available: {reason}")
