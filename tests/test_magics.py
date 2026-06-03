"""Tests for Remora Jupyter magics."""

from __future__ import annotations

import numpy as np
import pytest
from IPython.testing.globalipapp import get_ipython

from remora.jupyter.magics import RemoraMagics


@pytest.fixture(scope="session")
def ip():
    """Get a persistent IPython shell for testing."""
    ip = get_ipython()
    ip.extension_manager.load_extension("remora.jupyter.magics")
    return ip


def test_remora_magic_cpu(ip):
    """Test %%remora with the default CPU target."""
    result = ip.run_cell_magic("remora", "", "iota 5")
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.arange(5, dtype=np.int32))


def test_remora_magic_interp(ip):
    """Test %%remora with the interp target."""
    result = ip.run_cell_magic("remora", "--target interp", "iota 3")
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.arange(3, dtype=np.int32))


def test_remora_magic_out(ip):
    """Test the --out argument to bind results to Python variables."""
    ip.run_cell_magic("remora", "--out my_var", "iota 4")
    assert "my_var" in ip.user_ns
    assert isinstance(ip.user_ns["my_var"], np.ndarray)
    assert np.array_equal(ip.user_ns["my_var"], np.arange(4, dtype=np.int32))


def test_remora_magic_prelude(ip):
    """Test that the prelude is available in the magic."""
    result = ip.run_cell_magic("remora", "", "sum (iota 5)")
    assert int(result) == 10
