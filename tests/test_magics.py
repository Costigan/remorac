"""Tests for Remora Jupyter magics."""

from __future__ import annotations

import io
import sys

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


def test_remora_eval_expression(ip):
    """Test %remora_eval with a simple expression."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        ip.run_line_magic("remora_eval", "iota 3")
    finally:
        sys.stdout = old_stdout
    output = buf.getvalue().strip()
    assert "[0, 1, 2]" in output


def test_remora_eval_definition_and_use(ip):
    """Test %remora_eval with definition then use."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        ip.run_line_magic("remora_eval", "def inc x = x + 1")
        ip.run_line_magic("remora_eval", "inc 41")
    finally:
        sys.stdout = old_stdout
    output = buf.getvalue().strip()
    assert "42" in output


def test_remora_eval_reset(ip):
    """Test %remora_eval --reset clears session state."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        ip.run_line_magic("remora_eval", "--reset")
    finally:
        sys.stdout = old_stdout
    output = buf.getvalue().strip()
    assert "reset" in output.lower()


def test_remora_magic_types(ip):
    """Test %%remora --types prints type information."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        ip.run_cell_magic(
            "remora", "--types --syntax lisp",
            "(define/pi () (scale [xs (Array Float 4)] (Array Float 4)) (map (* 2.0) xs))",
        )
    finally:
        sys.stdout = old_stdout
    output = buf.getvalue()
    assert "scale" in output
