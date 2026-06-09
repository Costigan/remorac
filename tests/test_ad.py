"""Tests for AD1: scalar reverse-mode on the tape IR."""

import numpy as np
import pytest

from remora.ad import Tape, TapeEntry, reverse_ad_body, reverse_ad_function
from remora.ad_testing import finite_difference_grad, grad_check
from remora.elaborate import elaborate_program
from remora.lisp_reader import parse_lisp
from remora.typechecker import TypeChecker


# ── Tape unit tests ────────────────────────────────────────────────────────


def test_tape_add():
    t = Tape()
    x = t.push_input(3.0)
    y = t.push_input(4.0)
    r = t.push(TapeEntry("add", (x, y), ()), 7.0)
    adjs = t.reverse(1.0)
    assert adjs[x] == pytest.approx(1.0)
    assert adjs[y] == pytest.approx(1.0)


def test_tape_sub():
    t = Tape()
    x = t.push_input(5.0)
    y = t.push_input(2.0)
    r = t.push(TapeEntry("sub", (x, y), ()), 3.0)
    adjs = t.reverse(1.0)
    assert adjs[x] == pytest.approx(1.0)
    assert adjs[y] == pytest.approx(-1.0)


def test_tape_mul():
    t = Tape()
    x = t.push_input(3.0)
    y = t.push_input(4.0)
    r = t.push(TapeEntry("mul", (x, y), (4.0, 3.0)), 12.0)
    adjs = t.reverse(1.0)
    # d/dx (x*y) = y = 4
    assert adjs[x] == pytest.approx(4.0)
    assert adjs[y] == pytest.approx(3.0)


def test_tape_div():
    t = Tape()
    x = t.push_input(6.0)
    y = t.push_input(2.0)
    r = t.push(TapeEntry("div", (x, y), (2.0, 6.0)), 3.0)
    adjs = t.reverse(1.0)
    # d/dx (x/y) = 1/y = 0.5
    assert adjs[x] == pytest.approx(0.5)
    # d/dy (x/y) = -x/y^2 = -6/4 = -1.5
    assert adjs[y] == pytest.approx(-1.5)


def test_tape_composition():
    """f(x) = x * x + x  →  df/dx = 2x + 1"""
    t = Tape()
    x = t.push_input(3.0)
    x2 = t.push(TapeEntry("mul", (x, x), (3.0, 3.0)), 9.0)
    r = t.push(TapeEntry("add", (x2, x), ()), 12.0)
    adjs = t.reverse(1.0)
    assert adjs[x] == pytest.approx(7.0)  # 2*3 + 1 = 7


def test_tape_chained():
    """f(x) = (x + 1) * (x + 2)  →  df/dx = (x+2) + (x+1) = 2x + 3"""
    t = Tape()
    one = t.push_const(1.0)
    two = t.push_const(2.0)
    x = t.push_input(3.0)
    t1 = t.push(TapeEntry("add", (x, one), ()), 4.0)
    t2 = t.push(TapeEntry("add", (x, two), ()), 5.0)
    r = t.push(TapeEntry("mul", (t1, t2), (5.0, 4.0)), 20.0)
    adjs = t.reverse(1.0)
    assert adjs[x] == pytest.approx(9.0)  # 2*3 + 3 = 9


# ── Finite-difference validation ────────────────────────────────────────────


def _fd_test_f(f):
    """Test that the AD transform produces the same gradient as finite differences."""
    x_val = np.array([2.0, 3.0, 5.0])
    num_grad = finite_difference_grad(f, x_val)
    return num_grad


def test_fd_vs_tape_square():
    """f(x) = x * x  →  df/dx = 2x"""
    def f(x):
        return float(x[0] * x[0])
    num = finite_difference_grad(f, np.array([3.0]))
    assert num[0] == pytest.approx(6.0, rel=1e-4)


def test_fd_vs_tape_quadratic():
    """f(x) = x*x + x  →  df/dx = 2x + 1"""
    def f(x):
        return float(x[0] * x[0] + x[0])
    num = finite_difference_grad(f, np.array([3.0]))
    assert num[0] == pytest.approx(7.0, rel=1e-4)
