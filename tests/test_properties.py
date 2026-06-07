"""Property-based tests comparing compiled CPU output against the interpreter."""

import numpy as np
import pytest

from remora.runtime import evaluate_source, evaluate_source_compiled


def _assert_cpu_matches_interp(source: str):
    """Assert that compiled CPU and interpreter produce the same result."""
    interp = evaluate_source(source)
    compiled = evaluate_source_compiled(source)
    if interp.type != compiled.type:
        pytest.fail(f"type mismatch: {interp.type} vs {compiled.type}")
    np.testing.assert_equal(compiled.value, interp.value)


class TestCpuVsInterp:
    def test_scalar_arithmetic(self):
        _assert_cpu_matches_interp("1 + 2 * 3 - 4")

    def test_float_ops(self):
        _assert_cpu_matches_interp("1.0 + 2.0 * 3.0")

    def test_bool_logic(self):
        _assert_cpu_matches_interp("true && false || true")

    def test_array_literal(self):
        _assert_cpu_matches_interp("[1, 2, 3, 4, 5]")

    def test_map_i32(self):
        _assert_cpu_matches_interp("map (* 2) (iota 10)")

    def test_map_f32(self):
        _assert_cpu_matches_interp("map (* 2.0) (iota 10)")

    def test_binary_map(self):
        _assert_cpu_matches_interp("map (+) (iota 5) (iota 5)")

    def test_scalar_fold(self):
        _assert_cpu_matches_interp("fold (+) 0 (iota 10)")

    def test_float_fold(self):
        _assert_cpu_matches_interp("fold (+) 0.0 (iota 10)")

    def test_array_cell_fold(self):
        _assert_cpu_matches_interp(
            "let xs = [[1, 2], [3, 4]] in let init = [0, 0] in fold (+) init xs"
        )

    def test_nested_map_fold(self):
        _assert_cpu_matches_interp("let xs = iota 10 in fold (+) 0 (map (* 2) xs)")

    def test_dynamic_index(self):
        _assert_cpu_matches_interp("let xs = iota 10 in let idx = 5 in xs[idx]")

    def test_cell_map_index(self):
        _assert_cpu_matches_interp(
            "let xs = [[1, 2], [3, 4]] in map (\\row -> row[0] + row[1]) xs"
        )

    def test_prelude_sum(self):
        _assert_cpu_matches_interp("sum (iota 10)")

    def test_prelude_any(self):
        _assert_cpu_matches_interp("let xs = [true, false, true] in any xs")

    def test_prelude_max(self):
        _assert_cpu_matches_interp("max 3 7")

    def test_let_chain(self):
        _assert_cpu_matches_interp(
            "let a = iota 10 in "
            "let b = map (* 2) a in "
            "let c = map (+ 1) b in "
            "fold (+) 0 c"
        )

    def test_reverse(self):
        _assert_cpu_matches_interp("reverse [1, 2, 3, 4]")

    def test_dot_product(self):
        _assert_cpu_matches_interp("let a = iota 4 in let b = map (* 2) a in fold (+) 0 (map (*) a b)")


class TestRandomizedCpuVsInterp:
    """Property tests with randomized inputs comparing CPU vs interpreter."""

    @pytest.mark.parametrize("size", [1, 5, 10, 50])
    def test_map_scale_random_size(self, size):
        _assert_cpu_matches_interp(f"map (* 2) (iota {size})")

    @pytest.mark.parametrize("size", [1, 4, 16, 32])
    def test_fold_sum_random_size(self, size):
        _assert_cpu_matches_interp(f"fold (+) 0 (iota {size})")

    @pytest.mark.parametrize("size", [1, 3, 7, 15])
    def test_binary_map_random_size(self, size):
        _assert_cpu_matches_interp(f"map (+) (iota {size}) (iota {size})")

    @pytest.mark.parametrize("mult", [1, 3, 7, 11])
    def test_map_multiply_random_factor(self, mult):
        _assert_cpu_matches_interp(f"map (* {mult}) (iota 8)")

    def test_nested_lets_random(self):
        _assert_cpu_matches_interp(
            "let a = iota 6 in "
            "let b = map (* 3) a in "
            "let c = map (+ 1) b in "
            "fold (+) 0 c"
        )

    def test_cell_map_random(self):
        _assert_cpu_matches_interp(
            "let xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] in "
            "map (\\row -> fold (+) 0 row) xs"
        )
