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


def _evaluate_lisp(source: str):
    """Evaluate a Lisp-syntax source on compiled CPU."""
    return evaluate_source_compiled(source, include_prelude=False, syntax="lisp")


def _assert_implicit_matches_explicit(lisp_implicit: str, lisp_explicit: str):
    """Assert that Lisp implicit (auto-lifted) and explicit (map) produce same result."""
    r_imp = _evaluate_lisp(lisp_implicit)
    r_exp = _evaluate_lisp(lisp_explicit)
    if r_imp.type != r_exp.type:
        pytest.fail(f"type mismatch: {r_imp.type} vs {r_exp.type}")
    np.testing.assert_equal(r_imp.value, r_exp.value, 
        err_msg=f"implicit: {r_imp.value}, explicit: {r_exp.value}")


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


class TestRankPolymorphism:
    """Property tests for rank-polymorphic auto-lifting."""

    def test_scalar_auto_lift_binary(self):
        _assert_implicit_matches_explicit(
            "(+ [1 2 3] [4 5 6])",
            "(map + [1 2 3] [4 5 6])",
        )

    def test_scalar_auto_lift_unary(self):
        _assert_implicit_matches_explicit(
            "(* 2 [1 2 3 4 5])",
            "(map (* 2) [1 2 3 4 5])",
        )

    def test_scalar_auto_lift_commutative(self):
        _assert_implicit_matches_explicit(
            "(- [10 20 30] 5)",
            "(map (lambda (x) (- x 5)) [10 20 30])",
        )

    @pytest.mark.parametrize("size", [1, 3, 7, 15])
    def test_implicit_add_vs_explicit_map(self, size):
        arr = " ".join(str(i) for i in range(size))
        _assert_implicit_matches_explicit(
            f"(+ [{arr}] [{arr}])",
            f"(map + [{arr}] [{arr}])",
        )

    @pytest.mark.parametrize("size", [1, 3, 5, 10])
    def test_implicit_multiply_vs_explicit_map(self, size):
        arr = " ".join(str(i * 2) for i in range(size))
        _assert_implicit_matches_explicit(
            f"(* 3 [{arr}])",
            f"(map (* 3) [{arr}])",
        )

    def test_broadcast_vector_to_matrix(self):
        # Broadcast produces correct values: [10+1,10+2,10+3], [20+4,20+5,20+6]
        r = _evaluate_lisp("(+ [10 20] [[1 2 3] [4 5 6]])")
        expected = [[11, 12, 13], [24, 25, 26]]
        np.testing.assert_equal(r.value, expected)

    def test_broadcast_scalar_to_matrix(self):
        r = _evaluate_lisp("(* 2 [[1 2] [3 4]])")
        expected = [[2, 4], [6, 8]]
        np.testing.assert_equal(r.value, expected)

    def test_vector_cell_fold_auto_lift(self):
        r = _evaluate_lisp(
            "(define (sum-vec [v 1]) (fold + 0 v)) (sum-vec [[1 2] [3 4]])"
        )
        expected = [3, 7]  # [1+2, 3+4]
        np.testing.assert_equal(r.value, expected)

    def test_direct_vs_auto_lift_same_result(self):
        """Direct application (matching ranks) equals auto-lifted application."""
        src_def = "(define (sum-vec [v 1]) (fold + 0 v)) "
        r_direct = _evaluate_lisp(src_def + "(sum-vec [1 2 3 4])")
        # Direct: 1+2+3+4 = 10
        assert r_direct.value == 10
        # Lifted over 2x2 matrix: [[1,2],[3,4]] -> [1+2, 3+4] = [3, 7]
        r_lifted = _evaluate_lisp(src_def + "(sum-vec [[1 2] [3 4]])")
        np.testing.assert_equal(r_lifted.value, [3, 7])

    def test_lambda_auto_lift(self):
        _assert_implicit_matches_explicit(
            "((lambda (x) (* x 2)) [1 2 3])",
            "(map (lambda (x) (* x 2)) [1 2 3])",
        )

    def test_subtract_auto_lift(self):
        _assert_implicit_matches_explicit(
            "(- [5 3 1] [1 1 1])",
            "(map - [5 3 1] [1 1 1])",
        )


class TestRankPolymorphismErrors:
    """Type error tests for rank polymorphism."""

    def test_incompatible_frames_error(self):
        from remora.typechecker import TypeChecker
        from remora.lisp_reader import parse_lisp
        from remora.types import RemoraTypeError

        # [1 2] (rank 1) + [[1] [2] [3]] (rank 2, shape mismatch)
        # frame shapes: (2,) and (3, 1) - first dims differ
        # Actually [1 2] has shape (2), [[1][2][3]] has shape (3,1)
        # First dims 2 vs 3 are incompatible
        with pytest.raises(RemoraTypeError):
            tc = TypeChecker()
            prog = parse_lisp("(+ [1 2] [[1] [2] [3]])")
            tc.check_program(prog)


class TestPhase3Operators:
    """Property tests for Phase 3: reduce/scan/fold/trace."""

    def test_reduce_same_as_fold(self):
        r_fold = _evaluate_lisp("(fold + 0 (iota 10))")
        r_reduce = _evaluate_lisp("(reduce + 0 (iota 10))")
        assert r_fold.value == r_reduce.value

    def test_iscan_correctness(self):
        r = _evaluate_lisp("(iscan + 0 [2 10 5])")
        np.testing.assert_array_equal(r.value, [2, 12, 17])

    def test_escan_correctness(self):
        r = _evaluate_lisp("(escan + 0 [2 10 5])")
        np.testing.assert_array_equal(r.value, [0, 2, 12])

    def test_trace_same_as_iscan(self):
        r_iscan = _evaluate_lisp("(iscan + 0 [2 10 5])")
        r_trace = _evaluate_lisp("(trace + 0 [2 10 5])")
        np.testing.assert_array_equal(r_iscan.value, r_trace.value)

    def test_trace_right_correctness(self):
        r = _evaluate_lisp("(trace-right + 0 [2 10 5])")
        np.testing.assert_array_equal(r.value, [17, 15, 5])

    def test_fold_right_same_for_associative(self):
        r = _evaluate_lisp("(fold-right + 0 [1 2 3 4])")
        assert r.value == 10

    def test_scan_variants_aliases(self):
        r1 = _evaluate_lisp("(iscan + 0 [2 10 5])")
        r2 = _evaluate_lisp("(scan + 0 [2 10 5])")
        np.testing.assert_array_equal(r1.value, r2.value)

        r3 = _evaluate_lisp("(iscan/zero + 0 [2 10 5])")
        np.testing.assert_array_equal(r1.value, r3.value)

        r4 = _evaluate_lisp("(escan/zero + 0 [2 10 5])")
        r5 = _evaluate_lisp("(escan + 0 [2 10 5])")
        np.testing.assert_array_equal(r4.value, r5.value)

    @pytest.mark.parametrize("size", [1, 3, 7, 15])
    def test_scan_identity(self, size):
        arr = " ".join(str(i) for i in range(size))
        r = _evaluate_lisp(f"(iscan + 0 [{arr}])")
        # Prefix sum of [0, 1, ..., size-1] should be [0, 1, 3, 6, 10, ...]
        expected = np.cumsum(np.arange(size))
        np.testing.assert_array_equal(r.value, expected)


class TestPhase4Operators:
    """Property tests for Phase 4: additional primitives."""

    def test_length_correctness(self):
        assert _evaluate_lisp("(length [1 2 3 4 5])").value == 5
        assert _evaluate_lisp("(length [[1 2] [3 4] [5 6]])").value == 3

    def test_rotate_correctness(self):
        r = _evaluate_lisp("(rotate [1 2 3 4 5] 2)")
        np.testing.assert_array_equal(r.value, [3, 4, 5, 1, 2])

    def test_rotate_rank2_correctness(self):
        r = _evaluate_lisp("(rotate [[1 2] [3 4] [5 6]] 1)")
        np.testing.assert_array_equal(r.value, [[3, 4], [5, 6], [1, 2]])

    def test_rotate_identity(self):
        r = _evaluate_lisp("(rotate [1 2 3 4 5] 0)")
        np.testing.assert_array_equal(r.value, [1, 2, 3, 4, 5])

    def test_subarray_correctness(self):
        r = _evaluate_lisp("(subarray [[1 2 3] [4 5 6] [7 8 9]] [1 0] [2 2])")
        np.testing.assert_array_equal(r.value, [[4, 5], [7, 8]])

    def test_append_correctness(self):
        r = _evaluate_lisp("(append [1 2] [3 4 5])")
        np.testing.assert_array_equal(r.value, [1, 2, 3, 4, 5])

    def test_append_rank2_correctness(self):
        r = _evaluate_lisp("(append [[1 2] [3 4]] [[5 6] [7 8]])")
        np.testing.assert_array_equal(r.value, [[1, 2], [3, 4], [5, 6], [7, 8]])

    def test_select_correctness(self):
        assert _evaluate_lisp("(select #t 10 20)").value == 10
        assert _evaluate_lisp("(select #f 10 20)").value == 20

    def test_with_shape_correctness(self):
        r = _evaluate_lisp("(with-shape 5 [3 2])")
        np.testing.assert_array_equal(r.value, [[5, 5], [5, 5], [5, 5]])

    def test_box_unbox_passthrough(self):
        r = _evaluate_lisp("(unbox (box [1 2 3]) (len v) v)")
        np.testing.assert_array_equal(r.value, [1, 2, 3])

    def test_box_unbox_with_fold(self):
        r = _evaluate_lisp("(unbox (box [10 20 30]) (len xs) (fold + 0 xs))")
        assert r.value == 60


def _assert_lisp_compiled_matches_interp(source: str):
    """Assert that Lisp-syntax compiled CPU and interpreter produce the same result."""
    interp = evaluate_source(source, include_prelude=False, syntax="lisp")
    compiled = evaluate_source_compiled(source, include_prelude=False, syntax="lisp")
    if interp.type != compiled.type:
        pytest.fail(f"type mismatch: {interp.type} vs {compiled.type}")
    if isinstance(compiled.value, np.ndarray) and compiled.value.dtype.kind == 'f':
        np.testing.assert_allclose(compiled.value, interp.value, rtol=1e-5)
    elif isinstance(compiled.value, float):
        assert abs(compiled.value - interp.value) < 1e-5, \
            f"float mismatch: {compiled.value} vs {interp.value}"
    else:
        np.testing.assert_equal(compiled.value, interp.value)


class TestCompiledVsInterpreter:
    """Systematic compiled-vs-interpreter comparison for all operators."""

    # ── Arithmetic ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("src", [
        "(+ 10 20)",
        "(- 30 7)",
        "(* 6 7)",
        "(/ 10.0 3.0)",
        "(< 5 10)",
        "(> 10 5)",
        "(<= 5 5)",
        "(>= 10 5)",
        "(== 7 7)",
        "(!= 3 4)",
        "(&& #t #t)",
        "(&& #t #f)",
        "(|| #f #t)",
        "(|| #f #f)",
    ])
    def test_scalar_ops(self, src):
        _assert_lisp_compiled_matches_interp(src)

    # ── Array arithmetic (rank polymorphism) ─────────────────────────────

    @pytest.mark.parametrize("src", [
        "(+ [1 2 3] [4 5 6])",
        "((lambda (x) (* x 2)) [1 2 3])",
    ])
    def test_rank_polymorphism(self, src):
        _assert_lisp_compiled_matches_interp(src)

    # ── Map and fold ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("src", [
        "(map (lambda (x) (* x 3)) [1 2 3])",
        "(map + [1 2 3] [4 5 6])",
        "(fold + 0 [1 2 3 4 5])",
        "(fold (+) 0.0 (iota 5))",
    ])
    def test_map_fold(self, src):
        _assert_lisp_compiled_matches_interp(src)

    # ── Scan family ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("src", [
        "(iscan + 0 [2 10 5])",
        "(escan + 0 [2 10 5])",
        "(trace + 0 [2 10 5])",
        "(trace-right + 0 [2 10 5])",
        "(fold-right + 0 [1 2 3 4])",
        "(scan + 0 [2 10 5])",
        "(iscan/zero + 0 [2 10 5])",
        "(escan/zero + 0 [2 10 5])",
        "(iscan + [0 0 0] [[1 2 3] [4 5 6] [7 8 9]])",
    ])
    def test_scan_family(self, src):
        _assert_lisp_compiled_matches_interp(src)

    # ── Reduce family ────────────────────────────────────────────────────

    @pytest.mark.parametrize("src", [
        "(reduce + 0 (iota 5))",
        "(reduce/zero + 0 (iota 5))",
        "(reduce/1 + 0 (iota 5))",
    ])
    def test_reduce_family(self, src):
        _assert_lisp_compiled_matches_interp(src)

    # ── Views and primitives ─────────────────────────────────────────────

    @pytest.mark.parametrize("src", [
        "(length [1 2 3 4 5])",
        "(length [[1 2] [3 4] [5 6]])",
        "(rotate [1 2 3 4 5] 2)",
        "(rotate [1 2 3 4 5] 0)",
        "(rotate [[1 2] [3 4] [5 6]] 1)",
        "(subarray [[1 2 3] [4 5 6] [7 8 9]] [1 0] [2 2])",
        "(append [1 2] [3 4 5])",
        "(append [[1 2] [3 4]] [[5 6] [7 8]])",
        "(indices-of [10 20 30])",
        "(indices-of [[1 2] [3 4]])",
        "(with-shape 5 [3 2])",
        "(with-shape 7 [1 4])",
        "(select #t 42 99)",
        "(select #f 42 99)",
    ])
    def test_views_primitives(self, src):
        _assert_lisp_compiled_matches_interp(src)

    # ── Box/unbox ────────────────────────────────────────────────────────

    @pytest.mark.parametrize("src", [
        "(unbox (box [1 2 3]) (len v) v)",
        "(unbox (box [10 20 30]) (len xs) (fold + 0 xs))",
        "(unbox (iota1 5) (len v) v)",
        "(unbox (iota1 6) (len xs) (fold + 0 xs))",
    ])
    def test_box_ops(self, src):
        _assert_lisp_compiled_matches_interp(src)

    # ── Interpreter-only ops (no compiled path yet) ──────────────────────
    # These just verify the interpreter produces expected results

    def test_sort_interpreter(self):
        r = evaluate_source("(sort < [3 1 4 1])", include_prelude=False, syntax="lisp")
        np.testing.assert_array_equal(r.value, [1, 1, 3, 4])

    def test_grade_interpreter(self):
        r = evaluate_source("(grade < [3 1 4 1])", include_prelude=False, syntax="lisp")
        np.testing.assert_array_equal(r.value, [1, 3, 0, 2])

    # ── Filter / Replicate (compiled path) ────────────────────────────────

    @pytest.mark.parametrize("src", [
        "(filter (> 0) [1 -2 3 -4])",
        "(filter (> 0) [5 1 3])",
        "(filter (> 0) [-1 -2])",
        "(replicate [2 1 3] [10 20 30])",
        "(replicate [3] [42])",
        "(replicate [0 1 0] [7 8 9])",
    ])
    def test_filter_replicate_compiled(self, src):
        _assert_lisp_compiled_matches_interp(src)

    # ── Sort / Grade (compiled path) ──────────────────────────────────────

    @pytest.mark.parametrize("src", [
        "(sort < [3 1 4 1])",
        "(sort < [5 3 1 4 2])",
        "(sort < (iota 5))",
        "(grade < [3 1 4 1])",
        "(grade < [5 3 1 4 2])",
        "(grade < (iota 5))",
    ])
    def test_sort_grade_compiled(self, src):
        _assert_lisp_compiled_matches_interp(src)

    def test_filter_interpreter(self):
        r = evaluate_source("(filter (> 0) [1 -2 3 -4])", include_prelude=False, syntax="lisp")
        np.testing.assert_array_equal(r.value, [1, 3])

    def test_replicate_interpreter(self):
        r = evaluate_source("(replicate [2 1 3] [10 20 30])", include_prelude=False, syntax="lisp")
        np.testing.assert_array_equal(r.value, [10, 10, 20, 30, 30, 30])

    def test_filter_all_pass(self):
        r = evaluate_source("(filter (> 0) [1 2 3])", include_prelude=False, syntax="lisp")
        np.testing.assert_array_equal(r.value, [1, 2, 3])

    def test_filter_none_pass(self):
        r = evaluate_source("(filter (> 0) [-1 -2 -3])", include_prelude=False, syntax="lisp")
        np.testing.assert_array_equal(r.value, [])

    def test_replicate_single(self):
        r = evaluate_source("(replicate [3] [42])", include_prelude=False, syntax="lisp")
        np.testing.assert_array_equal(r.value, [42, 42, 42])

    # ── With-shape as sub-expression (compiled path) ──────────────────────

    @pytest.mark.parametrize("src", [
        "(map (+ 1) (with-shape 5 [3]))",
        "(fold + 0 (with-shape 2 [4]))",
        "(fold * 1 (with-shape 3 [3]))",
        "(with-shape (with-shape 1 [2]) [3])",
    ])
    def test_with_shape_subexpr(self, src):
        _assert_lisp_compiled_matches_interp(src)
