"""Tests for the Python integration API (remora.api and remora.codec)."""

import numpy as np
import pytest

from remora.api import RemoraFunction, RemoraRankMismatchError, compile_all, compile_function, define
from remora.codec import _transform_source
from remora.types import FLOAT, ArrayType, StaticDim


class TestRemoraFunction:

    def test_compile_and_call_unary(self):
        fn = compile_function(
            "(define/pi () (scale [xs (Array Float 4)] (Array Float 4)) (map (* 2.0) xs))",
            "scale", syntax="lisp", include_prelude=False,
        )
        result = fn(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0, 8.0])

    def test_compile_and_call_binary(self):
        fn = compile_function(
            "(define/pi () (add [xs (Array Float 3) ys (Array Float 3)] (Array Float 3)) (+ xs ys))",
            "add", syntax="lisp", include_prelude=False,
        )
        result = fn(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([10.0, 20.0, 30.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(result, [11.0, 22.0, 33.0])

    def test_compile_scalar_return(self):
        fn = compile_function(
            "(define/pi () (mysum [xs (Array Float 4)] Float) (fold + 0.0 xs))",
            "mysum", syntax="lisp", include_prelude=False,
        )
        result = fn(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert result == 10.0

    def test_auto_type_inference(self):
        fn = compile_function(
            "(define/pi () (neg [xs (Array Float 3)] (Array Float 3)) (map (* -1.0) xs))",
            "neg", syntax="lisp", include_prelude=False,
        )
        assert fn.name == "neg"
        assert len(fn.param_types) == 1
        result = fn(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(result, [-1.0, -2.0, -3.0])

    def test_repr(self):
        fn = compile_function(
            "(define/pi () (f [x (Array Float 3)] Float) (fold + 0.0 x))",
            "f", syntax="lisp", include_prelude=False,
        )
        assert "f" in repr(fn)
        assert "float[3]" in repr(fn)

    def test_auto_dtype_cast(self):
        fn = compile_function(
            "(define/pi () (id [xs (Array Float 3)] (Array Float 3)) xs)",
            "id", syntax="lisp", include_prelude=False,
        )
        result = fn(np.array([1, 2, 3], dtype=np.float64))
        assert result.dtype == np.float32


class TestJITBoundaryChecking:

    def test_wrong_rank_raises(self):
        fn = compile_function(
            "(define/pi () (f [xs (Array Float 4)] (Array Float 4)) (map (* 2.0) xs))",
            "f", syntax="lisp", include_prelude=False,
        )
        with pytest.raises(RemoraRankMismatchError, match="rank-1.*rank-2"):
            fn(np.array([[1, 2], [3, 4]], dtype=np.float32))

    def test_wrong_shape_raises(self):
        fn = compile_function(
            "(define/pi () (f [xs (Array Float 4)] (Array Float 4)) (map (* 2.0) xs))",
            "f", syntax="lisp", include_prelude=False,
        )
        with pytest.raises(RemoraRankMismatchError, match="size 4.*got 3"):
            fn(np.array([1, 2, 3], dtype=np.float32))

    def test_wrong_arg_count_raises(self):
        fn = compile_function(
            "(define/pi () (f [xs (Array Float 4)] (Array Float 4)) (map (* 2.0) xs))",
            "f", syntax="lisp", include_prelude=False,
        )
        with pytest.raises(TypeError, match="1 argument.*got 2"):
            fn(np.array([1, 2, 3, 4], dtype=np.float32), np.array([5, 6, 7, 8], dtype=np.float32))


class TestCompileAll:

    def test_compile_multiple_functions(self):
        fns = compile_all("""
(define/pi () (double [x (Array Float 3)] (Array Float 3)) (map (* 2.0) x))
(define/pi () (negate [x (Array Float 3)] (Array Float 3)) (map (* -1.0) x))
""", syntax="lisp", include_prelude=False)
        assert "double" in fns
        assert "negate" in fns
        np.testing.assert_array_equal(
            fns["double"](np.array([1, 2, 3], dtype=np.float32)),
            [2, 4, 6],
        )
        np.testing.assert_array_equal(
            fns["negate"](np.array([1, 2, 3], dtype=np.float32)),
            [-1, -2, -3],
        )


class TestCodecTransform:

    def test_transform_replaces_coding_line(self):
        source = "# coding: remora\nx = 1\n"
        result = _transform_source(source)
        assert "# coding: utf-8" in result
        assert "# coding: remora" not in result

    def test_transform_replaces_remora_block(self):
        source = """# coding: remora
# remora:begin
(define/pi () (f [x (Array Float 3)] (Array Float 3)) (map (* 2.0) x))
# remora:end
"""
        result = _transform_source(source)
        assert "remora_api" in result
        assert "compile_all" in result
        assert "# remora:begin" not in result

    def test_transform_preserves_python_code(self):
        source = """# coding: remora
import numpy as np
# remora:begin
(define/pi () (f [x (Array Float 3)] (Array Float 3)) x)
# remora:end
y = 42
"""
        result = _transform_source(source)
        assert "import numpy as np" in result
        assert "y = 42" in result

    def test_transform_end_to_end_execution(self):
        source = """# coding: remora
import numpy as np
# remora:begin
(define/pi () (scale [xs (Array Float 4)] (Array Float 4)) (map (* 2.0) xs))
# remora:end
__test_result__ = scale(np.array([1.0, 2.0, 3.0, 4.0]))
"""
        ns: dict = {}
        exec(compile(_transform_source(source), "<test>", "exec"), ns)
        np.testing.assert_array_equal(ns["__test_result__"], [2.0, 4.0, 6.0, 8.0])

    def test_unmatched_begin_raises(self):
        source = "# coding: remora\n# remora:begin\n(+ 1 2)\n"
        with pytest.raises(SyntaxError, match="without matching"):
            _transform_source(source)


class TestDefine:

    def test_define_single_function(self):
        fn = define(
            "(define/pi () (f [x (Array Float 3)] (Array Float 3)) (map (* 2.0) x))",
            syntax="lisp", include_prelude=False,
        )
        assert isinstance(fn, RemoraFunction)
        result = fn(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])

    def test_define_multiple_functions(self):
        result = define(
            "(define/pi () (a [x (Array Float 3)] (Array Float 3)) (map (* 2.0) x))\n"
            "(define/pi () (b [x (Array Float 3)] (Array Float 3)) (map (* -1.0) x))",
            syntax="lisp", include_prelude=False,
        )
        assert isinstance(result, dict)
        assert "a" in result
        assert "b" in result
        np.testing.assert_array_equal(
            result["a"](np.array([1, 2, 3], dtype=np.float32)), [2, 4, 6],
        )

    def test_define_no_functions_raises(self):
        with pytest.raises(ValueError, match="No function definitions"):
            define("(+ 1 2)", syntax="lisp", include_prelude=False)

    def test_define_accessible_from_package(self):
        import remora
        assert hasattr(remora, "define")
        assert remora.define is define
