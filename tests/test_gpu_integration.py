"""GPU integration tests — verify kernel correctness on actual CUDA hardware.

Requires an NVIDIA GPU with CUDA.  Skipped if GPU is unavailable;
fails hard if REMORA_TEST_GPU=1 is set.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import gpu_required_or_skip


def _gpu_runtime():
    from remora.runtime import CUDARuntime
    try:
        return CUDARuntime()
    except Exception as exc:
        gpu_required_or_skip(str(exc))


@pytest.fixture(scope="module")
def rt():
    runtime = _gpu_runtime()
    yield runtime
    runtime.close()


def _compile_hir_function(hfunc, kernel_name):
    from remora.codegen import generate_mlir_descriptor_abi_ptx
    return generate_mlir_descriptor_abi_ptx(hfunc, kernel_name=kernel_name)


class TestGPUMap:

    def test_scale(self, rt):
        from remora.compiler import compile_function_source_to_supported_gpu_artifacts
        from remora.executor import RemoraExecutor
        from remora.types import ArrayType, FLOAT, StaticDim

        source = "def scale xs = map (* 2.0) xs\nscale [1.0, 2.0, 3.0, 4.0, 5.0]"
        param_types = (ArrayType(FLOAT, (StaticDim(5),)),)
        art = compile_function_source_to_supported_gpu_artifacts(
            source, "scale", param_types, syntax="ml", include_prelude=False,
        )
        with RemoraExecutor(art.ptx_text, art.kernels, runtime=rt) as exe:
            result = exe.execute(art.kernels[0].name, [np.array([1, 2, 3, 4, 5], dtype=np.float32)])
        np.testing.assert_allclose(result, [2, 4, 6, 8, 10], atol=1e-5)

    def test_add_arrays(self, rt):
        from remora.compiler import compile_function_source_to_supported_gpu_artifacts
        from remora.executor import RemoraExecutor
        from remora.types import ArrayType, FLOAT, StaticDim

        source = "def add a b = a + b\nadd [1.0, 2.0, 3.0] [10.0, 20.0, 30.0]"
        arr3 = ArrayType(FLOAT, (StaticDim(3),))
        art = compile_function_source_to_supported_gpu_artifacts(
            source, "add", (arr3, arr3), syntax="ml", include_prelude=False,
        )
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([10, 20, 30], dtype=np.float32)
        with RemoraExecutor(art.ptx_text, art.kernels, runtime=rt) as exe:
            result = exe.execute(art.kernels[0].name, [a, b])
        np.testing.assert_allclose(result, [11, 22, 33], atol=1e-5)


class TestGPUSort:

    def test_sort_8_elements(self, rt):
        from remora.hir import HIRFunction, HIRParam, HIRSort, HIRVar
        from remora.types import ArrayType, FLOAT, StaticDim
        from remora.executor import RemoraExecutor

        arr = ArrayType(FLOAT, (StaticDim(8),))
        hf = HIRFunction("s", [HIRParam("xs", arr)], HIRSort(HIRVar("xs", arr), result_type=arr), return_type=arr)
        ptx, kernels, plan = _compile_hir_function(hf, "test_sort")
        xs = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            result = exe.execute("test_sort", [xs])
        np.testing.assert_array_equal(result, np.sort(xs))

    def test_sort_5_elements_non_power_of_2(self, rt):
        from remora.hir import HIRFunction, HIRParam, HIRSort, HIRVar
        from remora.types import ArrayType, FLOAT, StaticDim
        from remora.executor import RemoraExecutor

        arr = ArrayType(FLOAT, (StaticDim(5),))
        hf = HIRFunction("s", [HIRParam("xs", arr)], HIRSort(HIRVar("xs", arr), result_type=arr), return_type=arr)
        ptx, kernels, plan = _compile_hir_function(hf, "test_sort5")
        xs = np.array([5, 2, 8, 1, 3], dtype=np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            result = exe.execute("test_sort5", [xs])
        np.testing.assert_array_equal(result, np.sort(xs))

    def test_sort_with_duplicates(self, rt):
        from remora.hir import HIRFunction, HIRParam, HIRSort, HIRVar
        from remora.types import ArrayType, FLOAT, StaticDim
        from remora.executor import RemoraExecutor

        arr = ArrayType(FLOAT, (StaticDim(8),))
        hf = HIRFunction("s", [HIRParam("xs", arr)], HIRSort(HIRVar("xs", arr), result_type=arr), return_type=arr)
        ptx, kernels, plan = _compile_hir_function(hf, "test_sort_dup")
        xs = np.array([5, 5, 3, 3, 1, 1, 4, 4], dtype=np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            result = exe.execute("test_sort_dup", [xs])
        np.testing.assert_array_equal(result, np.sort(xs))
        from remora.hir import HIRFunction, HIRParam, HIRSort, HIRVar
        from remora.types import ArrayType, FLOAT, StaticDim
        from remora.executor import RemoraExecutor

        arr = ArrayType(FLOAT, (StaticDim(4),))
        hf = HIRFunction("s", [HIRParam("xs", arr)], HIRSort(HIRVar("xs", arr), result_type=arr), return_type=arr)
        ptx, kernels, plan = _compile_hir_function(hf, "test_sort_id")
        xs = np.array([1, 2, 3, 4], dtype=np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            result = exe.execute("test_sort_id", [xs])
        np.testing.assert_array_equal(result, xs)


class TestGPUGrade:

    def test_grade_8_elements(self, rt):
        from remora.hir import HIRFunction, HIRParam, HIRGrade, HIRVar
        from remora.types import ArrayType, FLOAT, INT, StaticDim
        from remora.executor import RemoraExecutor

        arr = ArrayType(FLOAT, (StaticDim(8),))
        iarr = ArrayType(INT, (StaticDim(8),))
        hf = HIRFunction("g", [HIRParam("xs", arr)], HIRGrade(HIRVar("xs", arr), result_type=iarr), return_type=iarr)
        ptx, kernels, plan = _compile_hir_function(hf, "test_grade")
        xs = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            result = exe.execute("test_grade", [xs])
        np.testing.assert_array_equal(xs[result], np.sort(xs))


    def test_grade_non_power_of_2(self, rt):
        from remora.hir import HIRFunction, HIRParam, HIRGrade, HIRVar
        from remora.types import ArrayType, FLOAT, INT, StaticDim
        from remora.executor import RemoraExecutor

        arr = ArrayType(FLOAT, (StaticDim(5),))
        iarr = ArrayType(INT, (StaticDim(5),))
        hf = HIRFunction("g", [HIRParam("xs", arr)], HIRGrade(HIRVar("xs", arr), result_type=iarr), return_type=iarr)
        ptx, kernels, plan = _compile_hir_function(hf, "test_grade5")
        xs = np.array([9, 1, 5, 3, 7], dtype=np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            result = exe.execute("test_grade5", [xs])
        np.testing.assert_array_equal(xs[result], np.sort(xs))


class TestGPUMatmul:

    def test_matmul_2x2(self, rt):
        from remora.hir import HIRFunction, HIRParam, HIRMatmul, HIRVar
        from remora.types import ArrayType, FLOAT, StaticDim
        from remora.executor import RemoraExecutor

        t22 = ArrayType(FLOAT, (StaticDim(2), StaticDim(2)))
        hf = HIRFunction(
            "mm", [HIRParam("a", t22), HIRParam("b", t22)],
            HIRMatmul(HIRVar("a", t22), HIRVar("b", t22), result_type=t22),
            return_type=t22,
        )
        ptx, kernels, plan = _compile_hir_function(hf, "test_mm")
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            result = exe.execute("test_mm", [a, b])
        np.testing.assert_allclose(result, a @ b, atol=1e-4)

    def test_matmul_4x3_3x2(self, rt):
        from remora.hir import HIRFunction, HIRParam, HIRMatmul, HIRVar
        from remora.types import ArrayType, FLOAT, StaticDim
        from remora.executor import RemoraExecutor

        t43 = ArrayType(FLOAT, (StaticDim(4), StaticDim(3)))
        t32 = ArrayType(FLOAT, (StaticDim(3), StaticDim(2)))
        t42 = ArrayType(FLOAT, (StaticDim(4), StaticDim(2)))
        hf = HIRFunction(
            "mm", [HIRParam("a", t43), HIRParam("b", t32)],
            HIRMatmul(HIRVar("a", t43), HIRVar("b", t32), result_type=t42),
            return_type=t42,
        )
        ptx, kernels, plan = _compile_hir_function(hf, "test_mm2")
        rng = np.random.default_rng(42)
        a = rng.standard_normal((4, 3)).astype(np.float32)
        b = rng.standard_normal((3, 2)).astype(np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            result = exe.execute("test_mm2", [a, b])
        np.testing.assert_allclose(result, a @ b, atol=1e-3)


    def test_matmul_16x16(self, rt):
        from remora.hir import HIRFunction, HIRParam, HIRMatmul, HIRVar
        from remora.types import ArrayType, FLOAT, StaticDim
        from remora.executor import RemoraExecutor

        t16 = ArrayType(FLOAT, (StaticDim(16), StaticDim(16)))
        hf = HIRFunction(
            "mm", [HIRParam("a", t16), HIRParam("b", t16)],
            HIRMatmul(HIRVar("a", t16), HIRVar("b", t16), result_type=t16),
            return_type=t16,
        )
        ptx, kernels, plan = _compile_hir_function(hf, "test_mm16")
        rng = np.random.default_rng(123)
        a = rng.standard_normal((16, 16)).astype(np.float32)
        b = rng.standard_normal((16, 16)).astype(np.float32)
        with RemoraExecutor(ptx, kernels, runtime=rt) as exe:
            result = exe.execute("test_mm16", [a, b])
        np.testing.assert_allclose(result, a @ b, atol=1e-2)


class TestGPUReduction:

    def test_fold_sum(self, rt):
        from remora.compiler import compile_function_source_to_supported_gpu_artifacts
        from remora.executor import RemoraExecutor
        from remora.types import ArrayType, FLOAT, StaticDim

        source = "def mysum xs = fold (+) 0.0 xs\nmysum [1.0, 2.0, 3.0, 4.0]"
        param_types = (ArrayType(FLOAT, (StaticDim(4),)),)
        art = compile_function_source_to_supported_gpu_artifacts(
            source, "mysum", param_types, syntax="ml", include_prelude=False,
        )
        xs = np.array([1, 2, 3, 4], dtype=np.float32)
        with RemoraExecutor(art.ptx_text, art.kernels, runtime=rt) as exe:
            result = exe.execute(art.kernels[0].name, [xs])
        np.testing.assert_allclose(float(result), 10.0, atol=1e-5)


class TestGPUStateFold:

    @pytest.mark.xfail(
        reason="AD source transform produces combinatorially large HIRScatterAdd trees "
               "with HIRMap targets; requires AD-level optimization or multi-kernel "
               "gradient compilation to fit in a single GPU kernel",
        strict=False,
    )
    def test_ad_optimize_on_gpu(self, rt):
        from remora.executor import execute_program_on_gpu
        source = open("examples/ad_optimize.lisp").read()
        result = execute_program_on_gpu(source, syntax="lisp", include_prelude=False)
        np.testing.assert_allclose(result, [0.512337, 0.433115, 0.911621], rtol=1e-3)
