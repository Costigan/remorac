import numpy as np
import pytest

from remora.codegen import CodegenUnavailable, KernelMeta
from remora.compiler import (
    compile_function_source_to_direct_ptx,
    compile_function_source_to_mlir_gpu_ptx,
    compile_function_source_to_rank1_mlir_gpu_ptx,
)
from remora.executor import RemoraExecutor, RemoraExecutorError, compute_output_shape, kernel_output_dtype
from remora.runtime import CUDARuntime, RuntimeUnavailable
from remora.types import FLOAT, ArrayType, StaticDim


class FakeKernel:
    def __init__(self):
        self.launches = []

    def launch(self, grid, block, args):
        self.launches.append((grid, block, args))


class FakeModule:
    def __init__(self, kernel):
        self.kernel = kernel
        self.closed = False

    def get_function(self, _name):
        return self.kernel

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self):
        self.kernel = FakeKernel()
        self.module = FakeModule(self.kernel)
        self.next_ptr = 0x2000
        self.loaded_ptx = None
        self.allocations = []
        self.frees = []
        self.host_to_device = []
        self.device_to_host = []
        self.synchronized = False
        self.closed = False

    def load_ptx(self, ptx):
        self.loaded_ptx = ptx
        return self.module

    def alloc(self, nbytes):
        ptr = self.next_ptr
        self.next_ptr += nbytes + 0x100
        self.allocations.append((ptr, nbytes))
        return ptr

    def free(self, ptr):
        self.frees.append(ptr)

    def copy_host_to_device(self, array, ptr):
        self.host_to_device.append((array.copy(), ptr))

    def copy_device_to_host(self, ptr, array):
        self.device_to_host.append((ptr, array.shape, array.dtype))
        array.fill(3)

    def synchronize(self):
        self.synchronized = True

    def close(self):
        self.closed = True


def test_compute_output_shape_and_dtype_from_kernel_metadata():
    meta = KernelMeta(
        name="scale",
        grid_dims=1,
        block_size=128,
        num_inputs=1,
        num_outputs=1,
        input_elem_types=["f32"],
        output_elem_types=["f32"],
        output_shape=(2, 3),
    )

    assert compute_output_shape(meta, [np.empty((9,), dtype=np.int32)]) == (2, 3)
    assert kernel_output_dtype(meta, []) == np.dtype(np.float32)


def test_remora_executor_launches_direct_abi_kernel_and_copies_output():
    runtime = FakeRuntime()
    meta = KernelMeta(
        name="scale",
        grid_dims=1,
        block_size=4,
        num_inputs=1,
        num_outputs=1,
        input_elem_types=["f32"],
        output_elem_types=["f32"],
        output_shape=(5,),
    )
    executor = RemoraExecutor("ptx", [meta], runtime=runtime)

    result = executor.execute("scale", [np.arange(5, dtype=np.float32)])

    assert runtime.loaded_ptx == "ptx"
    assert runtime.synchronized is True
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.full((5,), 3, dtype=np.float32))
    assert len(runtime.kernel.launches) == 1
    grid, block, args = runtime.kernel.launches[0]
    assert grid == (2, 1, 1)
    assert block == (4, 1, 1)
    assert len(args) == 2
    assert args[0].size0 == 5
    assert args[1].size0 == 5
    assert {ptr for ptr, _nbytes in runtime.allocations} == set(runtime.frees)


def test_remora_executor_execute_main_uses_shared_entrypoint():
    runtime = FakeRuntime()
    meta = KernelMeta(
        name="main",
        grid_dims=1,
        block_size=4,
        num_inputs=1,
        num_outputs=1,
        input_elem_types=["f32"],
        output_elem_types=["f32"],
        output_shape=(5,),
    )
    executor = RemoraExecutor("ptx", [meta], runtime=runtime)

    result = executor.execute_main([np.arange(5, dtype=np.float32)])

    assert runtime.loaded_ptx == "ptx"
    np.testing.assert_array_equal(result, np.full((5,), 3, dtype=np.float32))
    assert len(runtime.kernel.launches) == 1


def test_remora_executor_rejects_unknown_kernel_and_wrong_input_count():
    runtime = FakeRuntime()
    meta = KernelMeta(
        name="scale",
        grid_dims=1,
        block_size=4,
        num_inputs=1,
        num_outputs=1,
        input_elem_types=["f32"],
        output_elem_types=["f32"],
    )
    executor = RemoraExecutor("ptx", [meta], runtime=runtime)

    with pytest.raises(RemoraExecutorError, match="unknown kernel"):
        executor.execute("missing", [])
    with pytest.raises(RemoraExecutorError, match="expects 1 inputs"):
        executor.execute("scale", [])


def test_compile_function_source_to_direct_rank1_map_ptx():
    ptx, kernels, artifact = compile_function_source_to_direct_ptx(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        kernel_name="remora_scale",
    )

    assert artifact.function_name == "scale"
    assert ".visible .entry remora_scale" in ptx
    assert "mul.rn.f32" in ptx
    assert "ld.u64 %rd4, [%rd1+24]" in ptx
    assert "ld.u64 %rd7, [%rd1+32]" in ptx
    assert kernels == [
        KernelMeta(
            name="remora_scale",
            grid_dims=1,
            block_size=128,
            num_inputs=1,
            num_outputs=1,
            input_elem_types=["f32"],
            output_elem_types=["f32"],
            output_shape=(4,),
            output_dtype="float32",
        )
    ]


def test_compile_function_source_to_direct_rank2_and_rank3_map_ptx():
    rank2_ptx, rank2_kernels, _rank2_artifact = compile_function_source_to_direct_ptx(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),),
        kernel_name="remora_scale2d",
    )
    rank3_ptx, rank3_kernels, _rank3_artifact = compile_function_source_to_direct_ptx(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(2), StaticDim(3), StaticDim(4))),),
        kernel_name="remora_scale3d",
    )

    assert ".visible .entry remora_scale2d" in rank2_ptx
    assert "div.u64 %rd24, %rd3, %rd23;" in rank2_ptx
    assert "rem.u64 %rd25, %rd3, %rd23;" in rank2_ptx
    assert "mad.lo.s64 %rd20, %rd25, %rd26, %rd20;" in rank2_ptx
    assert rank2_kernels[0].output_shape == (2, 3)

    assert ".visible .entry remora_scale3d" in rank3_ptx
    assert "mul.lo.s64 %rd25, %rd23, %rd24;" in rank3_ptx
    assert "div.u64 %rd26, %rd3, %rd25;" in rank3_ptx
    assert "rem.u64 %rd29, %rd27, %rd24;" in rank3_ptx
    assert "mad.lo.s64 %rd20, %rd29, %rd31, %rd20;" in rank3_ptx
    assert rank3_kernels[0].output_shape == (2, 3, 4)


def test_compile_function_source_to_direct_binary_rank1_map_ptx():
    ptx, kernels, artifact = compile_function_source_to_direct_ptx(
        "def add xs ys = map (+) xs ys",
        "add",
        (
            ArrayType(FLOAT, (StaticDim(4),)),
            ArrayType(FLOAT, (StaticDim(4),)),
        ),
        kernel_name="remora_add",
    )

    assert artifact.function_name == "add"
    assert ".visible .entry remora_add" in ptx
    assert ".param .u64 input0_desc_param" in ptx
    assert ".param .u64 input1_desc_param" in ptx
    assert ".param .u64 output_desc_param" in ptx
    assert "add.rn.f32 %f3, %f1, %f2;" in ptx
    assert kernels == [
        KernelMeta(
            name="remora_add",
            grid_dims=1,
            block_size=128,
            num_inputs=2,
            num_outputs=1,
            input_elem_types=["f32"],
            output_elem_types=["f32"],
            output_shape=(4,),
            output_dtype="float32",
        )
    ]


def test_compile_function_source_to_direct_binary_rank2_and_rank3_map_ptx():
    rank2_ptx, rank2_kernels, _rank2_artifact = compile_function_source_to_direct_ptx(
        "def add xs ys = map (+) xs ys",
        "add",
        (
            ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
            ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
        ),
        kernel_name="remora_add2d",
    )
    rank3_ptx, rank3_kernels, _rank3_artifact = compile_function_source_to_direct_ptx(
        "def add xs ys = map (+) xs ys",
        "add",
        (
            ArrayType(FLOAT, (StaticDim(2), StaticDim(3), StaticDim(4))),
            ArrayType(FLOAT, (StaticDim(2), StaticDim(3), StaticDim(4))),
        ),
        kernel_name="remora_add3d",
    )

    assert "mad.lo.s64 %rd21, %rd26, %rd28, %rd21;" in rank2_ptx
    assert "mad.lo.s64 %rd22, %rd26, %rd36, %rd22;" in rank2_ptx
    assert rank2_kernels[0].num_inputs == 2
    assert rank2_kernels[0].output_shape == (2, 3)

    assert "div.u64 %rd27, %rd3, %rd26;" in rank3_ptx
    assert "mad.lo.s64 %rd21, %rd40, %rd44, %rd21;" in rank3_ptx
    assert "mad.lo.s64 %rd22, %rd40, %rd37, %rd22;" in rank3_ptx
    assert rank3_kernels[0].num_inputs == 2
    assert rank3_kernels[0].output_shape == (2, 3, 4)


def test_direct_ptx_rank4_maps_are_deferred_to_mlir_gpu_lowering():
    with pytest.raises(CodegenUnavailable, match="rank-1 through rank-3"):
        compile_function_source_to_direct_ptx(
            "def scale xs = map (* 2.0) xs",
            "scale",
            (ArrayType(FLOAT, (StaticDim(1), StaticDim(1), StaticDim(1), StaticDim(1))),),
            kernel_name="remora_scale4d",
        )


def test_compile_function_source_to_rank1_mlir_gpu_ptx():
    ptx, kernels, artifact = compile_function_source_to_rank1_mlir_gpu_ptx(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        kernel_name="remora_scale",
    )

    assert artifact.function_name == "scale"
    assert ".visible .entry remora_scale" in ptx
    assert ".param .u64 remora_scale_param_0" in ptx
    assert ".param .u64 remora_scale_param_1" in ptx
    assert "remora_scale_inner" in ptx
    assert kernels == [
        KernelMeta(
            name="remora_scale",
            grid_dims=1,
            block_size=0,
            num_inputs=1,
            num_outputs=1,
            input_elem_types=["f32"],
            output_elem_types=["f32"],
            output_shape=(4,),
            output_dtype="float32",
        )
    ]


def test_compile_function_source_to_rank1_binary_mlir_gpu_ptx():
    ptx, kernels, artifact = compile_function_source_to_rank1_mlir_gpu_ptx(
        "def add xs ys = map (+) xs ys",
        "add",
        (
            ArrayType(FLOAT, (StaticDim(4),)),
            ArrayType(FLOAT, (StaticDim(4),)),
        ),
        kernel_name="remora_add",
    )

    assert artifact.function_name == "add"
    assert ".visible .entry remora_add" in ptx
    assert ".param .u64 remora_add_param_0" in ptx
    assert ".param .u64 remora_add_param_1" in ptx
    assert ".param .u64 remora_add_param_2" in ptx
    assert "remora_add_inner" in ptx
    assert kernels == [
        KernelMeta(
            name="remora_add",
            grid_dims=1,
            block_size=0,
            num_inputs=2,
            num_outputs=1,
            input_elem_types=["f32"],
            output_elem_types=["f32"],
            output_shape=(4,),
            output_dtype="float32",
        )
    ]


def test_compile_function_source_to_rank2_unary_mlir_gpu_ptx():
    ptx, kernels, artifact = compile_function_source_to_mlir_gpu_ptx(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),),
        kernel_name="remora_scale2d",
    )

    assert artifact.function_name == "scale"
    assert ".visible .entry remora_scale2d" in ptx
    assert ".param .u64 remora_scale2d_param_0" in ptx
    assert ".param .u64 remora_scale2d_param_1" in ptx
    assert "remora_scale2d_inner" in ptx
    assert kernels == [
        KernelMeta(
            name="remora_scale2d",
            grid_dims=1,
            block_size=0,
            num_inputs=1,
            num_outputs=1,
            input_elem_types=["f32"],
            output_elem_types=["f32"],
            output_shape=(2, 3),
            output_dtype="float32",
        )
    ]


def test_remora_executor_runs_rank1_cuda_descriptor_round_trip_when_available():
    try:
        runtime = CUDARuntime()
    except RuntimeUnavailable as exc:
        pytest.skip(f"CUDA driver/device is not available: {exc}")

    ptx, kernels, _artifact = compile_function_source_to_direct_ptx(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        kernel_name="remora_scale",
    )
    try:
        executor = RemoraExecutor(ptx, kernels, runtime=runtime)
        result = executor.execute("remora_scale", [np.array([1, 2, 3, 4], dtype=np.float32)])
    except RuntimeUnavailable as exc:
        pytest.skip(f"CUDA PTX execution is not available: {exc}")
    finally:
        runtime.close()

    np.testing.assert_array_equal(result, np.array([2, 4, 6, 8], dtype=np.float32))


def test_remora_executor_runs_rank1_mlir_gpu_ptx_round_trip_when_available():
    try:
        runtime = CUDARuntime()
    except RuntimeUnavailable as exc:
        pytest.skip(f"CUDA driver/device is not available: {exc}")

    try:
        ptx, kernels, _artifact = compile_function_source_to_rank1_mlir_gpu_ptx(
            "def scale xs = map (* 2.0) xs",
            "scale",
            (ArrayType(FLOAT, (StaticDim(4),)),),
            kernel_name="remora_scale",
        )
        executor = RemoraExecutor(ptx, kernels, runtime=runtime)
        result = executor.execute_main([np.array([1, 2, 3, 4], dtype=np.float32)])
    except RuntimeUnavailable as exc:
        pytest.skip(f"CUDA PTX execution is not available: {exc}")
    finally:
        runtime.close()

    np.testing.assert_array_equal(result, np.array([2, 4, 6, 8], dtype=np.float32))


def test_remora_executor_runs_rank1_binary_mlir_gpu_ptx_round_trip_when_available():
    try:
        runtime = CUDARuntime()
    except RuntimeUnavailable as exc:
        pytest.skip(f"CUDA driver/device is not available: {exc}")

    try:
        ptx, kernels, _artifact = compile_function_source_to_rank1_mlir_gpu_ptx(
            "def add xs ys = map (+) xs ys",
            "add",
            (
                ArrayType(FLOAT, (StaticDim(4),)),
                ArrayType(FLOAT, (StaticDim(4),)),
            ),
            kernel_name="remora_add",
        )
        executor = RemoraExecutor(ptx, kernels, runtime=runtime)
        result = executor.execute_main(
            [
                np.array([1, 2, 3, 4], dtype=np.float32),
                np.array([10, 20, 30, 40], dtype=np.float32),
            ]
        )
    except RuntimeUnavailable as exc:
        pytest.skip(f"CUDA PTX execution is not available: {exc}")
    finally:
        runtime.close()

    np.testing.assert_array_equal(result, np.array([11, 22, 33, 44], dtype=np.float32))


def test_remora_executor_runs_rank2_unary_mlir_gpu_ptx_round_trip_when_available():
    try:
        runtime = CUDARuntime()
    except RuntimeUnavailable as exc:
        pytest.skip(f"CUDA driver/device is not available: {exc}")

    try:
        ptx, kernels, _artifact = compile_function_source_to_mlir_gpu_ptx(
            "def scale xs = map (* 2.0) xs",
            "scale",
            (ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),),
            kernel_name="remora_scale2d",
        )
        executor = RemoraExecutor(ptx, kernels, runtime=runtime)
        result = executor.execute_main([np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)])
    except RuntimeUnavailable as exc:
        pytest.skip(f"CUDA PTX execution is not available: {exc}")
    finally:
        runtime.close()

    np.testing.assert_array_equal(
        result,
        np.array([[2, 4, 6], [8, 10, 12]], dtype=np.float32),
    )


def test_remora_executor_runs_rank2_and_rank3_cuda_descriptor_round_trip_when_available():
    try:
        runtime = CUDARuntime()
    except RuntimeUnavailable as exc:
        pytest.skip(f"CUDA driver/device is not available: {exc}")

    rank2_ptx, rank2_kernels, _rank2_artifact = compile_function_source_to_direct_ptx(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),),
        kernel_name="remora_scale2d",
    )
    rank3_ptx, rank3_kernels, _rank3_artifact = compile_function_source_to_direct_ptx(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(2), StaticDim(2), StaticDim(1))),),
        kernel_name="remora_scale3d",
    )
    try:
        rank2_executor = RemoraExecutor(rank2_ptx, rank2_kernels, runtime=runtime)
        rank3_executor = RemoraExecutor(rank3_ptx, rank3_kernels, runtime=runtime)
        rank2 = rank2_executor.execute_main(
            [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)]
        )
        rank3 = rank3_executor.execute_main(
            [np.array([[[1], [2]], [[3], [4]]], dtype=np.float32)]
        )
    except RuntimeUnavailable as exc:
        pytest.skip(f"CUDA PTX execution is not available: {exc}")
    finally:
        runtime.close()

    np.testing.assert_array_equal(
        rank2,
        np.array([[2, 4, 6], [8, 10, 12]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        rank3,
        np.array([[[2], [4]], [[6], [8]]], dtype=np.float32),
    )
