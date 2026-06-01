import numpy as np
import pytest

from remora.codegen import KernelMeta
from remora.executor import RemoraExecutor, RemoraExecutorError, compute_output_shape, kernel_output_dtype
from remora.runtime import CUDARuntime, RuntimeUnavailable


RANK1_SCALE2_PTX = r"""
.version 6.0
.target sm_50
.address_size 64

.visible .entry remora_scale2(
    .param .u64 input_desc_param,
    .param .u64 output_desc_param
) {
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [input_desc_param];
    ld.param.u64 %rd2, [output_desc_param];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r4, %r2, %r3, %r1;
    cvt.s64.s32 %rd3, %r4;

    ld.u64 %rd4, [%rd1+24];
    setp.ge.s64 %p, %rd3, %rd4;
    @%p bra DONE;

    ld.u64 %rd5, [%rd1+8];
    ld.u64 %rd6, [%rd1+16];
    ld.u64 %rd7, [%rd1+32];
    mad.lo.s64 %rd8, %rd3, %rd7, %rd6;
    mul.lo.s64 %rd9, %rd8, 4;
    add.s64 %rd10, %rd5, %rd9;
    ld.global.f32 %f1, [%rd10];
    add.f32 %f2, %f1, %f1;

    ld.u64 %rd11, [%rd2+8];
    ld.u64 %rd12, [%rd2+16];
    ld.u64 %rd13, [%rd2+32];
    mad.lo.s64 %rd14, %rd3, %rd13, %rd12;
    mul.lo.s64 %rd15, %rd14, 4;
    add.s64 %rd16, %rd11, %rd15;
    st.global.f32 [%rd16], %f2;

DONE:
    ret;
}
"""


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


def test_remora_executor_runs_rank1_cuda_descriptor_round_trip_when_available():
    try:
        runtime = CUDARuntime()
    except RuntimeUnavailable as exc:
        pytest.skip(f"CUDA driver/device is not available: {exc}")

    meta = KernelMeta(
        name="remora_scale2",
        grid_dims=1,
        block_size=128,
        num_inputs=1,
        num_outputs=1,
        input_elem_types=["f32"],
        output_elem_types=["f32"],
        output_shape=(4,),
    )
    try:
        executor = RemoraExecutor(RANK1_SCALE2_PTX, [meta], runtime=runtime)
        result = executor.execute("remora_scale2", [np.array([1, 2, 3, 4], dtype=np.float32)])
    except RuntimeUnavailable as exc:
        pytest.skip(f"CUDA PTX execution is not available: {exc}")
    finally:
        runtime.close()

    np.testing.assert_array_equal(result, np.array([2, 4, 6, 8], dtype=np.float32))
