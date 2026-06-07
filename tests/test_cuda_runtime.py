import ctypes
import importlib.util

import numpy as np
import pytest

from remora.abi import make_memref_descriptor
from remora.runtime import CUDAKernel, CUDARuntime, RuntimeUnavailable, cuda_available
from conftest import gpu_required_or_skip


class FakeCuda:
    class CUresult:
        CUDA_SUCCESS = 0

    def __init__(self):
        self.launches = []

    def cuLaunchKernel(self, *args):
        self.launches.append(args)
        return self.CUresult.CUDA_SUCCESS


class FakeRuntime:
    def __init__(self):
        self._cuda = FakeCuda()
        self.next_ptr = 0x1000
        self.copied = []
        self.deferred = []

    def alloc(self, nbytes):
        ptr = self.next_ptr
        self.next_ptr += nbytes + 0x100
        return ptr

    def copy_host_bytes_to_device(self, host_address, device_ptr, nbytes):
        self.copied.append((host_address, device_ptr, nbytes))

    def _defer_free(self, ptr):
        self.deferred.append(ptr)


def test_cuda_kernel_launch_packs_descriptor_and_scalar_arguments():
    runtime = FakeRuntime()
    kernel = CUDAKernel(function=object(), runtime=runtime)
    descriptor = make_memref_descriptor(
        device_or_host_ptr=0xCAFE,
        shape=(4,),
        strides=(1,),
        dtype=np.float32,
    )

    kernel.launch((2, 1, 1), (128, 1, 1), [descriptor, 7, 1.5, True])

    assert len(runtime._cuda.launches) == 1
    assert runtime.copied == [
        (ctypes.addressof(descriptor), runtime.deferred[0], ctypes.sizeof(descriptor))
    ]
    assert runtime.deferred
    launch = runtime._cuda.launches[0]
    assert launch[1:7] == (2, 1, 1, 128, 1, 1)


def test_cuda_available_matches_driver_importability():
    assert cuda_available() == (
        importlib.util.find_spec("cuda") is not None
    )


def test_cuda_runtime_initialization_skips_without_live_driver():
    if importlib.util.find_spec("cuda") is None:
        pytest.skip("cuda-python is not installed")
    try:
        runtime = CUDARuntime()
    except RuntimeUnavailable as exc:
        gpu_required_or_skip(str(exc))
    else:
        runtime.close()
