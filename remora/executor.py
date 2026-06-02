"""High-level execution helpers for direct Remora ABI kernels."""

from __future__ import annotations

import numpy as np

from remora.abi import element_strides, make_memref_descriptor
from remora.codegen import KernelMeta
from remora.errors import RemoraError
from remora.runtime import CUDARuntime


class RemoraExecutorError(RemoraError):
    """Raised when high-level Remora execution cannot proceed."""


class RemoraExecutor:
    """Execute direct Remora ABI kernels through a CUDA runtime."""

    def __init__(
        self,
        ptx: str,
        kernels: list[KernelMeta],
        runtime: CUDARuntime | None = None,
    ) -> None:
        self._rt = CUDARuntime() if runtime is None else runtime
        self._owns_runtime = runtime is None
        self._module = self._rt.load_ptx(ptx)
        self._meta = {kernel.name: kernel for kernel in kernels}
        self._kernels = {
            kernel.name: self._module.get_function(kernel.name) for kernel in kernels
        }

    def execute(self, kernel_name: str, inputs: list[np.ndarray]) -> np.ndarray:
        """Run one direct Remora ABI kernel and return the host output array."""
        try:
            meta = self._meta[kernel_name]
            kernel = self._kernels[kernel_name]
        except KeyError as exc:
            raise RemoraExecutorError(f"unknown kernel {kernel_name}") from exc

        if meta.num_outputs != 1:
            raise RemoraExecutorError("only single-output kernels are supported")
        if meta.num_inputs != len(inputs):
            raise RemoraExecutorError(
                f"kernel {kernel_name} expects {meta.num_inputs} inputs, got {len(inputs)}"
            )

        host_inputs = [np.asarray(array) for array in inputs]
        output_shape = compute_output_shape(meta, host_inputs)
        output_dtype = kernel_output_dtype(meta, host_inputs)
        output = np.empty(output_shape, dtype=output_dtype)

        device_inputs: list[int] = []
        output_ptr: int | None = None
        try:
            for host_input in host_inputs:
                ptr = self._rt.alloc(host_input.nbytes)
                self._rt.copy_host_to_device(host_input, ptr)
                device_inputs.append(ptr)

            output_ptr = self._rt.alloc(output.nbytes)
            input_descs = [
                make_memref_descriptor(
                    ptr,
                    array.shape,
                    element_strides(array),
                    array.dtype,
                )
                for ptr, array in zip(device_inputs, host_inputs)
            ]
            output_desc = make_memref_descriptor(
                output_ptr,
                output.shape,
                element_strides(output),
                output.dtype,
            )
            block_size = int(meta.block_size or 256)
            element_count = max(1, int(np.prod(output.shape, dtype=np.int64)))
            grid_size = int((element_count + block_size - 1) // block_size)

            kernel.launch(
                (grid_size, 1, 1),
                (block_size, 1, 1),
                [*input_descs, output_desc],
            )
            self._rt.synchronize()
            self._rt.copy_device_to_host(output_ptr, output)
        finally:
            for ptr in device_inputs:
                self._rt.free(ptr)
            if output_ptr is not None:
                self._rt.free(output_ptr)

        return output

    def execute_main(self, inputs: list[np.ndarray] | None = None) -> np.ndarray:
        """Run the program entry kernel using the shared executor-style API."""
        kernel_name = self._main_kernel_name()
        return self.execute(kernel_name, [] if inputs is None else inputs)

    def close(self) -> None:
        self._module.close()
        if self._owns_runtime:
            self._rt.close()

    def __enter__(self) -> "RemoraExecutor":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def _main_kernel_name(self) -> str:
        if "main" in self._kernels:
            return "main"
        if len(self._kernels) == 1:
            return next(iter(self._kernels))
        raise RemoraExecutorError(
            "execute_main requires a kernel named main or exactly one compiled kernel"
        )


def compute_output_shape(meta: KernelMeta, inputs: list[np.ndarray]) -> tuple[int, ...]:
    """Compute a single output shape from kernel metadata and host inputs."""
    if meta.output_shape is not None:
        return tuple(int(dim) for dim in meta.output_shape)
    if inputs:
        return tuple(int(dim) for dim in inputs[0].shape)
    return ()


def kernel_output_dtype(meta: KernelMeta, inputs: list[np.ndarray]) -> np.dtype:
    """Compute a single output dtype from kernel metadata and host inputs."""
    if meta.output_dtype is not None:
        return np.dtype(meta.output_dtype)
    if meta.output_elem_types:
        return _dtype_from_mlir_or_numpy_name(meta.output_elem_types[0])
    if inputs:
        return np.dtype(inputs[0].dtype)
    return np.dtype(np.float32)


def _dtype_from_mlir_or_numpy_name(name: str) -> np.dtype:
    names = {
        "i1": np.bool_,
        "bool": np.bool_,
        "i32": np.int32,
        "int32": np.int32,
        "f32": np.float32,
        "float32": np.float32,
    }
    try:
        return np.dtype(names[name])
    except KeyError:
        return np.dtype(name)
