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

    def execute(
        self,
        kernel_name: str,
        inputs: list[np.ndarray],
        *,
        arena: Any | None = None,
    ) -> np.ndarray:
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

            if arena is not None:
                output_ptr = arena.alloc(output.nbytes)
            else:
                output_ptr = self._rt.alloc(output.nbytes)

            if meta.is_reduction:
                # Initialize output to 0 (for atomicAdd)
                # output.nbytes // 4 is the count of 32-bit words
                self._rt.memset_d32(output_ptr, 0, output.nbytes // 4)

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
            if meta.is_reduction and host_inputs:
                # Base grid size on input size for better parallelism
                input_count = int(np.prod(host_inputs[0].shape, dtype=np.int64))
                grid_size = int((input_count + block_size - 1) // block_size)
                # Limit grid size to avoid too many atomic operations if needed?
                # For now, let's just go with it.
            else:
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
            if output_ptr is not None and arena is None:
                self._rt.free(output_ptr)

        return output

    def execute_main(self, inputs: list[np.ndarray] | None = None, *, arena: Any | None = None) -> np.ndarray:
        """Run the program entry kernel using the shared executor-style API."""
        kernel_name = self._main_kernel_name()
        return self.execute(kernel_name, [] if inputs is None else inputs, arena=arena)

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


class GPUPtxContext:
    """Pre-loaded CUDA runtime and PTX module for reuse across launches."""

    def __init__(self, ptx_text: str):
        self._rt = CUDARuntime()
        self._mod = self._rt.load_ptx(ptx_text)
        self._pool: list[int] = []

    def get_kernel(self, name: str):
        return self._mod.get_function(name)

    def get_runtime(self) -> CUDARuntime:
        return self._rt

    def alloc_buffer(self, nbytes: int) -> int:
        """Get a reusable GPU buffer of at least *nbytes*, zeroing it."""
        if self._pool:
            ptr = self._pool.pop()
        else:
            ptr = self._rt.alloc(nbytes)
        self._rt.memset_d32(ptr, 0, nbytes // 4)
        return ptr

    def free_buffer(self, ptr: int) -> None:
        """Return a buffer to the pool for reuse."""
        self._pool.append(ptr)

    def close(self) -> None:
        for ptr in self._pool:
            self._rt.free(ptr)
        self._pool.clear()
        self._mod.close()
        self._rt.close()


def execute_program_on_gpu(
    source: str,
    *,
    include_prelude: bool = True,
) -> np.ndarray:
    """Compile a Remora body program to GPU PTX and execute it."""
    from remora.compiler import compile_source_to_ptx

    artifact = compile_source_to_ptx(source, include_prelude=include_prelude)
    return execute_program_from_ptx(artifact)


def execute_program_from_ptx(
    artifact: Any,
    *,
    context: GPUPtxContext | None = None,
) -> np.ndarray:
    """Execute a pre-compiled PTX artifact on GPU.

    Launches all kernels in order with intermediate buffers.
    If *context* is provided, reuses the pre-loaded CUDA runtime
    and PTX module to avoid recompilation overhead.
    Returns the output as a numpy array.
    """
    if not artifact.kernels:
        raise RemoraExecutorError(
            "No GPU kernels generated. Try a program with tensor operations."
        )
    result_type = artifact.compiler.return_type
    if result_type is None:
        raise RemoraExecutorError("Cannot determine result type for GPU execution")

    from remora.types import ArrayType, ScalarType
    from remora.runtime import _numpy_dtype

    if isinstance(result_type, ScalarType):
        output_shape: tuple[int, ...] = ()
        output_dtype = _numpy_dtype(result_type)
        output_nbytes = np.dtype(output_dtype).itemsize
    elif isinstance(result_type, ArrayType):
        output_shape = tuple(d.value for d in result_type.shape)
        output_dtype = _numpy_dtype(result_type.element)
        output_nbytes = int(np.prod(output_shape, dtype=np.int64)) * np.dtype(output_dtype).itemsize
    else:
        raise RemoraExecutorError(f"unsupported result type: {result_type}")

    output = np.empty(output_shape, dtype=output_dtype)

    if context is not None:
        rt = context.get_runtime()
        mod = context._mod
        use_pool = True
    else:
        rt = CUDARuntime()
        use_pool = False
    try:
        if not use_pool:
            mod = rt.load_ptx(artifact.ptx_text)
        buf_size = max(4096, output_nbytes * 4, 1024 * 1024)
        output_storage = np.zeros(buf_size, dtype=np.uint8)
        if use_pool:
            buf_ptr = context.alloc_buffer(buf_size)
        else:
            buf_ptr = rt.alloc(buf_size)
            rt.memset_d32(buf_ptr, 0, buf_size // 4)
        extra_bufs: list[int] = []

        for kernel_meta in artifact.kernels:
            kernel = mod.get_function(kernel_meta.name)
            num_params = kernel_meta.num_inputs + kernel_meta.num_outputs
            if num_params > 1:
                if use_pool:
                    extra_ptr = context.alloc_buffer(buf_size)
                else:
                    extra_ptr = rt.alloc(buf_size)
                    rt.memset_d32(extra_ptr, 0, buf_size // 4)
                extra_bufs.append(extra_ptr)
                params = [buf_ptr, extra_ptr]
            else:
                params = [buf_ptr]
            block_size = int(kernel_meta.block_size or 256)
            if output_shape:
                element_count = max(1, int(np.prod(output_shape, dtype=np.int64)))
            else:
                # Scalar output: use grid=1 for reduction, or small grid for safety
                element_count = 1
            grid_size = int((element_count + block_size - 1) // block_size)
            kernel.launch((grid_size, 1, 1), (block_size, 1, 1), params)
            rt.synchronize()
            # Swap buffers: output of this kernel becomes input of next
            if extra_bufs:
                if use_pool:
                    context.free_buffer(buf_ptr)
                else:
                    rt.free(buf_ptr)
                buf_ptr = extra_bufs.pop()

        rt.copy_device_to_host(buf_ptr, output_storage)
        result_offset = 0
        if output_shape == () and len(artifact.kernels) > 1:
            # For scalar results from multi-kernel programs, find the last
            # non-zero word in the buffer using numpy vector search.
            itemsize = np.dtype(output_dtype).itemsize
            flat = output_storage.view(output_dtype)
            nonzero = np.nonzero(flat)[0]
            if len(nonzero) > 0:
                result_offset = int(nonzero[-1]) * itemsize
        output_bytes = output_storage[result_offset:result_offset + output_nbytes]
        if output_shape:
            output[:] = output_bytes.view(output_dtype).reshape(output_shape)
        else:
            output[...] = output_bytes.view(output_dtype)[0]
        if use_pool:
            context.free_buffer(buf_ptr)
            for ptr in extra_bufs:
                context.free_buffer(ptr)
        else:
            rt.free(buf_ptr)
            for ptr in extra_bufs:
                rt.free(ptr)
        if context is None:
            mod.close()
    finally:
        if context is None:
            rt.close()

    return output


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
