"""High-level execution helpers for direct Remora ABI kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from remora.abi import element_strides, make_memref_descriptor
from remora.codegen import KernelMeta
from remora.errors import RemoraError
from remora.execution_plan import ExecutionPlan, KernelStep, LoopPlan, PlanStep
from remora.runtime import CUDARuntime, DeviceMemoryPool


class RemoraExecutorError(RemoraError):
    """Raised when high-level Remora execution cannot proceed."""


@dataclass
class _DeviceBuffer:
    """Tracks a device allocation for execute_plan."""
    ptr: int
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass
class DeviceArray:
    """A device-resident array bound to a RemoraExecutor.

    Holds a device pointer plus shape/dtype/nbytes so kernels can be
    chained on the GPU without round-tripping through host memory.
    Allocations come from the executor's memory pool.
    """
    executor: "RemoraExecutor"
    ptr: int
    shape: tuple[int, ...]
    dtype: np.dtype
    nbytes: int
    _freed: bool = False

    def to_numpy(self) -> np.ndarray:
        """Copy this device array back to host (D->H)."""
        return self.executor.download(self.ptr, self.shape, self.dtype)

    @classmethod
    def from_numpy(cls, executor: "RemoraExecutor", array: np.ndarray) -> "DeviceArray":
        """Upload a host array to the device (H->D)."""
        arr = np.ascontiguousarray(array)
        ptr = executor.alloc_and_upload(arr)
        return cls(executor, ptr, tuple(arr.shape), arr.dtype, int(arr.nbytes))

    def free(self) -> None:
        """Return this array's device buffer to the pool."""
        if not self._freed:
            self.executor.free_device(self.ptr, self.nbytes)
            self._freed = True


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute C-contiguous element strides for a given shape."""
    if not shape:
        return ()
    strides: list[int] = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


_PLAN_DTYPES = {
    "f32": np.float32,
    "float32": np.float32,
    "i32": np.int32,
    "int32": np.int32,
    "i1": np.bool_,
    "bool": np.bool_,
}


def _plan_dtype(name: str) -> np.dtype:
    try:
        return np.dtype(_PLAN_DTYPES[name])
    except KeyError:
        return np.dtype(name)


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
        self._modules = [self._rt.load_ptx(ptx)]
        self._meta: dict[str, KernelMeta] = {
            kernel.name: kernel for kernel in kernels
        }
        self._kernels = {
            kernel.name: self._modules[0].get_function(kernel.name)
            for kernel in kernels
        }
        # Prefer the runtime's shared buffer pool so allocations are recycled
        # across execute() calls and across executors that share the runtime.
        # Runtimes without one (e.g. lightweight test fakes) get a local pool
        # that this executor owns and drains on close().
        shared_pool = getattr(self._rt, "memory_pool", None)
        if shared_pool is not None:
            self._pool = shared_pool
            self._owns_pool = False
        else:
            self._pool = DeviceMemoryPool(self._rt)
            self._owns_pool = True

    def set_pool_enabled(self, enabled: bool) -> None:
        """Enable or disable the device memory pool.

        Delegates to the runtime's shared :class:`DeviceMemoryPool`.  When
        disabled, allocations go straight to the driver and any pooled buffers
        are drained.
        """
        self._pool.set_enabled(enabled)

    def _pool_alloc(self, nbytes: int) -> int:
        """Allocate device memory, reusing a pooled buffer of the same size class."""
        return self._pool.alloc(max(4, nbytes))

    def _pool_free(self, ptr: int, nbytes: int) -> None:
        """Return device memory to the pool for reuse."""
        self._pool.free(ptr, max(4, nbytes))

    def add_module(self, ptx: str, kernels: list[KernelMeta]) -> None:
        """Load additional kernels from a separate PTX module."""
        module = self._rt.load_ptx(ptx)
        self._modules.append(module)
        for kernel in kernels:
            self._meta[kernel.name] = kernel
            self._kernels[kernel.name] = module.get_function(kernel.name)

    def execute(
        self,
        kernel_name: str,
        inputs: list[Any],
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

        input_kinds = meta.input_kinds or ["array"] * meta.num_inputs
        if len(input_kinds) != len(inputs):
            raise RemoraExecutorError(
                f"kernel {kernel_name} metadata has {len(input_kinds)} input kinds for {len(inputs)} inputs"
            )
        host_inputs: list[np.ndarray] = []
        scalar_inputs: list[Any] = []
        for kind, value in zip(input_kinds, inputs):
            if kind == "array":
                host_inputs.append(np.asarray(value))
            elif kind == "scalar":
                scalar_inputs.append(value)
            else:
                raise RemoraExecutorError(f"kernel {kernel_name} has unsupported input kind {kind!r}")

        output_shape = compute_output_shape(meta, host_inputs)
        output_dtype = kernel_output_dtype(meta, host_inputs)
        output = np.empty(output_shape, dtype=output_dtype)

        device_inputs: list[int] = []
        device_input_sizes: list[int] = []
        output_ptr: int | None = None
        output_nbytes = output.nbytes
        try:
            for host_input in host_inputs:
                ptr = self._pool_alloc(host_input.nbytes)
                self._rt.copy_host_to_device(host_input, ptr)
                device_inputs.append(ptr)
                device_input_sizes.append(host_input.nbytes)

            if arena is not None:
                output_ptr = arena.alloc(output_nbytes)
            else:
                output_ptr = self._pool_alloc(output_nbytes)

            if meta.is_reduction:
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
            if meta.grid_size is not None:
                grid_size = meta.grid_size
            elif meta.is_reduction and host_inputs:
                input_count = int(np.prod(host_inputs[0].shape, dtype=np.int64))
                grid_size = int((input_count + block_size - 1) // block_size)
            else:
                element_count = max(1, int(np.prod(output.shape, dtype=np.int64)))
                grid_size = int((element_count + block_size - 1) // block_size)

            kernel.launch(
                (grid_size, 1, 1),
                (block_size, 1, 1),
                [*input_descs, *scalar_inputs, output_desc],
            )
            self._rt.synchronize()
            self._rt.copy_device_to_host(output_ptr, output)
        finally:
            for ptr, nbytes in zip(device_inputs, device_input_sizes):
                self._pool_free(ptr, nbytes)
            if output_ptr is not None and arena is None:
                self._pool_free(output_ptr, output_nbytes)

        return output

    def alloc_and_upload(self, array: np.ndarray) -> int:
        """Allocate device memory and upload an array; return the device pointer."""
        arr = np.ascontiguousarray(array)
        ptr = self._pool_alloc(arr.nbytes)
        self._rt.copy_host_to_device(arr, ptr)
        return ptr

    def download(self, device_ptr: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Copy device memory to a host numpy array."""
        out = np.empty(shape, dtype=dtype)
        self._rt.synchronize()
        self._rt.copy_device_to_host(device_ptr, out)
        return out

    def free_device(self, ptr: int, nbytes: int) -> None:
        """Release a device allocation managed outside execute()."""
        self._pool_free(ptr, nbytes)

    def execute_device(
        self,
        kernel_name: str,
        device_inputs: list[int],
        input_templates: list[np.ndarray],
        *,
        output_ptr: int | None = None,
    ) -> int:
        """Launch a kernel on device-resident data; return the output device ptr.

        *device_inputs* are raw device pointers previously uploaded via
        ``alloc_and_upload``.  *input_templates* are the corresponding
        host arrays (used only for shape/dtype/strides, NOT uploaded).

        Returns the output device pointer (allocated internally if
        *output_ptr* is None, otherwise reused).  Does not free inputs
        or the output — the caller owns all allocations.
        """
        meta = self._meta[kernel_name]
        kernel = self._kernels[kernel_name]
        if meta.num_outputs != 1:
            raise RemoraExecutorError("execute_device supports single-output kernels only")
        if len(device_inputs) != len(input_templates):
            raise RemoraExecutorError(
                f"mismatched device_inputs ({len(device_inputs)}) "
                f"and templates ({len(input_templates)})"
            )

        input_kinds = meta.input_kinds or ["array"] * meta.num_inputs
        if len(input_kinds) != len(device_inputs):
            raise RemoraExecutorError(
                f"kernel {kernel_name} reports {len(input_kinds)} input kinds "
                f"for {len(device_inputs)} inputs"
            )
        host_inputs = [np.asarray(t) for t in input_templates]
        scalar_values: list[Any] = []
        array_indices: list[int] = []
        for idx, kind in enumerate(input_kinds):
            if kind == "array":
                array_indices.append(idx)
            elif kind == "scalar":
                scalar_values.append(host_inputs[idx].item())
            else:
                raise RemoraExecutorError(f"unsupported input kind {kind!r}")
        array_hosts = [host_inputs[i] for i in array_indices]
        array_devices = [device_inputs[i] for i in array_indices]

        output_shape = compute_output_shape(meta, array_hosts)
        output_dtype = kernel_output_dtype(meta, array_hosts)
        out_nbytes = max(1, int(np.prod(output_shape, dtype=np.int64))) * output_dtype.itemsize

        own_output = output_ptr is None
        if own_output:
            output_ptr = self._pool_alloc(out_nbytes)
        if meta.is_reduction:
            self._rt.memset_d32(output_ptr, 0, max(1, out_nbytes // 4))

        input_descs = [
            make_memref_descriptor(ptr, arr.shape, element_strides(arr), arr.dtype)
            for ptr, arr in zip(array_devices, array_hosts)
        ]
        tmp_out = np.empty(output_shape, dtype=output_dtype)
        output_desc = make_memref_descriptor(
            output_ptr, output_shape, element_strides(tmp_out), output_dtype,
        )

        block_size = int(meta.block_size or 256)
        if meta.grid_size is not None:
            grid_size = meta.grid_size
        elif meta.is_reduction and host_inputs:
            input_count = int(np.prod(array_hosts[0].shape, dtype=np.int64))
            grid_size = int((input_count + block_size - 1) // block_size)
        else:
            element_count = max(1, int(np.prod(output_shape, dtype=np.int64)))
            grid_size = int((element_count + block_size - 1) // block_size)

        kernel.launch(
            (grid_size, 1, 1),
            (block_size, 1, 1),
            [*input_descs, *scalar_values, output_desc],
        )
        self._rt.synchronize()
        return output_ptr

    def execute_to_device(
        self,
        kernel_name: str,
        device_inputs: list[DeviceArray],
    ) -> DeviceArray:
        """Launch a kernel on device-resident ``DeviceArray`` inputs and
        return the result as a new device-resident ``DeviceArray`` (no
        host-device transfer)."""
        meta = self._meta[kernel_name]
        templates = [np.empty(da.shape, da.dtype) for da in device_inputs]
        input_kinds = meta.input_kinds or ["array"] * meta.num_inputs
        array_hosts = [t for t, k in zip(templates, input_kinds) if k == "array"]
        output_shape = compute_output_shape(meta, array_hosts)
        output_dtype = kernel_output_dtype(meta, array_hosts)
        out_nbytes = max(1, int(np.prod(output_shape, dtype=np.int64))) * output_dtype.itemsize
        out_ptr = self._pool_alloc(out_nbytes)
        self.execute_device(
            kernel_name,
            [da.ptr for da in device_inputs],
            templates,
            output_ptr=out_ptr,
        )
        return DeviceArray(self, out_ptr, tuple(output_shape), output_dtype, out_nbytes)

    def execute_plan(
        self,
        plan: ExecutionPlan,
        inputs: list[np.ndarray],
    ) -> np.ndarray:
        """Execute a multi-kernel plan and return the host output array.

        The plan describes a sequence of kernel launches with named
        intermediate buffers.  Input arrays are registered as
        ``input_0``, ``input_1``, etc.  Temporary buffers declared in
        ``plan.buffers`` are allocated on the device before execution
        and freed afterward.  ``LoopPlan`` steps run their body
        repeatedly with optional buffer swapping between iterations.
        """
        plan.validate()
        missing_kernels = plan.kernel_names() - set(self._kernels)
        if missing_kernels:
            raise RemoraExecutorError(
                f"Plan references unknown kernels: {sorted(missing_kernels)}"
            )

        registry: dict[str, _DeviceBuffer] = {}
        allocated_bufs: list[tuple[int, int]] = []

        try:
            for i, inp in enumerate(inputs):
                arr = np.ascontiguousarray(inp)
                ptr = self._pool_alloc(arr.nbytes)
                self._rt.copy_host_to_device(arr, ptr)
                registry[f"input_{i}"] = _DeviceBuffer(ptr, arr.shape, arr.dtype)
                allocated_bufs.append((ptr, arr.nbytes))

            for buf_spec in plan.buffers:
                dtype = _plan_dtype(buf_spec.dtype)
                n_elements = max(1, int(np.prod(buf_spec.shape, dtype=np.int64)))
                nbytes = n_elements * dtype.itemsize
                ptr = self._pool_alloc(nbytes)
                if buf_spec.init is not None:
                    init_array = np.array(buf_spec.init, dtype=dtype)
                    self._rt.copy_host_to_device(init_array, ptr)
                else:
                    self._rt.memset_d32(ptr, 0, max(1, nbytes // 4))
                registry[buf_spec.name] = _DeviceBuffer(
                    ptr, buf_spec.shape, dtype,
                )
                allocated_bufs.append((ptr, nbytes))

            for step in plan.steps:
                self._run_plan_step(step, registry)

            final = registry[plan.final_output]
            out_shape = plan.output_shape
            out_dtype = _plan_dtype(plan.output_dtype)
            output = np.empty(out_shape, dtype=out_dtype)
            self._rt.synchronize()
            self._rt.copy_device_to_host(final.ptr, output)
            return output
        finally:
            for ptr, nbytes in allocated_bufs:
                self._pool_free(ptr, nbytes)

    def execute_plan_to_device(
        self,
        plan: ExecutionPlan,
        device_inputs: list["DeviceArray"],
    ) -> "DeviceArray":
        """Run a multi-kernel plan on device-resident inputs, returning the
        result as a device-resident ``DeviceArray`` (no host transfer).

        Inputs are existing ``DeviceArray``s (registered as ``input_0``..).
        Plan intermediate buffers are pool-allocated and freed afterwards,
        except the ``final_output`` buffer, whose ownership transfers to the
        returned ``DeviceArray`` (the caller must ``free()`` it).
        """
        plan.validate()
        missing_kernels = plan.kernel_names() - set(self._kernels)
        if missing_kernels:
            raise RemoraExecutorError(
                f"Plan references unknown kernels: {sorted(missing_kernels)}"
            )
        if plan.final_output.startswith("input_"):
            raise RemoraExecutorError(
                "execute_plan_to_device requires final_output to be a plan buffer"
            )

        registry: dict[str, _DeviceBuffer] = {}
        for i, da in enumerate(device_inputs):
            registry[f"input_{i}"] = _DeviceBuffer(da.ptr, tuple(da.shape), da.dtype)

        owned: list[tuple[str, int, int]] = []
        for buf_spec in plan.buffers:
            dtype = _plan_dtype(buf_spec.dtype)
            n_elements = max(1, int(np.prod(buf_spec.shape, dtype=np.int64)))
            nbytes = n_elements * dtype.itemsize
            ptr = self._pool_alloc(nbytes)
            if buf_spec.init is not None:
                self._rt.copy_host_to_device(np.array(buf_spec.init, dtype=dtype), ptr)
            else:
                self._rt.memset_d32(ptr, 0, max(1, nbytes // 4))
            registry[buf_spec.name] = _DeviceBuffer(ptr, buf_spec.shape, dtype)
            owned.append((buf_spec.name, ptr, nbytes))

        for step in plan.steps:
            self._run_plan_step(step, registry)
        self._rt.synchronize()

        final = registry[plan.final_output]
        out_dtype = _plan_dtype(plan.output_dtype)
        out_nbytes = max(1, int(np.prod(plan.output_shape, dtype=np.int64))) * out_dtype.itemsize
        for name, ptr, nbytes in owned:
            if name != plan.final_output:
                self._pool_free(ptr, nbytes)
        return DeviceArray(self, final.ptr, tuple(plan.output_shape), out_dtype, out_nbytes)

    def execute_main(self, inputs: list[Any] | None = None, *, arena: Any | None = None) -> np.ndarray:
        """Run the program entry kernel using the shared executor-style API."""
        kernel_name = self._main_kernel_name()
        return self.execute(kernel_name, [] if inputs is None else inputs, arena=arena)

    def close(self) -> None:
        # A pool we own (no shared runtime pool) must be drained here; a shared
        # runtime pool is left intact for reuse and freed when the runtime is
        # closed (below, if we own the runtime).
        if self._owns_pool:
            self._pool.clear()
        for mod in self._modules:
            mod.close()
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

    def _run_plan_step(
        self,
        step: PlanStep,
        registry: dict[str, _DeviceBuffer],
    ) -> None:
        if isinstance(step, KernelStep):
            self._run_kernel_step(step, registry)
        elif isinstance(step, LoopPlan):
            for _ in range(step.count):
                for body_step in step.body:
                    self._run_kernel_step(body_step, registry)
                for a, b in step.swap_pairs:
                    registry[a], registry[b] = registry[b], registry[a]

    def _run_kernel_step(
        self,
        step: KernelStep,
        registry: dict[str, _DeviceBuffer],
    ) -> None:
        meta = self._meta[step.kernel_name]
        kernel = self._kernels[step.kernel_name]

        input_descs = []
        first_input_shape: tuple[int, ...] | None = None
        for ref in step.input_refs:
            buf = registry[ref]
            if first_input_shape is None:
                first_input_shape = buf.shape
            strides = _contiguous_strides(buf.shape)
            input_descs.append(
                make_memref_descriptor(buf.ptr, buf.shape, strides, buf.dtype)
            )

        out_buf = registry[step.output_ref]
        if step.is_reduction:
            nbytes = max(4, int(np.prod(out_buf.shape, dtype=np.int64)) * out_buf.dtype.itemsize)
            self._rt.memset_d32(out_buf.ptr, 0, max(1, nbytes // 4))

        out_strides = _contiguous_strides(out_buf.shape)
        output_desc = make_memref_descriptor(
            out_buf.ptr, out_buf.shape, out_strides, out_buf.dtype,
        )

        block_size = int(meta.block_size or 256)
        if meta.grid_size is not None:
            grid_size = meta.grid_size
        elif step.is_reduction and first_input_shape is not None:
            element_count = max(1, int(np.prod(first_input_shape, dtype=np.int64)))
            grid_size = int((element_count + block_size - 1) // block_size)
        else:
            element_count = max(1, int(np.prod(out_buf.shape, dtype=np.int64)))
            grid_size = int((element_count + block_size - 1) // block_size)

        kernel.launch(
            (grid_size, 1, 1),
            (block_size, 1, 1),
            [*input_descs, output_desc],
        )
        # No per-step synchronize: kernels serialize on the default
        # stream, and intermediate buffers are never read on the host
        # mid-plan.  execute_plan syncs once before the final D->H copy.


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
    syntax: str = "ml",
) -> np.ndarray:
    """Compile a Remora body program to GPU PTX and execute it.

    Tries a host-orchestrated loop plan for state-fold-over-iota
    programs (e.g. gradient descent).  Falls back to the standard
    IREE PTX pipeline for everything else.
    """
    from remora.compiler import compile_source, compile_source_to_ptx
    from remora.codegen import try_compile_state_fold_gpu

    art = compile_source(
        source, include_prelude=include_prelude, syntax=syntax, verify=False,
    )
    fold_result = try_compile_state_fold_gpu(art.hir)
    if fold_result is not None:
        ptx, kernels, plan = fold_result
        executor = RemoraExecutor(ptx, kernels)
        try:
            return executor.execute_plan(plan, [])
        finally:
            executor.close()

    artifact = compile_source_to_ptx(
        source, include_prelude=include_prelude, syntax=syntax,
    )
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
