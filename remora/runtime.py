"""CPU runtime helpers for Remora Dense Core."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
from shutil import which
import subprocess
import tempfile
from typing import Any, Callable

import numpy as np

from remora.abi import make_numpy_memref_descriptor
from remora.compiler import compile_function_source, compile_source
from remora.ast_nodes import BoolLit, FloatLit, FuncDef, IfExpr, IntLit, IotaExpr, VarExpr
from remora.display import format_result
from remora.errors import RemoraError
from remora.parser import parse_program
from remora.pipeline import (
    PipelineToolchain,
    PipelineUnavailable,
    detect_toolchain,
    run_cpu_pipeline_text,
    translate_mlir_to_llvmir,
)
from remora.prelude import with_prelude
from remora.typechecker import (
    TypeChecker,
    TypedApp,
    TypedArray,
    TypedCast,
    TypedDefinition,
    TypedExpr,
    TypedExprNode,
    TypedFold,
    TypedIf,
    TypedIndex,
    TypedLambda,
    TypedLeftSection,
    TypedLet,
    TypedMap,
    TypedOperatorFunc,
    TypedProgram,
    TypedRank,
    TypedRightSection,
    TypedShape,
)
from remora.types import ArrayType, BOOL, FLOAT, INT, RemoraType, ScalarType, StaticDim


class RuntimeUnavailable(RemoraError):
    """Raised when a requested runtime target is unavailable."""


class CUDAError(RuntimeUnavailable):
    """Raised when CUDA driver setup, memory operations, or launch fails."""


class EvaluationError(RemoraError):
    """Raised when CPU evaluation hits unsupported typed syntax."""


@dataclass(frozen=True)
class EvaluationResult:
    value: object
    type: RemoraType


@dataclass(frozen=True)
class CompiledCPUArtifact:
    library_path: Path
    temp_dir: tempfile.TemporaryDirectory[str]
    return_type: RemoraType
    cpu_threads: int | None = None

    def close(self) -> None:
        self.temp_dir.cleanup()


@dataclass(frozen=True)
class CompiledCPUFunctionArtifact:
    library_path: Path
    temp_dir: tempfile.TemporaryDirectory[str]
    function_name: str
    param_types: tuple[RemoraType, ...]
    return_type: RemoraType
    export_name: str
    cpu_threads: int | None = None

    def close(self) -> None:
        self.temp_dir.cleanup()


Value = object
Env = dict[str, Value]
CallableValue = Callable[..., Value]


def resolve_cpu_threads(cpu_threads: int | None = None) -> int | None:
    """Resolve requested CPU thread count from an explicit value or environment."""
    if cpu_threads is not None:
        return _validate_cpu_threads(cpu_threads, "cpu_threads")
    raw = os.environ.get("REMORA_NUM_THREADS")
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise EvaluationError("REMORA_NUM_THREADS must be a positive integer") from exc
    return _validate_cpu_threads(value, "REMORA_NUM_THREADS")


def _validate_cpu_threads(value: int, label: str) -> int:
    if value < 1:
        raise EvaluationError(f"{label} must be a positive integer")
    return int(value)


def cuda_available() -> bool:
    """Return whether the CUDA Python driver module can be imported."""
    try:
        _load_cuda_driver()
    except RuntimeUnavailable:
        return False
    return True


class CUDARuntime:
    """Small CUDA Driver API wrapper for direct Remora ABI kernels."""

    def __init__(self, device_idx: int = 0) -> None:
        self._cuda = _load_cuda_driver()
        self._ctx: Any | None = None
        self._deferred_frees: list[int] = []
        _cuda_check(self._cuda, self._cuda.cuInit(0), "cuInit failed")
        self._device = _cuda_value(
            self._cuda,
            self._cuda.cuDeviceGet(device_idx),
            "cuDeviceGet failed",
        )
        self._ctx = _cuda_value(
            self._cuda,
            self._cuda.cuCtxCreate(None, 0, self._device),
            "cuCtxCreate failed",
        )

    def load_ptx(self, ptx: str) -> "CUDAModule":
        module = _cuda_value(
            self._cuda,
            self._cuda.cuModuleLoadData(ptx.encode("utf-8")),
            "cuModuleLoadData failed",
        )
        return CUDAModule(module, self)

    def alloc(self, nbytes: int) -> int:
        if nbytes < 0:
            raise CUDAError("cannot allocate a negative number of bytes")
        ptr = _cuda_value(
            self._cuda,
            self._cuda.cuMemAlloc(int(nbytes)),
            "cuMemAlloc failed",
        )
        return int(ptr)

    def free(self, ptr: int) -> None:
        _cuda_check(self._cuda, self._cuda.cuMemFree(int(ptr)), "cuMemFree failed")

    def copy_host_to_device(self, host_array: np.ndarray, device_ptr: int) -> None:
        array = np.asarray(host_array)
        self.copy_host_bytes_to_device(int(array.ctypes.data), device_ptr, array.nbytes)

    def copy_device_to_host(self, device_ptr: int, host_array: np.ndarray) -> None:
        array = np.asarray(host_array)
        _cuda_check(
            self._cuda,
            self._cuda.cuMemcpyDtoH(int(array.ctypes.data), int(device_ptr), array.nbytes),
            "cuMemcpyDtoH failed",
        )

    def copy_host_bytes_to_device(self, host_address: int, device_ptr: int, nbytes: int) -> None:
        _cuda_check(
            self._cuda,
            self._cuda.cuMemcpyHtoD(int(device_ptr), int(host_address), int(nbytes)),
            "cuMemcpyHtoD failed",
        )

    def synchronize(self) -> None:
        try:
            _cuda_check(self._cuda, self._cuda.cuCtxSynchronize(), "cuCtxSynchronize failed")
        finally:
            self._free_deferred()

    def close(self) -> None:
        self._free_deferred()
        if self._ctx is not None:
            _cuda_check(self._cuda, self._cuda.cuCtxDestroy(self._ctx), "cuCtxDestroy failed")
            self._ctx = None

    def _defer_free(self, ptr: int) -> None:
        self._deferred_frees.append(int(ptr))

    def _free_deferred(self) -> None:
        pending = self._deferred_frees
        self._deferred_frees = []
        for ptr in pending:
            self.free(ptr)

    def __enter__(self) -> "CUDARuntime":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CUDAModule:
    """Loaded CUDA module for direct Remora ABI kernels."""

    def __init__(self, module: Any, runtime: CUDARuntime) -> None:
        self._module = module
        self._rt = runtime

    def get_function(self, name: str) -> "CUDAKernel":
        function = _cuda_value(
            self._rt._cuda,
            self._rt._cuda.cuModuleGetFunction(self._module, name.encode("utf-8")),
            f"cuModuleGetFunction failed for {name}",
        )
        return CUDAKernel(function, self._rt)

    def close(self) -> None:
        if self._module is not None:
            _cuda_check(
                self._rt._cuda,
                self._rt._cuda.cuModuleUnload(self._module),
                "cuModuleUnload failed",
            )
            self._module = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CUDAKernel:
    """CUDA kernel launcher with Remora descriptor-aware argument packing."""

    def __init__(self, function: Any, runtime: CUDARuntime) -> None:
        self._function = function
        self._rt = runtime

    def launch(
        self,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[object],
        *,
        shared_mem: int = 0,
        stream: int = 0,
    ) -> None:
        packed_args, kernel_args = _pack_cuda_kernel_args(args, self._rt)
        try:
            _cuda_check(
                self._rt._cuda,
                self._rt._cuda.cuLaunchKernel(
                    self._function,
                    int(grid[0]),
                    int(grid[1]),
                    int(grid[2]),
                    int(block[0]),
                    int(block[1]),
                    int(block[2]),
                    int(shared_mem),
                    stream,
                    kernel_args,
                    0,
                ),
                "cuLaunchKernel failed",
            )
        except TypeError as exc:
            raise CUDAError(f"cuLaunchKernel argument marshalling failed: {exc}") from exc
        # Keep scalar ctypes values alive until after cuLaunchKernel returns.
        _ = packed_args


def _pack_cuda_kernel_args(
    args: list[object],
    runtime: CUDARuntime,
) -> tuple[list[ctypes._SimpleCData | ctypes.Structure], int]:
    packed_args: list[ctypes._SimpleCData | ctypes.Structure] = []
    kernel_args: ctypes.Array[ctypes.c_void_p] = (ctypes.c_void_p * len(args))()
    for arg in args:
        if isinstance(arg, ctypes.Structure):
            descriptor_ptr = runtime.alloc(ctypes.sizeof(arg))
            runtime.copy_host_bytes_to_device(
                ctypes.addressof(arg),
                descriptor_ptr,
                ctypes.sizeof(arg),
            )
            runtime._defer_free(descriptor_ptr)
            c_arg: ctypes._SimpleCData | ctypes.Structure = ctypes.c_uint64(descriptor_ptr)
        elif isinstance(arg, bool):
            c_arg = ctypes.c_bool(arg)
        elif isinstance(arg, int):
            c_arg = ctypes.c_uint64(arg)
        elif isinstance(arg, float):
            c_arg = ctypes.c_float(arg)
        elif isinstance(arg, np.integer):
            c_arg = ctypes.c_int64(int(arg))
        elif isinstance(arg, np.floating):
            c_arg = ctypes.c_float(float(arg))
        else:
            raise CUDAError(f"unsupported CUDA kernel argument type {type(arg).__name__}")
        packed_args.append(c_arg)
        kernel_args[len(packed_args) - 1] = ctypes.cast(ctypes.byref(c_arg), ctypes.c_void_p)
    return packed_args, int(ctypes.cast(kernel_args, ctypes.c_void_p).value)


def _load_cuda_driver() -> Any:
    try:
        from cuda import cuda as cuda_driver

        return cuda_driver
    except Exception as first_exc:
        try:
            from cuda.bindings import driver as cuda_driver

            return cuda_driver
        except Exception as second_exc:
            raise RuntimeUnavailable(
                "cuda-python driver bindings are not available"
            ) from second_exc


def _cuda_success(cuda_driver: Any) -> Any:
    try:
        return cuda_driver.CUresult.CUDA_SUCCESS
    except AttributeError:
        return 0


def _cuda_error_code(result: object) -> object:
    if isinstance(result, tuple):
        return result[0]
    return result


def _cuda_check(cuda_driver: Any, result: object, message: str) -> None:
    error_code = _cuda_error_code(result)
    if error_code != _cuda_success(cuda_driver):
        raise CUDAError(f"{message}: {error_code}")


def _cuda_value(cuda_driver: Any, result: object, message: str) -> Any:
    if not isinstance(result, tuple):
        _cuda_check(cuda_driver, result, message)
        return None
    _cuda_check(cuda_driver, result[0], message)
    values = result[1:]
    if len(values) == 1:
        return values[0]
    return values


def evaluate_source(source: str, *, include_prelude: bool = True) -> EvaluationResult:
    program_source = with_prelude(source) if include_prelude else source
    typed = TypeChecker().check_program(parse_program(program_source))
    return evaluate_typed_program(typed)


def evaluate_source_compiled(
    source: str,
    *,
    include_prelude: bool = True,
    cpu_threads: int | None = None,
) -> EvaluationResult:
    artifact = CPUExecutor.compile_source(
        source,
        include_prelude=include_prelude,
        cpu_threads=cpu_threads,
    )
    try:
        value = CPUExecutor(artifact).execute_main([])
        return EvaluationResult(value, artifact.return_type)
    finally:
        artifact.close()


def evaluate_typed_program(program: TypedProgram) -> EvaluationResult:
    if program.body is None or program.type is None:
        raise EvaluationError("definition-only programs cannot be evaluated")

    env: Env = {}
    for definition in program.definitions:
        _bind_definition(definition, env)
    return EvaluationResult(_eval_expr(program.body, env), program.type)


class CPUExecutor:
    """Execute compiled Remora programs on CPU through LLVM and ctypes."""

    def __init__(self, artifact: CompiledCPUArtifact) -> None:
        self._artifact = artifact
        self._library = ctypes.CDLL(str(artifact.library_path))

    @classmethod
    def compile_source(
        cls,
        source: str,
        *,
        include_prelude: bool = True,
        toolchain: PipelineToolchain | None = None,
        cpu_threads: int | None = None,
    ) -> CompiledCPUArtifact:
        resolved_cpu_threads = resolve_cpu_threads(cpu_threads)
        compiler_artifact = compile_source(
            source,
            verify=False,
            include_prelude=include_prelude,
            export_output_descriptor=True,
        )
        if compiler_artifact.return_type is None:
            raise EvaluationError("definition-only programs cannot be compiled for CPU execution")

        toolchain = detect_toolchain() if toolchain is None else toolchain
        lowered = run_cpu_pipeline_text(compiler_artifact.mlir_text, toolchain=toolchain)
        llvm_ir = translate_mlir_to_llvmir(lowered, toolchain=toolchain)

        temp_dir = tempfile.TemporaryDirectory(prefix="remora-cpu-")
        root = Path(temp_dir.name)
        ll_path = root / "module.ll"
        obj_path = root / "module.o"
        so_path = root / "module.so"
        ll_path.write_text(llvm_ir, encoding="utf-8")

        llc = toolchain.llc
        if llc is None:
            temp_dir.cleanup()
            raise PipelineUnavailable("llc is required for compiled CPU execution")
        linker = which("gcc") or which("cc")
        if linker is None:
            temp_dir.cleanup()
            raise PipelineUnavailable("gcc or cc is required for compiled CPU execution")

        _run_checked(
            [
                llc,
                "-filetype=obj",
                "-relocation-model=pic",
                str(ll_path),
                "-o",
                str(obj_path),
            ],
            "llc failed during compiled CPU execution",
            temp_dir,
        )
        _run_checked(
            [linker, "-shared", str(obj_path), "-o", str(so_path)],
            "system linker failed during compiled CPU execution",
            temp_dir,
        )
        return CompiledCPUArtifact(
            so_path,
            temp_dir,
            compiler_artifact.return_type,
            resolved_cpu_threads,
        )

    def execute_main(self, inputs: list[np.ndarray] | None = None) -> object:
        if inputs is not None and len(inputs) != 0:
            raise EvaluationError("compiled CPU main does not accept inputs")
        return_type = self._artifact.return_type
        output = _empty_output_value(return_type)
        self.execute_main_into(output)
        if isinstance(return_type, ScalarType):
            return output.item()
        return output

    def execute_main_into(self, output: np.ndarray) -> None:
        output_type = self._artifact.return_type
        expected_shape = _result_shape(output_type)
        expected_dtype = _result_dtype(output_type)
        if output.shape != expected_shape:
            raise EvaluationError(
                f"compiled CPU output shape mismatch: expected {expected_shape}, got {output.shape}"
            )
        if output.dtype != expected_dtype:
            raise EvaluationError(
                f"compiled CPU output dtype mismatch: expected {expected_dtype}, got {output.dtype}"
            )

        descriptor = make_numpy_memref_descriptor(output)
        function = self._library._mlir_ciface_remora_main_out
        function.argtypes = [ctypes.POINTER(type(descriptor))]
        function.restype = None
        function(ctypes.byref(descriptor))


class CPUFunctionExecutor:
    """Execute one compiled top-level Remora function with descriptor inputs."""

    def __init__(self, artifact: CompiledCPUFunctionArtifact) -> None:
        self._artifact = artifact
        self._library = ctypes.CDLL(str(artifact.library_path))

    @classmethod
    def compile_source(
        cls,
        source: str,
        function_name: str,
        param_types: tuple[RemoraType, ...],
        *,
        include_prelude: bool = True,
        toolchain: PipelineToolchain | None = None,
        cpu_threads: int | None = None,
    ) -> CompiledCPUFunctionArtifact:
        resolved_cpu_threads = resolve_cpu_threads(cpu_threads)
        compiler_artifact = compile_function_source(
            source,
            function_name,
            param_types,
            verify=False,
            include_prelude=include_prelude,
        )
        toolchain = detect_toolchain() if toolchain is None else toolchain
        lowered = run_cpu_pipeline_text(compiler_artifact.mlir_text, toolchain=toolchain)
        llvm_ir = translate_mlir_to_llvmir(lowered, toolchain=toolchain)
        temp_dir = _compile_llvm_ir_to_shared_library(llvm_ir, toolchain)
        return CompiledCPUFunctionArtifact(
            temp_dir[1],
            temp_dir[0],
            function_name,
            param_types,
            compiler_artifact.return_type,
            "remora_call",
            resolved_cpu_threads,
        )

    def execute(self, *inputs: np.ndarray) -> EvaluationResult:
        output = _empty_output_value(self._artifact.return_type)
        self.execute_into(output, *inputs)
        if isinstance(self._artifact.return_type, ScalarType):
            return EvaluationResult(output.item(), self._artifact.return_type)
        return EvaluationResult(output, self._artifact.return_type)

    def execute_into(self, output: np.ndarray, *inputs: np.ndarray) -> None:
        if len(inputs) != len(self._artifact.param_types):
            raise EvaluationError(
                f"compiled CPU function expects {len(self._artifact.param_types)} inputs, got {len(inputs)}"
            )
        for index, (input_value, input_type) in enumerate(zip(inputs, self._artifact.param_types)):
            _validate_numpy_value(
                np.asarray(input_value),
                input_type,
                f"compiled CPU function input {index}",
            )
        _validate_numpy_value(
            output,
            self._artifact.return_type,
            "compiled CPU function output",
        )

        descriptors = [make_numpy_memref_descriptor(np.asarray(input_value)) for input_value in inputs]
        output_descriptor = make_numpy_memref_descriptor(output)
        function = getattr(self._library, f"_mlir_ciface_{self._artifact.export_name}")
        descriptor_types = [type(descriptor) for descriptor in descriptors]
        function.argtypes = [
            *(ctypes.POINTER(descriptor_type) for descriptor_type in descriptor_types),
            ctypes.POINTER(type(output_descriptor)),
        ]
        function.restype = None
        function(
            *(ctypes.byref(descriptor) for descriptor in descriptors),
            ctypes.byref(output_descriptor),
        )


def _run_checked(
    args: list[str],
    message: str,
    temp_dir: tempfile.TemporaryDirectory[str],
) -> None:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        temp_dir.cleanup()
        stderr = result.stderr.strip()
        raise RuntimeUnavailable(f"{message}: {stderr}")


def _compile_llvm_ir_to_shared_library(
    llvm_ir: str,
    toolchain: PipelineToolchain,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temp_dir = tempfile.TemporaryDirectory(prefix="remora-cpu-")
    root = Path(temp_dir.name)
    ll_path = root / "module.ll"
    obj_path = root / "module.o"
    so_path = root / "module.so"
    ll_path.write_text(llvm_ir, encoding="utf-8")

    llc = toolchain.llc
    if llc is None:
        temp_dir.cleanup()
        raise PipelineUnavailable("llc is required for compiled CPU execution")
    linker = which("gcc") or which("cc")
    if linker is None:
        temp_dir.cleanup()
        raise PipelineUnavailable("gcc or cc is required for compiled CPU execution")

    _run_checked(
        [
            llc,
            "-filetype=obj",
            "-relocation-model=pic",
            str(ll_path),
            "-o",
            str(obj_path),
        ],
        "llc failed during compiled CPU execution",
        temp_dir,
    )
    _run_checked(
        [linker, "-shared", str(obj_path), "-o", str(so_path)],
        "system linker failed during compiled CPU execution",
        temp_dir,
    )
    return temp_dir, so_path


def _empty_output_value(value_type: RemoraType) -> np.ndarray:
    return np.empty(_result_shape(value_type), dtype=_result_dtype(value_type))


def _result_shape(value_type: RemoraType) -> tuple[int, ...]:
    if isinstance(value_type, ScalarType):
        return ()
    if isinstance(value_type, ArrayType):
        return tuple(dim.value for dim in value_type.shape)
    raise RuntimeUnavailable(f"compiled CPU return type {value_type} is not supported")


def _result_dtype(value_type: RemoraType) -> np.dtype:
    if isinstance(value_type, ScalarType):
        return np.dtype(_numpy_dtype(value_type))
    if isinstance(value_type, ArrayType):
        return np.dtype(_numpy_dtype(value_type.element))
    raise RuntimeUnavailable(f"compiled CPU return type {value_type} is not supported")


def _validate_numpy_value(value: np.ndarray, value_type: RemoraType, label: str) -> None:
    expected_shape = _result_shape(value_type)
    expected_dtype = _result_dtype(value_type)
    if value.shape != expected_shape:
        raise EvaluationError(
            f"{label} shape mismatch: expected {expected_shape}, got {value.shape}"
        )
    if value.dtype != expected_dtype:
        raise EvaluationError(
            f"{label} dtype mismatch: expected {expected_dtype}, got {value.dtype}"
        )


def format_value(value: object) -> str:
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return format_result(value, ArrayType(BOOL, tuple(StaticDim(size) for size in value.shape)))
        if np.issubdtype(value.dtype, np.integer):
            return format_result(value, ArrayType(INT, tuple(StaticDim(size) for size in value.shape)))
        if np.issubdtype(value.dtype, np.floating):
            return format_result(value, ArrayType(FLOAT, tuple(StaticDim(size) for size in value.shape)))
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return format_result(value, BOOL)
    if isinstance(value, int):
        return format_result(value, INT)
    if isinstance(value, float):
        return format_result(value, FLOAT)
    return repr(value)


def _bind_definition(definition: TypedDefinition, env: Env) -> None:
    if isinstance(definition.definition, FuncDef):
        return
    if definition.value is None:
        raise EvaluationError("function definitions are deferred")
    env[definition.definition.name] = _eval_expr(definition.value, env)


def _eval_expr(expr: TypedExpr, env: Env) -> Value:
    if isinstance(expr, TypedCast):
        return _cast_scalar(_eval_expr(expr.value, env), expr.to_type)

    if isinstance(expr, TypedArray):
        values = [_eval_expr(element, env) for element in expr.elements]
        return np.array(values, dtype=_numpy_dtype(expr.type.element))

    if isinstance(expr, TypedMap):
        arrays = [_eval_expr(array, env) for array in expr.arrays]
        callable_value = _eval_callable(expr.func, env)
        return _map_value(callable_value, arrays, expr.cell_shape, expr.type)

    if isinstance(expr, TypedFold):
        array = _eval_expr(expr.array, env)
        acc = _eval_expr(expr.init, env)
        callable_value = _eval_callable(expr.func, env)
        if not isinstance(array, np.ndarray):
            raise EvaluationError("fold expects an array value")
        for item in array:
            acc = callable_value(acc, item)
        return _coerce_runtime_value(acc, expr.type)

    if isinstance(expr, TypedShape):
        return np.array(
            [dim.value for dim in _shape_dims(expr.array.type)],
            dtype=np.int32,
        )

    if isinstance(expr, TypedRank):
        return int(expr.array.type.rank)

    if isinstance(expr, TypedIndex):
        array = _eval_expr(expr.array, env)
        indices = tuple(int(_eval_expr(index, env)) for index in expr.indices)
        if not isinstance(array, np.ndarray):
            raise EvaluationError("indexing expects an array value")
        value = array[indices]
        return _coerce_runtime_value(value, expr.type)

    if isinstance(expr, TypedLambda):
        return _lambda_callable(expr, env)

    if isinstance(expr, TypedApp):
        if isinstance(expr.func, TypedExprNode) and isinstance(expr.func.expr, VarExpr):
            if expr.func.expr.name in _OPS:
                args = [_eval_expr(arg, env) for arg in expr.args]
                return _coerce_runtime_value(_apply_op(expr.func.expr.name, *args), expr.type)
        func = _eval_expr(expr.func, env)
        if not callable(func):
            raise EvaluationError("application target is not callable")
        args = [_eval_expr(arg, env) for arg in expr.args]
        return _coerce_runtime_value(func(*args), expr.type)

    if isinstance(expr, TypedLet):
        inner_env = dict(env)
        inner_env[expr.name] = _eval_expr(expr.value, env)
        return _eval_expr(expr.body, inner_env)

    if isinstance(expr, TypedIf):
        condition = _eval_expr(expr.condition, env)
        return _eval_expr(expr.then_branch if bool(condition) else expr.else_branch, env)

    if isinstance(expr, (TypedOperatorFunc, TypedLeftSection, TypedRightSection)):
        return _eval_callable(expr, env)

    if isinstance(expr, TypedExprNode):
        return _eval_expr_node(expr, env)

    raise EvaluationError(f"CPU evaluation for {type(expr).__name__} is deferred")


def _eval_expr_node(expr: TypedExprNode, env: Env) -> Value:
    ast = expr.expr
    if isinstance(ast, IntLit):
        return int(ast.value)
    if isinstance(ast, FloatLit):
        return float(ast.value)
    if isinstance(ast, BoolLit):
        return bool(ast.value)
    if isinstance(ast, VarExpr):
        try:
            return env[ast.name]
        except KeyError as exc:
            raise EvaluationError(f"unbound runtime variable '{ast.name}'") from exc
    if isinstance(ast, IotaExpr):
        if not isinstance(expr.type, ArrayType):
            raise EvaluationError("iota must have an array type")
        size = expr.type.shape[0]
        if not isinstance(size, StaticDim):
            raise EvaluationError("iota requires a static dimension")
        return np.arange(size.value, dtype=np.int32)
    raise EvaluationError(f"CPU evaluation for {type(ast).__name__} is deferred")


def _eval_callable(expr: TypedExpr, env: Env) -> CallableValue:
    if isinstance(expr, TypedLambda):
        return _lambda_callable(expr, env)
    if isinstance(expr, TypedOperatorFunc):
        return lambda left, right: _coerce_runtime_value(
            _apply_op(expr.expr.op, left, right), expr.type.result
        )
    if isinstance(expr, TypedLeftSection):
        left = _eval_expr(expr.arg, env)
        return lambda right: _coerce_runtime_value(
            _apply_op(expr.expr.op, left, right), expr.type.result
        )
    if isinstance(expr, TypedRightSection):
        right = _eval_expr(expr.arg, env)
        return lambda left: _coerce_runtime_value(
            _apply_op(expr.expr.op, left, right), expr.type.result
        )
    value = _eval_expr(expr, env)
    if not callable(value):
        raise EvaluationError("expected a callable value")
    return value


def _lambda_callable(expr: TypedLambda, env: Env) -> CallableValue:
    closed_env = dict(env)

    def call(*args: Value) -> Value:
        if len(args) != len(expr.params):
            raise EvaluationError("lambda arity mismatch")
        inner_env = dict(closed_env)
        for (name, _param_type), arg in zip(expr.params, args):
            inner_env[name] = arg
        return _eval_expr(expr.body, inner_env)

    return call


def _map_value(
    callable_value: CallableValue,
    arrays: list[Value],
    cell_shape: tuple[StaticDim, ...],
    result_type: RemoraType,
) -> Value:
    if len(arrays) == 2:
        return _binary_map_value(callable_value, arrays[0], arrays[1], result_type)
    if len(arrays) != 1:
        raise EvaluationError("map currently supports one or two arrays")
    array = arrays[0]
    if not isinstance(array, np.ndarray):
        return _coerce_runtime_value(callable_value(array), result_type)

    cell_rank = len(cell_shape)
    frame_rank = array.ndim - cell_rank
    if frame_rank < 0:
        raise EvaluationError("map cell rank exceeds array rank")

    if frame_rank == 0:
        return _coerce_runtime_value(callable_value(array), result_type)

    frame_shape = array.shape[:frame_rank]
    values = [
        callable_value(array[index] if cell_rank else array[index].item())
        for index in np.ndindex(frame_shape)
    ]
    if isinstance(result_type, ArrayType):
        return np.array(values, dtype=_numpy_dtype(result_type.element)).reshape(
            tuple(dim.value for dim in result_type.shape)
        )
    return _coerce_runtime_value(values[0], result_type)


def _binary_map_value(
    callable_value: CallableValue,
    left: Value,
    right: Value,
    result_type: RemoraType,
) -> Value:
    if not isinstance(left, np.ndarray) and not isinstance(right, np.ndarray):
        return _coerce_runtime_value(callable_value(left, right), result_type)
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
        raise EvaluationError("binary map currently expects both operands to be arrays or scalars")
    if left.shape != right.shape:
        raise EvaluationError("binary map expects matching array shapes")
    values = [
        callable_value(left[index].item(), right[index].item())
        for index in np.ndindex(left.shape)
    ]
    if isinstance(result_type, ArrayType):
        return np.array(values, dtype=_numpy_dtype(result_type.element)).reshape(
            tuple(dim.value for dim in result_type.shape)
        )
    return _coerce_runtime_value(values[0], result_type)


def _apply_op(op: str, left: Value, right: Value) -> Value:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return float(left) / float(right)
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "&&":
        return bool(left) and bool(right)
    if op == "||":
        return bool(left) or bool(right)
    raise EvaluationError(f"operator {op} is not supported by CPU evaluation")


def _coerce_runtime_value(value: Value, value_type: RemoraType) -> Value:
    if isinstance(value_type, ScalarType):
        return _cast_scalar(value, value_type)
    if isinstance(value_type, ArrayType):
        return np.asarray(value, dtype=_numpy_dtype(value_type.element)).reshape(
            tuple(dim.value for dim in value_type.shape)
        )
    return value


def _shape_dims(value_type: RemoraType) -> tuple[StaticDim, ...]:
    if isinstance(value_type, ArrayType):
        return value_type.shape
    if isinstance(value_type, ScalarType):
        return ()
    raise EvaluationError("shape/rank of function values is deferred")


def _cast_scalar(value: Value, value_type: ScalarType) -> Value:
    if isinstance(value, np.generic):
        value = value.item()
    if value_type == INT:
        return int(value)
    if value_type == FLOAT:
        return float(value)
    if value_type == BOOL:
        return bool(value)
    raise EvaluationError(f"cannot cast runtime value to {value_type}")


def _numpy_dtype(element_type: ScalarType) -> np.dtype:
    if element_type == INT:
        return np.dtype(np.int32)
    if element_type == FLOAT:
        return np.dtype(np.float32)
    if element_type == BOOL:
        return np.dtype(np.bool_)
    raise EvaluationError(f"unsupported array element type {element_type}")


_OPS = {"+", "-", "*", "/", "<", "<=", "==", "!=", "&&", "||"}
