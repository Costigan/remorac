"""Public compiler facade for Remora Dense Core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from remora.ad_source import GradientSourceArtifact

from remora.codegen import (
    CodegenUnavailable,
    KernelMeta,
    generate_mlir_descriptor_abi_ptx,
    generate_ptx,
    generate_rank1_f32_unary_mlir_descriptor_abi_ptx,
)
from remora.defunc import defunctionalize
from remora.elaborated import CoreProgram
from remora.elaborate import elaborate_program
from remora.erase import erase_to_hir
from remora.hir import HIRFunction, HIRParam, HIRProgram, lower_expr, lower_to_hir
from remora.lowering import MLIRLowering
from remora.lowering.types import RemoraLoweringError
from remora.parser import parse_program
from remora.lisp_reader import parse_lisp as parse_lisp_program
from remora.pipeline import run_validation_pipeline, verify_module_text
from remora.pipeline import PipelineUnavailable
from remora.prelude import with_prelude
from remora.typechecker import TypeChecker, TypeEnv, TypedApp, TypedGrad, TypedProgram
from remora.index import ShapeExpr
from remora.types import DimExpr, FuncType, RemoraType, RemoraTypeError
from remora.ast_nodes import AppExpr, FuncDef, IndexAppExpr, Program, VarExpr


def _parse_source(source: str, syntax: str = "ml") -> Program:
    if syntax == "lisp":
        return parse_lisp_program(source)
    return parse_program(source)
from remora.gpu_lowering import (
    GPUModuleScaffold,
    GPUScaffoldError,
    build_descriptor_abi_f32_reduction_gpu_module,
    build_gpu_scaffold_for_function,
)


@dataclass(frozen=True)
class CompilerArtifact:
    source: str
    typed: TypedProgram
    core: CoreProgram
    hir: HIRProgram
    mlir_module: object
    mlir_text: str

    @property
    def return_type(self) -> RemoraType | None:
        return self.typed.type


@dataclass(frozen=True)
class FunctionCompilerArtifact:
    source: str
    function_name: str
    function_type: FuncType
    hir_function: HIRFunction
    mlir_module: object
    mlir_text: str
    specialization_name: str | None = None
    index_args: tuple[DimExpr | ShapeExpr, ...] = ()

    @property
    def return_type(self) -> RemoraType:
        return self.function_type.result


@dataclass(frozen=True)
class PTXArtifact:
    compiler: CompilerArtifact
    ptx_text: str
    kernels: list[KernelMeta]


@dataclass(frozen=True)
class SupportedGPUFunctionArtifact:
    compiler: FunctionCompilerArtifact
    scaffold: GPUModuleScaffold
    ptx_text: str
    kernels: list[KernelMeta]


@dataclass(frozen=True)
class GradientCompilerArtifact:
    gradient_source: GradientSourceArtifact
    compiler: FunctionCompilerArtifact


@dataclass(frozen=True)
class GradientGPUArtifact:
    gradient_source: GradientSourceArtifact
    gpu: SupportedGPUFunctionArtifact


def compile_source(
    source: str,
    *,
    verify: bool = True,
    include_prelude: bool = True,
    export_output_descriptor: bool = False,
    syntax: str = "ml",
) -> CompilerArtifact:
    _maybe_include_prelude = include_prelude and syntax == "ml"
    program_source = with_prelude(source) if _maybe_include_prelude else source
    ast = _parse_source(program_source, syntax)
    typed = TypeChecker().check_program(ast)
    rewritten = _rewrite_applied_source_gradient(
        source,
        ast,
        typed,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    if rewritten is not None:
        typed = TypeChecker().check_program(rewritten)
    core = elaborate_program(typed)
    hir = defunctionalize(erase_to_hir(core))
    mlir_module = MLIRLowering().lower_program(
        hir,
        export_output_descriptor=export_output_descriptor,
    ).module
    if verify:
        run_validation_pipeline(mlir_module)
        verify_module_text(str(mlir_module))
    return CompilerArtifact(
        source=source,
        typed=typed,
        core=core,
        hir=hir,
        mlir_module=mlir_module,
        mlir_text=str(mlir_module),
    )


def compile_source_to_mlir(
    source: str,
    *,
    verify: bool = True,
    include_prelude: bool = True,
    syntax: str = "ml",
) -> str:
    return compile_source(
        source,
        verify=verify,
        include_prelude=include_prelude,
        export_output_descriptor=False,
        syntax=syntax,
    ).mlir_text


def compile_source_to_ptx(
    source: str,
    *,
    verify: bool = True,
    include_prelude: bool = True,
    syntax: str = "ml",
) -> PTXArtifact:
    artifact = compile_source(
        source,
        verify=verify,
        include_prelude=include_prelude,
        export_output_descriptor=False,
        syntax=syntax,
    )
    ptx_text, kernels = generate_ptx(artifact.mlir_module)
    return PTXArtifact(artifact, ptx_text, kernels)


def compile_function_source_to_rank1_mlir_gpu_ptx(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    include_prelude: bool = True,
    kernel_name: str | None = None,
) -> tuple[str, list[KernelMeta], FunctionCompilerArtifact]:
    """Compile one supported rank-1 unary/binary function to MLIR-derived GPU PTX."""
    artifact = compile_function_source(
        source,
        function_name,
        param_types,
        verify=False,
        include_prelude=include_prelude,
    )
    ptx, kernels = generate_rank1_f32_unary_mlir_descriptor_abi_ptx(
        artifact.hir_function,
        kernel_name=kernel_name,
    )
    return ptx, kernels, artifact


def compile_function_source_to_mlir_gpu_ptx(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    include_prelude: bool = True,
    kernel_name: str | None = None,
) -> tuple[str, list[KernelMeta], FunctionCompilerArtifact]:
    """Compile one supported function to MLIR-derived GPU PTX."""
    artifact = compile_function_source(
        source,
        function_name,
        param_types,
        verify=False,
        include_prelude=include_prelude,
    )
    ptx, kernels = generate_mlir_descriptor_abi_ptx(
        artifact.hir_function,
        kernel_name=kernel_name,
    )
    return ptx, kernels, artifact


def compile_function_source_to_supported_gpu_artifacts(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    include_prelude: bool = True,
    kernel_name: str | None = None,
    syntax: str = "ml",
) -> SupportedGPUFunctionArtifact:
    """Build the current inspection and execution GPU artifacts for one function.

    The returned scaffold is the `gpu.module` artifact for the rank-1 through
    rank-3 float unary/binary map slice. The returned PTX prefers the
    MLIR-derived descriptor-ABI path; when standalone NVPTX tools are not
    available it falls back to the older direct PTX slice so callers can still
    inspect/launch the supported ABI shape in constrained environments.
    """
    artifact = compile_function_source(
        source,
        function_name,
        param_types,
        verify=False,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    kernel = kernel_name or f"remora_{function_name}"
    ptx_text, kernels = generate_mlir_descriptor_abi_ptx(
        artifact.hir_function,
        kernel_name=kernel,
    )
    try:
        scaffold = build_gpu_scaffold_for_function(
            artifact.hir_function,
            kernel_name=kernel,
        )
    except GPUScaffoldError:
        scaffold = build_descriptor_abi_f32_reduction_gpu_module(
            artifact.hir_function,
            kernel_name=kernel,
        )

    return SupportedGPUFunctionArtifact(
        compiler=artifact,
        scaffold=scaffold,
        ptx_text=ptx_text,
        kernels=kernels,
    )


def compile_function_source(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    verify: bool = True,
    include_prelude: bool = True,
    export_name: str = "remora_call",
    syntax: str = "ml",
) -> FunctionCompilerArtifact:
    """Compile one top-level function with explicit static parameter types."""
    _maybe_include_prelude = include_prelude and syntax == "ml"
    program_source = with_prelude(source) if _maybe_include_prelude else source
    program = _parse_source(program_source, syntax)
    checker = TypeChecker()
    env = TypeEnv()
    function_def: FuncDef | None = None
    for definition in program.definitions:
        typed_definition, env = checker.check_definition(definition, env)
        if isinstance(definition, FuncDef) and definition.name == function_name:
            function_def = definition

    if function_def is None:
        raise ValueError(f"function {function_name!r} is not defined")

    typed_function = checker.specialize_top_level_function(
        function_def,
        param_types,
        env,
    )
    function_type = typed_function.type
    # Ensure the specialized body has no free index variables
    from remora.dependent_types import free_type_index_vars
    free_vars = free_type_index_vars(function_type)
    if free_vars:
        names = ", ".join(sorted(free_vars))
        raise RemoraTypeError(
            f"compiled function {function_name!r} has unspecialized "
            f"index variables: {names}"
        )
    internal_name = typed_function.specialization_name or function_name
    hir_function = HIRFunction(
        internal_name,
        [HIRParam(name, param_type) for name, param_type in typed_function.params],
        lower_expr(typed_function.body),
        function_type.result,
    )
    try:
        lowered = MLIRLowering().lower_function_descriptor_export(
            hir_function,
            export_name=export_name,
        )
        if verify:
            run_validation_pipeline(lowered.module)
            verify_module_text(str(lowered.module))
        mlir_module = lowered.module
        mlir_text = str(lowered.module)
    except RemoraLoweringError:
        mlir_module = None
        mlir_text = ""
    return FunctionCompilerArtifact(
        source=source,
        function_name=function_name,
        function_type=function_type,
        hir_function=hir_function,
        mlir_module=mlir_module,
        mlir_text=mlir_text,
        specialization_name=typed_function.specialization_name,
        index_args=typed_function.index_args,
    )


def compile_gradient_function_source(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    example_input: np.ndarray | None = None,
    *,
    gradient_name: str | None = None,
    include_prelude: bool = True,
    syntax: str = "ml",
    verify: bool = True,
) -> GradientCompilerArtifact:
    """Generate and compile a specialized unary gradient for the CPU path."""
    from remora.ad_source import generate_gradient_function_source

    gradient = generate_gradient_function_source(
        source,
        function_name,
        param_types,
        example_input,
        gradient_name=gradient_name,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    compiler = compile_function_source(
        gradient.source,
        gradient.function_name,
        gradient.param_types,
        verify=verify,
        include_prelude=False,
        syntax="lisp",
    )
    return GradientCompilerArtifact(gradient, compiler)


def compile_gradient_function_source_to_supported_gpu_artifacts(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    example_input: np.ndarray | None = None,
    *,
    gradient_name: str | None = None,
    include_prelude: bool = True,
    syntax: str = "ml",
    kernel_name: str | None = None,
) -> GradientGPUArtifact:
    """Generate and compile a specialized unary gradient for the GPU path."""
    from remora.ad_source import generate_gradient_function_source

    gradient = generate_gradient_function_source(
        source,
        function_name,
        param_types,
        example_input,
        gradient_name=gradient_name,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    kernel = kernel_name or f"remora_{gradient.function_name.replace('-', '_')}"
    gpu = compile_function_source_to_supported_gpu_artifacts(
        gradient.source,
        gradient.function_name,
        gradient.param_types,
        include_prelude=False,
        kernel_name=kernel,
        syntax="lisp",
    )
    return GradientGPUArtifact(gradient, gpu)


def compile_source_gradient_function(
    source: str,
    *,
    gradient_name: str | None = None,
    include_prelude: bool = True,
    syntax: str = "ml",
    verify: bool = True,
) -> GradientCompilerArtifact:
    """Compile the concrete source-level `(grad f)` request in a program body."""
    function_name, param_types = _source_gradient_request(
        source, include_prelude=include_prelude, syntax=syntax
    )
    return compile_gradient_function_source(
        source,
        function_name,
        param_types,
        gradient_name=gradient_name,
        include_prelude=include_prelude,
        syntax=syntax,
        verify=verify,
    )


def compile_source_gradient_function_to_supported_gpu_artifacts(
    source: str,
    *,
    gradient_name: str | None = None,
    include_prelude: bool = True,
    syntax: str = "ml",
    kernel_name: str | None = None,
) -> GradientGPUArtifact:
    """Compile a concrete source-level `(grad f)` request for the GPU path."""
    function_name, param_types = _source_gradient_request(
        source, include_prelude=include_prelude, syntax=syntax
    )
    return compile_gradient_function_source_to_supported_gpu_artifacts(
        source,
        function_name,
        param_types,
        gradient_name=gradient_name,
        include_prelude=include_prelude,
        syntax=syntax,
        kernel_name=kernel_name,
    )


def _source_gradient_request(
    source: str,
    *,
    include_prelude: bool,
    syntax: str,
) -> tuple[str, tuple[RemoraType, ...]]:
    program_source = with_prelude(source) if include_prelude and syntax == "ml" else source
    typed = TypeChecker().check_program(_parse_source(program_source, syntax))
    body = typed.body
    typed_grad = body.func if isinstance(body, TypedApp) else body
    if not isinstance(typed_grad, TypedGrad):
        raise ValueError("program body must be `(grad f)` or an application of it")
    if not isinstance(typed_grad.type, FuncType):
        raise ValueError("source-level gradient must be specialized to concrete parameter types")

    target = typed_grad.expr.func
    if isinstance(target, VarExpr):
        function_name = target.name
    elif isinstance(target, IndexAppExpr) and isinstance(target.func, VarExpr):
        function_name = target.func.name
    else:
        raise ValueError("source-level gradient must target a named function")
    return function_name, typed_grad.type.params


def _rewrite_applied_source_gradient(
    source: str,
    program: Program,
    typed: TypedProgram,
    *,
    include_prelude: bool,
    syntax: str,
) -> Program | None:
    """Replace an applied concrete `grad` body with a generated function call."""
    if not isinstance(typed.body, TypedApp) or not isinstance(typed.body.func, TypedGrad):
        if isinstance(typed.body, TypedGrad):
            raise ValueError(
                "bare `(grad f)` is a function value; use "
                "compile_source_gradient_function or apply it to an argument"
            )
        return None
    if not isinstance(program.body, AppExpr):
        raise AssertionError("typed gradient application must retain an AppExpr body")

    typed_grad = typed.body.func
    function_name = _typed_gradient_target_name(typed_grad)
    if not isinstance(typed_grad.type, FuncType):
        raise ValueError("source-level gradient must be specialized before application")

    existing_names = {
        definition.name
        for definition in program.definitions
        if isinstance(definition, FuncDef)
    }
    generated_name = _unique_gradient_name(function_name, existing_names)
    from remora.ad_source import generate_gradient_function_source

    gradient = generate_gradient_function_source(
        source,
        function_name,
        typed_grad.type.params,
        gradient_name=generated_name,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    generated_program = parse_lisp_program(gradient.source)
    if len(generated_program.definitions) != 1:
        raise AssertionError("generated gradient source must define one function")
    generated_definition = generated_program.definitions[0]
    if not isinstance(generated_definition, FuncDef):
        raise AssertionError("generated gradient source must define a function")

    replacement = AppExpr(
        VarExpr(generated_name, program.body.loc),
        list(program.body.args),
        program.body.loc,
    )
    return Program(
        [*program.definitions, generated_definition],
        replacement,
        program.loc,
    )


def _typed_gradient_target_name(typed_grad: TypedGrad) -> str:
    target = typed_grad.expr.func
    if isinstance(target, VarExpr):
        return target.name
    if isinstance(target, IndexAppExpr) and isinstance(target.func, VarExpr):
        return target.func.name
    raise ValueError("source-level gradient must target a named function")


def _unique_gradient_name(function_name: str, existing_names: set[str]) -> str:
    base = f"__remora_grad_{function_name.replace('-', '_')}"
    candidate = base
    suffix = 2
    while candidate in existing_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate
