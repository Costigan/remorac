"""Public compiler facade for Remora Dense Core."""

from __future__ import annotations

from dataclasses import dataclass

from remora.codegen import KernelMeta, generate_direct_remora_ptx, generate_ptx
from remora.defunc import defunctionalize
from remora.hir import HIRFunction, HIRParam, HIRProgram, lower_expr, lower_to_hir
from remora.lowering import MLIRLowering
from remora.parser import parse_program
from remora.pipeline import run_validation_pipeline, verify_module_text
from remora.prelude import with_prelude
from remora.typechecker import TypeChecker, TypeEnv, TypedProgram
from remora.types import FuncType, RemoraType
from remora.ast_nodes import FuncDef


@dataclass(frozen=True)
class CompilerArtifact:
    source: str
    typed: TypedProgram
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

    @property
    def return_type(self) -> RemoraType:
        return self.function_type.result


@dataclass(frozen=True)
class PTXArtifact:
    compiler: CompilerArtifact
    ptx_text: str
    kernels: list[KernelMeta]


def compile_source(
    source: str,
    *,
    verify: bool = True,
    include_prelude: bool = True,
    export_output_descriptor: bool = False,
) -> CompilerArtifact:
    program_source = with_prelude(source) if include_prelude else source
    typed = TypeChecker().check_program(parse_program(program_source))
    hir = defunctionalize(lower_to_hir(typed))
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
        hir=hir,
        mlir_module=mlir_module,
        mlir_text=str(mlir_module),
    )


def compile_source_to_mlir(
    source: str,
    *,
    verify: bool = True,
    include_prelude: bool = True,
) -> str:
    return compile_source(
        source,
        verify=verify,
        include_prelude=include_prelude,
        export_output_descriptor=False,
    ).mlir_text


def compile_source_to_ptx(
    source: str,
    *,
    verify: bool = True,
    include_prelude: bool = True,
) -> PTXArtifact:
    artifact = compile_source(
        source,
        verify=verify,
        include_prelude=include_prelude,
        export_output_descriptor=False,
    )
    ptx_text, kernels = generate_ptx(artifact.mlir_module)
    return PTXArtifact(artifact, ptx_text, kernels)


def compile_function_source_to_direct_ptx(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    include_prelude: bool = True,
    kernel_name: str | None = None,
) -> tuple[str, list[KernelMeta], FunctionCompilerArtifact]:
    """Compile a named function to direct Remora ABI PTX for supported GPU slices."""
    artifact = compile_function_source(
        source,
        function_name,
        param_types,
        verify=False,
        include_prelude=include_prelude,
    )
    ptx, kernels = generate_direct_remora_ptx(
        artifact.hir_function,
        kernel_name=kernel_name,
    )
    return ptx, kernels, artifact


def compile_function_source(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    verify: bool = True,
    include_prelude: bool = True,
    export_name: str = "remora_call",
) -> FunctionCompilerArtifact:
    """Compile one top-level function with explicit static parameter types."""
    program_source = with_prelude(source) if include_prelude else source
    program = parse_program(program_source)
    checker = TypeChecker()
    env = TypeEnv()
    function_def: FuncDef | None = None
    for definition in program.definitions:
        typed_definition, env = checker._check_definition(definition, env)
        if isinstance(definition, FuncDef) and definition.name == function_name:
            function_def = definition

    if function_def is None:
        raise ValueError(f"function {function_name!r} is not defined")

    function_type = checker._infer_top_level_function_type(function_def, param_types, env)
    typed_function = checker._typed_top_level_function(function_def, function_type, env)
    hir_function = HIRFunction(
        function_name,
        [HIRParam(name, param_type) for name, param_type in typed_function.params],
        lower_expr(typed_function.body),
        function_type.result,
    )
    lowered = MLIRLowering().lower_function_descriptor_export(
        hir_function,
        export_name=export_name,
    )
    if verify:
        run_validation_pipeline(lowered.module)
        verify_module_text(str(lowered.module))
    return FunctionCompilerArtifact(
        source=source,
        function_name=function_name,
        function_type=function_type,
        hir_function=hir_function,
        mlir_module=lowered.module,
        mlir_text=str(lowered.module),
    )
