"""Public compiler facade for Remora Dense Core."""

from __future__ import annotations

from dataclasses import dataclass

from remora.codegen import KernelMeta, generate_ptx
from remora.defunc import defunctionalize
from remora.hir import HIRProgram, lower_to_hir
from remora.lowering import MLIRLowering
from remora.parser import parse_program
from remora.pipeline import run_validation_pipeline, verify_module_text
from remora.typechecker import TypeChecker, TypedProgram
from remora.types import RemoraType


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
class PTXArtifact:
    compiler: CompilerArtifact
    ptx_text: str
    kernels: list[KernelMeta]


def compile_source(source: str, *, verify: bool = True) -> CompilerArtifact:
    typed = TypeChecker().check_program(parse_program(source))
    hir = defunctionalize(lower_to_hir(typed))
    mlir_module = MLIRLowering().lower_program(hir).module
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


def compile_source_to_mlir(source: str, *, verify: bool = True) -> str:
    return compile_source(source, verify=verify).mlir_text


def compile_source_to_ptx(source: str, *, verify: bool = True) -> PTXArtifact:
    artifact = compile_source(source, verify=verify)
    ptx_text, kernels = generate_ptx(artifact.mlir_module)
    return PTXArtifact(artifact, ptx_text, kernels)
