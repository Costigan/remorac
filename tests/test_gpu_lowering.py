import importlib.util

import pytest

from remora.compiler import compile_function_source
from remora.gpu_lowering import (
    GPUScaffoldError,
    build_gpu_scaffold_for_function,
    build_rank1_f32_unary_map_gpu_scaffold,
)
from remora.pipeline import (
    PipelineUnavailable,
    detect_toolchain,
    run_gpu_nvidia_scaffold_llvm_dialect_pipeline_text,
    run_gpu_nvidia_scaffold_nvvm_pipeline_text,
    verify_module_text,
)
from remora.types import FLOAT, INT, ArrayType, StaticDim


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("iree") is None,
    reason="IREE compiler MLIR bindings are not installed",
)


def parse_mlir(text: str):
    from iree.compiler import ir

    context = ir.Context()
    context.allow_unregistered_dialects = True
    with context, ir.Location.unknown(context):
        return ir.Module.parse(text)


def test_rank1_f32_unary_map_gpu_scaffold_is_parseable_gpu_mlir():
    scaffold = build_rank1_f32_unary_map_gpu_scaffold(size=4)
    module = parse_mlir(scaffold.text)
    text = str(module)

    assert scaffold.module_name == "remora_gpu"
    assert scaffold.kernel_name == "remora_map_rank1_f32"
    assert "gpu.module @remora_gpu" in text
    assert "gpu.func @remora_map_rank1_f32" in text
    assert "memref<4xf32>" in text
    assert " kernel " in text
    assert "gpu.thread_id" in text
    assert "gpu.block_id" in text
    assert "gpu.block_dim" in text
    assert "arith.cmpi ult" in text
    assert "scf.if" in text
    assert "memref.load" in text
    assert "arith.mulf" in text
    assert "memref.store" in text
    assert "gpu.return" in text
    assert ".visible .entry" not in text


def test_rank1_f32_unary_map_gpu_scaffold_uses_requested_size_and_multiplier():
    scaffold = build_rank1_f32_unary_map_gpu_scaffold(size=7, multiplier=3.5)
    module = parse_mlir(scaffold.text)
    text = str(module)

    assert "memref<7xf32>" in text
    assert "arith.constant 7 : index" in text
    assert "arith.constant 3.500000e+00 : f32" in text


def test_rank1_f32_unary_map_gpu_scaffold_rejects_invalid_size():
    with pytest.raises(GPUScaffoldError, match="positive"):
        build_rank1_f32_unary_map_gpu_scaffold(size=0)


def test_builds_gpu_scaffold_from_rank1_f32_scale_hir_function():
    artifact = compile_function_source(
        "def scale xs = map (* 2.5) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        verify=False,
    )

    scaffold = build_gpu_scaffold_for_function(artifact.hir_function)
    module = parse_mlir(scaffold.text)
    text = str(module)

    assert scaffold.kernel_name == "remora_scale_rank1_f32"
    assert "gpu.func @remora_scale_rank1_f32" in text
    assert "memref<4xf32>" in text
    assert "arith.constant 2.500000e+00 : f32" in text
    assert "arith.mulf" in text


def test_gpu_scaffold_is_accepted_by_external_mlir_verifier_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_external_verifier:
        pytest.skip("no external MLIR verifier is available")

    scaffold = build_rank1_f32_unary_map_gpu_scaffold(size=4)

    verify_module_text(scaffold.text, toolchain)


def test_gpu_scaffold_runs_minimal_nested_nvvm_pipeline_when_available():
    toolchain = detect_toolchain()
    if toolchain.mlir_opt is None:
        pytest.skip("mlir-opt is not available")

    scaffold = build_rank1_f32_unary_map_gpu_scaffold(size=4)
    try:
        lowered = run_gpu_nvidia_scaffold_nvvm_pipeline_text(
            scaffold.text,
            toolchain=toolchain,
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"minimal scaffold NVVM pipeline is not available: {exc}")

    assert "llvm.func @remora_map_rank1_f32" in lowered
    assert "nvvm.kernel" in lowered
    assert "nvvm.read.ptx.sreg.tid.x" in lowered
    assert "llvm.fmul" in lowered


def test_gpu_scaffold_runs_scaffold_llvm_dialect_pipeline_when_available():
    toolchain = detect_toolchain()
    if toolchain.mlir_opt is None:
        pytest.skip("mlir-opt is not available")

    scaffold = build_rank1_f32_unary_map_gpu_scaffold(size=4)
    try:
        lowered = run_gpu_nvidia_scaffold_llvm_dialect_pipeline_text(
            scaffold.text,
            toolchain=toolchain,
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"scaffold LLVM dialect pipeline is not available: {exc}")

    assert "llvm.func @remora_map_rank1_f32" in lowered
    assert "nvvm.kernel" in lowered
    assert "nvvm.read.ptx.sreg.tid.x" in lowered
    assert "llvm.cond_br" in lowered
    assert "llvm.br" in lowered
    assert "llvm.fmul" in lowered
    assert "scf." not in lowered
    assert "cf." not in lowered
    assert "arith." not in lowered
    assert "memref." not in lowered


def test_gpu_scaffold_from_function_rejects_non_float_inputs():
    artifact = compile_function_source(
        "def scale xs = map (* 2) xs",
        "scale",
        (ArrayType(INT, (StaticDim(4),)),),
        verify=False,
    )

    with pytest.raises(GPUScaffoldError, match="rank-1 float inputs"):
        build_gpu_scaffold_for_function(artifact.hir_function)


def test_gpu_scaffold_from_function_rejects_non_scale_maps():
    artifact = compile_function_source(
        "def inc xs = map (+ 1.0) xs",
        "inc",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        verify=False,
    )

    with pytest.raises(GPUScaffoldError, match="scale maps"):
        build_gpu_scaffold_for_function(artifact.hir_function)
