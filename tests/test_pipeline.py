import importlib.util

import pytest

from remora.defunc import defunctionalize
from remora.hir import lower_to_hir
from remora.codegen import CodegenUnavailable, generate_ptx
from remora.lowering import MLIRLowering
from remora.parser import parse_program
from remora.pipeline import (
    CPU_PIPELINE,
    FUSION_PIPELINE,
    GPU_NVIDIA_SCAFFOLD_LLVM_DIALECT_PIPELINE,
    GPU_NVIDIA_SCAFFOLD_NVVM_PIPELINE,
    PipelineUnavailable,
    build_cpu_pipeline,
    build_pipeline,
    build_validation_pipeline,
    detect_toolchain,
    run_cpu_pipeline_text,
    run_fusion_pipeline_text,
    run_pipeline,
    run_validation_pipeline,
    translate_mlir_to_llvmir,
    verify_module_text,
    _strip_trivial_memref_alloca_scopes,
)
from remora.typechecker import TypeChecker


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("iree") is None,
    reason="IREE compiler MLIR bindings are not installed",
)


def lowered_module(source: str):
    typed = TypeChecker().check_program(parse_program(source))
    hir = defunctionalize(lower_to_hir(typed))
    return MLIRLowering().lower_program(hir).module


def test_detect_toolchain_reports_available_bindings_and_missing_external_verifier():
    toolchain = detect_toolchain()

    assert toolchain.iree_passmanager is True
    assert toolchain.has_external_verifier == (
        toolchain.mlir_opt is not None or toolchain.iree_opt is not None
    )
    assert toolchain.has_standalone_mlir == (
        toolchain.mlir_opt is not None and toolchain.mlir_translate is not None
    )
    assert toolchain.has_nvptx_codegen == (
        toolchain.mlir_translate is not None and toolchain.llc is not None
    )
    assert toolchain.has_ptx_toolchain == (
        toolchain.llc is not None and toolchain.ptxas is not None
    )


def test_validation_pipeline_runs_on_lowered_module():
    module = lowered_module("fold (+) 0.0 (map (* 2.0) (iota 10))")

    run_validation_pipeline(module)

    assert "func.func @main() -> f32" in str(module)


def test_build_validation_pipeline_and_run_pipeline_directly():
    module = lowered_module("map (* 2) (iota 4)")
    with module.context:
        pass_manager = build_validation_pipeline()
        run_pipeline(module, pass_manager)

    assert "func.func @main() -> tensor<4xi32>" in str(module)


def test_external_verifier_accepts_lowered_module_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_external_verifier:
        pytest.skip("no external MLIR verifier is available")

    module = lowered_module("map (* 2) (iota 4)")

    verify_module_text(str(module), toolchain)


def test_unavailable_pipeline_reports_clear_error():
    with pytest.raises(PipelineUnavailable, match="not available"):
        build_pipeline("builtin.module(remora-this-pass-does-not-exist)")


def test_cpu_pipeline_is_gated_until_toolchain_is_pinned():
    try:
        build_cpu_pipeline()
    except PipelineUnavailable as exc:
        assert "not available" in str(exc)
    else:
        pytest.fail("CPU pipeline unexpectedly parsed; update Phase 6 validation tests")


def test_standalone_cpu_pipeline_lowers_to_llvm_dialect_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_standalone_mlir:
        pytest.skip("standalone MLIR tools are not available")

    module = lowered_module("map (* 2.0) (iota 10)")

    lowered = run_cpu_pipeline_text(str(module), toolchain=toolchain)
    llvm_ir = translate_mlir_to_llvmir(lowered, toolchain=toolchain)

    assert "llvm.func @main" in lowered
    assert "linalg.generic" not in lowered
    assert "define {" in llvm_ir
    assert "@main" in llvm_ir


def test_standalone_fusion_pipeline_fuses_nested_scalar_map_when_available():
    toolchain = detect_toolchain()
    if toolchain.mlir_opt is None:
        pytest.skip("mlir-opt is not available")

    module = lowered_module("map (* 3) (map (* 2) (iota 10))")

    before = str(module)
    after = run_fusion_pipeline_text(before, toolchain=toolchain)

    assert before.count("linalg.generic") == 3
    assert after.count("linalg.generic") == 1
    assert "arith.muli" in after


def test_pipeline_artifacts_match_code_constants():
    assert "linalg-fuse-elementwise-ops" in FUSION_PIPELINE
    assert "convert-to-llvm" in CPU_PIPELINE
    assert "gpu.module(convert-gpu-to-nvvm" in GPU_NVIDIA_SCAFFOLD_NVVM_PIPELINE
    assert "convert-cf-to-llvm" in GPU_NVIDIA_SCAFFOLD_LLVM_DIALECT_PIPELINE


def test_threaded_pipeline_strips_only_trivial_alloca_scopes():
    text = """module {
  func.func @f() {
    memref.alloca_scope  {
      scf.for %i = %c0 to %c1 step %c1 {
        "test.use"() : () -> ()
      }
    }
    memref.alloca_scope  {
      %0 = memref.alloca() : memref<1xf32>
      "test.use"(%0) : (memref<1xf32>) -> ()
    }
    return
  }
}
"""

    stripped = _strip_trivial_memref_alloca_scopes(text)

    assert 'scf.for %i = %c0 to %c1 step %c1' in stripped
    assert stripped.count("memref.alloca_scope") == 1
    assert "%0 = memref.alloca()" in stripped


def test_generate_ptx_with_iree_cuda_backend_when_available():
    toolchain = detect_toolchain()
    if toolchain.iree_compile is None:
        pytest.skip("iree-compile is not available")

    module = lowered_module("map (* 2) (iota 4)")

    try:
        ptx, kernels = generate_ptx(module, toolchain=toolchain)
    except CodegenUnavailable as exc:
        pytest.skip(f"CUDA PTX generation is not available: {exc}")

    assert ".version" in ptx
    assert ".visible .entry" in ptx
    assert kernels
    assert kernels[0].name.startswith("main_dispatch_")
    assert kernels[0].block_size > 0
    assert kernels[0].num_inputs >= 1


def test_generate_ptx_for_phase6_milestone_expression_when_available():
    toolchain = detect_toolchain()
    if toolchain.iree_compile is None:
        pytest.skip("iree-compile is not available")

    module = lowered_module("fold (+) 0.0 (map (* 2.0) (iota 1000))")

    try:
        ptx, kernels = generate_ptx(module, toolchain=toolchain)
    except CodegenUnavailable as exc:
        pytest.skip(f"CUDA PTX generation is not available: {exc}")

    assert ".target sm_80" in ptx
    assert len(kernels) >= 2
    assert any("broadcast_1000_f32" in kernel.name for kernel in kernels)
    assert any("generic_1000_f32" in kernel.name for kernel in kernels)
