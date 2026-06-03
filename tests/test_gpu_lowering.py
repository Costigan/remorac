import importlib.util
import re

import pytest

from remora.compiler import (
    compile_function_source,
    compile_function_source_to_mlir_gpu_ptx,
    compile_function_source_to_rank1_mlir_gpu_ptx,
    compile_function_source_to_supported_gpu_artifacts,
)
from remora.gpu_lowering import (
    GPUScaffoldError,
    build_descriptor_abi_f32_reduction_gpu_module,
    build_descriptor_abi_i32_map_gpu_module,
    build_f32_binary_map_gpu_scaffold,
    build_f32_unary_map_gpu_scaffold,
    build_gpu_scaffold_for_function,
    build_rank1_f32_unary_map_gpu_scaffold,
    extract_gpu_module_body_as_module,
)
from remora.hir import HIRFold, HIRFunction, HIRLit, HIRMap, HIRParam, HIRPrimCallable, HIRVar
from remora.pipeline import (
    PipelineUnavailable,
    assemble_ptx_text,
    detect_toolchain,
    lower_gpu_scaffold_to_nvptx_text,
    run_gpu_nvidia_scaffold_llvm_dialect_pipeline_text,
    run_gpu_nvidia_scaffold_nvvm_pipeline_text,
    translate_mlir_to_llvmir,
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
    try:
        from iree.compiler.dialects import nvvm
    except ImportError:
        pass

    try:
        with context, ir.Location.unknown(context):
            return ir.Module.parse(text)
    except Exception as exc:
        # Fallback to standalone mlir-opt if bindings fail (e.g. missing nvvm)
        toolchain = detect_toolchain()
        if toolchain.mlir_opt is not None:
             import subprocess
             result = subprocess.run(
                 [toolchain.mlir_opt],
                 input=text,
                 capture_output=True,
                 text=True,
                 check=False,
             )
             if result.returncode == 0:
                 return None # Parse success
        raise exc


def ptx_param_count(ptx_text: str, kernel_name: str) -> int:
    match = re.search(
        rf"\.visible\s+\.entry\s+{re.escape(kernel_name)}\((.*?)\)\s*(?://.*)?\n",
        ptx_text,
        flags=re.DOTALL,
    )
    assert match is not None
    return len(re.findall(r"\.param\s+\.\w+\s+[A-Za-z_.$][\w.$]*", match.group(1)))


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


def test_rank1_f32_reduction_descriptor_abi_module_is_parseable():
    function = HIRFunction(
        "sum",
        [HIRParam("xs", ArrayType(FLOAT, (StaticDim(4),)))],
        HIRFold(
            StaticDim(4),
            HIRPrimCallable("+", (FLOAT, FLOAT), FLOAT),
            HIRLit(0.0, FLOAT),
            HIRVar("xs", ArrayType(FLOAT, (StaticDim(4),))),
            FLOAT,
        ),
        FLOAT,
    )

    module = build_descriptor_abi_f32_reduction_gpu_module(function, kernel_name="remora_sum")
    parse_mlir(module.text)
    text = module.text

    assert "llvm.func @remora_sum(%input0_desc: !llvm.ptr, %output_desc: !llvm.ptr)" in text
    assert "nvvm.read.ptx.sreg.tid.x" in text
    assert "llvm.atomicrmw fadd" in text
    assert "nvvm.barrier0" in text


def test_rank1_i32_descriptor_abi_map_module_uses_integer_ops():
    array_type = ArrayType(INT, (StaticDim(4),))
    function = HIRFunction(
        "inc",
        [HIRParam("xs", array_type)],
        HIRMap(
            (StaticDim(4),),
            (),
            HIRPrimCallable("+", (INT, INT), INT, right_arg=HIRLit(2, INT)),
            [HIRVar("xs", array_type)],
            array_type,
        ),
        array_type,
    )

    module = build_descriptor_abi_i32_map_gpu_module(function, kernel_name="remora_inc")
    text = module.text

    assert "llvm.func @remora_inc(%input0_desc: !llvm.ptr, %output_desc: !llvm.ptr)" in text
    assert "%c = llvm.mlir.constant(2 : i32) : i32" in text
    assert "%y = llvm.add %x0, %c  : i32" in text
    assert "llvm.store %y, %out_elem_ptr : i32, !llvm.ptr" in text


def test_rank1_f32_unary_map_gpu_scaffold_uses_requested_size_and_multiplier():
    scaffold = build_rank1_f32_unary_map_gpu_scaffold(size=7, multiplier=3.5)
    module = parse_mlir(scaffold.text)
    text = str(module)

    assert "memref<7xf32>" in text
    assert "arith.constant 7 : index" in text
    assert "arith.constant 3.500000e+00 : f32" in text


def test_rank2_f32_unary_map_gpu_scaffold_uses_product_size_and_multi_indices():
    scaffold = build_f32_unary_map_gpu_scaffold(
        shape=(2, 3),
        operation="*",
        constant=2.0,
        kernel_name="remora_scale2d",
    )
    parse_mlir(scaffold.text)
    text = scaffold.text

    assert "gpu.func @remora_scale2d" in text
    assert "memref<2x3xf32>" in text
    assert "arith.constant 6 : index" in text
    assert "arith.divui %idx, %dim1 : index" in text
    assert "arith.remui %idx, %dim1 : index" in text
    assert "memref.load %input0[%i0, %i1]" in text
    assert "memref.store %y, %output[%i0, %i1]" in text


def test_rank3_f32_unary_map_gpu_scaffold_supports_left_section_division():
    scaffold = build_f32_unary_map_gpu_scaffold(
        shape=(2, 3, 4),
        operation="/",
        constant=3.0,
        constant_side="left",
        kernel_name="remora_inv3d",
    )
    parse_mlir(scaffold.text)
    text = scaffold.text

    assert "gpu.func @remora_inv3d" in text
    assert "memref<2x3x4xf32>" in text
    assert "arith.constant 24 : index" in text
    assert "arith.divui %idx, %plane : index" in text
    assert "arith.remui %rem0, %dim2 : index" in text
    assert "%y = arith.divf %c, %x0 : f32" in text
    assert "memref.store %y, %output[%i0, %i1, %i2]" in text


def test_rank1_through_rank3_binary_map_gpu_scaffolds_are_parseable():
    for shape, kernel_name in [
        ((4,), "remora_add1d"),
        ((2, 3), "remora_add2d"),
        ((2, 3, 4), "remora_add3d"),
    ]:
        scaffold = build_f32_binary_map_gpu_scaffold(
            shape=shape,
            operation="+",
            kernel_name=kernel_name,
        )
        parse_mlir(scaffold.text)
        text = scaffold.text

        dims = "x".join(str(dim) for dim in shape)
        assert f"gpu.func @{kernel_name}" in text
        assert f"memref<{dims}xf32>" in text
        assert "memref.load %input0" in text
        assert "memref.load %input1" in text
        assert "%y = arith.addf %x0, %x1 : f32" in text


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

    assert scaffold.kernel_name == "remora_scale_f32"
    assert "gpu.func @remora_scale_f32" in text
    assert "memref<4xf32>" in text
    assert "arith.constant 2.500000e+00 : f32" in text
    assert "arith.mulf" in text


def test_builds_gpu_scaffold_from_rank2_and_rank3_hir_functions():
    rank2_artifact = compile_function_source(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),),
        verify=False,
    )
    rank3_artifact = compile_function_source(
        "def div3 xs = map (3.0 /) xs",
        "div3",
        (ArrayType(FLOAT, (StaticDim(2), StaticDim(2), StaticDim(1))),),
        verify=False,
    )

    rank2_text = build_gpu_scaffold_for_function(rank2_artifact.hir_function).text
    rank3_text = build_gpu_scaffold_for_function(rank3_artifact.hir_function).text
    parse_mlir(rank2_text)
    parse_mlir(rank3_text)

    assert "memref<2x3xf32>" in rank2_text
    assert "arith.mulf" in rank2_text
    assert "memref<2x2x1xf32>" in rank3_text
    assert "arith.divf %x0, %c : f32" in rank3_text


def test_builds_binary_gpu_scaffold_from_hir_function():
    artifact = compile_function_source(
        "def add xs ys = map (+) xs ys",
        "add",
        (
            ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
            ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
        ),
        verify=False,
    )

    scaffold = build_gpu_scaffold_for_function(artifact.hir_function)
    parse_mlir(scaffold.text)
    text = scaffold.text

    assert "gpu.func @remora_add_f32" in text
    assert "memref<2x3xf32>" in text
    assert "memref.load %input1[%i0, %i1]" in text
    assert "%y = arith.addf %x0, %x1 : f32" in text


def test_supported_gpu_artifacts_prefer_mlir_descriptor_abi_ptx():
    artifact = compile_function_source_to_supported_gpu_artifacts(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        kernel_name="remora_scale",
    )

    parse_mlir(artifact.scaffold.text)

    assert "gpu.func @remora_scale" in artifact.scaffold.text
    assert ".visible .entry remora_scale" in artifact.ptx_text
    assert ".param .u64 remora_scale_param_0" in artifact.ptx_text
    assert ".param .u64 remora_scale_param_1" in artifact.ptx_text
    assert artifact.kernels[0].name == "remora_scale"
    assert artifact.kernels[0].num_inputs == 1
    assert artifact.kernels[0].output_shape == (4,)


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


def test_rank2_gpu_scaffold_runs_scaffold_llvm_dialect_pipeline_when_available():
    toolchain = detect_toolchain()
    if toolchain.mlir_opt is None:
        pytest.skip("mlir-opt is not available")

    scaffold = build_f32_unary_map_gpu_scaffold(
        shape=(2, 3),
        operation="*",
        constant=2.0,
        kernel_name="remora_scale2d",
    )
    try:
        lowered = run_gpu_nvidia_scaffold_llvm_dialect_pipeline_text(
            scaffold.text,
            toolchain=toolchain,
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"scaffold LLVM dialect pipeline is not available: {exc}")

    assert "llvm.func @remora_scale2d" in lowered
    assert "nvvm.kernel" in lowered
    assert "llvm.fmul" in lowered


def test_binary_gpu_scaffold_runs_scaffold_llvm_dialect_pipeline_when_available():
    toolchain = detect_toolchain()
    if toolchain.mlir_opt is None:
        pytest.skip("mlir-opt is not available")

    scaffold = build_f32_binary_map_gpu_scaffold(
        shape=(4,),
        operation="+",
        kernel_name="remora_add1d",
    )
    try:
        lowered = run_gpu_nvidia_scaffold_llvm_dialect_pipeline_text(
            scaffold.text,
            toolchain=toolchain,
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"scaffold LLVM dialect pipeline is not available: {exc}")

    assert "llvm.func @remora_add1d" in lowered
    assert "nvvm.kernel" in lowered
    assert "llvm.fadd" in lowered


def test_extracts_converted_gpu_module_for_device_translation():
    toolchain = detect_toolchain()
    if toolchain.mlir_opt is None:
        pytest.skip("mlir-opt is not available")

    scaffold = build_rank1_f32_unary_map_gpu_scaffold(size=4)
    lowered = run_gpu_nvidia_scaffold_llvm_dialect_pipeline_text(
        scaffold.text,
        toolchain=toolchain,
    )

    device_module = extract_gpu_module_body_as_module(lowered)

    assert "module {" in device_module
    assert "gpu.module" not in device_module
    assert "llvm.func @remora_map_rank1_f32" in device_module
    assert "nvvm.kernel" in device_module


def test_extracted_gpu_module_translates_to_nonempty_llvm_ir_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_standalone_mlir:
        pytest.skip("standalone MLIR tools are not available")

    scaffold = build_rank1_f32_unary_map_gpu_scaffold(size=4)
    lowered = run_gpu_nvidia_scaffold_llvm_dialect_pipeline_text(
        scaffold.text,
        toolchain=toolchain,
    )
    device_module = extract_gpu_module_body_as_module(lowered)

    llvm_ir = translate_mlir_to_llvmir(device_module, toolchain=toolchain)

    assert "define void @remora_map_rank1_f32" in llvm_ir
    assert "llvm.nvvm.read.ptx.sreg.tid.x" in llvm_ir
    assert "fmul float" in llvm_ir
    assert "ret void" in llvm_ir


def test_rank1_gpu_scaffold_compiles_to_nvptx_text_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_nvptx_codegen:
        pytest.skip("standalone NVPTX text tools are not available")

    scaffold = build_rank1_f32_unary_map_gpu_scaffold(size=4)
    try:
        ptx = lower_gpu_scaffold_to_nvptx_text(scaffold.text, toolchain=toolchain)
    except PipelineUnavailable as exc:
        pytest.skip(f"standalone NVPTX text generation is not available: {exc}")

    assert ".version" in ptx
    assert ".target sm_80" in ptx
    assert ".address_size 64" in ptx
    assert ".visible .entry remora_map_rank1_f32" in ptx
    assert "input_desc_param" not in ptx
    assert ".maxntid" not in ptx
    assert ptx_param_count(ptx, "remora_map_rank1_f32") == 10
    assert "ret;" in ptx


def test_rank2_gpu_scaffold_compiles_to_nvptx_text_with_exploded_memref_abi_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_nvptx_codegen:
        pytest.skip("standalone NVPTX text tools are not available")

    scaffold = build_f32_unary_map_gpu_scaffold(
        shape=(2, 3),
        operation="*",
        constant=2.0,
        kernel_name="remora_scale2d",
    )
    try:
        ptx = lower_gpu_scaffold_to_nvptx_text(scaffold.text, toolchain=toolchain)
    except PipelineUnavailable as exc:
        pytest.skip(f"standalone NVPTX text generation is not available: {exc}")

    assert ".visible .entry remora_scale2d" in ptx
    assert ".target sm_80" in ptx
    assert "input_desc_param" not in ptx
    assert ptx_param_count(ptx, "remora_scale2d") == 14
    assert "ld.param.u64" in ptx
    assert "mul.wide.u32" in ptx or "mul.wide.s32" in ptx


def test_supported_gpu_artifacts_document_scaffold_vs_descriptor_abi_boundary_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_nvptx_codegen:
        pytest.skip("standalone NVPTX text tools are not available")

    artifact = compile_function_source_to_supported_gpu_artifacts(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        kernel_name="remora_scale",
    )
    try:
        scaffold_ptx = lower_gpu_scaffold_to_nvptx_text(
            artifact.scaffold.text,
            toolchain=toolchain,
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"standalone NVPTX text generation is not available: {exc}")

    assert ptx_param_count(scaffold_ptx, "remora_scale") == 10
    assert ".param .u64 remora_scale_param_0" in artifact.ptx_text
    assert ".param .u64 remora_scale_param_1" in artifact.ptx_text
    assert ptx_param_count(artifact.ptx_text, "remora_scale") == 2


def test_rank1_mlir_gpu_ptx_exports_descriptor_abi_kernel_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_nvptx_codegen:
        pytest.skip("standalone NVPTX text tools are not available")

    try:
        ptx, kernels, artifact = compile_function_source_to_rank1_mlir_gpu_ptx(
            "def scale xs = map (* 2.0) xs",
            "scale",
            (ArrayType(FLOAT, (StaticDim(4),)),),
            kernel_name="remora_scale",
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"standalone NVPTX text generation is not available: {exc}")

    assert artifact.function_name == "scale"
    assert ".visible .entry remora_scale" in ptx
    assert ptx_param_count(ptx, "remora_scale") == 2
    assert kernels[0].name == "remora_scale"
    assert kernels[0].num_inputs == 1
    assert kernels[0].output_shape == (4,)


def test_rank1_binary_mlir_gpu_ptx_exports_descriptor_abi_kernel_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_nvptx_codegen:
        pytest.skip("standalone NVPTX text tools are not available")

    try:
        ptx, kernels, artifact = compile_function_source_to_rank1_mlir_gpu_ptx(
            "def add xs ys = map (+) xs ys",
            "add",
            (
                ArrayType(FLOAT, (StaticDim(4),)),
                ArrayType(FLOAT, (StaticDim(4),)),
            ),
            kernel_name="remora_add",
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"standalone NVPTX text generation is not available: {exc}")

    assert artifact.function_name == "add"
    assert ".visible .entry remora_add" in ptx
    assert ptx_param_count(ptx, "remora_add") == 3
    assert kernels[0].name == "remora_add"
    assert kernels[0].num_inputs == 2
    assert kernels[0].output_shape == (4,)


def test_rank2_unary_mlir_gpu_ptx_exports_descriptor_abi_kernel_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_nvptx_codegen:
        pytest.skip("standalone NVPTX text tools are not available")

    try:
        ptx, kernels, artifact = compile_function_source_to_mlir_gpu_ptx(
            "def scale xs = map (* 2.0) xs",
            "scale",
            (ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),),
            kernel_name="remora_scale2d",
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"standalone NVPTX text generation is not available: {exc}")

    assert artifact.function_name == "scale"
    assert ".visible .entry remora_scale2d" in ptx
    assert ptx_param_count(ptx, "remora_scale2d") == 2
    assert kernels[0].name == "remora_scale2d"
    assert kernels[0].num_inputs == 1
    assert kernels[0].output_shape == (2, 3)


def test_rank2_binary_mlir_gpu_ptx_exports_descriptor_abi_kernel_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_nvptx_codegen:
        pytest.skip("standalone NVPTX text tools are not available")

    try:
        ptx, kernels, artifact = compile_function_source_to_mlir_gpu_ptx(
            "def add xs ys = map (+) xs ys",
            "add",
            (
                ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
                ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
            ),
            kernel_name="remora_add2d",
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"standalone NVPTX text generation is not available: {exc}")

    assert artifact.function_name == "add"
    assert ".visible .entry remora_add2d" in ptx
    assert ptx_param_count(ptx, "remora_add2d") == 3
    assert kernels[0].name == "remora_add2d"
    assert kernels[0].num_inputs == 2
    assert kernels[0].output_shape == (2, 3)


def test_rank1_sum_mlir_gpu_ptx_exports_scalar_descriptor_abi_kernel_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_nvptx_codegen:
        pytest.skip("standalone NVPTX text tools are not available")

    try:
        ptx, kernels, artifact = compile_function_source_to_mlir_gpu_ptx(
            "def sum xs = fold (+) 0.0 xs",
            "sum",
            (ArrayType(FLOAT, (StaticDim(4),)),),
            include_prelude=False,
            kernel_name="remora_sum",
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"standalone NVPTX text generation is not available: {exc}")

    assert artifact.function_name == "sum"
    assert ".visible .entry remora_sum" in ptx
    assert ptx_param_count(ptx, "remora_sum") == 2
    assert kernels[0].name == "remora_sum"
    assert kernels[0].num_inputs == 1
    assert kernels[0].output_shape == ()
    assert kernels[0].output_dtype == "float32"


def test_rank1_dot_mlir_gpu_ptx_exports_scalar_descriptor_abi_kernel_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_nvptx_codegen:
        pytest.skip("standalone NVPTX text tools are not available")

    try:
        ptx, kernels, artifact = compile_function_source_to_mlir_gpu_ptx(
            "def dot xs ys = fold (+) 0.0 (map (*) xs ys)",
            "dot",
            (
                ArrayType(FLOAT, (StaticDim(4),)),
                ArrayType(FLOAT, (StaticDim(4),)),
            ),
            include_prelude=False,
            kernel_name="remora_dot",
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"standalone NVPTX text generation is not available: {exc}")

    assert artifact.function_name == "dot"
    assert ".visible .entry remora_dot" in ptx
    assert ptx_param_count(ptx, "remora_dot") == 3
    assert kernels[0].name == "remora_dot"
    assert kernels[0].num_inputs == 2
    assert kernels[0].output_shape == ()
    assert kernels[0].output_dtype == "float32"


def test_rank1_i32_unary_mlir_gpu_ptx_exports_descriptor_abi_kernel_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_nvptx_codegen:
        pytest.skip("standalone NVPTX text tools are not available")

    try:
        ptx, kernels, artifact = compile_function_source_to_mlir_gpu_ptx(
            "def inc xs = map (+ 2) xs",
            "inc",
            (ArrayType(INT, (StaticDim(4),)),),
            include_prelude=False,
            kernel_name="remora_inc",
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"standalone NVPTX text generation is not available: {exc}")

    assert artifact.function_name == "inc"
    assert ".visible .entry remora_inc" in ptx
    assert ptx_param_count(ptx, "remora_inc") == 2
    assert kernels[0].name == "remora_inc"
    assert kernels[0].num_inputs == 1
    assert kernels[0].output_shape == (4,)
    assert kernels[0].output_dtype == "int32"


def test_rank1_i32_binary_mlir_gpu_ptx_exports_descriptor_abi_kernel_when_available():
    toolchain = detect_toolchain()
    if not toolchain.has_nvptx_codegen:
        pytest.skip("standalone NVPTX text tools are not available")

    try:
        ptx, kernels, artifact = compile_function_source_to_mlir_gpu_ptx(
            "def add xs ys = map (+) xs ys",
            "add",
            (
                ArrayType(INT, (StaticDim(4),)),
                ArrayType(INT, (StaticDim(4),)),
            ),
            include_prelude=False,
            kernel_name="remora_iadd",
        )
    except PipelineUnavailable as exc:
        pytest.skip(f"standalone NVPTX text generation is not available: {exc}")

    assert artifact.function_name == "add"
    assert ".visible .entry remora_iadd" in ptx
    assert ptx_param_count(ptx, "remora_iadd") == 3
    assert kernels[0].name == "remora_iadd"
    assert kernels[0].num_inputs == 2
    assert kernels[0].output_shape == (4,)
    assert kernels[0].output_dtype == "int32"


def test_mlir_gpu_ptx_assembles_with_ptxas_when_available():
    toolchain = detect_toolchain()
    if toolchain.ptxas is None:
        pytest.skip("ptxas is not available")

    try:
        ptx, _kernels, _artifact = compile_function_source_to_mlir_gpu_ptx(
            "def scale xs = map (* 2.0) xs",
            "scale",
            (ArrayType(FLOAT, (StaticDim(4),)),),
            kernel_name="remora_scale",
        )
        cubin = assemble_ptx_text(ptx, toolchain=toolchain)
    except PipelineUnavailable as exc:
        pytest.skip(f"PTX assembly validation is not available: {exc}")

    assert cubin


def test_extract_gpu_module_reports_missing_module():
    with pytest.raises(GPUScaffoldError, match="was not found"):
        extract_gpu_module_body_as_module("module {}")


def test_gpu_scaffold_from_function_rejects_non_float_inputs():
    artifact = compile_function_source(
        "def scale xs = map (* 2) xs",
        "scale",
        (ArrayType(INT, (StaticDim(4),)),),
        verify=False,
    )

    with pytest.raises(GPUScaffoldError, match="rank-1 through rank-10 float inputs"):

        build_gpu_scaffold_for_function(artifact.hir_function)


def test_gpu_scaffold_from_function_rejects_nonliteral_unary_maps():
    function = HIRFunction(
        "addc",
        [
            HIRParam("xs", ArrayType(FLOAT, (StaticDim(4),))),
        ],
        HIRMap(
            frame_shape=(StaticDim(4),),
            cell_shape=(),
            func=HIRPrimCallable("+", (FLOAT,), FLOAT, right_arg=HIRVar("c", FLOAT)),
            arrays=[HIRVar("xs", ArrayType(FLOAT, (StaticDim(4),)))],
            result_type=ArrayType(FLOAT, (StaticDim(4),)),
        ),
        ArrayType(FLOAT, (StaticDim(4),)),
    )

    with pytest.raises(GPUScaffoldError, match="unary map requires a literal float section"):
        build_gpu_scaffold_for_function(function)


def test_gpu_scaffold_from_function_rejects_binary_operator_sections():
    function = HIRFunction(
        "addc",
        [
            HIRParam("xs", ArrayType(FLOAT, (StaticDim(4),))),
            HIRParam("ys", ArrayType(FLOAT, (StaticDim(4),))),
        ],
        HIRMap(
            frame_shape=(StaticDim(4),),
            cell_shape=(),
            func=HIRPrimCallable("+", (FLOAT, FLOAT), FLOAT, right_arg=HIRLit(1.0, FLOAT)),
            arrays=[
                HIRVar("xs", ArrayType(FLOAT, (StaticDim(4),))),
                HIRVar("ys", ArrayType(FLOAT, (StaticDim(4),))),
            ],
            result_type=ArrayType(FLOAT, (StaticDim(4),)),
        ),
        ArrayType(FLOAT, (StaticDim(4),)),
    )

    with pytest.raises(GPUScaffoldError, match="binary map does not support operator sections"):
        build_gpu_scaffold_for_function(function)


def test_gpu_scaffold_rejects_rank_above_ten_and_zero_dimensions():
    with pytest.raises(GPUScaffoldError, match="rank-1 through rank-10"):
        build_f32_unary_map_gpu_scaffold(shape=(1,) * 11, operation="*", constant=2.0)
    with pytest.raises(GPUScaffoldError, match="positive"):
        build_f32_binary_map_gpu_scaffold(shape=(2, 0), operation="+")


def test_gpu_scaffold_rejects_manual_mismatched_binary_shapes():
    function = HIRFunction(
        "bad_add",
        [
            HIRParam("xs", ArrayType(FLOAT, (StaticDim(2), StaticDim(3)))),
            HIRParam("ys", ArrayType(FLOAT, (StaticDim(3), StaticDim(2)))),
        ],
        HIRMap(
            frame_shape=(StaticDim(2), StaticDim(3)),
            cell_shape=(),
            func=HIRPrimCallable("+", (FLOAT, FLOAT), FLOAT),
            arrays=[HIRVar("xs", ArrayType(FLOAT, (StaticDim(2), StaticDim(3)))), HIRVar("ys", ArrayType(FLOAT, (StaticDim(3), StaticDim(2))))],
            result_type=ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
        ),
        ArrayType(FLOAT, (StaticDim(2), StaticDim(3))),
    )

    with pytest.raises(GPUScaffoldError, match="input and output shapes must match"):
        build_gpu_scaffold_for_function(function)
