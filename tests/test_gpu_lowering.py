import importlib.util

import pytest

from remora.gpu_lowering import (
    GPUScaffoldError,
    build_rank1_f32_unary_map_gpu_scaffold,
)


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
