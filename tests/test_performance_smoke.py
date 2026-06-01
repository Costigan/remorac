from __future__ import annotations

import importlib.util
import time

import pytest

from remora.compiler import compile_source_to_mlir
from remora.pipeline import detect_toolchain, run_cpu_pipeline_text, run_fusion_pipeline_text


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("iree") is None,
    reason="IREE compiler MLIR bindings are not installed",
)


SMOKE_CASES = {
    "vector_scale": ("map (* 2.0) (iota 1000)", 1),
    "map_chain": ("map (* 3.0) (map (* 2.0) (iota 1000))", 1),
    "vector_sum": ("fold (+) 0.0 (iota 1000)", 2),
    "dot": (
        "let xs = [1.0, 2.0, 3.0] in "
        "let ys = [4.0, 5.0, 6.0] in "
        "dot xs ys",
        1,
    ),
}


@pytest.mark.parametrize(("name", "source_and_expected"), SMOKE_CASES.items())
def test_fused_linalg_operation_count_smoke(name: str, source_and_expected):
    source, expected_count = source_and_expected
    toolchain = detect_toolchain()
    if toolchain.mlir_opt is None:
        pytest.skip("mlir-opt is not available")

    mlir = compile_source_to_mlir(source, verify=False)
    fused = run_fusion_pipeline_text(mlir, toolchain=toolchain)

    assert fused.count("linalg.generic") == expected_count, name


@pytest.mark.parametrize(("name", "source_and_expected"), SMOKE_CASES.items())
def test_cpu_pipeline_compile_time_smoke(name: str, source_and_expected):
    source, _expected_count = source_and_expected
    toolchain = detect_toolchain()
    if not toolchain.has_standalone_mlir:
        pytest.skip("standalone MLIR tools are not available")

    mlir = compile_source_to_mlir(source, verify=False)
    start = time.perf_counter()
    lowered = run_cpu_pipeline_text(mlir, toolchain=toolchain)
    elapsed = time.perf_counter() - start

    assert "llvm.func @main" in lowered, name
    assert "linalg.generic" not in lowered, name
    assert elapsed < 5.0, f"{name} CPU pipeline took {elapsed:.3f}s"
