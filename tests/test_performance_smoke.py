from __future__ import annotations

import importlib.util
import json
import time

import pytest

from remora.benchmark import benchmark_source, main as benchmark_main
from remora.compiler import compile_source_to_mlir
from remora.pipeline import detect_toolchain, run_cpu_pipeline_text, run_fusion_pipeline_text


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("iree") is None,
    reason="IREE compiler MLIR bindings are not installed",
)


SMOKE_CASES = {
    "vector_scale": ("map (* 2.0) (iota 1000)", 2, 1, 5.0),
    "map_chain": ("map (* 3.0) (map (* 2.0) (iota 1000))", 3, 1, 5.0),
    "vector_sum": ("fold (+) 0.0 (iota 1000)", 2, 2, 5.0),
    "dot": (
        "let xs = [1.0, 2.0, 3.0] in "
        "let ys = [4.0, 5.0, 6.0] in "
        "dot xs ys",
        2,
        1,
        5.0,
    ),
}


def smoke_metrics(source: str) -> tuple[str, str, str, float, float]:
    toolchain = detect_toolchain()
    if toolchain.mlir_opt is None:
        pytest.skip("mlir-opt is not available")

    mlir = compile_source_to_mlir(source, verify=False)
    start = time.perf_counter()
    fused = run_fusion_pipeline_text(mlir, toolchain=toolchain)
    fusion_elapsed = time.perf_counter() - start
    start = time.perf_counter()
    lowered = run_cpu_pipeline_text(mlir, toolchain=toolchain)
    cpu_elapsed = time.perf_counter() - start
    return mlir, fused, lowered, fusion_elapsed, cpu_elapsed


@pytest.mark.parametrize(("name", "source_and_expected"), SMOKE_CASES.items())
def test_fused_linalg_operation_count_smoke(name: str, source_and_expected):
    source, before_count, after_count, _cpu_budget_s = source_and_expected
    mlir, fused, _lowered, _fusion_elapsed, _cpu_elapsed = smoke_metrics(source)

    assert mlir.count("linalg.generic") == before_count, name
    assert fused.count("linalg.generic") == after_count, name


@pytest.mark.parametrize(("name", "source_and_expected"), SMOKE_CASES.items())
def test_cpu_pipeline_compile_time_smoke(name: str, source_and_expected):
    source, _before_count, _after_count, cpu_budget_s = source_and_expected
    _mlir, _fused, lowered, _fusion_elapsed, elapsed = smoke_metrics(source)

    assert "llvm.func @main" in lowered, name
    assert "linalg.generic" not in lowered, name
    assert elapsed < cpu_budget_s, f"{name} CPU pipeline took {elapsed:.3f}s"


def test_threaded_cpu_pipeline_emits_openmp_for_parallel_map():
    source = "map (* 2) (iota 16)"
    mlir = compile_source_to_mlir(source, verify=False)
    lowered = run_cpu_pipeline_text(mlir, toolchain=detect_toolchain(), threaded=True)

    assert "omp.parallel" in lowered
    assert "omp.wsloop" in lowered


def test_benchmark_source_records_cpu_thread_request():
    result = benchmark_source("map (* 2) (iota 4)", name="tiny", cpu_threads=1)

    assert result.name == "tiny"
    assert result.cpu_threads == 1
    assert result.linalg_generic_before >= result.linalg_generic_after_fusion
    assert result.llvm_func_count >= 1
    assert result.allocation_count >= 0


def test_benchmark_cli_emits_json(tmp_path, capsys):
    source = tmp_path / "bench.remora"
    source.write_text("map (* 2) (iota 4)", encoding="utf-8")

    assert benchmark_main(["--cpu-threads", "1", str(source)]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "bench"
    assert payload["cpu_threads"] == 1
    assert "allocation_count" in payload
    assert captured.err == ""


def test_benchmark_baselines_cover_smoke_cases():
    from pathlib import Path

    payload = json.loads(Path("docs/BENCHMARK_BASELINES.json").read_text(encoding="utf-8"))
    names = {case["name"] for case in payload["cases"]}

    assert set(SMOKE_CASES) <= names
    for case in payload["cases"]:
        assert case["max_linalg_generic_after_fusion"] >= 0
        assert case["max_allocation_count"] >= 0
