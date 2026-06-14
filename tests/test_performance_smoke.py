from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import time

import pytest

from remora.benchmark import (
    BASELINE_SOURCES,
    benchmark_function_compilation,
    benchmark_source,
    check_result_against_baseline,
    diagnose_cpu_pipeline_stages,
    main as benchmark_main,
)
import remora.benchmark as benchmark_module
from remora.compiler import compile_function_source, compile_source_to_mlir
from remora.pipeline import detect_toolchain, run_cpu_pipeline_text, run_fusion_pipeline_text
from remora.types import ArrayType, FLOAT, StaticDim


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("iree") is None,
    reason="IREE compiler MLIR bindings are not installed",
)


# (source, before_count, after_count, cpu_budget_s)
SMOKE_CASES = {
    "vector_scale": (BASELINE_SOURCES["vector_scale"], 2, 1, 5.0),
    "map_chain": (BASELINE_SOURCES["map_chain"], 3, 1, 5.0),
    "vector_sum": (BASELINE_SOURCES["vector_sum"], 2, 2, 5.0),
    "dot": (BASELINE_SOURCES["dot"], 2, 1, 5.0),
    "row_reduce": (BASELINE_SOURCES["row_reduce"], 1, 1, 5.0),
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


def test_threaded_cpu_pipeline_lowers_row_reduction():
    source = "let xs = [[1.0, 2.0], [3.0, 4.0]] in map (\\row -> fold (+) 0.0 row) xs"
    mlir = compile_source_to_mlir(source, verify=False)
    lowered = run_cpu_pipeline_text(mlir, toolchain=detect_toolchain(), threaded=True)

    assert "llvm.func @main" in lowered
    assert "linalg.generic" not in lowered
    assert "memref.alloca_scope" not in lowered
    assert "omp.wsloop" in lowered


def test_benchmark_source_records_cpu_thread_request():
    result = benchmark_source("map (* 2) (iota 4)", name="tiny", cpu_threads=1)

    assert result.name == "tiny"
    assert result.cpu_threads == 1
    assert result.cpu_vectorize is False
    assert result.linalg_generic_before >= result.linalg_generic_after_fusion
    assert result.llvm_func_count >= 1
    assert result.allocation_count >= 0


def test_benchmark_source_records_cpu_vectorize_request():
    result = benchmark_source(
        "map (* 2.0) (iota 4)",
        name="tiny-vectorized",
        cpu_vectorize=True,
    )

    assert result.cpu_vectorize is True
    assert result.llvm_func_count >= 1


def test_function_compile_benchmark_records_ir_metrics():
    result = benchmark_function_compilation(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        phase_timeout_s=10.0,
    )

    assert result.completed_phase == "shared_library_link"
    assert result.timed_out_phase is None
    assert result.error is None
    assert result.generated_source_bytes > 0
    assert result.function_prepare_s >= 0
    assert result.descriptor_compile_s >= 0
    assert result.hir_node_count > 0
    assert result.descriptor_mlir_bytes > 0
    assert result.linalg_generic_count > 0
    assert result.lowered_mlir_bytes
    assert result.llvm_ir_bytes
    assert result.llc_s is not None
    assert result.linker_s is not None
    assert result.object_bytes
    assert result.shared_library_bytes
    if Path("/usr/bin/time").is_file():
        assert result.cpu_pipeline_peak_rss_kb
        assert result.llvm_translation_peak_rss_kb
        assert result.llc_peak_rss_kb
        assert result.linker_peak_rss_kb


def test_function_compile_benchmark_can_persist_descriptor_mlir(tmp_path):
    mlir_path = tmp_path / "descriptor.mlir"
    result = benchmark_function_compilation(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        phase_timeout_s=10.0,
        descriptor_mlir_path=mlir_path,
    )

    persisted = mlir_path.read_text(encoding="utf-8")
    assert len(persisted.encode("utf-8")) == result.descriptor_mlir_bytes
    assert "func.func private @__remora_entry" in persisted


def test_cpu_pipeline_stage_diagnostics_complete_for_small_function():
    compiler_result = compile_function_source(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        verify=False,
    )
    stages = diagnose_cpu_pipeline_stages(
        compiler_result.mlir_text,
        phase_timeout_s=10.0,
    )

    assert len(stages) == len(benchmark_module.CPU_PIPELINE_PASSES)
    assert all(not stage.timed_out for stage in stages)
    assert all(stage.error is None for stage in stages)
    assert all(stage.output_bytes for stage in stages)


def test_cpu_pipeline_stage_diagnostics_can_skip_fusion():
    compiler_result = compile_function_source(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        verify=False,
    )
    stages = diagnose_cpu_pipeline_stages(
        compiler_result.mlir_text,
        phase_timeout_s=10.0,
        skip_passes=frozenset({"linalg-fuse-elementwise-ops"}),
    )

    assert len(stages) == len(benchmark_module.CPU_PIPELINE_PASSES) - 1
    assert stages[0].name.startswith("02:one-shot-bufferize")


def test_cpu_pipeline_stage_diagnostics_collect_mlir_timing():
    compiler_result = compile_function_source(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        verify=False,
    )
    stages = diagnose_cpu_pipeline_stages(
        compiler_result.mlir_text,
        phase_timeout_s=10.0,
        collect_pass_timing=True,
    )

    assert stages
    assert all(stage.timing_output for stage in stages)
    assert any("Execution time report" in stage.timing_output for stage in stages)


def test_cpu_pipeline_stage_diagnostics_support_prefix_canonicalization():
    compiler_result = compile_function_source(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        verify=False,
    )
    stages = diagnose_cpu_pipeline_stages(
        compiler_result.mlir_text,
        phase_timeout_s=10.0,
        prefix_passes=("canonicalize", "cse"),
    )

    assert stages[0].name == "pre01:canonicalize"
    assert stages[1].name == "pre02:cse"
    assert stages[2].name == "01:linalg-fuse-elementwise-ops"


def test_function_compile_benchmark_stage_diagnostics_report_timeout(monkeypatch):
    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("mlir-opt", timeout=0.01)

    monkeypatch.setattr(
        benchmark_module,
        "_run_external_with_optional_rss",
        time_out,
    )
    result = benchmark_function_compilation(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        phase_timeout_s=0.01,
        diagnose_cpu_stages=True,
    )

    assert result.completed_phase == "descriptor_compile"
    assert result.timed_out_phase == "cpu_stage:01:linalg-fuse-elementwise-ops"
    assert result.cpu_stage_results
    assert result.cpu_stage_results[0]["timed_out"] is True


def test_function_compile_benchmark_reports_cpu_pipeline_timeout(monkeypatch):
    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("mlir-opt", timeout=0.01)

    monkeypatch.setattr(
        benchmark_module,
        "_run_cpu_pipeline_with_timeout",
        time_out,
    )
    result = benchmark_function_compilation(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        phase_timeout_s=0.01,
    )

    assert result.completed_phase == "descriptor_compile"
    assert result.timed_out_phase == "cpu_mlir_pipeline"
    assert result.error is None
    assert result.descriptor_mlir_bytes > 0
    assert result.lowered_mlir_bytes is None
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["timed_out_phase"] == "cpu_mlir_pipeline"


def test_function_compile_benchmark_reports_llvm_translation_timeout(monkeypatch):
    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("mlir-translate", timeout=0.01)

    monkeypatch.setattr(
        benchmark_module,
        "_run_cpu_pipeline_with_timeout",
        lambda mlir, **kwargs: (mlir, None),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_translate_mlir_to_llvmir_with_timeout",
        time_out,
    )
    result = benchmark_function_compilation(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        phase_timeout_s=0.01,
    )

    assert result.completed_phase == "cpu_mlir_pipeline"
    assert result.timed_out_phase == "llvm_translation"
    assert result.llvm_ir_bytes is None


def test_native_compile_reports_llc_timeout(tmp_path, monkeypatch):
    toolchain = detect_toolchain()
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(args[0], timeout=0.01)

    monkeypatch.setattr(benchmark_module.subprocess, "run", run)
    result = benchmark_module._compile_llvm_ir_with_timeout(
        "define void @f() { ret void }",
        toolchain=toolchain,
        timeout_s=0.01,
    )

    assert calls
    assert result.completed_phase == "llvm_translation"
    assert result.timed_out_phase == "llc_object_generation"
    assert result.object_bytes is None


def test_native_compile_reports_linker_timeout(monkeypatch):
    toolchain = detect_toolchain()
    real_run = subprocess.run
    call_count = 0

    def run(args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise subprocess.TimeoutExpired(args[0], timeout=0.01)
        return real_run(args, **kwargs)

    monkeypatch.setattr(benchmark_module.subprocess, "run", run)
    result = benchmark_module._compile_llvm_ir_with_timeout(
        "define void @f() { ret void }",
        toolchain=toolchain,
        timeout_s=10.0,
    )

    assert call_count == 2
    assert result.completed_phase == "llc_object_generation"
    assert result.timed_out_phase == "shared_library_link"
    assert result.object_bytes
    assert result.shared_library_bytes is None


def test_benchmark_cli_emits_json(tmp_path, capsys):
    source = tmp_path / "bench.remora"
    source.write_text("map (* 2) (iota 4)", encoding="utf-8")

    assert benchmark_main(["--cpu-threads", "1", str(source)]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["name"] == "bench"
    assert payload["cpu_threads"] == 1
    assert payload["cpu_vectorize"] is False
    assert "allocation_count" in payload
    assert captured.err == ""


def test_function_benchmark_case_writes_json(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        benchmark_module,
        "_load_function_benchmark_case",
        lambda case: (
            "def scale xs = map (* 2.0) xs",
            "scale",
            (ArrayType(FLOAT, (StaticDim(4),)),),
            True,
            "ml",
            0.001,
        ),
    )
    output = tmp_path / "function-benchmark.json"

    assert benchmark_main(
        [
            "--case",
            "crater-cnn-gradient-k",
            "--phase-timeout",
            "10",
            "--json",
            str(output),
        ]
    ) == 0
    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))

    assert stdout_payload == file_payload
    assert file_payload["name"] == "crater-cnn-gradient-k"
    assert file_payload["function_name"] == "scale"
    assert file_payload["gradient_source_generation_s"] == 0.001
    assert file_payload["completed_phase"] == "shared_library_link"
    assert captured.err == ""


@pytest.mark.parametrize("size", [4, 8, 16])
def test_im2col_gradient_benchmark_cases_compile(size):
    source, function_name, param_types, include_prelude, syntax, generation_s = (
        benchmark_module._load_function_benchmark_case(
            f"im2col-gradient-{size}"
        )
    )
    result = benchmark_function_compilation(
        source,
        function_name,
        param_types,
        include_prelude=include_prelude,
        syntax=syntax,
        phase_timeout_s=10.0,
        gradient_source_generation_s=generation_s,
    )

    assert result.error is None
    assert result.descriptor_mlir_bytes > 0
    assert result.tensor_extract_count > 0
    assert result.tensor_insert_count == 0
    assert result.timed_out_phase is None


def test_function_benchmark_case_writes_partial_json_on_timeout(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(
        benchmark_module,
        "_load_function_benchmark_case",
        lambda case: (
            "def scale xs = map (* 2.0) xs",
            "scale",
            (ArrayType(FLOAT, (StaticDim(4),)),),
            True,
            "ml",
            0.001,
        ),
    )

    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("mlir-opt", timeout=0.01)

    monkeypatch.setattr(
        benchmark_module,
        "_run_cpu_pipeline_with_timeout",
        time_out,
    )
    output = tmp_path / "timed-out-function-benchmark.json"

    assert benchmark_main(
        [
            "--case",
            "crater-cnn-gradient-k",
            "--phase-timeout",
            "0.01",
            "--json",
            str(output),
        ]
    ) == 0
    captured = capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert json.loads(captured.out) == payload
    assert payload["completed_phase"] == "descriptor_compile"
    assert payload["timed_out_phase"] == "cpu_mlir_pipeline"
    assert payload["lowered_mlir_bytes"] is None
    assert captured.err == ""


def test_benchmark_cli_diagnoses_existing_mlir(tmp_path, capsys):
    mlir_path = tmp_path / "descriptor.mlir"
    json_path = tmp_path / "stages.json"
    artifact = compile_function_source(
        "def scale xs = map (* 2.0) xs",
        "scale",
        (ArrayType(FLOAT, (StaticDim(4),)),),
        verify=False,
    )
    mlir_path.write_text(artifact.mlir_text, encoding="utf-8")

    assert benchmark_main(
        [
            "--diagnose-mlir",
            str(mlir_path),
            "--phase-timeout",
            "10",
            "--skip-cpu-stage",
            "linalg-fuse-elementwise-ops",
            "--json",
            str(json_path),
        ]
    ) == 0
    captured = capsys.readouterr()
    stages = json.loads(json_path.read_text(encoding="utf-8"))

    assert json.loads(captured.out) == stages
    assert stages[0]["name"].startswith("02:one-shot-bufferize")
    assert all(stage["error"] is None for stage in stages)


def test_benchmark_cli_checks_baseline(tmp_path, capsys):
    source = tmp_path / "bench.remora"
    source.write_text("map (* 2) (iota 4)", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "bench",
                        "max_linalg_generic_after_fusion": 0,
                        "max_allocation_count": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert benchmark_main(["--baseline", str(baseline), str(source)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["name"] == "bench"
    assert "linalg_generic_after_fusion" in captured.err


def test_benchmark_cli_runs_suite(capsys):
    baseline = "docs/BENCHMARK_BASELINES.json"

    assert benchmark_main(["--baseline", baseline, "--suite"]) == 0
    captured = capsys.readouterr()
    results = json.loads(captured.out)

    assert isinstance(results, list)
    assert len(results) >= len(SMOKE_CASES)
    assert any(r["name"] == "vector_scale" for r in results)
    assert captured.err == ""


def test_benchmark_baseline_checker_reports_missing_case():
    result = benchmark_source("map (* 2) (iota 4)", name="missing")

    failures = check_result_against_baseline(result, {"cases": []})

    assert failures == ["benchmark baseline for 'missing' was not found"]


def test_benchmark_baselines_cover_smoke_cases():
    from pathlib import Path

    payload = json.loads(Path("docs/BENCHMARK_BASELINES.json").read_text(encoding="utf-8"))
    names = {case["name"] for case in payload["cases"]}

    assert set(SMOKE_CASES) <= names
    for case in payload["cases"]:
        assert case["max_linalg_generic_after_fusion"] >= 0
        assert case["max_allocation_count"] >= 0
