"""Small benchmark harness for Remora Dense Core."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, is_dataclass
import json
from pathlib import Path
from shutil import which
import subprocess
import sys
import tempfile
import time
from typing import Any

from remora.compiler import (
    compile_prepared_function,
    compile_source_to_mlir,
    prepare_function_source,
)
from remora.errors import RemoraError
from remora.hir import HIRExpr, HIRFunction, HIRParam
from remora.pipeline import (
    CPU_PIPELINE,
    CPU_PIPELINE_PASSES,
    PipelineToolchain,
    PipelineUnavailable,
    detect_toolchain,
    run_cpu_pipeline_text,
    run_fusion_pipeline_text,
)
from remora.runtime import evaluate_source_compiled, resolve_cpu_threads
from remora.types import RemoraType


BASELINE_SOURCES = {
    "vector_scale": "map (* 2.0) (iota 1000)",
    "map_chain": "map (* 3.0) (map (* 2.0) (iota 1000))",
    "vector_sum": "fold (+) 0.0 (iota 1000)",
    "dot": (
        "let xs = [1.0, 2.0, 3.0] in "
        "let ys = [4.0, 5.0, 6.0] in "
        "dot xs ys"
    ),
    "row_reduce": (
        "let xs = [[1.0, 2.0], [3.0, 4.0]] in "
        "map (\\row -> fold (+) 0.0 row) xs"
    ),
}

FUNCTION_BENCHMARK_CASES = (
    "crater-cnn-gradient-k",
    "im2col-gradient-4",
    "im2col-gradient-8",
    "im2col-gradient-16",
)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    cpu_threads: int | None
    cpu_vectorize: bool
    mlir_compile_s: float
    fusion_pipeline_s: float
    cpu_pipeline_s: float
    compiled_execution_s: float
    linalg_generic_before: int
    linalg_generic_after_fusion: int
    llvm_func_count: int
    allocation_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FunctionCompileBenchmarkResult:
    name: str
    function_name: str
    phase_timeout_s: float | None
    function_prepare_s: float
    descriptor_compile_s: float
    cpu_pipeline_s: float | None
    llvm_translation_s: float | None
    generated_source_bytes: int
    hir_node_count: int
    descriptor_mlir_bytes: int
    linalg_generic_count: int
    tensor_extract_count: int
    tensor_insert_count: int
    lowered_mlir_bytes: int | None
    llvm_ir_bytes: int | None
    completed_phase: str
    timed_out_phase: str | None
    error: str | None
    gradient_source_generation_s: float | None = None
    llc_s: float | None = None
    linker_s: float | None = None
    object_bytes: int | None = None
    shared_library_bytes: int | None = None
    cpu_pipeline_peak_rss_kb: int | None = None
    llvm_translation_peak_rss_kb: int | None = None
    llc_peak_rss_kb: int | None = None
    linker_peak_rss_kb: int | None = None
    cpu_stage_results: list[dict[str, object]] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark_function_compilation(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    *,
    name: str | None = None,
    include_prelude: bool = True,
    syntax: str = "ml",
    phase_timeout_s: float | None = None,
    toolchain: PipelineToolchain | None = None,
    gradient_source_generation_s: float | None = None,
    descriptor_mlir_path: Path | None = None,
    diagnose_cpu_stages: bool = False,
    skip_cpu_stages: frozenset[str] = frozenset(),
    collect_mlir_pass_timing: bool = False,
    pre_cpu_stages: tuple[str, ...] = (),
) -> FunctionCompileBenchmarkResult:
    """Measure descriptor and CPU lowering for one specialized function.

    The result is returned even when the external CPU MLIR phase times out, so
    large compiler regressions can be recorded as data instead of hanging the
    benchmark process.
    """
    benchmark_name = name or function_name
    toolchain = detect_toolchain() if toolchain is None else toolchain
    source_bytes = len(source.encode("utf-8"))

    start = time.perf_counter()
    prepared = prepare_function_source(
        source,
        function_name,
        param_types,
        include_prelude=include_prelude,
        syntax=syntax,
    )
    function_prepare_s = time.perf_counter() - start
    start = time.perf_counter()
    artifact = compile_prepared_function(prepared, verify=False)
    descriptor_compile_s = time.perf_counter() - start
    mlir = artifact.mlir_text
    if descriptor_mlir_path is not None and mlir:
        descriptor_mlir_path.write_text(mlir, encoding="utf-8")
    if not mlir:
        return FunctionCompileBenchmarkResult(
            name=benchmark_name,
            function_name=function_name,
            phase_timeout_s=phase_timeout_s,
            skip_passes=skip_cpu_stages,
            collect_pass_timing=collect_mlir_pass_timing,
            prefix_passes=pre_cpu_stages,
            function_prepare_s=function_prepare_s,
            descriptor_compile_s=descriptor_compile_s,
            cpu_pipeline_s=None,
            llvm_translation_s=None,
            generated_source_bytes=source_bytes,
            hir_node_count=_hir_node_count(artifact.hir_function),
            descriptor_mlir_bytes=0,
            linalg_generic_count=0,
            tensor_extract_count=0,
            tensor_insert_count=0,
            lowered_mlir_bytes=None,
            llvm_ir_bytes=None,
            completed_phase="descriptor_compile",
            timed_out_phase=None,
            error="descriptor lowering produced empty MLIR",
            gradient_source_generation_s=gradient_source_generation_s,
        )

    metrics = _descriptor_mlir_metrics(mlir)
    if diagnose_cpu_stages:
        stage_results = diagnose_cpu_pipeline_stages(
            mlir,
            toolchain=toolchain,
            phase_timeout_s=phase_timeout_s,
        )
        final_stage = stage_results[-1] if stage_results else None
        timed_out_phase = None
        error = None
        completed_phase = "descriptor_compile"
        lowered_mlir_bytes = None
        if final_stage is not None:
            if final_stage.timed_out:
                timed_out_phase = f"cpu_stage:{final_stage.name}"
            elif final_stage.error is not None:
                error = final_stage.error
            else:
                completed_phase = "cpu_stage_diagnostics"
                lowered_mlir_bytes = final_stage.output_bytes
        return FunctionCompileBenchmarkResult(
            name=benchmark_name,
            function_name=function_name,
            phase_timeout_s=phase_timeout_s,
            function_prepare_s=function_prepare_s,
            descriptor_compile_s=descriptor_compile_s,
            cpu_pipeline_s=sum(stage.elapsed_s for stage in stage_results),
            llvm_translation_s=None,
            generated_source_bytes=source_bytes,
            hir_node_count=_hir_node_count(artifact.hir_function),
            descriptor_mlir_bytes=metrics["descriptor_mlir_bytes"],
            linalg_generic_count=metrics["linalg_generic_count"],
            tensor_extract_count=metrics["tensor_extract_count"],
            tensor_insert_count=metrics["tensor_insert_count"],
            lowered_mlir_bytes=lowered_mlir_bytes,
            llvm_ir_bytes=None,
            completed_phase=completed_phase,
            timed_out_phase=timed_out_phase,
            error=error,
            gradient_source_generation_s=gradient_source_generation_s,
            cpu_stage_results=[stage.to_dict() for stage in stage_results],
        )
    start = time.perf_counter()
    try:
        lowered, cpu_pipeline_peak_rss_kb = _run_cpu_pipeline_with_timeout(
            mlir,
            toolchain=toolchain,
            timeout_s=phase_timeout_s,
        )
    except subprocess.TimeoutExpired:
        return FunctionCompileBenchmarkResult(
            name=benchmark_name,
            function_name=function_name,
            phase_timeout_s=phase_timeout_s,
            function_prepare_s=function_prepare_s,
            descriptor_compile_s=descriptor_compile_s,
            cpu_pipeline_s=time.perf_counter() - start,
            llvm_translation_s=None,
            generated_source_bytes=source_bytes,
            hir_node_count=_hir_node_count(artifact.hir_function),
            descriptor_mlir_bytes=metrics["descriptor_mlir_bytes"],
            linalg_generic_count=metrics["linalg_generic_count"],
            tensor_extract_count=metrics["tensor_extract_count"],
            tensor_insert_count=metrics["tensor_insert_count"],
            lowered_mlir_bytes=None,
            llvm_ir_bytes=None,
            completed_phase="descriptor_compile",
            timed_out_phase="cpu_mlir_pipeline",
            error=None,
            gradient_source_generation_s=gradient_source_generation_s,
        )
    except PipelineUnavailable as exc:
        return FunctionCompileBenchmarkResult(
            name=benchmark_name,
            function_name=function_name,
            phase_timeout_s=phase_timeout_s,
            function_prepare_s=function_prepare_s,
            descriptor_compile_s=descriptor_compile_s,
            cpu_pipeline_s=time.perf_counter() - start,
            llvm_translation_s=None,
            generated_source_bytes=source_bytes,
            hir_node_count=_hir_node_count(artifact.hir_function),
            descriptor_mlir_bytes=metrics["descriptor_mlir_bytes"],
            linalg_generic_count=metrics["linalg_generic_count"],
            tensor_extract_count=metrics["tensor_extract_count"],
            tensor_insert_count=metrics["tensor_insert_count"],
            lowered_mlir_bytes=None,
            llvm_ir_bytes=None,
            completed_phase="descriptor_compile",
            timed_out_phase=None,
            error=str(exc),
            gradient_source_generation_s=gradient_source_generation_s,
        )
    cpu_pipeline_s = time.perf_counter() - start

    start = time.perf_counter()
    try:
        llvm_ir, llvm_translation_peak_rss_kb = (
            _translate_mlir_to_llvmir_with_timeout(
            lowered,
            toolchain=toolchain,
            timeout_s=phase_timeout_s,
        )
        )
    except subprocess.TimeoutExpired:
        return FunctionCompileBenchmarkResult(
            name=benchmark_name,
            function_name=function_name,
            phase_timeout_s=phase_timeout_s,
            function_prepare_s=function_prepare_s,
            descriptor_compile_s=descriptor_compile_s,
            cpu_pipeline_s=cpu_pipeline_s,
            llvm_translation_s=time.perf_counter() - start,
            generated_source_bytes=source_bytes,
            hir_node_count=_hir_node_count(artifact.hir_function),
            descriptor_mlir_bytes=metrics["descriptor_mlir_bytes"],
            linalg_generic_count=metrics["linalg_generic_count"],
            tensor_extract_count=metrics["tensor_extract_count"],
            tensor_insert_count=metrics["tensor_insert_count"],
            lowered_mlir_bytes=len(lowered.encode("utf-8")),
            llvm_ir_bytes=None,
            completed_phase="cpu_mlir_pipeline",
            timed_out_phase="llvm_translation",
            error=None,
            gradient_source_generation_s=gradient_source_generation_s,
            cpu_pipeline_peak_rss_kb=cpu_pipeline_peak_rss_kb,
        )
    except PipelineUnavailable as exc:
        return FunctionCompileBenchmarkResult(
            name=benchmark_name,
            function_name=function_name,
            phase_timeout_s=phase_timeout_s,
            function_prepare_s=function_prepare_s,
            descriptor_compile_s=descriptor_compile_s,
            cpu_pipeline_s=cpu_pipeline_s,
            llvm_translation_s=time.perf_counter() - start,
            generated_source_bytes=source_bytes,
            hir_node_count=_hir_node_count(artifact.hir_function),
            descriptor_mlir_bytes=metrics["descriptor_mlir_bytes"],
            linalg_generic_count=metrics["linalg_generic_count"],
            tensor_extract_count=metrics["tensor_extract_count"],
            tensor_insert_count=metrics["tensor_insert_count"],
            lowered_mlir_bytes=len(lowered.encode("utf-8")),
            llvm_ir_bytes=None,
            completed_phase="cpu_mlir_pipeline",
            timed_out_phase=None,
            error=str(exc),
            gradient_source_generation_s=gradient_source_generation_s,
            cpu_pipeline_peak_rss_kb=cpu_pipeline_peak_rss_kb,
        )
    llvm_translation_s = time.perf_counter() - start

    native_result = _compile_llvm_ir_with_timeout(
        llvm_ir,
        toolchain=toolchain,
        timeout_s=phase_timeout_s,
    )

    return FunctionCompileBenchmarkResult(
        name=benchmark_name,
        function_name=function_name,
        phase_timeout_s=phase_timeout_s,
        function_prepare_s=function_prepare_s,
        descriptor_compile_s=descriptor_compile_s,
        cpu_pipeline_s=cpu_pipeline_s,
        llvm_translation_s=llvm_translation_s,
        generated_source_bytes=source_bytes,
        hir_node_count=_hir_node_count(artifact.hir_function),
        descriptor_mlir_bytes=metrics["descriptor_mlir_bytes"],
        linalg_generic_count=metrics["linalg_generic_count"],
        tensor_extract_count=metrics["tensor_extract_count"],
        tensor_insert_count=metrics["tensor_insert_count"],
        lowered_mlir_bytes=len(lowered.encode("utf-8")),
        llvm_ir_bytes=len(llvm_ir.encode("utf-8")),
        completed_phase=native_result.completed_phase,
        timed_out_phase=native_result.timed_out_phase,
        error=native_result.error,
        gradient_source_generation_s=gradient_source_generation_s,
        llc_s=native_result.llc_s,
        linker_s=native_result.linker_s,
        object_bytes=native_result.object_bytes,
        shared_library_bytes=native_result.shared_library_bytes,
        cpu_pipeline_peak_rss_kb=cpu_pipeline_peak_rss_kb,
        llvm_translation_peak_rss_kb=llvm_translation_peak_rss_kb,
        llc_peak_rss_kb=native_result.llc_peak_rss_kb,
        linker_peak_rss_kb=native_result.linker_peak_rss_kb,
    )


def _load_function_benchmark_case(
    case: str,
) -> tuple[str, str, tuple[RemoraType, ...], bool, str, float]:
    from remora.ad_source import generate_gradient_function_source
    from remora.types import ArrayType, FLOAT, StaticDim

    if case.startswith("im2col-gradient-"):
        image_size = int(case.rsplit("-", 1)[1])
        source = f"""
(define/pi ()
  (patch-sum [image (Array Float {image_size} {image_size})] Float)
  (fold + 0.0 (ravel (im2col image [3 3] 1))))
"""
        param_types = (
            ArrayType(
                FLOAT,
                (StaticDim(image_size), StaticDim(image_size)),
            ),
        )
        start = time.perf_counter()
        gradient = generate_gradient_function_source(
            source,
            "patch-sum",
            param_types,
            differentiate_input=0,
            include_prelude=False,
            syntax="lisp",
        )
        generation_s = time.perf_counter() - start
        return (
            gradient.source,
            gradient.function_name,
            param_types,
            False,
            "lisp",
            generation_s,
        )

    if case != "crater-cnn-gradient-k":
        raise ValueError(f"unknown function benchmark case {case!r}")

    from examples.crater_train import _CNN_FULL_LISP_SRC, _parameter_types

    param_types = _parameter_types()
    start = time.perf_counter()
    gradient = generate_gradient_function_source(
        _CNN_FULL_LISP_SRC,
        "cnn-loss",
        param_types,
        differentiate_input=0,
        include_prelude=False,
        syntax="lisp",
    )
    generation_s = time.perf_counter() - start
    return (
        gradient.source,
        gradient.function_name,
        param_types,
        False,
        "lisp",
        generation_s,
    )


def _run_cpu_pipeline_with_timeout(
    mlir_text: str,
    *,
    toolchain: PipelineToolchain,
    timeout_s: float | None,
) -> tuple[str, int | None]:
    if toolchain.mlir_opt is None:
        raise PipelineUnavailable(
            "mlir-opt is required for standalone MLIR pipeline validation"
        )
    result, peak_rss_kb = _run_external_with_optional_rss(
        [toolchain.mlir_opt, f"--pass-pipeline={CPU_PIPELINE}", "-"],
        input=mlir_text,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise PipelineUnavailable(
            f"standalone MLIR CPU pipeline failed: {stderr}"
        )
    return result.stdout, peak_rss_kb


@dataclass(frozen=True)
class CPUStageBenchmarkResult:
    name: str
    pass_pipeline: str
    elapsed_s: float
    input_bytes: int
    output_bytes: int | None
    peak_rss_kb: int | None
    timed_out: bool
    error: str | None
    timing_output: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def diagnose_cpu_pipeline_stages(
    mlir_text: str,
    *,
    toolchain: PipelineToolchain | None = None,
    phase_timeout_s: float | None = None,
    skip_passes: frozenset[str] = frozenset(),
    collect_pass_timing: bool = False,
    prefix_passes: tuple[str, ...] = (),
) -> list[CPUStageBenchmarkResult]:
    """Run production CPU passes one at a time and return stage metrics."""
    toolchain = detect_toolchain() if toolchain is None else toolchain
    if toolchain.mlir_opt is None:
        raise PipelineUnavailable("mlir-opt is required for CPU stage diagnostics")

    current = mlir_text
    results: list[CPUStageBenchmarkResult] = []
    stage_specs = [
        (f"pre{index + 1:02d}", pass_spec)
        for index, pass_spec in enumerate(prefix_passes)
    ] + [
        (f"{index + 1:02d}", pass_spec)
        for index, pass_spec in enumerate(CPU_PIPELINE_PASSES)
    ]
    for stage_prefix, pass_spec in stage_specs:
        if pass_spec in skip_passes:
            continue
        stage_name = f"{stage_prefix}:{pass_spec.split('{', 1)[0]}"
        pipeline = f"builtin.module({pass_spec})"
        command = [toolchain.mlir_opt]
        if collect_pass_timing:
            command.extend(
                ["--mlir-timing", "--mlir-timing-display=tree"]
            )
        command.extend([f"--pass-pipeline={pipeline}", "-"])
        input_bytes = len(current.encode("utf-8"))
        start = time.perf_counter()
        try:
            process, peak_rss_kb = _run_external_with_optional_rss(
                command,
                input=current,
                timeout=phase_timeout_s,
            )
        except subprocess.TimeoutExpired:
            results.append(
                CPUStageBenchmarkResult(
                    stage_name,
                    pipeline,
                    time.perf_counter() - start,
                    input_bytes,
                    None,
                    None,
                    True,
                    None,
                    None,
                )
            )
            break
        elapsed_s = time.perf_counter() - start
        if process.returncode != 0:
            results.append(
                CPUStageBenchmarkResult(
                    stage_name,
                    pipeline,
                    elapsed_s,
                    input_bytes,
                    None,
                    peak_rss_kb,
                    False,
                    process.stderr.strip(),
                    process.stderr.strip() if collect_pass_timing else None,
                )
            )
            break
        current = process.stdout
        results.append(
            CPUStageBenchmarkResult(
                stage_name,
                pipeline,
                elapsed_s,
                input_bytes,
                len(current.encode("utf-8")),
                peak_rss_kb,
                False,
                None,
                process.stderr.strip() if collect_pass_timing else None,
            )
        )
    return results


def _translate_mlir_to_llvmir_with_timeout(
    mlir_text: str,
    *,
    toolchain: PipelineToolchain,
    timeout_s: float | None,
) -> tuple[str, int | None]:
    if toolchain.mlir_translate is None:
        raise PipelineUnavailable(
            "mlir-translate is required for LLVM IR translation"
        )
    result, peak_rss_kb = _run_external_with_optional_rss(
        [toolchain.mlir_translate, "--mlir-to-llvmir"],
        input=mlir_text,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise PipelineUnavailable(
            f"LLVM IR translation failed: {stderr}"
        )
    return result.stdout, peak_rss_kb


_RSS_MARKER = "__REMORA_MAX_RSS_KB__="


def _run_external_with_optional_rss(
    args: list[str],
    *,
    input: str | None = None,
    timeout: float | None = None,
) -> tuple[subprocess.CompletedProcess[str], int | None]:
    time_executable = "/usr/bin/time"
    timed_args = args
    measure_rss = Path(time_executable).is_file()
    if measure_rss:
        timed_args = [
            time_executable,
            "-f",
            f"{_RSS_MARKER}%M",
            "--",
            *args,
        ]
    result = subprocess.run(
        timed_args,
        input=input,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if not measure_rss:
        return result, None

    peak_rss_kb: int | None = None
    stderr_lines: list[str] = []
    for line in result.stderr.splitlines():
        if line.startswith(_RSS_MARKER):
            try:
                peak_rss_kb = int(line[len(_RSS_MARKER) :])
            except ValueError:
                stderr_lines.append(line)
        else:
            stderr_lines.append(line)
    result.stderr = "\n".join(stderr_lines)
    return result, peak_rss_kb


@dataclass(frozen=True)
class _NativeCompileResult:
    completed_phase: str
    timed_out_phase: str | None
    error: str | None
    llc_s: float | None
    linker_s: float | None
    object_bytes: int | None
    shared_library_bytes: int | None
    llc_peak_rss_kb: int | None
    linker_peak_rss_kb: int | None


def _compile_llvm_ir_with_timeout(
    llvm_ir: str,
    *,
    toolchain: PipelineToolchain,
    timeout_s: float | None,
) -> _NativeCompileResult:
    if toolchain.llc is None:
        return _NativeCompileResult(
            "llvm_translation", None, "llc is required", None, None, None, None,
            None, None,
        )
    linker = which("gcc") or which("cc")
    if linker is None:
        return _NativeCompileResult(
            "llvm_translation",
            None,
            "gcc or cc is required",
            None,
            None,
            None,
            None,
            None,
            None,
        )

    with tempfile.TemporaryDirectory(prefix="remora-benchmark-") as temp_dir:
        root = Path(temp_dir)
        ll_path = root / "module.ll"
        obj_path = root / "module.o"
        so_path = root / "module.so"
        ll_path.write_text(llvm_ir, encoding="utf-8")

        start = time.perf_counter()
        try:
            llc_result, llc_peak_rss_kb = _run_external_with_optional_rss(
                [
                    toolchain.llc,
                    "-filetype=obj",
                    "-relocation-model=pic",
                    str(ll_path),
                    "-o",
                    str(obj_path),
                ],
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return _NativeCompileResult(
                "llvm_translation",
                "llc_object_generation",
                None,
                time.perf_counter() - start,
                None,
                None,
                None,
                None,
                None,
            )
        llc_s = time.perf_counter() - start
        if llc_result.returncode != 0:
            return _NativeCompileResult(
                "llvm_translation",
                None,
                f"llc failed: {llc_result.stderr.strip()}",
                llc_s,
                None,
                None,
                None,
                llc_peak_rss_kb,
                None,
            )
        object_bytes = obj_path.stat().st_size

        start = time.perf_counter()
        try:
            linker_result, linker_peak_rss_kb = _run_external_with_optional_rss(
                [linker, "-shared", str(obj_path), "-o", str(so_path)],
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return _NativeCompileResult(
                "llc_object_generation",
                "shared_library_link",
                None,
                llc_s,
                time.perf_counter() - start,
                object_bytes,
                None,
                llc_peak_rss_kb,
                None,
            )
        linker_s = time.perf_counter() - start
        if linker_result.returncode != 0:
            return _NativeCompileResult(
                "llc_object_generation",
                None,
                f"linker failed: {linker_result.stderr.strip()}",
                llc_s,
                linker_s,
                object_bytes,
                None,
                llc_peak_rss_kb,
                linker_peak_rss_kb,
            )
        return _NativeCompileResult(
            "shared_library_link",
            None,
            None,
            llc_s,
            linker_s,
            object_bytes,
            so_path.stat().st_size,
            llc_peak_rss_kb,
            linker_peak_rss_kb,
        )


def _descriptor_mlir_metrics(mlir: str) -> dict[str, int]:
    return {
        "descriptor_mlir_bytes": len(mlir.encode("utf-8")),
        "linalg_generic_count": mlir.count("linalg.generic"),
        "tensor_extract_count": mlir.count("tensor.extract"),
        "tensor_insert_count": (
            mlir.count("tensor.insert ") + mlir.count("tensor.insert_slice")
        ),
    }


def _hir_node_count(value: Any) -> int:
    seen: set[int] = set()

    def visit(node: Any) -> int:
        if isinstance(node, (str, bytes, int, float, bool, type(None))):
            return 0
        node_id = id(node)
        if node_id in seen:
            return 0
        if isinstance(node, (HIRExpr, HIRFunction, HIRParam)):
            seen.add(node_id)
            count = 1
        else:
            count = 0
        if is_dataclass(node):
            return count + sum(
                visit(getattr(node, field.name)) for field in fields(node)
            )
        if isinstance(node, (list, tuple)):
            return count + sum(visit(item) for item in node)
        return count

    return visit(value)


def benchmark_source(
    source: str,
    *,
    name: str = "program",
    cpu_threads: int | None = None,
    cpu_vectorize: bool = True,
    toolchain: PipelineToolchain | None = None,
) -> BenchmarkResult:
    """Compile and execute one source string, returning coarse timing metrics."""
    resolved_cpu_threads = resolve_cpu_threads(cpu_threads)
    toolchain = detect_toolchain() if toolchain is None else toolchain

    start = time.perf_counter()
    mlir = compile_source_to_mlir(source, verify=False)
    mlir_compile_s = time.perf_counter() - start

    start = time.perf_counter()
    fused = run_fusion_pipeline_text(mlir, toolchain=toolchain)
    fusion_pipeline_s = time.perf_counter() - start

    start = time.perf_counter()
    lowered = run_cpu_pipeline_text(mlir, toolchain=toolchain, vectorize=cpu_vectorize)
    cpu_pipeline_s = time.perf_counter() - start

    start = time.perf_counter()
    evaluate_source_compiled(
        source,
        cpu_threads=resolved_cpu_threads,
        cpu_vectorize=cpu_vectorize,
    )
    compiled_execution_s = time.perf_counter() - start

    return BenchmarkResult(
        name=name,
        cpu_threads=resolved_cpu_threads,
        cpu_vectorize=cpu_vectorize,
        mlir_compile_s=mlir_compile_s,
        fusion_pipeline_s=fusion_pipeline_s,
        cpu_pipeline_s=cpu_pipeline_s,
        compiled_execution_s=compiled_execution_s,
        linalg_generic_before=mlir.count("linalg.generic"),
        linalg_generic_after_fusion=fused.count("linalg.generic"),
        llvm_func_count=lowered.count("llvm.func"),
        allocation_count=_allocation_count(lowered),
    )


def _allocation_count(lowered_mlir: str) -> int:
    return lowered_mlir.count("llvm.call @malloc") + lowered_mlir.count("memref.alloc")


def check_result_against_baseline(
    result: BenchmarkResult,
    baselines: dict[str, object],
) -> list[str]:
    cases = baselines.get("cases")
    if not isinstance(cases, list):
        return ["benchmark baseline file must contain a cases list"]
    baseline = next(
        (case for case in cases if isinstance(case, dict) and case.get("name") == result.name),
        None,
    )
    if baseline is None:
        return [f"benchmark baseline for {result.name!r} was not found"]

    failures: list[str] = []
    max_fused = baseline.get("max_linalg_generic_after_fusion")
    if isinstance(max_fused, int) and result.linalg_generic_after_fusion > max_fused:
        failures.append(
            f"{result.name}: linalg_generic_after_fusion {result.linalg_generic_after_fusion} > {max_fused}"
        )
    max_allocs = baseline.get("max_allocation_count")
    if isinstance(max_allocs, int) and result.allocation_count > max_allocs:
        failures.append(f"{result.name}: allocation_count {result.allocation_count} > {max_allocs}")
    return failures


def run_benchmark_suite(
    baseline_path: Path,
    *,
    cpu_threads: int | None = None,
    cpu_vectorize: bool = True,
) -> tuple[list[BenchmarkResult], list[str]]:
    """Run all cases defined in the baseline file."""
    baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
    cases = baselines.get("cases", [])
    results: list[BenchmarkResult] = []
    all_failures: list[str] = []

    for case in cases:
        name = case.get("name")
        if not name:
            continue
        source = BASELINE_SOURCES.get(name)
        if not source:
            all_failures.append(f"No source string defined for baseline case {name!r}")
            continue

        result = benchmark_source(
            source,
            name=name,
            cpu_threads=cpu_threads,
            cpu_vectorize=cpu_vectorize,
        )
        results.append(result)
        all_failures.extend(check_result_against_baseline(result, baselines))

    return results, all_failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark a Remora Dense Core source file")
    parser.add_argument("file", type=Path, nargs="?", help="Remora source file")
    parser.add_argument("--name", default=None, help="benchmark case name")
    parser.add_argument(
        "--case",
        choices=FUNCTION_BENCHMARK_CASES,
        default=None,
        help="run a built-in specialized function compilation case",
    )
    parser.add_argument(
        "--phase-timeout",
        type=float,
        default=None,
        help="timeout in seconds for each supported external compilation phase",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the JSON result to this path",
    )
    parser.add_argument(
        "--descriptor-mlir",
        type=Path,
        default=None,
        help="diagnostic-only path for persisting descriptor MLIR",
    )
    parser.add_argument(
        "--diagnose-mlir",
        type=Path,
        default=None,
        help="run CPU stage diagnostics directly on an existing MLIR file",
    )
    parser.add_argument(
        "--diagnose-cpu-stages",
        action="store_true",
        help="run CPU MLIR passes sequentially and report per-stage metrics",
    )
    parser.add_argument(
        "--skip-cpu-stage",
        action="append",
        default=[],
        choices=CPU_PIPELINE_PASSES,
        help="skip one production pass during --diagnose-cpu-stages",
    )
    parser.add_argument(
        "--mlir-pass-timing",
        action="store_true",
        help="include MLIR pass-manager timing output in CPU stage diagnostics",
    )
    parser.add_argument(
        "--pre-cpu-stage",
        action="append",
        default=[],
        choices=("canonicalize", "cse"),
        help="run a diagnostic canonicalization pass before production CPU stages",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="requested CPU worker thread count; defaults to REMORA_NUM_THREADS when set",
    )
    vectorize_group = parser.add_mutually_exclusive_group()
    vectorize_group.add_argument(
        "--cpu-vectorize",
        dest="cpu_vectorize",
        action="store_true",
        help="use the affine/vector CPU lowering pipeline (default)",
    )
    vectorize_group.add_argument(
        "--no-cpu-vectorize",
        dest="cpu_vectorize",
        action="store_false",
        help="use the scalar CPU lowering pipeline",
    )
    parser.set_defaults(cpu_vectorize=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="optional benchmark baseline JSON file to check this result against",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="run all cases from the baseline file; requires --baseline",
    )
    args = parser.parse_args(argv)

    if args.suite and not args.baseline:
        print("remora-bench: --suite requires --baseline", file=sys.stderr)
        return 1
    if not args.suite and not args.file and not args.case and not args.diagnose_mlir:
        print(
            "remora-bench: file or --case is required unless using --suite",
            file=sys.stderr,
        )
        return 1
    if args.case and (args.file or args.suite):
        print(
            "remora-bench: --case cannot be combined with a file or --suite",
            file=sys.stderr,
        )
        return 1
    if args.diagnose_mlir and (args.case or args.file or args.suite):
        print(
            "remora-bench: --diagnose-mlir cannot be combined with a file, --case, or --suite",
            file=sys.stderr,
        )
        return 1

    try:
        if args.diagnose_mlir:
            mlir = args.diagnose_mlir.read_text(encoding="utf-8")
            payload = [
                stage.to_dict()
                for stage in diagnose_cpu_pipeline_stages(
                    mlir,
                    phase_timeout_s=args.phase_timeout,
                    skip_passes=frozenset(args.skip_cpu_stage),
                    collect_pass_timing=args.mlir_pass_timing,
                    prefix_passes=tuple(args.pre_cpu_stage),
                )
            ]
            failures = []
        elif args.case:
            (
                source,
                function_name,
                param_types,
                include_prelude,
                syntax,
                generation_s,
            ) = _load_function_benchmark_case(args.case)
            function_result = benchmark_function_compilation(
                source,
                function_name,
                param_types,
                name=args.name or args.case,
                include_prelude=include_prelude,
                syntax=syntax,
                phase_timeout_s=args.phase_timeout,
                gradient_source_generation_s=generation_s,
                descriptor_mlir_path=args.descriptor_mlir,
                diagnose_cpu_stages=args.diagnose_cpu_stages,
                skip_cpu_stages=frozenset(args.skip_cpu_stage),
                collect_mlir_pass_timing=args.mlir_pass_timing,
                pre_cpu_stages=tuple(args.pre_cpu_stage),
            )
            payload: object = function_result.to_dict()
            failures = []
        elif args.suite:
            results, failures = run_benchmark_suite(
                args.baseline,
                cpu_threads=args.cpu_threads,
                cpu_vectorize=args.cpu_vectorize,
            )
            payload = [r.to_dict() for r in results]
        else:
            source = args.file.read_text(encoding="utf-8")
            result = benchmark_source(
                source,
                name=args.name or args.file.stem,
                cpu_threads=args.cpu_threads,
                cpu_vectorize=args.cpu_vectorize,
            )
            failures = []
            if args.baseline is not None:
                baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
                failures = check_result_against_baseline(result, baseline)
            payload = result.to_dict()
        rendered = json.dumps(payload, sort_keys=True)
        print(rendered)
        if args.json is not None:
            args.json.write_text(rendered + "\n", encoding="utf-8")
    except (OSError, RemoraError, ValueError, json.JSONDecodeError) as exc:
        print(f"remora-bench: {exc}", file=sys.stderr)
        return 1

    if failures:
        for failure in failures:
            print(f"remora-bench: {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
