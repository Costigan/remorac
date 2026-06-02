"""Validate Remora's checked-in standalone MLIR pipelines."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.compiler import compile_source_to_mlir  # noqa: E402
from remora.pipeline import (  # noqa: E402
    CPU_PIPELINE,
    FUSION_PIPELINE,
    GPU_NVIDIA_SCAFFOLD_LLVM_DIALECT_PIPELINE,
    GPU_NVIDIA_SCAFFOLD_NVVM_PIPELINE,
    PipelineUnavailable,
    detect_toolchain,
    run_cpu_pipeline_text,
    run_external_pipeline_text,
    run_fusion_pipeline_text,
    translate_mlir_to_llvmir,
)


CASES = {
    "vector_scale": {
        "source": "map (* 2.0) (iota 10)",
        "before": 2,
        "after": 1,
    },
    "map_chain": {
        "source": "map (* 3) (map (* 2) (iota 10))",
        "before": 3,
        "after": 1,
    },
    "vector_sum": {
        "source": "fold (+) 0.0 (iota 10)",
        "before": 2,
        "after": 2,
    },
    "dot": {
        "source": (
            "let xs = [1.0, 2.0, 3.0] in "
            "let ys = [4.0, 5.0, 6.0] in "
            "dot xs ys"
        ),
        "before": 2,
        "after": 1,
    },
    "map_fold": {
        "source": "fold (+) 0.0 (map (* 2.0) (iota 10))",
        "before": 3,
        "after": 2,
        "miss_reason": (
            "MLIR 18's linalg-fuse-elementwise-ops fuses the iota producer into "
            "the map, but it leaves the reduction consumer materialized as a "
            "second linalg.generic in this textual pipeline."
        ),
    },
}


def main() -> int:
    toolchain = detect_toolchain()
    if not toolchain.has_standalone_mlir:
        print("standalone mlir-opt/mlir-translate are required", file=sys.stderr)
        return 1

    _check_artifact("docs/mlir-pipeline-cpu.txt", CPU_PIPELINE)
    _check_artifact("docs/mlir-pipeline-fusion.txt", FUSION_PIPELINE)

    for name, case in CASES.items():
        print(f"case: {name}")
        mlir = compile_source_to_mlir(case["source"], verify=False)
        fused = run_fusion_pipeline_text(mlir, toolchain=toolchain)
        cpu_lowered = run_cpu_pipeline_text(mlir, toolchain=toolchain)
        llvm_ir = translate_mlir_to_llvmir(cpu_lowered, toolchain=toolchain)

        before_count = mlir.count("linalg.generic")
        after_count = fused.count("linalg.generic")
        if before_count != case["before"]:
            raise PipelineUnavailable(
                f"{name}: expected {case['before']} linalg.generic ops before fusion, got {before_count}"
            )
        if after_count != case["after"]:
            raise PipelineUnavailable(
                f"{name}: expected {case['after']} linalg.generic ops after fusion, got {after_count}"
            )
        if "linalg.generic" in cpu_lowered:
            raise PipelineUnavailable(f"{name}: CPU pipeline left linalg.generic in output")
        if "llvm.func @main" not in cpu_lowered:
            raise PipelineUnavailable(f"{name}: CPU pipeline did not produce llvm.func @main")
        if "@main" not in llvm_ir:
            raise PipelineUnavailable(f"{name}: LLVM IR translation did not contain @main")
        print(f"  linalg.generic before fusion: {before_count}")
        print(f"  linalg.generic after fusion:  {after_count}")
        if "miss_reason" in case:
            print(f"  note: {case['miss_reason']}")

    nvidia_artifact = ROOT / "docs/mlir-pipeline-nvidia.txt"
    if not nvidia_artifact.is_file():
        print("missing docs/mlir-pipeline-nvidia.txt", file=sys.stderr)
        return 1
    nvidia_text = nvidia_artifact.read_text(encoding="utf-8")
    if GPU_NVIDIA_SCAFFOLD_NVVM_PIPELINE not in nvidia_text:
        raise PipelineUnavailable(
            "docs/mlir-pipeline-nvidia.txt does not mention the scaffold NVVM pipeline"
        )
    if GPU_NVIDIA_SCAFFOLD_LLVM_DIALECT_PIPELINE not in nvidia_text:
        raise PipelineUnavailable(
            "docs/mlir-pipeline-nvidia.txt does not mention the scaffold LLVM dialect pipeline"
        )
    print("nvidia pipeline artifact: scaffold-only until production gpu.module lowering lands")
    return 0


def _check_artifact(path: str, expected: str) -> None:
    artifact = ROOT / path
    if not artifact.is_file():
        raise PipelineUnavailable(f"missing {path}")
    text = artifact.read_text(encoding="utf-8").strip()
    if text != expected:
        raise PipelineUnavailable(f"{path} does not match remora.pipeline constant")


if __name__ == "__main__":
    raise SystemExit(main())
