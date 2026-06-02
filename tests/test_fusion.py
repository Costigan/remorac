import importlib.util

import pytest

from remora.compiler import compile_source_to_mlir
from remora.pipeline import detect_toolchain, run_fusion_pipeline_text


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("iree") is None,
    reason="IREE compiler MLIR bindings are not installed",
)


MAP_FOLD_MISS_REASON = (
    "MLIR 18's linalg-fuse-elementwise-ops fuses the iota producer into the "
    "map, but it leaves the reduction consumer materialized as a second "
    "linalg.generic in this textual pipeline."
)


def fused_mlir(source: str) -> tuple[str, str]:
    toolchain = detect_toolchain()
    if toolchain.mlir_opt is None:
        pytest.skip("mlir-opt is not available")
    before = compile_source_to_mlir(source, verify=False)
    after = run_fusion_pipeline_text(before, toolchain=toolchain)
    return before, after


def test_map_chain_fuses_to_one_linalg_generic():
    before, after = fused_mlir("map (* 3) (map (* 2) (iota 10))")

    assert before.count("linalg.generic") == 3
    assert after.count("linalg.generic") == 1
    assert "arith.muli" in after


def test_dot_binary_map_and_fold_fuse_to_one_linalg_generic():
    source = (
        "let xs = [1.0, 2.0, 3.0] in "
        "let ys = [4.0, 5.0, 6.0] in "
        "dot xs ys"
    )

    before, after = fused_mlir(source)

    assert before.count("linalg.generic") == 2
    assert after.count("linalg.generic") == 1
    assert "arith.mulf" in after
    assert "arith.addf" in after


def test_map_then_fold_records_the_current_materialization_reason():
    before, after = fused_mlir("fold (+) 0.0 (map (* 2.0) (iota 10))")

    assert before.count("linalg.generic") == 3
    assert after.count("linalg.generic") == 2, MAP_FOLD_MISS_REASON
    assert "arith.mulf" in after
    assert "arith.addf" in after
