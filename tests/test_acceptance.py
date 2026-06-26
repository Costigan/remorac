import json
import os
from pathlib import Path

from remora.cli import main


ACCEPTANCE_DIR = Path(__file__).parent / "acceptance"

# GPU acceptance cases require a live CUDA runtime. When REMORA_TEST_GPU is
# not explicitly "1" (CI sets it to "0"), degrade GPU cases to a skip rather
# than a failure — consistent with the conftest GPU-gating policy.
_GPU_REQUIRED = os.environ.get("REMORA_TEST_GPU") == "1"


def load_cases():
    return json.loads((ACCEPTANCE_DIR / "manifest.json").read_text(encoding="utf-8"))


def test_acceptance_manifest_cases(tmp_path, capsys):
    for case_num, case in enumerate(load_cases()):
        assert case["category"] in {"supported", "rejected", "deferred"}, case["name"]
        source = ACCEPTANCE_DIR / case["path"]

        if case["target"] == "cuda" and not _GPU_REQUIRED:
            continue

        args = ["--target", case["target"], str(source)]
        args.extend(case.get("args", []))
        if case["target"] == "cpu":
            args = ["--shared", "-o", str(tmp_path / f"case_{case_num}.so")] + args

        exit_code = main(args)
        captured = capsys.readouterr()

        assert exit_code == case["expect_exit"], case["name"]
        if case["category"] == "supported":
            assert exit_code == 0, case["name"]
        else:
            assert exit_code != 0, case["name"]
        if "expect_stdout" in case:
            assert captured.out == case["expect_stdout"], case["name"]
            assert captured.err == "", case["name"]
        if "expect_stderr_contains" in case:
            assert captured.out == "", case["name"]
            assert case["expect_stderr_contains"] in captured.err, case["name"]


def test_deferred_acceptance_cases_are_manifested_as_deferred():
    manifest_paths = {case["path"] for case in load_cases()}
    deferred_manifest_paths = {
        case["path"] for case in load_cases() if case["category"] == "deferred"
    }
    deferred_paths = {
        str(path.relative_to(ACCEPTANCE_DIR))
        for path in (ACCEPTANCE_DIR / "deferred").glob("*.remora")
    }

    assert deferred_paths <= manifest_paths
    assert deferred_paths == deferred_manifest_paths
