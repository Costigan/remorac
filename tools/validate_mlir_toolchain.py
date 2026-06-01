"""Report and validate the local MLIR/LLVM toolchain."""

from __future__ import annotations

from pathlib import Path
import importlib.metadata as metadata
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.pipeline import detect_toolchain  # noqa: E402


def main() -> int:
    toolchain = detect_toolchain()
    tools = {
        "mlir-opt": toolchain.mlir_opt,
        "mlir-translate": toolchain.mlir_translate,
        "llc": toolchain.llc,
        "ptxas": toolchain.ptxas,
        "iree-opt": toolchain.iree_opt,
        "iree-compile": toolchain.iree_compile,
    }

    versions: dict[str, int | None] = {}
    for name, path in tools.items():
        print(f"{name}: {path or 'missing'}")
        if path is None:
            versions[name] = None
            continue
        text = _version_text(path)
        print(_first_nonempty_line(text))
        versions[name] = _llvm_major(text)

    print(f"iree.compiler.passmanager: {toolchain.iree_passmanager}")
    try:
        iree_version = metadata.version("iree-compiler")
    except metadata.PackageNotFoundError:
        iree_version = "missing"
    print(f"iree-compiler package: {iree_version}")

    required = ["mlir-opt", "mlir-translate", "llc"]
    missing = [name for name in required if tools[name] is None]
    if missing:
        print(f"missing required standalone tools: {', '.join(missing)}", file=sys.stderr)
        return 1

    standalone_majors = {versions[name] for name in required}
    if len(standalone_majors) != 1:
        print(
            f"standalone LLVM/MLIR tools do not share one major version: {standalone_majors}",
            file=sys.stderr,
        )
        return 1

    standalone_major = next(iter(standalone_majors))
    print(f"standalone LLVM/MLIR major: {standalone_major}")

    iree_majors = {versions[name] for name in ("iree-opt", "iree-compile") if versions[name] is not None}
    if iree_majors and iree_majors != {standalone_major}:
        print(
            "warning: IREE inspection tools use a different LLVM major "
            f"({sorted(iree_majors)}) than standalone MLIR ({standalone_major})",
            file=sys.stderr,
        )

    if tools["ptxas"] is None:
        print("warning: ptxas is missing; standalone PTX assembly checks are skipped", file=sys.stderr)

    return 0


def _version_text(path: str) -> str:
    result = subprocess.run(
        [path, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout or result.stderr


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return f"  {line.strip()}"
    return "  <no version output>"


def _llvm_major(text: str) -> int | None:
    match = re.search(r"LLVM version\s+(\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


if __name__ == "__main__":
    raise SystemExit(main())
