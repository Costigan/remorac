"""Command-line entry points for Remora Dense Core."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from remora.codegen import CodegenUnavailable
from remora.compiler import compile_source_to_mlir, compile_source_to_ptx
from remora.errors import RemoraError
from remora.runtime import evaluate_source, format_value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remora Dense Core compiler")
    parser.add_argument("file", type=Path, help="Remora source file")
    parser.add_argument(
        "--target",
        choices=("cpu", "mlir", "ptx"),
        default="cpu",
        help="output target; cpu evaluates the program",
    )
    args = parser.parse_args(argv)

    try:
        source = args.file.read_text(encoding="utf-8")
        if args.target == "cpu":
            result = evaluate_source(source)
            print(format_value(result.value))
            return 0
        if args.target == "mlir":
            print(compile_source_to_mlir(source))
            return 0
        if args.target == "ptx":
            artifact = compile_source_to_ptx(source)
            print(artifact.ptx_text)
            return 0
        raise AssertionError(f"unknown target {args.target}")
    except (OSError, RemoraError, CodegenUnavailable) as exc:
        print(f"remorac: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
