"""Prelude source helpers for Remora Dense Core."""

from __future__ import annotations

from pathlib import Path


PRELUDE_PATH = Path(__file__).parent.parent / "stdlib" / "prelude.rem"


def prelude_source() -> str:
    return "\n".join(prelude_definition_sources())


def prelude_definition_sources() -> list[str]:
    return [
        line.strip()
        for line in PRELUDE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("def ")
    ]


def with_prelude(source: str) -> str:
    source = _strip_leading_ignored_lines(source)
    prelude = prelude_source()
    return f"{prelude}\n{source}" if source else prelude


def _strip_leading_ignored_lines(source: str) -> str:
    lines = source.splitlines()
    while lines and (not lines[0].strip() or lines[0].strip().startswith("--")):
        lines.pop(0)
    return "\n".join(lines).strip()
