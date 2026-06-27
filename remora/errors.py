"""Common exceptions for the Remora prototype."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpan:
    file: str | None
    line: int
    col: int
    end_line: int | None = None
    end_col: int | None = None

    def format(self) -> str:
        prefix = f"{self.file}:" if self.file else ""
        if self.end_line is not None and self.end_col is not None:
            return f"{prefix}{self.line}:{self.col}-{self.end_line}:{self.end_col}"
        return f"{prefix}{self.line}:{self.col}"


class RemoraError(Exception):
    span: SourceSpan | None = None

    def located(self, span: SourceSpan | None) -> "RemoraError":
        self.span = span
        return self

    def __str__(self) -> str:
        base = super().__str__()
        if self.span is not None:
            return f"{self.span.format()}: {base}"
        return base
