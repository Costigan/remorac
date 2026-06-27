"""Cost-annotation data structures (definitions only).

These are inert definitions for Workstream 7.  No scheduling decisions,
no cost math, no behavior changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StaticOrSymbolic = int | str


@dataclass(frozen=True)
class CostShape:
    elem_count: StaticOrSymbolic
    bytes_read: StaticOrSymbolic
    bytes_written: StaticOrSymbolic
    flops: StaticOrSymbolic
    temporary_bytes: StaticOrSymbolic
    irregularity: Literal["regular", "segmented", "ragged", "dynamic-call"]


@dataclass(frozen=True)
class ScheduleCandidate:
    backend: Literal["cpu", "gpu"]
    plan_kind: str
    estimated_cost: object
    requirements: list[str]
    fallback_reason: str | None
