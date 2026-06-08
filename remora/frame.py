"""Shared frame/cell decomposition for rank-polymorphic elaboration.

This module is the single owner of:
- cell-rank validation
- array suffix matching
- frame extraction from array types
- principal-frame selection for broadcasting
- cell-type candidates for implicit/explicit map lifting
- result-type framing
- replication/broadcasting obligations

Both dependent and non-dependent call sites should route through this module
so that frame/cell decisions are made consistently and can be recorded in the
structured typed core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from remora.ast_nodes import SourceLoc
from remora.index import DimExpr, DimLit
from remora.types import (
    ArrayType,
    FuncType,
    RemoraType,
    RemoraTypeError,
    ScalarType,
    enforce_rank_limit,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameCell:
    """A single resolved frame/cell decomposition decision.

    This record is designed to be stored in the typed core so that
    downstream passes (HIR lowering, AD, ...) do not need to
    rediscover the decomposition.
    """

    cell_type: RemoraType
    frame_shape: tuple[DimExpr, ...]

    @property
    def cell_rank(self) -> int:
        return self.cell_type.rank

    @property
    def frame_rank(self) -> int:
        return len(self.frame_shape)

    def __str__(self) -> str:
        return (
            f"FrameCell(cell={self.cell_type}, frame=["
            f"{','.join(str(d) for d in self.frame_shape)}])"
        )


# ---------------------------------------------------------------------------
# Cell-rank validation
# ---------------------------------------------------------------------------


def validate_cell_rank(
    array_rank: int, cell_rank: int, loc: SourceLoc | None = None
) -> None:
    """Raise if *cell_rank* cannot be a suffix of *array_rank*."""
    if array_rank < cell_rank:
        raise RemoraTypeError(
            f"array rank {array_rank} is too low for cell rank {cell_rank}",
            loc,
        )


# ---------------------------------------------------------------------------
# Suffix matching
# ---------------------------------------------------------------------------


def cell_matches_array_suffix(
    cell_type: RemoraType, array_type: RemoraType
) -> bool:
    """Return True when *cell_type* is a consistent suffix of *array_type*."""
    if isinstance(cell_type, FuncType) or isinstance(array_type, FuncType):
        return False
    if isinstance(cell_type, ScalarType):
        if isinstance(array_type, ScalarType):
            return cell_type == array_type
        return cell_type == array_type.element
    if isinstance(array_type, ScalarType):
        return False
    return (
        cell_type.element == array_type.element
        and cell_type.shape == array_type.shape[-cell_type.rank:]
    )


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def scalar_cell_and_frame(
    value_type: RemoraType, loc: SourceLoc | None = None
) -> tuple[ScalarType, tuple[DimExpr, ...]]:
    """Decompose *value_type* into a scalar element and its frame shape.

    This is the simplest decomposition: the cell is the innermost element
    type and the frame is everything else.
    """
    if isinstance(value_type, ScalarType):
        return value_type, ()
    if isinstance(value_type, ArrayType):
        return value_type.element, value_type.shape
    raise RemoraTypeError("map over function values is deferred", loc)


def decompose_argument(
    func_type: FuncType,
    array_type: RemoraType,
    loc: SourceLoc | None = None,
) -> FrameCell:
    """Decompose *array_type* into a frame and cell matching *func_type* params.

    This is the central decomposition used by map and implicit application
    inference.  It splits the array shape into a leading frame (the shared
    spatial dimensions) and a trailing cell (matching the function's
    parameter type).
    """
    if len(func_type.params) != 1:
        raise RemoraTypeError("map / lifting expects a unary function", loc)

    cell_type = func_type.params[0]
    cell_rank = cell_type.rank
    array_rank = array_type.rank

    validate_cell_rank(array_rank, cell_rank, loc)

    if not cell_matches_array_suffix(cell_type, array_type):
        raise RemoraTypeError(
            f"function cell type {cell_type} does not match {array_type}", loc
        )

    if isinstance(array_type, ArrayType):
        frame_shape: tuple[DimExpr, ...] = array_type.shape[
            : array_type.rank - cell_rank
        ]
    else:
        frame_shape = ()

    return FrameCell(cell_type, frame_shape)


# ---------------------------------------------------------------------------
# Result-type framing
# ---------------------------------------------------------------------------


def apply_frame(
    result_type: RemoraType, frame: tuple[DimExpr, ...]
) -> RemoraType:
    """Wrap *result_type* in the given frame shape.

    A scalar result under a non-empty frame becomes an ArrayType.
    An array result has the frame prepended to its shape.
    """
    if not frame:
        return result_type
    if isinstance(result_type, FuncType):
        raise RemoraTypeError("function-valued map results are deferred")
    if isinstance(result_type, ArrayType):
        return result_type.with_frame(frame)
    enforce_rank_limit(ArrayType(result_type, frame))
    return ArrayType(result_type, frame)


# ---------------------------------------------------------------------------
# Principal-frame selection
# ---------------------------------------------------------------------------


def principal_frame(
    frames: list[tuple[DimExpr, ...]], loc: SourceLoc | None = None
) -> tuple[DimExpr, ...] | None:
    """Determine the principal (longest) frame compatible with all given frames.

    The shorter frames must be prefixes of the longer frame (or empty).
    Returns *None* when frames are incompatible.  This is the authority for
    broadcasting replication obligations.
    """
    if not frames:
        return ()
    sorted_frames = sorted(frames, key=len, reverse=True)
    principal = sorted_frames[0]
    for other in sorted_frames[1:]:
        if not other:
            continue
        plen = len(principal)
        olen = len(other)
        if olen > plen:
            principal, other = other, principal
            plen, olen = olen, plen
        if principal[:olen] != other:
            return None
    return principal


# ---------------------------------------------------------------------------
# Cell-type candidates (for implicit/explicit map lifting)
# ---------------------------------------------------------------------------


def cell_type_candidates(
    value_type: RemoraType,
) -> list[RemoraType]:
    """Generate all valid cell types for *value_type* in rank order.

    For a scalar type there is exactly one candidate (the scalar itself).
    For an array type the candidates are:
      - the element type (scalar cell)
      - trailing-rank arrays from rank-1 up to the full array
    """
    if isinstance(value_type, ScalarType):
        return [value_type]
    candidates: list[RemoraType] = [value_type.element]
    for rank in range(1, value_type.rank + 1):
        candidates.append(
            ArrayType(value_type.element, value_type.shape[-rank:])
        )
    return candidates


# ---------------------------------------------------------------------------
# Convenience wrappers used during map / implicit-application elaboration
# ---------------------------------------------------------------------------


def infer_lifting(
    func_type: FuncType,
    array_type: RemoraType,
    loc: SourceLoc | None = None,
) -> tuple[tuple[DimExpr, ...], RemoraType]:
    """Decompose and frame the result type in one step (legacy compatibility).

    Returns ``(frame_shape, result_type)``.  New code should prefer
    :func:`decompose_argument` and :func:`apply_frame` directly.
    """
    fc = decompose_argument(func_type, array_type, loc)
    result_type = apply_frame(func_type.result, fc.frame_shape)
    return fc.frame_shape, result_type


# ---------------------------------------------------------------------------
# Broadcasting obligations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BroadcastObligation:
    """Record that a cell must be replicated across a principal frame."""

    cell_type: RemoraType
    cell_frame: tuple[DimExpr, ...]
    principal: tuple[DimExpr, ...]

    @property
    def needs_replication(self) -> bool:
        return len(self.cell_frame) < len(self.principal)


def broadcasting_obligations(
    left: FrameCell, right: FrameCell, loc: SourceLoc | None = None
) -> tuple[FrameCell, FrameCell, tuple[DimExpr, ...]] | None:
    """Resolve broadcasting for a binary operation.

    Returns ``(adjusted_left, adjusted_right, principal)`` where
    *adjusted_left* and *adjusted_right* carry the principal frame and
    *principal* is the shared frame.  Returns *None* when broadcasting
    is not possible.
    """
    principal = principal_frame(
        [left.frame_shape, right.frame_shape], loc
    )
    if principal is None:
        return None

    adjusted_left = FrameCell(left.cell_type, principal)
    adjusted_right = FrameCell(right.cell_type, principal)
    return adjusted_left, adjusted_right, principal
