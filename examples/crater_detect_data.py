"""Synthetic crater detection: data generation, target assignment, and decoding.

Pure-Python utilities for Phase 2 of the anchor-free crater detector.
No Remora compilation required for these functions — they generate
synthetic images and dense grid targets that can be consumed by a
Remora detector later.

Model contract:
  images:  [N, 1, 64, 64]  float32, ~[-1, 1]
  targets: [N, 8, 8, 4]    float32
  Each cell: [objectness (0|1), dx, dy, log_radius]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

TILE_SIZE = 64
GRID_SIZE = 8
CELL_SIZE = TILE_SIZE // GRID_SIZE  # 8
RADIUS_SCALE = float(CELL_SIZE)  # 8.0 — reference scale for log_radius


@dataclass(frozen=True)
class CraterParams:
    """A synthetic crater: center (x, y), radius in pixels."""
    cx: float
    cy: float
    radius: float


@dataclass(frozen=True)
class TargetInfo:
    """Diagnostic info about target assignment."""
    total_craters: int
    assigned: int
    conflicts: int
    out_of_bounds: int


def _ring_image(
    yx_grid: tuple[np.ndarray, np.ndarray],
    cx: float,
    cy: float,
    radius: float,
    rim_brightness: float = 0.9,
    center_darkness: float = 0.25,
    rim_width: float = 2.0,
) -> np.ndarray:
    """Return a crater ring image contribution (same shape as yx_grid[0]).

    Uses a bright ring and slightly darker interior.
    """
    yy, xx = yx_grid
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    inner = (radius - rim_width / 2)
    outer = (radius + rim_width / 2)
    ring = ((dist >= inner) & (dist <= outer)).astype(np.float32)
    center = (dist < inner).astype(np.float32)
    return rim_brightness * ring - center_darkness * center


def make_synthetic_image(
    craters: Sequence[CraterParams],
    *,
    noise_std: float = 0.04,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a single 64x64 grayscale crater image.

    Returns shape (1, 64, 64), float32, clipped to [-1, 1].
    """
    yy, xx = np.ogrid[:TILE_SIZE, :TILE_SIZE]
    rng = np.random.RandomState(seed)
    image = rng.normal(0.0, noise_std, size=(TILE_SIZE, TILE_SIZE)).astype(np.float32)
    for crater in craters:
        image += _ring_image((yy, xx), crater.cx, crater.cy, crater.radius)
    image = np.clip(image, -1.0, 1.0)
    return np.ascontiguousarray(image[np.newaxis, :, :])


def assign_targets(
    craters: Sequence[CraterParams],
) -> tuple[np.ndarray, TargetInfo]:
    """Assign crater parameters to an 8x8 grid, producing a dense target.

    Returns (target [8, 8, 4], TargetInfo).
    """
    target = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
    occupied: set[tuple[int, int]] = set()
    assigned = 0
    conflicts = 0
    out_of_bounds = 0

    for crater in craters:
        gx = int(np.floor(crater.cx / CELL_SIZE))
        gy = int(np.floor(crater.cy / CELL_SIZE))

        if not (0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE):
            out_of_bounds += 1
            continue

        key = (gy, gx)
        if key in occupied:
            conflicts += 1
            continue

        occupied.add(key)
        dx = (crater.cx - gx * CELL_SIZE) / CELL_SIZE
        dy = (crater.cy - gy * CELL_SIZE) / CELL_SIZE
        log_r = np.log(max(crater.radius, 1e-6) / RADIUS_SCALE)

        target[gy, gx, 0] = 1.0
        target[gy, gx, 1] = dx
        target[gy, gx, 2] = dy
        target[gy, gx, 3] = log_r
        assigned += 1

    info = TargetInfo(
        total_craters=len(craters),
        assigned=assigned,
        conflicts=conflicts,
        out_of_bounds=out_of_bounds,
    )
    return np.ascontiguousarray(target), info


def decode_target(target: np.ndarray) -> list[CraterParams]:
    """Decode a dense [8, 8, 4] target tensor back to crater parameters.

    Only cells with objectness >= 0.5 are decoded.
    """
    craters: list[CraterParams] = []
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            obj = float(target[gy, gx, 0])
            if obj < 0.5:
                continue
            dx = float(target[gy, gx, 1])
            dy = float(target[gy, gx, 2])
            log_r = float(target[gy, gx, 3])
            cx = (gx + dx) * CELL_SIZE
            cy = (gy + dy) * CELL_SIZE
            radius = RADIUS_SCALE * np.exp(log_r)
            craters.append(CraterParams(cx=cx, cy=cy, radius=radius))
    return craters


def make_detection_dataset(
    count: int = 8,
    *,
    min_craters: int = 1,
    max_craters: int = 4,
    min_radius: float = 3.0,
    max_radius: float = 12.0,
    seed: int = 1729,
) -> tuple[np.ndarray, np.ndarray, list[TargetInfo]]:
    """Generate a synthetic crater detection dataset.

    Returns (images [N, 1, 64, 64], targets [N, 8, 8, 4], list of TargetInfo).
    """
    if count < 1:
        raise ValueError("count must be >= 1")

    rng = np.random.RandomState(seed)
    images_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    infos: list[TargetInfo] = []

    for i in range(count):
        n_craters = rng.randint(min_craters, max_craters + 1)
        craters: list[CraterParams] = []
        for _ in range(n_craters):
            cx = rng.uniform(4.0, TILE_SIZE - 4.0)
            cy = rng.uniform(4.0, TILE_SIZE - 4.0)
            radius = rng.uniform(min_radius, max_radius)
            craters.append(CraterParams(cx=cx, cy=cy, radius=radius))

        image = make_synthetic_image(craters, seed=rng.randint(0, 2**31))
        target, info = assign_targets(craters)

        images_list.append(image)
        targets_list.append(target)
        infos.append(info)

    return (
        np.ascontiguousarray(np.stack(images_list, axis=0).astype(np.float32)),
        np.ascontiguousarray(np.stack(targets_list, axis=0).astype(np.float32)),
        infos,
    )


# ── Test helpers (also used by tests) ──────────────────────────────────────


def _max_center_error(original: list[CraterParams], decoded: list[CraterParams]) -> float:
    """Maximum Euclidean distance between paired original and decoded craters."""
    if len(original) != len(decoded):
        return float("inf")
    sorted_orig = sorted(original, key=lambda c: (c.cx, c.cy))
    sorted_dec = sorted(decoded, key=lambda c: (c.cx, c.cy))
    max_err = 0.0
    for o, d in zip(sorted_orig, sorted_dec):
        err = np.sqrt((o.cx - d.cx) ** 2 + (o.cy - d.cy) ** 2)
        max_err = max(max_err, err)
    return max_err


def _radius_error(original: list[CraterParams], decoded: list[CraterParams]) -> float:
    """Maximum absolute radius error between paired original and decoded craters."""
    sorted_orig = sorted(original, key=lambda c: (c.cx, c.cy))
    sorted_dec = sorted(decoded, key=lambda c: (c.cx, c.cy))
    max_err = 0.0
    for o, d in zip(sorted_orig, sorted_dec):
        max_err = max(max_err, abs(o.radius - d.radius))
    return max_err
