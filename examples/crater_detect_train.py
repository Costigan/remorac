"""Train an anchor-free crater grid detector (objectness-only) on synthetic data.

Uses a single compiled value-and-grad function returning gradients for kernel
k, bias b1, weight w2, and bias b2.  Python accumulates gradients over a
static batch and updates parameters with SGD.

Model contract:
  images:  [1, 64, 64]  float32
  targets: [64]          float32  (objectness, 0 or 1)
  logits:  [64]          float32  (unbounded)
  loss:    Float
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from examples.crater_detect_data import (
    CELL_SIZE,
    CraterParams,
    TargetInfo,
    assign_targets,
    make_synthetic_image,
)
from remora.ad_source import generate_value_and_grad_function_source
from remora.runtime import CPUFunctionExecutor
from remora.types import ArrayType, FLOAT, RemoraType, StaticDim

# ── hyper-parameters ────────────────────────────────────────────────────────

IMAGE_SIZE = 64
GRID_SIZE = 8
KERNEL_SIZE = 3
STRIDE = 8
PATCH_SIZE = KERNEL_SIZE ** 2  # 9
PATCHES_PER_AXIS = (IMAGE_SIZE - KERNEL_SIZE) // STRIDE + 1  # 8
PATCH_COUNT = PATCHES_PER_AXIS ** 2  # 64
DATA_SEED = 1729
PARAMETER_SEED = 22022022

TRAINABLE_NAMES = ("k", "b1", "w2", "b2")

# ── Remora source ───────────────────────────────────────────────────────────

_D = (
    f"(+ (* w2 (map relu (+ (map (lambda (p) (fold + 0.0 (map * p (ravel k)))) "
    f"(im2col image [{KERNEL_SIZE} {KERNEL_SIZE}] {STRIDE})) b1))) b2)"
)

_DETECTOR_LISP_SRC = f"""
(define/pi ()
  (relu [v Float] Float)
  (select (> v 0.0) v 0.0))

(define/pi ()
  (absv [v Float] Float)
  (select (> v 0.0) v (- 0.0 v)))

(define/pi ()
  (detect-loss [k (Array Float {KERNEL_SIZE} {KERNEL_SIZE}) b1 Float w2 Float b2 Float image (Array Float {IMAGE_SIZE} {IMAGE_SIZE}) target (Array Float {PATCH_COUNT})] Float)
  (fold + 0.0
    (+ (map relu {_D})
       (+ (* -1.0 (* {_D} target))
          (log (+ 1.0 (exp (- 0.0 (map absv {_D})))))))))
"""

# ── data helpers ────────────────────────────────────────────────────────────


def _objectness_target(target_grid: np.ndarray) -> np.ndarray:
    """Extract objectness channel from [8,8,4] and flatten to [64]."""
    return np.ascontiguousarray(target_grid[..., 0].ravel())


def make_dataset(
    n: int = 4,
    *,
    seed: int = DATA_SEED,
) -> tuple[list[np.ndarray], list[np.ndarray], list[TargetInfo]]:
    """Generate a synthetic crater detection dataset.

    Returns
        images   list of [64,64] float32  (squeezed from [1,64,64])
        targets  list of [64]   float32  (objectness channel, flattened)
        infos    diagnostic summary per example.
    """
    n_craters = max(n // 2 + 1, 1)
    rng = np.random.RandomState(seed)
    images: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    infos: list[TargetInfo] = []

    for i in range(n):
        craters: list[CraterParams] = []
        for _ in range(n_craters):
            cx = rng.uniform(4.0, IMAGE_SIZE - 4.0)
            cy = rng.uniform(4.0, IMAGE_SIZE - 4.0)
            radius = rng.uniform(3.0, 12.0)
            craters.append(CraterParams(cx=cx, cy=cy, radius=radius))

        img_1 = make_synthetic_image(craters, seed=rng.randint(0, 2**31))
        target_grid, info = assign_targets(craters)

        images.append(np.ascontiguousarray(img_1[0].astype(np.float32)))
        targets.append(_objectness_target(target_grid))
        infos.append(info)

    return images, targets, infos


def _objectness_target_grid(targets_flat: np.ndarray) -> np.ndarray:
    """Reshape [64] back to [8,8] with zeros for the remaining 3 channels."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
    grid[..., 0] = targets_flat.reshape(GRID_SIZE, GRID_SIZE)
    return grid


def initialize_parameters(
    *, seed: int = PARAMETER_SEED
) -> tuple[np.ndarray, np.float32, np.float32, np.float32]:
    rng = np.random.RandomState(seed)
    k = np.ascontiguousarray(rng.randn(KERNEL_SIZE, KERNEL_SIZE).astype(np.float32) * 0.1)
    b1 = np.float32(0.0)
    w2 = np.float32(rng.randn() * 0.1)
    b2 = np.float32(0.0)
    return k, b1, w2, b2


# ── parameter types ─────────────────────────────────────────────────────────


def _parameter_types() -> tuple[RemoraType, ...]:
    return (
        ArrayType(FLOAT, (StaticDim(KERNEL_SIZE), StaticDim(KERNEL_SIZE))),
        FLOAT,
        FLOAT,
        FLOAT,
        ArrayType(FLOAT, (StaticDim(IMAGE_SIZE), StaticDim(IMAGE_SIZE))),
        ArrayType(FLOAT, (StaticDim(PATCH_COUNT),)),
    )


# ── compiled functions ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class DetectorCompiledFunctions:
    """Compiled forward and value-and-grad function for native CPU execution."""

    _loss_exe: CPUFunctionExecutor
    _grad_exe: CPUFunctionExecutor

    def forward(
        self,
        k: np.ndarray,
        b1: float,
        w2: float,
        b2: float,
        image: np.ndarray,
        target_flat: np.ndarray,
    ) -> float:
        return float(np.asarray(self._loss_exe.execute(
            k, np.float32(b1), np.float32(w2), np.float32(b2), image, target_flat,
        ).value))

    def gradients(
        self,
        k: np.ndarray,
        b1: float,
        w2: float,
        b2: float,
        image: np.ndarray,
        target_flat: np.ndarray,
    ) -> tuple[np.ndarray, np.float32, np.float32, np.float32]:
        result = self._grad_exe.execute(
            k, np.float32(b1), np.float32(w2), np.float32(b2), image, target_flat,
        )

        def _flatten(value):
            if isinstance(value, (list, tuple)):
                out = []
                for v in value:
                    out.extend(_flatten(v))
                return out
            return [value]

        flat = _flatten(result.value)
        # Pair order: g_k, g_b1, g_w2, g_b2
        g_k = np.asarray(flat[0], dtype=np.float32)
        g_b1 = np.float32(flat[1])
        g_w2 = np.float32(flat[2])
        g_b2 = np.float32(flat[3])
        return g_k, g_b1, g_w2, g_b2

    @staticmethod
    def compile() -> DetectorCompiledFunctions:
        param_types = _parameter_types()
        grad_artifact = generate_value_and_grad_function_source(
            _DETECTOR_LISP_SRC,
            "detect-loss",
            param_types,
            differentiate_inputs=(0, 1, 2, 3),
            include_prelude=False,
            syntax="lisp",
        )
        grad_exe = CPUFunctionExecutor.compile_source(
            grad_artifact.source,
            grad_artifact.function_name,
            grad_artifact.param_types,
            include_prelude=False,
            syntax="lisp",
        )
        loss_exe = CPUFunctionExecutor.compile_source(
            _DETECTOR_LISP_SRC,
            "detect-loss",
            param_types,
            include_prelude=False,
            syntax="lisp",
        )
        return DetectorCompiledFunctions(
            _loss_exe=CPUFunctionExecutor(loss_exe),
            _grad_exe=CPUFunctionExecutor(grad_exe),
        )


# ── training ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DetectorTrainingResult:
    parameters: tuple[np.ndarray, np.float32, np.float32, np.float32]
    loss_history: tuple[float, ...]
    compile_seconds: float
    mean_step_seconds: float


def train(
    *,
    n: int = 8,
    epochs: int = 10,
    learning_rate: float = 0.05,
    data_seed: int = DATA_SEED,
    parameter_seed: int = PARAMETER_SEED,
    verbose: bool = True,
) -> DetectorTrainingResult:
    images, targets, infos = make_dataset(n=n, seed=data_seed)
    k, b1, w2, b2 = initialize_parameters(seed=parameter_seed)

    compile_start = perf_counter()
    compiled = DetectorCompiledFunctions.compile()
    compile_seconds = perf_counter() - compile_start
    if verbose:
        print(f"Compiled native execution ready ({compile_seconds:.1f}s)")

    loss_history: list[float] = []
    step_seconds: list[float] = []

    for epoch in range(epochs):
        indices = np.random.RandomState(epoch + parameter_seed).permutation(n)
        epoch_loss = 0.0
        for idx in indices:
            step_start = perf_counter()
            x_img = images[idx]

            loss = compiled.forward(k, float(b1), float(w2), float(b2), x_img, targets[idx])
            epoch_loss += loss

            g_k, g_b1, g_w2, g_b2 = compiled.gradients(
                k, float(b1), float(w2), float(b2), x_img, targets[idx],
            )

            if not np.all(np.isfinite(g_k)):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            if not (np.isfinite(g_b1) and np.isfinite(g_w2) and np.isfinite(g_b2)):
                raise FloatingPointError(f"non-finite scalar gradient at epoch {epoch}")

            k = np.asarray(k - learning_rate * g_k, dtype=np.float32)
            k = np.ascontiguousarray(k)
            b1 = np.float32(b1 - learning_rate * float(g_b1))
            w2 = np.float32(w2 - learning_rate * float(g_w2))
            b2 = np.float32(b2 - learning_rate * float(g_b2))

            step_seconds.append(perf_counter() - step_start)

        mean_loss = epoch_loss / n
        loss_history.append(mean_loss)
        if verbose:
            print(f"epoch {epoch:3d} loss {mean_loss:.6f}")

    mean_step_seconds = float(np.mean(step_seconds)) if step_seconds else 0.0
    if verbose:
        print(f"compile_seconds={compile_seconds:.1f} mean_step_seconds={mean_step_seconds:.3f}")
        print(f"initial loss {loss_history[0]:.6f} -> final {loss_history[-1]:.6f}")

    return DetectorTrainingResult(
        parameters=(k, b1, w2, b2),
        loss_history=tuple(loss_history),
        compile_seconds=compile_seconds,
        mean_step_seconds=mean_step_seconds,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DATA_SEED)
    args = parser.parse_args()

    train(
        n=args.examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        data_seed=args.seed,
        parameter_seed=PARAMETER_SEED,
        verbose=True,
    )


if __name__ == "__main__":
    main()
