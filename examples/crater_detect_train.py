"""Train an anchor-free crater grid detector on synthetic data.

Uses a single compiled value-and-grad function returning gradients for four
kernels (objectness, dx, dy, log_radius), plus shared bias b1, weight w2,
and bias b2 — 7 trainable parameters total.  Python accumulates gradients
across a mini-batch and updates parameters with SGD.

Model contract:
  images:       [1, 64, 64]  float32
  target grid:  [8, 8, 4]    float32  (objectness, dx, dy, log_radius)
  Output:       [64] × 4     float32  (one per channel, flattened)
  loss:         Float
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from examples.crater_detect_data import (
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
PATCH_COUNT = ((IMAGE_SIZE - KERNEL_SIZE) // STRIDE + 1) ** 2  # 64

CENTER_WEIGHT = 1.0
RADIUS_WEIGHT = 0.1

DATA_SEED = 1729
PARAMETER_SEED = 22022022

# ── Remora source (programmatically built for correct paren balancing) ──────


def _D(k_name: str) -> str:
    """One detector channel (3×3 dot over im2col + ReLU + w2 * + b2) → [64]."""
    return (
        f"(+ (* w2 (map relu (+ (map (lambda (p) (fold + 0.0 (map * p (ravel {k_name})))) "
        f"(im2col image [{KERNEL_SIZE} {KERNEL_SIZE}] {STRIDE})) b1))) b2)"
    )


def _bce(expr: str, target: str) -> str:
    """Stable BCE elementwise on [64] arrays — all primitive ops, no lambdas."""
    return (
        f"(+ (map relu {expr})"
        f" (+ (* -1.0 (* {expr} {target}))"
        f" (log (+ 1.0 (exp (- 0.0 (map absv {expr})))))))"
    )


def _masked_l2(expr: str, target: str, mask: str) -> str:
    """Masked L2 loss: sum(mask * (expr - target)^2)."""
    return f"(fold + 0.0 (* {mask} (* (- {expr} {target}) (- {expr} {target}))))"


def _build_detector_source() -> str:
    bce_loss = f"(fold + 0.0 {_bce(_D('k_obj'), 'target_obj')})"
    center = (
        f"(* {CENTER_WEIGHT} (+"
        f" {_masked_l2(_D('k_dx'), 'target_dx', 'target_obj')}"
        f" {_masked_l2(_D('k_dy'), 'target_dy', 'target_obj')}))"
    )
    radius = f"(* {RADIUS_WEIGHT} {_masked_l2(_D('k_logr'), 'target_logr', 'target_obj')})"
    body = f"(+ {bce_loss} (+ {center} {radius}))"

    params = " ".join([
        "k_obj (Array Float 3 3)",
        "k_dx (Array Float 3 3)",
        "k_dy (Array Float 3 3)",
        "k_logr (Array Float 3 3)",
        "b1 Float w2 Float b2 Float",
        f"image (Array Float {IMAGE_SIZE} {IMAGE_SIZE})",
        f"target_obj (Array Float {PATCH_COUNT})",
        f"target_dx (Array Float {PATCH_COUNT})",
        f"target_dy (Array Float {PATCH_COUNT})",
        f"target_logr (Array Float {PATCH_COUNT})",
    ])

    return "\n".join([
        "",
        "(define/pi ()",
        "  (relu [v Float] Float)",
        "  (select (> v 0.0) v 0.0))",
        "",
        "(define/pi ()",
        "  (absv [v Float] Float)",
        "  (select (> v 0.0) v (- 0.0 v)))",
        "",
        "(define/pi ()",
        f"  (detect-loss [{params}] Float)",
        f"  {body})",
        "",
    ])


_DETECTOR_LISP_SRC = _build_detector_source()

# ── data helpers ────────────────────────────────────────────────────────────


def _extract_channels(target_grid: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return four [64] channels from a [8,8,4] target grid."""
    return (
        np.ascontiguousarray(target_grid[..., 0].ravel()),
        np.ascontiguousarray(target_grid[..., 1].ravel()),
        np.ascontiguousarray(target_grid[..., 2].ravel()),
        np.ascontiguousarray(target_grid[..., 3].ravel()),
    )


def make_dataset(
    n: int = 4,
    *,
    seed: int = DATA_SEED,
) -> tuple[
    list[np.ndarray],
    list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    list[TargetInfo],
]:
    """Generate a synthetic crater detection dataset.

    Returns
        images   list of [64,64] float32
        targets  list of (obj[64], dx[64], dy[64], logr[64])
        infos    diagnostic summary per example.
    """
    n_craters = max(n // 2 + 1, 1)
    rng = np.random.RandomState(seed)
    images: list[np.ndarray] = []
    targets: list[tuple[np.ndarray, ...]] = []
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
        targets.append(_extract_channels(target_grid))
        infos.append(info)

    return images, targets, infos


def initialize_parameters(
    *, seed: int = PARAMETER_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.float32, np.float32, np.float32]:
    rng = np.random.RandomState(seed)
    kernels = tuple(
        np.ascontiguousarray(
            rng.randn(KERNEL_SIZE, KERNEL_SIZE).astype(np.float32) * 0.1
        )
        for _ in range(4)
    )
    return (*kernels, np.float32(0.0), np.float32(rng.randn() * 0.1), np.float32(0.0))


# ── parameter types ─────────────────────────────────────────────────────────


def _parameter_types() -> tuple[RemoraType, ...]:
    return (
        *(ArrayType(FLOAT, (StaticDim(KERNEL_SIZE), StaticDim(KERNEL_SIZE))),) * 4,
        FLOAT, FLOAT, FLOAT,
        ArrayType(FLOAT, (StaticDim(IMAGE_SIZE), StaticDim(IMAGE_SIZE))),
        *(ArrayType(FLOAT, (StaticDim(PATCH_COUNT),)),) * 4,
    )


# ── compiled functions ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class DetectorCompiledFunctions:
    """Compiled forward and value-and-grad (7 trainable parameters)."""

    _loss_exe: CPUFunctionExecutor
    _grad_exe: CPUFunctionExecutor

    def forward(
        self,
        kernels: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        b1: float,
        w2: float,
        b2: float,
        image: np.ndarray,
        channels: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        return float(np.asarray(self._loss_exe.execute(
            *kernels, np.float32(b1), np.float32(w2), np.float32(b2),
            image, *channels,
        ).value))

    def gradients(
        self,
        kernels: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        b1: float,
        w2: float,
        b2: float,
        image: np.ndarray,
        channels: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
               np.float32, np.float32, np.float32]:
        result = self._grad_exe.execute(
            *kernels, np.float32(b1), np.float32(w2), np.float32(b2),
            image, *channels,
        )

        def _flatten(value):
            if isinstance(value, (list, tuple)):
                out = []
                for v in value:
                    out.extend(_flatten(v))
                return out
            return [value]

        flat = _flatten(result.value)
        # Pair order: g_k_obj, g_k_dx, g_k_dy, g_k_logr, g_b1, g_w2, g_b2
        g_ks = tuple(np.asarray(flat[i], dtype=np.float32) for i in range(4))
        g_b1 = np.float32(flat[4])
        g_w2 = np.float32(flat[5])
        g_b2 = np.float32(flat[6])
        return g_ks, g_b1, g_w2, g_b2

    @staticmethod
    def compile() -> DetectorCompiledFunctions:
        param_types = _parameter_types()
        grad_artifact = generate_value_and_grad_function_source(
            _DETECTOR_LISP_SRC,
            "detect-loss",
            param_types,
            differentiate_inputs=tuple(range(7)),
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
    parameters: tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        np.float32, np.float32, np.float32,
    ]
    loss_history: tuple[float, ...]
    compile_seconds: float
    mean_step_seconds: float
    compiled: bool


def train(
    *,
    n: int = 8,
    batch_size: int = 4,
    epochs: int = 10,
    learning_rate: float = 0.01,
    data_seed: int = DATA_SEED,
    parameter_seed: int = PARAMETER_SEED,
    verbose: bool = True,
) -> DetectorTrainingResult:
    images, targets_list, infos = make_dataset(n=n, seed=data_seed)
    k_obj, k_dx, k_dy, k_logr, b1, w2, b2 = initialize_parameters(
        seed=parameter_seed,
    )

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
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            step_start = perf_counter()

            g_k_sum = [np.zeros_like(k_obj) for _ in range(4)]
            g_b1_sum = 0.0
            g_w2_sum = 0.0
            g_b2_sum = 0.0
            for idx in batch_idx:
                x_img = images[idx]
                ch = targets_list[idx]
                kernels = (k_obj, k_dx, k_dy, k_logr)
                loss_i = compiled.forward(kernels, float(b1), float(w2), float(b2), x_img, ch)
                epoch_loss += loss_i

                g_ks, g_b1, g_w2, g_b2 = compiled.gradients(
                    kernels, float(b1), float(w2), float(b2), x_img, ch,
                )
                for c in range(4):
                    g_k_sum[c] += g_ks[c]
                g_b1_sum += float(g_b1)
                g_w2_sum += float(g_w2)
                g_b2_sum += float(g_b2)

            bn = len(batch_idx)
            lr_bn = learning_rate / bn
            k_obj = np.asarray(k_obj - lr_bn * g_k_sum[0], dtype=np.float32)
            k_dx = np.asarray(k_dx - lr_bn * g_k_sum[1], dtype=np.float32)
            k_dy = np.asarray(k_dy - lr_bn * g_k_sum[2], dtype=np.float32)
            k_logr = np.asarray(k_logr - lr_bn * g_k_sum[3], dtype=np.float32)
            for arr in (k_obj, k_dx, k_dy, k_logr):
                arr = np.ascontiguousarray(arr)
            b1 = np.float32(b1 - lr_bn * g_b1_sum)
            w2 = np.float32(w2 - lr_bn * g_w2_sum)
            b2 = np.float32(b2 - lr_bn * g_b2_sum)

            step_seconds.append(perf_counter() - step_start)

        mean_loss = epoch_loss / n
        loss_history.append(mean_loss)
        if verbose:
            print(f"epoch {epoch:3d} loss {mean_loss:.6f}")

    mean_step_seconds = float(np.mean(step_seconds)) if step_seconds else 0.0
    if verbose:
        print(f"compile_seconds={compile_seconds:.1f} "
              f"mean_step_seconds={mean_step_seconds:.3f}")
        print(f"initial loss {loss_history[0]:.6f} -> final {loss_history[-1]:.6f}")

    return DetectorTrainingResult(
        parameters=((k_obj, k_dx, k_dy, k_logr), b1, w2, b2),
        loss_history=tuple(loss_history),
        compile_seconds=compile_seconds,
        mean_step_seconds=mean_step_seconds,
        compiled=True,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DATA_SEED)
    args = parser.parse_args()

    train(
        n=args.examples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        data_seed=args.seed,
        parameter_seed=PARAMETER_SEED,
        verbose=True,
    )


if __name__ == "__main__":
    main()
