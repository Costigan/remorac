"""Train the Section 7 crater CNN on a tiny deterministic dataset.

Images are float32 and normalized to approximately [-1, 1]. Labels are
float32 scalars: 1.0 means crater and 0.0 means no crater.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Callable

import numpy as np

from remora.ad_source import generate_gradient_function_source
from remora.lisp_reader import parse_lisp
from remora.runtime import _lambda_callable
from remora.typechecker import TypeChecker
from remora.types import ArrayType, FLOAT, FuncType, RemoraType, StaticDim


DATA_SEED = 1729
PARAMETER_SEED = 2718
TRAINABLE_NAMES = ("k", "b1", "w2", "b2", "w3", "b3")  # mask, x, y excluded
DROPOUT_SIZE = 900  # conv output size, where dropout is applied

_CNN_FULL_LISP_SRC = """
(define/pi ()
  (dot-patch [patch (Array Float 9) flat-k (Array Float 9)] Float)
  (fold + 0.0 (map * patch flat-k)))

(define/pi ()
  (conv2d [image (Array Float 32 32) kernel (Array Float 3 3) bias Float] (Array Float 900))
  (+ (map (lambda (p) (dot-patch p (ravel kernel))) (im2col image [3 3] 1)) bias))

(define/pi ()
  (dot-row [row (Array Float 900) x (Array Float 900)] Float)
  (fold + 0.0 (map * row x)))

(define/pi ()
  (linear [w (Array Float 4 900) x (Array Float 900)] (Array Float 4))
  (map (lambda (row) (dot-row row x)) w))

(define/pi ()
  (relu [v Float] Float)
  (select (> v 0.0) v 0.0))

(define/pi ()
  (bce [logit Float y Float] Float)
  (+ (select (> logit 0.0) logit 0.0)
     (+ (* -1.0 (* logit y))
        (log (+ 1.0 (exp (- 0.0 (select (> logit 0.0) logit (- 0.0 logit)))))))))

(define/pi ()
  (cnn-loss [k (Array Float 3 3) b1 Float w2 (Array Float 4 900) b2 (Array Float 4) w3 (Array Float 4) b3 Float mask (Array Float 900) x (Array Float 32 32) y Float] Float)
  (bce (+ (fold + 0.0 (* w3 (map relu (+ (linear w2 (* (map relu (conv2d x k b1)) mask)) b2)))) b3) y))
"""


@dataclass(frozen=True)
class TrainingResult:
    parameters: tuple[np.ndarray, ...]
    loss_history: tuple[float, ...]
    compile_seconds: float
    mean_step_seconds: float
    checkpoints: dict[int, tuple[np.ndarray, ...]]


def make_dropout_mask(
    size: int = DROPOUT_SIZE,
    *,
    keep_prob: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Return a flat dropout mask in {0.0, 1/keep_prob} for inverted dropout."""
    if not 0.0 < keep_prob <= 1.0:
        raise ValueError("keep_prob must be in (0, 1]")
    rng = np.random.RandomState(seed) if seed is not None else np.random
    mask = (rng.random(size) < keep_prob).astype(np.float32)
    return mask / keep_prob


def make_inference_mask(size: int = DROPOUT_SIZE) -> np.ndarray:
    """Return an all-ones mask with no scaling (no dropout at inference)."""
    return np.ones(size, dtype=np.float32)


def make_tiny_dataset(
    count: int = 8, *, seed: int = DATA_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Return balanced noisy images with or without a fixed crater-like ring."""
    if count < 2 or count % 2:
        raise ValueError("dataset count must be an even integer of at least 2")

    rng = np.random.RandomState(seed)
    images = rng.normal(0.0, 0.04, size=(count, 32, 32)).astype(np.float32)
    labels = np.zeros(count, dtype=np.float32)
    yy, xx = np.ogrid[:32, :32]
    radius = np.sqrt((yy - 15.5) ** 2 + (xx - 15.5) ** 2)
    ring = ((radius >= 5.0) & (radius <= 7.0)).astype(np.float32)
    center = (radius < 4.0).astype(np.float32)

    for index in range(count):
        if index % 2 == 0:
            images[index] += 0.9 * ring - 0.25 * center
            labels[index] = 1.0
        else:
            images[index] += -0.9 * ring + 0.25 * center

    images = np.clip(images, -1.0, 1.0)
    return np.ascontiguousarray(images), np.ascontiguousarray(labels)


def initialize_parameters(
    *, hidden_size: int = 4, seed: int = PARAMETER_SEED
) -> tuple[np.ndarray, ...]:
    """Initialize the six trainable tensors with deterministic float32 values."""
    if hidden_size != 4:
        raise ValueError("the current specialized CNN source requires hidden_size=4")
    rng = np.random.RandomState(seed)
    return (
        np.ascontiguousarray(rng.normal(0.08, 0.05, (3, 3)).astype(np.float32)),
        np.asarray(0.01, dtype=np.float32),
        np.ascontiguousarray(
            rng.normal(0.0, np.sqrt(2.0 / 900.0), (4, 900)).astype(np.float32)
        ),
        np.full(4, 0.01, dtype=np.float32),
        np.ascontiguousarray(rng.normal(0.0, 0.25, 4).astype(np.float32)),
        np.asarray(0.0, dtype=np.float32),
    )


def _parameter_types() -> tuple[RemoraType, ...]:
    return (
        ArrayType(FLOAT, (StaticDim(3), StaticDim(3))),
        FLOAT,
        ArrayType(FLOAT, (StaticDim(4), StaticDim(900))),
        ArrayType(FLOAT, (StaticDim(4),)),
        ArrayType(FLOAT, (StaticDim(4),)),
        FLOAT,
        ArrayType(FLOAT, (StaticDim(DROPOUT_SIZE),)),
        ArrayType(FLOAT, (StaticDim(32), StaticDim(32))),
        FLOAT,
    )


def _prepare_interpreted_function(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    result_type: RemoraType,
) -> Callable[..., object]:
    """Parse once and return the interpreter callable bound by the definition."""
    checker = TypeChecker()
    checker.check_program(parse_lisp(source))
    function = checker._functions.get(function_name)
    if function is None:
        raise ValueError(f"function {function_name!r} was not defined")
    specialized = checker._typed_top_level_function(
        function,
        FuncType(param_types, result_type),
        checker._build_prelude_env(),
    )
    return _lambda_callable(specialized, {})


@lru_cache(maxsize=1)
def _compile_training_functions() -> tuple[
    Callable[..., object], list[Callable[..., object]]
]:
    param_types = _parameter_types()
    gradient_sources = []
    for i in range(len(TRAINABLE_NAMES)):
        g = generate_gradient_function_source(
            _CNN_FULL_LISP_SRC,
            "cnn-loss",
            param_types,
            differentiate_input=i,
            include_prelude=False,
            syntax="lisp",
        )
        gradient_sources.append(g)

    forward = _prepare_interpreted_function(
        _CNN_FULL_LISP_SRC, "cnn-loss", param_types, FLOAT
    )
    gradients = [
        _prepare_interpreted_function(
            g.source,
            g.function_name,
            param_types,
            param_types[index],
        )
        for index, g in enumerate(gradient_sources)
    ]
    return forward, gradients


def train_tiny_dataset(
    *,
    epochs: int = 60,
    learning_rate: float = 0.1,
    example_count: int = 8,
    data_seed: int = DATA_SEED,
    parameter_seed: int = PARAMETER_SEED,
    checkpoint_every: int = 10,
    dropout_keep_prob: float = 0.5,
    dropout_seed: int = 42,
    verbose: bool = True,
) -> TrainingResult:
    """Run one-example SGD and return losses, timings, and parameter checkpoints."""
    images, labels = make_tiny_dataset(example_count, seed=data_seed)
    parameters = list(initialize_parameters(seed=parameter_seed))

    compile_start = perf_counter()
    forward, gradient_functions = _compile_training_functions()
    compile_seconds = perf_counter() - compile_start

    mask_rng = np.random.RandomState(dropout_seed)

    def arguments_with_dropout(image: np.ndarray, label: np.float32) -> tuple[object, ...]:
        mask = make_dropout_mask(keep_prob=dropout_keep_prob, seed=mask_rng.randint(0, 2**31))
        return (*parameters, mask, image, np.asarray(label, dtype=np.float32))

    def mean_loss() -> float:
        inference_mask = make_inference_mask()
        losses = [
            float(forward(*(*parameters, inference_mask, image, np.asarray(label, dtype=np.float32))))
            for image, label in zip(images, labels)
        ]
        return float(np.mean(losses))

    loss_history = [mean_loss()]
    checkpoints = {0: tuple(np.array(value, copy=True) for value in parameters)}
    step_seconds: list[float] = []

    for epoch in range(1, epochs + 1):
        for image, label in zip(images, labels):
            step_start = perf_counter()
            step_args = arguments_with_dropout(image, label)
            gradients = [
                np.asarray(gradient(*step_args), dtype=np.float32)
                for gradient in gradient_functions
            ]
            if not all(np.all(np.isfinite(gradient)) for gradient in gradients):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            updated_parameters = []
            for parameter, gradient in zip(parameters, gradients):
                updated = np.asarray(
                    parameter - learning_rate * gradient, dtype=np.float32
                )
                if updated.ndim > 0:
                    updated = np.ascontiguousarray(updated)
                updated_parameters.append(updated)
            parameters = updated_parameters
            if not all(np.all(np.isfinite(parameter)) for parameter in parameters):
                raise FloatingPointError(f"non-finite parameter at epoch {epoch}")
            step_seconds.append(perf_counter() - step_start)

        loss_history.append(mean_loss())
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            checkpoints[epoch] = tuple(np.array(value, copy=True) for value in parameters)
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"epoch {epoch:3d} loss {loss_history[-1]:.6f}")

    mean_step_seconds = float(np.mean(step_seconds)) if step_seconds else 0.0
    if verbose:
        print(f"data_seed={data_seed} parameter_seed={parameter_seed}")
        print(f"compile_seconds={compile_seconds:.3f}")
        print(f"mean_step_seconds={mean_step_seconds:.6f}")
        print(f"loss: {loss_history[0]:.6f} -> {loss_history[-1]:.6f}")

    return TrainingResult(
        parameters=tuple(parameters),
        loss_history=tuple(loss_history),
        compile_seconds=compile_seconds,
        mean_step_seconds=mean_step_seconds,
        checkpoints=checkpoints,
    )


@dataclass(frozen=True)
class BenchmarkResult:
    gradient_gen_seconds: list[float]
    forward_seconds: float
    gradient_step_seconds: float
    full_step_seconds: float
    peak_memory_kb: float


def run_benchmark() -> BenchmarkResult:
    """Profile the training pipeline and return timing / memory breakdown."""
    gradient_gen_times: list[float] = []
    param_types = _parameter_types()
    for i in range(len(TRAINABLE_NAMES)):
        t0 = perf_counter()
        generate_gradient_function_source(
            _CNN_FULL_LISP_SRC,
            "cnn-loss",
            param_types,
            differentiate_input=i,
            include_prelude=False,
            syntax="lisp",
        )
        gradient_gen_times.append(perf_counter() - t0)

    forward, gradient_functions = _compile_training_functions()

    rng = np.random.RandomState(42)
    image = np.asarray(rng.randn(32, 32).astype(np.float32))
    label = np.float32(1.0)
    mask = make_inference_mask()
    params = initialize_parameters()
    args = (*params, mask, image, label)

    # Warm-up
    forward(*args)
    for gf in gradient_functions:
        gf(*args)

    # Time forward
    t0 = perf_counter()
    for _ in range(10):
        forward(*args)
    forward_seconds = (perf_counter() - t0) / 10

    # Time all gradients
    t0 = perf_counter()
    for _ in range(10):
        for gf in gradient_functions:
            np.asarray(gf(*args), dtype=np.float32)
    gradient_step_seconds = (perf_counter() - t0) / 10

    # Time full step
    t0 = perf_counter()
    for _ in range(10):
        forward(*args)
        for gf in gradient_functions:
            np.asarray(gf(*args), dtype=np.float32)
    full_step_seconds = (perf_counter() - t0) / 10

    column_bytes = 900 * 9 * 4
    w2_grad_bytes = 4 * 900 * 4
    peak_memory_kb = (column_bytes + w2_grad_bytes) / 1024.0

    return BenchmarkResult(
        gradient_gen_seconds=gradient_gen_times,
        forward_seconds=forward_seconds,
        gradient_step_seconds=gradient_step_seconds,
        full_step_seconds=full_step_seconds,
        peak_memory_kb=peak_memory_kb,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--dropout-keep", type=float, default=0.5)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        result = run_benchmark()
        print("=== Benchmark ===")
        print("Gradient source generation (1 function, not cached):")
        for i, t in enumerate(result.gradient_gen_seconds):
            print(f"  param {TRAINABLE_NAMES[i]:>3s}: {t:.4f}s")
        total_gen = sum(result.gradient_gen_seconds)
        print(f"  total: {total_gen:.4f}s")
        print(f"Forward pass (avg 10):    {result.forward_seconds:.6f}s")
        print(f"All 6 gradients (avg 10): {result.gradient_step_seconds:.6f}s")
        print(f"Full step (avg 10):       {result.full_step_seconds:.6f}s")
        print(f"Peak intermediate memory: {result.peak_memory_kb:.1f} KB")
        return

    train_tiny_dataset(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        example_count=args.examples,
        dropout_keep_prob=args.dropout_keep,
    )


if __name__ == "__main__":
    main()
