"""Train binary logistic regression on a deterministic synthetic dataset.

Uses a compiled value-and-grad function for w and b.  Falls back
to the interpreter when native compilation is unavailable.

Per-example loss is computed via the Remora interpreter; gradients
are obtained from a single compiled value-and-grad call (or per-input
interpreted gradient functions as a fallback).  Python accumulates
gradients over a static batch and updates parameters with SGD.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Callable

import numpy as np

from remora.ad_source import generate_gradient_function_source, generate_value_and_grad_function_source
from remora.lisp_reader import parse_lisp
from remora.runtime import CPUFunctionExecutor, _lambda_callable
from remora.typechecker import TypeChecker
from remora.types import ArrayType, FLOAT, FuncType, RemoraType, StaticDim

DATA_SEED = 1729
PARAMETER_SEED = 22022022
DEFAULT_D = 4
DEFAULT_B = 8
TRAINABLE_NAMES = ("w", "b")

_LOGISTIC_LISP_SRC = """
(define/pi ()
  (bce [logit Float y Float] Float)
  (+ (select (> logit 0.0) logit 0.0)
     (+ (* -1.0 (* logit y))
        (log (+ 1.0 (exp (- 0.0 (select (> logit 0.0) logit (- 0.0 logit)))))))))

(define/pi ([D Dim])
  (logistic-loss [w (Array Float D) b Float x (Array Float D) y Float] Float)
  (bce (+ (fold + 0.0 (map * x w)) b) y))
"""


@dataclass(frozen=True)
class TrainingResult:
    parameters: tuple[np.ndarray, object]
    loss_history: tuple[float, ...]
    compile_seconds: float
    mean_step_seconds: float
    compiled: bool


def make_dataset(
    n: int = DEFAULT_B,
    d: int = DEFAULT_D,
    *,
    seed: int = DATA_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]:
    rng = np.random.RandomState(seed)
    x_data = np.ascontiguousarray(rng.randn(n, d).astype(np.float32))
    w_true = np.ascontiguousarray(rng.randn(d).astype(np.float32) * 0.5)
    b_true = np.float32(rng.randn() * 0.1)
    logits = x_data @ w_true + b_true + rng.randn(n).astype(np.float32) * 0.05
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    y_data = np.ascontiguousarray((probs > 0.5).astype(np.float32))
    return x_data, y_data, w_true, b_true


def initialize_parameters(d: int = DEFAULT_D, *, seed: int = PARAMETER_SEED) -> tuple[np.ndarray, np.float32]:
    rng = np.random.RandomState(seed)
    return (
        np.ascontiguousarray(rng.randn(d).astype(np.float32) * 0.1),
        np.float32(0.0),
    )


def _parameter_types(d: int) -> tuple[RemoraType, ...]:
    return (
        ArrayType(FLOAT, (StaticDim(d),)),
        FLOAT,
        ArrayType(FLOAT, (StaticDim(d),)),
        FLOAT,
    )


def _prepare_interpreted_function(
    source: str,
    function_name: str,
    param_types: tuple[RemoraType, ...],
    result_type: RemoraType,
) -> Callable[..., object]:
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


class CompiledLogisticFunctions:
    """Compiled value-and-grad function for native CPU execution."""

    def __init__(self, d: int) -> None:
        param_types = _parameter_types(d)

        gradient_source = generate_value_and_grad_function_source(
            _LOGISTIC_LISP_SRC,
            "logistic-loss",
            param_types,
            differentiate_inputs=(0, 1),
            include_prelude=False,
            syntax="lisp",
        )

        self._grad_executor = CPUFunctionExecutor(
            CPUFunctionExecutor.compile_source(
                gradient_source.source,
                gradient_source.function_name,
                gradient_source.param_types,
                include_prelude=False,
                syntax="lisp",
            )
        )
        self._forward = _prepare_interpreted_function(
            _LOGISTIC_LISP_SRC, "logistic-loss", param_types, FLOAT
        )

    def forward(self, *args: object) -> float:
        return float(self._forward(
            *(np.asarray(a, dtype=np.float32) for a in args)
        ))

    def gradients(self, *args: object) -> tuple[np.ndarray, ...]:
        result = self._grad_executor.execute(
            *(np.asarray(a, dtype=np.float32) for a in args)
        )
        return tuple(np.asarray(g, dtype=np.float32) for g in result.value)


@lru_cache(maxsize=1)
def _compile_interpreted_functions(d: int) -> tuple[
    Callable[..., object], list[Callable[..., object]]
]:
    param_types = _parameter_types(d)
    gradient_sources = []
    for i in range(len(TRAINABLE_NAMES)):
        g = generate_gradient_function_source(
            _LOGISTIC_LISP_SRC,
            "logistic-loss",
            param_types,
            differentiate_input=i,
            include_prelude=False,
            syntax="lisp",
        )
        gradient_sources.append(g)

    forward = _prepare_interpreted_function(
        _LOGISTIC_LISP_SRC, "logistic-loss", param_types, FLOAT
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


def _try_compiled(d: int) -> tuple[CompiledLogisticFunctions | None, str | None]:
    try:
        return CompiledLogisticFunctions(d), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def train_binary_logistic(
    *,
    n: int = DEFAULT_B,
    d: int = DEFAULT_D,
    epochs: int = 200,
    learning_rate: float = 0.1,
    data_seed: int = DATA_SEED,
    parameter_seed: int = PARAMETER_SEED,
    verbose: bool = True,
    use_compiled: bool | None = None,
) -> TrainingResult:
    x_data, y_data, w_true, b_true = make_dataset(n, d, seed=data_seed)
    w, b = initialize_parameters(d, seed=parameter_seed)

    compile_start = perf_counter()

    compiled_failure: str | None = None
    if use_compiled is False:
        compiled = None
    elif use_compiled is True:
        compiled = CompiledLogisticFunctions(d)
    else:
        compiled, compiled_failure = _try_compiled(d)
    if compiled is not None:
        compiled_mode = True
        forward_fn = compiled.forward
        grad_fn = compiled.gradients
        if verbose:
            print("Using compiled native execution (single value-and-grad function)")
    else:
        compiled_mode = False
        interp_forward, interp_grads = _compile_interpreted_functions(d)
        forward_fn = lambda *a: float(interp_forward(*a))  # type: ignore[assignment]
        grad_fn = lambda *a: tuple(  # type: ignore[assignment]
            np.asarray(g(*a), dtype=np.float32) for g in interp_grads
        )
        if verbose:
            print("Using interpreted execution (2 separate gradient functions)")
            if compiled_failure is not None:
                print(f"Compiled execution unavailable: {compiled_failure}")

    compile_seconds = perf_counter() - compile_start

    def mean_loss():
        losses = [
            float(forward_fn(
                np.asarray(w, dtype=np.float32),
                np.asarray(b, dtype=np.float32),
                np.asarray(x_i, dtype=np.float32),
                np.asarray(y_i, dtype=np.float32),
            ))
            for x_i, y_i in zip(x_data, y_data)
        ]
        return float(np.mean(losses))

    loss_history = [mean_loss()]
    step_seconds: list[float] = []

    for epoch in range(1, epochs + 1):
        indices = np.random.RandomState(epoch + parameter_seed).permutation(n)
        for idx in indices:
            step_start = perf_counter()
            x_i = x_data[idx]
            y_i = y_data[idx]
            args = (
                np.asarray(w, dtype=np.float32),
                np.asarray(b, dtype=np.float32),
                np.asarray(x_i, dtype=np.float32),
                np.asarray(y_i, dtype=np.float32),
            )

            gradients = list(grad_fn(*args))
            if not all(np.all(np.isfinite(g)) for g in gradients):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")

            w = np.asarray(w - learning_rate * gradients[0], dtype=np.float32)
            w = np.ascontiguousarray(w)
            b = np.float32(b - learning_rate * gradients[1])

            if not np.all(np.isfinite(w)) or not np.isfinite(b):
                raise FloatingPointError(f"non-finite parameter at epoch {epoch}")

            step_seconds.append(perf_counter() - step_start)

        loss_history.append(mean_loss())
        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            print(f"epoch {epoch:3d} loss {loss_history[-1]:.6f}")

    mean_step_seconds = float(np.mean(step_seconds)) if step_seconds else 0.0
    if verbose:
        print(f"data_seed={data_seed} parameter_seed={parameter_seed} D={d}")
        print(f"compile_seconds={compile_seconds:.3f}")
        print(f"mean_step_seconds={mean_step_seconds:.6f}")
        print(f"loss: {loss_history[0]:.6f} -> {loss_history[-1]:.6f}")

    return TrainingResult(
        parameters=(w, b),
        loss_history=tuple(loss_history),
        compile_seconds=compile_seconds,
        mean_step_seconds=mean_step_seconds,
        compiled=compiled_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--examples", type=int, default=DEFAULT_B)
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--compiled", action="store_true", help="Force compiled native execution")
    args = parser.parse_args()

    train_binary_logistic(
        n=args.examples,
        d=args.d,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_compiled=True if args.compiled else None,
    )


if __name__ == "__main__":
    main()
