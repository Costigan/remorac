"""Tests for the synthetic crater detector training script (Phase 2)."""

import numpy as np
import pytest

import examples.crater_detect_train as detect_train
from examples.crater_detect_train import (
    DetectorCompiledFunctions,
    DetectorTrainingResult,
    initialize_parameters,
    make_dataset,
    train,
)


def _is_known_compiled_runtime_blocker(exc: Exception) -> bool:
    message = str(exc)
    return "undefined symbol: memrefCopy" in message


def test_training_decreases_loss():
    result = train(
        n=4, epochs=3, learning_rate=0.1,
        data_seed=1729, parameter_seed=42,
        verbose=False,
    )
    initial = result.loss_history[0]
    final = result.loss_history[-1]
    assert final < initial, (
        f"loss did not decrease: {initial:.6f} -> {final:.6f}"
    )
    assert np.isfinite(initial)
    assert np.isfinite(final)
    assert all(np.all(np.isfinite(p)) for p in result.parameters)


def test_one_step_lowers_loss():
    k, b1, w2, b2 = initialize_parameters(seed=99)
    img, tgt, infos = make_dataset(n=1, seed=42)

    compiled = DetectorCompiledFunctions.compile()
    initial = compiled.forward(k, float(b1), float(w2), float(b2), img[0], tgt[0])
    assert np.isfinite(initial)

    g = compiled.gradients(k, float(b1), float(w2), float(b2), img[0], tgt[0])
    lr = 0.1
    k_new = np.asarray(k - lr * g[0], dtype=np.float32)
    b1_new = np.float32(b1 - lr * float(g[1]))
    w2_new = np.float32(w2 - lr * float(g[2]))
    b2_new = np.float32(b2 - lr * float(g[3]))

    new_loss = compiled.forward(k_new, float(b1_new), float(w2_new), float(b2_new), img[0], tgt[0])
    assert new_loss < initial, (
        f"one step did not lower loss: {initial:.6f} -> {new_loss:.6f}"
    )


def test_gradients_are_finite():
    k, b1, w2, b2 = initialize_parameters(seed=1)
    img, tgt, infos = make_dataset(n=1, seed=1)

    compiled = DetectorCompiledFunctions.compile()
    g = compiled.gradients(k, float(b1), float(w2), float(b2), img[0], tgt[0])

    assert np.all(np.isfinite(g[0])), "g_k contains non-finite values"
    assert np.isfinite(g[1]), "g_b1 is non-finite"
    assert np.isfinite(g[2]), "g_w2 is non-finite"
    assert np.isfinite(g[3]), "g_b2 is non-finite"


def test_loss_finite_for_random_parameters():
    """Detector loss must be finite for random parameters on synthetic data."""
    k, b1, w2, b2 = initialize_parameters(seed=42)
    img, tgt, infos = make_dataset(n=4, seed=1729)

    compiled = DetectorCompiledFunctions.compile()
    for i in range(4):
        loss = compiled.forward(k, float(b1), float(w2), float(b2), img[i], tgt[i])
        assert np.isfinite(loss), f"loss not finite for example {i}: {loss}"
