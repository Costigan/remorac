"""Tests for the 4-channel synthetic crater detector training (Phase 2)."""

import numpy as np
import pytest

import examples.crater_detect_train as detect_train
from examples.crater_detect_train import (
    DetectorCompiledFunctions,
    initialize_parameters,
    make_dataset,
    train,
)


def _is_known_compiled_runtime_blocker(exc: Exception) -> bool:
    message = str(exc)
    return "undefined symbol: memrefCopy" in message


def test_training_decreases_loss():
    result = train(
        n=4, batch_size=2, epochs=3, learning_rate=0.01,
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


def test_one_step_lowers_loss():
    *kerns, b1, w2, b2 = initialize_parameters(seed=99)
    img, tgt_list, infos = make_dataset(n=1, seed=42)

    compiled = DetectorCompiledFunctions.compile()
    kernels = tuple(kerns)
    ch = tgt_list[0]
    initial = compiled.forward(kernels, float(b1), float(w2), float(b2), img[0], ch)
    assert np.isfinite(initial)

    g_ks, g_b1, g_w2, g_b2 = compiled.gradients(kernels, float(b1), float(w2), float(b2), img[0], ch)
    lr = 0.01
    new_kerns = tuple(np.asarray(k - lr * g_ks[i], dtype=np.float32) for i, k in enumerate(kerns))
    new_b1 = np.float32(b1 - lr * float(g_b1))
    new_w2 = np.float32(w2 - lr * float(g_w2))
    new_b2 = np.float32(b2 - lr * float(g_b2))

    new_loss = compiled.forward(new_kerns, float(new_b1), float(new_w2), float(new_b2), img[0], ch)
    assert new_loss < initial, (
        f"one step did not lower loss: {initial:.6f} -> {new_loss:.6f}"
    )


def test_gradients_are_finite():
    *kerns, b1, w2, b2 = initialize_parameters(seed=1)
    img, tgt_list, infos = make_dataset(n=1, seed=1)

    compiled = DetectorCompiledFunctions.compile()
    g_ks, g_b1, g_w2, g_b2 = compiled.gradients(
        tuple(kerns), float(b1), float(w2), float(b2), img[0], tgt_list[0],
    )

    for i, g in enumerate(g_ks):
        assert np.all(np.isfinite(g)), f"g_k[{i}] has non-finite values"
    assert np.isfinite(g_b1), "g_b1 is non-finite"
    assert np.isfinite(g_w2), "g_w2 is non-finite"
    assert np.isfinite(g_b2), "g_b2 is non-finite"


def test_loss_finite_for_random_parameters():
    *kerns, b1, w2, b2 = initialize_parameters(seed=42)
    img, tgt_list, infos = make_dataset(n=4, seed=1729)

    compiled = DetectorCompiledFunctions.compile()
    kernels = tuple(kerns)
    for i in range(4):
        loss = compiled.forward(kernels, float(b1), float(w2), float(b2), img[i], tgt_list[i])
        assert np.isfinite(loss), f"loss not finite for example {i}: {loss}"
        assert loss >= -1e-6, f"loss should be non-negative, got {loss}"


def test_batch_gradient_accumulation():
    *kerns, b1, w2, b2 = initialize_parameters(seed=42)
    img, tgt_list, infos = make_dataset(n=4, seed=1729)

    compiled = DetectorCompiledFunctions.compile()
    kernels = tuple(kerns)
    g_k_sum = [np.zeros_like(k) for k in kernels]
    g_b1_sum = 0.0
    g_w2_sum = 0.0
    g_b2_sum = 0.0
    for i in range(4):
        g_ks, g_b1, g_w2, g_b2 = compiled.gradients(
            kernels, float(b1), float(w2), float(b2), img[i], tgt_list[i],
        )
        for c in range(4):
            g_k_sum[c] += g_ks[c]
        g_b1_sum += float(g_b1)
        g_w2_sum += float(g_w2)
        g_b2_sum += float(g_b2)

    for i in range(4):
        assert np.all(np.isfinite(g_k_sum[i]))
        assert not np.allclose(g_k_sum[i], 0.0, atol=1e-12), f"summed g_k[{i}] is zero"
    assert np.isfinite(g_b1_sum) and abs(g_b1_sum) > 1e-12
    assert np.isfinite(g_w2_sum) and abs(g_w2_sum) > 1e-12
    assert np.isfinite(g_b2_sum) and abs(g_b2_sum) > 1e-12
