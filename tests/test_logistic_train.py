import numpy as np
import pytest

import examples.logistic_train as logistic_train
from examples.logistic_train import (
    CompiledLogisticFunctions,
    _compile_interpreted_functions,
    initialize_parameters,
    make_dataset,
    train_binary_logistic,
)


def _is_known_compiled_runtime_blocker(exc: Exception) -> bool:
    message = str(exc)
    return "undefined symbol: memrefCopy" in message


def test_synthetic_data_is_deterministic():
    x1, y1, w1, b1 = make_dataset(n=8, d=4, seed=1729)
    x2, y2, w2, b2 = make_dataset(n=8, d=4, seed=1729)

    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(w1, w2)
    assert b1 == b2

    x3, y3, _, _ = make_dataset(n=8, d=4, seed=42)
    assert not np.array_equal(y1, y3)


def test_forward_loss_is_finite():
    d = 4
    w, b = initialize_parameters(d)
    x_data, y_data, _, _ = make_dataset(n=4, d=d)

    interp_forward, _ = _compile_interpreted_functions(d)
    for x_i, y_i in zip(x_data, y_data):
        loss = interp_forward(
            np.asarray(w, dtype=np.float32),
            np.asarray(b, dtype=np.float32),
            np.asarray(x_i, dtype=np.float32),
            np.asarray(y_i, dtype=np.float32),
        )
        assert np.isfinite(float(loss)), f"loss not finite: {loss}"


def test_one_step_lowers_loss():
    d = 4
    w, b = initialize_parameters(d, seed=42)
    x_data, y_data, _, _ = make_dataset(n=8, d=d, seed=99)

    interp_forward, interp_grads = _compile_interpreted_functions(d)

    def loss_fn(w_arr, b_val, x_vec, y_val):
        return float(interp_forward(
            np.asarray(w_arr, dtype=np.float32),
            np.asarray(b_val, dtype=np.float32),
            np.asarray(x_vec, dtype=np.float32),
            np.asarray(y_val, dtype=np.float32),
        ))

    def grads_fn(w_arr, b_val, x_vec, y_val):
        args = (
            np.asarray(w_arr, dtype=np.float32),
            np.asarray(b_val, dtype=np.float32),
            np.asarray(x_vec, dtype=np.float32),
            np.asarray(y_val, dtype=np.float32),
        )
        g_w = np.asarray(interp_grads[0](*args), dtype=np.float32)
        g_b = np.float32(interp_grads[1](*args))
        return g_w, g_b

    x_i = x_data[0]
    y_i = y_data[0]
    initial_loss = loss_fn(w, b, x_i, y_i)

    lr = 0.5
    grad_w, grad_b = grads_fn(w, b, x_i, y_i)
    w_new = np.asarray(w - lr * grad_w, dtype=np.float32)
    b_new = np.float32(b - lr * grad_b)
    new_loss = loss_fn(w_new, b_new, x_i, y_i)

    assert new_loss < initial_loss, (
        f"loss did not decrease after one step: {initial_loss:.6f} -> {new_loss:.6f}"
    )


def test_training_decreases_loss():
    result = train_binary_logistic(
        n=4, d=4, epochs=5, learning_rate=0.5, verbose=False,
        use_compiled=False,
    )
    initial_loss = result.loss_history[0]
    final_loss = result.loss_history[-1]
    assert final_loss < initial_loss, (
        f"loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}"
    )
    assert all(np.all(np.isfinite(p)) for p in result.parameters)
    assert not result.compiled


def test_strict_compiled_mode_raises_on_compile_failure(monkeypatch):
    def fail_compile(d):
        raise RuntimeError("compiled path unavailable")

    monkeypatch.setattr(logistic_train, "CompiledLogisticFunctions", fail_compile)

    with pytest.raises(RuntimeError, match="compiled path unavailable"):
        train_binary_logistic(
            n=4, d=4, epochs=1, verbose=False, use_compiled=True,
        )


def test_compiled_gradients_match_interpreter():
    from remora.pipeline import detect_toolchain

    tc = detect_toolchain()
    if tc.mlir_opt is None or tc.llc is None:
        pytest.skip("mlir-opt and llc required for compiled execution")

    d = 4
    w, b = initialize_parameters(d)
    x_data, y_data, _, _ = make_dataset(n=1, d=d)
    x_i = x_data[0]
    y_i = np.float32(y_data[0])

    interp_forward, interp_grads = _compile_interpreted_functions(d)

    def interp_args():
        return (
            np.asarray(w, dtype=np.float32),
            np.asarray(b, dtype=np.float32),
            np.asarray(x_i, dtype=np.float32),
            np.asarray(y_i, dtype=np.float32),
        )

    interpreted = [
        np.asarray(g(*interp_args()), dtype=np.float32)
        for g in interp_grads
    ]

    try:
        compiled_obj = CompiledLogisticFunctions(d)
        compiled_grads = compiled_obj.gradients(*interp_args())
    except Exception as exc:
        if _is_known_compiled_runtime_blocker(exc):
            pytest.skip(f"compiled runtime support unavailable: {exc}")
        raise

    assert len(compiled_grads) == len(interpreted)
    for i, (cg, ig) in enumerate(zip(compiled_grads, interpreted)):
        assert cg.shape == ig.shape, (
            f"gradient {i} shape mismatch: {cg.shape} vs {ig.shape}"
        )
        np.testing.assert_allclose(cg, ig, rtol=1e-3, atol=1e-5,
                                   err_msg=f"gradient {i} mismatch")


def test_forward_losses_match_compiled_vs_interpreter():
    from remora.pipeline import detect_toolchain

    tc = detect_toolchain()
    if tc.mlir_opt is None or tc.llc is None:
        pytest.skip("mlir-opt and llc required for compiled execution")

    d = 4
    w, b = initialize_parameters(d)
    x_data, y_data, _, _ = make_dataset(n=1, d=d)
    x_i = x_data[0]
    y_i = np.float32(y_data[0])

    interp_forward, _ = _compile_interpreted_functions(d)

    try:
        compiled_obj = CompiledLogisticFunctions(d)
    except Exception as exc:
        if _is_known_compiled_runtime_blocker(exc):
            pytest.skip(f"compiled runtime support unavailable: {exc}")
        raise

    args = (
        np.asarray(w, dtype=np.float32),
        np.asarray(b, dtype=np.float32),
        np.asarray(x_i, dtype=np.float32),
        np.asarray(y_i, dtype=np.float32),
    )

    interp_loss = float(interp_forward(*args))
    compiled_loss = compiled_obj.forward(*args)

    assert np.isfinite(interp_loss)
    assert np.isfinite(compiled_loss)
    assert interp_loss == pytest.approx(compiled_loss, rel=1e-6)
