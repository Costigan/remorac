import numpy as np
import pytest

import examples.crater_train as crater_train
from examples.crater_train import (
    CompiledTrainingFunctions,
    _compile_interpreted_functions,
    initialize_parameters,
    make_inference_mask,
    run_benchmark,
    train_tiny_dataset,
)


def _is_known_compiled_runtime_blocker(exc: Exception) -> bool:
    message = str(exc)
    return "undefined symbol: memrefCopy" in message


def test_tiny_crater_training_decreases_loss():
    """Train for a few steps and verify loss decreases."""
    result = train_tiny_dataset(
        epochs=2,
        learning_rate=0.1,
        example_count=2,
        checkpoint_every=2,
        verbose=False,
        use_compiled=False,
    )

    initial_loss = result.loss_history[0]
    final_loss = result.loss_history[-1]
    assert final_loss < initial_loss, (
        f"loss did not decrease: {initial_loss:.6f} → {final_loss:.6f}"
    )
    assert all(np.all(np.isfinite(parameter)) for parameter in result.parameters)
    assert {0, 2}.issubset(result.checkpoints)
    assert not result.compiled


def test_benchmark_produces_reasonable_numbers():
    result = run_benchmark()

    assert result.gradient_gen_seconds < 5.0
    assert result.forward_seconds < 1.0, f"forward: {result.forward_seconds:.4f}s"
    assert result.gradient_step_seconds < 30.0, f"gradients: {result.gradient_step_seconds:.4f}s"
    assert result.full_step_seconds < 30.0, f"full step: {result.full_step_seconds:.4f}s"
    assert result.peak_memory_kb < 1024.0


def test_strict_compiled_mode_raises_on_compile_failure(monkeypatch):
    def fail_compile():
        raise RuntimeError("compiled path unavailable")

    monkeypatch.setattr(crater_train, "CompiledTrainingFunctions", fail_compile)

    with pytest.raises(RuntimeError, match="compiled path unavailable"):
        train_tiny_dataset(
            epochs=1,
            example_count=2,
            verbose=False,
            use_compiled=True,
        )


def test_compiled_gradients_match_interpreter():
    """Compiled value-and-grad function produces same gradients as interpreter."""
    from remora.pipeline import detect_toolchain

    tc = detect_toolchain()
    if tc.mlir_opt is None or tc.llc is None:
        pytest.skip("mlir-opt and llc required for compiled execution")

    params = initialize_parameters()
    mask = make_inference_mask()
    rng = np.random.RandomState(42)
    image = np.asarray(rng.randn(32, 32).astype(np.float32))
    label = np.float32(1.0)

    # Interpreted gradients (6 separate functions)
    interp_forward, interp_grads = _compile_interpreted_functions()
    interpreted = [
        np.asarray(g(*params, mask, image, label), dtype=np.float32)
        for g in interp_grads
    ]

    # Compiled gradients (single multi-output function)
    try:
        compiled_obj = CompiledTrainingFunctions()
        compiled_grads = compiled_obj.gradients(*params, mask, image, label)
    except Exception as exc:
        if _is_known_compiled_runtime_blocker(exc):
            pytest.skip(f"compiled runtime support unavailable: {exc}")
        raise

    assert len(compiled_grads) == len(interpreted)
    for i, (cg, ig) in enumerate(zip(compiled_grads, interpreted)):
        assert cg.shape == ig.shape, f"gradient {i} shape mismatch: {cg.shape} vs {ig.shape}"
        np.testing.assert_allclose(cg, ig, rtol=1e-3, atol=1e-5,
                                   err_msg=f"gradient {i} mismatch")
