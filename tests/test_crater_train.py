import numpy as np

from examples.crater_train import run_benchmark, train_tiny_dataset


def test_tiny_crater_training_decreases_loss():
    """Train for a few steps and verify loss decreases.

    Uses small epochs/examples because interpreted gradient functions
    are typechecked at ~40 s each (6 gradients + 1 forward ≈ 260 s).
    """
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

    # Single multi-output gradient source generation call
    assert result.gradient_gen_seconds < 5.0

    # Forward and gradient step should be sub-second (interpreted mode)
    assert result.forward_seconds < 1.0, f"forward: {result.forward_seconds:.4f}s"
    assert result.gradient_step_seconds < 30.0, f"gradients: {result.gradient_step_seconds:.4f}s"
    assert result.full_step_seconds < 30.0, f"full step: {result.full_step_seconds:.4f}s"

    # Peak memory should be under 1 MB (for 32x32 model)
    assert result.peak_memory_kb < 1024.0
