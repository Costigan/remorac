import numpy as np

from examples.crater_train import run_benchmark, train_tiny_dataset


def test_tiny_crater_training_decreases_loss():
    result = train_tiny_dataset(
        epochs=5,
        learning_rate=0.1,
        example_count=8,
        checkpoint_every=5,
        verbose=False,
    )

    initial_loss = result.loss_history[0]
    final_loss = result.loss_history[-1]
    assert final_loss < initial_loss * 0.95
    assert all(np.all(np.isfinite(parameter)) for parameter in result.parameters)
    assert {0, 5}.issubset(result.checkpoints)


def test_benchmark_produces_reasonable_numbers():
    result = run_benchmark()

    # Gradient generation should be sub-second per parameter
    assert all(t < 5.0 for t in result.gradient_gen_seconds)
    assert sum(result.gradient_gen_seconds) < 10.0

    # Forward and gradient step should be sub-second
    assert result.forward_seconds < 1.0
    assert result.gradient_step_seconds < 5.0
    assert result.full_step_seconds < 5.0

    # Peak memory should be under 1 MB (for 32x32 model)
    assert result.peak_memory_kb < 1024.0
