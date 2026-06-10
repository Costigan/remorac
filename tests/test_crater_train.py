import numpy as np

from examples.crater_train import train_tiny_dataset


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
