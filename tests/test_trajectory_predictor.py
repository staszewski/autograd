import numpy as np
from autograd.tensor import Tensor
from autograd.models.trajectory_predictor import TrajectoryMLP
from autograd.datasets.trajectory_generator import (
    generate_straight_line,
    generate_circular_loiter,
    create_sliding_window_samples,
)

np.random.seed(42)


def test_model_output_shape():
    model = TrajectoryMLP()
    X = Tensor(np.random.randn(10, 5).astype(np.float32), requires_grad=False)

    output = model.forward(X)

    assert output.data.shape == (6, 5), f"Expected (6, 5), got {output.data.shape}"


def test_prediction_is_deterministic():
    """
    Test that same input always produces same output (no randomness in forward pass)
    """
    model = TrajectoryMLP()

    X = Tensor(np.random.randn(10, 3).astype(np.float32), requires_grad=False)

    output1 = model.forward(X)
    output2 = model.forward(X)

    assert np.allclose(output1.data, output2.data), (
        "Forward pass should be deterministic"
    )


def test_loss_decreases_during_training():
    trajectory = generate_straight_line(num_points=50, velocity=(1, 1), noise=0.1)
    X, y = create_sliding_window_samples(trajectory)

    X_train = Tensor(X.T.astype(np.float32), requires_grad=False)
    y_train = Tensor(y.T.astype(np.float32), requires_grad=False)

    model = TrajectoryMLP()
    losses = model.train(X_train, y_train, epochs=50, lr=0.01, verbose=False)

    assert losses[40] < losses[10], "Loss should decrease during training"
    assert losses[-1] < losses[0], "Final loss should be less than initial"


def test_perfect_straight_line_prediction():
    trajectory = generate_straight_line(num_points=50, velocity=(2.0, 1.0), noise=0.0)

    X, y = create_sliding_window_samples(trajectory)

    X_train = Tensor(X.T.astype(np.float32), requires_grad=False)
    y_train = Tensor(y.T.astype(np.float32), requires_grad=False)

    model = TrajectoryMLP()
    losses = model.train(X_train, y_train, epochs=200, lr=0.01, verbose=False)

    final_loss = losses[-1]
    assert final_loss < 10.0, f"Loss too high: {final_loss}"


def test_mlp_beats_baseline_on_curves():
    trajectory = generate_circular_loiter(
        num_points=100, radius=10.0, angular_velocity=0.5, noise=0.02
    )
    X, y = create_sliding_window_samples(trajectory)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = TrajectoryMLP()
    X_train_t = Tensor(X_train.T.astype(np.float32), requires_grad=False)
    y_train_t = Tensor(y_train.T.astype(np.float32), requires_grad=False)
    model.train(X_train_t, y_train_t, epochs=200, lr=0.01, verbose=False)

    X_test_t = Tensor(X_test.T.astype(np.float32), requires_grad=False)
    predictions = model.forward(X_test_t)

    mlp_predictions = predictions.data.T
    mlp_mse = np.mean((mlp_predictions - y_test) ** 2)

    baseline_predictions = []
    for i in range(len(X_test)):
        pos_4 = X_test[i, 8:10]  # Position at t=4
        pos_3 = X_test[i, 6:8]  # Position at t=3

        velocity = pos_4 - pos_3

        # Predict next 3 positions using constant velocity
        pred_5 = pos_4 + velocity
        pred_6 = pred_5 + velocity
        pred_7 = pred_6 + velocity

        baseline_predictions.append(np.concatenate([pred_5, pred_6, pred_7]))

    baseline_predictions = np.array(baseline_predictions)
    baseline_mse = np.mean((baseline_predictions - y_test) ** 2)

    improvement = (baseline_mse - mlp_mse) / baseline_mse * 100

    print("✓ Circular trajectory (curved path):")
    print(f"  Baseline MSE: {baseline_mse:.4f}")
    print(f"  MLP MSE: {mlp_mse:.4f}")
    print(f"  Improvement: {improvement:.1f}%")

    assert mlp_mse < baseline_mse, "MLP should beat baseline on curves"
    assert improvement > 20, f"Expected >20% improvement, got {improvement:.1f}%"

