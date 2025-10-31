import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from autograd.tensor import Tensor
from autograd.models.trajectory_predictor import TrajectoryMLP
from autograd.datasets.trajectory_generator import (
    generate_circular_loiter,
    create_sliding_window_samples,
)

np.random.seed(42)


def train_model_on_circular():
    """Train model on circular trajectory and return model + test data"""
    print("Generating circular trajectory...")
    trajectory = generate_circular_loiter(
        num_points=100, radius=10.0, angular_velocity=0.5, noise=0.02
    )

    X, y = create_sliding_window_samples(trajectory)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training on {len(X_train)} samples...")
    model = TrajectoryMLP()
    X_train_t = Tensor(X_train.T.astype(np.float32), requires_grad=False)
    y_train_t = Tensor(y_train.T.astype(np.float32), requires_grad=False)
    losses = model.train(X_train_t, y_train_t, epochs=300, lr=0.01, verbose=True)
    print(f"\nTraining complete! Loss: {losses[0]:.2f} → {losses[-1]:.2f}")

    return model, trajectory, X_test, y_test, losses


def compute_baseline_predictions(X_test):
    """Compute constant-velocity baseline predictions"""
    baseline_predictions = []
    for i in range(len(X_test)):
        pos_4 = X_test[i, 8:10]
        pos_3 = X_test[i, 6:8]
        velocity = pos_4 - pos_3

        pred_5 = pos_4 + velocity
        pred_6 = pred_5 + velocity
        pred_7 = pred_6 + velocity

        baseline_predictions.append(np.concatenate([pred_5, pred_6, pred_7]))

    return np.array(baseline_predictions)


def create_comparison_plot(model, trajectory, X_test, y_test):
    """Create side-by-side comparison of baseline vs ML"""
    print("\nCreating comparison visualization...")

    X_test_t = Tensor(X_test.T.astype(np.float32), requires_grad=False)
    ml_predictions = model.forward(X_test_t).data.T

    baseline_predictions = compute_baseline_predictions(X_test)

    ml_mse = np.mean((ml_predictions - y_test) ** 2)
    baseline_mse = np.mean((baseline_predictions - y_test) ** 2)
    improvement = (baseline_mse - ml_mse) / baseline_mse * 100

    plt.figure(figsize=(16, 5))

    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        "b-",
        alpha=0.3,
        linewidth=2,
        label="Full trajectory",
    )
    ax1.scatter(
        trajectory[::5, 0], trajectory[::5, 1], c="blue", s=30, alpha=0.5, zorder=3
    )
    ax1.set_xlabel("X Position", fontsize=11)
    ax1.set_ylabel("Y Position", fontsize=11)
    ax1.set_title(
        "Circular Trajectory\n(Training Data)", fontsize=12, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")
    ax1.legend()

    # Plot 2: Baseline predictions (constant velocity)
    ax2 = plt.subplot(1, 3, 2)
    # Show one example prediction
    example_idx = len(X_test) // 2
    input_positions = X_test[example_idx].reshape(5, 2)
    true_future = y_test[example_idx].reshape(3, 2)
    baseline_future = baseline_predictions[example_idx].reshape(3, 2)

    ax2.plot(
        input_positions[:, 0],
        input_positions[:, 1],
        "bo-",
        markersize=8,
        linewidth=2,
        label="Observed (last 5)",
    )
    ax2.plot(
        true_future[:, 0],
        true_future[:, 1],
        "g*-",
        markersize=15,
        linewidth=2,
        label="Ground truth",
    )
    ax2.plot(
        baseline_future[:, 0],
        baseline_future[:, 1],
        "rx-",
        markersize=12,
        linewidth=2,
        label="Baseline pred",
    )
    ax2.set_xlabel("X Position", fontsize=11)
    ax2.set_ylabel("Y Position", fontsize=11)
    ax2.set_title(
        f"Baseline (Constant Velocity)\nMSE: {baseline_mse:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis("equal")

    # Plot 3: ML predictions
    ax3 = plt.subplot(1, 3, 3)
    ml_future = ml_predictions[example_idx].reshape(3, 2)

    ax3.plot(
        input_positions[:, 0],
        input_positions[:, 1],
        "bo-",
        markersize=8,
        linewidth=2,
        label="Observed (last 5)",
    )
    ax3.plot(
        true_future[:, 0],
        true_future[:, 1],
        "g*-",
        markersize=15,
        linewidth=2,
        label="Ground truth",
    )
    ax3.plot(
        ml_future[:, 0],
        ml_future[:, 1],
        "mo-",
        markersize=12,
        linewidth=2,
        label="ML pred",
    )
    ax3.set_xlabel("X Position", fontsize=11)
    ax3.set_ylabel("Y Position", fontsize=11)
    ax3.set_title(
        f"ML Predictor\nMSE: {ml_mse:.2f} ({improvement:.1f}% better!)",
        fontsize=12,
        fontweight="bold",
        color="darkgreen",
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axis("equal")

    plt.tight_layout()
    plt.savefig("trajectory_prediction_comparison.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: trajectory_prediction_comparison.png")

    print(f"\n{'=' * 50}")
    print("RESULTS:")
    print(f"{'=' * 50}")
    print(f"Baseline MSE:     {baseline_mse:.4f}")
    print(f"ML MSE:           {ml_mse:.4f}")
    print(f"Improvement:      {improvement:.1f}%")
    print(f"{'=' * 50}\n")

    return ml_mse, baseline_mse, improvement


def create_metrics_plot(losses, ml_mse, baseline_mse):
    """Create metrics visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Training loss curve
    ax1.plot(losses, linewidth=2, color="darkblue")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("MSE Loss", fontsize=11)
    ax1.set_title("Training Loss Over Time", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(losses))

    # Comparison bar chart
    methods = ["Constant\nVelocity", "ML\nPredictor"]
    mse_values = [baseline_mse, ml_mse]
    colors = ["#ff6b6b", "#51cf66"]

    bars = ax2.bar(
        methods, mse_values, color=colors, alpha=0.7, edgecolor="black", linewidth=2
    )
    ax2.set_ylabel("Mean Squared Error", fontsize=11)
    ax2.set_title("Prediction Accuracy Comparison", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, mse_values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("trajectory_prediction_metrics.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: trajectory_prediction_metrics.png")


def create_rolling_prediction_animation(model, trajectory):
    """Create animation showing rolling predictions"""
    fig, ax = plt.subplots(figsize=(10, 10))

    X_full, y_full = create_sliding_window_samples(trajectory)

    def animate(i):
        ax.clear()

        input_pos = X_full[i].reshape(5, 2)
        true_future = y_full[i].reshape(3, 2)

        X_t = Tensor(X_full[i : i + 1].T.astype(np.float32), requires_grad=False)
        ml_pred = model.forward(X_t).data.T.reshape(3, 2)

        pos_4 = X_full[i, 8:10]
        pos_3 = X_full[i, 6:8]
        velocity = pos_4 - pos_3
        baseline_pred = np.array(
            [pos_4 + velocity, pos_4 + 2 * velocity, pos_4 + 3 * velocity]
        )

        ax.plot(trajectory[:, 0], trajectory[:, 1], "k-", alpha=0.1, linewidth=1)

        ax.plot(
            input_pos[:, 0],
            input_pos[:, 1],
            "bo-",
            markersize=10,
            linewidth=3,
            label="Observed",
            zorder=3,
        )
        ax.plot(
            true_future[:, 0],
            true_future[:, 1],
            "g*-",
            markersize=18,
            linewidth=3,
            label="True future",
            zorder=4,
        )
        ax.plot(
            baseline_pred[:, 0],
            baseline_pred[:, 1],
            "rx--",
            markersize=14,
            linewidth=2,
            label="Baseline",
            alpha=0.7,
            zorder=2,
        )
        ax.plot(
            ml_pred[:, 0],
            ml_pred[:, 1],
            "mo-",
            markersize=14,
            linewidth=3,
            label="ML",
            zorder=3,
        )

        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Y Position", fontsize=12)
        ax.set_title(
            f"Rolling Prediction (Frame {i + 1}/{len(X_full)})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.set_xlim(trajectory[:, 0].min() - 2, trajectory[:, 0].max() + 2)
        ax.set_ylim(trajectory[:, 1].min() - 2, trajectory[:, 1].max() + 2)

    anim = animation.FuncAnimation(
        fig, animate, frames=min(50, len(X_full)), interval=200, repeat=True
    )
    anim.save("trajectory_prediction_animation.gif", writer="pillow", fps=5, dpi=100)
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  TRAJECTORY PREDICTION DEMO")
    print("  Comparing ML vs Constant-Velocity Baseline")
    print("=" * 60)

    model, trajectory, X_test, y_test, losses = train_model_on_circular()

    ml_mse, baseline_mse, improvement = create_comparison_plot(
        model, trajectory, X_test, y_test
    )
    create_metrics_plot(losses, ml_mse, baseline_mse)
    create_rolling_prediction_animation(model, trajectory)

    print("  DEMO COMPLETE!")
