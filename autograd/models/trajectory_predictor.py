from autograd.tensor import Tensor
from autograd.multi_layer_mlp import MultiLayerMLP
from autograd.optimizer import SGD
from autograd.loss import MSELoss
import numpy as np


class TrajectoryMLP:
    """
    Multi-layer perceptron for predicting future drone trajectory positions.

    Architecture: 10 input → 32 → 16 → 6 output
    Input: Last 5 positions (5 × 2 = 10 values)
    Output: Next 3 positions (3 × 2 = 6 values)
    """

    def __init__(self):
        # 10 input → 32 → 16 → 6 output
        self.mlp = MultiLayerMLP(input_size=10, hidden_sizes=[32, 16], output_size=6)

    def forward(self, positions):
        """
        Forward pass through the network.

        Args:
            positions: Tensor shape (batch, 10) - last 5 positions flattened

        Returns:
            Tensor shape (batch, 6) - next 3 positions flattened
        """
        return self.mlp.forward(positions)

    def train(self, X_train, y_train, epochs=100, lr=0.01, verbose=True):
        """
        Train the model using MSE loss and SGD optimizer.

        Args:
            X_train: Tensor (num_samples, 10) - input sequences
            y_train: Tensor (num_samples, 6) - target outputs
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Print loss during training

        Returns:
            losses: List of loss values for each epoch
        """
        optimizer = SGD(self.mlp.parameters(), lr=lr)

        losses = []

        for epoch in range(epochs):
            predictions = self.forward(X_train)
            loss = MSELoss.apply(predictions, y_train)
            loss.backward()
            optimizer.step(grad_clip=1.0)
            optimizer.zero_grad()
            loss_value = float(loss.data)
            losses.append(loss_value)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d}/{epochs}: Loss = {loss_value:.6f}")

        return losses

    def predict(self, last_5_positions):
        """
        Convenience method for single prediction.

        Args:
            last_5_positions: numpy array of shape (5, 2) or (10,)
                             - last 5 observed (x, y) positions

        Returns:
            numpy array of shape (3, 2) - predicted next 3 (x, y) positions
        """
        # Flatten if needed
        if last_5_positions.shape == (5, 2):
            input_vector = last_5_positions.flatten()
        else:
            input_vector = last_5_positions

        # Ensure correct shape
        assert input_vector.shape == (10,), (
            f"Expected shape (10,), got {input_vector.shape}"
        )

        # Create tensor (batch of 1)
        X = Tensor(input_vector.reshape(1, 10).astype(np.float32), requires_grad=False)

        # Forward pass
        output = self.forward(X)

        # Extract predictions and reshape to (3, 2)
        predictions = output.data.reshape(3, 2)

        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set.

        Args:
            X_test: Tensor (num_samples, 10)
            y_test: Tensor (num_samples, 6)

        Returns:
            mse: Mean squared error on test set
        """
        predictions = self.forward(X_test)
        mse = np.mean((predictions.data - y_test.data) ** 2)
        return mse
