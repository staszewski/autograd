import numpy as np
from autograd import Tensor


class MultiLayerMLP:
    """
    Multi-layer perceptron with flexible architecture.
    Supports any number of hidden layers.
    """

    def __init__(self, input_size, hidden_sizes, output_size, activation="relu"):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes, e.g. [32, 16] for two layers
            output_size: Number of output features
            activation: Activation function ('relu', 'sigmoid', 'tanh')

        Example:
            mlp = MultiLayerMLP(input_size=10, hidden_sizes=[32, 16], output_size=6)
            # Creates: 10 → 32 → 16 → 6
        """
        self.input_size = input_size
        self.hidden_sizes = (
            hidden_sizes if isinstance(hidden_sizes, list) else [hidden_sizes]
        )
        self.output_size = output_size
        self.activation = activation

        self._init_weights_and_biases()

    def _init_weights_and_biases(self):
        """Initialize weights and biases for all layers using Xavier initialization"""
        self.weights = []
        self.biases = []

        # Build layers: input → hidden1 → hidden2 → ... → output
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]

            scale = (
                np.sqrt(2.0 / in_size)
                if self.activation == "relu"
                else np.sqrt(1.0 / in_size)
            )

            W = Tensor(np.random.randn(out_size, in_size) * scale, requires_grad=True)
            b = Tensor(np.zeros((out_size, 1)), requires_grad=True)

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Tensor of shape (input_size, batch_size) - features as rows

        Returns:
            Tensor of shape (output_size, batch_size)
        """
        a = x

        num_layers = len(self.weights)

        for i in range(num_layers - 1):
            z = self.weights[i] @ a + self.biases[i]

            if self.activation == "relu":
                a = z.relu()
            elif self.activation == "sigmoid":
                a = z.sigmoid()
            elif self.activation == "tanh":
                a = z.tanh()
            else:
                raise ValueError(f"Activation '{self.activation}' not supported")

        z_out = self.weights[-1] @ a + self.biases[-1]

        return z_out

    def __call__(self, x):
        """Allow calling as mlp(x) instead of mlp.forward(x)"""
        return self.forward(x)

    def parameters(self):
        """Return all trainable parameters (weights and biases)"""
        params = []
        for W, b in zip(self.weights, self.biases):
            params.append(W)
            params.append(b)
        return params
