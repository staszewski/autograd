import numpy as np
from autograd import Tensor

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self._init_weights_and_biases()

    def _init_weights_and_biases(self):
        # To answer: Why we can't have input_size @ hidden_size but hidden_size @ input_size? 
        # Layer 1: input to hidden
        self.W1 = Tensor(np.random.randn(self.hidden_size, self.input_size) * 0.1, requires_grad=True)
        self.b1 = Tensor(np.random.randn(self.hidden_size, 1), requires_grad=True)

        # Layer 2: hidden to output
        self.W2 = Tensor(np.random.randn(self.output_size, self.hidden_size) * 0.1, requires_grad=True)
        self.b2 = Tensor(np.random.randn(self.output_size, 1), requires_grad=True)

    def __call__(self, x):
        # Layer 1: input to hidden
        z1 = self.W1 @ x + self.b1
        a1 = z1.relu()

        # Layer 2: hidden to output
        z2 = self.W2 @ a1 + self.b2
        return z2

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]