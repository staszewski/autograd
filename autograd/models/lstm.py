from autograd.tensor import Tensor
import numpy as np
from autograd.arithmetic import TanhOperation, SigmoidOperation


class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        # Forget gate
        self.W_xf = Tensor(np.random.randn(input_size, hidden_size) * 0.1, requires_grad=True)
        self.W_hf = Tensor(np.random.randn(hidden_size, hidden_size) * 0.1, requires_grad=True)
        self.b_f = Tensor(np.ones((1, hidden_size)), requires_grad=True)

        # Input gate
        self.W_xi = Tensor(np.random.randn(input_size, hidden_size) * 0.1, requires_grad=True)
        self.W_hi = Tensor(np.random.randn(hidden_size, hidden_size) * 0.1, requires_grad=True)
        self.b_i = Tensor(np.zeros((1, hidden_size)), requires_grad=True)

        # Candidate weights
        self.W_xg = Tensor(np.random.randn(input_size, hidden_size) * 0.1, requires_grad=True)
        self.W_hg = Tensor(np.random.randn(hidden_size, hidden_size) * 0.1, requires_grad=True)
        self.b_g = Tensor(np.zeros((1, hidden_size)), requires_grad=True)

        # Output gate
        self.W_xo = Tensor(np.random.randn(input_size, hidden_size) * 0.1, requires_grad=True)
        self.W_ho = Tensor(np.random.randn(hidden_size, hidden_size) * 0.1, requires_grad=True)
        self.b_o = Tensor(np.zeros((1, hidden_size)), requires_grad=True)

        # Output layer
        self.W_output = Tensor(np.random.randn(hidden_size, output_size) * 0.1, requires_grad=True)
        self.b_output = Tensor(np.zeros((1, output_size)), requires_grad=True)

    def step(self, input_data, prev_hidden, prev_cell):
        if prev_hidden is None:
            prev_hidden = Tensor(np.zeros((1, self.hidden_size)))
        if prev_cell is None:
            prev_cell = Tensor(np.zeros((1, self.hidden_size)))

        forget_gate = SigmoidOperation.apply(input_data @ self.W_xf + prev_hidden @ self.W_hf + self.b_f)
        input_gate = SigmoidOperation.apply(input_data @ self.W_xi + prev_hidden @ self.W_hi + self.b_i)
        candidate_cell = TanhOperation.apply(input_data @ self.W_xg + prev_hidden @ self.W_hg + self.b_g)
        output_gate = SigmoidOperation.apply(input_data @ self.W_xo + prev_hidden @ self.W_ho + self.b_o)

        new_cell = forget_gate * prev_cell + input_gate * candidate_cell
        new_hidden = output_gate * TanhOperation.apply(new_cell)

        return new_hidden, new_cell
