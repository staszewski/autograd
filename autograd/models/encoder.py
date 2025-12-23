from autograd.tensor import Tensor
import numpy as np
from autograd.models.lstm import LSTM


class Encoder(LSTM):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)

    def encode(self, sentence: str, data_processor):
        hidden = Tensor(np.zeros((1, self.hidden_size)))
        cell = Tensor(np.zeros((1, self.hidden_size)))

        for char in sentence:
            onehot = data_processor.char_to_onehot(char)
            onehot_tensor = Tensor(onehot)

            hidden, cell = self.step(onehot_tensor, hidden, cell)

        return hidden, cell
