import numpy as np


class ShakespeareData:
    def __init__(self):
        with open("autograd/datasets/shakespear.txt", "r") as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def char_to_onehot(self, char):
        char_idx = self.char_to_idx[char]
        char_zeros = np.zeros(self.vocab_size)

        char_zeros[char_idx] = 1

        return char_zeros.reshape(1, self.vocab_size)

    def create_sequences(self, seq_length):
        text_length = len(self.text)
        if seq_length <= 0 or seq_length > text_length:
            return []

        result = []
        for i in range(text_length - seq_length):
            input = self.text[i : i + seq_length]
            target = self.text[i + 1 : i + seq_length + 1]
            result.append((input, target))

        return result
