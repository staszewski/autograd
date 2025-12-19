# https://codingowen.github.io/projects/recurrent_nn_from_scratch/ inspiration
# https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt
from autograd.datasets.shakespeare_data import ShakespeareData
from autograd.operations.log_softmax import LogSoftmaxOperation
from autograd.operations.nll_loss import NLLLoss
from autograd.tensor import Tensor
from autograd.arithmetic import TanhOperation
import numpy as np


class RNN:
    def __init__(self, input_size=2, hidden_size=128, output_size=2):
        self.input = Tensor(np.random.randn(input_size, hidden_size) * 0.1, requires_grad=True)
        self.hidden = Tensor(np.random.randn(hidden_size, hidden_size) * 0.1, requires_grad=True)
        self.bias = Tensor(np.zeros((1, hidden_size)), requires_grad=True)
        self.output = Tensor(np.random.randn(hidden_size, output_size) * 0.1, requires_grad=True)
        self.bias_output = Tensor(np.zeros((1, output_size)), requires_grad=True)

        self.hidden_size = hidden_size

    def step(self, input_data, prev_hidden=None):
        if prev_hidden is None:
            prev_hidden = Tensor(np.zeros((1, self.hidden_size)))

        input_contribution = input_data @ self.input

        hidden_contribution = prev_hidden @ self.hidden

        combined = input_contribution + hidden_contribution + self.bias
        new_hidden = TanhOperation.apply(combined)

        return new_hidden

    def predict_char_probs(self, hidden_state):
        logits = hidden_state @ self.output
        logits = logits + self.bias_output

        probs = LogSoftmaxOperation.apply(logits, axis=1)
        return probs

    def forward_sequence(self, input_string, data_processor):
        predictions = []
        hidden = Tensor(np.zeros((1, self.hidden_size)), requires_grad=True)
        for char in input_string:
            onehot = data_processor.char_to_onehot(char)
            onehot_tensor = Tensor(onehot)
            hidden = self.step(onehot_tensor, hidden)
            pred = self.predict_char_probs(hidden)
            predictions.append(pred)

        return predictions


def compute_sequence_loss(predictions, target_string, data_processor):
    total_loss = 0
    for prediction, target_char in zip(predictions, target_string):
        target_oneshot = data_processor.char_to_onehot(target_char)
        target_tensor = Tensor(target_oneshot)

        loss = NLLLoss.apply(prediction, target_tensor, axis=1)
        total_loss += loss

    avg_loss = total_loss / len(target_string)
    return avg_loss


def generate_text(rnn, data_processor, seed_text="H", length=100, temperature=1.0):
    hidden = Tensor(np.zeros((1, rnn.hidden_size)))
    for char in seed_text:
        onehot = data_processor.char_to_onehot(char)
        onehot_tensor = Tensor(onehot)
        hidden = rnn.step(onehot_tensor, hidden)

    generated = seed_text
    current_char = seed_text[-1]

    for n in range(length):
        onehot = data_processor.char_to_onehot(current_char)
        onehot_tensor = Tensor(onehot)
        hidden = rnn.step(onehot_tensor, hidden)
        log_probs = rnn.predict_char_probs(hidden)

        logits = log_probs.data[0]
        scaled = logits / temperature
        probs = np.exp(scaled - np.max(scaled))
        probs = probs / np.sum(probs)

        next_idx = np.random.choice(len(probs), p=probs)
        next_char = data_processor.idx_to_char[next_idx]

        generated += next_char
        current_char = next_char

    return generated


def train_network(epochs=100, lr=0.01):
    dataset = ShakespeareData()
    nn = RNN(input_size=dataset.vocab_size, output_size=dataset.vocab_size)
    sequence_len = 5
    word_sequences = dataset.create_sequences(sequence_len)
    word_sequences = word_sequences[:10000]

    for epoch in range(epochs):
        epoch_loss = 0
        for seq in word_sequences:
            input_seq, target_seq = seq

            predictions = nn.forward_sequence(input_seq, dataset)

            loss = compute_sequence_loss(predictions, target_seq, dataset)

            epoch_loss += loss.data[0][0]

            nn.input.zero_grad()
            nn.hidden.zero_grad()
            nn.bias.zero_grad()
            nn.output.zero_grad()
            nn.bias_output.zero_grad()

            loss.backward()

            nn.input._data -= lr * nn.input._grad
            nn.hidden._data -= lr * nn.hidden._grad
            nn.bias._data -= lr * nn.bias._grad
            nn.output._data -= lr * nn.output._grad
            nn.bias_output._data -= lr * nn.bias_output._grad

        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(word_sequences)
            print(f"Epoch {epoch}/{epochs}, avg loss: {avg_loss:.4f}")

    print(f"Training complete. Final loss: {avg_loss:.4f}")
    return nn, dataset


def main():
    trained_nn, dataset = train_network()
    print("Generated text:")
    print(generate_text(trained_nn, dataset))


if __name__ == "__main__":
    main()
