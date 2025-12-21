from autograd.datasets.shakespeare_data import ShakespeareData
from autograd.tensor import Tensor
import numpy as np
from autograd.arithmetic import TanhOperation, SigmoidOperation
from autograd.operations.log_softmax import LogSoftmaxOperation
from autograd.operations.nll_loss import NLLLoss


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

    def forward_sequence(self, input_string, data_processor):
        predictions = []
        hidden = Tensor(np.zeros((1, self.hidden_size)), requires_grad=True)
        cell = Tensor(np.zeros((1, self.hidden_size)), requires_grad=True)
        for char in input_string:
            onehot = data_processor.char_to_onehot(char)
            onehot_tensor = Tensor(onehot)
            hidden, cell = self.step(onehot_tensor, hidden, cell)
            pred = self.predict_char_probs(hidden)
            predictions.append(pred)

        return predictions

    def predict_char_probs(self, hidden_state):
        logits = hidden_state @ self.W_output
        logits = logits + self.b_output

        probs = LogSoftmaxOperation.apply(logits, axis=1)
        return probs


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
    cell = Tensor(np.zeros((1, rnn.hidden_size)))
    for char in seed_text:
        onehot = data_processor.char_to_onehot(char)
        onehot_tensor = Tensor(onehot)
        hidden, cell = rnn.step(onehot_tensor, hidden, cell)

    generated = seed_text
    current_char = seed_text[-1]

    for n in range(length):
        onehot = data_processor.char_to_onehot(current_char)
        onehot_tensor = Tensor(onehot)
        hidden, cell = rnn.step(onehot_tensor, hidden, cell)
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
    nn = LSTM(input_size=dataset.vocab_size, hidden_size=128, output_size=dataset.vocab_size)
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

            # Forget gate
            nn.W_xf.zero_grad()
            nn.W_hf.zero_grad()
            nn.b_f.zero_grad()

            # Input gate
            nn.W_xi.zero_grad()
            nn.W_hi.zero_grad()
            nn.b_i.zero_grad()

            # Candidate
            nn.W_xg.zero_grad()
            nn.W_hg.zero_grad()
            nn.b_g.zero_grad()

            # Output gate
            nn.W_xo.zero_grad()
            nn.W_ho.zero_grad()
            nn.b_o.zero_grad()

            # Output layer
            nn.W_output.zero_grad()
            nn.b_output.zero_grad()

            loss.backward()

            # Forget gate
            nn.W_xf._data -= lr * nn.W_xf._grad
            nn.W_hf._data -= lr * nn.W_hf._grad
            nn.b_f._data -= lr * nn.b_f._grad

            # Input gate
            nn.W_xi._data -= lr * nn.W_xi._grad
            nn.W_hi._data -= lr * nn.W_hi._grad
            nn.b_i._data -= lr * nn.b_i._grad

            # Candidate
            nn.W_xg._data -= lr * nn.W_xg._grad
            nn.W_hg._data -= lr * nn.W_hg._grad
            nn.b_g._data -= lr * nn.b_g._grad

            # Output gate
            nn.W_xo._data -= lr * nn.W_xo._grad
            nn.W_ho._data -= lr * nn.W_ho._grad
            nn.b_o._data -= lr * nn.b_o._grad

            # Output layer
            nn.W_output._data -= lr * nn.W_output._grad
            nn.b_output._data -= lr * nn.b_output._grad

        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(word_sequences)
            print(f"Epoch {epoch}/{epochs}, avg loss: {avg_loss:.4f}")

    print(f"Training complete. Final loss: {avg_loss:.4f}")
    return nn, dataset


if __name__ == "__main__":
    trained_nn, dataset = train_network()

    print("Generated text:")
    print(generate_text(trained_nn, dataset))
