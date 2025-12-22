from autograd.models.lstm import LSTM
from autograd.operations.log_softmax import LogSoftmaxOperation
from autograd.tensor import Tensor


class Decoder(LSTM):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)

    def predict_char_probs(self, hidden_state):
        logits = hidden_state @ self.W_output
        logits = logits + self.b_output

        probs = LogSoftmaxOperation.apply(logits, axis=1)
        return probs

    def decode_train(self, context_h, context_c, target_sentence, data):
        hidden = context_h  # from encoder?
        cell = context_c
        predictions = []

        current_char = "<START>"

        for target_char in target_sentence:
            onehot = data.char_to_onehot(current_char)
            onehot_tensor = Tensor(onehot)

            hidden, cell = self.step(onehot_tensor, hidden, cell)
            prediction = self.predict_char_probs(hidden)
            predictions.append(prediction)

            current_char = target_char
        # END TOKEN?
        onehot = data.char_to_onehot(current_char)
        onehot_tensor = Tensor(onehot)

        hidden, cell = self.step(onehot_tensor, hidden, cell)
        prediction = self.predict_char_probs(hidden)
        predictions.append(prediction)

        return predictions

    def decode_generate(self, context_h, context_c, data, max_length=50):
        hidden = context_h  # from encoder?
        cell = context_c
        generated = []

        current_char = "<START>"

        for i in range(max_length):
            onehot = data.char_to_onehot(current_char)
            onehot_tensor = Tensor(onehot)

            hidden, cell = self.step(onehot_tensor, hidden, cell)

            log_probs = self.predict_char_probs(hidden)
            predicted_idx = log_probs.data.argmax()
            next_char = data.idx_to_char[predicted_idx]

            if next_char == "<END>" or next_char == "<PAD>":
                break

            generated.append(next_char)

            current_char = next_char

        return "".join(generated)
