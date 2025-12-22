"""
FOR each (english, polish) pair:
    1. Encode English → context vectors
    2. Decode Polish (training mode) → predictions
    3. Compare predictions with target → loss
    4. Backprop through BOTH encoder AND decoder
    5. Update ALL weights
"""

from autograd.models.decoder import Decoder
from autograd.models.encoder import Encoder
from autograd.operations.nll_loss import NLLLoss
from autograd.tensor import Tensor
from autograd.datasets.eng_to_pl_data import EnglishToPolishTranslationData


def compute_sequence_loss(predictions, target_sequence, data_processor):
    total_loss = 0
    for prediction, target_char in zip(predictions, target_sequence):
        target_oneshot = data_processor.char_to_onehot(target_char)
        target_tensor = Tensor(target_oneshot)

        loss = NLLLoss.apply(prediction, target_tensor, axis=1)
        total_loss += loss

    if len(predictions) > len(target_sequence):
        last_prediction = predictions[-1]
        end_oneshot = data_processor.char_to_onehot("<END>")
        end_tensor = Tensor(end_oneshot)
        loss = NLLLoss.apply(last_prediction, end_tensor, axis=1)
        total_loss += loss

    avg_loss = total_loss / len(predictions)
    return avg_loss


def zero_all_gradients(encoder, decoder):
    # Encoder LSTM weights (12 tensors)
    encoder.W_xf.zero_grad()
    encoder.W_hf.zero_grad()
    encoder.b_f.zero_grad()
    encoder.W_xi.zero_grad()
    encoder.W_hi.zero_grad()
    encoder.b_i.zero_grad()
    encoder.W_xg.zero_grad()
    encoder.W_hg.zero_grad()
    encoder.b_g.zero_grad()
    encoder.W_xo.zero_grad()
    encoder.W_ho.zero_grad()
    encoder.b_o.zero_grad()

    # Decoder LSTM weights (12 tensors)
    decoder.W_xf.zero_grad()
    decoder.W_hf.zero_grad()
    decoder.b_f.zero_grad()
    decoder.W_xi.zero_grad()
    decoder.W_hi.zero_grad()
    decoder.b_i.zero_grad()
    decoder.W_xg.zero_grad()
    decoder.W_hg.zero_grad()
    decoder.b_g.zero_grad()
    decoder.W_xo.zero_grad()
    decoder.W_ho.zero_grad()
    decoder.b_o.zero_grad()

    # Decoder output layer (2 tensors)
    decoder.W_output.zero_grad()
    decoder.b_output.zero_grad()


def update_weights(encoder, decoder, lr):
    # Encoder LSTM weights (12 tensors)
    encoder.W_xf._data -= lr * encoder.W_xf._grad
    encoder.W_hf._data -= lr * encoder.W_hf._grad
    encoder.b_f._data -= lr * encoder.b_f._grad
    encoder.W_xi._data -= lr * encoder.W_xi._grad
    encoder.W_hi._data -= lr * encoder.W_hi._grad
    encoder.b_i._data -= lr * encoder.b_i._grad
    encoder.W_xg._data -= lr * encoder.W_xg._grad
    encoder.W_hg._data -= lr * encoder.W_hg._grad
    encoder.b_g._data -= lr * encoder.b_g._grad
    encoder.W_xo._data -= lr * encoder.W_xo._grad
    encoder.W_ho._data -= lr * encoder.W_ho._grad
    encoder.b_o._data -= lr * encoder.b_o._grad

    # Decoder LSTM weights (12 tensors)
    decoder.W_xf._data -= lr * decoder.W_xf._grad
    decoder.W_hf._data -= lr * decoder.W_hf._grad
    decoder.b_f._data -= lr * decoder.b_f._grad
    decoder.W_xi._data -= lr * decoder.W_xi._grad
    decoder.W_hi._data -= lr * decoder.W_hi._grad
    decoder.b_i._data -= lr * decoder.b_i._grad
    decoder.W_xg._data -= lr * decoder.W_xg._grad
    decoder.W_hg._data -= lr * decoder.W_hg._grad
    decoder.b_g._data -= lr * decoder.b_g._grad
    decoder.W_xo._data -= lr * decoder.W_xo._grad
    decoder.W_ho._data -= lr * decoder.W_ho._grad
    decoder.b_o._data -= lr * decoder.b_o._grad

    # Decoder output layer (2 tensors)
    decoder.W_output._data -= lr * decoder.W_output._grad
    decoder.b_output._data -= lr * decoder.b_output._grad


def test_translation(encoder, decoder, data):
    test_sentences = ["i love you", "have fun", "why me?"]

    for english in test_sentences:
        context_h, context_c = encoder.encode(english, data)

        polish = decoder.decode_generate(context_h, context_c, data, max_length=30)

        print(f"{english} -> {polish}")


def train_network(epochs=200, lr=0.01, max_pairs=5000):
    data = EnglishToPolishTranslationData()
    pairs = data.get_pairs()[:max_pairs]

    print(f"Total pairs: {len(pairs)}")

    encoder = Encoder(input_size=data.vocab_size, hidden_size=256, output_size=data.vocab_size)
    decoder = Decoder(input_size=data.vocab_size, hidden_size=256, output_size=data.vocab_size)

    for epoch in range(epochs):
        epoch_loss = 0

        for english, polish in pairs:
            # print(f"'{english}' ({len(english)}) → '{polish}' ({len(polish)})")
            context_h, context_c = encoder.encode(english, data)

            predictions = decoder.decode_train(context_h, context_c, polish, data)

            loss = compute_sequence_loss(predictions, polish, data)
            epoch_loss += loss.data[0][0]

            zero_all_gradients(encoder, decoder)

            loss.backward()

            update_weights(encoder, decoder, lr)

        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(pairs)
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

            test_translation(encoder, decoder, data)


if __name__ == "__main__":
    train_network()
