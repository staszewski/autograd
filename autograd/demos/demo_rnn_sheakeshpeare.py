# TODO: RNN based on shakespear (from karpathy?)
# https://codingowen.github.io/projects/recurrent_nn_from_scratch/ inspiration
# https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt
#
# 0. imports
from autograd.datasets.shakespeare_data import ShakespeareData


# 1. Data processing for text
# 2. Define RNN
# 3. forward
# 4. loss computation
# 5. training loop
# 6. demo with text generation


def main():
    dataset = ShakespeareData()

    res = dataset.create_sequences(5)
    print(res)


if __name__ == "__main__":
    main()
