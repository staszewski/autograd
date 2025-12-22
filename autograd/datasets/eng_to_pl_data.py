import csv
import numpy as np


class EnglishToPolishTranslationData:
    def __init__(self) -> None:
        self.pairs = []
        with open("data/eng_to_pl/eng_to_pl.tsv", encoding="utf-8") as tsv_file:
            reader = csv.reader(tsv_file, delimiter="\t")
            for row in reader:
                eng = row[1]
                pol = row[3]
                self.pairs.append((eng.lower(), pol.lower()))

        self.build_vocabulary()

    def build_vocabulary(self):
        all_chars = set().union(*(eng + pol for eng, pol in self.pairs))
        self.chars = sorted(all_chars)
        self.chars = ["<PAD>", "<START>", "<END>"] + self.chars

        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.vocab_size = len(self.chars)

    def char_to_onehot(self, char):
        char_idx = self.char_to_idx[char]
        char_zeros = np.zeros(self.vocab_size)

        char_zeros[char_idx] = 1

        return char_zeros.reshape(1, self.vocab_size)

    def get_pairs(self):
        return self.pairs

