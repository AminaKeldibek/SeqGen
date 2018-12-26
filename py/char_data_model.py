import numpy as np


class CharDataModel(object):

    def __init__(self):
        data = open('../data/dinos.txt', 'r').read()
        data = data.lower()

        chars = sorted(set(data))
        self.char_to_idx = {char: i for i, char in enumerate(chars)}
        self.idx_to_char = {i: char for i, char in enumerate(chars)}

        data = data.split("\n")
        data = data[:2]  # delete me
        self.vocab_size = len(self.char_to_idx)

        self.X, self.Y = [None] * len(data), [None] * len(data)
        for i in np.arange(len(data)):
            self.X[i] = [None] + [self.char_to_idx[s] for s in data[i]]
            self.Y[i] = self.X[i][1:] + [self.char_to_idx['\n']]
