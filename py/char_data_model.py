import numpy as np


class CharDataModel(object):

    def __init__(self, shuffle=True):
        data = open('../data/dinos.txt', 'r').read()
        data = data.lower()

        chars = sorted(set(data))
        self.char_to_idx = {char: i for i, char in enumerate(chars)}
        self.idx_to_char = {i: char for i, char in enumerate(chars)}
        self.termin_elem = '\n'

        data = data.split("\n")
        if shuffle:
            np.random.seed(0)
            np.random.shuffle(data)

        self.vocab_size = len(self.char_to_idx)
        self.train_size = len(data)

        self.X, self.Y = [None] * len(data), [None] * len(data)
        for i in np.arange(len(data)):
            self.X[i] = [None] + [self.char_to_idx[s] for s in data[i]]
            self.Y[i] = self.X[i][1:] + [self.char_to_idx['\n']]
