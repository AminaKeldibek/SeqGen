import numpy as np
from rnn_model import rnn_fwd_one_step


def sample(y_hat):
    return np.argmax(y_hat)


def gen_sequence(params, max_seq_len, termin_elem, a_0, x_0):
    '''
    Generates sequence from trained model of given size.
    Args:
        params:
        max_seq_len:
        termin_elem:
        a_prev:
        x_0:
    Returns:
        sequence: list of numerical representation of sequence
    '''
    sequence = []
    x = x_0
    a_prev = a_0
    elem = None

    while (elem != termin_elem | len(sequence) < max_seq_len):
        a_prev, y_hat = rnn_fwd_one_step(params, a_prev, x)
        elem = sample(y_hat)
        sequence.append(elem)

        x = np.zeros_like(x)
        x[elem, 0] = 1

    return sequence


def decode_seq(sequence, decode_dict):
    return ''.join([decode_dict[s] for s in sequence])
