import numpy as np
from scipy.special import softmax
from initialization import xavier_weight_init


def init_params(n_a, n_x, n_y):
    '''
    Initializes RNN model parameters.

    Args:
        n_a: size of hidden layer
        n_x: size of input layer
        n_y: size of output layer

    Returns:
        a_0: initial vector for prev hidden state, numpy vector of shape (n_a, 1)
        params: python dictionary of:
                Wax: numpy weight matrix of shape (n_a, n_x)
                Waa: numpy weight matrix of shape (n_a, n_a)
                Wya: numpy weight matrix of shape (n_y, n_a)
                ba: numpy weight matrix of shape (n_a, a)
                by: numpy weight matrix of shape (n_y, 1)
    '''
    a_0 = np.zeros((n_a, 1))
    param_names = ['Wax', 'Waa', 'Wya', 'ba', 'by']
    param_shapes = [(n_a, n_x), (n_a, n_a), (n_y, n_a), (n_a, 1), (n_y, 1)]

    xavier_init = xavier_weight_init()

    param_mats = list(map(xavier_init, param_shapes[:3]))
    param_mats += list(map(lambda s: np.zeros(s), param_shapes[3:]))

    params = dict(zip(param_names, param_mats))

    return a_0, params


def rnn_fwd_one_step(params, a_prev, x):
    '''
    Performs one step of RNN forward propagation

    Args:
        x: one-hot vector of shape (vocab_size, 1)
    '''
    Wax, Waa, Wya = params['Wax'], params['Waa'], params['Wya']
    ba, by = params['ba'], params['by']

    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + ba)
    y_hat = softmax(np.dot(Wya, a_next) + by)

    return a_next, y_hat


def rnn_fwd(X, a_0, params, vocab_size):
    '''
    Performs RNN forward propagation over sequence.

    Args:
        X: python list of integers representing one input sequence
        a_0: see init_params
        params: see init_params
        vocab_size: number of unique sequence values
    '''
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a_0)

    for t in np.arange(len(X)):
        x[t] = np.zeros((vocab_size, 1))
        if X[t] is not None:
            x[t][X[t]] = 1

        a[t], y_hat[t] = rnn_fwd_one_step(params, a[t-1], x[t])

    cache = {'x': x, 'a': a, 'y_hat': y_hat}

    return cache


def calc_cost(y_hat, Y):
    '''
    Calculates cross entropy loss.

    Args:
        y_hat: dictionary of y_hat per timestamp of one sequence
        Y: python list of integers representing one sequencer
    '''
    loss = 0
    for t in np.arange(len(y_hat)):
        loss -= np.log(y_hat[t][Y[t], 0])

    return loss


def rnn_bwd(X, Y, params, cache):
    grads = {'dWya': 0, 'dby': 0, 'dWax': 0, 'dWaa': 0, 'dba': 0}
    return grads
