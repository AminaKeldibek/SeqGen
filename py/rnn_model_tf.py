import numpy as np
import tensorflow as tf

from initialization import xavier_weight_init


class SeqGenModel():
    def __init__(self, n_a, n_x, n_y):
        '''
        Initializes RNN model params.

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
        xavier_init = xavier_weight_init()

        self.a_0 = tf.get_variable("a_0", (n_a, 1), tf.float32,
                                   tf.zeros_initializer())

        self.Wax = tf.get_variable("Wax", shape=(n_a, n_x), initializer=xavier_init)
        self.Waa = tf.get_variable("Waa", shape=(n_a, n_a), initializer=xavier_init)
        self.Wya = tf.get_variable("Wya", shape=(n_y, n_x), initializer=xavier_init)
        self.ba = tf.get_variable("ba", shape=(n_a, 1), initializer=tf.zeros_initializer())
        self.by = tf.get_variable("by", shape=(n_y, 1), initializer=tf.zeros_initializer())

    def rnn_fwd_one_step(self, a_prev, x):
        '''
        Performs one step of RNN forward propagation

        Args:
            x: one-hot vector of shape (vocab_size, 1)
        Returns:
            a_next: tensor of shape (n_a, 1)
            y_hat: tensor of shape (vocab_size, 1)
        '''
        a_next = tf.tanh(tf.matmul(self.Wax, x) + tf.matmul(self.Waa, a_prev) + self.ba)
        y_hat = tf.nn.softmax(tf.matmul(self.Wya, a_next) + self.by)

        return a_next, y_hat

    def rnn_fwd(self, X, vocab_size):
        '''
        Performs RNN forward propagation over sequence.

        Args:
            X: python list of integers representing one input sequence
            a_0: see init_params
            params: see init_params
            vocab_size: number of unique sequence values
        '''
        x, a, y_hat = {}, {}, {}
        a[-1] = tf.Variable(self.a_0)

        for t in np.arange(len(X)):
            x[t] = tf.zeros((vocab_size, 1))
            if X[t] is not None:
                x[t][X[t]] = 1

            a[t], y_hat[t] = self.rnn_fwd_one_step(a[t-1], x[t])

        cache = {'x': x, 'a': a, 'y_hat': y_hat}

        return cache

    def calc_cost(self, y_hat, Y):
        '''
        Calculates cross entropy loss.

        Args:
            y_hat: dictionary of y_hat per timestamp of one sequence
            Y: python list of integers representing one sequencer
        '''
        loss = 0
        for t in np.arange(len(y_hat)):
            loss -= tf.log(y_hat[t][Y[t], 0])

        return loss

    def train(self, loss, lr):
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = opt.minimize(loss)

        return train_op
