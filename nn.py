#!/usr/bin/env python3

"""

nn
==

This module contains a neural network class and some useful neural network functions.

"""


import numpy as np


class Model:
    
    def __init__(self, data, attributes):
        """

        Parameters
        ----------
        data : dict
        attributes : dict

        """
        self.features = attributes['features']
        self.activation_type = data['activation_type']
        self.layers = data['layers']
        self.p = data['p']
        self.predict_log_value = data['predict_log_value']
        self.mu_x = data['mu_x']
        self.std_x = data['std_x']
        self.mu_y = data['mu_y']
        self.std_y = data['std_y']
        assert len(self.features) == self.layers[0]
        assert len(self.mu_x) == self.layers[0]
        assert len(self.std_x) == self.layers[0]
    
     
    def predict(self, x_raw_new):
        """
        Return prediction from neural network model.
    
        Parameters
        ----------
        x_ray_new : real array(N,M) or array(N) or list(N)
            Number of features N.
            Number of data examples M.

        """
        x_raw_new = np.array(x_raw_new)
        if x_raw_new.ndim == 1:
            x_raw_new = np.atleast_2d(x_raw_new).T
        elif x_raw_new.ndim > 2:
            sys.exit('Unexpected input dimension of apartment features')
        # Normalize and scale input
        x = norm_and_scale(x_raw_new, self.mu_x, self.std_x)
        # Calculate output from neural network
        y = hypothesis(x, self.p, self.layers, self.activation_type)
        # Go back to un-normalized and un-scaled output
        y_raw = y*self.std_y + self.mu_y
        if self.predict_log_value:
            y_raw = np.exp(y_raw)
        y_raw = y_raw.flatten()
        if y_raw.size == 1:
            y_raw = y_raw[0]
        return y_raw


def norm_and_scale(x, mu, std):
    """
    Return normalized data.
    """
    if x.ndim == 1:
        # Normalize and scale input
        x_ns = (x - mu)/std
    elif x.ndim == 2:
        # Number of features
        n = np.shape(x)[0]
        # Normalize and scale input
        x_ns = np.zeros_like(x)
        for i in range(n):
            x_ns[i,:] = (x[i,:] - mu[i])/std[i]
    else:
        sys.exit("Not implemented yet...")
    return x_ns


def hypothesis(x, p, layers, activation_type='sigmoid',
                  logistic_output=False):
    """
    Return the neural network prediction.

    Forward propagation.

    """
    if activation_type == 'sigmoid':
        g = lambda v: sigmoid(v)
    elif activation_type == 'ReLu':
        g = lambda v: relu(v)
    else:
        sys.exit("This activation function has not been implemented.")

    # Number of features and number of data examples
    n, m = np.shape(x)
    # Model parameters, unrolled
    w, b = unroll(p, layers)
    # Model predictions
    h = np.zeros(m, dtype=np.float)

    # Parallelized version, no loop over examples
    z = x
    for j, (wm, bv) in enumerate(zip(w, b)):
        z = np.dot(wm, z) + bv
        # Last layer is treated specially
        if j == len(layers) - 2:
            if logistic_output:
                h = g(z)
            else:
                h = z
        else:
            z = g(z)
    return h


def sigmoid(z):
    """
    Return the sigmoid function.
    """
    return 1/(1 + np.exp(-z))


def dsigmoid(z):
    """
    Return the derivative of the sigmoid function.
    """
    return sigmoid(z)*(1-sigmoid(z))


def relu(z):
    """
    Return the ReLu function.
    """
    return z * (z > 0)


def drelu(z):
    """
    Return the derivative of the ReLu function.
    """
    return 1.*(z>0)


def unroll(p, layers):
    w = []
    b = []
    j, k = 0, 0
    for i in range(len(layers[:-1])):
        k = j + layers[i+1]*layers[i]
        w.append(p[j:k].reshape((layers[i+1],layers[i])))
        j = k
        k += layers[i+1]
        b.append(p[j:k].reshape((layers[i+1],1)))
        j = k
    return w, b


def inroll(w, b):
    # Number of parameters
    r = 0
    for i in range(len(w)):
        r += w[i].size + b[i].size
    p = np.zeros(r, dtype=np.float)
    # Fill p
    j, k = 0, 0
    for i in range(len(w)):
        k = j + w[i].size
        p[j:k] = w[i].reshape(w[i].size)
        j = k
        k += b[i].size
        p[j:k] = b[i].reshape(b[i].size)
        j = k
    return p


