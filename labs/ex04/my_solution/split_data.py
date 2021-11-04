# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    N = x.size
    k = np.random.permutation(N)
    x = np.take(x, k)
    y = np.take(y, k)
    sep = int(N*ratio)
    x_train, y_train = x[0:sep], y[0:sep]
    x_test, y_test = x[sep:], y[sep:]
    return x_train, y_train, x_test, y_test
