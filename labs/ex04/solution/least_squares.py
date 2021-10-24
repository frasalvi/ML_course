# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
import numpy.linalg as npl


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns optimal weights, MSE
    # ***************************************************
    w = npl.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    N = y.size
    e = y - np.dot(tx, w)
    mse = npl.norm(e)**2 / (2*N)
    return mse, w
