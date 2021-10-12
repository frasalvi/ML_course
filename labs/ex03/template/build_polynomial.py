# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    if(len(x.shape) == 1):
        x = x[np.newaxis]
    X = x.T ** np.arange(degree+1)
#     X = np.vander(x, degree+1, increasing=True)
    return X
