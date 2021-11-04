# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
import numpy.linalg as npl


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    N = y.size
    w = npl.solve(tx.T @ tx + 2*N*lambda_, tx.T @ y)
    e = y - tx @ w
    mse = npl.norm(e, 2)**2 / (2*N) + lambda_ * npl.norm(w, 2)**2
    return mse, w
