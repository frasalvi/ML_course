# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy.linalg as npl
import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = y.size
    e = y - np.dot(tx, w)
    # MSE
#     L = np.dot(e.T, e) / (2*N)
    L = npl.norm(e, 2)**2 / (2*N)
    # MAE
#     L = npl.norm(e, 1) / N
    return L
