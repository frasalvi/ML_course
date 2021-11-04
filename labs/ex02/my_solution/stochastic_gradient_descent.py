# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    N = y.size
    e = y - np.dot(tx, w)
    gradL = -np.dot(tx.T, e) / N
    return gradL


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    w = initial_w
    ws = [w]
    losses = []
    for batch_y, batch_tx in batch_iter(y, tx, batch_size, max_iters):
        gradient = compute_stoch_gradient(batch_y, batch_tx, w)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(compute_loss(y, tx, w))
    return losses, ws