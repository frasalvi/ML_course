# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:21:13 2021

@author: franc
"""

import numpy as np
import numpy.linalg as npl


def distance(v1, v2):
    return npl.norm(v1 - v2)


def pairwise_naive(P, Q):
    p = P.shape[0]
    q = Q.shape[0]
    D = np.empty((p, q))
    for i in range(p):
        for j in range(q):
            D[i, j] = distance(P[i], Q[j])
    return D


def pairwise(P, Q):
    D = npl.norm(P[:, np.newaxis, :] - Q[np.newaxis, :, :], axis=2)
    return D


P = np.random.randint(0, 10, (6, 2))
Q = np.random.randint(0, 10, (8, 2))

D1 = pairwise_naive(P, Q)
D2 = pairwise(P, Q)
