# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:52:12 2021

@author: franc
"""

import numpy as np
import numpy.linalg as npl


def likelihood(x, *theta):
    '''
    Parameters
    ----------
    x : list of n list of d float
        The data sample. Each point is an array of d features.
    thetai : [mui, sigmai]
        The parameters of the i-model. mui is the d-list mean, sigmai the dxd
        covariance matrix

    Returns a, list of most likely model
    -------
    '''
    mu = [t[0] for t in theta]
    sigma = [t[1] for t in theta]
    d = x.shape[1]
    p = [1 / ((2*np.pi)**(d/2) * abs(npl.det(Sigma))**.5)
         * np.exp(-1/2*np.diagonal(np.dot(x-Mu,
                                          np.dot(npl.inv(Sigma), (x-Mu).T))))
         for Mu, Sigma in zip(mu, sigma)]
    a = np.argmax(list(zip(*p)), axis=1)
    return a+1


theta = ([np.array([4, 5, 2]), np.eye(3)],
         [np.array([7, 5, 2]), np.eye(3)])
x = np.array([[6, 7, 6], [5, 4, 2], [10, 3, 5], [6, 3, 9]])
a = likelihood(x, *theta)
