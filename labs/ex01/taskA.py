

import numpy as np


def standardise(A):
    A -= np.mean(A, axis=0)
    A /= np.std(A, axis=0)
    return A


a = np.random.rand(4, 6)
standardise(a)
print(np.mean(a, axis=0))
print(np.std(a, axis=0))
