__author__ = 'johannes'

import numpy as np


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def gaussian_kernel(x1, x2, sigma):
    t = x1 - x2
    if len(t.shape) == 1:
        res = np.exp((-1 * np.linalg.norm(t, ord=2) ** 2)/(2*sigma**2))
    else:
        res = np.exp((-1 * np.linalg.norm(t, axis=1, ord=2) ** 2)/(2*sigma**2))
    return res