__author__ = 'arenduchintala'
import numpy as np
from numpy import float32 as DTYPE


def pointwise_multiply(m1, m2):
    return np.multiply(m1, m2)


def clip(m1):
    m1[m1 < 1.0e-100] = 0.0
    return m1


def normalize(m1):
    if np.sum(m1) > 0.0:
        try:
            return m1 / np.sum(m1)
        except FloatingPointError:
            print np.sum(m1)
            print np.sum(m1) > 0.0
            raise FloatingPointError()
    else:
        return np.zeros(np.shape(m1))


def matrix_multiply(m1, m2):
    return m1.dot(m2)
