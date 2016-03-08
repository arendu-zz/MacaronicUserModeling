__author__ = 'arenduchintala'
import numpy as np
from numpy import float32 as DTYPE


def pointwise_multiply(m1, m2):
    return np.multiply(m1, m2)


def clip(m1):
    m1[m1 < 1.0e-100] = 0.0  # we dont have to worry about these being negative...
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


def dd_matrix_multiply(m1, m2):
    return m1.dot(m2)


def sd_matrix_multiply(s1, d2):
    return s1.dot(d2)


def sd_pointwise_multiply(s1, d2):
    raise NotImplementedError("not implemented pointwise multiply for sparse-dense matrix")


def ss_matix_multiply(s1, s2):
    return s1.dot(s2)
