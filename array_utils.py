__author__ = 'arenduchintala'
import numpy as np
import time
import itertools
from scipy import sparse
from numpy import float32 as DTYPE

np.set_printoptions(linewidth=300, precision=2)


class PhiWrapper(object):
    def __init__(self, phi_en_en, phi_en_en_w1, phi_en_de):
        self.phi_en_en = phi_en_en
        self.phi_en_en_w1 = phi_en_en_w1
        self.phi_en_de = phi_en_de


class ThetaWrapper(object):
    def __init__(self, theta_en_en_names, theta_en_en, theta_en_de_names, theta_en_de):
        self.theta_en_en_names = theta_en_en_names
        self.theta_en_de_names = theta_en_de_names
        self.theta_en_en = theta_en_en
        self.theta_en_de = theta_en_de


def pointwise_multiply(m1, m2):
    return np.multiply(m1, m2)


def clip(m1):
    m1[m1 < 1.0e-100] = 0.0  # we dont have to worry about these being negative...
    return m1


def normalize(m1):
    s = np.sum(m1)
    if s > 0.0:
        try:
            return m1 / s
        except FloatingPointError:
            print s
            print s > 0.0
            raise FloatingPointError()
    else:
        m1.fill(0)
        return m1


def induce_s_pointwise_multiply_clip(d1, d2, k=1000):
    if __debug__: assert np.shape(d1) == np.shape(d2)
    indices = (-d1).argpartition(k, axis=None)[:k]
    x, y = np.unravel_index(indices, d1.shape)
    result = np.zeros_like(d2)
    result[x, y] = d1[x, y] * d2[x, y]
    return result


def induce_s(m1, k=1000):
    if __debug__: assert np.shape(m1)[1] == 1
    if k > np.size(m1):
        return m1
    else:
        indices = (-m1).argpartition(k, axis=None)[:k]
        x, y = np.unravel_index(indices, m1.shape)
        new_m1 = np.zeros_like(m1)
        new_m1[x, y] = m1[x, y]
        return new_m1


def induce_s_mutliply_clip(s1, d2, k=1000):
    if __debug__: assert np.shape(d2)[0] < np.shape(d2)[1]
    if __debug__: assert np.shape(s1)[0] == np.shape(d2)[1] and np.shape(s1)[1] == 1
    k = k if k < np.shape(s1)[0] else np.shape(s1)[0] - 1
    s1_abs = np.abs(s1)
    s1_abs = np.reshape(s1_abs, (np.size(s1),))
    max_idx = np.argpartition(s1_abs, -k)[-k:]
    d2_trunc = d2[:, max_idx]
    s1_trunc = s1[max_idx, :]
    return d2_trunc.dot(s1_trunc)


def induce_s_multiply_threshold(s1, d2):
    raise NotImplementedError("do not use it seems very slow..")
    if __debug__: assert np.shape(d2)[0] < np.shape(d2)[1]
    if __debug__: assert np.shape(s1)[0] == np.shape(d2)[1] and np.shape(s1)[1] == 1
    tt = 1.0 / np.size(s1) ** 2
    s1_approx = s1[np.abs(s1) > tt]
    s1_approx_nz = np.nonzero(s1_approx)[0]
    s1_trunc = s1[s1_approx_nz, :]
    d2_trunc = d2[:, s1_approx_nz]
    return d2_trunc.dot(s1_trunc), s1_approx_nz


def dd_matrix_multiply(m1, m2):
    return m1.dot(m2)


def make_sparse_and_dot(m1, m2, k=100):
    m1_shaped = np.reshape(m1, np.size(m1))
    m2_shaped = np.reshape(m2, np.size(m2))
    m1_max_idx = np.argpartition(m1_shaped, -k)[-k:]
    m2_max_idx = np.argpartition(m2_shaped, -k)[-k:]
    d = {}
    for x, y in itertools.product(m1_max_idx, m2_max_idx):
        d[x, y] = m1[x, 0] * m2[0, y]
    return d


def sparse_multiply_and_normalize(s_m1, m2):
    m2_z = np.zeros_like(m2)
    n = 0.0
    for (x, y), v in s_m1.iteritems():
        m2_z[x, y] = m2[x, y] * v
        n += m2_z[x, y]
    for x, y in s_m1:
        m2_z[x, y] = m2_z[x, y] / n
    return m2_z


def sd_matrix_multiply(s1, d2):
    return s1.dot(d2)


def sd_pointwise_multiply(s1, d2):
    raise NotImplementedError("not implemented pointwise multiply for sparse-dense matrix")


def ss_matix_multiply(s1, s2):
    return s1.dot(s2)


def make_adapt_phi(phi, num_adaptations):
    adapt_phi = np.zeros((np.shape(phi)[0], np.shape(phi)[1] * (num_adaptations + 1)))
    r = range(0, np.shape(phi)[1])
    adapt_phi[:, r] = phi
    return adapt_phi


def set_adaptation(f_size, adapt_phi, active_adaptations):
    f = f_size
    r_0 = range(f_size)
    for i in active_adaptations:
        st = i * f
        r = range(st, st + f)
        adapt_phi[:, r] = adapt_phi[:, r_0]
    else:
        pass
    return adapt_phi


def set_adaptation_off(f_size, adapt_phi, active_adaptations):
    f = f_size
    for i in active_adaptations:
        st = i * f
        r = range(st, st + f)
        adapt_phi[:, r] = 0
    else:
        pass
    return adapt_phi


def set_original(phi, adapt_phi):
    r = range(0, np.shape(phi)[1])
    adapt_phi[:, r] = phi
    return adapt_phi


if __name__ == '__main__':
    s = 10
    f = 4
    u = 5
    active_user = [1, 4]
    ss2 = time.time()
    phi = np.random.rand(s, f)
    phi_off = np.zeros_like(phi)
    adapt_phi = np.zeros((np.shape(phi)[0], np.shape(phi)[1] * (u + 1)))
    adapt_phi = set_original(phi, adapt_phi)
    adapt_phi = set_adaptation(f, adapt_phi, [1, 3])
    print adapt_phi
