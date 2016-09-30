__author__ = 'arenduchintala'
import numpy as np
cimport numpy as np
import warnings
import time
import itertools
from scipy import sparse

np.set_printoptions(linewidth=300, precision=12)


cpdef pointwise_multiply(m1, m2):
    #lm1 = np.log(m1)
    #lm2 = np.log(m2)
    #return np.exp(lm1 + lm2)
     return np.multiply(m1, m2)
    
cpdef clip(m1):
    m1[m1 < 1.0e-100] = 0.0  # we dont have to worry about these being negative...
    return m1


cpdef sparse_normalize(m1, c_idx, r_idx):
    s = np.sum(m1[np.ix_(c_idx, r_idx)])
    m1[np.ix_(c_idx, r_idx)] = m1[np.ix_(c_idx, r_idx)] / s
    return m1


cpdef normalize(m1):
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


cpdef induce_s_pointwise_multiply_clip(d1, d2):
    cdef int K = 100
    if __debug__: assert np.shape(d1) == np.shape(d2)
    indices = (-d1).argpartition(K, axis=None)[:K]
    x, y = np.unravel_index(indices, d1.shape)
    result = np.zeros_like(d2)
    result[x, y] = d1[x, y] * d2[x, y]
    return result


cpdef induce_s(m1):
    cdef int K = 100
    if __debug__: assert np.shape(m1)[1] == 1
    if K > np.size(m1):
        return m1
    else:
        indices = (-m1).argpartition(K, axis=None)[:K]
        x, y = np.unravel_index(indices, m1.shape)
        new_m1 = np.zeros_like(m1)
        new_m1[x, y] = m1[x, y]
        return new_m1


cpdef induce_s_mutliply_clip(s1, d2):
    cdef int K = 100
    if __debug__: assert np.shape(d2)[0] < np.shape(d2)[1]
    if __debug__: assert np.shape(s1)[0] == np.shape(d2)[1] and np.shape(s1)[1] == 1
    s1_abs = np.abs(s1)
    s1_abs = np.reshape(s1_abs, (np.size(s1),))
    max_idx = np.argpartition(s1_abs, -K)[-K:]
    d2_trunc = d2[:, max_idx]
    s1_trunc = s1[max_idx, :]
    return d2_trunc.dot(s1_trunc)


cpdef induce_s_multiply_threshold(s1, d2):
    raise NotImplementedError("do not use it seems very slow..")
    if __debug__: assert np.shape(d2)[0] < np.shape(d2)[1]
    if __debug__: assert np.shape(s1)[0] == np.shape(d2)[1] and np.shape(s1)[1] == 1
    tt = 1.0 / np.size(s1) ** 2
    s1_approx = s1[np.abs(s1) > tt]
    s1_approx_nz = np.nonzero(s1_approx)[0]
    s1_trunc = s1[s1_approx_nz, :]
    d2_trunc = d2[:, s1_approx_nz]
    return d2_trunc.dot(s1_trunc), s1_approx_nz


cpdef np.ndarray[np.float32_t, ndim=2]  dense_dot(np.ndarray[np.float32_t, ndim=2] m1, np.ndarray[np.float32_t, ndim=2] m2):
    return m1.dot(m2)

cpdef np.ndarray[np.float32_t, ndim=2]  dense_pointwise_multiply(np.ndarray[np.float32_t, ndim=2] m1, np.ndarray[np.float32_t, ndim=2] m2):
    return np.multiply(m1, m2)

cpdef make_sparse_and_dot(m1, m2):
    cdef int K = 100
    m1_shaped = np.reshape(m1, np.size(m1))
    m2_shaped = np.reshape(m2, np.size(m2))
    m1_max_idx = np.argpartition(m1_shaped, -K)[-K:]
    m2_max_idx = np.argpartition(m2_shaped, -K)[-K:]
    d = {}
    for x, y in itertools.product(m1_max_idx, m2_max_idx):
        d[x, y] = m1[x, 0] * m2[0, y]
    return d


cpdef np.ndarray[np.float32_t, ndim=2] sparse_pointwise_multiply(np.ndarray[np.float32_t, ndim=2] sparse_m, 
        np.ndarray[np.int_t, ndim=1] c_idx, 
        np.ndarray[np.int_t, ndim=1] r_idx, 
        np.ndarray[np.float32_t, ndim=2] dense_m):
    z = np.zeros_like(dense_m)
    z[np.ix_(c_idx, r_idx)] = np.multiply(sparse_m[np.ix_(c_idx, r_idx)], dense_m[np.ix_(c_idx, r_idx)])
    return z


cpdef sparse_dot(np.ndarray[np.float32_t, ndim=2] m1,  np.ndarray[np.float32_t, ndim=2] m2):
    cdef int K = 100
    assert m1.shape[0] == m2.shape[1]
    assert m1.shape[1] == m2.shape[0] == 1
    n = m1.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] m1_idx = np.empty(K, dtype=np.int)
    cdef np.ndarray[np.int_t, ndim=1] m2_idx = np.empty(K, dtype=np.int)
    cdef np.ndarray[np.float32_t, ndim=2] out = np.zeros((n,n), dtype=np.float)
    m1_idx = np.argpartition(-m1, K-1, axis=0)[:K].ravel()
    m2_idx = np.argpartition(-m2, K-1)[:, :K].ravel()
    #out = np.zeros((n, n))
    out[np.ix_(m1_idx, m2_idx)] = np.dot(m1[m1_idx], m2[:, m2_idx])
    return out, m1_idx, m2_idx


cpdef sparse_multiply_and_normalize(s_m1, m2):
    m2_z = np.zeros_like(m2)
    m2_d = {}
    n = 0.0
    for (x, y), v in s_m1.iteritems():
        m2_z[x, y] = m2[x, y] * v
        n += m2_z[x, y]
    for x, y in s_m1:
        m2_z[x, y] = m2_z[x, y] / n
        m2_d[x, y] = m2_z[x, y]
    return m2_z, m2_d


cpdef sd_matrix_multiply(s1, d2):
    return s1.dot(d2)


cpdef sd_pointwise_multiply(s1, d2):
    raise NotImplementedError("not implemented pointwise multiply for sparse-dense matrix")


cpdef ss_matix_multiply(s1, s2):
    return s1.dot(s2)


cpdef make_adapt_phi(phi, num_adaptations):
    adapt_phi = np.zeros((np.shape(phi)[0], np.shape(phi)[1] * (num_adaptations + 1)))
    r = range(0, np.shape(phi)[1])
    adapt_phi[:, r] = phi
    return adapt_phi


cpdef set_adaptation(f_size, adapt_phi, active_adaptations):
    f = f_size
    r_0 = range(f_size)
    for i in active_adaptations:
        st = i * f
        r = range(st, st + f)
        adapt_phi[:, r] = adapt_phi[:, r_0]
    else:
        pass
    return adapt_phi


cpdef set_adaptation_off(f_size, adapt_phi, active_adaptations):
    f = f_size
    for i in active_adaptations:
        st = i * f
        r = range(st, st + f)
        adapt_phi[:, r] = 0
    else:
        pass
    return adapt_phi


cpdef set_original(phi, adapt_phi):
    r = range(0, np.shape(phi)[1])
    adapt_phi[:, r] = phi
    return adapt_phi


cpdef sparse_vec_mat_dot(np.ndarray[np.float32_t, ndim=2]  vec, np.ndarray[np.float32_t, ndim=2] mat):
    cdef int K = 100
    cdef np.ndarray[np.int_t, ndim=1] m1_idx = np.empty(K, dtype=np.int)
    if vec.shape[0] == 1:
        # row vector
        m_idx = np.argpartition(-vec[0,:], K-1)[:K]
        r = np.dot(vec[0,m_idx], mat[m_idx,:])
        return r
    else:
        #col vector
        m_idx = np.argpartition(-vec[:,0], K-1)[:K]
        r = np.dot(mat[:, m_idx], vec[m_idx])
        return r


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
