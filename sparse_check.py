__author__ = 'arenduchintala'
import numpy as np
from scipy import sparse
import time

if __name__ == '__main__':

    a = np.random.randint(0, 10, (10,))
    kkk = [np.argpartition(a, -3)][-3:]
    print a
    print kkk

    print 'pok'
    feature_size = 5
    for d in [500, 1000, 2000, 5000]:
        domain_size = d
        print '\nsize:', domain_size
        for t in range(5, 9):
            threshold = t * 0.1
            print '\t\nthreshold:', threshold

            np_creation_time = []
            for _ in range(5):
                np_t = time.time()
                a = np.random.rand(domain_size, domain_size, feature_size)
                np_creation_time.append(time.time() - np_t)

            a = np.reshape(a, (domain_size * domain_size, feature_size))
            a[a < threshold] = 0
            a_sparse_np = a

            csr_creation_time = []
            for _ in range(5):
                crt = time.time()
                a_sparse_csr = sparse.csr_matrix(a)
                csr_creation_time.append(time.time() - crt)

            csc_creation_time = []
            for _ in range(5):
                cct = time.time()
                a_sparse_csc = sparse.csc_matrix(a)
                csc_creation_time.append(time.time() - cct)

            print '\tnp  creation time:', sum(np_creation_time) / len(np_creation_time)
            print '\tcsr creation time:', sum(csr_creation_time) / len(csr_creation_time)
            print '\tcsc creation time:', sum(csc_creation_time) / len(csc_creation_time)

            b = np.random.rand(domain_size * domain_size, 1)

            mul_time = []
            for _ in range(5):
                np_mul = time.time()
                f = a_sparse_np.T.dot(b)
                mul_time.append(time.time() - np_mul)

            print '\tnp mul time :', sum(mul_time) / len(mul_time)

            mul_time = []
            for _ in range(5):
                np_mul = time.time()
                f = a_sparse_csr.T.dot(b)
                mul_time.append(time.time() - np_mul)

            print '\tcsr mul time:', sum(mul_time) / len(mul_time)

            mul_time = []
            for _ in range(5):
                np_mul = time.time()
                f = a_sparse_csc.T.dot(b)
                mul_time.append(time.time() - np_mul)

            print '\tcsc mul time:', sum(mul_time) / len(mul_time)
