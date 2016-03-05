__author__ = 'arenduchintala'
import itertools
import numpy as np
from scipy import sparse
from LBP import FactorNode, FactorGraph, VariableNode, VAR_TYPE_PREDICTED, PotentialTable
import timeit

np.seterr(all='raise')
np.set_printoptions(precision=4, suppress=True)
if __name__ == '__main__':

    en_domain = ['en_' + str(i) for i in range(1000)]
    de_domain = ['de_' + str(i) for i in range(100)]

    f_en_en = ['f1', 'f2', 'f3', 'f4', 'f5']

    f_en_en_theta = np.ones((1, len(f_en_en)))
    pre_fire_en_en = np.random.rand(len(en_domain) * len(en_domain), len(f_en_en))
    pre_fire_en_en[pre_fire_en_en > 0.8] = 1.0
    pre_fire_en_en[pre_fire_en_en < 1.0] = 0.0
    # pre_fire_en_en = sparse.csr_matrix(pre_fire_en_en)

    f_en_de = ['x', 'y', 'z']
    f_en_de_theta = np.ones((1, len(f_en_de)))
    pre_fire_en_de = np.random.rand(len(en_domain) * len(de_domain), len(f_en_de))
    pre_fire_en_de[pre_fire_en_de > 0.5] = 1.0
    pre_fire_en_de[pre_fire_en_de < 0.2] = 0.0
    # pre_fire_en_de = sparse.csr_matrix(pre_fire_en_de)
    print timeit.timeit()
    fg = FactorGraph(theta_en_en=f_en_en_theta,
                     theta_en_de=f_en_de_theta,
                     phi_en_en=pre_fire_en_en,
                     phi_en_de=pre_fire_en_de)

    variables = []
    for i in range(20):
        v = VariableNode(id=i, var_type=VAR_TYPE_PREDICTED,
                         domain_type='en',
                         domain=en_domain,
                         supervised_label=en_domain[np.random.randint(0, 100)])
        variables.append(v)

    factors = []
    for v1 in variables:
        # print 'making u-factor', v1.id
        f = FactorNode(id=len(factors), factor_type='en_de', observed_domain_size=len(de_domain))
        o_idx = np.random.randint(len(de_domain) / 2, len(de_domain))
        p = PotentialTable(v_id2dim={v1.id: 0}, table=None, observed_dim=o_idx)
        f.add_varset_with_potentials(varset=[v1], ptable=p)
        factors.append(f)

    for v1 in variables:
        for v2 in variables:
            if v1.id < v2.id:
                # print 'making bi-factor', v1.id, v2.id
                f = FactorNode(id=len(factors), factor_type='en_en')
                p = PotentialTable(v_id2dim={v1.id: 0, v2.id: 1}, table=None, observed_dim=None)
                f.add_varset_with_potentials(varset=[v1, v2], ptable=p)
                factors.append(f)
            else:
                pass

    for f in factors:
        fg.add_factor(f)

    for f in fg.factors:
        f.potential_table.make_potentials()

    # print en_de_pot
    # print en_en_pot

    fg.initialize()
    print fg.isLoopy
    for v in fg.variables:
        m = v.get_marginal()
        print np.reshape(m.m, (np.size(m.m),))
    fg.treelike_inference(3)
    grad_en_de, grad_en_en = fg.get_gradient()
    fg.theta_en_en += 0.1 * grad_en_en
    fg.theta_en_de += 0.1 * grad_en_de
