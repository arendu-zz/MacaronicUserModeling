__author__ = 'arenduchintala'
import itertools
import numpy as np
from scipy import sparse
from LBP import FactorNode, FactorGraph, VariableNode, VAR_TYPE_PREDICTED, PotentialTable, VAR_TYPE_GIVEN
import timeit

np.seterr(all='raise')
np.set_printoptions(precision=4, suppress=True)
if __name__ == '__main__':

    en_domain = ['en_' + str(i) for i in range(50)]
    de_domain = ['de_' + str(i) for i in range(10)]

    f_en_en = ['f1', 'f2', 'f3', 'f4', 'f5']

    f_en_en_theta = np.zeros((1, len(f_en_en)))
    phi_en_en = np.random.rand(len(en_domain) * len(en_domain), len(f_en_en))
    phi_en_en[phi_en_en > 0.8] = 1.0
    phi_en_en[phi_en_en < 1.0] = 0.0
    # pre_fire_en_en = sparse.csr_matrix(pre_fire_en_en)

    f_en_de = ['x', 'y', 'z']
    f_en_de_theta = np.zeros((1, len(f_en_de)))
    phi_en_de = np.random.rand(len(en_domain) * len(de_domain), len(f_en_de))
    phi_en_de[phi_en_de > 0.5] = 1.0
    phi_en_de[phi_en_de < 0.2] = 0.0
    # pre_fire_en_de = sparse.csr_matrix(pre_fire_en_de)
    print timeit.timeit()
    for _ in range(10):
        fg = FactorGraph(theta_en_en=f_en_en_theta,
                         theta_en_de=f_en_de_theta,
                         phi_en_en=phi_en_en,
                         phi_en_de=phi_en_de)

        variables = []
        sent_len = 12
        pred_or_given = [1] * 12
        pred_or_given[5] = 0
        for i in range(sent_len):
            p_or_g = VAR_TYPE_PREDICTED if pred_or_given[i] == 1 else VAR_TYPE_GIVEN
            v = VariableNode(id=i, var_type=p_or_g,
                             domain_type='en',
                             domain=en_domain,
                             supervised_label=en_domain[np.random.randint(0, len(en_domain))])
            variables.append(v)

        factors = []
        for v1 in variables:
            # print 'making u-factor', v1.
            if v1.var_type == VAR_TYPE_PREDICTED:
                f = FactorNode(id=len(factors), factor_type='en_de',
                               observed_domain_type='de',
                               observed_domain_size=len(de_domain))
                o_idx = np.random.randint(len(de_domain) / 2, len(de_domain))
                p = PotentialTable(v_id2dim={v1.id: 0}, table=None, observed_dim=o_idx)
                f.add_varset_with_potentials(varset=[v1], ptable=p)
                factors.append(f)
            elif v1.var_type == VAR_TYPE_GIVEN:
                pass
            else:
                raise BaseException("vars are given or predicted only (no latent)")

        for v1 in variables:
            for v2 in variables:
                if v1.id < v2.id:
                    # print 'making bi-factor', v1.id, v2.id
                    if v1.var_type == VAR_TYPE_PREDICTED and v2.var_type == VAR_TYPE_PREDICTED:

                        f = FactorNode(id=len(factors), factor_type='en_en')
                        p = PotentialTable(v_id2dim={v1.id: 0, v2.id: 1}, table=None, observed_dim=None)
                        f.add_varset_with_potentials(varset=[v1, v2], ptable=p)
                        factors.append(f)

                    elif v1.var_type == VAR_TYPE_GIVEN and v2.var_type == VAR_TYPE_GIVEN:
                        pass
                    else:
                        v_given = v1 if v1.var_type == VAR_TYPE_GIVEN else v2
                        v_pred = v1 if v1.var_type == VAR_TYPE_PREDICTED else v2

                        f = FactorNode(id=len(factors),
                                       factor_type='en_en',
                                       observed_domain_type='en',
                                       observed_domain_size=len(en_domain))
                        o_idx = np.random.randint(0, len(en_domain))
                        p = PotentialTable(v_id2dim={v_pred.id: 0}, table=None, observed_dim=o_idx)
                        f.add_varset_with_potentials(varset=[v_pred], ptable=p)
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
        # for v in fg.variables:
        # m = v.get_marginal()
        #    print np.reshape(m.m, (np.size(m.m),))
        fg.treelike_inference(3)
        grad_en_de, grad_en_en = fg.get_gradient()
        fg.theta_en_en += 0.1 * grad_en_en
        fg.theta_en_de += 0.1 * grad_en_de
        f_en_en_theta = fg.theta_en_en
        f_en_de_theta = fg.theta_en_de
        print f_en_de_theta
        print f_en_en_theta
