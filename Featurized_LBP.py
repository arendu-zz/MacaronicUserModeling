__author__ = 'arenduchintala'
import itertools
import numpy as np
from scipy import sparse
from LBP import FactorNode, FactorGraph, VariableNode, VAR_TYPE_PREDICTED, PotentialTable

np.seterr(all='raise')
np.set_printoptions(precision=4, suppress=True)
if __name__ == '__main__':
    fg = FactorGraph()
    en_domain = ['en_' + str(i) for i in range(300)]
    de_domain = ['de_' + str(i) for i in range(100)]
    f_labels = ['en_same', 'en_odd', 'en_even', 'de_odd', 'de_even']
    en_en_features = {}
    for en1, en2 in itertools.product(en_domain, en_domain):
        f = [0, 0, 0, 0, 0]  # same size as f_labels
        if en1 == en2:
            f[0] = 1
        if int(en1.split('_')[1]) % 2 == 1 and int(en2.split('_')[1]) % 2 == 1:
            f[1] = 1
        if int(en1.split('_')[1]) % 2 == 1 and int(en2.split('_')[1]) % 2 == 1:
            f[2] = 1
        en_en_features[en1, en2] = f

    en_en_theta = np.zeros((len(en_domain), len(en_domain)))  # assumes \thetas are 0 so exp(\theta \dot features) = 1
    en_en_pot = np.exp(en_en_theta)

    en_de_features = {}
    for en1, de2 in itertools.product(en_domain, de_domain):
        f = [0, 0, 0, 0, 0]
        if int(en1.split('_')[1]) % 2 == 1 and int(de2.split('_')[1]) % 2 == 1:
            f[3] = 1
        if int(en1.split('_')[1]) % 2 == 1 and int(de2.split('_')[1]) % 2 == 1:
            f[4] = 1
        en_de_features[en1, de2] = f
    en_de_theta = np.zeros((len(en_domain), 1))  # assumes \thetas are 0 so exp \theta dot features are 1
    en_de_pot = np.exp(en_de_theta)

    variables = []
    for i in range(20):
        v = VariableNode(id=i, var_type=VAR_TYPE_PREDICTED,
                         domain_type='en',
                         domain=en_domain,
                         supervised_label=en_domain[np.random.randint(0, 100)])
        variables.append(v)
    factors = []
    for v1 in variables:
        print 'making u-factor', v1.id
        f = FactorNode(id=len(factors), factor_type='en_de')
        p = PotentialTable(v_id2dim={v1.id: 0}, table=en_de_pot)
        f.add_varset_with_potentials(varset=[v1], ptable=p)
        factors.append(f)

    for v1 in variables:
        for v2 in variables:
            if v1.id < v2.id:
                print 'making bi-factor', v1.id, v2.id
                f = FactorNode(id=len(factors), factor_type='en_en')
                p = PotentialTable(v_id2dim={v1.id: 0, v2.id: 1}, table=en_en_pot)
                f.add_varset_with_potentials(varset=[v1, v2], ptable=p)
                factors.append(f)
            else:
                pass

    for f in factors:
        fg.add_factor(f)

    # print en_de_pot
    # print en_en_pot
    fg.initialize()
    print fg.isLoopy
    for v in fg.variables:
        m = v.get_factor_beliefs()
        print np.reshape(m.m, (np.size(m.m),))
    fg.treelike_inference(3)
    cg = fg.get_cell_gradient()
    for k, g in cg.iteritems():
        if k == 'en_en':
            cgtg = np.multiply(g, en_en_pot)
            en_en_theta += cgtg * 1.0
            en_en_pot = np.exp(en_en_theta)
        elif k == 'en_de':
            cgtg = np.multiply(g, en_de_pot)
            en_de_theta += cgtg * 1.0
            en_de_pot = np.exp(en_de_theta)
    print en_de_pot
    print en_en_pot
    for f in factors:
        if f.factor_type == 'en_en':
            f.potential_table.table = en_en_pot
        else:
            f.potential_table.table = en_de_pot
    fg.treelike_inference(3)
    for v in fg.variables:
        m = v.get_factor_beliefs()
        print np.reshape(m.m, (np.size(m.m),))
