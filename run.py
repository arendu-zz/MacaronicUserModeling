__author__ = 'arenduchintala'
import numpy as np
from LBP import VAR_TYPE_PREDICTED, VariableNode, PotentialTable, FactorGraph, FactorNode

if __name__ == '__main__':
    d = open('big_data.txt').readlines()
    n, k = [int(i) for i in d[0].split()]

    pots = {}
    for l in d[1:]:

        items = l.split()
        items = [float(i) for i in items]
        if len(items) == 3:
            fac_id, outcome, value = items
            outcome = int(outcome - 1)
            fac_id = int(fac_id - 1)
            pot = pots.get(fac_id, np.zeros((k, 1)))
            pot[outcome] = value
            pots[fac_id] = pot
        else:
            fac_id, outcome1, outcome2, value = items
            outcome1 = int(outcome1 - 1)
            outcome2 = int(outcome2 - 1)
            fac_id = int(fac_id - 1)
            pot = pots.get(fac_id, np.zeros((k, k)))
            pot[outcome1, outcome2] = value
            pots[fac_id] = pot

    variables_nodes = []
    for i in range(n):
        v = VariableNode(id=i, var_type=VAR_TYPE_PREDICTED, domain_type=None,
                         domain=["outcome_" + str(int(i + 1)) for i in range(k)],
                         supervised_label="outcome_" + str(int(i + 1)))
        variables_nodes.append(v)

    factor_nodes = []
    for fac_id, fac_potential in pots.iteritems():
        f = FactorNode(fac_id)
        factor_nodes.append(f)

    fg = FactorGraph()
    for f in factor_nodes[:n]:
        p = PotentialTable(v_id2dim={variables_nodes[f.id].id: 0}, table=pots[f.id])
        f.add_varset_with_potentials(varset=[variables_nodes[f.id]], ptable=p)
        fg.add_factor(f)

    for idx, f in enumerate(factor_nodes[n:-1]):
        p = PotentialTable(v_id2dim={variables_nodes[idx].id: 0, variables_nodes[idx + 1].id: 1}, table=pots[f.id])
        f.add_varset_with_potentials(varset=[variables_nodes[idx], variables_nodes[idx + 1]], ptable=p)
        fg.add_factor(f)

    f = factor_nodes[2 * n - 1]
    p = PotentialTable(v_id2dim={variables_nodes[0].id: 0, variables_nodes[n - 1].id: 1}, table=pots[f.id])
    f.add_varset_with_potentials(varset=[variables_nodes[0], variables_nodes[n - 1]], ptable=p)
    fg.add_factor(f)

    fg.initialize()
    fg.treelike_inference(3)
    # fg.hw_inf(10)
    for v in fg.variables:
        m = v.get_factor_beliefs()
        print 'Var:', v.id, np.reshape(m.m, (np.size(m.m),))
    print 'has loops?', fg.isLoopy
