__author__ = 'arenduchintala'
import numpy as np

global VAR_TYPE_LATENT, VAR_TYPE_PREDICTED
VAR_TYPE_PREDICTED = 'var_type_predicted'
VAR_TYPE_LATENT = 'var_type_latent'


class FactorGraph():
    def __init__(self):
        self.variables = []
        self.factors = []
        self.messages = {}
        self.normalize_messages = False

    def add_factor(self, fac):
        assert fac not in self.factors
        self.factors.append(fac)
        fac.graph = self
        for v in fac.varset:
            if v not in self.variables:
                self.variables.append(v)
                v.graph = self

    def initialize(self):
        fs = sorted([(f.id, f) for f in self.factors])
        self.factors = [f for fid, f in fs]
        vs = sorted([(v.id, v) for v in self.variables])
        self.variables = [v for vid, v in vs]

        for f in self.factors:
            assert len(f.potential_table.var_id2dim) == len(f.varset)
            vids = [vid for d, vid in sorted([(d, v) for v, d in f.potential_table.var_id2dim.iteritems()])]
            vars = [self.variables[vid] for vid in vids]
            assert np.shape(f.potential_table.table) == tuple([len(v.domain) for v in vars])
            # todo: this check above is not comprehensive must check if dimsion of the potential table
            # todo: matches in the correct order of variables given by var_id2dim
            if len(f.varset) == 1:
                self.messages[str(f), str(f.varset[0])] = Message.new_message(f.varset[0].domain, 1.0)
            else:
                for v in f.varset:
                    self.messages[str(v), str(f)] = Message.new_message(v.domain, 1.0)
                    self.messages[str(f), str(v)] = Message.new_message(v.domain, 1.0)

    def run_inference(self, iterations):
        n = len(self.variables)
        for fu in self.factors[:n]:
            fu.update_message_to(fu.varset[0])
            fu.update_message_to(fu.varset[0])
        for _ in range(iterations):
            print 'iteration', _
            for i in range(1, n + 1):
                fi = i + n
                vi = n if (i + 1) == n else (i + 1) % n
                fii = n + 1 + i % n
                print vi, fi, fii
                self.factors[fi - 1].update_message_to(self.variables[vi - 1])
                self.variables[vi - 1].update_message_to(self.factors[fii - 1])
                print 'ok'

            for i in sorted(range(1, n + 1), reverse=True):
                fi = n + i
                vi = i
                fii = n + (i - 2) % n + 1 if i != 1 else 2 * n
                self.factors[fi - 1].update_message_to(self.variables[vi - 1])
                self.variables[vi - 1].update_message_to(self.factors[fii - 1])


class VariableNode():
    def __init__(self, id, var_type, domain):
        assert isinstance(id, int)
        self.id = id
        self.var_type = var_type
        self.domain = domain
        self.factor_neighbors = []  # list of factorNodes
        self.graph = None

    def __str__(self):
        return "X_" + str(self.id)

    def __eq__(self, other):
        assert type(other) == type(self)
        return self.id == other.id

    def display(self, m):
        isinstance(m, Message)
        raise NotImplementedError()

    def add_factor(self, fc):
        assert isinstance(fc, FactorNode)
        self.factor_neighbors.append(fc)

    def init_message_to(self, fc, init_m):
        assert isinstance(fc, FactorNode)
        assert isinstance(init_m, Message)
        # todo: should we prune initial messages?
        self.messages[self.id, fc.id] = init_m

    def update_message_to(self, fc):
        assert isinstance(fc, FactorNode)
        assert fc in self.factor_neighbors
        new_m = Message.new_message(self.domain, 1.0)
        for other_fc in self.factor_neighbors:
            if other_fc is not fc:
                m = self.graph.messages[str(other_fc), str(self)]  # from other factors to this variable
                new_m = pointwise_multiply(m, new_m)
        if self.graph.normalize_messages:
            new_m.renormalize()
        self.graph.messages[str(self), str(fc)] = new_m

    def get_marginal(self):
        new_m = None
        for fc in self.factor_neighbors:
            m = self.graph.messages[str(fc), str(self)]  # incoming message
            new_m = pointwise_multiply(m, new_m)
        new_m.renormalize()
        return new_m


class FactorNode():
    def __init__(self, id):
        assert isinstance(id, int)
        self.id = id
        self.varset = []
        self.potential_table = None
        self.graph = None

    def __str__(self):
        return 'F_' + str(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def init_message_to(self, var, init_m):
        assert isinstance(var, VariableNode)
        assert isinstance(init_m, Message)
        self.graph.messages[str(self), str(var)] = init_m

    def add_varset_with_potentials(self, varset, ptable):
        assert isinstance(ptable, PotentialTable)
        assert len(varset) == len(ptable.var_id2dim)
        if len(varset) > 2:
            raise NotImplementedError("Currently supporting unary and pairwise factors...")
        for v in varset:
            assert v not in self.varset
            self.varset.append(v)
            v.add_factor(self)
        self.potential_table = ptable

    def update_message_to(self, var):
        pass
        other_vars = [v for v in self.varset if v.id != var.id]
        if len(other_vars) == 0:
            # this is a unary factor
            new_m = np.copy(self.potential_table.table)
            self.graph.messages[str(self), str(var)] = Message(new_m)
        else:
            o_var = other_vars[0]
            o_var_dim = self.potential_table.var_id2dim[o_var.id]
            msg = self.graph.messages[str(o_var), str(self)]
            if o_var_dim == 1:
                mul = np.multiply(msg.m, self.potential_table.table)
                marginalized = np.sum(mul, 1)
            else:
                mul = np.multiply(msg.m, self.potential_table.table.T).T
                marginalized = np.sum(mul, 0)
            new_m = Message(marginalized)
            if self.graph.normalize_messages:
                new_m.renormalize()
            self.graph.messages[str(self), str(var)] = new_m


class ObservedFactor(FactorNode):
    def __init__(self, id, ):
        FactorNode.__init__(self, id, )


class Message():
    def __init__(self, m):
        assert isinstance(m, np.ndarray)
        self.m = m

    def __str__(self):
        return np.array_str(self.m)

    def renormalize(self):
        self.m = self.m / np.sum(self.m)

    @staticmethod
    def new_message(domain, init):
        m = np.ones((len(domain))) * init
        return Message(m)


class PotentialTable():
    def __init__(self, v_id2dim, table):
        assert isinstance(table, np.ndarray)
        if len(np.shape(table)) > 1:
            assert np.shape(table)[0] == np.shape(table)[1]
        self.var_id2dim = v_id2dim
        self.table = table


def pointwise_multiply(m1, m2):
    assert isinstance(m1, Message)
    assert isinstance(m2, Message) or m2 is None
    if m2 is None:
        return m1
    elif m1 is None:
        return m2
    else:
        new_m = np.multiply(m1.m, m2.m)
    return Message(new_m)


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
            pot = pots.get(fac_id, np.zeros((k,)))
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
        v = VariableNode(id=i, var_type=VAR_TYPE_PREDICTED, domain=[int(i + 1) for i in range(k)])
        variables_nodes.append(v)

    factor_nodes = []
    for fac_id, fac_potential in pots.iteritems():
        f = FactorNode(fac_id)
        factor_nodes.append(f)

    for f in factor_nodes[:n]:
        p = PotentialTable(v_id2dim={variables_nodes[f.id].id: 0}, table=pots[f.id])
        f.add_varset_with_potentials(varset=[variables_nodes[f.id]], ptable=p)
    for idx, f in enumerate(factor_nodes[n:-1]):
        p = PotentialTable(v_id2dim={variables_nodes[idx].id: 0, variables_nodes[idx + 1].id: 1}, table=pots[f.id])
        f.add_varset_with_potentials(varset=[variables_nodes[idx], variables_nodes[idx + 1]], ptable=p)

    f = factor_nodes[2 * n - 1]
    p = PotentialTable(v_id2dim={variables_nodes[0].id: 1, variables_nodes[n - 1].id: 0}, table=pots[f.id])
    f.add_varset_with_potentials(varset=[variables_nodes[0], variables_nodes[n - 1]], ptable=p)

    fg = FactorGraph()
    for fac in factor_nodes:
        fg.add_factor(fac)
    fg.initialize()
    fg.run_inference(10)
    for v in fg.variables:
        print 'Var:', v.id, v.get_marginal()
    print 'ok'
