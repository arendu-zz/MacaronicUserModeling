__author__ = 'arenduchintala'
import numpy as np
from numpy import float32 as DTYPE
import pdb

global VAR_TYPE_LATENT, VAR_TYPE_PREDICTED, VAR_TYPE_GIVEN, UNARY_FACTOR, BINARY_FACTOR
VAR_TYPE_PREDICTED = 'var_type_predicted'
VAR_TYPE_GIVEN = 'var_type_given'
VAR_TYPE_LATENT = 'var_type_latent'
UNARY_FACTOR = 'unary_factor'
BINARY_FACTOR = 'binary_factor'


class FactorGraph():
    def __init__(self, theta_en_en, theta_en_de, phi_en_en, phi_en_de):
        self.theta_en_en = theta_en_en
        self.theta_en_de = theta_en_de
        self.phi_en_en = phi_en_en
        self.phi_en_de = phi_en_de
        self.variables = []
        self.factors = []
        self.messages = {}
        self.normalize_messages = True
        self.isLoopy = None
        self.regularization_param = 0.1

    def add_factor(self, fac):
        assert fac not in self.factors
        self.factors.append(fac)
        fac.graph = self
        for v in fac.varset:
            if v not in self.variables:
                self.variables.append(v)
                v.graph = self

    def get_message_schedule(self, root):
        assert isinstance(root, VariableNode)
        _schedule = []
        _seen = []
        _stack = [root]
        while len(_stack) > 0:
            _n = _stack.pop(0)
            if str(_n) not in _seen:
                _seen.append(str(_n))
                if isinstance(_n, VariableNode):
                    [_schedule.append((_fn, _n)) for _fn in _n.facset if str(_fn) not in _seen]
                    [_stack.append(_fn) for _fn in _n.facset if str(_fn) not in _seen]
                elif isinstance(_n, FactorNode):
                    [_schedule.append((_vn, _n)) for _vn in _n.varset if str(_vn) not in _seen]
                    [_stack.append(_vn) for _vn in _n.varset if str(_vn) not in _seen]
                else:
                    raise NotImplementedError("Only handles 2 kinds of nodes, variables and factors")
        return _schedule

    def has_loops(self):
        _seen = []
        _stack = [(self.variables[0], None)]
        while len(_stack) > 0:
            _n, _nparent = _stack.pop()
            if str(_n) in _seen:
                return True
            _seen.append(str(_n))
            if isinstance(_n, VariableNode):
                [_stack.append((_fn, _n)) for _fn in _n.facset if _fn != _nparent]
            elif isinstance(_n, FactorNode):
                [_stack.append((_vn, _n)) for _vn in _n.varset if _vn != _nparent]
            else:
                raise NotImplementedError("Only handles 2 kinds of nodes, variables and factors")
        return False

    def initialize(self):
        fs = sorted([(f.id, f) for f in self.factors])
        self.factors = [f for fid, f in fs]
        vs = sorted([(v.id, v) for v in self.variables])
        self.variables = [v for vid, v in vs]
        self.isLoopy = self.has_loops()

        for f in self.factors:
            assert len(f.potential_table.var_id2dim) == len(f.varset)
            vars = [self.variables[vid] for d, vid in
                    sorted([(d, v) for v, d in f.potential_table.var_id2dim.iteritems()])]
            if len(vars) == 2:
                assert np.shape(f.potential_table.table) == tuple([len(v.domain) for v in vars])
            else:
                assert np.shape(f.potential_table.table) == (len(vars[0].domain), 1)
                # todo: this check above is not comprehensive must check if dimsion of the potential table
            # todo: matches in the correct order of variables given by var_id2dim
            if len(f.varset) == 1:
                self.messages[str(f), str(f.varset[0])] = Message.new_message(f.varset[0].domain, 1.0)
            else:
                for v in f.varset:
                    self.messages[str(v), str(f)] = Message.new_message(v.domain, 1.0)
                    self.messages[str(f), str(v)] = Message.new_message(v.domain, 1.0)

    def treelike_inference(self, iterations):
        iterations = iterations if self.isLoopy else 1
        for _ in range(iterations):
            print 'iteration', _
            _root = self.variables[np.random.randint(0, len(self.variables))]
            _schedule = self.get_message_schedule(_root)
            print 'leaves to root...', str(_root)
            for frm, to in reversed(_schedule):
                if isinstance(to, FactorNode) and len(to.varset) < 2:
                    pass
                else:
                    # print 'send msg', str(frm), '->', str(to)
                    frm.update_message_to(to)
            print 'root to leaves...', str(_root)
            for to, frm in _schedule:
                if isinstance(to, FactorNode) and len(to.varset) < 2:
                    pass
                else:
                    # print 'send msg', str(frm), '->', str(to)
                    frm.update_message_to(to)

    def hw_inf(self, iterations):
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
                # print vi, fi, fii
                self.factors[fi - 1].update_message_to(self.variables[vi - 1])
                self.variables[vi - 1].update_message_to(self.factors[fii - 1])
                # print 'ok'

            for i in sorted(range(1, n + 1), reverse=True):
                fi = n + i
                vi = i
                fii = n + (i - 2) % n + 1 if i != 1 else 2 * n
                self.factors[fi - 1].update_message_to(self.variables[vi - 1])
                self.variables[vi - 1].update_message_to(self.factors[fii - 1])

    def get_gradient(self):
        grad_en_de = np.zeros_like(self.theta_en_de)
        grad_en_en = np.zeros_like(self.theta_en_en)
        for f in self.factors:
            if f.factor_type == 'en_en':
                grad_en_en += f.get_gradient()
            elif f.factor_type == 'en_de':
                grad_en_de += f.get_gradient()
            else:
                raise BaseException('only 2 kinds of factors allowed...')
        reg_en_de = self.regularization_param * self.theta_en_de
        reg_en_en = self.regularization_param * self.theta_en_en
        grad_en_en -= reg_en_en
        grad_en_de -= reg_en_de
        return grad_en_de, grad_en_en


class VariableNode():
    def __init__(self, id, var_type, domain_type, domain, supervised_label):
        assert isinstance(id, int)
        assert supervised_label in domain
        self.id = id
        self.var_type = var_type
        self.domain = domain
        self.facset = []  # set of neighboring factors
        self.graph = None
        self.supervised_label = supervised_label
        self.supervised_label_index = self.domain.index(supervised_label)
        self.domain_type = domain_type

    def __str__(self):
        return "X_" + str(self.id)

    def __eq__(self, other):
        return isinstance(other, VariableNode) and self.id == other.id

    def display(self, m):
        isinstance(m, Message)
        raise NotImplementedError()

    def add_factor(self, fc):
        assert isinstance(fc, FactorNode)
        self.facset.append(fc)

    def init_message_to(self, fc, init_m):
        assert isinstance(fc, FactorNode)
        assert isinstance(init_m, Message)
        # todo: should we prune initial messages?
        self.messages[self.id, fc.id] = init_m

    def update_message_to(self, fc):
        assert isinstance(fc, FactorNode)
        assert fc in self.facset
        new_m = Message.new_message(self.domain, 1.0)
        for other_fc in self.facset:
            if other_fc is not fc:
                m = self.graph.messages[str(other_fc), str(self)]  # from other factors to this variable
                new_m = pointwise_multiply(m, new_m)
                assert np.shape(new_m.m) == np.shape(m.m)
        if self.graph.normalize_messages:
            new_m.renormalize()
        self.graph.messages[str(self), str(fc)] = new_m

    def get_marginal(self):
        new_m = Message.new_message(self.domain, 1.0)
        for fc in self.facset:
            m = self.graph.messages[str(fc), str(self)]  # incoming message
            new_m = pointwise_multiply(m, new_m)
        new_m.renormalize()
        return new_m


class FactorNode():
    def __init__(self, id, factor_type=None, observed_domain_type=None, observed_value=None, observed_domain_size=None):
        assert isinstance(id, int)
        self.id = id
        self.varset = []  # set of neighboring variables
        self.potential_table = None
        self.factor_type = factor_type
        self.graph = None
        self.observed_domain_type = observed_domain_type
        self.observed_value = observed_value
        self.observed_domain_size = observed_domain_size

    def __str__(self):
        return 'F_' + str(self.id)

    def __eq__(self, other):
        return isinstance(other, FactorNode) and self.id == other.id

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
        ptable.add_factor(self)
        self.potential_table = ptable

    def get_theta(self):
        if self.factor_type == 'en_en':
            return self.graph.theta_en_en
        elif self.factor_type == 'en_de':
            return self.graph.theta_en_de
        else:
            raise BaseException("only 2 factor types are supported right now..")

    def get_phi(self):
        if self.factor_type == 'en_en':
            return self.graph.phi_en_en
        elif self.factor_type == 'en_de':
            return self.graph.phi_en_de
        else:
            raise BaseException("only 2 feature value types are supported right now..")

    def get_shape(self):
        if len(self.varset) == 1:
            return len(self.varset[0].domain), self.observed_domain_size
        elif len(self.varset) == 2:
            return len(self.varset[0].domain), len(self.varset[1].domain)
        else:
            raise BaseException("only unary or binary factors are supported...")

    def update_message_to(self, var):
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
                mul = np.multiply(msg.m.T, self.potential_table.table)
                marginalized = np.sum(mul, 1)
            else:
                mul = np.multiply(msg.m.T, self.potential_table.table.T).T
                marginalized = np.sum(mul, 0)
            new_m = Message(marginalized)
            if self.graph.normalize_messages:
                new_m.renormalize()
            assert np.shape(new_m.m) == np.shape(self.graph.messages[str(self), str(var)].m)
            self.graph.messages[str(self), str(var)] = new_m

    def get_marginal(self):
        # takes the last messages coming into this factor
        # normalizes the messages
        # multiplies it into a matrix with same dim as the potnetials
        r = None
        c = None
        if len(self.varset) == 1:
            m = Message.new_message(self.varset[0].domain, 1.0)
            m.renormalize()
            marginal = m.m
        else:
            for v in self.varset:
                vd = self.potential_table.var_id2dim[v.id]
                m = self.graph.messages[str(v), str(self)]
                m.renormalize()
                if vd == 0:
                    c = np.reshape(m.m, (np.size(m.m), 1))  # col vector
                elif vd == 1:
                    r = np.reshape(m.m, (1, np.size(m.m)))  # row vector
                else:
                    raise NotImplementedError("only supports pairwise factors..")
            marginal = np.dot(c, r)
        marginal = np.multiply(marginal, self.potential_table.table)
        marginal = marginal / np.sum(marginal)
        return marginal

    def get_observed_factor(self):
        of = np.zeros_like(self.potential_table.table)
        cell = sorted([(self.potential_table.var_id2dim[v.id], v.supervised_label_index) for v in self.varset])
        cell = tuple([o for d, o in cell])
        of[cell] = 1.0
        return of

    def get_on_the_fly_feature_values(self):
        raise NotImplementedError("the matrix with feature values that are computed on the fly goes here...")

    def get_gradient(self):
        g = self.cell_gradient()
        if self.factor_type == 'en_de':
            sparse_g = np.zeros(self.get_shape())
            g = np.reshape(g, (np.size(g),))
            sparse_g[:, self.potential_table.observed_dim] = g
            g = sparse_g
        else:
            # g is already  a matrix
            pass
        f_ij = np.reshape(g, (np.size(g), 1))
        grad1 = f_ij.T.dot(self.get_phi())
        '''on_the_fly_feature_values = self.get_on_the_fly_feature_values()
        grad2 = on_the_fly_feature_values * f_ij
        fg = np.concatenate((grad1, grad2), axis=1)'''
        return grad1

    def cell_gradient(self):
        obs = self.get_observed_factor()
        marginal = self.get_marginal()  # this is the same because I assume binary features  values at the "cell level"
        exp = marginal
        g = obs - exp
        return g


class ObservedFactor(FactorNode):
    def __init__(self, id, observed_domain_type, observed_value):
        FactorNode.__init__(self, id, factor_type=UNARY_FACTOR)
        self.observed_domain_type = observed_domain_type
        self.observed_value = observed_value


class Message():
    def __init__(self, m):
        assert isinstance(m, np.ndarray)
        if np.shape(m) != (np.size(m), 1):
            self.m = np.reshape(m, (np.size(m), 1))
        else:
            self.m = m

    def __str__(self):
        return np.array_str(self.m)

    def renormalize(self):
        self.m = self.m / np.sum(self.m)

    @staticmethod
    def new_message(domain, init):
        m = np.ones((len(domain), 1)) * init
        m.astype(DTYPE)
        return Message(m)


class PotentialTable():
    def __init__(self, v_id2dim, table=None, observed_dim=None):
        self.factor = None
        self.observed_dim = observed_dim
        self.var_id2dim = v_id2dim

        if table is not None:
            assert isinstance(table, np.ndarray)
            if observed_dim is not None:
                assert len(v_id2dim) == 1
                if v_id2dim[v_id2dim.keys()[0]] == 0:
                    self.table = np.reshape(table[:, observed_dim], (np.shape(table)[0], 1))
                    self.var_id2dim = v_id2dim
                else:
                    raise NotImplementedError("a unary factor should always be a column vector")
            else:
                self.var_id2dim = v_id2dim
                self.table = table
            self.table.astype(DTYPE)
            if len(np.shape(self.table)) > 1:
                assert np.shape(self.table)[0] == np.shape(self.table)[1] or np.shape(self.table)[1] == 1
        else:
            pass

    def make_potentials(self):
        # table = np.exp(np.multiply(self.factor.get_theta(), self.factor.get_phi()))
        # table = np.sum(table, 1)
        table = self.factor.get_phi().dot(self.factor.get_theta().T)
        table = np.exp(table)
        table_shape = self.factor.get_shape()
        table = np.reshape(table, table_shape)
        if self.observed_dim is not None:
            table = np.reshape(table[:, self.observed_dim], (np.shape(table)[0], 1))
        else:
            pass
        self.table = table
        self.table.astype(DTYPE)
        if len(np.shape(self.table)) > 1:
            assert np.shape(self.table)[0] == np.shape(self.table)[1] or np.shape(self.table)[1] == 1

    def add_factor(self, factor):
        assert isinstance(factor, FactorNode)
        assert self.factor is None
        self.factor = factor


def pointwise_multiply(m1, m2):
    assert isinstance(m1, Message)
    assert isinstance(m2, Message)
    if m2 is None:
        return m1
    elif m1 is None:
        return m2
    else:
        assert np.shape(m1.m) == np.shape(m2.m)
        new_m = np.multiply(m1.m, m2.m)
    return Message(new_m)
