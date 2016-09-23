__author__ = 'arenduchintala'
import sys
import numpy as np
from numpy import float64 as DTYPE
import random
import c_array_utils as au
from scipy import sparse
import pdb
import time

global VAR_TYPE_LATENT, VAR_TYPE_PREDICTED, VAR_TYPE_GIVEN, UNARY_FACTOR, BINARY_FACTOR
VAR_TYPE_PREDICTED = 'var_type_predicted'
VAR_TYPE_GIVEN = 'var_type_given'
VAR_TYPE_LATENT = 'var_type_latent'
UNARY_FACTOR = 'unary_factor'
BINARY_FACTOR = 'binary_factor'


class FactorGraph():
    def __init__(self,
                 theta_en_en_names,
                 theta_en_de_names,
                 theta_en_en,
                 theta_en_de,
                 phi_en_en,
                 phi_en_de,
                 phi_en_en_w1):
        self.theta_en_en = theta_en_en
        self.theta_en_de = theta_en_de
        self.theta_en_en_names = theta_en_en_names,
        self.theta_en_de_names = theta_en_de_names,
        self.phi_en_en = phi_en_en
        self.phi_en_en_w1 = phi_en_en_w1
        self.phi_en_de = phi_en_de
        self.pot_en_en = None
        self.pot_en_en_w1 = None
        self.pot_en_en_clamp = None
        self.pot_en_de = None
        self.variables = {}
        self.factors = []
        self.messages = {}
        self.normalize_messages = True
        self.isLoopy = None
        self.regularization_param = 0.01
        self.learning_rate = 0.1
        self.report_times = False
        self.cg_times = []
        self.it_times = []
        self.gg_times = []
        self.sgg_times = []
        self.active_domains = {}
        self.use_approx_beliefs = False
        self.use_approx_gradient = False
        if isinstance(self.theta_en_en_names, tuple):
            self.theta_en_en_names = self.theta_en_en_names[0]
        if isinstance(self.theta_en_de_names, tuple):
            self.theta_en_de_names = self.theta_en_de_names[0]

    def display_timing_info(self):
        if self.report_times and len(self.variables) > 5:
            #self.cg_times = []
            #self.exp_cg_times = []
            #self.obs_cg_times = []
            #self.diff_cg_times = []
            #self.gg_times = []
            #self.sgg_times = []
            print '\ncgtimes    :', np.sum(self.cg_times) / len(self.cg_times), 'total', np.sum(self.cg_times), 'len', len(self.cg_times)
            print 'ggtimes    :', np.sum(self.gg_times) / len(self.gg_times), 'total', np.sum(self.gg_times), 'len', len(self.gg_times)
            print 'sggtimes    :', np.sum(self.sgg_times) / len(self.sgg_times), 'total', np.sum(self.sgg_times), 'len', len(self.sgg_times)
            print 'it_times    :', np.sum(self.it_times) / len(self.it_times), 'total', np.sum(self.it_times), 'len', len(self.it_times)
            print 'num vars   :', len(self.variables)
        else:
            pass
        return True


    def get_precision_counts(self):
        p_at_0 = 0
        p_at_25 = 0
        p_at_50 = 0
        totals = 0
        for f in self.factors:
            if f.factor_type == 'en_de':
                sl, slp, prediction = f.varset[0].get_max_vocab(50)
                totals +=1
                for rank, (p_label, p_prob) in enumerate(prediction):
                    if sl == p_label:
                        if rank == 0:
                            p_at_0 +=1
                            p_at_25 +=1
                            p_at_50 +=1
                        elif rank < 26:
                            p_at_25 +=1
                            p_at_50 +=1
                        elif rank < 51:
                            p_at_50 += 1
                        else:
                            pass
                    else:
                        pass
            else:
                pass
        return p_at_0, p_at_25, p_at_50, totals


    def to_string(self):
        position_factors = sorted([(f.position, f) for f in self.factors if f.position is not None])
        fg_dct = {}
        for p, f in position_factors:
            if f.factor_type == 'en_de':
                de_label = f.word_label
                sl, slp, pred = f.varset[0].get_max_vocab(50)
                
                pred = ' '.join([p1 + ' ' + p2 for p1, p2 in pred])
                fg_dct[p] = ' '.join([de_label, sl, slp, pred])
            if f.factor_type == 'en_en':
                guess_label = f.word_label
                fg_dct[p] = ' '.join(['', guess_label, ''])
        fg_strings = [fg_dct[k] for k in sorted(fg_dct)]
        return fg_strings

    def to_dist(self):
        factor_dist = []
        position_factors = sorted([(f.position, f) for f in self.factors if f.position is not None])
        for p, f in position_factors:
            if f.factor_type == 'en_de':
                v = f.varset[0]
                truth = v.truth_label if v.truth_label is not None else 'None'
                guess = v.supervised_label if v.supervised_label is not None else 'None'
                m = v.get_marginal()
                #print 'here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                #print m.m
                #print np.sum(m.m)
                #print m.m.shape
                #i = ' '.join(['%0.4f' % i for i in np.log(m.m)])
                i = ' '.join(['%0.6f' % i for i in np.log(m.m)])
                d = ' ||| '.join([truth, guess, i])
                factor_dist.append(d)
        factor_dist = '\n'.join(factor_dist)
        return factor_dist

    def add_factor(self, fac):
        if __debug__: assert fac not in self.factors
        self.factors.append(fac)
        fac.graph = self
        for v in fac.varset:
            if v.id not in self.variables:
                # self.variables.append(v)
                self.variables[v.id] = v
                v.graph = self

    def get_message_schedule(self, root):
        if __debug__: assert isinstance(root, VariableNode)
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
        _rand_key = random.sample(self.variables.keys(), 1)[0]
        _root = self.variables[_rand_key]
        _stack = [(_root, None)]
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
        if __debug__: assert len(self.variables) > 0
        if __debug__: assert len(self.factors) > 0
        fs = sorted([(f.id, f) for f in self.factors])
        self.factors = [f for fid, f in fs]
        vs = sorted([(v.id, v) for v in self.variables.values()])
        # self.variables = [v for vid, v in vs]
        self.isLoopy = self.has_loops()

        for f in self.factors:
            if __debug__: assert len(f.potential_table.var_id2dim) == len(f.varset)
            vars = [self.variables[vid] for d, vid in
                    sorted([(d, v) for v, d in f.potential_table.var_id2dim.iteritems()])]
            if len(vars) == 2:
                if __debug__: assert np.shape(f.potential_table.table) == tuple([len(v.domain) for v in vars])
            else:
                if __debug__: assert np.shape(f.potential_table.table) == (len(vars[0].domain), 1)
                # todo: this check above is not comprehensive must check if dimsion of the potential table
            # todo: matches in the correct order of variables given by var_id2dim
            if len(f.varset) == 1:
                self.messages[str(f), str(f.varset[0])] = Message.new_message(f.varset[0].domain,
                                                                              1.0 / len(f.varset[0].domain))
            else:
                for v in f.varset:
                    self.messages[str(v), str(f)] = Message.new_message(v.domain, 1.0 / float(len(v.domain)))
                    self.messages[str(f), str(v)] = Message.new_message(v.domain, 1.0 / len(v.domain))

    def treelike_inference(self, iterations):
        iterations = iterations if self.isLoopy else 1
        for _ in range(iterations):
            if self.report_times: it = time.time()
            # print 'inference iterations', _
            _rand_key = random.sample(self.variables.keys(), 1)[0]
            _root = self.variables[_rand_key]
            _schedule = self.get_message_schedule(_root)
            # print 'leaves to root...', str(_root)
            for frm, to in reversed(_schedule):
                if isinstance(to, FactorNode) and len(to.varset) < 2:
                    pass
                else:
                    # print 'send msg', str(frm), '->', str(to)
                    #sys.stderr.write('0-')
                    frm.update_message_to(to)
                    #sys.stderr.write('-0')
            # print 'root to leaves...', str(_root)
            for to, frm in _schedule:
                if isinstance(to, FactorNode) and len(to.varset) < 2:
                    pass
                else:
                    # print 'send msg', str(frm), '->', str(to)
                    #sys.stderr.write('1-')
                    frm.update_message_to(to)
                    #sys.stderr.write('-1')
            if self.report_times: self.it_times.append(time.time() - it)
        return True

    def get_posterior_probs(self):
        log_posterior = 0.0
        for v_key, v in self.variables.iteritems():
            m = v.get_marginal()
            _l = np.log(m.m[v.supervised_label_index])
            if _l == float('-inf'):
                sys.stderr.write('err -inf' + str( m.m[v.supervised_label_index]))
                log_posterior += -99.99
            else:
                log_posterior += np.sum(_l)
        return log_posterior

    def get_max_postior_label(self, top=10):
        label_guesses = []
        for v_key, v in self.variables.iteritems():
            s, sp, g = v.get_max_vocab(top)
            g_str = ' '.join([i + ' ' + p for i, p in g])
            label_guesses.append(s + ' ' + sp + ' ' + g_str)
        return label_guesses

    def hw_inf(self, iterations):
        raise BaseException("This method assumes self.variables is a list.. depricated...")
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
        grad_en_en, grad_en_de = self.get_unregularized_gradeint()
        reg_en_de = self.regularization_param * self.theta_en_de
        reg_en_en = self.regularization_param * self.theta_en_en
        grad_en_en -= reg_en_en
        grad_en_de -= reg_en_de
        return grad_en_de, grad_en_en

    def get_unregularized_gradeint(self):
        grad_en_de = np.zeros_like(self.theta_en_de, dtype=DTYPE)
        grad_en_en = np.zeros_like(self.theta_en_en, dtype=DTYPE)
        for f in self.factors:
            if f.factor_type == 'en_en':
                gr = f.get_gradient()
                if f.gap == 1:
                    idx = self.theta_en_en_names.index('pmi_w1')
                    grad_en_en[0, idx] += gr[0, 0]
                elif f.gap is None or f.gap > 1:
                    idx = self.theta_en_en_names.index('pmi')
                    grad_en_en[0, idx] += gr[0, 0]
                else:
                    raise BaseException('unknown feature name')
            elif f.factor_type == 'en_de':
                grad_en_de += f.get_gradient()
            else:
                raise BaseException('only 2 kinds of factors allowed...')
        return grad_en_en, grad_en_de

    def return_gradient(self):
        # adds learning rate to regularized gradient
        grad_en_de, grad_en_en = self.get_gradient()
        g_en_en = self.learning_rate * grad_en_en
        g_en_de = self.learning_rate * grad_en_de
        return g_en_en, g_en_de

    def update_theta(self):
        grad_en_de, grad_en_en = self.get_gradient()
        self.theta_en_en += (self.learning_rate * grad_en_en)
        self.theta_en_de += (self.learning_rate * grad_en_de)
        return self.theta_en_en, self.theta_en_de


class VariableNode():
    def __init__(self, id, var_type, domain_type, domain, supervised_label):
        if not isinstance(id, int):
            print 'id ' , id, 'not an int'
        if supervised_label not in domain:
            print supervised_label, 'not in' , domain
            exit(-1)
        self.id = id
        self.var_type = var_type
        self.domain = domain
        self.facset = []  # set of neighboring factors
        self.graph = None
        self.supervised_label = supervised_label
        self.supervised_label_index = self.domain.index(supervised_label)
        self.domain_type = domain_type
        self.truth_label = None
        self.truth_label_index = None

    def set_truth_label(self, tl):
        self.truth_label = tl

    def __str__(self):
        return "X_" + str(self.id)

    def __eq__(self, other):
        return isinstance(other, VariableNode) and self.id == other.id

    def display(self, m):
        isinstance(m, Message)
        raise NotImplementedError()

    def add_factor(self, fc):
        if __debug__: assert isinstance(fc, FactorNode)
        self.facset.append(fc)

    def init_message_to(self, fc, init_m):
        if __debug__: assert isinstance(fc, FactorNode)
        if __debug__: assert isinstance(init_m, Message)
        # todo: should we prune initial messages?
        self.messages[self.id, fc.id] = init_m

    def update_message_to(self, fc):
        if __debug__: assert isinstance(fc, FactorNode)
        if __debug__: assert fc in self.facset
        #sys.stderr.write('~')
        new_m = Message.new_message(self.domain, 1.0 / len(self.domain))
        for other_fc in self.facset:
            if other_fc is not fc:
                m = self.graph.messages[str(other_fc), str(self)]  # from other factors to this variable
                new_m = pointwise_multiply(m, new_m)
                if __debug__: assert np.shape(new_m.m) == np.shape(m.m)
        if self.graph.normalize_messages:
            new_m.renormalize()
        self.graph.messages[str(self), str(fc)] = new_m

    def get_marginal(self):
        new_m = Message.new_message(self.domain, 1.0 / len(self.domain))
        for fc in self.facset:
            m = self.graph.messages[str(fc), str(self)]  # incoming message
            new_m = pointwise_multiply(m, new_m)
        if self.graph.normalize_messages:
            new_m.renormalize()
        return new_m

    def get_max_vocab(self, top):
        m = self.get_marginal()
        a = np.reshape(m.m, (np.size(m.m, )))
        max_idx = np.argpartition(a, -top)[-top:]
        max_idx = max_idx[np.argsort(a[max_idx])]
        max_vocab = [(self.domain[i], '%0.4f' % np.log(a[i])) for i in max_idx]
        max_vocab.reverse()
        return self.supervised_label, '%0.4f' % np.log(m.m[self.supervised_label_index]), max_vocab


class FactorNode():
    def __init__(self, id, factor_type=None, observed_domain_type=None, observed_value=None, observed_domain_size=None):
        if __debug__: assert isinstance(id, int)
        self.id = id
        self.varset = []  # set of neighboring variables
        self.potential_table = None
        self.factor_type = factor_type
        self.graph = None
        self.observed_domain_type = observed_domain_type
        self.observed_value = observed_value
        self.observed_domain_size = observed_domain_size
        self.position = None
        self.word_label = None
        self.gap = None
        self.connect_type = None

    def __str__(self):
        return 'F_' + str(self.id)

    def __eq__(self, other):
        return isinstance(other, FactorNode) and self.id == other.id

    def init_message_to(self, var, init_m):
        if __debug__: assert isinstance(var, VariableNode)
        if __debug__: assert isinstance(init_m, Message)
        self.graph.messages[str(self), str(var)] = init_m

    def add_varset_with_potentials(self, varset, ptable):
        if __debug__: assert isinstance(ptable, PotentialTable)
        if len(varset) == 2:
            if __debug__: assert varset[0] != varset[1]
        if __debug__: assert len(varset) == len(ptable.var_id2dim)
        if len(varset) > 2:
            raise NotImplementedError("Currently supporting unary and pairwise factors...")

        for v in varset:
            if __debug__: assert v not in self.varset
            self.varset.append(v)
            v.add_factor(self)
        ptable.add_factor(self)
        self.potential_table = ptable

    def get_pot(self):
        if self.factor_type == 'en_en':
            if self.gap > 1:
                return self.graph.pot_en_en
            elif self.gap == 1:
                return self.graph.pot_en_en_w1
            else:
                raise BaseException("only 2 kinds of distances are supported ...")
        elif self.factor_type == 'en_de':
            return self.graph.pot_en_de
        else:
            raise BaseException("only two kinds of potentials are supported...")

    '''
    def get_theta(self):
        if self.factor_type == 'en_en':
            return self.graph.theta_en_en
        elif self.factor_type == 'en_de':
            return self.graph.theta_en_de
        else:
            raise BaseException("only 2 factor types are supported right now..")
    '''

    def get_phi(self):
        if self.factor_type == 'en_en':
            if self.gap > 1:
                return self.graph.phi_en_en
            elif self.gap == 1:
                return self.graph.phi_en_en_w1
            else:
                raise BaseException("only 2 distances supported at the moment")
        elif self.factor_type == 'en_de':
            return self.graph.phi_en_de
        else:
            raise BaseException("only 2 feature value types are supported right now..")

    '''def get_phi_csc(self):
        if self.factor_type == 'en_en':
            return self.graph.phi_en_en_csc
        elif self.factor_type == 'en_de':
            return self.graph.phi_en_de_csc
        else:
            raise BaseException("only 2 feature value types are supported right now..")
    '''

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
            new_m = Message(new_m)
            if self.graph.normalize_messages:
                new_m.renormalize()
            self.graph.messages[str(self), str(var)] = new_m
        else:
            o_var = other_vars[0]
            o_var_dim = self.potential_table.var_id2dim[o_var.id]
            msg = self.graph.messages[str(o_var), str(self)]
            if o_var_dim == 1:
                # mul = np.multiply(msg.m.T, self.potential_table.table)
                # marginalized = np.sum(mul, 1)
                test_marginalized = np.dot(self.potential_table.table, msg.m)
                # test_marginalized = np.reshape(test_marginalized, np.shape(marginalized))
                # if __debug__: assert  np.allclose(test_marginalized, marginalized)
            else:
                # mul = np.multiply(msg.m.T, self.potential_table.table.T).T
                # marginalized = np.sum(mul, 0)
                test_marginalized = np.dot(msg.m.T, self.potential_table.table)
                # test_marginalized = np.reshape(test_marginalized, np.shape(marginalized))
                # if __debug__: assert  np.allclose(test_marginalized, marginalized)

            new_m = Message(test_marginalized)
            if self.graph.normalize_messages:
                new_m.renormalize()
            if __debug__: assert np.shape(new_m.m) == np.shape(self.graph.messages[str(self), str(var)].m)
            self.graph.messages[str(self), str(var)] = new_m

    def get_factor_beliefs(self):
        # takes the last messages coming into this factor
        # normalizes the messages
        # multiplies it into a matrix with same dim as the potentials
        r = None
        c = None
        if len(self.varset) == 1:
            #m = Message.new_message(self.varset[0].domain, 1.0 / len(self.varset[0].domain))
            #marginals = m.m
            #assert marginals.shape == self.potential_table.table.shape
            #beliefs = np.multiply(marginals, self.potential_table.table)  # O(n)
            beliefs = au.normalize(self.potential_table.table)  # todo: make this faster?
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
            if self.graph.use_approx_beliefs:
                #sys.stderr.write('+')
                # approx_marginals = au.make_sparse_and_dot(c, r)
                approx_marginals, c_idx, r_idx = au.sparse_dot(c, r) 
                # approx_beliefs_mat, _ = au.sparse_multiply_and_normalize(approx_marginals, self.potential_table.table)
                # beliefs = approx_beliefs_mat
                beliefs = au.sparse_pointwise_multiply(approx_marginals, c_idx, r_idx, self.potential_table.table)
                # beliefs = np.multiply(approx_marginals, self.potential_table.table)  # O(n)
                # beliefs = au.normalize(beliefs)
                beliefs = au.sparse_normalize(beliefs, c_idx, r_idx)
            else:
                #sys.stderr.write('-')
                marginals = au.dd_matrix_multiply(c, r)  # np.dot(c, r)
                #assert marginals.shape == self.potential_table.table.shape
                beliefs = np.multiply(marginals, self.potential_table.table)  # O(n)
                beliefs = au.normalize(beliefs)
                pass
                # if c[10, 0] != c[11, 0]:
                #    print 'c not uniform...'
        return beliefs

    def get_observed_factor_as_array(self):
        # of = np.zeros_like(self.potential_table.table)
        cell = sorted([(self.potential_table.var_id2dim[v.id], v.supervised_label_index) for v in self.varset])
        cell = tuple([o for d, o in cell])
        # print 'cell', cell, self.potential_table.table.shape
        # of[cell] = 1.0
        return [cell]

    def get_observed_factor(self):
        of = np.zeros_like(self.potential_table.table, dtype=DTYPE)
        cell = sorted([(self.potential_table.var_id2dim[v.id], v.supervised_label_index) for v in self.varset])
        cell = tuple([o for d, o in cell])
        of[cell] = 1.0
        return of


    def get_gradient(self):
        if self.graph.report_times: cg = time.time()
        g = self.cell_gradient()
        #g = self.cell_gradient_alt()
        #assert  np.allclose(g, g_alt)
        if self.graph.report_times: self.graph.cg_times.append(time.time() - cg)
        if self.graph.report_times: sgg = time.time()
        if self.potential_table.observed_dim is not None:
            sparse_g = np.zeros(self.get_shape(), dtype=DTYPE)
            g = np.reshape(g, (np.size(g),))
            sparse_g[:, self.potential_table.observed_dim] = g
            g = sparse_g
        else:
            # g is already  a matrix
            pass
        if self.graph.report_times: self.graph.sgg_times.append(time.time() - sgg)
        f_ij = np.reshape(g, (np.size(g), 1))
        if self.graph.report_times: gg = time.time()
        # print 'nz appx, orig, full :', np.count_nonzero(f_ij_approx), np.count_nonzero(f_ij), np.size(f_ij)
        if self.graph.use_approx_gradient:
            grad_approx = au.induce_s_mutliply_clip(f_ij, self.get_phi().T)
            grad1 = grad_approx.T
        else:
            grad1 = (self.get_phi().T.dot(f_ij)).T
        # if __debug__: assert  np.allclose(grad1, grad2.T)
        if self.graph.report_times: self.graph.gg_times.append(time.time() - gg)
        return grad1

    def cell_gradient(self):
        obs_counts = self.get_observed_factor()
        exp_counts = self.get_factor_beliefs()  # this is the same because I assume binary features  values at the "cell level"
        cell_g = obs_counts - exp_counts
        return cell_g

    def cell_gradient_alt(self):
        cell = self.get_observed_factor_as_array()
        exp_counts = self.get_factor_beliefs()  # this is the same because I assume binary features  values at the "cell level"
        d = -exp_counts
        assert len(cell) == 1
        for c in cell:
            d[c] = 1.0 +  d[c]
        return d

class ObservedFactor(FactorNode):
    def __init__(self, id, observed_domain_type, observed_value):
        FactorNode.__init__(self, id, factor_type=UNARY_FACTOR)
        self.observed_domain_type = observed_domain_type
        self.observed_value = observed_value


class Message():
    def __init__(self, m):
        if __debug__: assert isinstance(m, np.ndarray)
        if __debug__: assert np.size(m[m < 0.0]) == 0
        if np.shape(m) != (np.size(m), 1):
            self.m = np.reshape(m, (np.size(m), 1))
        else:
            self.m = m

    def __str__(self):
        return np.array_str(self.m)

    def renormalize(self):
        s = np.sum(self.m)
        if s > 0:
            # self.m = self.m / np.sum(self.m)
            self.m = au.normalize(self.m)
        else:
            e = np.empty_like(self.m)
            e.fill(1.0 / np.size(self.m))
            self.m = e
        if __debug__: assert np.size(self.m[self.m < 0.0]) == 0
        if __debug__: assert np.abs(np.sum(self.m) - 1.0) < 1e-10

    @staticmethod
    def new_message(domain, init):
        m = np.empty((len(domain), 1))
        m.fill(init)
        if m.dtype != DTYPE:
            m = m.astype(DTYPE)
        return Message(m)


class PotentialTable():
    def __init__(self, v_id2dim, table=None, observed_dim=None):
        self.factor = None
        self.observed_dim = observed_dim
        self.var_id2dim = v_id2dim

        if table is not None:
            if __debug__: assert isinstance(table, np.ndarray)
            if observed_dim is not None:
                if __debug__: assert len(v_id2dim) == 1
                if v_id2dim[v_id2dim.keys()[0]] == 0:
                    self.table = np.reshape(table[:, observed_dim], (np.shape(table)[0], 1))
                    self.var_id2dim = v_id2dim
                else:
                    raise NotImplementedError("a unary factor should always be a column vector")
            else:
                self.var_id2dim = v_id2dim
                self.table = table
            if self.table.dtype != DTYPE:
                self.table = self.table.astype(DTYPE)
            if len(np.shape(self.table)) > 1:
                if __debug__: assert np.shape(self.table)[0] == np.shape(self.table)[1] or np.shape(self.table)[1] == 1
        else:
            pass

    def slice_potentials(self):
        table = self.factor.get_pot()
        # if __debug__: assert  np.allclose(table_from_pot, table)
        table_shape = self.factor.get_shape()
        table = np.reshape(table, table_shape)

        if self.observed_dim is not None:
            table = np.reshape(table[:, self.observed_dim], (np.shape(table)[0], 1))
        else:
            pass
        self.table = table
        if self.table.dtype != DTYPE:
            self.table = self.table.astype(DTYPE)
        if len(np.shape(self.table)) > 1:
            if __debug__: assert np.shape(self.table)[0] == np.shape(self.table)[1] or np.shape(self.table)[1] == 1

    def add_factor(self, factor):
        if __debug__: assert isinstance(factor, FactorNode)
        if __debug__: assert self.factor is None
        self.factor = factor


def pointwise_multiply(m1, m2):
    if __debug__: assert isinstance(m1, Message)
    if __debug__: assert isinstance(m2, Message)
    if m2 is None:
        return m1
    elif m1 is None:
        return m2
    else:
        # print 'tt:', np.sum(m1.m), np.sum(m2.m)
        if __debug__: assert np.shape(m1.m) == np.shape(m2.m)
        # if __debug__: assert  np.abs(np.sum(m1.m) - 1) < 1e-5
        new_m = au.pointwise_multiply(m1.m, m2.m)
    return Message(new_m)
