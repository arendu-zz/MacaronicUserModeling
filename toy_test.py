__author__ = 'arenduchintala'
import itertools
import sys
# sys.path.insert(0, '/Users/arenduchintala/Projects2/MacaronicWebApp/python-scripts/')
from training_classes import TrainingInstance, Guess, SimpleNode
import json
import numpy as np
from scipy import sparse
from LBP import FactorNode, FactorGraph, VariableNode, VAR_TYPE_PREDICTED, PotentialTable, VAR_TYPE_GIVEN
import timeit
import codecs

np.seterr(all='raise')
np.set_printoptions(precision=4, suppress=True)


def find_guess(simplenode_id, guess_list):
    for cg in guess_list:
        if simplenode_id == cg.id:
            guess = cg
            return guess
    return None


def get_var_node_pair(sorted_current_sent, current_guesses, current_revealed, en_domain):
    var_node_pairs = []

    for idx, simplenode in enumerate(sorted_current_sent):
        if simplenode.lang == 'en':
            v = VariableNode(id=idx, var_type=VAR_TYPE_GIVEN, domain_type='en', domain=en_domain,
                             supervised_label=simplenode.l2_word)

        else:
            guess = find_guess(simplenode.id, current_guesses)
            if guess is None:
                guess = find_guess(simplenode.id, current_revealed)
                var_type = VAR_TYPE_GIVEN
            else:
                var_type = VAR_TYPE_PREDICTED
            assert guess is not None
            try:
                v = VariableNode(id=idx, var_type=var_type,
                                 domain_type='en',
                                 domain=en_domain,
                                 supervised_label=guess.guess)

            except AssertionError:
                print 'something bad...'
        var_node_pairs.append((v, simplenode))
    return var_node_pairs


def load_fg(fg, ti, en_domain, de2id, en2id):
    ordered_current_sent = sorted([(simplenode.position, simplenode) for simplenode in ti.current_sent])
    ordered_current_sent = [simplenode for position, simplenode in ordered_current_sent]
    var_node_pairs = get_var_node_pair(ordered_current_sent, ti.current_guesses, ti.current_revealed_guesses, en_domain)
    factors = []

    # create Ve x Vg factors
    for v, simplenode in var_node_pairs:
        if v.var_type == VAR_TYPE_PREDICTED:
            f = FactorNode(id=len(factors), factor_type='en_de', observed_domain_size=len(de_domain))
            o_idx = de2id[simplenode.l2_word]
            p = PotentialTable(v_id2dim={v.id: 0}, table=None, observed_dim=o_idx)
            f.add_varset_with_potentials(varset=[v], ptable=p)
            factors.append(f)
        elif v.var_type == VAR_TYPE_GIVEN:
            pass
        else:
            raise BaseException("vars are given or predicted only (no latent)")

    for idx_1, (v1, simplenode_1) in enumerate(var_node_pairs):
        for idx_2, (v2, simplenode_2) in enumerate(var_node_pairs[idx_1 + 1:]):
            if v1.var_type == VAR_TYPE_PREDICTED and v2.var_type == VAR_TYPE_PREDICTED:
                f = FactorNode(id=len(factors), factor_type='en_en')
                p = PotentialTable(v_id2dim={v1.id: 0, v2.id: 1}, table=None, observed_dim=None)
                f.add_varset_with_potentials(varset=[v1, v2], ptable=p)
                factors.append(f)
            elif v1.var_type == VAR_TYPE_GIVEN and v2.var_type == VAR_TYPE_GIVEN:
                pass
            else:
                v_given = v1 if v1.var_type == VAR_TYPE_GIVEN else v2
                simplenode_given = simplenode_1 if v1.var_type == VAR_TYPE_GIVEN else simplenode_2
                v_pred = v1 if v1.var_type == VAR_TYPE_PREDICTED else v2
                simplenode_pred = simplenode_1 if v1.var_type == VAR_TYPE_PREDICTED else simplenode_2
                f = FactorNode(id=len(factors),
                               factor_type='en_en',
                               observed_domain_type='en',
                               observed_domain_size=len(en_domain))
                try:
                    o_idx = en2id[v_given.supervised_label]  # either a users guess OR a revealed word
                except:
                    print 'wtf?'
                p = PotentialTable(v_id2dim={v_pred.id: 0}, table=None, observed_dim=o_idx)
                f.add_varset_with_potentials(varset=[v_pred], ptable=p)
                factors.append(f)
            pass
    for f in factors:
        fg.add_factor(f)

    for f in fg.factors:
        f.potential_table.make_potentials()
    return fg


if __name__ == '__main__':
    training_instances = codecs.open('mturk-data/ti.200').readlines()
    de_domain = [i.strip() for i in codecs.open('mturk-data/vocab.200.de', 'r', 'utf8').readlines()]
    en_domain = [i.strip() for i in codecs.open('mturk-data/vocab.200.en.lower', 'r', 'utf8').readlines()]
    en2id = dict((e, idx) for idx, e in enumerate(en_domain))
    de2id = dict((d, idx) for idx, d in enumerate(de_domain))
    print len(en_domain), len(de_domain)
    # en_domain = ['en_' + str(i) for i in range(500)]
    # de_domain = ['de_' + str(i) for i in range(100)]

    f_en_en = ['f1', 'f2']

    f_en_en_theta = np.ones((1, len(f_en_en)))
    phi_en_en = np.random.rand(len(en_domain) * len(en_domain), len(f_en_en))
    phi_en_en[phi_en_en > 0.5] = 1.0
    phi_en_en[phi_en_en < 1.0] = 0.0
    # pre_fire_en_en = sparse.csr_matrix(pre_fire_en_en)

    f_en_de = ['x', 'y']
    f_en_de_theta = np.ones((1, len(f_en_de)))
    phi_en_de = np.random.rand(len(en_domain) * len(de_domain), len(f_en_de))
    phi_en_de[phi_en_de > 0.5] = 1.0
    phi_en_de[phi_en_de < 0.2] = 0.0
    # pre_fire_en_de = sparse.csr_matrix(pre_fire_en_de)


    print timeit.timeit()
    for t_idx, training_instance in enumerate(training_instances):
        j_ti = json.loads(training_instance)
        ti = TrainingInstance.from_dict(j_ti)
        fg = FactorGraph(theta_en_en=f_en_en_theta,
                         theta_en_de=f_en_de_theta,
                         phi_en_en=phi_en_en,
                         phi_en_de=phi_en_de)

        fg = load_fg(fg, ti, en_domain, de2id=de2id, en2id=en2id)

        fg.initialize()
        print 'initialized', t_idx
        print fg.isLoopy
        # for v in fg.variables:
        # m = v.get_marginal()
        #    print np.reshape(m.m, (np.size(m.m),))
        fg.treelike_inference(3)
        grad_en_de, grad_en_en = fg.get_gradient()
        fg.theta_en_en += 0.05 * grad_en_en
        fg.theta_en_de += 0.05 * grad_en_de
        f_en_en_theta = fg.theta_en_en
        f_en_de_theta = fg.theta_en_de
        print f_en_de_theta
        print f_en_en_theta
