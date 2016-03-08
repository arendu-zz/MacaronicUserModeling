__author__ = 'arenduchintala'
import pdb
import random
from training_classes import TrainingInstance
import json
import numpy as np
import sys
from optparse import OptionParser
from LBP import FactorNode, FactorGraph, VariableNode, VAR_TYPE_PREDICTED, PotentialTable, VAR_TYPE_GIVEN
import time
import codecs
from numpy import float32 as DTYPE
from scipy import sparse
from multiprocessing import Pool

global f_en_en_theta, f_en_de_theta
np.seterr(divide='raise', over='raise', under='ignore')

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


def create_factor_graph(ti, learning_rate, theta_en_en, theta_en_de, phi_en_en, phi_en_de, en_domain, de2id, en2id):
    ordered_current_sent = sorted([(simplenode.position, simplenode) for simplenode in ti.current_sent])
    ordered_current_sent = [simplenode for position, simplenode in ordered_current_sent]
    var_node_pairs = get_var_node_pair(ordered_current_sent, ti.current_guesses, ti.current_revealed_guesses, en_domain)
    factors = []

    len_en_domain = len(en_domain)
    len_de_domain = len(de_domain)
    fg = FactorGraph(theta_en_en=theta_en_en,
                     theta_en_de=theta_en_de,
                     phi_en_en=phi_en_en,
                     phi_en_de=phi_en_de)
    fg.learning_rate = learning_rate

    history_feature = np.zeros((len_en_domain, len_de_domain))
    history_feature.astype(DTYPE)
    for pg in ti.past_correct_guesses:
        i = en2id[pg.guess]
        j = de2id[pg.l2_word]
        history_feature[i, :] += 1.0
        history_feature[:, j] += 1.0
        history_feature[i, j] += 1.0
    history_feature = np.reshape(history_feature, (np.shape(fg.phi_en_de)[0],))
    fg.phi_en_de[:, -1] = history_feature

    pot_en_en = fg.phi_en_en.dot(fg.theta_en_en.T)
    pot_en_en = np.exp(pot_en_en)
    fg.pot_en_en = pot_en_en

    pot_en_de = fg.phi_en_de.dot(fg.theta_en_de.T)
    pot_en_de = np.exp(pot_en_de)
    fg.pot_en_de = pot_en_de

    # covert to sparse phi
    fg.phi_en_de_csc = sparse.csc_matrix(fg.phi_en_de)
    fg.phi_en_en_csc = sparse.csc_matrix(fg.phi_en_en)

    # create Ve x Vg factors
    for v, simplenode in var_node_pairs:
        if v.var_type == VAR_TYPE_PREDICTED:
            f = FactorNode(id=len(factors), factor_type='en_de', observed_domain_size=len_de_domain)
            o_idx = de2id[simplenode.l2_word]
            p = PotentialTable(v_id2dim={v.id: 0}, table=None, observed_dim=o_idx)
            f.add_varset_with_potentials(varset=[v], ptable=p)
            factors.append(f)
        elif v.var_type == VAR_TYPE_GIVEN:
            pass
        else:
            raise BaseException("vars are given or predicted only (no latent)")
    # create Ve x Ve factors
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
                v_pred = v1 if v1.var_type == VAR_TYPE_PREDICTED else v2
                f = FactorNode(id=len(factors),
                               factor_type='en_en',
                               observed_domain_type='en',
                               observed_domain_size=len_en_domain)
                o_idx = en2id[v_given.supervised_label]  # either a users guess OR a revealed word -> see line 31,36
                p = PotentialTable(v_id2dim={v_pred.id: 0}, table=None, observed_dim=o_idx)
                f.add_varset_with_potentials(varset=[v_pred], ptable=p)
                factors.append(f)
            pass

    for f in factors:
        fg.add_factor(f)
    for f in fg.factors:
        f.potential_table.slice_potentials()
    sys.stderr.write('.')
    return fg


def batch_check(training_instance, theta_en_en, theta_en_de, phi_en_en, phi_en_de, lr, en_domain, de2id, en2id):
    j_ti = json.loads(training_instance)
    ti = TrainingInstance.from_dict(j_ti)
    sent_id = ti.current_sent[0].sent_id
    # sys.stderr.write('sent id:' + str(sent_id))
    # print 'in:', sent_id, theta_en_en, theta_en_de
    fg = create_factor_graph(ti=ti,
                             learning_rate=lr,
                             theta_en_de=theta_en_de,
                             theta_en_en=theta_en_en,
                             phi_en_en=phi_en_en,
                             phi_en_de=phi_en_de,
                             en_domain=en_domain,
                             de2id=de2id,
                             en2id=en2id)

    fg.initialize()
    # fg.treelike_inference(3)
    return sent_id


def batch_check_accumulate(p):
    print 'completed', p


if __name__ == '__main__':
    global f_en_en_theta, f_en_de_theta

    opt = OptionParser()
    # insert options here
    opt.add_option('--ti', dest='training_instances', default='')
    opt.add_option('--end', dest='en_domain', default='')
    opt.add_option('--ded', dest='de_domain', default='')
    opt.add_option('--phi_wiwj', dest='phi_wiwj', default='')
    opt.add_option('--phi_ed', dest='phi_ed', default='')
    opt.add_option('--phi_ped', dest='phi_ped', default='')
    opt.add_option('--cpu', dest='cpus', default='')
    (options, _) = opt.parse_args()

    if options.training_instances == '' or options.en_domain == '' or options.de_domain == '' or options.phi_wiwj == '' or options.phi_ed == '' or options.phi_ped == '':
        sys.stderr.write(
            'Usage: python real_phi_test.py\n\
            --ti [training instance file]\n \
            --end [en domain file]\n \
            --ded [de domain file]\n \
            --phi_wiwj [wiwj file]\n \
            --phi_ed [ed file]\n \
            --phi_ped [ped file]\n'
            '--cpu [4 by default]\n')
        exit(1)
    else:
        pass
    print 'reading in  ti and domains...'
    cpu_count = 4 if options.cpus.strip() == '' else int(options.cpus)
    training_instances = codecs.open(options.training_instances).readlines()
    de_domain = [i.strip() for i in codecs.open(options.de_domain, 'r', 'utf8').readlines()]
    en_domain = [i.strip() for i in codecs.open(options.en_domain, 'r', 'utf8').readlines()]
    en2id = dict((e, idx) for idx, e in enumerate(en_domain))
    de2id = dict((d, idx) for idx, d in enumerate(de_domain))
    print len(en_domain), len(de_domain)
    # en_domain = ['en_' + str(i) for i in range(500)]
    # de_domain = ['de_' + str(i) for i in range(100)]
    print 'read ti and domains...'
    f_en_en = ['f1', 'dummy_ee']

    # f_en_en_theta = np.random.rand(1, len(f_en_en)) - 0.5  # zero mean random values
    f_en_en_theta = np.zeros((1, len(f_en_en)))
    print 'reading phi wiwj'
    phi_en_en1 = np.loadtxt(options.phi_wiwj)
    phi_en_en1[phi_en_en1 < 1.0 / len(en_domain)] = 0.0  # make sparse...

    phi_en_en1 = np.reshape(phi_en_en1, (len(en_domain) * len(en_domain), 1))
    ss = np.shape(phi_en_en1)
    phi_en_en2 = np.random.rand(ss[0], ss[1])
    phi_en_en2[phi_en_en2 < 0.5] = 0
    phi_en_en = np.concatenate((phi_en_en1, phi_en_en2), axis=1)
    phi_en_en.astype(DTYPE)

    f_en_de = ['x', 'y', 'dummy_ef', 'history']
    # f_en_de_theta = np.random.rand(1, len(f_en_de)) - 0.5  # zero mean random values
    f_en_de_theta = np.zeros((1, len(f_en_de)))
    print 'reading phi ed'
    phi_en_de1 = np.loadtxt(options.phi_ed)
    phi_en_de1[phi_en_de1 < 0.5] = 0.0
    phi_en_de1 = np.reshape(phi_en_de1, (len(en_domain) * len(de_domain), 1))

    print 'reading phi ped'
    phi_en_de2 = np.loadtxt(options.phi_ped)
    phi_en_de2[phi_en_de2 < 0.5] = 0.0
    phi_en_de2 = np.reshape(phi_en_de2, (len(en_domain) * len(de_domain), 1))
    ss = np.shape(phi_en_de2)
    phi_en_de3 = np.random.rand(ss[0], ss[1])
    phi_en_de3[phi_en_de3 < 0.5] = 0
    phi_en_de4 = np.zeros_like(phi_en_de1)
    phi_en_de = np.concatenate((phi_en_de1, phi_en_de2, phi_en_de3, phi_en_de4), axis=1)
    phi_en_de.astype(DTYPE)
    lr = 0.1
    pool = Pool(processes=cpu_count)
    for ti in training_instances:
        pool.apply_async(batch_check, args=(
            ti,
            f_en_en_theta,
            f_en_de_theta,
            phi_en_en,
            phi_en_de, lr,
            en_domain,
            de2id,
            en2id), callback=batch_check_accumulate)
    pool.close()
    pool.join()
    print 'done.'