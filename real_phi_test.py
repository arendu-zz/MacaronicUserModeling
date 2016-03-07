__author__ = 'arenduchintala'
import pdb
import random
from training_classes import TrainingInstance, Guess, SimpleNode
import json
import numpy as np
import sys
from optparse import OptionParser
from LBP import FactorNode, FactorGraph, VariableNode, VAR_TYPE_PREDICTED, PotentialTable, VAR_TYPE_GIVEN
import time
import codecs
from numpy import float32 as DTYPE

np.seterr(divide='raise', over='raise', under='ignore')

np.set_printoptions(precision=4, suppress=True)

global make_phi_sparse
make_phi_sparse = True


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
    len_en_domain = len(en_domain)
    len_de_domain = len(de_domain)

    history_feature = np.zeros((len_en_domain, len_de_domain))
    history_feature.astype(DTYPE)
    for pg in ti.past_correct_guesses:
        i = en2id[pg.guess]
        j = de2id[pg.l2_word]
        history_feature[i, :] -= 0.5
        history_feature[:, j] -= 0.5
        history_feature[i, j] += 1
    history_feature = np.reshape(history_feature, (np.shape(fg.phi_en_de)[0],))
    fg.phi_en_de[:, -1] = history_feature

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

    return fg


if __name__ == '__main__':
    global make_phi_sparse
    opt = OptionParser()
    # insert options here
    opt.add_option('--ti', dest='training_instances', default='')
    opt.add_option('--end', dest='en_domain', default='')
    opt.add_option('--ded', dest='de_domain', default='')
    opt.add_option('--phi_wiwj', dest='phi_wiwj', default='')
    opt.add_option('--phi_ed', dest='phi_ed', default='')
    opt.add_option('--phi_ped', dest='phi_ped', default='')
    (options, _) = opt.parse_args()

    if options.training_instances == '' or options.en_domain == '' or options.de_domain == '' or options.phi_wiwj == '' or options.phi_ed == '' or options.phi_ped == '':
        sys.stderr.write(
            'Usage: python real_phi_test.py\n\
            --ti [training instance file]\n \
            --end [en domain file]\n \
            --ded [de domain file]\n \
            --phi_wiwj [wiwj file]\n \
            --phi_ed [ed file]\n \
            --phi_ped [ped file]\n')
        exit(1)
    else:
        pass
    print 'reading in  ti and domains...'

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
    if make_phi_sparse:
        phi_en_en1[phi_en_en1 < 1.0 / len(en_domain)] = 0.0  # make sparse...
    phi_en_en1 = np.reshape(phi_en_en1, (len(en_domain) * len(en_domain), 1))
    ss = np.shape(phi_en_en1)
    phi_en_en2 = np.random.rand(ss[0], ss[1])
    if make_phi_sparse:
        phi_en_en2[phi_en_en2 < 0.5] = 0
    phi_en_en = np.concatenate((phi_en_en1, phi_en_en2), axis=1)
    phi_en_en.astype(DTYPE)
    # phi_en_en = np.random.rand(len(en_domain) * len(en_domain), len(f_en_en))
    # phi_en_en[phi_en_en > 0.8] = 1.0
    # phi_en_en[phi_en_en < 1.0] = 0.0
    # pre_fire_en_en = sparse.csr_matrix(pre_fire_en_en)

    f_en_de = ['x', 'y', 'dummy_ef', 'history']
    # f_en_de_theta = np.random.rand(1, len(f_en_de)) - 0.5  # zero mean random values
    f_en_de_theta = np.zeros((1, len(f_en_de)))
    print 'reading phi ed'
    phi_en_de1 = np.loadtxt(options.phi_ed)
    if make_phi_sparse:
        phi_en_de1[phi_en_de1 < 0.5] = 0.0
    phi_en_de1 = np.reshape(phi_en_de1, (len(en_domain) * len(de_domain), 1))

    print 'reading phi ped'
    phi_en_de2 = np.loadtxt(options.phi_ped)
    if make_phi_sparse:
        phi_en_de2[phi_en_de2 < 0.5] = 0.0
    phi_en_de2 = np.reshape(phi_en_de2, (len(en_domain) * len(de_domain), 1))
    ss = np.shape(phi_en_de2)
    phi_en_de3 = np.random.rand(ss[0], ss[1])
    if make_phi_sparse:
        phi_en_de3[phi_en_de3 < 0.5] = 0
    phi_en_de4 = np.zeros_like(phi_en_de1)
    phi_en_de = np.concatenate((phi_en_de1, phi_en_de2, phi_en_de3, phi_en_de4), axis=1)
    phi_en_de.astype(DTYPE)

    fg = None
    split_ratio = int(len(training_instances) * 0.33)
    test_instances = training_instances[:split_ratio]
    all_training_instances = training_instances[split_ratio:]
    lr = 0.1
    for epoch in range(10):
        print 'epoch', epoch
        lp = 0.0
        if epoch % 3 == 0:
            for t_idx, training_instance in enumerate(test_instances):
                j_ti = json.loads(training_instance)
                ti = TrainingInstance.from_dict(j_ti)

                fg = FactorGraph(theta_en_en=f_en_en_theta if fg is None else fg.theta_en_en,
                                 theta_en_de=f_en_de_theta if fg is None else fg.theta_en_de,
                                 phi_en_en=phi_en_en,
                                 phi_en_de=phi_en_de)

                fg = load_fg(fg, ti, en_domain, de2id=de2id, en2id=en2id)

                for f in fg.factors:
                    f.potential_table.make_potentials()  # can this be made faster??
                fg.initialize()
                fg.treelike_inference(3)
                lp += fg.get_posterior_probs()
                # print t_idx
                fg.get_max_postior_label(top=5)
                # print '...'
            print 'log sum of posteriors probs for subervised labels:', lp
        load_times = []
        grad_times = []
        inference_times = []
        mp_times = []
        random.shuffle(all_training_instances)
        for t_idx, training_instance in enumerate(all_training_instances):
            sys.stderr.write('.')
            j_ti = json.loads(training_instance)
            ti = TrainingInstance.from_dict(j_ti)
            lt = time.time()
            fg = FactorGraph(theta_en_en=f_en_en_theta if fg is None else fg.theta_en_en,
                             theta_en_de=f_en_de_theta if fg is None else fg.theta_en_de,
                             phi_en_en=phi_en_en,
                             phi_en_de=phi_en_de)
            fg.learning_rate = lr

            fg = load_fg(fg, ti, en_domain, de2id=de2id, en2id=en2id)
            load_times.append(time.time() - lt)
            mp = time.time()
            for f in fg.factors:
                f.potential_table.make_potentials()  # can this be made faster??
            mp_times.append(time.time() - mp)
            it = time.time()
            fg.initialize()
            fg.treelike_inference(3)
            inference_times.append(time.time() - it)
            gt = time.time()
            fg.update_theta()
            grad_times.append(time.time() - gt)
        print 'ave load times:', sum(load_times) / float(len(load_times))
        print 'ave mp times  :', sum(mp_times) / float(len(mp_times))
        print 'ave inf times :', sum(inference_times) / float(len(inference_times))
        print 'ave grad times:', sum(grad_times) / float(len(grad_times))
        lr *= 0.5
    print 'theta final:', fg.theta_en_de, fg.theta_en_en
