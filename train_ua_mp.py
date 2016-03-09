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
from time import ctime
import codecs
from numpy import float32 as DTYPE
from scipy import sparse
from multiprocessing import Pool
import itertools
import array_utils

global f_en_en_theta, f_en_de_theta, prediction_probs, writer, n_up
n_up = 0
np.seterr(divide='raise', over='raise', under='ignore')

np.set_printoptions(precision=4, suppress=True)


def find_guess(simplenode_id, guess_list):
    for cg in guess_list:
        if simplenode_id == cg.id:
            guess = cg
            return guess
    return None


def save_params(w, ee_theta, ed_theta, ee_names, ed_names, adaptation2id):
    en_de_adapt_names = [(0, 'original')] + sorted([(i, '_'.join(k)) for k, i in adaptation2id.iteritems()])
    en_de_adapt_names = [k for i, k in en_de_adapt_names]

    ed_adapt_names = ['_'.join([uname, basic]) for uname, basic in
                      itertools.product(en_de_adapt_names, ed_names)]
    print en_de_adapt_names
    print ed_names
    print np.shape(ed_theta), (len(en_de_adapt_names), len(ed_names))
    en_de_adapt_weights = np.reshape(ed_theta, (len(en_de_adapt_names), len(ed_names)))

    en_en_adapt_names = [(0, 'original')] + sorted([(i, '_'.join(k)) for k, i in adaptation2id.iteritems()])
    en_en_adapt_names = [k for i, k in en_en_adapt_names]

    ee_adapt_names = ['_'.join([uname, basic]) for uname, basic in
                      itertools.product(en_en_adapt_names, ee_names)]

    print en_en_adapt_names
    print ee_names
    print np.shape(ee_theta), (len(en_en_adapt_names), len(ee_names))
    en_en_adapt_weights = np.reshape(ee_theta, (len(en_en_adapt_names), len(ee_names)))

    w.write('\n'.join(ee_adapt_names))
    w.write('\n')
    w.write(np.array_str(en_en_adapt_weights))

    w.write('\n'.join(ed_adapt_names))
    w.write('\n')
    w.write(np.array_str(en_de_adapt_weights))

    w.flush()
    w.close()


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


def create_factor_graph(ti,
                        learning_rate,
                        theta_en_en,
                        theta_en_de,
                        phi_en_en,
                        phi_en_de,
                        basic_f_en_en,
                        basic_f_en_de,
                        en_domain,
                        de2id,
                        en2id,
                        adaptation2id):
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
    history_feature = np.reshape(history_feature, (np.shape(fg.phi_en_de)[0], 1))
    # print 'here'
    # print basic_f_en_de.index('history')
    # print np.shape(fg.phi_en_de), np.shape(history_feature), type(fg.phi_en_de), type(history_feature)
    fg.phi_en_de[:, basic_f_en_de.index('history')] = history_feature
    # print 'here after history'
    # adapt
    user_id = ti.user_id
    user_experience = ''
    '''
    if len(ti.past_sentences_seen) < 10:
        user_experience = 'novice'
    elif len(ti.past_sentences_seen) < 40
        user_experience = 'beginner'
    else:
        user_experience = 'familiar'
    '''

    on_id = adaptation2id[user_id, user_experience]
    print 'on_id', on_id
    fg.phi_en_en = array_utils.set_adaptation(len(basic_f_en_en), fg.phi_en_en, [on_id])
    fg.phi_en_de = array_utils.set_adaptation(len(basic_f_en_de), fg.phi_en_de, [on_id])

    fg.phi_en_de_csc = fg.phi_en_de.tocsc()  # sparse.csc_matrix(fg.phi_en_de)
    fg.phi_en_en_csc = fg.phi_en_en.tocsc()  # sparse.csc_matrix(fg.phi_en_en)
    # print 'here2'
    pot_en_en = fg.phi_en_en.dot(fg.theta_en_en.T)
    pot_en_en = np.exp(pot_en_en)
    fg.pot_en_en = pot_en_en

    pot_en_de = fg.phi_en_de.dot(fg.theta_en_de.T)
    pot_en_de = np.exp(pot_en_de)
    fg.pot_en_de = pot_en_de

    # covert to sparse phi


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
    sys.stderr.write('ok load graph...')
    return fg


def batch_predictions(training_instance,
                      f_en_en_theta,
                      f_en_de_theta,
                      adapt_phi_en_en,
                      adapt_phi_en_de, lr,
                      en_domain,
                      de2id,
                      en2id,
                      basic_f_en_en,
                      basic_f_en_de,
                      adaptation2id):
    j_ti = json.loads(training_instance)
    ti = TrainingInstance.from_dict(j_ti)
    sent_id = ti.current_sent[0].sent_id
    # sys.stderr.write('sent id:' + str(sent_id))
    # print 'in:', sent_id, theta_en_en, theta_en_de
    fg = create_factor_graph(ti=ti,
                             learning_rate=lr,
                             theta_en_de=f_en_de_theta,
                             theta_en_en=f_en_en_theta,
                             phi_en_en=adapt_phi_en_en,
                             phi_en_de=adapt_phi_en_de,
                             basic_f_en_en=basic_f_en_en,
                             basic_f_en_de=basic_f_en_de,
                             en_domain=en_domain,
                             de2id=de2id,
                             en2id=en2id,
                             adaptation2id=adaptation2id)

    fg.initialize()
    fg.treelike_inference(3)

    return fg.get_posterior_probs()


def batch_prediction_probs_accumulate(p):
    global prediction_probs, n_up
    prediction_probs += p
    if n_up % 10 == 0:
        writer.write(str(n_up) + ' pred prob:' + str(prediction_probs) + '\n')
        writer.flush()
    n_up += 1


def batch_sgd(training_instance,
              theta_en_en,
              theta_en_de,
              phi_en_en,
              phi_en_de, lr,
              en_domain,
              de2id,
              en2id,
              basic_f_en_en,
              basic_f_en_de,
              adaptation2id):
    j_ti = json.loads(training_instance)
    ti = TrainingInstance.from_dict(j_ti)
    sent_id = ti.current_sent[0].sent_id
    fg = create_factor_graph(ti=ti,
                             learning_rate=lr,
                             theta_en_de=theta_en_de,
                             theta_en_en=theta_en_en,
                             phi_en_en=phi_en_en,
                             phi_en_de=phi_en_de,
                             basic_f_en_en=basic_f_en_en,
                             basic_f_en_de=basic_f_en_de,
                             en_domain=en_domain,
                             de2id=de2id,
                             en2id=en2id,
                             adaptation2id=adaptation2id)

    fg.initialize()
    # sys.stderr.write('.')
    fg.treelike_inference(3)
    # sys.stderr.write('.')
    # f_en_en_theta, f_en_de_theta = fg.update_theta()
    g_en_en, g_en_de = fg.return_gradient()
    # turn off adapt_phi
    return [sent_id, g_en_en, g_en_de]


def batch_sgd_accumulate(result):
    global f_en_en_theta, f_en_de_theta, n_up
    f_en_en_theta += result[1]
    f_en_de_theta += result[2]
    if n_up % 10 == 0:
        writer.write(str(n_up) + ' ' + np.array_str(f_en_en_theta) + ' ' + np.array_str(f_en_de_theta) + '\n')
        writer.flush()
    n_up += 1
    # print 'received', result[0], f_en_en_theta, f_en_de_theta


if __name__ == '__main__':
    global f_en_en_theta, f_en_de_theta, prediction_probs, writer, n_up

    opt = OptionParser()
    # insert options here
    opt.add_option('--ti', dest='training_instances', default='')
    opt.add_option('--end', dest='en_domain', default='')
    opt.add_option('--ded', dest='de_domain', default='')
    opt.add_option('--phi_wiwj', dest='phi_wiwj', default='')
    opt.add_option('--phi_ed', dest='phi_ed', default='')
    opt.add_option('--phi_ped', dest='phi_ped', default='')
    opt.add_option('--users', dest='user_list', default='')
    opt.add_option('--cpu', dest='cpus', default='')
    (options, _) = opt.parse_args()

    if options.user_list.strip() == '' or options.training_instances == '' or options.en_domain == '' or options.de_domain == '' or options.phi_wiwj == '' or options.phi_ed == '' or options.phi_ped == '':
        sys.stderr.write(
            'Usage: python real_phi_test.py\n\
                    --ti [training instance file]\n \
                    --end [en domain file]\n \
                    --ded [de domain file]\n \
                    --phi_wiwj [wiwj file]\n \
                    --phi_ed [ed file]\n \
                    --phi_ped [ped file]\n'
            '--cpu [4 by default]'
            '--users [users file list]\n')
        exit(1)
    else:
        pass

    cpu_count = 4 if options.cpus.strip() == '' else int(options.cpus)
    print 'cpu count:', cpu_count
    print 'reading in  ti and domains...'
    training_instances = codecs.open(options.training_instances).readlines()
    t_now = '-'.join(ctime().split())
    model_param_writer_name = options.training_instances + '.cpu' + str(cpu_count) + '.' + t_now + '.adapt.params'
    writer = open(model_param_writer_name, 'w')
    de_domain = [i.strip() for i in codecs.open(options.de_domain, 'r', 'utf8').readlines()]
    en_domain = [i.strip() for i in codecs.open(options.en_domain, 'r', 'utf8').readlines()]
    en2id = dict((e, idx) for idx, e in enumerate(en_domain))
    de2id = dict((d, idx) for idx, d in enumerate(de_domain))
    users = [i.strip() for i in codecs.open(options.user_list, 'r', 'utf8').readlines()]
    experience = ['']
    adaptation2id = {}
    for u, e in itertools.product(users, experience):
        adaptation2id[u, e] = len(adaptation2id) + 1
    print len(en_domain), len(de_domain)
    print 'read ti and domains...'
    basic_f_en_en = ['skipgram']

    # f_en_en_theta = np.random.rand(1, len(f_en_en)) - 0.5  # zero mean random values
    f_en_en_theta = np.ones((1, len(basic_f_en_en)))
    print 'reading phi wiwj'
    phi_en_en1 = np.loadtxt(options.phi_wiwj)
    phi_en_en1[phi_en_en1 < 2.0 / len(en_domain)] = 0.0  # make sparse...
    print np.count_nonzero(phi_en_en1)
    phi_en_en1 = np.reshape(phi_en_en1, (len(en_domain) * len(en_domain), 1))
    ss = np.shape(phi_en_en1)
    phi_en_en = np.concatenate((phi_en_en1,), axis=1)
    phi_en_en.astype(DTYPE)
    phi_en_en_off = np.zeros_like(phi_en_en)
    phi_en_en_off.astype(DTYPE)

    basic_f_en_de = ['ed', 'ped', 'history']
    # f_en_de_theta = np.random.rand(1, len(f_en_de)) - 0.5  # zero mean random values
    f_en_de_theta = np.zeros((1, len(basic_f_en_de)))
    print 'reading phi ed'
    phi_en_de1 = np.loadtxt(options.phi_ed)
    phi_en_de1[phi_en_de1 < 0.5] = 0.0
    phi_en_de1 = np.reshape(phi_en_de1, (len(en_domain) * len(de_domain), 1))

    print 'reading phi ped'
    phi_en_de2 = np.loadtxt(options.phi_ped)
    phi_en_de2[phi_en_de2 < 0.5] = 0.0
    phi_en_de2 = np.reshape(phi_en_de2, (len(en_domain) * len(de_domain), 1))
    phi_en_de3 = np.zeros_like(phi_en_de1)
    phi_en_de = np.concatenate((phi_en_de1, phi_en_de2, phi_en_de3), axis=1)
    phi_en_de.astype(DTYPE)
    phi_en_de_off = np.zeros_like(phi_en_de)
    phi_en_de_off.astype(DTYPE)

    # appending features for adaptation..
    print 'adding adaptation features...'
    f_en_en_theta = np.ones((1, len(basic_f_en_en) * (len(adaptation2id) + 1)))
    f_en_en_theta *= 0.2646
    f_en_de_theta = np.zeros((1, len(basic_f_en_de) * (len(adaptation2id) + 1)))
    old = np.array([4.74, 2.3359, 3.3745])
    for o in range(len(adaptation2id) + 1):
        st = o * len(basic_f_en_de)
        f_en_de_theta[0, range(st, st + len(basic_f_en_de))] = old
    adapt_phi_en_en = array_utils.make_adapt_phi(phi_en_en, len(adaptation2id))
    adapt_phi_en_en = sparse.lil_matrix(adapt_phi_en_en)
    adapt_phi_en_de = array_utils.make_adapt_phi(phi_en_de, len(adaptation2id))
    adapt_phi_en_de = sparse.lil_matrix(adapt_phi_en_de)

    assert np.shape(f_en_en_theta)[1] == np.shape(adapt_phi_en_en)[1]
    print 'done..'
    split_ratio = int(len(training_instances) * 0.1)
    test_instances = training_instances[:split_ratio]
    all_training_instances = training_instances[split_ratio:]
    prediction_probs = 0.0
    lr = 0.1

    w = codecs.open(model_param_writer_name + '.init', 'w')
    save_params(w, f_en_en_theta, f_en_de_theta, basic_f_en_en, basic_f_en_de, adaptation2id)
    pool = Pool(processes=cpu_count)
    for ti in test_instances:
        pool.apply_async(batch_predictions, args=(
            ti,
            f_en_en_theta,
            f_en_de_theta,
            adapt_phi_en_en,
            adapt_phi_en_de, lr,
            en_domain,
            de2id,
            en2id,
            basic_f_en_en,
            basic_f_en_de,
            adaptation2id), callback=batch_prediction_probs_accumulate)
    pool.close()
    pool.join()
    print '\nprediction probs:', prediction_probs
    for epoch in range(2):
        lr = 0.05
        print 'epoch:', epoch, 'theta:', f_en_en_theta, f_en_de_theta
        random.shuffle(all_training_instances)
        pool = Pool(processes=cpu_count)
        for ti in all_training_instances:
            pool.apply_async(batch_sgd, args=(
                ti,
                f_en_en_theta,
                f_en_de_theta,
                adapt_phi_en_en,
                adapt_phi_en_de, lr,
                en_domain,
                de2id,
                en2id,
                basic_f_en_en,
                basic_f_en_de,
                adaptation2id), callback=batch_sgd_accumulate)
        pool.close()
        pool.join()
        print '\nepoch:', epoch, f_en_en_theta, f_en_de_theta
        prediction_probs = 0.0
        pool = Pool(processes=cpu_count)
        for ti in test_instances:
            pool.apply_async(batch_predictions, args=(
                ti,
                f_en_en_theta,
                f_en_de_theta,
                adapt_phi_en_en,
                adapt_phi_en_de, lr,
                en_domain,
                de2id,
                en2id,
                basic_f_en_en,
                basic_f_en_de,
                adaptation2id), callback=batch_prediction_probs_accumulate)
        pool.close()
        pool.join()
        lr *= 0.5
        print '\nprediction probs:', prediction_probs

print '\ntheta final:', f_en_en_theta, f_en_de_theta
w = codecs.open(model_param_writer_name + '.final', 'w')
save_params(w, f_en_en_theta, f_en_de_theta, basic_f_en_en, basic_f_en_de, adaptation2id)
'''
f_en_en_theta = np.reshape(f_en_en_theta, (np.size(f_en_en_theta),))
f_en_de_theta = np.reshape(f_en_de_theta, (np.size(f_en_de_theta),))
en_en_names = basic_f_en_en
en_en_adapt_names = [(0, 'original')] + sorted([(i, k) for k, i in adaptation2id.iteritems()])
en_en_adapt_names = [sum((uname, basic), ()) for uname, basic in
                     itertools.product([k for i, k in en_en_adapt_names], basic_f_en_en)]
en_en_adapt_weights = np.reshape(f_en_en_theta, (len(en_en_adapt_names), len(basic_f_en_en)))

en_de_names = basic_f_en_de
en_de_adapt_names = [(0, 'original')] + sorted([(i, k) for k, i in adaptation2id.iteritems()])
en_de_adapt_names = [sum((uname, basic), ()) for uname, basic in
                     itertools.product([k for i, k in en_de_adapt_names], basic_f_en_de)]
en_de_adapt_weights = np.reshape(f_en_de_theta, (len(en_de_adapt_names), len(basic_f_en_de)))
w.write('\n'.join(en_en_adapt_names))
w.write('\n')
w.write(np.array_str(en_en_adapt_weights))

w.write('\n'.join(en_de_adapt_names))
w.write('\n')
w.write(np.array_str(en_de_adapt_weights))

w.flush()
w.close()
'''
