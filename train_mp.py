__author__ = 'arenduchintala'
import pdb
import random
from training_classes import TrainingInstance
import json
import numpy as np
import sys
from optparse import OptionParser
from LBP import FactorNode, FactorGraph, VariableNode, VAR_TYPE_PREDICTED, PotentialTable, VAR_TYPE_GIVEN
from time import ctime
import codecs
from numpy import float32 as DTYPE
from multiprocessing import Pool, Lock
from array_utils import PhiWrapper

global f_en_en_theta, f_en_de_theta, prediction_probs, intermediate_writer, prediction_str, final_writer, n_up
global lock
global options
n_up = 0
lock = Lock()
np.seterr(divide='raise', over='raise', under='ignore')

np.set_printoptions(precision=4, suppress=True)


def error_msg():
    sys.stderr.write(
        'Usage: python real_phi_test.py\n\
                --ti [training instance file]\n \
                --end [en domain file]\n \
                --ded [de domain file]\n \
                --phi_pmi [pmi file]\n \
                --phi_pmi_w1 [pmi w1 file]\n \
                --phi_ed [ed file]\n \
                --phi_ped [ped file]\n \
                --cpu [4 by default]\n \
                --save_params [save params to this file] or \n \
                --load_params [load params from this file]] --save_predictions [save predictions to file]\n')


def find_guess(simplenode_id, guess_list):
    for cg in guess_list:
        if simplenode_id == cg.id:
            guess = cg
            return guess
    return None


def read_params(params_file):
    p1, p2 = codecs.open(params_file, 'r', 'utf8').read().split('ED_F:')
    _, p1 = p1.split('EE_F:')
    p1_lines = p1.split('\n')
    een = p1_lines[0].split()
    eet = np.array([float(i) for i in p1_lines[1].split()[1:]])
    eet = np.reshape(eet, (1, np.size(eet)))
    p2_lines = p2.split('\n')
    edn = p2_lines[0].split()
    edt = np.array([float(i) for i in p2_lines[1].split()[1:]])
    edt = np.reshape(edt, (1, np.size(edt)))
    print 'loaded params...'
    return een, eet, edn, edt


def save_params(w, ee_theta, ed_theta, ee_names, ed_names):
    w.write('\t'.join(['EE_F:'] + ee_names) + '\n')
    fl = [item for sublist in ee_theta.tolist() for item in sublist]
    n_o = ['Original'.ljust(15)] + ['%0.6f' % i for i in fl]
    w.write('\t'.join(n_o) + '\n')
    w.write('\t'.join(['ED_F:'] + ed_names) + '\n')
    fl = [item for sublist in ed_theta.tolist() for item in sublist]
    n_o = ['Original'.ljust(15)] + ['%0.6f' % i for i in fl]
    w.write('\t'.join(n_o) + '\n')
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


def create_factor_graph(ti, learning_rate,
                        theta_en_en_names, theta_en_de_names,
                        theta_en_en, theta_en_de,
                        phi_wrapper,
                        en_domain,
                        de2id, en2id):
    ordered_current_sent = sorted([(simplenode.position, simplenode) for simplenode in ti.current_sent])
    ordered_current_sent = [simplenode for position, simplenode in ordered_current_sent]
    var_node_pairs = get_var_node_pair(ordered_current_sent, ti.current_guesses, ti.current_revealed_guesses, en_domain)
    factors = []

    len_en_domain = len(en_domain)
    len_de_domain = len(de_domain)
    fg = FactorGraph(theta_en_en_names=theta_en_en_names,
                     theta_en_de_names=theta_en_de_names,
                     theta_en_en=theta_en_en,
                     theta_en_de=theta_en_de,
                     phi_en_en=phi_wrapper.phi_en_en,
                     phi_en_de=phi_wrapper.phi_en_de,
                     phi_en_en_w1=phi_wrapper.phi_en_en_w1)

    fg.learning_rate = learning_rate
    if options.history:
        history_feature = np.zeros((len_en_domain, len_de_domain))
        history_feature.astype(DTYPE)
        for pg in ti.past_correct_guesses:
            i = en2id[pg.guess]
            j = de2id[pg.l2_word]
            history_feature[i, :] -= 0.01
            history_feature[:, j] -= 0.01
            history_feature[i, j] += 1.02
        history_feature = np.reshape(history_feature, (np.shape(fg.phi_en_de)[0],))
        fg.phi_en_de[:, -2] = history_feature
    if options.session_history:
        incorrect_history = np.zeros((len_en_domain, len_de_domain))
        incorrect_history.astype(DTYPE)
        for ig in ti.past_guesses_for_current_sent:
            if not ig.revealed:
                i = en2id[ig.guess]
                j = de2id[ig.l2_word]
                incorrect_history[i, j] -= 1.00
                # if it was close then give some positive weight for close words
                # for that i need to precompute closeness in my vocabulary
        incorrect_history = np.reshape(incorrect_history, (np.shape(fg.phi_en_de)[0],))
        fg.phi_en_de[:, -1] = incorrect_history
    theta_pmi = fg.theta_en_en[0, theta_en_en_names.index('pmi')]

    pot_en_en = fg.phi_en_en * theta_pmi  # fg.phi_en_en.dot(fg.theta_en_en.T)
    pot_en_en = np.exp(pot_en_en)
    fg.pot_en_en = pot_en_en

    theta_pmi_w1 = fg.theta_en_en[0, theta_en_en_names.index('pmi_w1')]
    pot_en_en_w1 = fg.phi_en_en_w1 * theta_pmi_w1  # fg.phi_en_en_w1.dot(fg.theta_en_en.T)
    pot_en_en_w1 = np.exp(pot_en_en_w1)
    fg.pot_en_en_w1 = pot_en_en_w1
    pot_en_de = fg.phi_en_de.dot(fg.theta_en_de.T)
    pot_en_de = np.exp(pot_en_de)
    fg.pot_en_de = pot_en_de
    # covert to sparse phi
    # fg.phi_en_de_csc = sparse.csc_matrix(fg.phi_en_de)
    # fg.phi_en_en_csc = sparse.csc_matrix(fg.phi_en_en)

    # create Ve x Vg factors
    for v, simplenode in var_node_pairs:
        if v.var_type == VAR_TYPE_PREDICTED:
            f = FactorNode(id=len(factors), factor_type='en_de', observed_domain_size=len_de_domain)
            o_idx = de2id[simplenode.l2_word]
            p = PotentialTable(v_id2dim={v.id: 0}, table=None, observed_dim=o_idx)
            f.add_varset_with_potentials(varset=[v], ptable=p)
            f.position = v.id
            f.gap = 0
            f.word_label = simplenode.l2_word
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
                f.position = None
                f.word_label = None
                f.gap = abs(v1.id - v2.id)
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
                f.position = v_given.id
                f.gap = abs(v_given.id - v_pred.id)
                f.word_label = v_given.supervised_label
                factors.append(f)
            pass

    for f in factors:
        fg.add_factor(f)
    for f in fg.factors:
        f.potential_table.slice_potentials()
    sys.stderr.write('.')
    return fg


def batch_predictions(training_instance,
                      theta_en_en_names, theta_en_de_names,
                      theta_en_en, theta_en_de,
                      phi_wapper, lr,
                      en_domain, de2id, en2id):
    j_ti = json.loads(training_instance)
    ti = TrainingInstance.from_dict(j_ti)
    sent_id = ti.current_sent[0].sent_id

    fg = create_factor_graph(ti=ti,
                             learning_rate=lr,
                             theta_en_en_names=theta_en_en_names,
                             theta_en_de_names=theta_en_de_names,
                             theta_en_de=theta_en_de,
                             theta_en_en=theta_en_en,
                             phi_wrapper=phi_wapper,
                             en_domain=en_domain,
                             de2id=de2id,
                             en2id=en2id)

    fg.initialize()
    fg.treelike_inference(3)
    p = fg.get_posterior_probs()
    fgs = '\n'.join(['*SENT_ID:' + str(sent_id)] + fg.to_string())
    return [p, fgs]


def batch_prediction_probs_accumulate(result):
    global prediction_probs, intermediate_writer, prediction_str, n_up
    prediction_probs += result[0]
    fgs = result[1]
    if prediction_str is not None:
        prediction_str = prediction_str + fgs + '\n'
    else:
        prediction_str = fgs + '\n'
    n_up += 1
    sys.stderr.write('*')


def batch_sgd(training_instance,
              theta_en_en_names, theta_en_de_names,
              theta_en_en, theta_en_de,
              phi_wrapper, lr,
              en_domain, de2id, en2id):
    j_ti = json.loads(training_instance)
    ti = TrainingInstance.from_dict(j_ti)
    sent_id = ti.current_sent[0].sent_id

    fg = create_factor_graph(ti=ti,
                             learning_rate=lr,
                             theta_en_en_names=theta_en_en_names,
                             theta_en_de_names=theta_en_de_names,
                             theta_en_de=theta_en_de,
                             theta_en_en=theta_en_en,
                             phi_wrapper=phi_wrapper,
                             en_domain=en_domain,
                             de2id=de2id,
                             en2id=en2id)

    fg.initialize()

    fg.treelike_inference(3)
    # sys.stderr.write('.')
    # f_en_en_theta, f_en_de_theta = fg.update_theta()
    # sys.stderr.write('|')

    g_en_en, g_en_de = fg.return_gradient()
    # sys.stderr.write('+')

    p = fg.get_posterior_probs()

    return [sent_id, p, g_en_en, g_en_de]


def batch_sgd_accumulate(result):
    global f_en_en_theta, f_en_de_theta, n_up, prediction_probs
    lock.acquire()
    prediction_probs += result[1]
    f_en_en_theta += result[2]
    f_en_de_theta += result[3]
    sys.stderr.write('*')
    lock.release()
    if n_up % 10 == 0:
        intermediate_writer.write(
            str(n_up) + ' ' + np.array_str(f_en_en_theta) + ' ' + np.array_str(f_en_de_theta) + '\n')
        intermediate_writer.flush()
    n_up += 1
    # print 'received', result[0], f_en_en_theta, f_en_de_theta


if __name__ == '__main__':
    global f_en_en_theta, f_en_de_theta, prediction_probs, prediction_str, intermediate_writer, n_up

    opt = OptionParser()
    # insert options here
    opt.add_option('--ti', dest='training_instances', default='')
    opt.add_option('--end', dest='en_domain', default='')
    opt.add_option('--ded', dest='de_domain', default='')
    opt.add_option('--phi_pmi', dest='phi_pmi', default='')
    opt.add_option('--phi_pmi_w1', dest='phi_pmi_w1', default='')
    opt.add_option('--phi_ed', dest='phi_ed', default='')
    opt.add_option('--phi_ped', dest='phi_ped', default='')
    opt.add_option('--cpu', dest='cpus', default='')
    opt.add_option('--save_params', dest='save_params_file', default='')
    opt.add_option('--load_params', dest='load_params_file', default='')
    opt.add_option('--save_predictions', dest='save_predictions_file', default='')
    opt.add_option('--history', dest='history', default=False, action='store_true')
    opt.add_option('--session_history', dest='session_history', default=False, action='store_true')
    (options, _) = opt.parse_args()

    if options.training_instances == '' or options.en_domain == '' or options.de_domain == '' or options.phi_pmi_w1 == '' or options.phi_pmi == '' or options.phi_ed == '' or options.phi_ped == '':
        error_msg()
        exit(1)
    elif options.save_params_file == '' and options.load_params_file == '' and options.save_predictions_file == '':
        error_msg()
        exit(1)
    else:
        pass

    cpu_count = 4 if options.cpus.strip() == '' else int(options.cpus)
    print 'cpu count:', cpu_count
    mode = 'training' if options.load_params_file == '' else 'predicting'
    if mode == 'training':
        f_en_en_names = ['pmi', 'pmi_w1']
        f_en_en_theta = np.zeros((1, len(f_en_en_names)))
        f_en_de_names = ['ed', 'ped', 'full_history', 'hit_history']
        f_en_de_theta = np.zeros((1, len(f_en_de_names)))
    else:
        een, eet, edn, edt = read_params(options.load_params_file)
        save_predictions_file = options.save_predictions_file
        f_en_en_names = een
        f_en_en_theta = eet
        f_en_de_names = edn
        f_en_de_theta = edt

    print 'reading in  ti and domains...'
    training_instances = codecs.open(options.training_instances).readlines()
    training_instances = training_instances[:10]

    de_domain = [i.strip() for i in codecs.open(options.de_domain, 'r', 'utf8').readlines()]
    en_domain = [i.strip() for i in codecs.open(options.en_domain, 'r', 'utf8').readlines()]
    en2id = dict((e, idx) for idx, e in enumerate(en_domain))
    de2id = dict((d, idx) for idx, d in enumerate(de_domain))
    print len(en_domain), len(de_domain)

    print 'reading phi pmi'
    phi_en_en1 = np.loadtxt(options.phi_pmi)
    phi_en_en1 = np.reshape(phi_en_en1, (len(en_domain) * len(en_domain), 1))
    phi_en_en = np.concatenate((phi_en_en1,), axis=1)
    phi_en_en.astype(DTYPE)
    print 'reading phi pmi w1'
    phi_en_en_w1 = np.loadtxt(options.phi_pmi_w1)
    phi_en_en_w1 = np.reshape(phi_en_en1, (len(en_domain) * len(en_domain), 1))
    phi_en_en_w1.astype(DTYPE)

    print 'reading phi ed'
    phi_en_de1 = np.loadtxt(options.phi_ed)
    phi_en_de1 = np.reshape(phi_en_de1, (len(en_domain) * len(de_domain), 1))

    print 'reading phi ped'
    phi_en_de2 = np.loadtxt(options.phi_ped)
    phi_en_de2 = np.reshape(phi_en_de2, (len(en_domain) * len(de_domain), 1))
    phi_en_de3 = np.zeros_like(phi_en_de1)  # place holder for history
    phi_en_de4 = np.zeros_like(phi_en_de1)  # place holder for incorrect history
    phi_en_de = np.concatenate((phi_en_de1, phi_en_de2, phi_en_de3, phi_en_de4), axis=1)
    phi_en_de.astype(DTYPE)

    phi_wrapper = PhiWrapper(phi_en_en, phi_en_en_w1, phi_en_de)
    lock = Lock()
    t_now = '-'.join(ctime().split())
    model_param_writer_name = options.training_instances + '.cpu' + str(cpu_count) + '.' + t_now + '.params'
    intermediate_writer = open(model_param_writer_name + '.tmp', 'w')
    if mode == 'training':
        for epoch in range(3):
            lr = 0.05
            prediction_probs = 0.0
            print 'epoch:', epoch
            print f_en_en_names, f_en_en_theta
            print f_en_de_names, f_en_de_theta
            random.shuffle(training_instances)
            pool = Pool(processes=cpu_count)
            for ti in training_instances:
                pool.apply_async(batch_sgd, args=(
                    ti,
                    f_en_en_names,
                    f_en_de_names,
                    f_en_en_theta,
                    f_en_de_theta,
                    phi_wrapper,
                    lr,
                    en_domain,
                    de2id,
                    en2id), callback=batch_sgd_accumulate)
            pool.close()
            pool.join()
            print '\nepoch:', epoch
            print f_en_en_names, f_en_en_theta
            print f_en_de_names, f_en_de_theta
            print '\nprediction probs:', prediction_probs
            lr *= 0.75
        print '\ntheta final:', f_en_en_theta, f_en_de_theta
        final_writer = codecs.open(model_param_writer_name + '.final', 'w')
        save_params(final_writer, f_en_en_theta, f_en_de_theta, f_en_en_names, f_en_de_names)
        final_writer = codecs.open(options.save_params_file, 'w', 'utf8')
        save_params(final_writer, f_en_en_theta, f_en_de_theta, f_en_en_names, f_en_de_names)
    else:
        print 'predicting...'
        print f_en_en_names, f_en_en_theta
        print f_en_de_names, f_en_de_theta
        prediction_str = ''
        pool = Pool(processes=cpu_count)
        lr = 0.05
        n_up = 0
        prediction_probs = 0.0
        for ti in training_instances:
            pool.apply_async(batch_predictions, args=(
                ti,
                f_en_en_names,
                f_en_de_names,
                f_en_en_theta,
                f_en_de_theta,
                phi_wrapper,
                lr,
                en_domain,
                de2id,
                en2id), callback=batch_prediction_probs_accumulate)
        pool.close()
        pool.join()
        print '\nprediction probs:', prediction_probs, n_up
        final_writer = codecs.open(save_predictions_file, 'w', 'utf8')
        final_writer.write(prediction_str)
