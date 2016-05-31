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
global domain2theta
n_up = 0
lock = Lock()
np.seterr(divide='raise', over='raise', under='ignore')

np.set_printoptions(precision=4, suppress=True)


def apply_regularization(reg, grad, lr, theta):
    rg = reg * theta
    grad -= rg
    grad *= lr
    return grad


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
    d2t = {}
    p1, p2 = codecs.open(params_file, 'r', 'utf8').read().split('ED_F:')
    p1 = p1.strip()
    p2 = p2.strip()
    _, p1 = p1.split('EE_F:')
    p1_lines = p1.split('\n')
    een = p1_lines[0].split()  # first line has names
    eet = np.array([float(i) for i in p1_lines[1].split()[1:]])  # second line has original weights
    eet = np.reshape(eet, (1, np.size(eet)))
    for p1_line in p1_lines[2:]:  # third line onwards has adapted weights
        items = p1_line.split()
        d = items[0]
        d_eet = np.array([float(i) for i in items[1:]])
        d_eet = np.reshape(d_eet, (1, np.size(d_eet)))
        d2t['en_en', d.strip()] = d_eet

    p2_lines = p2.split('\n')
    edn = p2_lines[0].split()  # first line has names
    edt = np.array([float(i) for i in p2_lines[1].split()[1:]])  # second line has original weights
    edt = np.reshape(edt, (1, np.size(edt)))
    for p2_line in p2_lines[2:]:
        items = p2_line.split()
        d = items[0]
        d_edt = np.array([float(i) for i in items[1:]])
        d_edt = np.reshape(d_edt, (1, np.size(d_edt)))
        d2t['en_de', d] = d_edt
    print 'loaded params...', len(d2t), 'adapted params..'
    return een, eet, edn, edt, d2t


def save_params(w, ee_theta, ed_theta, ee_names, ed_names, d2t):
    w.write('\t'.join(['EE_F:'] + ee_names) + '\n')
    fl = [item for sublist in ee_theta.tolist() for item in sublist]
    n_o = ['Original'.ljust(15)] + ['%0.6f' % i for i in fl]
    w.write('\t'.join(n_o) + '\n')
    for ft, d in d2t:
        if ft == 'en_en':
            dt = d2t[ft, d]
            fl = [item for sublist in dt.tolist() for item in sublist]
            n_o = [d.ljust(15)] + ['%0.6f' % i for i in fl]
            w.write('\t'.join(n_o) + '\n')
    w.write('\t'.join(['ED_F:'] + ed_names) + '\n')
    fl = [item for sublist in ed_theta.tolist() for item in sublist]
    n_o = ['Original'.ljust(15)] + ['%0.6f' % i for i in fl]
    w.write('\t'.join(n_o) + '\n')
    for ft, d in d2t:
        if ft == 'en_de':
            dt = d2t[ft, d]
            fl = [item for sublist in dt.tolist() for item in sublist]
            n_o = [d.ljust(15)] + ['%0.6f' % i for i in fl]
            w.write('\t'.join(n_o) + '\n')
    w.flush()
    w.close()


def get_var_node_pair(sorted_current_sent, current_guesses, current_revealed, en_domain):
    var_node_pairs = []

    for idx, simplenode in enumerate(sorted_current_sent):
        truth = simplenode.l1_parent
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
                if var_type == VAR_TYPE_PREDICTED:
                    v.set_truth_label(truth)

            except AssertionError:
                print 'something bad...'
        var_node_pairs.append((v, simplenode))
    return var_node_pairs


def create_factor_graph(ti, learning_rate,
                        theta_en_en_names, theta_en_de_names,
                        theta_en_en, theta_en_de,
                        phi_wrapper,
                        en_domain,
                        de2id, en2id, d2t):
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

    if options.user_adapt:
        d = ti.user_id
        fg.active_domains['en_en', d] = 1
        fg.active_domains['en_de', d] = 1
        sys.stderr.write('+')
    elif options.experience_adapt:
        d = len(ti.past_sentences_seen)
        fg.active_domains['en_en', d] = 1
        fg.active_domains['en_de', d] = 1
        sys.stderr.write('=')
    elif not options.user_adapt and not options.experience_adapt:
        pass
    else:
        raise BaseException("2 domains not supported simultaniously")

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

    theta_pmi_w1 = fg.theta_en_en[0, theta_en_en_names.index('pmi_w1')]
    pot_en_en_w1 = fg.phi_en_en_w1 * theta_pmi_w1  # fg.phi_en_en_w1.dot(fg.theta_en_en.T)

    for ft, d in fg.active_domains:
        if ft == 'en_en':
            t = d2t[ft, d]
            d_pmi = t[0, theta_en_en_names.index('pmi')]
            pot_en_en += fg.phi_en_en * d_pmi
            d_pmi_w1 = t[0, theta_en_en_names.index('pmi_w1')]
            pot_en_en_w1 += fg.phi_en_en_w1 * d_pmi_w1

    pot_en_de = fg.phi_en_de.dot(fg.theta_en_de.T)

    for ft, d in fg.active_domains:
        if ft == 'en_de':
            t = d2t[ft, d]
            pot_en_de += fg.phi_en_de.dot(t.T)

    fg.pot_en_de = np.exp(pot_en_de)
    fg.pot_en_en_w1 = np.exp(pot_en_en_w1)
    fg.pot_en_en = np.exp(pot_en_en)

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
                      en_domain, de2id, en2id, d2t):
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
                             en2id=en2id, d2t=d2t)

    fg.initialize()
    fg.treelike_inference(3)
    p = fg.get_posterior_probs()
    fgs = '\n'.join(['*SENT_ID:' + str(sent_id)] + fg.to_string())
    factor_dist = fg.to_dist()
    return [p, fgs, factor_dist]


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
              en_domain, de2id, en2id, d2t):
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
                             en2id=en2id,
                             d2t=d2t)

    fg.initialize()

    fg.treelike_inference(3)
    # sys.stderr.write('.')
    # f_en_en_theta, f_en_de_theta = fg.update_theta()
    # sys.stderr.write('|')
    if options.user_adapt or options.experience_adapt:
        g_en_en, g_en_de = fg.get_unregularized_gradeint()
        sample_ag = {}
        for f_type, d in fg.active_domains:
            g = g_en_en.copy() if f_type == 'en_en' else g_en_de.copy()
            t = domain2theta[f_type, d]
            r = fg.regularization_param
            l = fg.learning_rate
            sample_ag[f_type, d] = apply_regularization(r * 0.001, g, l, t)  # use a smaller regularization term
        g_en_en = apply_regularization(r, g_en_en, l, fg.theta_en_en)
        g_en_de = apply_regularization(r, g_en_de, l, fg.theta_en_de)
    else:
        sample_ag = None
        g_en_en, g_en_de = fg.return_gradient()
    # sys.stderr.write('+')

    p = fg.get_posterior_probs()

    return [sent_id, p, g_en_en, g_en_de, sample_ag]


def batch_sgd_accumulate(result):
    global f_en_en_theta, f_en_de_theta, n_up, prediction_probs, domain2theta
    if options.user_adapt or options.experience_adapt:
        lock.acquire()
        prediction_probs += result[1]
        f_en_en_theta += result[2]
        f_en_de_theta += result[3]
        sample_ag = result[4]
        for f_type, d in sample_ag:
            ag = sample_ag[f_type, d]
            domain2theta[f_type, d] += ag
        sys.stderr.write('*')
        lock.release()
    else:
        lock.acquire()
        prediction_probs += result[1]
        f_en_en_theta += result[2]
        f_en_de_theta += result[3]
        sys.stderr.write('*')
        lock.release()
        # print 'received', result[0], f_en_en_theta, f_en_de_theta


if __name__ == '__main__':
    random.seed(1234)
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
    opt.add_option('--user_adapt', dest='user_adapt', default=False, action='store_true')
    opt.add_option('--experience_adapt', dest='experience_adapt', default=False, action='store_true')

    (options, _) = opt.parse_args()

    if options.training_instances == '' or options.en_domain == '' or options.de_domain == '' or options.phi_pmi_w1 == '' or options.phi_pmi == '' or options.phi_ed == '' or options.phi_ped == '':
        error_msg()
        exit(1)
    elif options.save_params_file == '' and options.load_params_file == '' and options.save_predictions_file == '':
        error_msg()
        exit(1)
    else:
        pass

    print options.user_adapt, ' is the user adapt'
    print options.experience_adapt, 'is the experienence adapt'

    cpu_count = 4 if options.cpus.strip() == '' else int(options.cpus)
    print 'cpu count:', cpu_count

    domains = []
    if options.user_adapt and options.experience_adapt:
        sys.stderr.write("Currently only supports 1 type of adaptation.")
        exit(1)

    if options.user_adapt:
        try:
            domains = [d.strip() for d in codecs.open(options.training_instances + '.users').readlines()]
        except IOError:
            sys.stderr.write("user_adapt option can not find .users file.")
            exit(1)
    if options.experience_adapt:
        try:
            domains = [d.strip() for d in codecs.open(options.training_instances + '.experience').readlines()]
        except IOError:
            sys.stderr.write("experience_adapt option can not find .experience file.")
            exit(1)

    mode = 'training' if options.load_params_file == '' else 'predicting'
    if mode == 'training':
        f_en_en_names = ['pmi', 'pmi_w1']
        f_en_en_theta = np.zeros((1, len(f_en_en_names)))
        f_en_de_names = ['ed', 'ped', 'full_history', 'hit_history']
        f_en_de_theta = np.zeros((1, len(f_en_de_names)))
        domain2theta = {}
        for d in domains:
            domain2theta['en_en', d] = f_en_en_theta.copy()
            domain2theta['en_de', d] = f_en_de_theta.copy()
    else:
        ext = '.user_adapt' if options.user_adapt else ('.exp_adapt' if options.experience_adapt else '')
        print 'trying to read', options.load_params_file + ext
        een, eet, edn, edt, d2t = read_params(options.load_params_file + ext)
        save_predictions_file = options.save_predictions_file
        f_en_en_names = een
        f_en_en_theta = eet
        f_en_de_names = edn
        f_en_de_theta = edt
        domain2theta = d2t

    print 'reading in  ti and domains...'
    training_instances = codecs.open(options.training_instances).readlines()

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
    t_now = '-'.join(ctime().split())
    model_param_writer_name = options.training_instances + '.cpu' + str(cpu_count) + '.' + t_now + '.params'
    intermediate_writer = open(model_param_writer_name + '.tmp', 'w')
    if mode == 'training':
        for epoch in range(4):
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
                    en2id,
                    domain2theta), callback=batch_sgd_accumulate)
            pool.close()
            pool.join()
            print '\nepoch:', epoch
            print f_en_en_names, f_en_en_theta
            print f_en_de_names, f_en_de_theta
            print '\nprediction probs:', prediction_probs
            lr *= 0.75
            ext = '.user_adapt' if options.user_adapt else ('.exp_adapt' if options.experience_adapt else '')
            final_writer = codecs.open(options.save_params_file + ext + '.iter' + str(epoch), 'w', 'utf8')
            save_params(final_writer, f_en_en_theta, f_en_de_theta, f_en_en_names, f_en_de_names, domain2theta)
        print '\ntheta final:', f_en_en_theta, f_en_de_theta
        ext = '.user_adapt' if options.user_adapt else ('.exp_adapt' if options.experience_adapt else '')
        final_writer = codecs.open(options.save_params_file + ext, 'w', 'utf8')
        save_params(final_writer, f_en_en_theta, f_en_de_theta, f_en_en_names, f_en_de_names, domain2theta)
    else:
        print 'predicting...'
        print f_en_en_names, f_en_en_theta
        print f_en_de_names, f_en_de_theta
        prediction_str = ''
        lr = 0.05
        n_up = 0
        prediction_probs = 0.0
        ext = '.user_adapt' if options.user_adapt else ('.exp_adapt' if options.experience_adapt else '')
        final_writer = codecs.open(save_predictions_file + ext, 'w', 'utf8')
        final_dist_writer = codecs.open(save_predictions_file + ext + '.dist', 'w', 'utf8')
        for ti in training_instances:
            p, fgs, factor_dist = batch_predictions(ti,
                                                    f_en_en_names,
                                                    f_en_de_names,
                                                    f_en_en_theta,
                                                    f_en_de_theta,
                                                    phi_wrapper,
                                                    lr,
                                                    en_domain,
                                                    de2id,
                                                    en2id,
                                                    domain2theta)
            prediction_probs += p
            prediction_str = prediction_str + fgs + '\n'
            final_writer.write(fgs + '\n')
            final_writer.flush()
            final_dist_writer.write(factor_dist + '\n')
            final_dist_writer.flush()
        print '\nprediction probs:', prediction_probs, n_up
        final_dist_writer.close()
        final_writer.close()
