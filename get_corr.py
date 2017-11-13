#!/usr/bin/env python 
__author__ = 'arenduchintala'
import sys
import codecs
from optparse import OptionParser
import numpy as np
#from gensim.models.word2vec import Word2Vec
from scipy.stats.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors
import random

reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'


def cosine_sim(v1, v2):
    cs = np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
    return cs


def renormalize(w):
    return w / np.sum(w)


def weighted_mean_vec(ws, vs):
    assert isinstance(vs, list)
    assert isinstance(ws, np.ndarray)
    try:
        assert abs(np.sum(ws) - 1.0) < 1e-4
    except AssertionError:
        print 'sum', np.sum(ws)
        print 'size', np.size(ws)
        print 'len vecs', len(vs)
        exit(1)
    mv = np.zeros_like(vs[0])
    for w, v in zip(ws, vs):
        mv += w * v
    return mv


def get_vec(model, word):
    try:
        vec = model[word]
    except KeyError:
        sys.stderr.write('no vec for ' + word + ' using <unk> \n')
        vec = model['__unk__']
        vec = np.random.rand(50)
        #vec = rare.copy()
    return vec


if __name__ == '__main__':
    np.random.seed(33334)
    opt = OptionParser()
    # insert options here
    opt.add_option('--vocab_en', dest='vocab_file', default='')
    opt.add_option('--dist', dest='dist_file', default='')
    opt.add_option('--glove', dest='glove_file', default='')
    opt.add_option('--top', dest='get_top', default=False, action='store_true')
    (options, _) = opt.parse_args()

    if options.dist_file == '' or options.vocab_file == '' or options.glove_file == '':
        sys.stderr.write(
            'Usage: python --vocab_en [vocab file] --glove [glove file] --dist [prediction distribution file]\n')
        exit(1)
    else:
        pass

    dist_lines = codecs.open(options.dist_file, 'r', 'utf8').readlines()
    #w2v_model = Word2Vec.load(options.word2vec_model)
    w2v_model = dict(
        (items.split()[0].strip(), np.array([float(n) for n in items.split()[1:]])) for items in
        open(options.glove_file).readlines())
    vocabs = [v.strip() for v in codecs.open(options.vocab_file, 'r', 'utf8').readlines()]
    vocab2id = dict((v, idx) for idx, v in enumerate(vocabs))
    id2vector = dict((idx, get_vec(w2v_model, v)) for idx, v in enumerate(vocabs))
    vocab_vecs = [get_vec(w2v_model, v) for v in vocabs]
    writer = codecs.open(options.dist_file + '.summary', 'w', 'utf8')
    guess_correct = 0
    guess_total = 0
    expected_pg2truth = []
    ug2truth = []
    wrongs = []
    for dist_line in dist_lines[:]:
        try:
            truth, user_guess, pred_dist = dist_line.strip().split('|||')
            pred_dist = np.array([float(p) for p in pred_dist.split()])
            tl = truth.split()
            w_tl = np.ones((len(tl),)) * (1.0 / len(tl))  # uniformly weight truth word vectors
            i_tl = [vocab2id[v.strip()] for v in tl]  # index of each truth word
            v_tl = [id2vector[idx] for idx in i_tl]  # vector of each truth word

            ugl = user_guess.split()
            w_ug = np.ones((len(ugl),)) * (1.0 / len(ugl))
            i_ug = [vocab2id[v.strip()] for v in ugl]  # index of each user guess
            v_ug = [id2vector[idx] for idx in i_ug]  # vector of each user guess
            if truth.strip() == user_guess.strip():
                cos_u2t = 1.0
            else:
                cos_u2t = cosine_sim(weighted_mean_vec(w_tl, v_tl), weighted_mean_vec(w_ug, v_ug))

            if not options.get_top:
                w_pg = np.exp(pred_dist)
            else:
                w_pg = np.exp(pred_dist)
                w_pg_top = np.argpartition(w_pg, -1)[-1:]
                w_pg_bottom = np.argpartition(w_pg, len(vocabs) - 1)[:len(vocabs) - 1]
                w_pg[w_pg_bottom] = 0.0
            w_pg = renormalize(w_pg)
            v_pg = vocab_vecs
            cos_pg2t = cosine_sim(weighted_mean_vec(w_tl, v_tl), weighted_mean_vec(w_pg, v_pg))

            expected_pg2truth.append(cos_pg2t)
            ug2truth.append(cos_u2t)
            if cos_u2t == 1.0 and cos_pg2t < 0.6:
                top_pgs = [vocabs[idx] for idx in np.argpartition(np.exp(pred_dist), -10)[-10:]]
                wrongs.append(truth + ' ||| ' + user_guess + ' ||| ' + ' '.join(top_pgs))
        except:
            sys.stderr.write('error in:' + truth + user_guess)
    print 'done'
    p_corrcoef, p_pval = pearsonr(expected_pg2truth, ug2truth)
    s_corrcoef, s_pval = spearmanr(expected_pg2truth, ug2truth)
    '''
    plt.scatter(expected_pg2truth, ug2truth, alpha=0.01)
    plt.xlabel('Predicted Quality (expected)')
    plt.ylabel('Actual Quality')
    plt.title('Cosine Similarity Relation')
    plt.savefig(options.dist_file + '.plot.png')
    plt.clf()
    plt.set_cmap('YlOrRd')
    plt.hist2d(expected_pg2truth, ug2truth, bins=100, norm=LogNorm())
    plt.colorbar()
    plt.xlabel('Predicted Quality (expected)', sizze=10)
    plt.ylabel('Actual Quality')
    plt.title('Cosine Similarity Relation')
    plt.show()
    plt.savefig(options.dist_file + '.hist.png')
    '''
    print '\n'.join(wrongs)
    print 'pearsons:', p_corrcoef, p_pval
    print 'spearmans', s_corrcoef, s_pval

    '''
    0.460724729798 0.0  for expected
    0.472198720725 0.0

    0.302539571376 6.57924022781e-179 for best
    0.341276038753 3.51095361267e-230

    0.411618362509 0.0 for top 10
    0.43867110401 0.0

    '''
