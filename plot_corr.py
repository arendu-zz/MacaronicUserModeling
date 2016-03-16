__author__ = 'arenduchintala'
import sys
import codecs
from optparse import OptionParser
import json
import gensim
from gensim.models.word2vec import Word2Vec
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt
import matplotlib
import pdb

'''reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'
'''


def cosine_sim(v1, v2):
    return np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))


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
    return vec


if __name__ == '__main__':
    opt = OptionParser()
    # insert options here
    opt.add_option('--vocab', dest='vocab_file', default='')
    opt.add_option('--dist', dest='dist_file', default='')
    opt.add_option('--vec', dest='word2vec_model', default='')
    (options, _) = opt.parse_args()

    if options.dist_file == '' or options.vocab_file == '' or options.word2vec_model == '':
        sys.stderr.write(
            'Usage: python --vocab [vocab file] --vec [word2vec model] --dist [prediction distribution file]\n')
        exit(1)
    else:
        pass

    w2v_model = Word2Vec.load(options.word2vec_model)
    dist_lines = codecs.open(options.dist_file, 'r', 'utf8').readlines()

    vocabs = [v.strip() for v in codecs.open(options.vocab_file, 'r', 'utf8').readlines()]
    vocab2id = dict((v, idx) for idx, v in enumerate(vocabs))
    id2vector = dict((idx, get_vec(w2v_model, v)) for idx, v in enumerate(vocabs))
    vocab_vecs = [get_vec(w2v_model, v) for v in vocabs]

    guess_correct = 0
    guess_total = 0
    expected_pg2truth = []
    ug2truth = []

    for dist_line in dist_lines:
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

            cos_u2t = cosine_sim(weighted_mean_vec(w_tl, v_tl), weighted_mean_vec(w_ug, v_ug))

            w_pg = np.exp(pred_dist)
            # w_pg_top = np.argpartition(w_pg, -1)[-1:]
            w_pg_bottom = np.argpartition(w_pg, len(vocabs) - 1)[:len(vocabs) - 1]
            w_pg[w_pg_bottom] = 0.0
            w_pg = renormalize(w_pg)
            v_pg = vocab_vecs
            cos_pg2t = cosine_sim(weighted_mean_vec(w_tl, v_tl), weighted_mean_vec(w_pg, v_pg))

            expected_pg2truth.append(cos_pg2t)
            ug2truth.append(cos_u2t)
        except:
            pass
    p_corrcoef, p_pval = pearsonr(expected_pg2truth, ug2truth)
    s_corrcoef, s_pval = spearmanr(expected_pg2truth, ug2truth)
    plt.scatter(expected_pg2truth, ug2truth)
    plt.xlabel('sim(pg,t)')
    plt.ylabel('sim(lg,t)')
    plt.title('Cosine Similarity Relation')
    plt.savefig(options.dist_file + '.plot.best.png')
    print p_corrcoef, p_pval
    print s_corrcoef, s_pval

    '''
    0.335967 1.05945535069e-222  for expected
    0.287609642002 3.9861983184e-161

    0.36013 5.92051261798e-258 for best
    0.261565189546 1.2960524574e-132

    '''
    # heatmap, xedges, yedges = np.histogram2d(expected_pg2truth, ug2truth, bins=100)
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # plt.clf()
    # plt.imsave(heatmap.T, extent=extent, origin='lower')
    # plt.savefig(options.dist_file + '.heatmap.png')
