# -*- coding: utf-8 -*-
__author__ = 'arenduchintala'
import sys
import codecs
from optparse import OptionParser
import string
from ed import edsimple

reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'


def fmt_align(i, o):
    i_f = None
    o_f = None
    if i in string.ascii_letters or i == '<eps>':
        i_f = i
    else:
        i_f = '__SYM__'
    if o in string.ascii_letters or o == '<eps>':
        o_f = o
    else:
        o_f = '__SYM__'
    return i_f, o_f


if __name__ == '__main__':
    opt = OptionParser()
    opt.add_option('--syn0', dest='output_word_vecs', default='')
    opt.add_option('--syn1', dest='context_word_vecs', default='')
    opt.add_option('--ev', dest='en_vocab', default='')
    opt.add_option('--dv', dest='de_vocab', default='')
    (options, _) = opt.parse_args()

    # todo: make real feature vals for en-en
    import numpy as np

    n = np.zeros((1306, 471))
    for d_idx, d in enumerate(codecs.open(options.de_vocab, 'r', 'utf8').readlines()):
        for e_idx, e in enumerate(codecs.open(options.en_vocab, 'r', 'utf8').readlines()):
            ed_score, ed_alignments = edsimple(d, e)
            ed_score = float(ed_score) / float(max(len(d), len(e)))
            ed_alignments_fix = [fmt_align(i, o) for i, o in ed_alignments]
            n[e_idx, d_idx] = ed_score
    np.savetxt('toy.en-de.ed.feature.vals', n, fmt='%0.4f')
