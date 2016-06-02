#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import codecs
from optparse import OptionParser
import json
from training_classes import TrainingInstance, SimpleNode

reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'

if __name__ == '__main__':
    opt = OptionParser()
    # insert options here
    opt.add_option('--ti', dest='instance_file', default='')
    opt.add_option('--sent', dest='sent_file', default='')
    (options, _) = opt.parse_args()
    if options.instance_file == '' or options.sent_file == '':
        sys.stderr.write("Usage: python get_instance_tf.py "
                         "--sent [id2sent file] "
                         "--ti [instance file]\n")
        exit(1)
    else:
        pass
    sentid2words = {}
    l2_vocab = set([])
    for sent in codecs.open(options.sent_file, 'r', 'utf8').readlines():
        s = sent.split()
        words = [i.lower() for i in s[1:]]
        sentid2words[int(s[0])] = words
        l2_vocab.update(words)

    for training_instance in codecs.open(options.instance_file, 'r', 'utf8').readlines():
        j_ti = json.loads(training_instance)
        ti = TrainingInstance.from_dict(j_ti)
        w2c = {}
        for p in ti.past_sentences_seen:
            for w in sentid2words[int(p) + 50]:  # the plus 50 is to make sent_id in sent_file match those in ti_file
                w2c[w] = w2c.get(w, 0) + 1
        for sn in ti.current_sent:
            if sn.lang == 'de':
                w = sn.l2_word.lower()
                w2c[w] = w2c.get(w, 0) + 1
            else:
                pass
        c_sum = 0
        alpha = 0.0
        for w, c in w2c.iteritems():
            c_sum += (c + alpha)

        for v in l2_vocab:
            w2c[v] = float(w2c.get(v, 0) + alpha)  / float(c_sum)
        output = []
        for w, c in sorted(w2c.iteritems()):
            output.append(w + '\t' + '%.6f' % c)
        print '\t'.join(output)
