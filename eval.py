__author__ = 'arenduchintala'
import sys
import codecs
from optparse import OptionParser

reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'

if __name__ == '__main__':
    opt = OptionParser()
    # insert options here
    opt.add_option('-p', dest='prediction_file', default='')
    (options, _) = opt.parse_args()

    if options.prediction_file.strip() == '':
        sys.stderr.write('Usage: python eval.py -p [prediction file]\n')
        exit(1)
    else:
        pass
    correct_at_0 = 0
    correct_at_5 = 0
    correct_at_10 = 0
    correct_at_15 = 0
    correct_at_25= 0
    correct_at_50 = 0
    correct_at_100 = 0
    total = 0
    model_correct_on = {}
    model_most_wrong_on = []
    for line in codecs.open(options.prediction_file, 'r', 'utf8').readlines():
        items = line.split()
        if len(items) < 15:
            continue
        de_word = items[:1][0]
        words = [i for idx, i in enumerate(items[1:]) if idx % 2 == 0]
        probs = [i for idx, i in enumerate(items[1:]) if idx % 2 != 0]
        label = words[0]
        label_prob = probs[0]
        guesses = words[1:]
        guesses_prob = probs[1:]
        if label in guesses:
            mco = model_correct_on.get(label, 0)
            mco += 1
            model_correct_on[label] = mco
            correct_at_15 += 1
            pos = guesses.index(label)
            if pos <= 10:
                correct_at_10 += 1
            if pos <= 5:
                correct_at_5 += 1
            if pos == 0:
                correct_at_0 += 1
        else:
            d = abs(float(label_prob) - float(guesses_prob[0]))
            model_most_wrong_on.append((d, label, guesses[0]))
        total += 1
    print 'report'
    print 'prec at 0 ', float(correct_at_0) * 100 / total
    print 'prec at 5 ', float(correct_at_5) * 100 / total
    print 'prec at 10', float(correct_at_10) * 100 / total
    print 'prec at 15', float(correct_at_15) * 100 / total
    tups = sorted([(c, l) for l, c in model_correct_on.iteritems()], reverse=True)
    most_correct = [l + ',' + str(c) for c, l in tups]
    print 'most correct  :', ' '.join(most_correct[:20])
    most_wrong = sorted(model_most_wrong_on, reverse=True)
    most_wrong = [label + ',' + guesses + ',' + str(d) for d, label, guesses in most_wrong]
    print 'most incorrect:', ' '.join(most_wrong[:20])
