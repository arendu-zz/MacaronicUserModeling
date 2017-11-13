#!/usr/bin/env python
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
    #insert options here
    opt.add_option('-a', dest='predictions', default='')
    (options, _) = opt.parse_args()
    if options.predictions == '':
        sys.stderr.write('Usage: python -a [prediction file]\n')
        exit(1)
    else:
        pass
    lines = codecs.open(options.predictions, 'r', 'utf8').readlines()
    total_guesses = 0
    correct_at_0 = 0
    correct_at_25 = 0
    correct_at_50 = 0
    for line in lines:
        items = line.split()
        if len(items) > 100:
            de_word = items[0]
            user_guess = items[1]
            model_prob_for_user_guess = items[2]
            preds = [i for idx,i in enumerate(items[3:]) if idx % 2 == 0] 
            preds_probs = [float(i) for idx,i in enumerate(items[3:]) if idx % 2 == 1] 
            if user_guess == preds[0]:
                correct_at_25 += 1
                correct_at_50 +=1
                correct_at_0 += 1
            elif user_guess in preds[:25]:
                correct_at_50 +=1
                correct_at_25 += 1
            elif user_guess in preds:
                correct_at_50 +=1
            else:
                pass
            total_guesses +=1

    print correct_at_0, correct_at_25, correct_at_50, total_guesses
    print float(correct_at_0) / float(total_guesses)
    print float(correct_at_25) / float(total_guesses)
    print float(correct_at_50) / float(total_guesses)
