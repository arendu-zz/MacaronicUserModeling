__author__ = 'arenduchintala'
import sys
import codecs
from optparse import OptionParser
import json
from training_classes import TrainingInstance, Guess, SimpleNode

reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'

if __name__ == '__main__':
    opt = OptionParser()
    # insert options here
    opt.add_option('--ti', dest='training_instances', default='')
    (options, _) = opt.parse_args()

    if options.training_instances == '':
        sys.stderr.write('Usage: python --ti [training instances file] \n')
        exit(1)
    else:
        pass

    training_instances = codecs.open(options.training_instances).readlines()
    for t_idx, training_instance in enumerate(training_instances):
        j_ti = json.loads(training_instance)
        ti = TrainingInstance.from_dict(j_ti)
        print ti.current_sent
