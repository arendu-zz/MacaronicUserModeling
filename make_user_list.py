__author__ = 'arenduchintala'
import sys
import codecs
from optparse import OptionParser
import json

reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'

if __name__ == '__main__':
    opt = OptionParser()
    # insert options here
    opt.add_option('--ti', dest='ti_file', default='mturk-data/ti.50')
    (options, _) = opt.parse_args()
    if options.ti_file.strip() == '':
        sys.stderr.write('Usage: python make_user_list.py --ti [user list file]\n')
        exit(1)
    else:
        pass
    delete_lines = []
    user_list = []
    replacement = []
    for idx, ti in enumerate(codecs.open(options.ti_file, 'r', 'utf8').readlines()):
        j_ti = json.loads(ti)
        user_id = j_ti['user_id']
        if user_id.strip() == '' or user_id.lower() == 'demo':
            delete_lines.append(idx)
        else:
            replacement.append(ti)
            user_list.append(user_id)
    w = codecs.open(options.ti_file, 'w')
    w.write(''.join(replacement))
    w.flush()
    w.close()
    w = codecs.open(options.ti_file + '.users', 'w')
    user_list = sorted(list(set(user_list)))
    w.write('\n'.join(user_list))
    w.flush()
    w.close()
