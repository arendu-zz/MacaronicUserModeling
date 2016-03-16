__author__ = 'arenduchintala'
import pdb
import json
import sys
import codecs

'''
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'
'''
low_users = {'', 'A10FSA4NWA7NPP', 'A1640H4RXH8NZE', 'A17J1CE7N49Z9D', 'A1BUBB41AO8TZ9', 'A1FG4M4370KNFI',
             'A1H95TGQZSN1P1', 'A1NITMHYG5YL3H', 'A1NZFJHVJ9CNTO', 'A1SZCPDUF1M409', 'A2416IYI3FQMAG', 'A24Z9RP5YZZ2TY',
             'A2HOUSLZG9KA91', 'A2VBSFSJXLZZ7A', 'A2W3A42TRMJ861', 'A3205NXKRZBL8G', 'A34KJAFLPZN4D6', 'A36662QQDZ9J3R',
             'A37NN01EVQNIIQ', 'A3D1VTLX623K6I', 'A3NBOAYT9H3LO0', 'A3OHF7VE4XO3UH', 'A8BW189UIEIJA', 'AD1WGUMVD6KED',
             'AEAPXBRB1R653', 'AOIR8V07FYMH5', 'AXMGMXPMH8YZW', 'DEMO'}


def get_results(db, q):
    db.query(q)
    res = db.use_result()
    fields = [i[0] for i in res.describe()]
    rows = []
    r = res.fetch_row()
    while r != ():
        rows.append(r[0])
        r = res.fetch_row()
    return fields, rows


if __name__ == '__main__':
    ti_file = sys.argv[1]
    keep_training_instances = []
    for idx, line in enumerate(codecs.open(ti_file, 'r', 'utf8').readlines()):
        jti = json.loads(line)
        user_id = jti['user_id'].strip()
        if user_id not in low_users:
            keep_training_instances.append(line)
        else:
            sys.stderr.write('removing line:' + str(idx) + ' with user:' + user_id + '\n')
            pass
    w = codecs.open(ti_file, 'w', 'utf8')
    w.write(''.join(keep_training_instances))
    w.flush()
    w.close()
