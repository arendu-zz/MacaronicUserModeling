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
    #opt.add_option('-e', dest='example_option', default='example default')
    (options, _) = opt.parse_args()
    print options
    pass