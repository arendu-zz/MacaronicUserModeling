__author__ = 'arenduchintala'
from editdistance import EditDistance
import enchant
import sys
import pdb


class TrainingInstance(dict):
    def __init__(self,
                 user_id,
                 past_correct_guesses,
                 past_sentences_seen,
                 past_guesses_for_current_sent,
                 current_sent,
                 current_revealed_guesses,
                 current_guesses):
        dict.__init__(self)
        self.__dict__ = self
        # these are the inputs
        self.user_id = user_id
        self.past_correct_guesses = past_correct_guesses
        self.past_sentences_seen = past_sentences_seen
        self.past_guesses_for_current_sent = past_guesses_for_current_sent
        self.current_sent = current_sent
        self.current_revealed_guesses = current_revealed_guesses
        # this is the output
        self.current_guesses = current_guesses

    @staticmethod
    def from_dict(_dict):
        ti = TrainingInstance(user_id=_dict['user_id'],
                              past_correct_guesses=list(map(Guess.from_dict, _dict['past_correct_guesses'])),
                              past_sentences_seen=_dict['past_sentences_seen'],
                              past_guesses_for_current_sent=list(
                                  map(Guess.from_dict, _dict['past_guesses_for_current_sent'])),
                              current_sent=list(map(SimpleNode.from_dict, _dict['current_sent'])),
                              current_revealed_guesses=list(map(Guess.from_dict, _dict['current_revealed_guesses'])),
                              current_guesses=list(map(Guess.from_dict, _dict['current_guesses'])))
        return ti


def check_ignore_case(w, d):
    if d.check(w) or w in ['__unk__', '__copy__', '__blank__', '*', '&quot;']:
        return True
    else:
        s_lower = [i.lower().replace("'", "") for i in d.suggest(w)]
        if len(s_lower) == 0:
            return False
        else:
            if w.lower() in s_lower:
                return True
            else:
                return False


def copy_or_not(w, l2, english_d, ed):
    '''if w.lower() == u'dienste':
        pdb.set_trace()'''
    w = w.lower()
    l2 = l2.lower()
    if w == l2:
        return '__copy__', l2, 'de'
    else:
        pass

    if w.isdigit():
        best_suggestion = []
    else:
        try:
            lower_sugguestions = [s.lower() for s in english_d.suggest(w)]
            best_suggestion = sorted([(ed.editdistance_simple(s, w)[0], s) for s in lower_sugguestions])[0][1]
        except IndexError:
            sys.stderr.write('no suggestions for: ' + w + ',' + l2 + ',' + ' '.join(lower_sugguestions) + '\n')
            return '__copy__', l2, 'de'
        best_suggestion = sorted([(len(bs), bs) for bs in best_suggestion.split()])[-1][1]

    sys.stderr.write('ed for ' + w + '\n')
    if len(best_suggestion) == 0:
        sys.stderr.write('german ed was lower, guess:' + w + ',en correction:[]' + ', german context:' + l2 + '\n')
        return '__copy__', l2, 'de'
    else:
        ed2_en = ed.editdistance_simple(w, best_suggestion)[0]
        ed2_de = ed.editdistance_simple(w, l2)[0]
        if ed2_de >= ed2_en:
            sys.stderr.write(
                'english ed was lower, guess:' + w + ', correction:' + best_suggestion + ', german context:' + l2 + '\n')
            return best_suggestion, best_suggestion, 'en'
        else:
            sys.stderr.write(
                'german ed was lower, guess:' + w + ',en correction:' + best_suggestion + ', german context:' + l2 + '\n')
            return '__copy__', l2, 'de'


class Guess(dict):
    def __init__(self, id, guess, revealed, l2_word, reference=None):
        dict.__init__(self)
        self.__dict__ = self
        if guess.strip() == '':
            self.guess = '__blank__'
        elif guess.strip().lower() == '__blank__' or guess.strip().lower() == '__unk__' or guess.strip().lower() == '__copy__':
            self.guess = guess
        else:
            if len(guess.split()) > 1:
                sys.stderr.write("phrasal guess:" + guess + '\n')
            guess = sorted([(len(g), g) for g in guess.split()])[-1][1]
            guess = guess[:-1] if guess[-1] == '*' and len(guess) > 1 else guess
            if True:  # the or True condition means at this stage we accept what ever the make_training_instance.py file has generated.
                self.guess = guess.lower().replace("'", "")
            else:
                pass
        self.l2_word = l2_word
        self.id = id
        self.revealed = revealed
        self.reference  = reference

    def copy(self, new_id=None):
        if new_id:
            g = Guess(id=new_id, guess=self.guess, revealed=self.revealed, l2_word=self.l2_word)
        else:
            g = Guess(id=self.id, guess=self.guess, revealed=self.revealed, l2_word=self.l2_word)
        return g

    def __str__(self):
        return ','.join([str(self.id), str(self.guess), str(self.revealed)])

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __cmp__(self, other):
        if str(self) == str(other):
            return 0
        elif str(self) > str(other):
            return 1
        else:
            return -1

    @staticmethod
    def from_dict(_dict):
        g = Guess(id=tuple(_dict['id']),
                  guess=_dict['guess'],
                  revealed=_dict['revealed'],
                  reference = _dict.get('reference', None),
                  l2_word=_dict['l2_word'])
        return g


class SimpleNode(dict):
    def __init__(self, sent_id, id, l2_word, position, lang, l1_parent):
        dict.__init__(self)
        self.__dict__ = self
        self.sent_id = sent_id
        self.id = id
        if lang == 'en':
            self.l2_word = l2_word.lower().replace("'", "")
        else:
            self.l2_word = l2_word
        if lang == 'de':
            self.l1_parent = l1_parent.lower().replace("'", "")
        else:
            self.l1_parent = l1_parent
        self.position = int(position)
        self.lang = lang

    def __cmp__(self, other):
        if self.position == other.position:
            return 0
        elif self.position > other.position:
            return 1
        else:
            return -1

    @staticmethod
    def from_dict(_dict):
        s = SimpleNode(sent_id=_dict['sent_id'],
                       id=tuple(_dict['id']),
                       l2_word=_dict['l2_word'],
                       l1_parent=_dict['l1_parent'],
                       position=_dict['position'],
                       lang=_dict['lang'])
        return s
