########################################################################
# From pyenchant package spell checker
#####################################################################
from nltk.metrics import edit_distance
import enchant


class SpellingReplacer(object):
    def __init__(self, dict_name = 'en_GB', max_dist = 2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = 2

    def replace(self, word):
        if self.spell_dict.check(word):
            return word
        suggestions = self.spell_dict.suggest(word)

        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word


##################
def spell_check(word_list):
    checked_list = []
    for item in word_list:
        replacer = SpellingReplacer()
        r = replacer.replace(item)
        checked_list.append(r)
    return checked_list


# From Dr.Omar Link
#################################################

def reduce_lengthening_single(word):
    pattern = re.compile(r"(.)\1{2,}")
    r = pattern.sub(r"\1\1", word)
    return r


def reduce_lengthening_list(word_list):
    checked_list = []
    for item in word_list:
        pattern = re.compile(r"(.)\1{2,}")
        r = pattern.sub(r"\1\1", item)
        checked_list.append(r)
    return checked_list


# from pattern.en import spelling
# pattern not working in my case

word = "amazzziiing"
word_wlf = reduce_lengthening_list(word) #calling function defined above
# print (word_wlf) #word lengthening isn't being able to fix it completely

# correct_word = spelling(word_wlf)
# print (correct_word)


#################################3

from collections import Counter


def words(text): return re.findall(r'\w+', text.lower())


WORDS = Counter(words(open("J:\\DSU\\CITI\\big_text.txt").read()))


def P(word, N=sum(WORDS.values())):
    # "Probability of `word`."
    return WORDS[word] / N


def correction(word_list):
    # "Most probable spelling correction for word."
    checked_list = []
    for item in word_list:
        r = max(candidates(item), key=P)
        checked_list.append(r)
    return checked_list


def candidates(word):
    # "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    # "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


word_list = ['sugrrr', 'sugr', 'glooko', 'likethat', 'llikethat', 'aweeesooome', 'teh' , 'Dibetes', 'monitoring', 'hospitl',
             'Diaxcom','booold','ttoday', 'helpfulIn', 'processMy', 'Hangover', 'applewatch', 'promotioncode', 'NewYork', 'Johnson',
             'WashingtonDC', 'bottleneck', 'misinterpretation', 'conception', 'do-or-die', 'challengng','understand','completlly','posssible','controlaction',
             'hypractive', 'CGM', 'multple', 'Apple-iphone', 'reaction', 'positve', 'compound', 'Systemanalysis', 'device','detrimntal',
             'infomation', 'eitherway', 'splitted', 'Oftentimes', 'informationAfter' ,'annnual', 'processess' ,'goooooood', 'glucoose', 'worldcalss']
print(type(word_list))
print('Normal Pyenchant NLTK')
print(spell_check(word_list))
print('Dr. Omar Link')
print (reduce_lengthening_list( word_list))
print('Peter Norvigs code')  # Most pop. spell checkers dealing with split, delete, transpose, inserts, replace
print(correction(word_list))
print('Auto-correct code')

from autocorrect import spell
auto_list = []

for w in word_list:
    auto_list.append(spell(w))
print(auto_list)

# One more cleaning strategy with regular expression : (There are many words as in line in our text document, So
# we try to split them )
import re
# line = 'appleWatch applicationMy generatedMany' is converted into apple Watch, application My,  generated Many
print('Splitting from regular expression')
new_word_list = []
print(word_list[0])
for item in word_list:
    result = re.sub('(?<=[a-z])(?=[A-Z][a-z])', '\t', item,)
    new_word_list.append(result.split())
print (new_word_list)

# the above code does not deal with things as applewatch so we tried to implement using compound word splitter
# It split successfully to the compound word as likethat but Dibetes split into Di bet es as well
import splitter
print('Splitting using Splitter')
new_list = []
for item in word_list:
    words = splitter.split(item)
    if words == '':
        words = item
    new_list.append(words)
print(new_list)