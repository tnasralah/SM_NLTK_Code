#####################################################################
from nltk.metrics import edit_distance
import enchant


class SpellingReplacer(object):
    def __init__(self, dict_name = 'en_GB', max_dist = 2):
        self.spell_dict = enchant.Dict(dict_name)
        #print(list(enchant.Dict(dict_name)))
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

word_list = ['sugrrr', 'sugr', 'glooko', 'likethat', 'llikethat']
print(spell_check(word_list))

