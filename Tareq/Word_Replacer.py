# The following code
from nltk.corpus import wordnet
import re

import enchant
from nltk.metrics import edit_distance


class SpellingReplacer(object):
    def __init__(self, dict_name='en', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist

    def replace(self, word):
        if self.spell_dict.check(word):
            return word

        suggestions = self.spell_dict.suggest(word)
        if suggestions and edit_distance(word, suggestions[0]) <=self.max_dist:
            return suggestions[0]
        else:
            return word


replacement_patterns = [
(r'won\'t', 'will not'),
(r'can\'t', 'can not'),
(r'i\'m', 'i am'),
(r'ain\'t', 'is not'),
(r'(\w+)\'ll', '\g<1> will'),
(r'(\w+)n\'t', '\g<1> not'),
(r'(\w+)\'ve', '\g<1> have'),
(r'(\w+)\'s', '\g<1> is'),
(r'(\w+)\'re', '\g<1> are'),
(r'(\w+)\'d', '\g<1> would')
]
class RegexpReplacer(object):
	def __init__(self, patterns=replacement_patterns):
		self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
	def replace(self, text):
		s = text
		for (pattern, repl) in self.patterns:
			s = re.sub(pattern, repl, s)
		return s

replacer = SpellingReplacer()
print (replacer.replace('cookbok'))  #cookbook
	
	
class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word

replacer = RegexpReplacer()

print ("I should've done that thing I didn't do, but it's nice. can't is a contraction")
print(replacer.replace("I should've done that thing I didn't do, but it's nice. can't is a contraction"))
'I should have done that thing I did not do, but it is nice.  can not is a contraction'
print ('-----------------------------------------------')
replacer = RepeatReplacer()
print (replacer.replace('loooooveee'))
'love'
print(replacer.replace('oooooh'))
'oh'
print(replacer.replace('goose'))
'gose'
print(replacer.replace('googleee'))
'google'

---
    cooked_parsed = doc.lower().strip().replace("\n", " ").replace(".", " ").replace("-", ' ')
    # cooked_parsed = re.sub(r'[^\w\s]', '', cooked_parsed).replace("  "," ")
    # # # Remove HTML tags:
    cooked_parsed = re.sub('<[^<]+?>', '', cooked_parsed)
    # Remove punctuation
    normalized = result = re.sub(r"http\S+", " ", cooked_parsed)
    normalized = result = re.sub(r"than\S+", "", normalized)
    normalized = result = re.sub(r"@\S+", "", normalized)
    normalized = re.sub(r'[^\w\s]', '', normalized).replace("  ", " ")
    normalized=re.sub(r'[^a-zA-Z5\s]+', '', normalized)
    # Standardize words (remove multiple letters):
    normalized = ''.join(''.join(s)[:2] for _, s in itertools.groupby(normalized))
    # normalized = TextBlob(normalized)
    # normalized=' '.join(normalized.noun_phrases)
    stop_free = " ".join([i for i in normalized.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    # normalized = normalized.lower().strip().replace("\n", " ").replace(".", " ").replace("-", ' ')
---
