# The following code
from nltk.corpus import wordnet
import re

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

