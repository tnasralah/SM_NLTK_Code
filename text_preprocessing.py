## Text Data Cleaning

## Importing the discussion as the list from local directory
docs=[]
filepath="J:/DSU/CITI/s_discussions_DT.txt"
cnt=0
with open(filepath) as f:
    for line in f:
        docs.append(line)
        cnt = cnt+ 1
        print (line)
print (cnt)

print(type(docs))
# So, looking at the output, there is 239 of the discussions in the list
## So, for the text preprocessing and cleaning we carried out the following steps:

###################################################################################################
## 1. Correcting the words (haven't....have not, I'm...I am etc.)
## Some replacement patterns created, we can add further if our requirement increases
replacement_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'can not'),
    (r'i\'m', 'i am'),
    (r'I\'m', 'I am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would'),
    (r'&', ' and '),
    (r'/', ' or ')       # Replacing symbols into meaningful words
]


class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s

replacer = RegexpReplacer()

## Reference from VIP book
## print(replacer.replace("I should've done that thing I didn't do, but it's nice. can't is a contraction")) ## Works well

## Implement in our doc list
new_doc = [replacer.replace(docs[i]) for i in range(0, cnt)]
print(new_doc)
## Works well as we see I'am changed to I am and & changed to and in doc 13

## Looking at the document we can even replace cgm, bp, US  to corresponding long form.. but do we really require..???

#############################################################################
##2. Long words replacement
## e,g looooove...love, ....... to .  So, this will create a good structure
import re
from nltk.corpus import wordnet

print(wordnet.synsets('ball')) ## Looks like there is synset for ball
print(wordnet.synsets('mall')) ## cannot is empty

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


replacer = RepeatReplacer()

print(replacer.replace('ball'))
print(replacer.replace('The ball is on the floooor')) ## works well for ball # infact works well like this word by word...
                                ## So, the problem is if we carry this before tokenizing, it read a list as a text and nothing is matched to
                                ## wordnet, so every doulbe occurance is removed will..wil, Apple..Aple, hopefully..hopefuly etc.
## challenges to handle this
## Looks like to correct some of structure like looove, we are doing it at greater expense.... :(



# Word tokenization for every documents into words..
# Rather than tokenizing, this below approach works pretty well and also do spelling correction on the flow
words=[]


for i in range(0,cnt):
    # Removing html address referred
    new_doc[i]=re.sub(r'http\S+', 'url', new_doc[i])
    new_doc[i]=re.sub(r'www\S+', 'url', new_doc[i])   #successfully replaced
    # separating into words based on white space
    new_doc[i]= re.findall(r'\w+', new_doc[i])
    # replacing looove to love
    new_doc[i] = [replacer.replace(words) for words in new_doc[i]]
    # checking and correcting spelling
    new_doc[i] = [sp_replacer.replace(words) for words in new_doc[i]]
## Working well poooor replaced to por but also small to smal, cannot to canot

##################################################################################
## 3. Now, the next things is to do the spelling corrrection
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
        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word

sp_replacer = SpellingReplacer()
print(sp_replacer.replace('cookbok'))  # cookbook
print (sp_replacer.replace('hsa'))     # has
print (sp_replacer.replace('anynoe'))  # anyone     ## Works well
print (sp_replacer.replace('anynoe hsa cookbok'))  ## But cannot work as this string is not found in dictionary..


######
from autocorrect import spell  # This module looks simpler to implement than pyenchant
print(spell('jmups'))          # Works
print(spell('Dog jmups'))       # Doesn't works so, same problem


## Implementing in our document
# Already implemented along with  2

###########################################
# 4. Replacing of https:// or www. to word referred as 'url'
# Implemented in 2 before repeat eliminator and spell checkers

##########################################
# 5.