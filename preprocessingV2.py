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
import re

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
    (r'&', ' and ')
                                # Replacing symbols into meaningful words
]


class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s

reg_replacer = RegexpReplacer()

## Reference from VIP book
## print(replacer.replace("I should've done that thing I didn't do, but it's nice. can't is a contraction")) ## Works well

## Implement in our doc list
new_doc = [reg_replacer.replace(docs[i]) for i in range(0, cnt)]
print('Original Document')
print(docs[0])

print("Regular Expression Replaced Document as I'm changed to I am")
print(new_doc[0])

## Works well as we see I'm changed to I am

## Looking at the document we can even replace cgm, bp, US  to corresponding long form.. but do we really require..???

#############################################################################
##2. Long words replacement
## e,g looooove...love, ....... to .  So, this will create a good structure

from nltk.corpus import wordnet

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


rc_replacer = RepeatReplacer()

##################################################################################
## 3. Now, the next things is to do the spelling correction
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

################################################
#Implementing all above methods
for i in range(0,cnt):
    #Removing html address referred to url
    new_doc[i]=re.sub(r'http\S+', 'url', new_doc[i])
    new_doc[i]=re.sub(r'www\S+', 'url', new_doc[i])   #successfully replaced

print('http and www replaced to url')
print(new_doc[0])
##############################################

for i in range(0,cnt):
    #separating into words based on white space
    new_doc[i]= re.findall(r'\w+', new_doc[i])

print('After separating into individual words')
print(new_doc[0])

############################################
for i in range(0,cnt):
    #replacing looove to love
    new_doc[i] = [rc_replacer.replace(words) for words in new_doc[i]]
print('Replacing repeated character as seen twooo to two')
print(new_doc[0])
#################################################
for i in range(0,cnt):
    #checking and correcting spelling
    new_doc[i] = [sp_replacer.replace(words) for words in new_doc[i]]

print('Replacing wrong spelling as seen wlil to will')
print(new_doc[0])

## issues...works well as expected..but seen that apps...changed to pas Gloko changed to Gloom
# So, need of customization of dictionary in wordnet.

## Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

for i in range(0,cnt):
    # lemmatization
    new_doc[i] = [lemmatizer.lemmatize(words) for words in new_doc[i]]

print('Doc after lemmatization')
print(new_doc[0])   # seems working good as companies changed to company

## POS tagging
import nltk
for i in range(0,cnt):
    new_doc[i]=nltk.pos_tag(new_doc[i])
print('Parts of sppech tagged')
print(new_doc[150][0])

## Chunking and chinking


#So, if we thing the cleaning and preprocessing process is completed, we can merge to make a list of document as original docs
#######
# cleaned_doc=[]
# def list_sent(sentence, sent_str):
#     for i in sentence:
#         sent_str += str(i) + " "
#     sent_str = sent_str[:-1]

# for i in range(0,cnt):
#     print(' '.join(word for word in new_doc[i])
#
#
#  print(' '.join(word for word in new_doc[0]))
#
# print(new_doc[0][507])
#
# from nltk.tag import pos_tag
#
# sentence = "Michael Jackson likes to eat at McDonalds")
# tagged_sent = nltk.pos_tag(sentence.split())
#     # [('Michael', 'NNP'), ('Jackson', 'NNP'), ('likes', 'VBZ'), ('to', 'TO'), ('eat', 'VB'), ('at', 'IN'), ('McDonalds', 'NNP')]
#
# propernouns = [word for word, pos in tagged_sent if pos == 'NNP']
#     # ['Michael','Jackson', 'McDonalds']
#
# import pandas as pd
# class LemmaTokenizer(object):
#     def __init__(self):
#         self.wnl = WordNetLemmatizer()
#     def __call__(self, doc):
#         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
#
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
# vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words = 'english')
# X = vectorizer.fit_transform(new_doc[0])
# df_tf = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
# print(df_tf)

