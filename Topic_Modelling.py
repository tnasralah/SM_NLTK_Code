import gensim
import pandas as pd
import numpy as np
from gensim.models import LdaModel, LsiModel
from sklearn.feature_extraction.text import TfidfVectorizer

### This file basically convert the discussions into the bag of words after the clean data is provided as the input
## So, we can implement with the new_str in our case

docs=[]
filepath="J:/DSU/CITI/s_discussions_DT.txt"
cnt=0
with open(filepath) as f:
    for line in f:
        new_str.append(line)
        cnt = cnt+ 1
        #print (line)
print (cnt)

print(type(new_str))

# new_str = [
# "The Russian foreign ministry said the UK staff would be expelled from Moscow within a week in response to Britain's decision to expel 23 Russian diplomats.",
# "The UK government says they were poisoned with a nerve agent of a type developed by Russia called Novichok - the Russian government denies any involvement in the attack.",
# "BBC diplomatic correspondent James Robbins said the move was significant because the Council fosters people-to-people relationships and, as it serves young people, could be crucial for the UK's relationship with a post-Putin Russia.",
# "Det Sgt Nick Bailey, who was part of the initial response to the incident, remains in a serious but stable condition in hospital after being exposed to the chemical."
# ]


# create a CountVectorizer instance
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words = 'english')

# transform the new_str into vectors of word frequency
X = vectorizer.fit_transform(new_str)
print(X)

# cover the dense matrix to a data frame
df_tf = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())

# print the data frame
print(df_tf)

# X by default is a sparse matrix
# the todense() method will convert X to a dense matrix
X.todense()
print(X)

print(vectorizer)

# Use term presence instead of term frequency

vectorizer = CountVectorizer(stop_words = 'english', binary=True)
X = vectorizer.fit_transform(new_str)
df_tf = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
print(df_tf)

## Using tfidf

vectorizer = TfidfVectorizer(stop_words = 'english', norm="l1")
X = vectorizer.fit_transform(new_str)
df_tf = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
print(df_tf)

df_tf.apply(np.sum, axis=1)
# the sum is always 1 due to normalization

vectorizer = TfidfVectorizer(stop_words = 'english', norm=None)
X = vectorizer.fit_transform(new_str)
df_tf = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
print(df_tf)

# Including bigrams and trigrams
vectorizer = TfidfVectorizer(stop_words = 'english', norm=None)
X = vectorizer.fit_transform(new_str)
df_tf = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
print(df_tf)

# Lemmatizing and stemming
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words = 'english')
X = vectorizer.fit_transform(new_str)
df_tf = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
print(df_tf)

# POS tagging
from nltk.corpus import stopwords
stopwords = set(nltk.corpus.stopwords.words("english"))
print(stopwords)

###############################
# conver words to lower case
lower = map(str.lower, new_str)

# replace punctuation with space
no_punc = map(lambda x: re.sub("[^a-z]", " ", x), lower)

# tokenize each document
tokenized = map(nltk.word_tokenize, no_punc)

# pos tag teach document
tagged = map(nltk.pos_tag, tokenized)

# remove stopwords
# stopwords can only be removed after POS tags are generated. Otherwise, it will influence the POS tagging results.
stopwords = nltk.corpus.stopwords.words("english")
def remove_stopwords(doc):
    out = []
    for word in doc:
        if word[0] not in stopwords: out.append(word)
    return out

no_stopwords = map(remove_stopwords, tagged)

print(no_stopwords)

###################################################
# convert the lists of tagged words to string so that scikit-learn can tokenize them
tagged_new_str = map(str, no_stopwords)
print(tagged_new_str)

###############################################
# vecotrize
import re
vectorizer = CountVectorizer(token_pattern=r"\('[^ ]+', '[^ ]+'\)", lowercase=False)
X = vectorizer.fit_transform(tagged_new_str)
df_tf = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
print(df_tf)

################################################