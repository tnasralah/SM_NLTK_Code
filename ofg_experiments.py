from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.corpus.reader.util import read_line_block

import gensim
from gensim import corpora


"""
# This is the default function for reading line blocks found in nltk.reader.corpus.util
# This code can be used as a base for creating a replacement function. 
# Otherwise, the code is not needed.

def read_line_block(stream):
    toks = []
    for i in range(20):
        line = stream.readline()
        if not line: return toks
        toks.append(line.rstrip('\n'))
    return toks

"""


# Reading corpus
corpus_root = '/Users/elgayaro/nltk_data/corpora/tudiab'
ww = PlaintextCorpusReader(corpus_root, r'(?!README|\.).*\.txt', para_block_reader=read_line_block)

# Displaying sample corpus
print('The files in this corpus are: ', ww.fileids())
print('Number of words (before pre-processing) in the corpus = ', len(set(ww.words())))
print('There are documents/paragraphs/threads in the corpus = ', len(ww.paras()))

# print(ww.words())
# print(ww.sents()[0])
# print(ww.paras()[0])
# print(ww.raw()[:10])


"""
# Generating frequency distributions
ww_freq = FreqDist(ww.words())
print('Frequency for "dexcom" = ', ww_freq['dexcom'])
print(ww_freq.most_common(30))
ww_freq.plot(50, cumulative=True)

# Generating ngrams
bi_grams = ngrams(ww.words(), 2)
print(type(bi_grams))
print(list(bi_grams)[:10])
"""

# Defining stop words
stopwords = stopwords.words('english')
add_stopwords = ['http', 'https', '://', 'www', 'com', '8800', '...', '....', 'yep', '.).', '.:).', '++..', 'github']
[stopwords.append(st) for st in add_stopwords]


"""
# Filter short words
filtered_words = [w for w in ww.words() if len(w) >= 3]

# Removing stop words
words = [w.lower() for w in filtered_words if w.lower() not in stopwords]
print(words)
print(len(words))
"""

"""
Pre-processing of paragraphs
"""


# Filter short words and stopwords from paragraphs
filtered_paras = [[] for i in range(len(ww.paras()))]

i = 0
for p in ww.paras():
    for s in p:
        for w in s:
            if len(w) >= 3 and w.lower() not in stopwords:
                filtered_paras[i].append(w.lower())
    i += 1
print(len(filtered_paras))
print('+++ First filetered parapgraph +++')
print(filtered_paras[0])

"""
# Generating ngrams
bi_grams_para = []
tri_grams_para = []

# Generating ngrams in paragraphs
for p in filtered_paras:
    bi_grams_para.append(list(ngrams(p, 2)))
    tri_grams_para.append(list(ngrams(p, 3)))

print('\n+++ ngram in first paragraphs')
print('++++ bi_grams +++')
print(bi_grams_para[:1])
print('++++ tri_grams +++')
print(tri_grams_para[:1])


# Generating ngrams across entire corpus (all paragraphs)
# bi_grams
bi_grams = []
for p in bi_grams_para:
    [bi_grams.append(bg) for bg in p]
print(bi_grams[:10])

bi_gramf = FreqDist(bi_grams)
print('\n+++ Frequent bi_grams')
print(bi_gramf.most_common(20))
bi_gramf.plot(50, cumulative=True)

# tri_grams
tri_grams = []
for p in tri_grams_para:
    [tri_grams.append(bg) for bg in p]
print(tri_grams[:10])

tri_gramf = FreqDist(tri_grams)
print('\n+++ Frequent tri_grams')
print(tri_gramf.most_common(20))
tri_gramf.plot(50, cumulative=True)
"""


# Preparing Document-Term Matrix

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(filtered_paras)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(p) for p in filtered_paras]
print('Number of items in dictionary = ', len(dictionary.iteritems()))
print('Number of paragraphs = ', len(doc_term_matrix))

"""
# Exploring the doc_term_matrix
# doc_term_matrix is a list of lists
# Each document/paragraph is a list of tuples. Each tuple represents an index to the corresponding
# term and the frequency of the term

print(type(doc_term_matrix))

print(len(doc_term_matrix[0]))
print(len(doc_term_matrix[1]))
print(len(doc_term_matrix[2]))
print(len(doc_term_matrix[3]))

print('\n+++')
print(doc_term_matrix[0])
print('\n+++')
print(doc_term_matrix[1])
print('\n+++')
print(doc_term_matrix[2])
print('\n+++')
print(doc_term_matrix[0][0])
"""


# Running LDA Model
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)
print(ldamodel.print_topics(num_topics=3, num_words=3))
