import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.corpus.reader.util import read_line_block

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.util import ngrams
from nltk.probability import FreqDist

import gensim
from gensim import corpora


from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


# Reading corpus
corpus_root = '/Users/elgayaro/nltk_data/corpora/tudiab'
ww = PlaintextCorpusReader(corpus_root, r'(?!README|\.).*\.txt', para_block_reader=read_line_block)

# Displaying sample corpus
print('The files in this corpus are: ', ww.fileids())
print('Number of words (before pre-processing) in the corpus = ', len(set(ww.words())))
print('There are documents/paragraphs/threads in the corpus = ', len(ww.paras()))
# print(ww.words())
# print(ww.sents()[0])
# print(ww.paras()[10])
# print(ww.raw()[:10])


# Defining stop words
stopwords = stopwords.words('english')
add_stopwords = ['http', 'https', '://', 'www', 'com', '8800', '...', '....', 'yep', '.).', '](#', '.:).',
                  '++..', 'github', 'etc', 'also', 'org', 'gee', 'let', 'know', 'ever',
                 'vcntr', 'falseamount', 'isig']
[stopwords.append(st) for st in add_stopwords]


# Filter short words and stopwords from paragraphs, and lemmatize
filtered_paras = [[] for i in range(len(ww.paras()))]

wnl = WordNetLemmatizer()
i = 0
for p in ww.paras():
    for s in p:
        ts = nltk.pos_tag(s)
        tls = [wnl.lemmatize(w, get_wordnet_pos(pos)) for (w, pos) in ts if get_wordnet_pos(pos) != '']
        for w in tls:
            if len(w) >= 3 and w.isalpha() and w.lower() not in stopwords:
                filtered_paras[i].append(w.lower())
    i += 1

print('Number of words (after pre-processing) in the corpus = ',
      len(set([word for p in filtered_paras for word in p])))
print('There are documents/paragraphs/threads in the corpus = ', len(filtered_paras))

print('+++ First filetered parapgraph +++')
print(filtered_paras[0])


# Generating ngrams
uni_grams_para = []
bi_grams_para = []
tri_grams_para = []

# Generating ngrams in paragraphs
for p in filtered_paras:
    uni_grams_para.append(list(ngrams(p, 1)))
    bi_grams_para.append(list(ngrams(p, 2)))
    tri_grams_para.append(list(ngrams(p, 3)))

print('\n+++ ngram in first paragraphs')
print('\n+++ uni_grams +++')
print(uni_grams_para[:1])
print('\n+++ bi_grams +++')
print(bi_grams_para[:1])
print('\n+++ tri_grams +++')
print(tri_grams_para[:1])


# Generating ngrams across entire corpus (all paragraphs)
# bi_grams
uni_grams = []
for p in uni_grams_para:
    [uni_grams.append(bg) for bg in p]
# print(uni_grams[:10])

uni_gramf = FreqDist(uni_grams)
print('\n+++ Frequent uni_grams')
print(uni_gramf.most_common(40))
uni_gramf.plot(50, cumulative=False)

# bi_grams
bi_grams = []
for p in bi_grams_para:
    [bi_grams.append(bg) for bg in p]
# print(bi_grams[:10])

bi_gramf = FreqDist(bi_grams)
print('\n+++ Frequent bi_grams')
print(bi_gramf.most_common(40))
bi_gramf.plot(50, cumulative=False)

# tri_grams
tri_grams = []
for p in tri_grams_para:
    [tri_grams.append(bg) for bg in p]
# print(tri_grams[:10])

tri_gramf = FreqDist(tri_grams)
print('\n+++ Frequent tri_grams')
print(tri_gramf.most_common(40))
tri_gramf.plot(50, cumulative=False)


#
print('+++++++++++++ Replacing tri_grams ++++++++++++++++')
trigram_lexicon = [['one', 'touch', 'ultra'], ['tudiabetes', 'org', 'group'],
            ['continuous', 'glucose', 'monitoring'], ['time', 'per', 'day'],
            ['blood', 'glucose', 'meter'], ['continuous', 'glucose', 'monitor'],
            ['continuous', 'glucose', 'monitoring'],
            ['blood', 'glucose', 'level'], ['low', 'blood', 'sugar'],
            ['one', 'touch', 'meter'], ['glucose', 'monitoring', 'system'],
            ['high', 'blood', 'sugar'], ['blood', 'sugar', 'level'],
            ['red', 'blood', 'cell'], ['durable', 'medical', 'equipment'],
            ['accu', 'chek', 'aviva'], ['dexcom', 'seven', 'plus'],
            ['use', 'one', 'touch'], ['blood', 'sugar', 'control'],
            ['tudiabetes', 'org', 'forum'], ['contour', 'next', 'link'],
            ['letter', 'medical', 'necessity'], ['low', 'carb', 'diet'],
            ['cross', 'blue', 'shield']]

ngram_count = 3
trigram_paras = []
for p in filtered_paras:
    if len(p) >= ngram_count:
        trigram_p = []
        i = 0
        while i <= (len(p) - ngram_count):
            if p[i:(i + ngram_count)] in trigram_lexicon:
                trigram_p.append(p[i]+'_'+p[i+1]+'_'+p[i+2])
 #               print('exchange made')
 #               print(p[i] + '_' + p[i + 1] + '_' + p[i + 2])

                i += 3
            else:
                trigram_p.append(p[i])
                i += 1

        while i < len(p):
            trigram_p.append(p[i])
            i += 1

        trigram_paras.append(trigram_p)
    else:
        trigram_paras.append(p)

print('Number of words (after trigram-replacement) in the corpus = ',
      len(set([word for p in trigram_paras for word in p])))
print('There are documents/paragraphs/threads in the corpus = ', len(trigram_paras))

for i in range(1):
    print(filtered_paras[i])
    print(trigram_paras[i])

print(len(set(filtered_paras[1])))
print(len(set(trigram_paras[1])))


"""
# Preparing Document-Term Matrix

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(filtered_paras)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(p) for p in filtered_paras]
print('Number of items in dictionary = ', len(dictionary.iteritems()))
print('Number of paragraphs = ', len(doc_term_matrix))

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

"""
# Running LDA Model
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)
print(ldamodel.print_topics(num_topics=3, num_words=3))
"""