from nltk.corpus import PlaintextCorpusReader
from nltk.corpus.reader.util import read_line_block
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.corpus import stopwords

"""
def read_line_block(stream):
    toks = []
    for i in range(20):
        line = stream.readline()
        if not line: return toks
        toks.append(line.rstrip('\n'))
    return toks
"""

#Reading corpus
corpus_root = '/Users/elgayaro/nltk_data/corpora/tudiab'
#ww = PlaintextCorpusReader(corpus_root, '.*')
ww = PlaintextCorpusReader(corpus_root, r'(?!README|\.).*\.txt', para_block_reader=read_line_block)

# Displaying sample corpus
#print(type(ww))
print(ww.fileids())

#print(ww.words())
print('There are', len(ww.words()), 'in this corpus')
#print(ww.sents()[0])
print(ww.paras()[0])
#print(ww.raw()[:10])
print('There are ', len(ww.paras()), 'thread discussion in this corpus')

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
add_stopwords = ['http', '://', 'www', 'com', '...', '....', 'yep', '.).', '.:).', '++..']
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
            if len(w) >=3 and w.lower() not in stopwords:
                filtered_paras[i].append(w.lower())
    i += 1
print(len(filtered_paras))
print(filtered_paras[0])

print('+++++++++++++++++')
bi_grams_para = []
for p in filtered_paras:
    bi_grams_para.append(list(ngrams(p, 2)))

print(bi_grams_para[:2])

bi_grams = []
for p in bi_grams_para:
    [bi_grams.append(bg) for bg in p]
print(bi_grams[:10])

bi_gramf = FreqDist(bi_grams)
print(bi_gramf.most_common(20))
bi_gramf.plot(50, cumulative = True)

# bi_grams = ngrams([w for w in filtered_paras[0]], 2)
# print(list(bi_grams)[:10])

