import itertools
import math
import pickle
import pandas as pd
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
from sklearn.feature_extraction.text import CountVectorizer

# ====================================================================================
def computeTF(wordDic, bow):
    tfDict={}
    bowCount=len(bow)
    for word,count in wordDic.items():
        tfDict[word]= count/float(bowCount)
    return tfDict
# ====================================================================================
def computeIDF(docList):
    idfDict={}
    N= len(docList)
    # count the number of documents that contain a word w
    idfDict=dict.fromkeys(docList[0].keys(),0)
    for doc in docList:
        for word,val in doc.items():
            idfDict[word] +=1
    # divides N by donminator above, take the log of that
    for word, val in idfDict.items():
        idfDict[word]=math.log(N/float(val))

    return idfDict
# ====================================================================================
def computeTFIDF(tfBow,idfs):
    tfidf={}
    for word,val in tfBow.items():
        tfidf[word]=val * idfs[word]
    return tfidf
# ====================================================================================
def TFIDF_1(paras):
    bowlist=[]
    wordset={}
    worddicts=[]
    for p in paras:
        bowlist.append(word_tokenize(p))
    for b in bowlist:
        wordset=set(b).union(wordset)
    print (wordset)
    for i in range(len(paras)):
        worddicts.append(dict.fromkeys(wordset,0))
    print (worddicts)

    i=0
    for p in bowlist:
        for w in p:
            worddicts[i][w]+=1
        i+=1
        if i==len(paras):break
    df=pd.DataFrame(worddicts)
    print(df)

    for i in range(len (paras)): print (computeTF(worddicts[i],bowlist[i]))
    tfBow=[ computeTF(worddicts[i],bowlist[i]) for i in range(len (paras))]
    idfs=computeIDF(worddicts)
    print ("idfs = ",idfs)

    print( "TFIDF ================================")

    for i in range(len (paras)): print (computeTFIDF(tfBow[i],idfs))
    tfidf_list=[computeTFIDF(tfBow[i],idfs) for i in range(len (paras)) ]
    print (tfidf_list)

    df2= pd.DataFrame(tfidf_list)
    # print (df2)
    # df.to_csv('csv_outputs/TFIDF_Out.csv')


    bi_gramf = FreqDist(list(itertools.chain.from_iterable(bowlist)))
    # print (bi_gramf)
    print(bi_gramf.most_common(20))
    # bi_gramf.plot(50, cumulative = True)
# ====================================================================================
    '''
    CountVectorizer(analyzer=u'word', binary=False, decode_error=u'strict',
            dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern=u'(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)
    '''
def Count_Vect(paras):
    vect= CountVectorizer(max_features=20)
    vect.fit(paras)
    print(vect.get_feature_names())
    print (len(vect.get_feature_names()))
    dtm = vect.transform(paras)
    print (dtm)
    df= pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
    df.to_csv('csv_outputs/Count_Vect.csv')

# ====================================================================================
    '''
    TfidfVectorizer(input=’content’, encoding=’utf-8’, decode_error=’strict’, 
                strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, 
                analyzer=’word’, stop_words=None, token_pattern=’(?u)\b\w\w+\b’, ngram_range=(1, 1), 
                max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, 
                dtype=<class ‘numpy.int64’>, norm=’l2’, use_idf=True, smooth_idf=True, sublinear_tf=False)
    '''
def CreateDTM(paras):
    # vect = TfidfVectorizer(max_features=10,min_df=10,max_df=40)
    vect = TfidfVectorizer(max_features=20,ngram_range=(2,2))
    dtm = vect.fit_transform(paras)  # create DTM
    print (dtm)
    # create pandas dataframe of DTM
    df = pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
    df.to_csv('csv_outputs/TFIDF_Vect.csv')
    return pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())

# ====================================================================================

if __name__ == '__main__':

    save_clean_lines = open("clean_diabApp.pickle", "rb")
    paras = pickle.load(save_clean_lines)
    save_clean_lines.close()

    print ("-------------------------Count Vectorizer-------------------")
    Count_Vect(paras)
    print ("-------------------------TF-IDF Vectorizer-------------------")
    CreateDTM(paras)








