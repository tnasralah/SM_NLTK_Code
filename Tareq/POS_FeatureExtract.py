"""
 Ch7. NLTK - Extracting Information from Text
"""
import pickle, re
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag, RegexpParser
from nltk import conlltags2tree, tree2conlltags
from random import shuffle


def pos_preprocess(document):
    sentences = re.sub(r"http\S+", " url ", document)
    sentences = re.sub(r"www\S+", " url ", sentences)
    sentences = sentences.replace('.', '. ')
    sentences = sent_tokenize(sentences)
    sentences = [word_tokenize(sent) for sent in sentences]
    sentences = [pos_tag(sent) for sent in sentences]
    print('--------\n', sentences, '\n=================================================')
    for i in sentences:
        print(i)
    print('----------------------------------------------------------------------------')
    return sentences


# =================================================================

def NP_Chunker(sentence):
    """
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    This rule says that an NP chunk should be formed whenever the chunker finds an optional
    determiner (DT) followed by any number of adjectives (JJ) and then a noun (NN).

    <DT>?<JJ.*>*<NN.*>+.
    This will chunk any sequence of tokens beginning with an optional
    determiner, followed by zero or more adjectives of any type (including relative adjectives
    like  earlier/JJR), followed by one or more nouns of any type.

    chunkGram = " {<RB.?>*<VB.?>*<NNP>+<NN>?}"
    """
    grammar_exp = r"""
      NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
          {<NNP>+}                # chunk sequences of proper nouns
    """
    chunkParser = RegexpParser(grammar_exp)
    r = chunkParser.parse(sentence)
    print(r.draw())

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = RegexpParser(grammar)
    result = cp.parse(sentence)
    print(result)

    result.draw()


# =================================================================
def Ext_Chunks(sents):
    NP_li = []
    # print(sents)
    grammar_exp = r"""
      CHUNK: {<NN><NN.*><NN.*>+}   # chunk determiner/possessive, adjectives and noun
             }<NNP>+{              # chunk sequences of proper nouns
    """
    # cp = nltk.RegexpParser('CHUNK:  {<NN><NN.*><NN.*>+}}<NNP>{')
    cp = nltk.RegexpParser(grammar_exp)
    # cp = nltk.RegexpParser('CHUNK:  {<DT>?<JJ.*>*<NN.*>+}')

    for sent in sents:
        tree = cp.parse(sent)
        # print(tree.draw())
        for subtree in tree.subtrees():
            if subtree.label() == 'CHUNK':
                print(subtree)
                iob_tags = tree2conlltags(subtree)
                iob_tree = conlltags2tree(iob_tags)
                print(iob_tags)
                print(iob_tree)
                chunk_words = str(subtree).replace('/DT', '').replace('/JJS', '').replace('/JJ', '').replace('/NNS',
                                                                                                             '').replace(
                    '/NNP', '').replace('(CHUNK', '').replace(')', '').replace('/NN', '').replace('\n', '')
                NP_li.append(chunk_words)
                print(chunk_words, '\n')
    print('----------------------------------------------------------------\n',NP_li)
    return NP_li


# =================================================================

def main():
    # Read the cleanlines pickle
    save_clean_lines = open("pickles\TechAll_raw_lines.pickle", "rb")
    # save_clean_lines = open("pickles/clean_diabApp.pickle", "rb")
    c_lines = pickle.load(save_clean_lines)
    save_clean_lines.close()
    sents = []
    np_all = []
    docs = c_lines
    for doc in docs[:30]:
        print(doc)
        sents += pos_preprocess(doc)

    # for s in sents[:1]:
    #      NP_Chunker(s)

    np_all += Ext_Chunks(sents)
    print(len(np_all),' ', len(set(np_all)))

    long_chunks = [w for w in set(np_all) if len(w)>15]
    c_long = []
    for i in long_chunks:
        x=[]
        for j in i.split():
            if len(j)>=4:
                x.append(j)
        c_long.append(' '.join(x))
    print("long chunks:\n", long_chunks,'\nClean long chunks:\n',c_long)


if __name__ == '__main__':
    main()
