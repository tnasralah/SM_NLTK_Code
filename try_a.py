## playing with gutenberg corpus and diabetes text
## some basic analysis and exploration and plotting
## could read and strip sentence from diabetes but unsuccessful to create corpora (so , we put our files inside gutenberg and it magically works)
## could show the plot of occurance of word (useful for exp. data analysis) and also performed POS tagging.
## Have demo from US president saying "America" and "Citizen" , so that we can use same concept to show the occurance of word on a grouped tweets or discussions

import nltk
import matplotlib
from nltk.corpus import gutenberg


# Now that we've downloaded all the NLTK corpus content, let us go ahead and
# load in the text from s_discussions_DT" via Gutenberg:
from nltk.text import Text
discuss = Text(nltk.corpus.gutenberg.words('s_discussions_DT.txt'))

#converting every words to lower case
discuss = nltk.Text(word.lower() for word in discuss)
print(discuss)
#with open('C:\\nltk_data\corpora\diabetes\s_discussions_DT.txt', 'r') as myfile:
#    discuss=myfile.read()

## tagging every words into POS
from nltk import word_tokenize
sample_text=gutenberg.raw("s_discussions_DT.txt")
text = word_tokenize(sample_text)
print(nltk.pos_tag(text))

## Lemmatization
#from nltk.stem import WordNetLemmatizer
#lemmatizer = WordNetLemmatizer()
#print(lemmatizer.lemmatize(sample_text))

## Replacing synonyms
#from replacers import WordReplacer
#replacer = WordReplacer({'bday': 'birthday'})
#replacer.replace('bday')

# NLTK also provides other texts from Gutenberg. We can view those by
# running the following command:
print(nltk.corpus.gutenberg.fileids())

print(type(discuss))
print(len(discuss))

# count the word "the" on each occurrence.
print(len(set(discuss)))

# Specific Word Count: How many times does a specific word occur
# in a text?
print(discuss.count("technology"))

# Concordance: Shows occurence of word in context of use.
discuss.concordance("diabetes")

# Dispersion Plot: Location of where a word is in the text.
# Example:
#   Give a visual representation of where the words "discuss", "Rabbit",
#   "Hatter", and "Queen" appear in "discuss in Wonderland".
discuss.dispersion_plot(["diabetes", "type", "technology", "sensor"])

# Frequency Distributions: What are the most frequent words (specifically,
# tokens), that are used in a given text.

# First, use NLTK to generate a frequncy distribution dictionary-like object.
fdist = nltk.FreqDist(discuss)

# What are the top 50 most common words in "discuss in Wonderland"?
fdist.plot(50, cumulative=False, title="50 most common tokens in s-discussions_DT")

# Observe that the x-axis consists of punctuation, which may not
# be precisely what we are going for. It is possible to remove this
# from the words that we plot by filtering out the punctuation.
fdist_no_punc = nltk.FreqDist(
        dict((word, freq) for word, freq in fdist.items() if word.isalpha()))

fdist_no_punc.plot(50,
                   cumulative=False,
                   title="50 most common tokens (no punctuation)")

# This plot gives us a bit more useful information, but it still contains an
# awful lot of punctuation that we do not particularly care to see. In a
# similar fashion, we may filter this out.

# We may not obtain too much information on the above plot, since
# many of the words on the x-axis are words like "and", "the", "in",
# etc. These types of common English words are referred to as
# stopwords. NLTK provides a method to identify such words.
stopwords = nltk.corpus.stopwords.words('english')
fdist_no_punc_no_stopwords = nltk.FreqDist(
        dict((word, freq) for word, freq in fdist.items() if word not in stopwords and word.isalpha()))

# Replot fdist after stopwords filtered out.
fdist_no_punc_no_stopwords.plot(50,
                                cumulative=False,
                                title="50 most common tokens (no stopwords or punctuation)")


################################################################################
# We shall pepper in a few NLP terms from time to time to reduce the
# overwhelm of encountering too many new terms all at once.

print(fdist.hapaxes())

# Collocations: A pair or group of words that are habitually juxtaposed.
print(discuss.collocations())

################################################################################
# Get words from text (what we did in Part 1):
discuss_words = nltk.corpus.gutenberg.words('s_discussions_DT.txt')
# Note that Python does not print out the entire list or words. The ellipsis
# (...) sequence denotes that there is more content that is supressed from output.
print(discuss_words)

# Get characters from "discussion":
discuss_chars = nltk.corpus.gutenberg.raw('s_discussions_DT.txt')
print(discuss_chars)

# Get sentences from "discussion":
discuss_sents = nltk.corpus.gutenberg.sents('s_discussions_DT.txt')
print(discuss_sents)

# With the above chars, words, and sentences extracted from "discussion",
# we can make use of these to calculate some cursory information on the text:

# Average word length:
print(int(len(discuss_chars) / len(discuss_words)))

# Average sentence length:
print(int(len(discuss_words) / len(discuss_sents)))

# Let us turn the above two metrics into functions, and determine the average
# word length and sentence length of all the texts in the Gutenberg collection.


def avg_word_len(num_chars, num_words):
    return int(num_chars/num_words)


def avg_sent_len(num_words, num_sents):
    return int(num_words/num_sents)

# Let us loop through each address. While doing so, let us keep a running tally
# of the number of times the word "America" is used in each address.

# Loop through each inaugural address:
for fileid in nltk.corpus.inaugural.fileids():
    america_count = 0
    # Loop through all words in current inaugural address:
    for w in nltk.corpus.inaugural.words(fileid):
        # We convert the word to lowercase before checking
        # This makes checking for the occurrence more consistent.
        # Note that the "startswith" function also catches words like
        # "American", "Americans", etc.
        if w.lower().startswith('america'):
            america_count += 1
    # Output both the inaugural address name and count for America:
    president = fileid[5:-4]
    year = fileid[:4]
    print("President " + president +
          " of year " + year +
          " said America " + str(america_count) + " times. ")

# Say I also want to see how many times the word "citizen" is present in
# each of the inaugural addresses. It may be preferable to consider a plot
# output as opposed to one that simply outputs to terminal.

# Let us consider a conditional frequency distribution, that is a frequency
# distribution that is a collection of frequency distributions run under
# different conditions.

# Recall the FreqDist function took a list as input. NLTK provides a
# ConditionalFreqDist function as well which takes a list of pairs.
# Each pair has the form (condition, event).

# In our example, we care about the case when either the word "America"
# or "citizen" is used in each of the inaugural addresses. In other words,
# encountering the phrase "America" or "citizen" are the conditions we
# care about, and the events are one for each year of the inaugural address.

cfd = nltk.ConditionalFreqDist(
            (target, fileid[:4])
            for fileid in nltk.corpus.inaugural.fileids()
            for w in nltk.corpus.inaugural.words(fileid)
            for target in ['america', 'citizen']
            if w.lower().startswith(target))
cfd.plot()
