import unicodedata
import re
import torchwordemb


######################################################################
# We will be representing each word in a language as a one-hot
# vector, or giant vector of zeros except for a single one (at the index
# of the word). 
#
# .. figure:: /_static/img/seq-seq-images/word-encoding.png
#    :alt:
#
#


#TODO Använd en dsitributional word embedding, till exempel word2vec eller GloVe istället för one-hot vector.


######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Vocab`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.
#


class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


#TODO Vokabuläret kommer inte att kunna innehålla alla ord. Okända ord kan ersättas med antingen en token UNK eller med en metod som behåller någon sorts information om det ersatta ordet (fråga Evelina eller Marco)


######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
#

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFC', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase and trim


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?,:;])", r" \1", s)
    s = re.sub(r"[^a-öA-Ö.!?]+", r" ", s) #remove non-letter characters
    return s


######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. 
#

def readData(filename):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(filename, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    triplets = [[s for s in l.split('\t')] for l in lines]

    vocabulary = Vocab(filename)

    return vocabulary, triplets


######################################################################
# The maximum length is 30 words (that includes ending punctuation)
#


def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length


def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(filename, max_length=10, sts_threshold=0.5, vocab_limit=None):
    vocabulary, triplets = readData(filename)
    print("Read %s sentence pairs" % len(triplets))
    pairs = [[normalizeString(s[1]), normalizeString(s[0])] for s in triplets if float(s[2])>sts_threshold]
    print("Number of sentence pairs with Sentence Similarity above %s: %s" % (sts_threshold, len(pairs)))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    vocabulary.addWord("<SOS>")
    vocabulary.addWord("<EOS>")
    for pair in pairs[:vocab_limit]:
        vocabulary.addSentence(pair[0])
        vocabulary.addSentence(pair[1])
    print("Counted words:")
    print(vocabulary.name, vocabulary.n_words)
    return vocabulary, pairs



def prepareDataAndWordEmbeddings(data_filename, embedding_weights_filename, max_length=10, sts_threshold=0.5):
    _, triplets = readData(data_filename)
    print("Read %s sentence pairs" % len(triplets))
    pairs = [[normalizeString(s[1]), normalizeString(s[0])] for s in triplets if float(s[2])>sts_threshold]
    print("Number of sentence pairs with Sentence Similarity above %s: %s" % (sts_threshold, len(pairs)))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    vocab_dict, embeddings = torchwordemb.load_word2vec_text(embedding_weights_filename)
    vocabulary = Vocab(embedding_weights_filename)
    for w in vocab_dict:
        vocabulary.addWord(w)
    print("Vocabulary size:")
    print(vocabulary.name, vocabulary.n_words)
    return vocabulary, pairs, embeddings

