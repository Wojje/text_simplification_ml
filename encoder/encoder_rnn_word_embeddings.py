######################################################################
# The Seq2Seq Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.
#
# A `Sequence to Sequence network <http://arxiv.org/abs/1409.3215>`__, or
# seq2seq network, or `Encoder Decoder
# network <https://arxiv.org/pdf/1406.1078v3.pdf>`__, is a model
# consisting of two RNNs called the encoder and decoder. The encoder reads
# an input sequence and outputs a single vector, and the decoder reads
# that vector to produce an output sequence.
#
# .. figure:: /_static/img/seq-seq-images/seq2seq.png
#    :alt:
#
# Unlike sequence prediction with a single RNN, where every input
# corresponds to an output, the seq2seq model frees us from sequence
# length and order, which makes it ideal for translation between two
# languages.
#
# Consider the sentence "Je ne suis pas le chat noir" → "I am not the
# black cat". Most of the words in the input sentence have a direct
# translation in the output sentence, but are in slightly different
# orders, e.g. "chat noir" and "black cat". Because of the "ne/pas"
# construction there is also one more word in the input sentence. It would
# be difficult to produce a correct translation directly from the sequence
# of input words.
#
# With a seq2seq model the encoder creates a single vector which, in the
# ideal case, encodes the "meaning" of the input sequence into a single
# vector — a single point in some N dimensional space of sentences.
#


######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn

use_cuda = torch.cuda.is_available()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_weights, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        embed_num = embedding_weights.size(0)
        embed_dim = embedding_weights.size(1)
        self.embedding = nn.Embedding(embed_num, embed_dim)
        self.embedding.weight = nn.Parameter(embedding_weights)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        result = [Variable(torch.zeros(1, 1, self.hidden_size)),
                  Variable(torch.zeros(1, 1, self.hidden_size))]
        if use_cuda:
            return [result[0].cuda(), result[1].cuda()]
        else:
            return result

