from torch.autograd import Variable
import torch

use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1

def indexesFromSentence(vocabulary, sentence):
    return [vocabulary[word] for word in sentence.split(' ') if vocabulary.get(word)!=None]
#TODO ändra så att ord som inte finns i vokabuläret ersätts av en speciell token och iställe för att bara slängas bort

def variableFromSentence(vocabulary, sentence):
    indexes = indexesFromSentence(vocabulary, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair, vocabulary):
    input_variable = variableFromSentence(vocabulary, pair[0])
    target_variable = variableFromSentence(vocabulary, pair[1])
    return (input_variable, target_variable)

