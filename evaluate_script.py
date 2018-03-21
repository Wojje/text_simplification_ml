#!/usr/bin/env python

import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import nltk
import os
import argparse

import torch
from torch import optim
import torch.nn as nn

use_cuda = torch.cuda.is_available()

import helper_modules.prepare_data as prepare_data
import eval_iter.evaluate as evaluate
import helper_modules.vocabulary_helper as vocabulary_helper
import helper_modules.plotting_helper as plotting_helper
from encoder.encoder_rnn_word_embeddings import EncoderRNN
from decoder.attn_decoder_rnn_word_embeddings import AttnDecoderRNN


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='saved_models/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()


TEACHER_FORCING_RATIO = 0.5
DROPOUT_P = 0.1
N_ITERS = 80000
HIDDEN_SIZE = 1024
PRINT_EVERY=1000
PLOT_EVERY=100
CHECKPOINT_AND_EVALUATE_EVERY=2000
LEARNING_RATE=0.01

DATA_FILE_PATH = 'data/uniqueMaximum.txt'
EMBEDDINGS_FILE_PATH = 'word_embeddings/swectors-300dim.txt'
MAX_LENGTH = 30
VOCAB_SIZE = 90000
STS_THRESHOLD = 0.6  #använd bara par med Sentence Similarity över 0.6

start_iter = 1
best_eval_score = 0
training_losses = []
eval_scores = []




######################################################################
# PREPARING DATA
# =======================


vocabulary, pairs, embeddings = prepare_data.prepareDataAndWordEmbeddings(DATA_FILE_PATH, EMBEDDINGS_FILE_PATH, max_length=MAX_LENGTH, sts_threshold=STS_THRESHOLD, vocab_limit=VOCAB_SIZE)
print(random.choice(pairs))
cut = int(len(pairs)*0.8)
cut2 = int(len(pairs)*0.9)


######################################################################
# MODEL
# =======================
#


encoder = EncoderRNN(vocabulary.n_words, HIDDEN_SIZE, embedding_weights=embeddings)
decoder = AttnDecoderRNN(HIDDEN_SIZE, vocabulary.n_words, embedding_weights=embeddings, dropout_p=DROPOUT_P, max_length=MAX_LENGTH)
# TODO Implementera möjlighet för flera layers.
for param in encoder.embedding.parameters():
    param.requires_grad = False
for param in decoder.embedding.parameters():
    param.requires_grad = False


if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

encoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()), lr=LEARNING_RATE)
decoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, decoder.parameters()), lr=LEARNING_RATE)

path_to_default_checkpoint = 'saved_models/model_best.pth.tar'
checkpoint = torch.load(path_to_default_checkpoint)
if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
else:
    print("=> no checkpoint found at '{}'. Using default at '{}'.'".format(args.resume, path_to_default_checkpoint))

start_iter = checkpoint['iteration']+1
best_eval_score = checkpoint['best_bleu']
training_losses = checkpoint['training_losses']
eval_scores = checkpoint['bleu_scores']
indices = checkpoint['shuffled_indices']
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
print("=> loaded checkpoint '{}' (iteration {})"
      .format(args.resume, checkpoint['iteration']))



#train_set = [pairs[i] for i in indices[:cut]]
validate_set = [pairs[i] for i in indices[cut:cut2]]
#test_set = [pairs[i] for i in indices[cut2:]]

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


stop = False
while not stop:
    input_sentence = input('Type a sentence or leave blank to randomly select one from the validation set: ')
    input_sentence = prepare_data.normalizeString(input_sentence)
    if not input_sentence:
        pair = random.choice(validate_set)
        print('>', pair[0])
        print('=', pair[1])
        input_sentence = pair[0]
    output_words, attentions = evaluate.evaluate(encoder, decoder, input_sentence, vocabulary, max_length=MAX_LENGTH)
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
    print('')
    showAttention(input_sentence, output_words, attentions)
    stop = input('Type any character to stop. Enter to continue. ')
    if stop:
        confirmed = False
        while not confirmed:
            stop = input('Are you sure you want to exit the program? [y/N] ')
            if stop not in ['y', 'Y', 'n', 'N'] and stop:
                print('please enter y or n.')
                confirmed = False
            elif stop == 'y' or stop == 'Y':
                stop = True
                confirmed = True
            elif stop == 'n' or stop == 'N':
                stop = False
                confirmed = True
            else:
                confirmed = True
