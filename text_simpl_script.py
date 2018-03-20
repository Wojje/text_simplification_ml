#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified from: `Sean Robertson <https://github.com/spro/practical-pytorch>`_



**Requirements**
"""
import random
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import shutil
import nltk
import os
import argparse

import torch
from torch import optim
import torch.nn as nn

use_cuda = torch.cuda.is_available()

import helper_modules.prepare_data as prepare_data
import train_iter.train as train
import eval_iter.evaluate as evaluate
import helper_modules.vocabulary_helper as vocabulary_helper
import helper_modules.timer_helper as timer_helper
import helper_modules.plotting_helper as plotting_helper
from encoder.encoder_rnn_word_embeddings import EncoderRNN
from decoder.attn_decoder_rnn_word_embeddings import AttnDecoderRNN


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()


TEACHER_FORCING_RATIO = 0.5
DROPOUT_P = 0.1
N_ITERS = 75000
HIDDEN_SIZE = 1024
PRINT_EVERY=1000
PLOT_EVERY=100
CHECKPOINT_AND_EVALUATE_EVERY=1000
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
indices = list(range(len(pairs)))
random.shuffle(indices)
cut = int(len(pairs)*0.8)
cut2 = int(len(pairs)*0.9)


######################################################################
# MODEL
# =======================
#


encoder = EncoderRNN(vocabulary.n_words, HIDDEN_SIZE, embedding_weights=embeddings)
decoder = AttnDecoderRNN(HIDDEN_SIZE, vocabulary.n_words, embedding_weights=embeddings, dropout_p=DROPOUT_P, max_length=MAX_LENGTH)
# TODO Frys de förtränade word embeddingvikterna så att de inte tränas.
for param in encoder.embedding.parameters():
    param.requires_grad = False
for param in decoder.embedding.parameters():
    param.requires_grad = False


if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

encoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()), lr=LEARNING_RATE)
decoder_optimizer = optim.SGD(filter(lambda p: p.requires_grad, decoder.parameters()), lr=LEARNING_RATE)

if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
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
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))




######################################################################
# TRAINING
# =======================
#

train_set = [pairs[i] for i in indices[:cut]]
validate_set = [pairs[i] for i in indices[cut:cut2]]
test_set = [pairs[i] for i in indices[cut2:]]
print("Training set: %d" %len(train_set))
print("Validation set: %d" %len(validate_set))
print("Test set: %d" %len(test_set))

encoder.train()
decoder.train()
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every
training_pairs = [vocabulary_helper.variablesFromPair(random.choice(train_set), vocabulary)
                  for i in range(N_ITERS)]
criterion = nn.NLLLoss()


start = time.time()

for iter in range(start_iter, N_ITERS + 1):
    training_pair = training_pairs[iter - 1]
    input_variable = training_pair[0]
    target_variable = training_pair[1]

    loss = train.train(input_variable, target_variable, encoder,
                 decoder, encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
    print_loss_total += loss
    plot_loss_total += loss

    if iter % PRINT_EVERY == 0:
        print_loss_avg = print_loss_total / PRINT_EVERY
        print_loss_total = 0
        print('%s (%d %d%%) %.4f' % (timer_helper.timeSince(start, iter / N_ITERS),
                                     iter, iter / N_ITERS * 100, print_loss_avg))
        evaluateRandomly(encoder,decoder,n=1)

    if iter % PLOT_EVERY == 0:
        plot_loss_avg = plot_loss_total / PLOT_EVERY
        training_losses.append(plot_loss_avg)
        plot_loss_total = 0

    if iter % CHECKPOINT_AND_EVALUATE_EVERY == 0:
        score = 0
        n_validation_pairs = len(validate_set)
        encoder.eval()
        decoder.eval()
        for pair in [random.choice(validate_set) for _ in range(n_validation_pairs)]:
            output_words, attentions = evaluate.evaluate(encoder, decoder, pair[0], vocabulary, max_length=MAX_LENGTH)
            score += nltk.translate.bleu_score.sentence_bleu([pair[1]], output_words)
        encoder.train()
        decoder.train()
        eval_score = score / n_validation_pairs
        eval_scores.append(eval_score)
        is_best = eval_score > best_eval_score
        best_eval_score = max(eval_score, best_eval_score)
        print("Average BLEU score after %d iterations: %d" % (iter, eval_score))
        checkpoint = {
            'iteration': iter,
            'shuffled_indices': indices,
            'training_losses': training_losses,
            'bleu_scores': eval_scores,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'encoder_optimizer' : encoder_optimizer.state_dict(),
            'decoder_optimizer' : decoder_optimizer.state_dict(),
            'best_bleu': best_eval_score
        }
        filename = "saved_models/checkpoint.pth.tar"
        torch.save(checkpoint, filename)
        if is_best:
            shutil.copyfile(filename, 'saved_models/model_best.pth.tar')

#plotting_helper.showPlot(plot_losses)
#plotting_helper.showPlot(eval_losses)
#plt.figure()
#fig, ax = plt.subplots()
# this locator puts ticks at regular intervals
#loc = ticker.MultipleLocator(base=0.2)
#ax.yaxis.set_major_locator(loc)
#plt.plot(plot_losses)
#plt.plot(eval_losses)
#plt.show()


######################################################################
# EVALUATION
# =======================
#



######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(training_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate.evaluate(encoder, decoder, pair[0], vocabulary, max_length=MAX_LENGTH)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')



######################################################################
#

#evaluateRandomly(encoder, decoder)


######################################################################
# Visualizing Attention
# ---------------------
#
# A useful property of the attention mechanism is its highly interpretable
# outputs. Because it is used to weight specific encoder outputs of the
# input sequence, we can imagine looking where the network is focused most
# at each time step.
#
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
#

output_words, attentions = evaluate.evaluate(
    encoder, decoder, "det här är en ganska svår mening , alla skulle inte förstå den .", vocabulary, max_length=MAX_LENGTH)
#plt.matshow(attentions.numpy())


######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#

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


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate.evaluate(
        encoder, decoder, input_sentence, vocabulary, max_length=MAX_LENGTH)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)



#evaluateAndShowAttention("alla människor kan ibland behöva stöd och hjälp för att underlätta vardagen")

#evaluateAndShowAttention("vi består av ett bibliotek samt en bokbuss .")


######################################################################
# Exercises
# =========
#
# -  Try with a different dataset
#
#    -  Another language pair
#    -  Human → Machine (e.g. IOT commands)
#    -  Chat → Response
#    -  Question → Answer
#
# -  Replace the embeddings with pre-trained word embeddings such as word2vec or
#    GloVe
# -  Try with more layers, more hidden units, and more sentences. Compare
#    the training time and results.
# -  If you use a translation file where pairs have two of the same phrase
#    (``I am test \t I am test``), you can use this as an autoencoder. Try
#    this:
#
#    -  Train as an autoencoder
#    -  Save only the Encoder network
#    -  Train a new Decoder for translation from there
#
