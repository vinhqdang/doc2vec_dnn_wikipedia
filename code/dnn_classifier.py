import csv
import numpy as np
from sklearn import metrics, cross_validation
import pandas

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import skflow

import random
import time

import cPickle as pickle

from nltk.corpus import stopwords

import string

### Training data

qualities = ["STUB","START","C","B","GA","FA"]

print('Read labels')

def read_label (file_name):
    Y = []
    with open(file_name) as f:
        for line in f:
            label = line.strip().replace('\n','')
            Y.append (qualities.index (label))
    return np.asarray (Y)

Y_train = read_label ("doc2vec_train_label.txt")
Y_test = read_label ("doc2vec_test_label.txt")

print('Read content')

def read_wiki_content (file_name):
    X = []
    with open(file_name) as f:
        for line in f:
            _arr = line.split ()
            X.append (map (float, _arr))
    return np.asarray (X)

X_train = read_wiki_content ("doc2vec_train_content.txt")
X_test = read_wiki_content ("doc2vec_test_content.txt")

print ("Dimension of input: ", len(X_train[0]))


# pickle.dump (X_train, open ("xtrain.p", b))
# pickle.dump (X_test, open ("xtest.p", b))

# X_train = pickle.load (open ("xtrain.p", rb))
# X_test = pickle.load (open ("xtest.p", rb))

### Process vocabulary

# print('Process vocabulary')

# MAX_DOCUMENT_LENGTH = 1000

# vocab_processor = skflow.preprocessing.VocabularyProcessor(max_document_length = MAX_DOCUMENT_LENGTH, min_frequency = 0)
# X_train = np.array(list(vocab_processor.fit_transform(X_train)))
# X_test = np.array(list(vocab_processor.transform(X_test)))

# n_words = len(vocab_processor.vocabulary_)
# print('Total words: %d' % n_words)

# pickle.dump (X_train, open ("xtrain.p", b))
# pickle.dump (X_test, open ("xtest.p", b))

# X_train = pickle.load (open ("xtrain.p", rb))
# X_test = pickle.load (open ("xtest.p", rb))

### Models

# print('Build model')

# EMBEDDING_SIZE = 300

# # def average_model(X, y):
# #     word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
# #         embedding_size=EMBEDDING_SIZE, name='words')
# #     features = tf.reduce_max(word_vectors, reduction_indices=1)
# #     return skflow.models.logistic_regression(features, y)

# n_words = 500

# def rnn_model(X, y):
#     """Recurrent neural network model to predict from sequence of words
#     to a class."""
#     # Convert indexes of words into embeddings.
#     # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
#     # maps word indexes of the sequence into [batch_size, sequence_length,
#     # EMBEDDING_SIZE].
#     word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
#         embedding_size=EMBEDDING_SIZE, name='words')
#     # Split into list of embedding per word, while removing doc length dim.
#     # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
#     word_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
#     # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
#     cell = rnn_cell.GRUCell(EMBEDDING_SIZE)
#     # Create an unrolled Recurrent Neural Networks to length of
#     # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
#     _, encoding = rnn.rnn(cell, word_list, dtype=tf.float32)
#     # Given encoding of RNN, take encoding of last step (e.g hidden size of the
#     # neural network of last step) and pass it as features for logistic
#     # regression over output classes.
#     return skflow.models.logistic_regression(encoding[-1], y)

# classifier = skflow.TensorFlowEstimator(model_fn=rnn_model, n_classes=6,
#     steps=100, optimizer='Adam', learning_rate=0.01, continue_training=True)

# # Continuesly train for 1000 steps & predict on test set.

# max_acc = 0
# max_confusion_matrix = None
# max_cnt = 0
# cnt = 0
# # while True:
# for i in range (20):
#     cnt = cnt + 1
#     classifier.fit(X_train, Y_train, logdir="./logdir/enwiki_classification_rnn_words")
#     prediction = classifier.predict(X_test)
#     score = metrics.accuracy_score(Y_test, prediction)
#     print('Accuracy: {0:f}'.format(score))
#     confusion_matrix = metrics.confusion_matrix (Y_test, prediction)
#     print(confusion_matrix)

#     if (score > max_acc):
#         max_acc = score
#         max_confusion_matrix = confusion_matrix
#         max_cnt = cnt

print ('Using DNN')
hidden_units = [2000,1000,500,200]
steps = 50000
early_stopping_rounds = 5000

print ("Parameters: ", hidden_units, " steps = ", steps, "   early_stopping_rounds = ", early_stopping_rounds)
classifier = skflow.TensorFlowDNNClassifier(hidden_units=hidden_units,
                                            n_classes=6, steps=steps,
                                            early_stopping_rounds=early_stopping_rounds)
print ('Fit model')
classifier.fit(X_train, Y_train, logdir = "./logdir/doc2vec_dnn")

print ('Predicting')
prediction = classifier.predict(X_test)

score2 = metrics.accuracy_score(prediction, Y_test)

confusion_matrix = metrics.confusion_matrix (Y_test, prediction)

print (confusion_matrix)
print (score2)