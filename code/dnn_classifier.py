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