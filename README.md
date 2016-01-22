# Introduction

This repository contains our implementation to classify quality of Wikipedia articles by using Doc2Vec and Deep Neural Networks (DNN).

## Repository structure

All the implementation is stored in the directory ``code``.

There are 2 Python programs:
- preprocess.py
- dnn_classifier.py

And other data files.

### Requirements:

The following tools / libraries are required to execute the program:

- python 2.7 (suggested to use [Anaconda distribution](https://anaconda.org/))
- [gensim](https://radimrehurek.com/gensim/index.html)
- [tensorflow](https://www.tensorflow.org)
- [skflow](https://github.com/tensorflow/skflow)

# Running the implementation:

The implementation is separated to 2 phases: preprocessing (convert documents to vectors by using Doc2Vec) and classification (classify documents to quality classes)

## Preprocessing phase

The raw data are contained in TXT files (for each quality class, and for train and test set). Because github limits the size of an individual file to 100MB, I have to compress file "train_fa.txt" to "train_fa.txt.zip". You should unzip this file before move on.

```bash
unzip train_fa.txt.zip
```

Then

```bash
python preprocess.py
```

The program will read all TXT file, and use Doc2Vec to convert them to vectors with size of 500.

At the first time, the program will save the model to "enwiki_quality.d2v" file so you can reuse in the future.

**Be careful**

Please note that the ``preprocess.py`` takes a lot of time (several hours on my Macbook Mid-2014) and request a lot of memory. Please prepare for it.

## Classifying

After the preprocessing phase finish, you can run the classifier

```bash
python dnn_classifier.py
```

The program will load the trained vectors (as the output of the previous phase), apply DNN and report the accuracy.

The log will be stored in ``logdir/doc2vec_dnn``.