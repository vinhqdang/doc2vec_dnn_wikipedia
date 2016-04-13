# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# shuffle
from random import shuffle

# logging
import logging
import os.path
import sys
import cPickle as pickle

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

#sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

sources = {'train_fa.txt':'TRAIN_FA',
            'train_ga.txt':'TRAIN_GA',
            'train_b.txt':'TRAIN_B',
            'train_c.txt':'TRAIN_C',
            'train_start.txt':'TRAIN_START',
            'train_stub.txt':'TRAIN_STUB'
            }

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=1, window=10, size=500, sample=1e-4, negative=5, workers=7)

model.build_vocab(sentences.to_array())

for epoch in range(20):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm())

model.save('./enwiki_quality_train.d2v')

def convert_array_to_string (data):
    res = ""
    for i in range(len(data)):
        res = res + str (data[i])
        if (i < len(data) - 1):
            res = res + '\t'
    return res

def write_array_to_file (file_name, array_data):
    f = open (file_name, "w")
    for i in range (len(array_data)):
        f.write (str(array_data[i]) + "\n")
    f.close ()

qualities = ['FA','GA','B','C','START','STUB']
train_labels = [0] * 23577
train_content_file = "doc2vec_train_content_separated.txt"
train_label_file = "doc2vec_train_label_separated.txt"
train_cnt = 0
for i in range (len(qualities)):
    for j in range (30000):
                key = 'TRAIN_' + qualities[i] + "_" + str(j)
                data = model.docvecs[key]
                if (len(data) == 500):
                    with open(train_content_file, "a") as myfile:
                        myfile.write(convert_array_to_string (data))
                        myfile.write("\n")
                    train_labels [train_cnt] = qualities[i]
                    train_cnt += 1

write_array_to_file (file_name = train_label_file, array_data = train_labels)

