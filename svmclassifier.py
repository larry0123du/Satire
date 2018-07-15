# coding: utf-8

# from io import open
# import cPickle
import os
import random
from random import shuffle


from pickle import load, dump

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.ensemble import BaggingClassifier

import numpy as np

import matplotlib.pyplot as plt

import argparse

import logging

from collections import Counter

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)


# File names
BASE = '/homes/du113/scratch/data/'
FAKE_TEXT = BASE + "text/fake"
TRUE_TEXT = BASE + "text/true"

FAKE_FILE = ["train.txt", "dev.txt", "test.txt"]
# FAKE_FILE = ["train.txt", "dev.txt", "Spoof_SatireWorld.txt"]

TRUE_FILE = ["true_train_1.txt", "true_train_2.txt","true_train_3.txt",
             "true_train_4.txt", "true_train_5.txt", "true_train_6.txt",
             "true_validation_1.txt", "true_validation_2.txt", 
             "true_test_1.txt", "true_test_2.txt"]
'''
TRUE_FILE = ["true_train_1.txt", "true_train_2.txt","true_train_3.txt",
             "true_train_4.txt", "true_train_5.txt", "true_train_6.txt",
             "true_validation_1.txt", "true_validation_2.txt", 
             "Cnn_out.txt", "fox_out.txt"]
'''


def parse():
    # 2 args
    # use pretrained model
    # save the model
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('-s', '--save', type=str, default=None)

    args = parser.parse_args()
    return args


# helpers
def makedata(text_path, label):
    # input: path to the data file, correct label
    # output: a list of tuples
    with open(text_path) as f:
        lines = f.readlines()
        
    data = []
    doc = [] 

    for line in lines:
        if line == '******\n':
            data.append((' '.join(doc), label))
            doc = [] 
        else:
            doc.append(line.strip())

    return data
    
def load_fake():
    # output: 3 list of tuples
    fake = []
    for file_name in FAKE_FILE:
        text_path = os.path.join(FAKE_TEXT, file_name)
        data = makedata(text_path, 0)
        logging.warning('loading {} fake data'.format(len(data)))
        fake.append(data)
    return fake[0], fake[1], fake[2]

def load_true():
    # output: 3 list of tuples
    train = []
    dev = []
    test = []
    for i in range(6):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        train += makedata(text_path, 1)
    logging.warning('loading {} true data'.format(len(train)))
        
    for i in range(6, 8):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        dev += makedata(text_path, 1)
    logging.warning('loading {} true data'.format(len(dev)))
        
    for i in range(8, 10):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        test += makedata(text_path, 1)
    logging.warning('loading {} true data'.format(len(test)))
        
    return train, dev, test


def test_write(data):
    with open('test.txt', 'w') as f:
        f.write('\n******\n'.join(data))


def loaddata(): 
    # return: a dictionary of pairs of lists
    fake_train, fake_dev, fake_test = load_fake()
    true_train, true_dev, true_test = load_true()

    train, dev, test = fake_train + true_train, fake_dev + true_dev, fake_test + true_test
    train_text, train_label = prepare(train)
    dev_text, dev_label = prepare(dev)
    test_text, test_label = prepare(test)

    return {'train': (train_text, train_label), \
            'dev': (dev_text, dev_label), \
            'test': (test_text, test_label)}


def prepare(data):
    # input: data (a list of (text, label))
    # output: a list of text and another list of label
    shuffle(data)
    text, label = zip(*data)
    return text, label


# visualizing features
def plot_coefficients(classifier, feature_names, top_features=20):
    coefs = [e.coef_.ravel() for e in classifier.estimators_]

    coef = coefs[0]
    
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    # create plot
    plt.figure(figsize=(15, 10))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)

    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()


def train(dataset, p=1.0):
    # input: dataset, portion of the data to use
    # return: model
    args = parse()
    model = args.model
    save = args.save

    if model is not None:
        logging.warning('loading model from ' + model)
        with open(model, 'rb') as fid:
            satire_clf = load(fid)

    else:
        logging.warning('initializing and training svm')
        c = 1e-3
        satire_clf = Pipeline([
            ('vect',CountVectorizer(ngram_range=(1,2))),
            ('clf',BaggingClassifier(
                LinearSVC(C=c,class_weight='balanced',
                    loss='hinge',
                    verbose=2,
                    max_iter=200,
                    random_state=42),
                max_samples=1.0/10,
                n_jobs=-1,
                verbose=1)),
        ])

        train_text, train_label = dataset['train']

        # only use a portion of the training data
        if p < 1.0:
            logging.warning('using {} of the training data'.format(p))
            num = int(len(train_label) * p)
            train_text, train_label = train_text[:num], train_label[:num]

        label_distr = Counter(train_label)
        logging.warning('satire sample: {}'.format(label_distr[0]))
        logging.warning('true sample: {}'.format(label_distr[1]))

        logging.warning('start training')
        satire_clf.fit(train_text, train_label)

        if save:
            logging.warning('saving model to {}'.format(save))
            # save the classifier
            with open('/homes/du113/scratch/satire-models/' + save, 'wb') as fid:
                dump(satire_clf, fid)

    return satire_clf


def get_coeffs(satire_clf):
    coefs = sum([list(e.coef_.ravel()) for e in satire_clf.named_steps['clf'].estimators_], [])
    # print len(coefs)
    '''
    # print len(satire_clf.get_params(deep=True))
    # plot_coefficients(satire_clf, satire_vec.get_feature_names())
    '''

def validate(satire_clf, dataset):
    dev_text, dev_label = dataset['dev']
    test_text, test_label = dataset['test']

    # test on the validation data
    dev_pred = satire_clf.predict(dev_text)

    dev_lab_distr = Counter(dev_label)
    logging.warning('satire lab: {}'.format(dev_lab_distr[0]))
    logging.warning('true lab: {}'.format(dev_lab_distr[1]))
    pred_distr = Counter(dev_pred)
    logging.warning('satire predictions: {}'.format(pred_distr[0]))
    logging.warning('true predictions: {}'.format(pred_distr[1]))

    acc = accuracy_score(dev_label, dev_pred)
    prec = precision_score(dev_label, dev_pred)
    rec = recall_score(dev_label, dev_pred)
    f1 = f1_score(dev_label, dev_pred)

    cm = confusion_matrix(dev_label, dev_pred)
    logging.warning('TP: {}\tFP: {}\tTN: {}\tFN: {}'.format(cm[1,1], cm[0,1], cm[0,0], cm[1,0]))

    logging.warning('dev acc: {}\tprec: {}\trec: {}\tf1: {}'.format(acc, prec, rec, f1))
    print('dev acc: {}\tprec: {}\trec: {}\tf1: {}'.format(acc, prec, rec, f1))

    # test on the test data
    test_pred = satire_clf.predict(test_text)

    test_lab_distr = Counter(test_label)
    logging.warning('satire lab: {}'.format(test_lab_distr[0]))
    logging.warning('true lab: {}'.format(test_lab_distr[1]))
    pred_distr = Counter(test_pred)
    logging.warning('satire predictions: {}'.format(pred_distr[0]))
    logging.warning('true predictions: {}'.format(pred_distr[1]))

    acc = accuracy_score(test_label, test_pred)
    prec = precision_score(test_label, test_pred)
    rec = recall_score(test_label, test_pred)
    f1 = f1_score(test_label, test_pred)

    cm = confusion_matrix(test_label, test_pred)
    logging.warning('TP: {}\tFP: {}\tTN: {}\tFN: {}'.format(cm[1,1], cm[0,1], cm[0,0], cm[1,0]))

    logging.warning('test acc: {}\tprec: {}\trec: {}\tf1: {}'.format(acc, prec, rec, f1))
    print('test acc: {}\tprec: {}\trec: {}\tf1: {}'.format(acc, prec, rec, f1))


def main():
    step = 0.001
    dataset = loaddata()

    for i in range(6):
        svm = train(dataset, min(1, step * np.exp(i)))
        validate(svm, dataset)


if __name__ == '__main__':
    main()

