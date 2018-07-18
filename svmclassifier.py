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

# FAKE_FILE = ["train.txt", "dev.txt", "test.txt"]
# FAKE_FILE = ["train.txt", "dev.txt", "Spoof_SatireWorld.txt"]
FAKE_FILE = ["train.txt", "dev.txt", "DM_HP_satireNews.txt"]

'''
TRUE_FILE = ["true_train_1.txt", "true_train_2.txt","true_train_3.txt",
             "true_train_4.txt", "true_train_5.txt", "true_train_6.txt",
             "true_validation_1.txt", "true_validation_2.txt", 
             "true_test_1.txt", "true_test_2.txt"]
'''
'''
TRUE_FILE = ["true_train_1.txt", "true_train_2.txt","true_train_3.txt",
             "true_train_4.txt", "true_train_5.txt", "true_train_6.txt",
             "true_validation_1.txt", "true_validation_2.txt", 
             "Cnn_out.txt", "fox_out.txt"]
'''
TRUE_FILE = ["true_train_1.txt", "true_train_2.txt","true_train_3.txt",
             "true_train_4.txt", "true_train_5.txt", "true_train_6.txt",
             "true_validation_1.txt", "true_validation_2.txt", 
             "ABC_trueNews1.txt", "ABC_trueNews2.txt"]


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
        data = makedata(text_path, 1)
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
        train += makedata(text_path, 0)
    logging.warning('loading {} true data'.format(len(train)))
        
    for i in range(6, 8):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        dev += makedata(text_path, 0)
    logging.warning('loading {} true data'.format(len(dev)))
        
    for i in range(8, 10):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        test += makedata(text_path, 0)
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
def plot_coefficients(classifier, top_features=20):
    coefs = get_coeffs(classifier)
    feature_names = classifier.named_steps['vect'].get_feature_names()
    feature_names = np.array(feature_names)

    with open('/homes/du113/scratch/satire-models/features.p', 'wb') as fid:
        dump({'coefs':coefs, 'fn':feature_names}, fid)

    fig, ax = plt.subplots(nrows=len(coefs), ncols=1, figsize=(10, 5))

    for i, coef in enumerate(coefs):
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # for debugging
        # logging.warning(top_coefficients.shape)
        # logging.warning(top_coefficients.dtype)

        # create plot
        # plt.figure(figsize=(15, 10))
        
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        ax[i].bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)

        ax[i].set_xticks(np.arange(1, 1 + 2 * top_features))
        ax[i].set_xticklabels(feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()


# visualizing features
def print_coefficients(classifier, top_features=20):
    coefs = get_coeffs(classifier)
    feature_names = classifier.named_steps['vect'].get_feature_names()
    feature_names = np.array(feature_names)

    with open('/homes/du113/scratch/satire-models/features.p', 'wb') as fid:
        dump({'coefs':coefs, 'fn':feature_names}, fid)

    coef = coefs[0]
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    
    logging.warning('top fake coefs')
    logging.warning(feature_names[top_negative_coefficients])
    logging.warning('top true coefs')
    logging.warning(feature_names[top_positive_coefficients])


def get_most_frequent(data, label, topk=20):
    fake_counter = CountVectorizer(ngram_range=(1,2))
    true_counter = CountVectorizer(ngram_range=(1,2))

    data, label = np.array(data), np.array(label, dtype='bool')
    true_data = data[label]
    fake_data = data[np.invert(label)]
    fake = fake_counter.fit_transform(fake_data)
    true = true_counter.fit_transform(true_data)

    fake_grams = np.array(fake_counter.get_feature_names())
    # for debugging
    # logging.warning(fake_grams.shape)
    # logging.warning(fake_grams.dtype)

    fake_count = np.asarray(fake.sum(axis=0)).squeeze()

    # for debugging
    # logging.warning(fake_count.shape)
    # logging.warning(fake_count.dtype)

    true_grams = np.array(true_counter.get_feature_names())
    true_count = np.asarray(true.sum(axis=0)).squeeze()

    top_fake = np.argsort(fake_count)[-topk:]

    # for debugging
    # logging.warning(top_fake.shape)
    # logging.warning(top_fake.dtype)

    top_true = np.argsort(true_count)[-topk:]

    logging.warning('top fake words:')
    logging.warning(fake_grams[top_fake])
    logging.warning('top true words:')
    logging.warning(true_grams[top_true])


def train(dataset, p=1.0):
    # input: dataset, portion of the data to use
    # return: model
    args = parse()
    model = args.model
    save = args.save

    if model is not None:
        logging.warning('loading model from ' + model)
        with open('/homes/du113/scratch/satire-models/' + model, 'rb') as fid:
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

            # debugging
            # get_most_frequent(train_text, train_label)

        label_distr = Counter(train_label)
        logging.warning('satire sample: {}'.format(label_distr[1]))
        logging.warning('true sample: {}'.format(label_distr[0]))

        logging.warning('start training')
        satire_clf.fit(train_text, train_label)

        if save:
            logging.warning('saving model to {}'.format(save))
            # save the classifier
            with open('/homes/du113/scratch/satire-models/' + save, 'wb') as fid:
                dump(satire_clf, fid)

    # plot_coefficients(satire_clf)
    # print_coefficients(satire_clf)

    return satire_clf


def get_coeffs(satire_clf):
    # return: list of np arrays (n_estimators x n_dimensions)
    coefs = [e.coef_.ravel() for e in satire_clf.named_steps['clf'].estimators_]
    return coefs
    # plot_coefficients(satire_clf, satire_vec.get_feature_names())

def validate(satire_clf, dataset):
    dev_text, dev_label = dataset['dev']
    test_text, test_label = dataset['test']

    # debugging
    # get_most_frequent(dev_text, dev_label)
    # test on the validation data
    dev_pred = satire_clf.predict(dev_text)

    dev_lab_distr = Counter(dev_label)
    logging.warning('satire lab: {}'.format(dev_lab_distr[1]))
    logging.warning('true lab: {}'.format(dev_lab_distr[0]))
    pred_distr = Counter(dev_pred)
    logging.warning('satire predictions: {}'.format(pred_distr[1]))
    logging.warning('true predictions: {}'.format(pred_distr[0]))

    acc = accuracy_score(dev_label, dev_pred)
    prec = precision_score(dev_label, dev_pred)
    rec = recall_score(dev_label, dev_pred)
    f1 = f1_score(dev_label, dev_pred)

    cm = confusion_matrix(dev_label, dev_pred)
    logging.warning('TP: {}\tFP: {}\tTN: {}\tFN: {}'.format(cm[0,0], cm[1,0], cm[1,1], cm[0,1]))

    logging.warning('dev acc: {}\tprec: {}\trec: {}\tf1: {}'.format(acc, prec, rec, f1))
    print('dev acc: {}\tprec: {}\trec: {}\tf1: {}'.format(acc, prec, rec, f1))

    # debugging
    # get_most_frequent(test_text, test_label)
    # test on the test data
    test_pred = satire_clf.predict(test_text)

    test_lab_distr = Counter(test_label)
    logging.warning('satire lab: {}'.format(test_lab_distr[1]))
    logging.warning('true lab: {}'.format(test_lab_distr[0]))
    pred_distr = Counter(test_pred)
    logging.warning('satire predictions: {}'.format(pred_distr[1]))
    logging.warning('true predictions: {}'.format(pred_distr[0]))

    acc = accuracy_score(test_label, test_pred)
    prec = precision_score(test_label, test_pred)
    rec = recall_score(test_label, test_pred)
    f1 = f1_score(test_label, test_pred)

    cm = confusion_matrix(test_label, test_pred)
    logging.warning('TP: {}\tFP: {}\tTN: {}\tFN: {}'.format(cm[0,0], cm[1,0], cm[1,1], cm[0,1]))

    logging.warning('test acc: {}\tprec: {}\trec: {}\tf1: {}'.format(acc, prec, rec, f1))
    print('test acc: {}\tprec: {}\trec: {}\tf1: {}'.format(acc, prec, rec, f1))


def main():
    step = 0.001
    dataset = loaddata()

    # for i in range(6):
    # svm = train(dataset, min(1, step * np.exp(i)))
    svm = train(dataset)
    validate(svm, dataset)


if __name__ == '__main__':
    main()

