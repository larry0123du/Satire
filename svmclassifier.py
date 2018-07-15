# coding: utf-8

# from io import open
# import cPickle
from pickle import load, dump

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import BaggingClassifier

import numpy as np

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default=None)

args = parser.parse_args()
model = args.model

# File names
BASE = '/homes/du113/scratch/data/'
FAKE_TEXT = BASE + "text/fake"
TRUE_TEXT = BASE + "text/true"

# FAKE_FILE = ["train.txt", "dev.txt", "test.txt"]
FAKE_FILE = ["train.txt", "dev.txt", "Spoof_SatireWorld.txt"]

'''
TRUE_FILE = ["true_train_1.txt", "true_train_2.txt","true_train_3.txt",
             "true_train_4.txt", "true_train_5.txt", "true_train_6.txt",
             "true_validation_1.txt", "true_validation_2.txt", 
             "true_test_1.txt", "true_test_2.txt"]
'''
TRUE_FILE = ["true_train_1.txt", "true_train_2.txt","true_train_3.txt",
             "true_train_4.txt", "true_train_5.txt", "true_train_6.txt",
             "true_validation_1.txt", "true_validation_2.txt", 
             "Cnn_out.txt", "fox_out.txt"]


# helpers
def makedata(text_path, label):
    with open(text_path) as f:
        # data = f.read().strip().split('******\n')
        lines = f.readlines()
        
    data = []
    doc = ''

    for line in lines:
        if line == '******\n':
            data.append((doc, label))
            doc = ''
        else:
            doc += line

    # data = list(zip(data, [label for _ in data]))
    return data
    
def load_fake():
    fake = []
    for file_name in FAKE_FILE:
        text_path = os.path.join(FAKE_TEXT, file_name)
        data = makedata(text_path, 0)
        print('loading {} fake data'.format(len(data)))
        fake.append(data)
    return fake[0], fake[1], fake[2]

def load_true():
    train = []
    dev = []
    test = []
    for i in range(6):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        train += makedata(text_path, 1)
    print('loading {} true data'.format(len(train)))
        
    for i in range(6, 8):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        dev += makedata(text_path, 1)
    print('loading {} true data'.format(len(dev)))
        
    for i in range(8, 10):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        test += makedata(text_path, 1)
    print('loading {} true data'.format(len(test)))
        
    return train, dev, test


def test_write(data):
    with open('test.txt', 'w') as f:
        f.write('\n******\n'.join(data))
# In[19]:


import os
fake_train, fake_dev, fake_test = load_fake()

'''
#debugging purpose
fake_test_text, _ = zip(*fake_test)
test_write(fake_test_text)

raise Exception
'''

true_train, true_dev, true_test = load_true()


# In[22]:


import random
from random import shuffle

def prepare(data):
    # input: data (a list of (text, label))
    # output: a list of text and another list of label
    shuffle(data)
    text, label = zip(*data)
    return text, label


# In[27]:


train, dev, test = fake_train + true_train, fake_dev + true_dev, fake_test + true_test
train_text, train_label = prepare(train)
dev_text, dev_label = prepare(dev)
test_text, test_label = prepare(test)

'''
print test_text[1]
print test_label[1]
'''

# visualizing features
def plot_coefficients(classifier, feature_names, top_features=20):
    # coef = classifier.coef_.ravel()
    # coef = np.array(coef)
    coefs = [e.coef_.ravel() for e in classifier.estimators_]

    # coefs[0].shape

    coef = coefs[0]
    
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    '''
    with open('svm_top.pkl', 'wb') as fid:
        cPickle.dump(top_coefficients, fid)
    '''

    # print top_coefficients.shape
    # create plot
    plt.figure(figsize=(15, 10))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    # print feature_names.shape

    # plt.subplot(len(coefs), 1, i)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()

# In[30]:

if model is not None:
    print('loading model from', model)
    with open(model, 'rb') as fid:
        satire_clf = load(fid)

else:
    print('initializing and training svm')
    c = 1e-3
    # for c in [1e-1, 1e-2, 1e-3, 1e-4]:
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
    '''
    satire_vec = CountVectorizer(ngram_range=(1,2))
    train_text = satire_vec.fit_transform(train_text)
    # print satire_vec.get_feature_names()

    satire_clf = BaggingClassifier(
            LinearSVC(C=c,class_weight='balanced',
                loss='hinge',
                verbose=2,
                max_iter=100,
                random_state=42),
            max_samples=1.0/10,
            n_jobs=-1,
            verbose=1) 
    '''
    satire_clf.fit(train_text, train_label)

    # save the classifier
    with open('/homes/du113/scratch/satire-models/py3_best_svm_july13.pkl', 'wb') as fid:
        dump(satire_clf, fid)

coefs = sum([list(e.coef_.ravel()) for e in satire_clf.named_steps['clf'].estimators_], [])
# print len(coefs)
'''
# print len(satire_clf.get_params(deep=True))
# plot_coefficients(satire_clf, satire_vec.get_feature_names())
'''
# test on the validation data
dev_pred = satire_clf.predict(dev_text)

acc = accuracy_score(dev_label, dev_pred)
prec = precision_score(dev_label, dev_pred)
rec = recall_score(dev_label, dev_pred)
f1 = f1_score(dev_label, dev_pred)

print('dev acc: {}\tprec: {}\trec: {}\tf1: {}'.format(acc, prec, rec, f1))

# test on the test data
test_pred = satire_clf.predict(test_text)

acc = accuracy_score(test_label, test_pred)
prec = precision_score(test_label, test_pred)
rec = recall_score(test_label, test_pred)
f1 = f1_score(test_label, test_pred)

print('test acc: {}\tprec: {}\trec: {}\tf1: {}'.format(acc, prec, rec, f1))

