import os

''' commented on jluy 11
FAKE_TEXT = "data/text/fake"
TRUE_TEXT = "data/text/true"
FAKE_DOC = "data/doc/norm_fake"
TRUE_DOC = "data/doc/norm_true"
FAKE_SENT = "data/sent/norm_fake"
TRUE_SENT = "data/sent/norm_true"
FAKE_FILE = ["train.txt", "dev.txt", "test.txt"]
TRUE_FILE = ["true_train_1.txt", "true_train_2.txt", "true_train_3.txt", "true_train_4.txt", "true_train_5.txt",
             "true_train_6.txt", "true_validation_1.txt", "true_validation_2.txt", "true_test_1.txt", "true_test_2.txt"]
'''

BASE = '/homes/du113/scratch/'
FAKE_TEXT = BASE + "data/text/fake"
TRUE_TEXT = BASE + "data/text/true"
FAKE_DOC = BASE + "data/doc/norm_fake"
TRUE_DOC = BASE + "data/doc/norm_true"
FAKE_SENT = BASE + "data/sent/norm_fake"
TRUE_SENT = BASE + "data/sent/norm_true"

# original data
'''
FAKE_FILE = ["train.txt", "dev.txt", "test.txt"]
TRUE_FILE = ["true_train_1.txt", "true_train_2.txt", "true_train_3.txt", "true_train_4.txt", "true_train_5.txt",
             "true_train_6.txt", "true_validation_1.txt", "true_validation_2.txt", "true_test_1.txt", "true_test_2.txt"]
'''
# new 1
'''
FAKE_FILE = ["train.txt","dev.txt","spoof_news_cat.txt"]
TRUE_FILE = ["true_train_1.txt", "true_train_2.txt", "true_train_3.txt", "true_train_4.txt", "true_train_5.txt",
             "true_train_6.txt", "true_validation_1.txt", "true_validation_2.txt", "Fox_news_2018_7_9.txt", "Cnn_news.txt"]
'''
# new 2
'''
FAKE_FILE = ["train.txt","dev.txt","Spoof_SatireWorld.txt"]
TRUE_FILE = ["true_train_1.txt", "true_train_2.txt", "true_train_3.txt", "true_train_4.txt", "true_train_5.txt",
             "true_train_6.txt", "true_validation_1.txt", "true_validation_2.txt", "fox_out.txt", "Cnn_out.txt"]
'''
# newest as of July 17th
FAKE_FILE = ["train.txt", "dev.txt", "DM_HP_satireNews.txt"]
TRUE_FILE = ["true_train_1.txt", "true_train_2.txt","true_train_3.txt",
             "true_train_4.txt", "true_train_5.txt", "true_train_6.txt",
             "true_validation_1.txt", "true_validation_2.txt", 
             "ABC_trueNews1.txt", "ABC_trueNews2.txt"]

homedic = os.getcwd()

def load_doc(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    return [l.strip() for l in lines] # delete space


def load_sent(file_name):
    docs = []
    doc = []
    with open(file_name) as f:
        lines = f.readlines()
    for line in lines:
        if line == "******\n":
            docs.append(doc)
            doc = []
        else:
            doc.append(line.strip())
    return docs


def load_feature_set(text_path, sent_path, doc_path, label, doc=False, sent_ling=True, doc_ling=True):
    text = load_sent(text_path)
    if doc:
        text = list2doc(text)

    # added july 18th
    if sent_ling:
        sent_feature = load_sent(sent_path)
    if doc_ling:
        doc_features = load_doc(doc_path)
    # print(text[0],len(text[1]))
    # print(len(text),len(sent_feature),len(doc_features))
    # print(type(doc_features))
    # print(doc_features)

    # assert len(text) == len(sent_feature) == len(doc_features)
    labels = [label for _ in range(len(text))]
    if sent_ling and doc_ling:
        return list(zip(text, sent_feature, doc_features, labels))
    elif sent_ling and not doc_ling:
        return list(zip(text, sent_feature, labels))
    elif not sent_ling and doc_ling:
        return list(zip(text, doc_features, labels))
    elif not sent_ling and not doc_ling:
        return list(zip(text, labels))


# text_path = os.path.join(homedic, 'toydata_featuretest.txt')
# sent_path = os.path.join(homedic, 'para_ling.txt')
# doc_path = os.path.join(homedic, 'doc_ling.txt')
# load_feature_set(text_path, sent_path, doc_path, label = 1, doc=False, sent_ling=True, doc_ling=True)



def load_true(doc=False, sent_ling=True, doc_ling=True):
    train = []
    dev = []
    test = []
    for i in range(6):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        sent_path = os.path.join(TRUE_SENT, file_name)
        doc_path = os.path.join(TRUE_DOC, file_name)
        train.append(load_feature_set(text_path, sent_path, doc_path, label=1,
                                      doc=doc, sent_ling=sent_ling, doc_ling=doc_ling))
    for i in range(6, 8):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        sent_path = os.path.join(TRUE_SENT, file_name)
        doc_path = os.path.join(TRUE_DOC, file_name)
        dev += load_feature_set(text_path, sent_path, doc_path, label=1,
                                doc=doc, sent_ling=sent_ling, doc_ling=doc_ling)
    for i in range(8, 10):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        sent_path = os.path.join(TRUE_SENT, file_name)
        doc_path = os.path.join(TRUE_DOC, file_name)
        test += load_feature_set(text_path, sent_path, doc_path, label=1,
                                 doc=doc, sent_ling=sent_ling, doc_ling=doc_ling)
    return train, dev, test


def load_fake(doc=False, sent_ling=True, doc_ling=True):
    fake = []
    for file_name in FAKE_FILE:
        text_path = os.path.join(FAKE_TEXT, file_name)
        sent_path = os.path.join(FAKE_SENT, file_name)
        doc_path = os.path.join(FAKE_DOC, file_name)
        features = load_feature_set(text_path, sent_path, doc_path, label=0,
                                    doc=doc, sent_ling=sent_ling, doc_ling=doc_ling)
        fake.append(features)
    return fake[0], fake[1], fake[2]


def list2doc(docs):
    """
        convert a list of sentences to a single document
        """
    doc_docs = []
    s = ''
    for doc in docs:
        if isinstance(doc, tuple):
            doc = doc[0]
        else:
            doc = doc
        for sent in doc:
            s += sent + ' '
        doc_docs.append(s)
        s = ''
    return doc_docs


def list2file(docs, file):
    with open(file, 'a') as f:
        for doc in docs:
            f.write(doc[0])
            f.write('\n')


if __name__ == '__main__':
    print()
    train, dev, test = load_fake(True, False, False)
    print(len(train), len(dev), len(test))
    list2file(train, "fake_doc_train.txt")
    list2file(dev, "fake_doc_dev.txt")
    list2file(test, "fake_doc_test.txt")

