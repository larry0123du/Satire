import os
import math
import nltk
# nltk.download('all')
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.data import load
import re
import textacy
import textacy.datasets
import spacy
from collections import Counter
import logging


FAKE_TEXT = "data/text/fake"
TRUE_TEXT = "data/text/true"
FAKE_DOC = "data/doc/norm_fake"
TRUE_DOC = "data/doc/norm_true"
FAKE_SENT = "data/sent/norm_fake"
TRUE_SENT = "data/sent/norm_true"
FAKE_FILE = ["train.txt", "dev.txt", "test.txt"]
TRUE_FILE = ["true_train_1.txt", "true_train_2.txt", "true_train_3.txt", "true_train_4.txt", "true_train_5.txt",
             "true_train_6.txt", "true_validation_1.txt", "true_validation_2.txt", "true_test_1.txt", "true_test_2.txt"]

homedic = os.getcwd()
tagdict = load('help/tagsets/upenn_tagset.pickle') # POS tag keys


def readability_f(text):
	text_ = textacy.Doc(text)
	ts = textacy.TextStats(text_)
	Flesch_reading_ease_index = ts.readability_stats['flesch_reading_ease']
	Gunning_fog_index = ts.readability_stats['gunning_fog_index']
	Automated_index = ts.readability_stats['automated_readability_index']
	ColemanLiau_index = ts.readability_stats['coleman_liau_index']
	return [Flesch_reading_ease_index, Gunning_fog_index, Automated_index, ColemanLiau_index]


def LIWC_f(text):
	pass


def POS_f(text):
	word_list = nltk.pos_tag(word_tokenize(text))
	POS_dict = Counter([j for i,j in word_list])
	d = {}
	for key, value in POS_dict.items():
		d[key] = value
	
	d_final = {}
	for key in tagdict.keys():
		if key in d.keys():
			d_final[key] = d[key]
		else: 
			d_final[key] = 0
	return list(d_final.values())


def structural_f(text):
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)
	num_words = len(tokens)	
	log_num_words = math.log(num_words)
	num_words_puncts = len(word_tokenize(text))
	num_punctuations = num_words_puncts - num_words
	num_digits = sum(c.isdigit() for c in text)
	num_Cap_letters = len(re.findall(r'[A-Z]',text))
	num_sentences = len(sent_tokenize(text))

	# text_ = textacy.Doc(text)
	# ts = textacy.TextStats(text_)
	# print(ts.basic_counts.keys())
	# (['n_sents', 'n_words', 'n_chars', 'n_syllables', 'n_unique_words',/
	#  'n_long_words', 'n_monosyllable_words', 'n_polysyllable_words'])
	return [num_words, log_num_words, num_punctuations, num_digits, num_Cap_letters, num_sentences]

def ling_f(text):
	readability_feature = readability_f(text)
	POS_feature = POS_f(text)
	structural_feature = structural_f(text)
	all_feature = [readability_feature, POS_feature,structural_feature]
	all_feature = sum(all_feature, []) # 1d flatten list, dim = 55
	# print(all_feature,len(all_feature))
	return all_feature


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
            # print('doc appended')
            doc = []
        else:
            doc.append(line.strip())
    # print(len(docs),docs)
    return docs


def write_doc_ling_f(in_file_name,out_file_name):
	with open(out_file_name,'w') as f:
		for doc in load_sent(in_file_name):
			doc_ling = ling_f(str(doc))		
			f.writelines(str(doc_ling)+'\n') 


def write_para_ling_f(in_file_name,out_file_name):
    doc = []
    with open(in_file_name) as f:
        lines = f.readlines()
    with open(out_file_name,'w') as f:
        for line in lines:
            if line == "******\n":
                # docs.append(doc)
                f.writelines(str(doc)+'\n')
                f.writelines('******\n')
                # print('doc appended')
                doc = []
            else:
                para_feature = ling_f(line)
                doc.append(para_feature)

# for debug #
# text = 'I like NLP, and we 2 are doing it. I hope we can finish this project in 10 week.'
# ling_f(text)
# write_doc_ling_f('doc_ling.txt')
# write_para_ling_f('toydata_featuretest.txt','para_ling.txt')

def write_doc_para():
    logging.info("writing fake feature data...")
    for file_name in FAKE_FILE:
        text_path = os.path.join(FAKE_TEXT, file_name)
        sent_path = os.path.join(FAKE_SENT, file_name)
        doc_path = os.path.join(FAKE_DOC, file_name)
        write_doc_ling_f(text_path,doc_path)
        write_para_ling_f(text_path,sent_path)
    logging.info("complete establishing fake feature data...")
    logging.info("writing true feature data train...")

    for i in range(6):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        sent_path = os.path.join(TRUE_SENT, file_name)
        doc_path = os.path.join(TRUE_DOC, file_name)
        write_doc_ling_f(text_path,doc_path)
        write_para_ling_f(text_path,sent_path)

    logging.info("writing true feature data dev...")
    for i in range(6, 8):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        sent_path = os.path.join(TRUE_SENT, file_name)
        doc_path = os.path.join(TRUE_DOC, file_name)
        write_doc_ling_f(text_path,doc_path)
        write_para_ling_f(text_path,sent_path)

    logging.info("writing true feature data test...")
    for i in range(8, 10):
        file_name = TRUE_FILE[i]
        text_path = os.path.join(TRUE_TEXT, file_name)
        sent_path = os.path.join(TRUE_SENT, file_name)
        doc_path = os.path.join(TRUE_DOC, file_name)
        write_doc_ling_f(text_path,doc_path)
        write_para_ling_f(text_path,sent_path)

write_doc_para()

