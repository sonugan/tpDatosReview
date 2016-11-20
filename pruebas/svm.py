# -*- coding: utf-8 -*-,
import csv
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
import helpers as helper
from sklearn import svm
from sklearn import cross_validation
from nltk.corpus import stopwords

stopwordlist = stopwords.words('english')

clf = svm.LinearSVC()
if True:
    #
    target, data = helper.get_train_data('../../Data/train_prep.csv', vectorizer = helper.get_vectorizer(min_df = 3, ngram_range = (2,2)), pred_pos=6, text_pos=9,tf_idf=True, remove_header=True)
    print "NSP + NHTML + LOW + MIN3 + BIG + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

if False:
    #[ 0.8220164   0.82262331  0.82218015] --> Mejor
    target, data = helper.get_train_data('../../Data/train.csv', vectorizer = helper.get_vectorizer(min_df = 3, ngram_range = (2,2)), pred_pos=6, text_pos=9,tf_idf=True, remove_header=True)
    print "LOW + MIN3 + BIG + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

if False:
    #
    target, data = helper.get_train_data('../../Data/train.csv', vectorizer = helper.get_vectorizer(min_df = 3, ngram_range = (3,3)), pred_pos=6, text_pos=9,tf_idf=True, remove_header=True)
    print "LOW + MIN3 + TRIG + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

if False:
    #
    target, data = helper.get_train_data('../../Data/train.csv', vectorizer = helper.get_vectorizer(min_df = 4, ngram_range = (2,2)), pred_pos=6, text_pos=9,tf_idf=True, remove_header=True)
    print "LOW + MIN4 + BIG + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

if False:
    #
    target, data = helper.get_train_data('../../Data/train.csv', vectorizer = helper.get_vectorizer(min_df = 5, ngram_range = (2,2)), pred_pos=6, text_pos=9,tf_idf=True, remove_header=True)
    print "LOW + MIN5 + BIG + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

if False:
    #
    target, data = helper.get_train_data('../../Data/train_prep.csv', vectorizer = helper.get_vectorizer(min_df = 5, ngram_range = (2,2)), pred_pos=0, text_pos=1,tf_idf=True, remove_header=True)
    print "NP + NHTML + LOW + MIN5 + BIG + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

if False:
    #
    target, data = helper.get_train_data('../../Data/train.csv', vectorizer = helper.get_vectorizer(min_df = 3, ngram_range = (1,1)), pred_pos=6, text_pos=9,tf_idf=True, remove_header=True)
    print "LOW + MIN3 + UNI + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

####
if False:
    #
    target, data = helper.get_train_data('../../Data/train.csv', vectorizer = helper.get_vectorizer(stop_words=stopwordlist, min_df = 3, ngram_range = (2,2)), pred_pos=6, text_pos=9,tf_idf=True)
    print "LOW + MIN3 + BIG + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

if False:
    #
    target, data = helper.get_train_data('../../Data/train.csv', vectorizer = helper.get_vectorizer(stop_words=stopwordlist, min_df = 3, ngram_range = (1,1)), pred_pos=6, text_pos=9,tf_idf=True)
    print "LOW + MIN3 + UNI + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)