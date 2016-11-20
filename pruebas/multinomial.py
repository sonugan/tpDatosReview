# -*- coding: utf-8 -*-,
import csv
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
import helpers as helper
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from nltk.corpus import stopwords

stopwordlist = stopwords.words('english')

clf = MultinomialNB(alpha=0.01)
if False:
    #[ 0.79903422  0.79911206  0.79957912]
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
    #[ 0.79754201  0.79762117  0.79744831]
    target, data = helper.get_train_data('../../Data/train.csv', vectorizer = helper.get_vectorizer(min_df = 4, ngram_range = (2,2)), pred_pos=6, text_pos=9,tf_idf=True, remove_header=True)
    print "LOW + MIN4 + BIG + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

if False:
    #[ 0.79539802  0.7945998   0.79562756]
    target, data = helper.get_train_data('../../Data/train.csv', vectorizer = helper.get_vectorizer(min_df = 5, ngram_range = (2,2)), pred_pos=6, text_pos=9,tf_idf=True, remove_header=True)
    print "LOW + MIN5 + BIG + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

if False:
    #[ 0.79427655  0.79444148  0.79476861]
    target, data = helper.get_train_data('../../Data/train_prep.csv', vectorizer = helper.get_vectorizer(min_df = 5, ngram_range = (2,2)), pred_pos=0, text_pos=1,tf_idf=True, remove_header=True)
    print "NP + NHTML + LOW + MIN5 + BIG + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

if False:
    #[ 0.69582484  0.69573908  0.69606692]
    target, data = helper.get_train_data('../../Data/train.csv', vectorizer = helper.get_vectorizer(min_df = 3, ngram_range = (1,1)), pred_pos=6, text_pos=9,tf_idf=True, remove_header=True)
    print "LOW + MIN3 + UNI + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

####
if False:
    #[ 0.78581418  0.785318    0.78801472]
    target, data = helper.get_train_data('../../Data/train.csv', vectorizer = helper.get_vectorizer(stop_words=stopwordlist, min_df = 3, ngram_range = (2,2)), pred_pos=6, text_pos=9,tf_idf=True)
    print "LOW + MIN3 + BIG + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)

if False:
    #[ 0.69826108  0.69849657  0.69843521]
    target, data = helper.get_train_data('../../Data/train.csv', vectorizer = helper.get_vectorizer(stop_words=stopwordlist, min_df = 3, ngram_range = (1,1)), pred_pos=6, text_pos=9,tf_idf=True)
    print "LOW + MIN3 + UNI + TFIDF"
    this_scores = cross_validation.cross_val_score(clf, data, target, cv = 3)
    print(this_scores)