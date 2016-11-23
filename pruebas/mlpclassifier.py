# -*- coding: utf-8 -*-,
import csv
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
import helpers as helper
from sklearn.neural_network import MLPClassifier
from sklearn import cross_validation
from nltk.corpus import stopwords
##
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
##
stopwordlist = stopwords.words('english')

if True:
    clf = MLPClassifier(verbose=True, tol=0.001, learning_rate="adaptive", max_iter=10, early_stopping=True, alpha=0.0001, hidden_layer_sizes=(5, 5, 10, 5), random_state=1)
    
    train_y, train_x = helper.get_train_data('../../Data/train_prep.csv', vectorizer = helper.get_vectorizer(stop_words=stopwordlist, min_df = 3, ngram_range = (2,2)), pred_pos=0, text_pos=1,tf_idf=True, remove_header=False)
    ids, test_x = helper.get_train_data('../../Data/test_prep.csv', vectorizer = helper.get_vectorizer(stop_words=stopwordlist, min_df = 3, ngram_range = (2,2)),tf_idf=True, remove_header=True)

    print "LOW + MIN3 + BIG + TFIDF"
    #x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=0)
    clf.fit(train_x, train_y)
    y_predicted = enumerate(clf.predict(test_x))
    helper.saveKaggleFile('../kaggle/mlpclassifier_test.csv', y_predicted, ids)


    