from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split as tts
import helpers as helper
import csv
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from imblearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin
from imblearn.under_sampling import NearMiss
from sklearn.cluster import KMeans

stopwordlist = stopwords.words('english')

def saveConfusionMatrix(m, path):
    with open(path, 'w') as f:
        f.write(np.array2string(m, separator=', '))

if True:
    vectorizer = CountVectorizer(stop_words=stopwordlist, min_df=3)
    tfidf = TfidfTransformer()
    
    pipeline = make_pipeline(vectorizer, tfidf, DecisionTreeClassifier())

    print "LOW + MIN3 + BIG + TFIDF + NS"
    target, data = helper.get_data('../../Data/train_prep.csv', pred_pos=0, text_pos=1, remove_header=False)

    print 'Train'
    #Split!!!
    x_train, x_test, y_train, y_test = tts(data, target, random_state=0)
    pipeline.fit(x_train, y_train)
    y_predicted = pipeline.predict(x_test)

    m = confusion_matrix(y_test, y_predicted)
    print m
    saveConfusionMatrix(m, '../../Data/m_dt-1.m')#mejora bastante
