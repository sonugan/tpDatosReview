from sklearn.pipeline import Pipeline
from sklearn import svm
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

class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


if False:
    vectorizer = CountVectorizer(stop_words=stopwordlist, min_df=3)
    tfidf = TfidfTransformer()
    
    pipeline = make_pipeline(vectorizer, tfidf, svm.LinearSVC())

    print "LOW + MIN3 + BIG + TFIDF + NS"
    target, data = helper.get_data('../../Data/train_prep.csv', pred_pos=0, text_pos=1, remove_header=False)

    #Kmeans!!
    print 'KMeans'
    cincos = []
    resto = []
    resto_t = []
    for idx, d in enumerate(data):
        if target[idx] == '5':
            cincos.append(d)
        else:
            resto.append(d)
            resto_t.append(target[idx])

    data_5 = vectorizer.fit_transform(cincos)
    data_5 = tfidf.fit_transform(data_5)

    kmeans = KMeans(n_clusters=5, random_state=0, verbose=1, max_iter=4).fit(data_5)
    for idx, c in enumerate(kmeans.labels_):
        resto.append(cincos[idx])
        resto_t.append('5' + str(c))
    data = resto
    target = resto_t

    print 'Train'
    #Split!!!
    x_train, x_test, y_train, y_test = tts(data, target, random_state=0)
    pipeline.fit(x_train, y_train)
    y_predicted = pipeline.predict(x_test)

    pred_list = []
    for p in y_predicted:
        if p == '1' or p=='2' or p=='3' or p=='4':
            pred_list.append(p)
        else:
            pred_list.append('5')
    
    y_list = []
    for p in y_test:
        if p == '1' or p=='2' or p=='3' or p=='4':
            y_list.append(p)
        else:
            y_list.append('5')

    y_test = y_list
    y_predicted = pred_list
    m = confusion_matrix(y_test, y_predicted)
    print m
    saveConfusionMatrix(m, '../../Data/m_s-1.m')#mejora bastante

    
if True:
    vectorizer = CountVectorizer(stop_words=stopwordlist, min_df=3, ngram_range=(2, 2))
    tfidf = TfidfTransformer()
    
    pipeline = make_pipeline(vectorizer, tfidf, svm.LinearSVC())

    print "LOW + MIN3 + BIG + TFIDF + NS"
    target, data = helper.get_data('../../Data/train_prep.csv', pred_pos=0, text_pos=1, remove_header=False)

    #Kmeans!!
    print 'KMeans'
    cincos = []
    resto = []
    resto_t = []
    for idx, d in enumerate(data):
        if target[idx] == '5':
            cincos.append(d)
        else:
            resto.append(d)
            resto_t.append(target[idx])

    data_5 = vectorizer.fit_transform(cincos)
    data_5 = tfidf.fit_transform(data_5)

    kmeans = KMeans(n_clusters=5, random_state=0, verbose=1, max_iter=4).fit(data_5)
    for idx, c in enumerate(kmeans.labels_):
        resto.append(cincos[idx])
        resto_t.append('5' + str(c))
    data = resto
    target = resto_t

    print 'Train'
    #Split!!!
    x_train, x_test, y_train, y_test = tts(data, target, random_state=0)
    pipeline.fit(x_train, y_train)
    y_predicted = pipeline.predict(x_test)

    pred_list = []
    for p in y_predicted:
        if p == '1' or p=='2' or p=='3' or p=='4':
            pred_list.append(p)
        else:
            pred_list.append('5')
    
    y_list = []
    for p in y_test:
        if p == '1' or p=='2' or p=='3' or p=='4':
            y_list.append(p)
        else:
            y_list.append('5')

    y_test = y_list
    y_predicted = pred_list
    m = confusion_matrix(y_test, y_predicted)
    print m
    saveConfusionMatrix(m, '../../Data/m_s-2.m')    

if False:
    vectorizer = CountVectorizer(stop_words=stopwordlist, min_df=3)
    tfidf = TfidfTransformer()
    
    pipeline = make_pipeline(vectorizer, tfidf, svm.LinearSVC())

    print "LOW + MIN3 + BIG + TFIDF + NS"
    target, data = helper.get_data('../../Data/train_prep.csv', pred_pos=0, text_pos=1, remove_header=False)

    #Kmeans!!
    print 'KMeans'
    cincos = []
    resto = []
    resto_t = []
    for idx, d in enumerate(data):
        if target[idx] == '5':
            cincos.append(d)
        else:
            resto.append(d)
            resto_t.append(target[idx])

    data_5 = vectorizer.fit_transform(cincos)
    data_5 = tfidf.fit_transform(data_5)

    kmeans = KMeans(n_clusters=5, random_state=0, verbose=1, max_iter=10).fit(data_5)
    for idx, c in enumerate(kmeans.labels_):
        resto.append(cincos[idx])
        resto_t.append('5' + str(c))
    data = resto
    target = resto_t

    print 'Train'
    #Split!!!
    x_train, x_test, y_train, y_test = tts(data, target, random_state=0)
    pipeline.fit(x_train, y_train)
    y_predicted = pipeline.predict(x_test)

    pred_list = []
    for p in y_predicted:
        if p == '1' or p=='2' or p=='3' or p=='4':
            pred_list.append(p)
        else:
            pred_list.append('5')
    
    y_list = []
    for p in y_test:
        if p == '1' or p=='2' or p=='3' or p=='4':
            y_list.append(p)
        else:
            y_list.append('5')

    y_test = y_list
    y_predicted = pred_list
    m = confusion_matrix(y_test, y_predicted)
    print m
    saveConfusionMatrix(m, '../../Data/m_s-3.m')