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
import random

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

    
if False:
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

####kaggle!!

if False:#1
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
    x_train = data
    y_train = target
    #x_train, x_test, y_train, y_test = tts(data, target, random_state=0)
    pipeline.fit(x_train, y_train)

    print "Predict!!"
    ids, x_test = helper.get_data('../../Data/test_prep.csv', pred_pos=0, text_pos=1, remove_header=True)
    y_predicted = pipeline.predict(x_test)

    pred_list = []
    for p in y_predicted:
        if p == '1' or p=='2' or p=='3' or p=='4':
            pred_list.append(p)
        else:
            pred_list.append('5')
    
    y_predicted = pred_list
    
    helper.saveKaggleFile('../kaggle/svc_kmeans_test_1.csv', y_predicted, ids)


if False:#2
    vectorizer = CountVectorizer(min_df=3)
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
    x_train = data
    y_train = target
    #x_train, x_test, y_train, y_test = tts(data, target, random_state=0)
    pipeline.fit(x_train, y_train)

    print "Predict!!"
    ids, x_test = helper.get_data('../../Data/test_prep.csv', pred_pos=0, text_pos=1, remove_header=True)
    y_predicted = pipeline.predict(x_test)

    pred_list = []
    for p in y_predicted:
        if p == '1' or p=='2' or p=='3' or p=='4':
            pred_list.append(p)
        else:
            pred_list.append('5')
    
    y_predicted = pred_list
    
    helper.saveKaggleFile('../kaggle/svc_kmeans_test.csv', y_predicted, ids)


def train_subset(x_train, y_train, x_test, porc_corte, pipeline):
    x_train_1, x_test_1, y_train_1, y_test_1 = tts(x_train, y_train, random_state=0, test_size=porc_corte)
    pipeline.fit(x_train_1, y_train_1)

    print "Predict!!"
    return pipeline.predict(x_test)

##UNifica las clases spliteadas por kmeans
def unificar_kmeans(y_predicted, not_spit_clases, split_class):
    pred_list = []
    for p in y_predicted:
        if p in not_spit_clases:
            pred_list.append(p)
        else:
            pred_list.append(split_class)
    return pred_list
    

###Retona la prediccion dada una lista de predicciones
###si hay empate, retorna la clase que encontro primero
def get_pred(pred_list):
    #pred_list = random.shuffle(pred_list)
    preds = dict()
    for p in pred_list:
        if p not in preds:
            preds[p] = 1
        else:
            preds[p] += 1

    prediction = None
    for p in preds:
        if prediction == None:
            prediction = p
        else:
            if preds[p] < preds[prediction]:
                prediction = p
    return prediction

if False:#3
    vectorizer = CountVectorizer(min_df=3)
    tfidf = TfidfTransformer()
    
    pipeline = make_pipeline(vectorizer, tfidf, svm.LinearSVC())

    print "LOW + MIN3 + BIG + TFIDF + NS"
    target, data = helper.get_data('../../Data/train_prep.csv', pred_pos=0, text_pos=1, remove_header=False)
    
    print 'Train'

    ids, x_test = helper.get_data('../../Data/test_prep.csv', pred_pos=0, text_pos=1, remove_header=True)
    
    y_pred1 = train_subset(data, target, x_test, 0.25, pipeline)
    print "1"
    y_pred2 = train_subset(data, target, x_test, 0.30, pipeline)
    print "2"
    y_pred3 = train_subset(data, target, x_test, 0.35, pipeline)
    print "3"
    y_pred4 = train_subset(data, target, x_test, 0.40, pipeline)
    print "4"
    y_pred5 = train_subset(data, target, x_test, 0.45, pipeline)
    print "5"
    y_pred6 = train_subset(data, target, x_test, 0.50, pipeline)
    print "6"
    y_pred7 = train_subset(data, target, x_test, 0.55, pipeline)
    print "listo"
    y_predicted = []
    for idx, id in enumerate(ids):
        preds = [y_pred1[idx], y_pred2[idx], y_pred3[idx], y_pred4[idx], y_pred5[idx], y_pred6[idx], y_pred7[idx]]        
        y_predicted.append(get_pred(preds))

    helper.saveKaggleFile('../kaggle/svc_kmeans_test_3.csv', y_predicted, ids)

if True:#4
    vectorizer = CountVectorizer(min_df=3)
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
    ids, x_test = helper.get_data('../../Data/test_prep.csv', pred_pos=0, text_pos=1, remove_header=True)
    
    not_spit_clases = ['1','2','3','4']

    y_pred1 = unificar_kmeans(train_subset(data, target, x_test, 0.25, pipeline), not_spit_clases, '5') 
    print "1"
    y_pred2 = unificar_kmeans(train_subset(data, target, x_test, 0.30, pipeline), not_spit_clases, '5')
    print "2"
    y_pred3 = unificar_kmeans(train_subset(data, target, x_test, 0.35, pipeline), not_spit_clases, '5')
    print "3"
    y_pred4 = unificar_kmeans(train_subset(data, target, x_test, 0.40, pipeline), not_spit_clases, '5')
    print "4"
    y_pred5 = unificar_kmeans(train_subset(data, target, x_test, 0.45, pipeline), not_spit_clases, '5')
    print "5"
    y_pred6 = unificar_kmeans(train_subset(data, target, x_test, 0.50, pipeline), not_spit_clases, '5')
    print "6"
    y_pred7 = unificar_kmeans(train_subset(data, target, x_test, 0.55, pipeline), not_spit_clases, '5')
    print "listo"
    y_predicted = []
    for idx, id in enumerate(ids):
        preds = [y_pred1[idx], y_pred2[idx], y_pred3[idx], y_pred4[idx], y_pred5[idx], y_pred6[idx], y_pred7[idx]]        
        y_predicted.append(get_pred(preds))

    
    helper.saveKaggleFile('../kaggle/svc_kmeans_test_4.csv', y_predicted, ids)
