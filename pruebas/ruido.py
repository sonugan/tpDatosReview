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

##Retorna ruido
def generateNoise(path, pos_pred, pos_text, noisy_class, count, other_clases):
    with open(path, 'r') as csvi:
        readeri = csv.reader(csvi)

        #Todos los registros de una clase que quiero ensuciar
        seed_list = []
        for l in readeri:
            if l[pos_pred] == noisy_class:
                seed_list.append(l[pos_text])
        noisy_list = []
        noisy_target = []
        for i in range(count):
            noisy_target.append(random.choice(other_clases))
            noisy_list.append(random.choice(seed_list))
        return [noisy_target, noisy_list]


if True:
    vectorizer = CountVectorizer(stop_words=stopwordlist, min_df=3)
    tfidf = TfidfTransformer()
    
    pipeline = make_pipeline(vectorizer, tfidf, svm.LinearSVC())

    print "LOW + MIN3 + BIG + TFIDF + NS"
    target, data = helper.get_data('../../Data/train_prep.csv', pred_pos=0, text_pos=1, remove_header=False)

    noisy_target, noisy_data = generateNoise('../../Data/train_prep.csv', 0, 1, '5', 20000, ['1','2','3','4'])
    for t in noisy_target:
        target.append(t)
    for d in noisy_data:
        data.append(d)
   
    print 'Train'
    #Split!!!
    x_train, x_test, y_train, y_test = tts(data, target, random_state=0)
    pipeline.fit(x_train, y_train)
    y_predicted = pipeline.predict(x_test)

    m = confusion_matrix(y_test, y_predicted)
    print m
    saveConfusionMatrix(m, '../../Data/m_r-1.m')
