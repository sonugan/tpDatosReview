# -*- coding: utf-8 -*-,
import helpers as helper
import re
import csv
import unicodecsv as csv
import prep_divided
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

stopwordlist = stopwords.words('english')


train_path = '../../Data/train_prep'
test_path='../../Data/test_prep'
vocabulario = None
##Separo un set de test
if True:
    train_path = '../../Data/train-train_prep_voc'
    test_path='../../Data/test-test_prep_voc'
    
    target, data = helper.get_data('../../Data/train_prep.csv')
    print "split!!"

    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=0)
    
    with open(test_path + '.csv', 'wb') as csvt:
        writer = csv.writer(csvt)
        for idx, x in enumerate(x_test):
            writer.writerow([y_test[idx], x])
    
    with open(train_path + '.csv', 'wb') as csvt:
        writer = csv.writer(csvt)
        for idx, x in enumerate(x_train):
            writer.writerow([y_train[idx], x])       

    #Armo los diccionarios y elimino las claves que no en 5 del resto de las clases
    voc = dict()
    voc['1'] = dict()
    voc['2'] = dict()
    voc['3'] = dict()
    voc['4'] = dict()
    voc['5'] = dict()
    for idx, x in enumerate(x_train):
        if x not in voc[y_train[idx]]:
            voc[y_train[idx]] = x
    count = 0
    vocab = dict()
    for c in ['1','2','3','4']:
        for x in voc[c]:
            if count % 1000 == 0:
                print count
            if x not in voc['5'] and x not in vocab:
                vocab[x] = 1
            count += 1
    
    vocabulario = []
    with open(train_path + 'voc--.csv', 'wb') as csvo:
        writero = csv.writer(csvo)
        for x in vocab:
            vocabulario.append(x)
            writero.writerow([x])
if True:
    print "Obtengo los datos"
    x_target, x_data = helper.get_train_data(train_path + '.csv', vectorizer = helper.get_vectorizer(vocabulary=vocabulario, min_df = 3), pred_pos=0, text_pos=1,tf_idf=True, remove_header=False)
    y_target, y_data = helper.get_train_data(test_path + '.csv', vectorizer = helper.get_vectorizer(vocabulary=vocabulario ,min_df = 3), pred_pos=0, text_pos=1,tf_idf=True, remove_header=False)
    print "LOW + MIN3 + BIG + TFIDF"
    clf = MultinomialNB(alpha=0.01)
    clf.fit(x_data, x_target)
    y_predicted = clf.predict(y_data)
    print(confusion_matrix(y_target, y_predicted))
    print(accuracy_score(y_target, y_predicted))
    
if False:
    count = 0
    with open('../../Data/train_prep_voc1.csv', 'r') as csv4:
        with open('../../Data/train_prep_voc5.csv', 'r') as csv5:
            with open('../../Data/train_prep_voc1-5.csv', 'wb') as csvo:
                reader4 = csv.reader(csv4)
                reader5 = csv.reader(csv5)
                writer = csv.writer(csvo)
                words5 = []
                for word in reader5:
                    words5.append(word[1])
                for word in reader4:
                    if(count % 1000 == 0):
                        print(count)
                    if word[1] not in words5:
                        writer.writerow(word)
                    count += 1

#Armo un archivo con los cosos que no tiene el 5
if False:
    count = 0
    realwords5 = []
    with open('../../Data/train_prep_voc_all-5.csv', 'wb') as csvo:
        writer_o = csv.writer(csvo)
        with open('../../Data/train_prep_voc5.csv', 'r') as csv5:
            reader5 = csv.reader(csv5)
            words5 = []            
            for word in reader5:
                words5.append(word)
            print "1"
            with open('../../Data/train_prep_voc1-5.csv', 'r') as csvin:
                reader_i = csv.reader(csvin)
                words = []
                for line in reader_i:
                    words.append(line[0])
                    writer_o.writerow([line[0]])
                count = 0
                for w in words5:
                    if count % 100 == 0:
                        print count
                    if w[0] not in words:
                        realwords5.append(w)
                    count += 1
            print "2"
            with open('../../Data/train_prep_voc2-5.csv', 'r') as csvin:
                reader_i = csv.reader(csvin)
                words = []
                for line in reader_i:
                    words.append(line[0])
                    writer_o.writerow([line[0]])
                count = 0
                for w in words5:
                    if count % 100 == 0:
                        print count
                    if w[0] not in words:
                        realwords5.append(w)
                    count += 1
            print "3"
            with open('../../Data/train_prep_voc3-5.csv', 'r') as csvin:
                reader_i = csv.reader(csvin)
                words = []
                for line in reader_i:
                    words.append(line[0])
                    writer_o.writerow([line[0]])
                count = 0
                for w in words5:
                    if count % 100 == 0:
                        print count
                    if w[0] not in words:
                        realwords5.append(w)
                    count += 1
            print "4"
            with open('../../Data/train_prep_voc4-5.csv', 'r') as csvin:
                reader_i = csv.reader(csvin)
                words = []
                for line in reader_i:
                    words.append(line[0])
                    writer_o.writerow([line[0]])
                count = 0
                for w in words5:
                    if count % 100 == 0:
                        print count
                    if w[0] not in words:
                        realwords5.append(w)
                    count += 1
            #Escribo los que están solo en el 5:
            print "5"
            count = 0
            for w in realwords5:
                if count % 100 == 0:
                    print count
                writer_o.writerow(w)
                count += 1
    
#Elimino las palabras que aparecen mas de una vez                
if False:
    count = 0
    words = dict()
    with open('../../Data/train_prep_voc_all-5.csv', 'r') as csv4:
        reader4 = csv.reader(csv4)
        for w in reader4:
            if count % 100 == 0:
                print count
            if w[0] not in words:
                words[w[0]] = 1
            count += 1
    with open('../../Data/train_prep_voc_all-5.csv', 'wb') as csv4:
        writer4 = csv.writer(csv4)
        for k in words:
            writer4.writerow([k])
                