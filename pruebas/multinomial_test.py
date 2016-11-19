# -*- coding: utf-8 -*-,
import csv
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from nltk.corpus import stopwords

stopwordlist = stopwords.words('english')

data_train = []
target = []

count = 0
'''
	Toma del archivo train.csv (cambiar el path) todos los textos de las reviews y los agrega a una lista (data)
	y las clasificaciones de cada review: 1,2,3,4,5 ...
	count muestra el progreso en el avance del procesamiento del archivo de train
'''
print('Leo el archivo de train-----------')
with open('../../Data/train_prep.csv', 'r') as csvfile1:
	spamreader = csv.reader(csvfile1)
	for rowlist in spamreader:
		if(count % 1000 == 0):
			print(count)
		data_train.append(rowlist[1])
		target.append(rowlist[0])
		count+=1

data_test = []
ids = []

count = 0
print('Leo el archivo de test-----------')
with open('../../Data/test_prep.csv', 'r') as csvfile1:
	spamreader = csv.reader(csvfile1)
	for rowlist in spamreader:
		if(count % 1000 == 0):
			print(count)
		data_test.append(rowlist[1])
		ids.append(rowlist[0])
		count+=1

## Genera la matriz para SP
def genSP(data_train, data_test):
	count_vect = CountVectorizer()
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	x_new_counts = count_vect.transform(data_test)
	x_new_counts_tfidf = tfidf_transformer.transform(x_new_counts)
	return X_train_tfidf, x_new_counts_tfidf

## Genera la matriz para ES
def genES(data_train, data_test):
	count_vect = CountVectorizer(stop_words=stopwordlist)
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	x_new_counts = count_vect.transform(data_test)
	x_new_counts_tfidf = tfidf_transformer.transform(x_new_counts)
	return X_train_tfidf, x_new_counts_tfidf

## Genera la matriz para MIN3
def genMIN3(data_train, data_test):
	count_vect = CountVectorizer(min_df = 3)
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	x_new_counts = count_vect.transform(data_test)
	x_new_counts_tfidf = tfidf_transformer.transform(x_new_counts)
	return X_train_tfidf, x_new_counts_tfidf

## Genera la matriz para MIN3 BIGR
def genMIN3BIGR(data_train, data_test, ngram):
	count_vect = CountVectorizer(min_df = 3, ngram_range=(ngram,ngram))
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	x_new_counts = count_vect.transform(data_test)
	x_new_counts_tfidf = tfidf_transformer.transform(x_new_counts)
	return X_train_tfidf, x_new_counts_tfidf


## Genera la matriz para ES + MIN3
def genESMIN3(data_train, data_test):
	count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3)
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	x_new_counts = count_vect.transform(data_test)
	x_new_counts_tfidf = tfidf_transformer.transform(x_new_counts)
	return X_train_tfidf, x_new_counts_tfidf

## Genera la matriz para ES + MIN3 + [BIGR(2),TRIG(3)]
def genESMIN3BIGR(data_train, data_test, ngram):
	count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3, ngram_range=(ngram,ngram))
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	x_new_counts = count_vect.transform(data_test)
	x_new_counts_tfidf = tfidf_transformer.transform(x_new_counts)
	return X_train_tfidf, x_new_counts_tfidf

## Genera la matriz para ES + MIN3 + [BIGR(2),TRIG(3)...NGRAM(N)]
def genESMIN3MULTIGR(data_train, data_test, ngram_from, ngram_to):
	count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3, ngram_range=(ngram_from,ngram_to))
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	x_new_counts = count_vect.transform(data_test)
	x_new_counts_tfidf = tfidf_transformer.transform(x_new_counts)
	return X_train_tfidf, x_new_counts_tfidf


## Genera los tokens stemizados
stemmer = PorterStemmer()
def stem_tokens(tokens):
	    stemmed = []
	    for item in tokens:
		stemmed.append(stemmer.stem(item))
	    return stemmed

def tokenize_steam(text):
	    tokens = word_tokenize(text)
	    stems = stem_tokens(tokens)
	    return stems

## Genera la matriz para STEM
def genSTEM(data_train, data_test):
	count_vect = CountVectorizer(tokenizer=tokenize_steam)
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	x_new_counts = count_vect.transform(data_test)
	x_new_counts_tfidf = tfidf_transformer.transform(x_new_counts)
	return X_train_tfidf, x_new_counts_tfidf

##Genera los tokens lematizados
lmtzr = WordNetLemmatizer()
def lemma_tokens(tokens):
	    lemmed = []
	    for item in tokens:
		lemmed.append(lmtzr.lemmatize(item))
	    return lemmed

def tokenize_lemma(text):
	    tokens = word_tokenize(text)
	    stems = lemma_tokens(tokens)
	    return stems


## Genera la matriz para LEMMA
def genLEMMA(data_train, data_test):
	count_vect = CountVectorizer(tokenizer=tokenize_lemma)
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	x_new_counts = count_vect.transform(data_test)
	x_new_counts_tfidf = tfidf_transformer.transform(x_new_counts)
	return X_train_tfidf, x_new_counts_tfidf

## Genera la matriz para ES + MIN3 + STEM
def genESMIN3STEM(data_train, data_test):
	count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3, tokenizer=tokenize_steam)
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	x_new_counts = count_vect.transform(data_test)
	x_new_counts_tfidf = tfidf_transformer.transform(x_new_counts)
	return X_train_tfidf, x_new_counts_tfidf

## Genera la matriz para ES + MIN3 + LEMMA
def genESMIN3LEMMA(data_train, data_test):
	count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3, tokenizer=tokenize_lemma)
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	x_new_counts = count_vect.transform(data_test)
	x_new_counts_tfidf = tfidf_transformer.transform(x_new_counts)
	return X_train_tfidf, x_new_counts_tfidf


###Entreno
clf = MultinomialNB(alpha=0.01)
x_train, x_test = genESMIN3MULTIGR(data_train, data_test, 1, 2)
clf.fit(x_train, target)


count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
##Creo el archivo para subir al kaggle
with open('../kaggle/multinomial_test.csv', 'wb') as csvfile2:
	writer = csv.writer(csvfile2)
 	writer.writerow(['Id', 'Prediction'])
	for idx, prediction in enumerate(clf.predict(x_test)):
    		writer.writerow([str(ids[idx]), str(prediction)])
		if(prediction == '1'):
			count1+=1
		if(prediction == '2'):
			count2+=1
		if(prediction == '3'):
			count3+=1
		if(prediction == '4'):
			count4+=1
		if(prediction == '5'):
			count1+=1


print('Resultados:')
print('Cantidad de 1:' + str(count1))
print('Cantidad de 2:' + str(count2))
print('Cantidad de 3:' + str(count3))
print('Cantidad de 4:' + str(count4))
print('Cantidad de 5:' + str(count5))
