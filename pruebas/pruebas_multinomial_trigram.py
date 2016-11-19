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

data = []
target = []

count = 0
'''
	Toma del archivo train.csv (cambiar el path) todos los textos de las reviews y los agrega a una lista (data)
	y las clasificaciones de cada review: 1,2,3,4,5 ...
	count muestra el progreso en el avance del procesamiento del archivo de train
'''
with open('../../Data/train_prep.csv', 'r') as csvfile1:
	spamreader = csv.reader(csvfile1)
	for rowlist in spamreader:
		if(count % 1000 == 0):
			print(count)
		data.append(rowlist[1])
		target.append(rowlist[0])
		count+=1
##Split Folds:

def calcScores(alphas, data, target):
	scores = []
	for a in alphas:
		mult = MultinomialNB(alpha = a)
		print("Multinomial alpha = :" + str(a))
		this_scores = cross_validation.cross_val_score(mult, data, target, cv = 3)
		print(this_scores)
		scores.append(np.mean(this_scores))
	return scores

alphas = [0.01]
allScores = []

## Genera la matriz para SP
def genSP(data_train, ngram):
	count_vect = CountVectorizer(ngram_range=(ngram,ngram))
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
	return X_train_tfidf

## Genera la matriz para ES
def genES(data_train, ngram):
	count_vect = CountVectorizer(stop_words=stopwordlist, ngram_range=(ngram,ngram))
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
	return X_train_tfidf

## Genera la matriz para MIN3
def genMIN3(data_train, ngram):
	count_vect = CountVectorizer(min_df = 3, ngram_range=(ngram,ngram))
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	return X_train_tfidf

## Genera la matriz para ES + MIN3
def genESMIN3(data_train, ngram):
	count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3, ngram_range=(ngram,ngram))
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	return X_train_tfidf

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
def genSTEM(data_train, ngram):
	count_vect = CountVectorizer(tokenizer=tokenize_steam, ngram_range=(ngram,ngram))
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	return X_train_tfidf

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
def genLEMMA(data_train, ngram):
	count_vect = CountVectorizer(tokenizer=tokenize_lemma, ngram_range=(ngram,ngram))
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	return X_train_tfidf

## Genera la matriz para ES + MIN3 + STEM
def genESMIN3STEM(data_train, ngram):
	count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3, tokenizer=tokenize_steam, ngram_range=(ngram,ngram))
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	return X_train_tfidf

## Genera la matriz para ES + MIN3 + LEMMA
def genESMIN3LEMMA(data_train, ngram):
	count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3, tokenizer=tokenize_lemma, ngram_range=(ngram,ngram))
	x_train_counts = count_vect.fit_transform(data_train)

	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

	return X_train_tfidf


### SP

print('Scores SP')
allScores.append(['SP'] + calcScores(alphas, genSP(data, 3), target))

#### ES
print('Scores ES')
allScores.append(['ES'] + calcScores(alphas, genES(data,3), target))

#### MIN3
print('Scores MIN3')
allScores.append(['MIN3'] + calcScores(alphas, genMIN3(data, 3), target))

#### ES + MIN3
print('Scores ES + MIN3')
allScores.append(['ES + MIN3'] + calcScores(alphas, genESMIN3(data,3), target))

#### STEM
print('Scores STEM')
allScores.append(['STEM'] + calcScores(alphas, genSTEM(data,3), target))

#### LEMMA
print('Scores LEMMA')
allScores.append(['LEMMA'] + calcScores(alphas, genLEMMA(data,3), target))

#### ES + MIN3 + STEM
print('Scores ES + MIN3 + STEM')
allScores.append(['ES + MIN3 + STEM'] + calcScores(alphas, genESMIN3STEM(data, 3), target))


#### ES + MIN3 + LEMMA
print('Scores ES + MIN3 + LEMMA')
allScores.append(['ES + MIN3 + LEMMA'] + calcScores(alphas, genESMIN3LEMMA(data, 3), target))


####Guardo los datos obtenidos
with open('scores_train_multinomial_trigram.csv', 'wb') as csvfile2:
	writer = csv.writer(csvfile2)
	for score in allScores:
       		writer.writerow(score)

