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
		this_scores = cross_validation.cross_val_score(mult, X_train_tfidf, target, cv = 5)
		print(this_scores)
		scores.append(np.mean(this_scores))
	return scores

alphas = np.arange(0,1.1,0.1)
allScores = []

### SP
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores SP')
allScores.append(['SP'] + calcScores(alphas, X_train_tfidf, target))

#### ES
count_vect = CountVectorizer(stop_words=stopwordlist)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores ES')
allScores.append(['ES'] + calcScores(alphas, X_train_tfidf, target))

#### MIN3
count_vect = CountVectorizer(min_df = 3)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores MIN3')
allScores.append(['MIN3'] + calcScores(alphas, X_train_tfidf, target))

#### ES + MIN3
count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores ES + MIN3')
allScores.append(['ES + MIN3'] + calcScores(alphas, X_train_tfidf, target))

#### STEAM
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize_steam(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

count_vect = CountVectorizer(tokenizer=tokenize_steam)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores STEAM')
allScores.append(['STEAM'] + calcScores(alphas, X_train_tfidf, target))

#### LEMMA
lmtzr = WordNetLemmatizer()

def lemma_tokens(tokens, stemmer):
    lemmed = []
    for item in tokens:
        lemmed.append(lmtzr.lemmatize(item))
    return lemmed

def tokenize_lemma(text):
    tokens = word_tokenize(text)
    stems = lemma_tokens(tokens, stemmer)
    return stems

count_vect = CountVectorizer(tokenizer=tokenize_lemma)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores LEMMA')
allScores.append(['LEMMA'] + calcScores(alphas, X_train_tfidf, target))

#### ES + MIN3 + STEAM
count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3, tokenizer=tokenize_steam)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores ES + MIN3 + STEAM')
allScores.append(['ES + MIN3 + STEAM'] + calcScores(alphas, X_train_tfidf, target))


#### ES + MIN3 + LEMMA
count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3, tokenizer=tokenize_lemma)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores ES + MIN3 + LEMMA')
allScores.append(['ES + MIN3 + LEMMA'] + calcScores(alphas, X_train_tfidf, target))


####Guardo los datos obtenidos
with open('scores_train_multinomial.csv', 'wb') as csvfile2:
	writer = csv.writer(csvfile2)
	for score in allScores:
       		writer.writerow(score)

'''
plt.figure(figsize=(4, 3))
plt.plot(alphas, np.array(scores1))
plt.plot(alphas, np.array(scores2))
plt.plot(alphas, np.array(scores3))

plt.ylabel('CV score')
plt.xlabel('alpha')
plt.legend(['SP', 'BIN', 'ES + MIN3'], loc='upper left')
plt.show()
'''
