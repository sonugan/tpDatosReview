# -*- coding: utf-8 -*-,
import csv
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
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

####Guardo los datos obtenidos
def saveScores(scores):
	with open('scores_train_svc.csv', 'a') as csvfile2:
		writer = csv.writer(csvfile2)
		allScores = []
		for score in scores:
			allScores.append(score)
       		writer.writerow(allScores)

def calcScores(ces, data, target):
	scores = []
	for c in ces:
		svc = svm.LinearSVC()
		print("SVC C = :" + str(c))
		this_scores = cross_validation.cross_val_score(svc, X_train_tfidf, target, cv = 2)
		print(this_scores)
		scores.append(np.mean(this_scores))
	return scores

ces = np.arange(0.1,1.2,0.2)
'''
### SP
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores SP')
saveScores(['SP'] + calcScores(ces, X_train_tfidf, target))

#### ES
count_vect = CountVectorizer(stop_words=stopwordlist)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores ES')
saveScores(['ES'] + calcScores(ces, X_train_tfidf, target))

#### MIN3
count_vect = CountVectorizer(min_df = 3)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores MIN3')
saveScores(['MIN3'] + calcScores(ces, X_train_tfidf, target))

#### ES + MIN3
count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print("Scale matrix")
X_train_tfidf = preprocessing.scale(X_train_tfidf, with_mean=False)

print('Scores ES + MIN3')
print('Fin train')
'''
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
'''
count_vect = CountVectorizer(tokenizer=tokenize_steam)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores STEAM')
saveScores(['STEAM'] + calcScores(ces, X_train_tfidf, target))
'''
#### LEMMA
lmtzr = WordNetLemmatizer()

def lemma_tokens(tokens, lmtzr):
    lemmed = []
    for item in tokens:
        lemmed.append(lmtzr.lemmatize(item))
    return lemmed

def tokenize_lemma(text):
    tokens = word_tokenize(text)
    stems = lemma_tokens(tokens, lmtzr)
    return stems
'''
count_vect = CountVectorizer(tokenizer=tokenize_lemma)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores LEMMA')
saveScores(['LEMMA'] + calcScores(ces, X_train_tfidf, target))
'''
#### ES + MIN3 + STEAM
count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3, tokenizer=tokenize_steam)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)


print('Scores ES + MIN3 + STEAM')
saveScores(['ES + MIN3 + STEAM'] + calcScores(ces, X_train_tfidf, target))


#### ES + MIN3 + LEMMA
count_vect = CountVectorizer(stop_words=stopwordlist, min_df = 3, tokenizer=tokenize_lemma)
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

print('Scores ES + MIN3 + LEMMA')
saveScores(['ES + MIN3 + LEMMA'] + calcScores(ces, X_train_tfidf, target))

print("Fin!!")


