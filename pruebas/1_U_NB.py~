# -*- coding: utf-8 -*-,
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation

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

count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

alphas = np.arange(0,1.1,0.1)
scores = []
for a in alphas:
	mult = MultinomialNB(alpha = a)
	print("Multinomial alpha = :" + str(a))
	this_scores = cross_validation.cross_val_score(mult, X_train_tfidf, target, cv = 5)
	print(this_scores)
	scores.append(np.mean(this_scores))

plt.figure(figsize=(4, 3))
plt.plot(alphas, np.array(scores))
plt.ylabel('CV score')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.show()

plt.figure(figsize=(4, 3))
plt.plot(alphas, np.array(scores))
plt.ylabel('CV score')
plt.xlabel('alpha')
plt.show()

