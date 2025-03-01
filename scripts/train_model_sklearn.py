# -*- coding: utf-8 -*-,
'''
	Se requiere instalar sklearn 0.17: http://scikit-learn.org/stable/install.html 
	 y nltk para las stopwords, pero si quieren pueden armar el listado a mano ustedes: http://www.nltk.org/install.html
	
	
	El codigo y los ejemplos los tome del siguiente link: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
	Aca hay info de cross validation y como realizarla usando la libreria: http://scikit-learn.org/stable/modules/cross_validation.html
	
	Si tienen tiempo de ver mirar un poco el sitio oficial de scikit-learn es muy interesante, está lleno de ejemplos y muy bien documentado
'''

import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfTransformer

stopwordlist = stopwords.words('english')

data = []
target = []

dataTest = []
targetTest = []
count = 0
'''
	Toma del archivo train.csv (cambiar el path) todos los textos de las reviews y los agrega a una lista (data)
	y las clasificaciones de cada review: 1,2,3,4,5 ...
	count muestra el progreso en el avance del procesamiento del archivo de train
'''
with open('../../Data/train.csv', 'r') as csvfile1:
	spamreader = csv.reader(csvfile1)
	for rowlist in spamreader:
		if(count % 1000 == 0):
			print(count)
		if(count >0 and count <= 10):
			dataTest.append(rowlist[9])
			targetTest.append(rowlist[6])
		if(count > 10):
			data.append(rowlist[9])
			target.append(rowlist[6])
		count+=1

data_train, data_test, target_train, target_test = cross_validation.train_test_split(data, target, test_size=0.4, random_state=0)

'''
	CountVectorizer genera una matriz dispersa (x_train_counts) usando scipy de terminos por documento con la cantidad de ocurrencias de los mismos en cadad doc.
	Pueden agregarse varios filtros para los terminos. 
	Yo probé con los mas simples: 
		* lowercase: pasa todos los terminos a lowercase,
		* stop_words: se le pasa un listado de las stopwords que se desea eliminar, use el de nltk en ingles
		* min_df: filra los terminos que no tienen al menos min_df ocurrencias
	Aca esta el link a la documentacion: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
	
	Como es una matriz que tiene la cantidad de ocurrencias, se puede aplicar tf_idf antes de entrenar el modelo
'''
count_vect = CountVectorizer(lowercase=True, stop_words=stopwordlist, min_df = 3)
x_train_counts = count_vect.fit_transform(data_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

'''
	Como segui el ejemplo, use un modelo multinomial, pero tiene todo un listado de modelos supervisados: http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
	
	clf es el modelo entrenado
'''
clf1 = MultinomialNB().fit(X_train_tfidf, target_train)
clf2 = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True).fit(X_train_tfidf, target_train)
clf3 = MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True).fit(X_train_tfidf, target_train)
clf4 = MultinomialNB(alpha=0.3, class_prior=None, fit_prior=True).fit(X_train_tfidf, target_train)
clf5 = MultinomialNB(alpha=0.2, class_prior=None, fit_prior=True).fit(X_train_tfidf, target_train)
clf6 = MultinomialNB(alpha=0, class_prior=None, fit_prior=True).fit(X_train_tfidf, target_train)
clf7 = MultinomialNB(alpha=0, fit_prior=True).fit(X_train_tfidf, target_train)
clf8 = MultinomialNB(alpha=0, fit_prior=False).fit(X_train_tfidf, target_train)


'''
	para poder clasificar un registro, se debe transformar el mismo para que tenga la misma dimension del modelo (x_new_counts)
	dataTest es un listado de en este caso 10 registros que saque del set de train
'''
x_new_counts = count_vect.transform(data_test)
x_new_counts_tfidf = tfidf_transformer.transform(x_new_counts)

'''
	le indico al modelo que me de una predicción de los registros transformados
'''
#predicted = clf.predict(x_new_counts)

print(clf1.score(x_new_counts, target_test))
print(clf2.score(x_new_counts, target_test))
print(clf3.score(x_new_counts, target_test))
print(clf4.score(x_new_counts, target_test))
print(clf5.score(x_new_counts, target_test))
print(clf6.score(x_new_counts, target_test))
print(clf7.score(x_new_counts, target_test))
print(clf8.score(x_new_counts, target_test))


