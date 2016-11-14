#Preprocesamiento inicial de los sets de train y test
#Marca las negaciones
#Elimina HTML, las palabras que tengan numeros en medio y signos de puntuacion y pasa caracteres a minusculas

# -*- coding: utf-8 -*-,
import csv
import re
from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

sentim_analyzer = SentimentAnalyzer()
negs = sentim_analyzer.all_words([mark_negation(row) for row in [['silly', 'and', 'alert', ',', 'thirteen', 'conversations', 'about', 'one',
'thing', 'is', 'a', 'small', 'gem', '.']]])
for neg in negs:
	print(neg)

'''
# Archivo con el preprocesamiento basico
def basicPrep(inputFile, outputFile, posPred, posText):
	count = 0
	with open(inputFile, 'r') as csvfile1:
		spamreader = csv.reader(csvfile1)
		with open(outputFile, 'wb') as csvfile2:
			writer = csv.writer(csvfile2)
			for rowlist in spamreader:
				if(count % 1000 == 0):
					print(count)

				text = rowlist[posText]

				text = re.sub(r'<.*?>','', text) #elimino los codigos html
				text = re.sub(r'^\w*\d\w*','', text) #elimino todas las palabras con numeros en medio
				text = re.sub(r'[^\w\s]','', text) #elimino los signos de puntuacion
				text = text.lower()

				writer.writerow([rowlist[posPred], text])
				count+=1

# Remueve stop words
def prepES(inputFile, outputFile, posPred, posText):
	stopwordlist = stopwords.words('english')
	count = 0
	with open(inputFile, 'r') as csvfile1:
		spamreader = csv.reader(csvfile1)
		with open(outputFile, 'wb') as csvfile2:
			writer = csv.writer(csvfile2)
			for rowlist in spamreader:
				if(count % 1000 == 0):
					print(count)

				text = rowlist[posText]

				writer.writerow([rowlist[posPred], text])
				count+=1

print('Train')
####### Me quedo con la prediccion y con el texto
basicPrep('../../Data/train.csv', '../../Data/train_prep.csv', 6, 9)


print('Test')
#######Me quedo con el ID del registro y con el texto
basicPrep('../../Data/test.csv', '../../Data/test_prep.csv', 0, 8)
'''

'''
with open('../../Data/train.csv', 'r') as csvfile1:
	spamreader = csv.reader(csvfile1)
	with open('../../Data/train_prep.csv', 'wb') as csvfile2:
		writer = csv.writer(csvfile2)
		for rowlist in spamreader:
			if(count % 1000 == 0):
				print(count)

			text = rowlist[9]

			text = re.sub(r'<.*?>','', text) #elimino los codigos html
			text = re.sub(r'^\w*\d\w*','', text) #elimino todas las palabras con numeros en medio
			text = re.sub(r'[^\w\s]','', text) #elimino los signos de puntuacion
			text = text.lower()

        		writer.writerow([rowlist[6], text])
			count+=1

#######Me quedo con el ID del registro y con el texto

count = 0
print('Test')
with open('../../Data/test.csv', 'r') as csvfile1:
	spamreader = csv.reader(csvfile1)
	with open('../../Data/test_prep.csv', 'wb') as csvfile2:
		writer = csv.writer(csvfile2)
		for rowlist in spamreader:
			if(count % 1000 == 0):
				print(count)

			text = rowlist[8]

			text = re.sub(r'<.*?>','', text) #elimino los codigos html
			text = re.sub(r'^\w*\d\w*','', text) #elimino todas las palabras con numeros en medio
			text = re.sub(r'[^\w\s]','', text) #elimino los signos de puntuacion
			text = text.lower()

        		writer.writerow([rowlist[0], text])
			count+=1

'''
