#Preprocesamiento inicial de los sets de train y test
#Elimina HTML, las palabras que tengan signos de puntuacion y pasa caracteres a minusculas

# -*- coding: utf-8 -*-,
import csv
import re
from nltk.corpus import stopwords

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
				if(count > 0):
					text = rowlist[posText]

					text = re.sub(r'<.*?>','', text) #elimino los codigos html
					#text = re.sub(r'^\w*\d\w*','', text) #elimino todas las palabras con numeros en medio
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
basicPrep('../../Data/train.csv', '../../Data/train_prep_nums_lower.csv', 6, 9)


print('Test')
#######Me quedo con el ID del registro y con el texto
basicPrep('../../Data/test.csv', '../../Data/test_prep_nums_lower.csv', 0, 8)

