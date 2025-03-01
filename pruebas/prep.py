#Preprocesamiento inicial de los sets de train y test
#Elimina HTML, las palabras que tengan numeros en medio y signos de puntuacion y pasa caracteres a minusculas

# -*- coding: utf-8 -*-,
import csv
import re
import prep_emoticon as emoticon
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

					#text = emoticon.replace_emoticons(text)
					text = re.sub(r'<.*?>','', text) #elimino los codigos html
					
					#text = re.sub(r'^\w*\d\w*','', text) #elimino todas las palabras con numeros en medio
					text = re.sub(r'\d', '', text)#elimino los numeros
					text = re.sub(r'[^\w\s]','', text) #elimino los signos de puntuacion

					text = re.sub(' +', ' ', text)#elimino los espacios en blanco extra
					text = text.lower()

					writer.writerow([rowlist[posPred], text])
				count+=1

def basicPrepS(inputFile, targetFile, outputFile, posPred, posText):
	count = 1
	targets = []
	with open(inputFile, 'r') as csvfile1:
		with open(targetFile, 'r') as targetF:
			treader = csv.reader(targetF)
			for t in treader:
				targets.append(t[posPred])
		spamreader = csv.reader(csvfile1)
		with open(outputFile, 'wb') as csvfile2:
			writer = csv.writer(csvfile2)
			for rowlist in spamreader:
				if(count % 1000 == 0):
					print(count)
				text = rowlist[posText]

				#text = emoticon.replace_emoticons(text)
				text = re.sub(r'<.*?>','', text) #elimino los codigos html
					
				#text = re.sub(r'^\w*\d\w*','', text) #elimino todas las palabras con numeros en medio
				text = re.sub(r'\d', '', text)#elimino los numeros
				text = re.sub(r'[^\w\s]','', text) #elimino los signos de puntuacion

				text = re.sub(' +', ' ', text)#elimino los espacios en blanco extra
				text = text.lower()

				writer.writerow([targets[count], text])
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
#basicPrep('../../Data/train_NO_NAMES.csv', '../../Data/train_prep_NO_NAMES.csv', 6, 9)
basicPrepS('../../Data/train_NO_NAMES.csv', '../../Data/train_prep.csv', '../../Data/train_prep_NO_NAMES.csv', 0, 0)

print('Test')
#######Me quedo con el ID del registro y con el texto
#basicPrep('../../Data/test.csv', '../../Data/test_prep_emot.csv', 0, 8)


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
