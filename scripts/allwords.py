##Toma el archivo train.csv y extrae todas las palabras eliminando los signos de puntuacion y los codigos html
###y convierte cada una a minusculas
##Entrada: train.csv
##Salida: allwords.w

# -*- coding: utf-8 -*-,
import csv
import sys
import csv
import nltk
import re
import codecs
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords

csv.field_size_limit(sys.maxsize)
    
#Elimina el codigo html dentro de las reviews

def cleanhtml(raw_html):
	cleanr =re.compile('<.*?>')
	cleantext = re.sub(cleanr,'', raw_html)
	return cleantext


def countWords(line):
	text = cleanhtml(line)
	nonPunct = re.compile(".*[A-Za-z0-9].*")
	filtered = [w.lower().encode('utf-8') for w in TweetTokenizer().tokenize(text) if nonPunct.match(w)]
	#return Counter(filtered)
	return filtered
    
def saveWords():
    with codecs.open('../words/allwords.w', 'wb', encoding='utf-8') as vwfile:
        for w in words:
		vwfile.write(unicode(w,'utf-8') + u':' + str(words[w]).encode('utf-8') + u'\n')


count = 0
words = {}#Counter({''})

with open('../../Data/train.csv', 'r') as csvfile1:
	spamreader = csv.reader(csvfile1)
	for rowlist in spamreader:
		if(count % 1000 == 0):
			print(count)
		#if(count > 0):
			#words = words + countWords(rowlist[9])
		if(count > 0):
			for w in countWords(rowlist[9]):
				if w in words:			
					words[w]+=1
				if not w in words:
					words[w] = 1

		count = count + 1

print("Fin lectura")            
saveWords()
print("Fin")
