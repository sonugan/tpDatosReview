##Quita del archivo allwords.w todas las stopwords
###Entrada: allwords.w
###Salida: nonstopwords.w

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

stopwordlist = stopwords.words('english')

with open('../words/allwords.w', 'r') as filer:
    	with codecs.open('../words/nonstopwords.w', 'wb', encoding='utf-8') as vwfile:
		for w in filer.readlines():
			ws = w[:w.index(':')]
			if(unicode(ws,'utf-8') not in stopwordlist):
				vwfile.write(unicode(w,'utf-8'))
    
print("Fin words")

count = 0
with open('../words/allwords_cleaned_test.c', 'r') as filer:
    	with codecs.open('../words/nonstopwords_cleaned_train.c', 'wb', encoding='utf-8') as vwfile:
		for line in filer.readlines():
			if(count % 1000 == 0):
				print(count)
			cleanedLine = []
			for w in line.split():
				if(unicode(w,'utf-8') not in stopwordlist):
					cleanedLine.append(w)
			vwfile.write(unicode(' '.join(cleanedLine) + '\n','utf-8'))
			count+=1
print("Fin cleaned")    
