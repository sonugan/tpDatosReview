##Quita del archivo allwords.w todas las stopwords
###Entrada: allwords.w
###Salida: nonstopwords.w

# -*- coding: utf-8 -*-,
import sys
import re
import codecs
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse, io

##cargo la matriz
#newm = io.mmread("test.mtx")
#print(newm.tocsr().toarray())


indptr = [0]
indices = []
data = []
vocabulary = {}

count = 0
with open('../words/allwords_cleaned_train.c', 'r') as filer:
	for d in filer.readlines():
		if(count % 1000 == 0):
			print(count)
		count+=1
		for term in d:
			index = vocabulary.setdefault(term, len(vocabulary))
			indices.append(index)
			data.append(1)
			indptr.append(len(indices))
#csr_matrix((data, indices, indptr), dtype=int).toarray()
m = csr_matrix((data, indices, indptr), dtype=int)
io.mmwrite('../matrix/allwords_train.m', m)

print("Fin")

