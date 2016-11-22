# -*- coding: utf-8 -*-,
import helpers as helper
import re
import csv
import unicodecsv as csv
from spacy.en import English

parser = English()

def tokenize_text(text):
    textArray = []

    parsed = parser(text)
    for t in parsed:
        textArray.append(t.orth_)
        
    return textArray



count = 0
vocab = dict()
with open('../../Data/train_prep_4.csv', 'r') as csvfile1:
    spamreader = csv.reader(csvfile1)
    for text in spamreader:
        if(count % 1000 == 0):
            print(count)
        
        for token in tokenize_text(text[1]):
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1
        count+=1

with open('../../Data/train_prep_voc4.csv', 'wb') as csvfile2:
    writer = csv.writer(csvfile2)
    for token in vocab:
        writer.writerow([token, str(vocab[token])])
                

