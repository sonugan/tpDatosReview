# -*- coding: utf-8 -*-,

import csv
import re
from nltk import word_tokenize

def reviewsPerType(path, pos_pred, pos_text):
    count = 0
    dic_count = dict()
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            if(count > 0):
                prediction = line[pos_pred]
                if(prediction in dic_count):
                    dic_count[prediction] += 1
                else:
                    dic_count[prediction] = 1
            count += 1
    return dic_count

print(reviewsPerType('../../Data/train.csv', 6, 9))            

def extractFeacturesPerType(path, path_output, type, pos_pred, pos_text):
    count = 0
    vocabulary = dict()
    with open(path, 'r') as csv_input:
        reader = csv.reader(csv_input)
        for line in reader:
            if count % 1000 == 0:
                print(count)
            prediction = line[pos_pred]
            if(prediction == type):
                for token in word_tokenize(line[pos_text]):
                    if(utf_tok in vocabulary):
                        vocabulary[utf_tok] += 1
                    else:
                        vocabulary[utf_tok] = 1
            count+=1
        with open(path_output, 'wb') as csv_output:
            writer = csv.writer(csv_output)
            writer.writerow(vocabulary)

            
#Veo cuales son los features de cada uno de los tipos
extractFeacturesPerType('../../Data/train.csv', '../../Data/train_prep_1.csv', '1', 6, 9)