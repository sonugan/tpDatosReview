# -*- coding: utf-8 -*-,
import helpers as helper
import re
import csv
import unicodecsv as csv
from spacy.en import English

parser = English()

def removeNames(text):
    removedNames = []
    textArray = []

    parsed = parser(text)
    for e in parsed.ents:
        removedNames.append(e.orth_)

    for t in parsed:
        if t.orth_ not in removedNames:
            textArray.append(t.orth_)
        
    return [u' '.join(textArray), removedNames]



count = 0
target, data = helper.get_data('../../Data/test.csv', 0, 8, True)
with open('../../Data/test_NO_NAMES.csv', 'wb') as csvfile2:
    writer = csv.writer(csvfile2)
    with open('../../Data/test_NAMES.csv', 'wb') as csvfile3:
        writerNames = csv.writer(csvfile3)
        for text in data:
            if(count % 1000 == 0):
                print(count)
            if(count > 0):
                text = text.decode('utf8')
                text, names = removeNames(text)
                writer.writerow([text])
                if names != None:
                    for name in names:
                        writerNames.writerow([name])
            count+=1    

