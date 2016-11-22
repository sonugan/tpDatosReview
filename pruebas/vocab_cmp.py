# -*- coding: utf-8 -*-,
import helpers as helper
import re
import csv
import unicodecsv as csv
from nltk.corpus import stopwords

stopwordlist = stopwords.words('english')

if False:
    count = 0
    with open('../../Data/train_prep_voc1.csv', 'r') as csv4:
        with open('../../Data/train_prep_voc5.csv', 'r') as csv5:
            with open('../../Data/train_prep_voc1-5.csv', 'wb') as csvo:
                reader4 = csv.reader(csv4)
                reader5 = csv.reader(csv5)
                writer = csv.writer(csvo)
                words5 = []
                for word in reader5:
                    words5.append(word[1])
                for word in reader4:
                    if(count % 1000 == 0):
                        print(count)
                    if word[1] not in words5:
                        writer.writerow(word)
                    count += 1

#Armo un archivo con los cosos que no tiene el 5
if False:
    count = 0
    realwords5 = []
    with open('../../Data/train_prep_voc_all-5.csv', 'wb') as csvo:
        writer_o = csv.writer(csvo)
        with open('../../Data/train_prep_voc5.csv', 'r') as csv5:
            reader5 = csv.reader(csv5)
            words5 = []            
            for word in reader5:
                words5.append(word)
            print "1"
            with open('../../Data/train_prep_voc1-5.csv', 'r') as csvin:
                reader_i = csv.reader(csvin)
                words = []
                for line in reader_i:
                    words.append(line[0])
                    writer_o.writerow([line[0]])
                count = 0
                for w in words5:
                    if count % 100 == 0:
                        print count
                    if w[0] not in words:
                        realwords5.append(w)
                    count += 1
            print "2"
            with open('../../Data/train_prep_voc2-5.csv', 'r') as csvin:
                reader_i = csv.reader(csvin)
                words = []
                for line in reader_i:
                    words.append(line[0])
                    writer_o.writerow([line[0]])
                count = 0
                for w in words5:
                    if count % 100 == 0:
                        print count
                    if w[0] not in words:
                        realwords5.append(w)
                    count += 1
            print "3"
            with open('../../Data/train_prep_voc3-5.csv', 'r') as csvin:
                reader_i = csv.reader(csvin)
                words = []
                for line in reader_i:
                    words.append(line[0])
                    writer_o.writerow([line[0]])
                count = 0
                for w in words5:
                    if count % 100 == 0:
                        print count
                    if w[0] not in words:
                        realwords5.append(w)
                    count += 1
            print "4"
            with open('../../Data/train_prep_voc4-5.csv', 'r') as csvin:
                reader_i = csv.reader(csvin)
                words = []
                for line in reader_i:
                    words.append(line[0])
                    writer_o.writerow([line[0]])
                count = 0
                for w in words5:
                    if count % 100 == 0:
                        print count
                    if w[0] not in words:
                        realwords5.append(w)
                    count += 1
            #Escribo los que están solo en el 5:
            print "5"
            count = 0
            for w in realwords5:
                if count % 100 == 0:
                    print count
                writer_o.writerow(w)
                count += 1
    
#Elimino las palabras que aparecen mas de una vez                
if True:
    count = 0
    words = dict()
    with open('../../Data/train_prep_voc_all-5.csv', 'r') as csv4:
        reader4 = csv.reader(csv4)
        for w in reader4:
            if count % 100 == 0:
                print count
            if w[0] not in words:
                words[w[0]] = 1
            count += 1
    with open('../../Data/train_prep_voc_all-5.csv', 'wb') as csv4:
        writer4 = csv.writer(csv4)
        for k in words:
            writer4.writerow([k])
                