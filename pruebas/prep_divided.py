#Didive el archvio de train, uno por cada clase

# -*- coding: utf-8 -*-,
import csv
import re

def divide(inputFile, outputFile, target):
	count = 0
	with open(inputFile, 'r') as csvfile1:
		spamreader = csv.reader(csvfile1)
		with open(outputFile, 'wb') as csvfile2:
			writer = csv.writer(csvfile2)
			for rowlist in spamreader:
				if(count % 1000 == 0):
					print(count)
				if(count > 0 and rowlist[0] == target):
					writer.writerow([rowlist[1]])
				count+=1
print('1--------')
####### Tomo los clasificados como 1
#divide('../../Data/train_prep.csv', '../../Data/train_prep_1.csv', '1')

print('2--------')
####### Tomo los clasificados como 1
#divide('../../Data/train_prep.csv', '../../Data/train_prep_2.csv', '2')

print('3--------')
####### Tomo los clasificados como 1
#divide('../../Data/train_prep.csv', '../../Data/train_prep_3.csv', '3')

print('4--------')
####### Tomo los clasificados como 1
#divide('../../Data/train_prep.csv', '../../Data/train_prep_4.csv', '4')

print('5--------')
####### Tomo los clasificados como 1
#divide('../../Data/train_prep.csv', '../../Data/train_prep_5.csv', '5')


