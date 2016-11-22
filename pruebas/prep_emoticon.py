# -*- coding: utf-8 -*-
#Preprocesamiento inicial de los sets de train y test
#Elimina HTML, las palabras que tengan numeros en medio y signos de puntuacion y pasa caracteres a minusculas

import csv
import re
from nltk.corpus import stopwords


# Remueve stop words
def prepES(inputFile, regs):
	cantidad = 0
	count = 0
	cantuno = 0
	with open(inputFile, 'r') as csvfile1:
		spamreader = csv.reader(csvfile1)
		for rowlist in spamreader:
			#if(count % 1000 == 0):
			#	print(count)

			text = rowlist[8]
			lista = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
			#for reg in regs:
			#	lista = re.findall(reg, text)
			if(len(lista) > 0 and rowlist[6] == '5'):
				print(lista + [str(rowlist[6])])
				cantidad += 1
			if(rowlist[6] == '5'):
				cantuno += 1
			count+=1
	print(str(cantidad))
	print(str(cantuno))
print('Train')

def replace_emoticons(text):
	reg_Happiness = r'(?::|=)(?:-)?(?:\)|\])'
	reg_Sadness = r'(?::|=)(?:-)?(?:\(|\[)' 
	reg_Wink =  r'(?:\;)(?:-)?(?:\))'
	reg_kidding = r'(?::|\;)(?:-)?(?:\p|\P)'
	reg_Amused = r'(?::|=)(?:-)?(?:d|D)'
	reg_Anger = r'(?:\>)(?::|=)(?:-)?(?:\()'
	reg_Kiss =  r'(?::)(?:-)?(?:\*)'
	reg_Confused =  r'(?:o|O|0)(?:.)?(?:o|O|0)'
	reg_Devil = r'(?:3)(?::)(?:-)?(?:\))'
	reg_Cool = r'(?:8|\B)(?:-)?(?:\))'
	reg_Unsure = r'(?::)(?:-)?(?:\\|\/)'
	reg_Cry = r'(?::)?(?:\’)(?:-)?(?:\()'
	reg_Love = r'(?:<3)'
	reg_Shy = r'(?:\^\_\^)'
	reg_Blessed =  r'(?:o|O)(?::)(?:-)?(?:\))'
	reg_Hug =  r'(?:\>\(\^\_\^\)\<)|(?:\<\(\^\_\^\)\>)'
	reg_Squint =  r'(?:\-_\-)'
	reg_Surprised =  r'(?::)(?:-)?(?:o|O)'

	regs = [{'pattern':reg_Cry, 'replace':'__CRY__'}, {'pattern':reg_Devil, 'replace':'__DEVIL__'}, {'pattern':reg_Happiness, 'replace':'__HAPPY__'}, {'pattern':reg_Hug, 'replace':'__HUG__'}, {'pattern':reg_kidding, 'replace':'__KIDDING__'}, {'pattern':reg_Kiss, 'replace':'__KISS__'}, {'pattern':reg_Love, 'replace': '__LOVE__'},{'pattern':reg_Sadness, 'replace': '__SADNESS__'}, {'pattern':reg_Shy, 'replace':'__SHY__'}, {'pattern':reg_Squint, 'replace':'__SQUINT__'}, {'pattern':reg_Surprised, 'replace':'__SURPRISED__'}, {'pattern':reg_Unsure, 'replace':'__UNSURE__'}, {'pattern':reg_Wink, 'replace': '__WINK__'}]
	for reg in regs:
		text = re.sub(reg['pattern'], reg['replace'], text)
	return text
####### Me quedo con la prediccion y con el texto
#prepES('../../Data/test.csv')

#######Me quedo con el ID del registro y con el texto
#basicPrep('../../Data/test.csv', '../../Data/test_prep.csv', 0, 8)
'''
count = 0
with open(inputFile, 'r') as csvfile1:
	spamreader = csv.reader(csvfile1)
	for rowlist in spamreader:
		if(count % 1000 == 0):
			print(count)
		if(count > 0):
			text = rowlist[posText]
			text = text.lower()

			count+=1
'''


'''
#smiley_pattern = '(:\(|:\)|:-\))' # matches only the smileys ":)" and ":("
s = "Ju;ps<3t: to:)) =) te:-ps:i-pt ::-) :(:-(( ():: :):) :]:( :p ;) h=Do;-)l;--)a :[! hol>:-( aa:-da:-*a >:--( ---:-ii* o.O\
o.Oso asdjfasjdfj3:)ds3:(lf  asd B-) j 8)    :/ SD :-\  :’-(  :’’-("

#reg_smiley = '(?::|;|=)(?:-)?(?:\)|\(|D|P)'
#print(re.findall(reg_smiley,s))
reg_Happiness = r'(?::|=)(?:-)?(?:\)|\])'
print(re.findall(reg_Happiness,s))

reg_Sadness = r'(?::|=)(?:-)?(?:\(|\[)' 
print(re.findall(reg_Sadness,s))

reg_Wink =  r'(?:\;)(?:-)?(?:\))'
print(re.findall(reg_Wink,s))

reg_kidding = r'(?::|\;)(?:-)?(?:\p|\P)'
print(re.findall(reg_kidding,s))

reg_Amused = r'(?::|=)(?:-)?(?:d|D)'
print(re.findall(reg_Amused,s))

reg_Anger = r'(?:\>)(?::|=)(?:-)?(?:\()'
print(re.findall(reg_Anger,s))

reg_Kiss =  r'(?::)(?:-)?(?:\*)'
print(re.findall(reg_Kiss,s))

reg_Confused =  r'(?:o|O|0)(?:.)?(?:o|O|0)'
print(re.findall(reg_Confused,s))

#reg_Embarrased =  r'(?:\>)(?::|=)(?:-)?(?:\()'
#print(re.findall(reg_Embarrased,s))

reg_Devil = r'(?:3)(?::)(?:-)?(?:\))'
print(re.findall(reg_Devil,s))

reg_Cool = r'(?:8|\B)(?:-)?(?:\))'
print(re.findall(reg_Cool,s))

reg_Unsure = r'(?::)(?:-)?(?:\\|\/)'
print(re.findall(reg_Unsure,s))

reg_Cry = r'(?::)?(?:\’)(?:-)?(?:\()'
print(re.findall(reg_Cry,s))

reg_Love = r'(?:<3)'
print(re.findall(reg_Love,s))

s = "^_^ ^_|^"

reg_Shy = r'(?:\^\_\^)'
print(re.findall(reg_Shy,s))

s = "O:-))) -O:)--="

reg_Blessed =  r'(?:o|O)(?::)(?:-)?(?:\))'
print(re.findall(reg_Blessed,s))

s = ">(^_^)< <(^_^)> asdf >(^_^))<"

reg_Hug =  r'(?:\>\(\^\_\^\)\<)|(?:\<\(\^\_\^\)\>)'
print(re.findall(reg_Hug,s))

s="-_---- -__-"

reg_Squint =  r'(?:\-_\-)'
print(re.findall(reg_Squint,s))

s= ":-oooo----:--o ::o"

reg_Surprised =  r'(?::)(?:-)?(?:o|O)'
print(re.findall(reg_Surprised,s))

regs = [reg_Cry, reg_Devil, reg_Happiness, reg_Hug, reg_kidding, reg_Kiss, reg_Love, reg_Sadness, reg_Shy, reg_Squint, reg_Surprised, reg_Unsure, reg_Wink]
#prepES('../../Data/train.csv', regs)
'''
'''
def test_match(s):
    print 'Value: %s; Result: %s' % (
        s,
        'Matches!' if re.match(smiley_pattern, s) else 'Doesn\'t match.'
    )

should_match = [
    ':)',   # Single smile
    ':(',   # Single frown
    ':):)', # Two smiles
    ':(:(', # Two frowns
    ':):(', # Mix of a smile and a frown
	':-)'
]
should_not_match = [
    '',         # Empty string
    ':(foo',    # Extraneous characters appended
    'foo:(',    # Extraneous characters prepended
    ':( :(',    # Space between frowns
    ':( (',     # Extraneous characters and space appended
    ':(('       # Extraneous duplicate of final character appended
]

print('The following should all match:')
for x in should_match: test_match(x)

print('')   # Newline for output clarity

print('The following should all not match:')
for x in should_not_match: test_match(x)
'''








