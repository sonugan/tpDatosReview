# -*- coding: utf-8 -*-,
import csv
from nltk.stem.wordnet import WordNetLemmatizer          
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from nltk.corpus import stopwords


'''
	Toma del archivo train.csv (cambiar el path) todos los textos de las reviews y los agrega a una lista (data)
	y las clasificaciones de cada review: 1,2,3,4,5 ...
	count muestra el progreso en el avance del procesamiento del archivo de train
'''
def get_train_data(path, vectorizer = None, pred_pos = 0, text_pos = 1, tf_idf=False, remove_header=False):
    data = []
    target = []
    count = 0
    with open(path, 'r') as csvfile1:
        spamreader = csv.reader(csvfile1)
        for rowlist in spamreader:
            if not remove_header or count > 0:
                data.append(rowlist[text_pos])
                target.append(rowlist[pred_pos])
            count += 1
    x_train_counts = vectorizer.fit_transform(data)
    if tf_idf == True:
        tfidf_transformer = TfidfTransformer()
        x_train_counts = tfidf_transformer.fit_transform(x_train_counts)
    return [target, x_train_counts]

def get_data(path, pred_pos=0, text_pos=1, remove_header=False):
    data = []
    target = []
    count = 0
    with open(path, 'r') as csvfile1:
        spamreader = csv.reader(csvfile1)
        for rowlist in spamreader:
            if not remove_header or count > 0:
                data.append(rowlist[text_pos])
                target.append(rowlist[pred_pos])
            count += 1
 
    return [target, data]

def get_vectorized_data(path, filter_list, pred_pos = 0, text_pos = 1, tf_idf=False, remove_header=False):
    data = []
    target = []
    count = 0
    with open(path, 'r') as csvfile1:
        spamreader = csv.reader(csvfile1)
        for rowlist in spamreader:
            if not remove_header or count > 0:
                data.append(rowlist[text_pos])
                target.append(rowlist[pred_pos])
            count += 1
    x_train_counts = vectorizer.fit_transform(data)
    if tf_idf == True:
        tfidf_transformer = TfidfTransformer()
        x_train_counts = tfidf_transformer.fit_transform(x_train_counts)
    return [target, x_train_counts]


### SP
#ES: stop_words != None
#MIN3: min_df = 3
#ES + MIN3: stop_words != None, min_df = 3
#STEAM: stem = True
#LEMMA: lemma = True
def get_vectorizer(stop_words=None, min_df = 3, stem=False, lemma=False, lowercase=True, ngram_range = (1,1), vocabulary = None):
    tokenizer = None
    if stem:
        tokenizer = tokenize_steam
    if lemma:
        tokenizer = tokenize_lemma
    return CountVectorizer(stop_words=stop_words, min_df=min_df, tokenizer=tokenizer, lowercase=lowercase, ngram_range=ngram_range, vocabulary= vocabulary)

#### STEAM
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize_steam(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

#### LEMMA
lmtzr = WordNetLemmatizer()
def lemma_tokens(tokens, stemmer):
    lemmed = []
    for item in tokens:
        lemmed.append(lmtzr.lemmatize(item))
    return lemmed

def tokenize_lemma(text):
    tokens = word_tokenize(text)
    stems = lemma_tokens(tokens, stemmer)
    return stems

def saveKaggleFile(path, predictions, ids):
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    ##Creo el archivo para subir al kaggle
    with open(path, 'wb') as csvfile2:
        writer = csv.writer(csvfile2)
        writer.writerow(['Id', 'Prediction'])
        for idx, prediction in enumerate(predictions):
            writer.writerow([str(ids[idx]), str(prediction)])
            if(prediction == '1'):
                count1+=1
            if(prediction == '2'):
                count2+=1
            if(prediction == '3'):
                count3+=1
            if(prediction == '4'):
                count4+=1
            if(prediction == '5'):
                count5+=1

    print('Resultados:')
    print('Cantidad de 1:' + str(count1))
    print('Cantidad de 2:' + str(count2))
    print('Cantidad de 3:' + str(count3))
    print('Cantidad de 4:' + str(count4))
    print('Cantidad de 5:' + str(count5))


'''
coso = dict()
target, data = get_train_data('../../Data/train_prep.csv', vectorizer = get_vectorizer(), pred_pos=0, text_pos=1)
for t in target:
    if t in coso:
        coso[t] += 1
    else:
        coso[t] = 1
print coso'''
#{'1': 41775, '3': 34118, '2': 23871, '5': 290403, '4': 64593}

#Armo un archivo para cada tipo:

'''
count = 0

with open('../../Data/train_prep.csv', 'r') as csvfile1:
    spamreader = csv.reader(csvfile1)
    with open('../../Data/train_prep_5.csv', 'wb') as csv_out:
        writer = csv.writer(csv_out)
        for rowlist in spamreader:
            if rowlist[0] == '5':   
                writer.writerow([rowlist[0],rowlist[1]])
            count += 1
            
count = 0            
with open('../../Data/train_prep.csv', 'r') as csvfile1:
    spamreader = csv.reader(csvfile1)
    with open('../../Data/train_prep_2.csv', 'wb') as csv_out:
        writer = csv.writer(csv_out)
        for rowlist in spamreader:
            if count < 23871 and rowlist[0] == '2':   
                writer.writerow([rowlist[0],rowlist[1]])
            count += 1
count = 0
with open('../../Data/train_prep.csv', 'r') as csvfile1:
    spamreader = csv.reader(csvfile1)
    with open('../../Data/train_prep_3.csv', 'wb') as csv_out:
        writer = csv.writer(csv_out)
        for rowlist in spamreader:
            if count < 23871 and rowlist[0] == '3':   
                writer.writerow([rowlist[0],rowlist[1]])
            count += 1          
count = 0
with open('../../Data/train_prep.csv', 'r') as csvfile1:
    spamreader = csv.reader(csvfile1)
    with open('../../Data/train_prep_4.csv', 'wb') as csv_out:
        writer = csv.writer(csv_out)
        for rowlist in spamreader:
            if count < 23871 and rowlist[0] == '4':   
                writer.writerow([rowlist[0],rowlist[1]])
            count += 1              
count = 0
with open('../../Data/train_prep.csv', 'r') as csvfile1:
    spamreader = csv.reader(csvfile1)
    with open('../../Data/train_prep_5.csv', 'wb') as csv_out:
        writer = csv.writer(csv_out)
        for rowlist in spamreader:
            if count < 23871 and rowlist[0] == '5':   
                writer.writerow([rowlist[0],rowlist[1]])
            count += 1            

with open('../../Data/train_prep_min.csv', 'wb') as csv_out:
    writer = csv.writer(csv_out)
    with open('../../Data/train_prep_1.csv', 'r') as csvfile1:
        spamreader = csv.reader(csvfile1)
        for rowlist in spamreader: 
            writer.writerow(rowlist)
    with open('../../Data/train_prep_2.csv', 'r') as csvfile1:
        spamreader = csv.reader(csvfile1)
        for rowlist in spamreader: 
            writer.writerow(rowlist)
    with open('../../Data/train_prep_3.csv', 'r') as csvfile1:
        spamreader = csv.reader(csvfile1)
        for rowlist in spamreader: 
            writer.writerow(rowlist)
    with open('../../Data/train_prep_4.csv', 'r') as csvfile1:
        spamreader = csv.reader(csvfile1)
        for rowlist in spamreader: 
            writer.writerow(rowlist)        
    with open('../../Data/train_prep_5.csv', 'r') as csvfile1:
        spamreader = csv.reader(csvfile1)
        for rowlist in spamreader: 
            writer.writerow(rowlist)
'''           
