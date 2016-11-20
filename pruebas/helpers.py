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

### SP
#ES: stop_words != None
#MIN3: min_df = 3
#ES + MIN3: stop_words != None, min_df = 3
#STEAM: stem = True
#LEMMA: lemma = True
def get_vectorizer(stop_words=None, min_df = 3, stem=False, lemma=False, lowercase=True, ngram_range = (1,1)):
    tokenizer = None
    if stem:
        tokenizer = tokenize_steam
    if lemma:
        tokenizer = tokenize_lemma
    return CountVectorizer(stop_words=stop_words, min_df=min_df, tokenizer=tokenizer, lowercase=lowercase, ngram_range=ngram_range)

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