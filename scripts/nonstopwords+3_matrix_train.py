import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

stopwordlist = stopwords.words('english')

data = []
target = []

dataTest = []
targetTest = []
count = 0
with open('../../Data/train.csv', 'r') as csvfile1:
	spamreader = csv.reader(csvfile1)
	for rowlist in spamreader:
		if(count % 1000 == 0):
			print(count)
		if(count >0 and count <= 10):
			dataTest.append(rowlist[9])
			targetTest.append(rowlist[6])
		if(count > 10):
			data.append(rowlist[9])
			target.append(rowlist[6])
		count+=1


count_vect = CountVectorizer(lowercase=True, stop_words=stopwordlist, min_df = 3, ngram_range=(2,2))
x_train_counts = count_vect.fit_transform(data)
print(x_train_counts.shape)


clf = MultinomialNB().fit(x_train_counts, target)

x_new_counts = count_vect.transform(dataTest)

predicted = clf.predict(x_new_counts)

print(targetTest)
print(predicted)
