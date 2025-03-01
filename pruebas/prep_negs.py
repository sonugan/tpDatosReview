from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentences = ["VADER is smart, handsome, and funny.", # positive sentence example
	"VADER is smart, handsome, and funny!", # punctuation emphasis handled correctly (sentiment intensity adjusted)
	"VADER is very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
	"VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
	"VADER is VERY SMART, handsome, and FUNNY!!!",# combination of signals - VADER appropriately adjusts intensity
	"VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",# booster words & punctuation make this close to ceiling for score
	"The book was good.",         # positive sentence
]
paragraph = "It was one of the worst movies I've seen, despite good reviews. \
Unbelievably bad acting!! Poor direction. VERY poor production. \
The movie was bad. Very bad movie. VERY bad movie. VERY BAD movie. VERY BAD movie!"

from nltk import tokenize
lines_list = tokenize.sent_tokenize(paragraph)
sentences.extend(lines_list)

tricky_sentences = [
	"Most automated sentiment analysis tools are shit.",
	"VADER sentiment analysis is the shit.",
	"Sentiment analysis has never been good.",
	"Sentiment analysis with VADER has never been this good.",
	"Warren Beatty has never been so entertaining.",
	"but then it breaks",
	"usually around the time the 90 day warranty expires",
	"the twin towers collapsed today",
	"However, Mr. Carter solemnly argues, his client carried out the kidnapping \
	under orders and in the ''least offensive way possible.''"
]
sentences.extend(tricky_sentences)
sid = SentimentIntensityAnalyzer()
for sentence in sentences:
	print(sentence)
	ss = sid.polarity_scores(sentence)
	for k in sorted(ss):
	    print(str(k) + str(ss[k]))
	print()
