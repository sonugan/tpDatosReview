from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import helpers as helper
from nltk.corpus import stopwords

stopwordlist = stopwords.words('english')


clf = AdaBoostClassifier(n_estimators=100)
x_target, x_data = helper.get_train_data('../../Data/train_prep.csv', vectorizer = helper.get_vectorizer(stop_words=stopwordlist, min_df = 3), pred_pos=0, text_pos=1,tf_idf=True, remove_header=False)

scores = cross_val_score(clf, x_data, x_target)
print scores.mean()