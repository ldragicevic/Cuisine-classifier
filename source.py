import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from preprocessing import *
import re


def processSingleIngredient(word):

    //return re.sub('[^a-zA-Z]+', '', word).lower()


pd.set_option('display.expand_frame_repr', False)

train_df = pd.read_json('input/train.json')
train_df['ingredients_string'] = [processSingleIngredient(w) for w in train_df['ingredients']]

count_vectorizer = CountVectorizer(analyzer='word', stop_words='english', preprocessor=processSingleIngredient,
                                   max_df=1,
                                   min_df=1)

X_train_counts = count_vectorizer.fit_transform(train_df['ingredients_string']).toarray()

print(X_train_counts)
exit(-1)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# clf = MultinomialNB().fit(X_train_tfidf, train_df['cuisine'])
clf = RandomForestClassifier().fit(X_train_tfidf, train_df['cuisine'])

test_df = pd.read_json('input/test.json')
prepare_input_data(test_df)

X_test_counts = count_vect.transform(test_df['ingredients_string'])
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = clf.predict(X_test_tfidf)
for i in range(0, 10):
    print(predicted[i])

output = pd.DataFrame(data={"id": test_df["id"], "cuisine": predicted})
output.to_csv("luka_randomforest.csv", index=False, quoting=3)
