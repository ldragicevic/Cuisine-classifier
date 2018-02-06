import re
import pandas as pd
from sklearn.svm import LinearSVC
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.expand_frame_repr', False)

lemmatizer = WordNetLemmatizer()
regex = re.compile('[^a-zA-Z]')

# Load data
train_data = pd.read_json('input/train.json')
test_data = pd.read_json('input/test.json')

# Clean data
train_data['ingredients_str'] = [regex.sub(' ', ' '.join(w).strip()).lower() for w in train_data['ingredients']]
train_data['ingredients_clean'] = [' '.join([lemmatizer.lemmatize(w) for w in line.split(' ')]) for line in
                                   train_data['ingredients_str']]

test_data['ingredients_str'] = [regex.sub(' ', ' '.join(w).strip()).lower() for w in test_data['ingredients']]
test_data['ingredients_clean'] = [' '.join([lemmatizer.lemmatize(w) for w in line.split(' ')]) for line in
                                  test_data['ingredients_str']]

tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words='english', token_pattern=r'\w+',
                        max_df=0.57, max_features=2 ** 12, use_idf=True, norm='l2')
train_tfidf = tfidf.fit_transform(train_data['ingredients_clean']).todense()
train_target = train_data['cuisine']

test_tfidf = tfidf.transform(test_data['ingredients_clean']).todense()

classifier = LinearSVC(dual=False, penalty='l2', C=0.85)
classifier.fit(train_tfidf, train_target)

predictions = classifier.predict(test_tfidf)

test_data['cuisine'] = predictions
test_data[['id', 'cuisine']].to_csv('LUKAAA1.csv', index=False)
