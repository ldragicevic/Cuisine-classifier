import re
import pandas as pd
import numpy as np
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import Dense

pd.set_option('display.expand_frame_repr', False)

# Constants
HIDDEN_LAYER_NODE_COUNT = 100
REG_EXPR = re.compile('[^a-zA-Z]')
LEMMATIZER = WordNetLemmatizer()
TFIDF_VECTORIZER = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words='english', token_pattern=r'\w+',
                                   max_df=0.50, max_features=2 ** 12, use_idf=True, norm='l2')

# Load data
train_data = pd.read_json('input/train.json')
test_data = pd.read_json('input/test.json')

# Fit target
le = LabelEncoder().fit(train_data['cuisine'])
le_cuisine_number_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
le_number_cuisine_mapping = {v: k for k, v in le_cuisine_number_mapping.items()}

CUISINE_COUNT = len(le_cuisine_number_mapping)

# Conversion cuisine name to cuisine number
train_data['cuisine_number'] = [le_cuisine_number_mapping[w] for w in train_data['cuisine']]

# Clean data
train_data['ingredients_string'] = [REG_EXPR.sub(' ', ' '.join(w).strip()).lower() for w in train_data['ingredients']]
train_data['ingredients_clean'] = [' '.join([LEMMATIZER.lemmatize(w) for w in line.split(' ')]) for line in
                                   train_data['ingredients_string']]

test_data['ingredients_string'] = [REG_EXPR.sub(' ', ' '.join(w).strip()).lower() for w in test_data['ingredients']]
test_data['ingredients_clean'] = [' '.join([LEMMATIZER.lemmatize(w) for w in line.split(' ')]) for line in
                                  test_data['ingredients_string']]

# Vectorizing input features
train_tfidf_matrix = TFIDF_VECTORIZER.fit_transform(train_data['ingredients_clean']).toarray()
test_tfidf_matrix = TFIDF_VECTORIZER.transform(test_data['ingredients_clean']).toarray()

# Neural Network Model
X = train_tfidf_matrix
Y = train_data['cuisine_number']

model = Sequential()
model.add(Dense(HIDDEN_LAYER_NODE_COUNT, input_shape=(len(X[0]),), activation='relu'))
model.add(Dense(CUISINE_COUNT, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.fit(X, Y, epochs=200, batch_size=200)
model.fit(X, Y, epochs=30, batch_size=500)

# Predict cuisine; all outputs value
X = test_tfidf_matrix
predictions_output_all = model.predict(X)

predictions = [le_number_cuisine_mapping[np.argmax(p)] for p in predictions_output_all]

test_data['cuisine'] = predictions
test_data[['id', 'cuisine']].to_csv('OUTPUTLUKA_ep_30_batch_500.csv', index=False)
