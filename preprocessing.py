# import pandas as pd
# from nltk.corpus import wordnet as wn
#
#
# def is_noun(word):
#     print(word + '      ' + wn.synsets(word)[0].pos() if wn.synsets(word) else ' x ')
#     return True if wn.synsets(word) and wn.synsets(word)[0].pos() != 'n' else False
#
#
# def prepare_input_data(data_path):
#     data = pd.read_json('input/train.json')
#     for i in range(0, len(data['ingredients'])):
#         print(">" + ' '.join(data['ingredients'][i]))
#         ingredients_sum = ' '.join(data['ingredients'][i]).split()
#         for word in ingredients_sum:
#             is_noun(word)  # + ' true ' if is_noun(word) else ' false ')
#         exit(1)
#
#
# # data['ingredients_string'] = [re.sub('a-zA-Z', '', ((','.join(w)).replace(' ', '').replace(',', ' ').lower()))
# #                              for w in data['ingredients']]
#
# data = prepare_input_data('input/train.json')
#
# # words = ['amazing', 'interesting', 'love', 'great', 'nice', 'fresh', 'big']
#
# # for w in words:
#
# # tmp = wn.synsets(w)[0].pos()
# # print(w, ":", tmp)

measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Francisco', 'temperature': 18.},
]

from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()

vec.fit_transform(measurements).toarray()

print(vec.get_feature_names())
