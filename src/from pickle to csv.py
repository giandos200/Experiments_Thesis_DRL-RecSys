from __future__ import unicode_literals
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

def Rated_Items(test_dict):
    rated = {}
    for u in tqdm(test_dict.keys()):
        rated[u] = [it for it, rat, timestamp in test_dict[u]]

    return rated

with open("models/train.pkl", 'rb') as f:
    train = pickle.load(f)

with open("models/test.pkl", 'rb') as f:
    test = pickle.load(f)


def transform_to_csv(train):
    list = []
    for u in tqdm(train.keys()):
        for it, rat, t in train[u]:
            list.append((u, it, rat, t))
    return list


train_list = transform_to_csv(train)
test_list = transform_to_csv(test)
train_df = pd.DataFrame(train_list, columns=["userId", "movieId", "rating", "timestamp"])
test_df = pd.DataFrame(test_list, columns=["userId", "movieId", "rating", "timestamp"])
train_df.to_csv(r'dict_vari/train.csv', sep=',')
test_df.to_csv(r'dict_vari/test.csv', sep='\t', encoding='utf-8')

print(train_df['rating'], train_df["rating"])

Rated = Rated_Items(train)

a_file = open("dict_vari/Rated_train.pkl", "wb")
pickle.dump(Rated, a_file)
a_file.close()
