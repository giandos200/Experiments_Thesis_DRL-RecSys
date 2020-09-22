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

with open("prova_ml20_f10_g0_99/train.pkl", 'rb') as f:
    train = pickle.load(f)

with open("prova_ml20_f10_g0_99/test.pkl", 'rb') as f:
    test = pickle.load(f)

with open("frame_s5_gamma_0_99/rec.pkl", 'rb') as f:
    rec = pickle.load(f)

def transform_rec_to_csv(train):
    list = []
    for u in tqdm(train.keys()):
        for i in range(len(train[u])):
            if len(train[u][i])>3:
                pass
            else:
                it, score, rat = train[u][i]
                list.append((u, it, rat))
    return list

def transform_to_csv(train):
    list = []
    for u in tqdm(train.keys()):
        for it, rat, t in train[u]:
            list.append((u, it, rat, t))
    return list

rec_list = transform_rec_to_csv(rec)
rec_df = pd.DataFrame(rec_list, columns=["userId", "movieId", "rating"])
print("ecco")
rec_df.to_csv(r'prova_ml20_f10_g0_99/rec.tsv', sep='\t', encoding='utf-8', header=None, index=False)
input('solo_rec')
train_list = transform_to_csv(train)
test_list = transform_to_csv(test)
train_df = pd.DataFrame(train_list, columns=["userId", "movieId", "rating", "timestamp"])
test_df = pd.DataFrame(test_list, columns=["userId", "movieId", "rating", "timestamp"])
train_df.to_csv(r'prova_ml20_f10_g0_99/train_tab.csv', sep='\t', encoding='utf-8', header=None, index=False)
test_df.to_csv(r'prova_ml20_f10_g0_99/test_tab.csv', sep='\t', encoding='utf-8', header=None, index=False)

input("ecco")

print(train_df['rating'], train_df["rating"])

Rated = Rated_Items(train)

a_file = open("prova_ml20_f10_g0_99/Rated_train.pkl", "wb")
pickle.dump(Rated, a_file)
a_file.close()
