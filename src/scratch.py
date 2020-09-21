import torch
import recnn
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

with open("ml20_pca128.pkl", 'rb') as f:
    embedding = pickle.load(f)

def train_test(dict):

    train_dict = {}
    test_dict = {}
    eliminati = 0
    for u in tqdm(dict.keys()):
        num = len(dict[u])
        if num < 20:
            eliminati += 1
            pass
        if num > 20:
            cutoff = round(num*0.8) - 1
            train_dict[u] = dict[u][:cutoff]
            test_dict[u] = dict[u][cutoff:]
        if u == dict.keys()[-1]:
            print(u, "<----ultimo user--/\--users eliminati---->", eliminati)
    return train_dict, test_dict

def Dict_userBased(dfuser, df):

    ratings = {}
    train_dict = {}
    test_dict = {}
    eliminati=0
    users = list(dfuser.unique())
    users.sort()
    for u in tqdm(users):
        select = df[dfuser == u]
        ratings[u] = list(zip(select['itemId'], select['rating'], select['timestamp']))
        num = len(ratings[u])
        if num<20:
            eliminati +=1
            print(eliminati)
            pass
        if num>=20:
            cutoff = round(num * 0.8) - 1
            train_dict[u] = ratings[u][:cutoff]
            test_dict[u] = ratings[u][cutoff:]
        if u == users[-1]:
            print(u, "<----ultimo user--/\--users eliminati---->", eliminati)

    return ratings, train_dict, test_dict

df = pd.read_csv("ml-1m/ratings.csv", sep=',')
df = df.sort_values('timestamp')
df['itemId'] = df['movieId']
#df.drop(columns=['timestamp', 'movieId'])
# df = df.groupby('userId').count()
# user_drop = 0
# for i in tqdm(df['movieId']):
#     if i < 20:
#         user_drop += 1
# print(user_drop)
user_based_dict, train_dict, test_dict = Dict_userBased(df['userId'], df)
# train_dict, test_dict = train_test(user_based_dict)
with open('models/train.pkl', 'wb') as f:
    pickle.dump(train_dict,f)
    f.close()

with open('models/test.pkl', 'wb') as f:
    pickle.dump(test_dict,f)
    f.close()