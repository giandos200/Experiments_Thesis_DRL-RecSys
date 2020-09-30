import pandas as pd
import numpy as np
import recmetrics
import matplotlib.pyplot as plt
from surprise import Reader, SVD, Dataset
from surprise.model_selection import train_test_split
import pickle
from tqdm import tqdm

with open("ml-1m/train.pkl", 'rb') as f:
    train = pickle.load(f)

with open("ml-1m/test.pkl", 'rb') as f:
    test = pickle.load(f)

with open("k-nn top10 ranking/rec_drr.pkl", 'rb') as f:
    rec = pickle.load(f)

rmse = 0
denom = 0

item_cov = []
test_item_cov = []
for i in tqdm(test.keys()):


    for it, score , rat in rec[i]:
        for item, rating, t in test[i]:
            test_item_cov.append(item)
            if item == it:
                denom += 1
                rmse += ((rat+10)/4 -rating)**2
        item_cov.append(it)
    print(denom)
item_notevoli = 0
for i in test.keys():
    for item,rating, t in test[i]:
        if rating >= 4:
            item_notevoli += 1



print(len(set(item_cov)))
print(len(set(test_item_cov)))
rmse = (rmse/denom)**(1/2)
print(rmse)
print(item_notevoli)



