import pandas as pd
import numpy as np
import recmetrics
import matplotlib.pyplot as plt
from surprise import Reader, SVD, Dataset
from surprise.model_selection import train_test_split
import pickle
from tqdm import tqdm

with open("models/train.pkl", 'rb') as f:
    train = pickle.load(f)

with open("models/test.pkl", 'rb') as f:
    test = pickle.load(f)

with open("frame_s10_gamma_0_8/rec.pkl", 'rb') as f:
    rec = pickle.load(f)

rmse = 0
denom = 0

for i in tqdm(test.keys()):

    for it, score , rat in rec[i]:
        for item, rating, t in test[i]:
            if item == it:
                denom += 1
                rmse += ((rat+10)/4 -rating)**2
    print(denom)
rmse = (rmse/denom)**(1/2)
print(rmse)



