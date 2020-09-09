import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim
import numpy as np
from scipy.spatial import distance

from tqdm.auto import tqdm
import pickle
import gc
import json
import pandas as pd

import matplotlib.pyplot as plt
#%matplotlib inline

# == recnn ==
import sys
sys.path.append("../../")
import recnn


cuda = torch.device('cuda')
frame_size = 10
# https://drive.google.com/open?id=1t0LNCbqLjiLkAMFwtP8OIYU-zPUCNAjK
meta = json.load(open('parsed/omdb.json'))
tqdm.pandas()
frame_size = 10
batch_size = 1
# embeddgings: https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL
dirs = recnn.data.env.DataPath(
    base="",
    embeddings="ml20_pca128.pkl",
    ratings="ml-20m/ratings.csv",
    cache="frame_env.pkl", # cache will generate after you run
    use_cache=True
)
env = recnn.data.env.FrameEnv(dirs, frame_size, batch_size)

ddpg = recnn.nn.models.Actor(1290, 128, 256).to(cuda)
td3 = recnn.nn.models.Actor(1290, 128, 256).to(cuda)
ddpg.load_state_dict(torch.load('models/ddpg_policy.pt'))
#td3.load_state_dict(torch.load('models/td3_policy.pt'))

#test_batch = next(iter(env.test_dataloader))
train_batch = next(iter(env.train_dataloader))
state, action, reward, next_state, done = recnn.data.get_base_batch(train_batch)

def rank(gen_action, metric):
    scores = []
    for i in env.base.key_to_id.keys():
        if i == 0 or i == '0':
            continue
        scores.append([i, metric(env.base.embeddings[env.base.key_to_id[i]], gen_action)])
    scores = list(sorted(scores, key = lambda x: x[1]))
    scores = scores[:10]
    ids = [i[0] for i in scores]
    for i in range(10):
        scores[i].extend([meta[str(scores[i][0])]['omdb'][key]  for key in ['Title',
                                'Genre', 'Language', 'Released', 'imdbRating']])
        # scores[i][3] = ' '.join([genres_dict[i] for i in scores[i][3]])

    indexes = ['id', 'score', 'Title', 'Genre', 'Language', 'Released', 'imdbRating']
    table_dict = dict([(key,[i[idx] for i in scores]) for idx, key in enumerate(indexes)])
    table = pd.DataFrame(table_dict)
    return table

ddpg_action = ddpg(state)
# pick random action
ddpg_action = ddpg_action[np.random.randint(0, state.size(0), 1)[0]].detach().cpu().numpy()

from pandas.plotting import table
import subprocess
import matplotlib.pyplot as plt
#%matplotlib inline

#from jupyterthemes import jtplot
#jtplot.style(theme='grade3')

print(rank(ddpg_action, distance.euclidean))

print(rank(ddpg_action, distance.cosine))

rank(ddpg_action, distance.correlation) # looks similar to cosine

rank(ddpg_action, distance.canberra)

rank(ddpg_action, distance.minkowski)

rank(ddpg_action, distance.chebyshev)

rank(ddpg_action, distance.braycurtis)

rank(ddpg_action, distance.cityblock)

###############TD3##############

td3_action = td3(state)
# pick random action
td3_action = td3_action[np.random.randint(0, state.size(0), 1)[0]].detach().cpu().numpy()

from pandas.plotting import table
import matplotlib.pyplot as plt
#%matplotlib inline

#from jupyterthemes import jtplot
#jtplot.style(theme='grade3')

rank(td3_action, distance.euclidean)

rank(td3_action, distance.cosine)

rank(td3_action, distance.correlation) # looks similar to cosine

rank(td3_action, distance.canberra)

rank(td3_action, distance.minkowski)

rank(td3_action, distance.chebyshev)

rank(td3_action, distance.braycurtis)

rank(td3_action, distance.cityblock)

