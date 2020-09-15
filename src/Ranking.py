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
from numba import jit
import matplotlib.pyplot as plt
#%matplotlib inline

# == recnn ==
import sys
sys.path.append("../../")
import recnn

with open("ml20_pca128.pkl", 'rb') as f:
    embedding = pickle.load(f)

with open("dict_vari/Rated_train.pkl", 'rb') as f:
    Rated = pickle.load(f)

with open("models/train.pkl", 'rb') as f:
    train_dict = pickle.load(f)

cuda = torch.device('cuda')
frame_size = 10
# https://drive.google.com/open?id=1t0LNCbqLjiLkAMFwtP8OIYU-zPUCNAjK
meta = json.load(open('parsed/omdb.json'))
tqdm.pandas()
frame_size = 10
batch_size = 1
#embeddgings: https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL
# dirs = recnn.data.env.DataPath(
#     base="",
#     embeddings="ml20_pca128.pkl",
#     ratings="ml-20m/ratings.csv",
#     cache="frame_env.pkl", # cache will generate after you run
#     use_cache=False
# )
# env = recnn.data.env.FrameEnv(dirs, frame_size, batch_size)

ddpg = recnn.nn.models.Actor(129*frame_size, 128, 256).to(cuda)

#td3 = recnn.nn.models.Actor(129*frame_size, 128, 256).to(cuda)
ddpg.load_state_dict(torch.load('models/ddpg_policy.pt'))
#td3.load_state_dict(torch.load('models/td3_policy.pt'))
Qvalue = recnn.nn.models.Critic(129*frame_size, 128, 256).to(cuda)
Qvalue.load_state_dict(torch.load('models/ddpg_value.pt'))

@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta -= uv/np.sqrt(uu*vv)
    return np.abs(cos_theta)

#@jit(nopython=True)
def recommendation(dict, embeddings, metric, rated, topk, frame_size):
    recommendations = {}
    for u in tqdm(dict.keys()):
        state = []
        for i in range(topk):
            if i == 0:
                state = [np.append(embeddings[it].numpy(), rat) for it, rat, t in dict[u][-frame_size:]]
                state = (torch.from_numpy(np.asarray(state).flatten()).float()).to(cuda)
            ddpg_action = ddpg(state)
            Qvalue_action = Qvalue(state, ddpg_action)
            state = torch.cat([state[129:],torch.cat([ddpg_action,Qvalue_action])])
            ddpg_action = ddpg_action.detach().cpu().numpy()
            Qvalue_action = Qvalue_action.detach().cpu().numpy()
            scores = []
            for j in embeddings.keys():
                if j in rated[u]:
                    pass
                else:
                    scores.append([j, metric(embeddings[j].numpy(), ddpg_action)])
            scores = list(sorted(scores, key=lambda x: x[1]))[0]
            if i == 0:
                recommendations[u] = [scores, Qvalue_action]
            else:
                recommendations[u].append([scores,Qvalue_action])

    return recommendations

Dict_rec = recommendation(train_dict, embedding, cosine_similarity_numba, Rated, topk=10, frame_size=10)

input()

#test_batch = next(iter(env.test_dataloader))
train_batch = next(iter(env.train_dataloader))
state, action, reward, next_state, done = recnn.data.get_base_batch(train_batch)
test_batch = next(iter(env.test_dataloader))
state2, action2, reward2, next_state2, done2 = recnn.data.get_base_batch(test_batch)
iola = state[0]
iola2 = action[0]
print(state.size(), state2.size())

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

Qvalue_action = Qvalue(iola,iola2)
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

# td3_action = td3(state)
# # pick random action
# td3_action = td3_action[np.random.randint(0, state.size(0), 1)[0]].detach().cpu().numpy()
#
# from pandas.plotting import table
# import matplotlib.pyplot as plt
# #%matplotlib inline
#
# #from jupyterthemes import jtplot
# #jtplot.style(theme='grade3')
#
# rank(td3_action, distance.euclidean)
#
# rank(td3_action, distance.cosine)
#
# rank(td3_action, distance.correlation) # looks similar to cosine
#
# rank(td3_action, distance.canberra)
#
# rank(td3_action, distance.minkowski)
#
# rank(td3_action, distance.chebyshev)
#
# rank(td3_action, distance.braycurtis)
#
# rank(td3_action, distance.cityblock)

