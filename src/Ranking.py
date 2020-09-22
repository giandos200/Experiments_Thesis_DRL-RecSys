import json
import pickle
# == recnn ==
import sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from numba import jit
from scipy.spatial import distance
from tqdm.auto import tqdm



class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size, init_w=3e-1):
        super(Actor, self).__init__()

        #self.drop_layer = nn.Dropout(p=0.5)

        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        # state = self.state_rep(state)
        x = F.relu(self.linear1(state))
        #x = self.drop_layer(x)
        x = F.relu(self.linear2(x))
        #x = self.drop_layer(x)
        # x = torch.tanh(self.linear3(x)) # in case embeds are -1 1 normalized
        x = torch.tanh(self.linear3(x))  # in case embeds are standard scaled / wiped using PCA whitening
        # return state, x
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size, init_w=3e-5):
        super(Critic, self).__init__()

        #self.drop_layer = nn.Dropout(p=0.5)

        self.linear1 = nn.Linear(input_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        #x = self.drop_layer(x)
        x = F.relu(self.linear2(x))
        #x = self.drop_layer(x)
        x = self.linear3(x)
        return x

# %matplotlib inline
sys.path.append("../../")
import recnn

with open("ml1_pca128_norm.pkl", 'rb') as f:
    embedding = pickle.load(f)

with open("dict_vari/Rated_train.pkl", 'rb') as f:
    Rated = pickle.load(f)

with open("ml-1m/train.pkl", 'rb') as f:
    train_dict = pickle.load(f)

cuda = torch.device('cuda')
frame_size = 5
# https://drive.google.com/open?id=1t0LNCbqLjiLkAMFwtP8OIYU-zPUCNAjK
meta = json.load(open('parsed/omdb.json'))
tqdm.pandas()
batch_size = 1
# embeddgings: https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL
dirs = recnn.data.env.DataPath(
    base="",
    embeddings="ml1_pca128_norm.pkl",
    ratings="dict_vari/train.csv",
    cache="frame_env.pkl",  # cache will generate after you run
    use_cache=False
)
env = recnn.data.env.FrameEnv(dirs, frame_size, batch_size)

ddpg = Actor(129 * frame_size, 128, 256).to(cuda)

# td3 = recnn.nn.models.Actor(129*frame_size, 128, 256).to(cuda)
ddpg.load_state_dict(torch.load('frame_s5_gamma_0_99/ddpg_policy.pt')) #policy
# td3.load_state_dict(torch.load('models/td3_policy.pt'))
Qvalue = Critic(129 * frame_size, 128, 256).to(cuda)
Qvalue.load_state_dict(torch.load('frame_s5_gamma_0_99/ddpg_value.pt')) #value




def recommendation(dict, embeddings, cos, rated, topk, frame_size):
    recommendations = {}
    for u in tqdm(dict.keys()):
        state = []
        for i in range(topk):
            if i == 0:
                list1 = torch.cat([embeddings[it] for it, rat, t in dict[u][-frame_size:]], dim=0)
                rat = torch.FloatTensor([rat for it, rat, t in dict[u][-frame_size:]])
                state = torch.cat([list1, rat]).unsqueeze(0).to(cuda)
            ddpg_action = ddpg(state)
            #Qvalue_action = Qvalue(state, ddpg_action)
            #Qvalue_action = Qvalue_action.detach().cpu().item()
            output = torch.abs(1-cos(ddpg_action, env.base.embeddings.to(cuda))).cpu()
            item = env.base.embeddings[torch.argmin(output).item()].to(cuda)
            Qvalue_action = Qvalue(state, item.unsqueeze(0))
            scores = (env.base.id_to_key[torch.argmin(output).item()], torch.min(output).item(), Qvalue_action.cpu().item())
            if scores[0] in rated[u] or Qvalue_action < 0:
                 sorte, index = torch.sort(output)
                 for v in (env.base.id_to_key.keys()):
                     item = env.base.embeddings[index[v + 1].item()].to(cuda)
                     Qvalue_action = Qvalue(state, item.unsqueeze(0))
                     scores = env.base.id_to_key[index[v + 1].item()], sorte[v + 1].item(), Qvalue_action.cpu().item()
                     if scores[0] in Rated[u] or Qvalue_action < 0:
                         continue
                     else:
                         Rated[u].append(env.base.id_to_key[index[v + 1].item()])
                         break
            state = torch.cat([torch.cat([state[:,128:128 * (frame_size)], item.unsqueeze(0)], 1), torch.cat([state[:,-frame_size+1:], Qvalue_action], 1)], 1)
            if i == 0:
                recommendations[u] = [list(scores)]
            else:
                recommendations[u].append(list(scores))

    return recommendations


cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#cos = nn.DataParallel(cos, device_ids=[0,1,2,3])
Dict_rec = recommendation(train_dict, embedding, cos, Rated, topk=20, frame_size=5)

with open('frame_s5_gamma_0_99/rec.pkl', 'wb') as f:
    pickle.dump(Dict_rec,f)
    f.close()


input()

# test_batch = next(iter(env.test_dataloader))
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
    scores = list(sorted(scores, key=lambda x: x[1]))
    scores = scores[:10]
    ids = [i[0] for i in scores]
    for i in range(10):
        scores[i].extend([meta[str(scores[i][0])]['omdb'][key] for key in ['Title',
                                                                           'Genre', 'Language', 'Released',
                                                                           'imdbRating']])
        # scores[i][3] = ' '.join([genres_dict[i] for i in scores[i][3]])

    indexes = ['id', 'score', 'Title', 'Genre', 'Language', 'Released', 'imdbRating']
    table_dict = dict([(key, [i[idx] for i in scores]) for idx, key in enumerate(indexes)])
    table = pd.DataFrame(table_dict)
    return table


ddpg_action = ddpg(state)
# pick random action
ddpg_action = ddpg_action[np.random.randint(0, state.size(0), 1)[0]].detach().cpu().numpy()

Qvalue_action = Qvalue(iola, iola2)
# %matplotlib inline

# from jupyterthemes import jtplot
# jtplot.style(theme='grade3')

print(rank(ddpg_action, distance.euclidean))

print(rank(ddpg_action, distance.cosine))

rank(ddpg_action, distance.correlation)  # looks similar to cosine

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
