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
        #x = torch.tanh(self.linear3(x)) # in case embeds are -1 1 normalized
        x = self.linear3(x)  # in case embeds are standard scaled / wiped using PCA whitening
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

with open("/home/giandomenico/Experiments_Thesis/src/ml1_pca128.pkl", 'rb') as f:
    embedding = pickle.load(f)

with open("/home/giandomenico/Experiments_Thesis/src/ml-1m/Rated_train.pkl", 'rb') as f:
    Rated = pickle.load(f)

with open("/home/giandomenico/Experiments_Thesis/src/ml-1m/train.pkl", 'rb') as f:
    train_dict = pickle.load(f)

cuda = torch.device('cuda')
frame_size = 10
# https://drive.google.com/open?id=1t0LNCbqLjiLkAMFwtP8OIYU-zPUCNAjK
#meta = json.load(open('parsed/omdb.json'))
tqdm.pandas()
batch_size = 1
# embeddgings: https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL
dirs = recnn.data.env.DataPath(
    base="",
    embeddings="/home/giandomenico/Experiments_Thesis/src/ml1_pca128.pkl",
    ratings="/home/giandomenico/Experiments_Thesis/src/ml-1m/train.csv",
    cache="frame_env.pkl",  # cache will generate after you run
    use_cache=False
)
env = recnn.data.env.FrameEnv(dirs, frame_size, batch_size)

ddpg = Actor(129 * frame_size, 128, 256).to(cuda)

# td3 = recnn.nn.models.Actor(129*frame_size, 128, 256).to(cuda)
ddpg.load_state_dict(torch.load('ddpg_policy.pt')) #policy
# td3.load_state_dict(torch.load('models/td3_policy.pt'))
Qvalue = Critic(129 * frame_size, 128, 256).to(cuda)
Qvalue.load_state_dict(torch.load('ddpg_value.pt')) #value


def recommendation(dict, embeddings, cos, rated, topk, frame_size):
    recommendations = {}
    for u in tqdm(dict.keys()):
        state = []
        #for i in range(topk):
        i=0
        list1 = torch.cat([embeddings[it] for it, rat, t in dict[u][-frame_size:]], dim=0)
        rat = torch.FloatTensor([rat for it, rat, t in dict[u][-frame_size:]])
        state = torch.cat([list1, rat]).unsqueeze(0).to(cuda)
        ddpg_action = ddpg(state)
        # Qvalue_action = Qvalue(state, ddpg_action)
        # Qvalue_action = Qvalue_action.detach().cpu().item()
        output = cos(ddpg_action, env.base.embeddings.to(cuda)).cpu()
        sorte, index = torch.sort(output, descending=True)
        for j in index.tolist():
            if i == topk:
                break
            Qvalue_action = Qvalue(state, env.base.embeddings[j].unsqueeze(0).to(cuda)).cpu().item()
            if env.base.id_to_key[j] in Rated[u] or Qvalue_action <= 0:
                continue
            else:
                scores = (env.base.id_to_key[j], sorte[i].item(), Qvalue_action)
                if i == 0:
                    recommendations[u] = [list(scores)]
                else:
                    recommendations[u].append(list(scores))
            i += 1


        # item = env.base.embeddings[torch.argmax(output).item()].to(cuda)
        # Qvalue_action = Qvalue(state, item.unsqueeze(0))
        # scores = (env.base.id_to_key[torch.argmax(output).item()], torch.max(output).item(), Qvalue_action.cpu().item())
        # if scores[0] in rated[u] or Qvalue_action <= 0:
        #     sorte, index = torch.sort(output, descending=True)
        #     for v in (env.base.id_to_key.keys()):
        #         item = env.base.embeddings[index[v].item()].to(cuda)
        #         Qvalue_action = Qvalue(state, item.unsqueeze(0))
        #         scores = env.base.id_to_key[index[v].item()], sorte[v].item(), Qvalue_action.cpu().item()
        #         if scores[0] in Rated[u] or Qvalue_action < 0:
        #             continue
        #         else:
        #             Rated[u].append(env.base.id_to_key[index[v].item()])
        #             break
        # else:
        #     Rated[u].append(env.base.id_to_key[torch.argmax(output).item()])
        # state = torch.cat([torch.cat([state[:, 128:128 * (frame_size)], item.unsqueeze(0)], 1),
        #                    torch.cat([state[:, -frame_size + 1:], Qvalue_action], 1)], 1)
        # if i == 0:
        #     recommendations[u] = [list(scores)]
        # else:
        #     recommendations[u].append(list(scores))

    return recommendations


cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#cos = nn.DataParallel(cos, device_ids=[0,1,2,3])
Dict_rec = recommendation(train_dict, embedding, cos, Rated, topk=20, frame_size=10)
print(Dict_rec[1])
#
with open('rec.pkl', 'wb') as f:
    pickle.dump(Dict_rec,f)
    f.close()

list1 = torch.cat([embedding[it] for it, rat, t in train_dict[1][-frame_size:]], dim=0)
rat = torch.FloatTensor([rat for it, rat, t in train_dict[1][-frame_size:]])
state = torch.cat([list1, rat]).unsqueeze(0).to(cuda)
