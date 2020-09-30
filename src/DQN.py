import gc

import recnn as recnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
#%matplotlib inline


# == recnn ==
import sys
sys.path.append("../../")
#import recnn

device = torch.device('cuda')
# ---
frame_size = 10
batch_size = 10
embed_dim  = 128
# ---

tqdm.pandas()
ratings = pd.read_csv('ml-1m/ratings.csv')
keys = list(sorted(ratings['movieId'].unique()))
key_to_id = dict(zip(keys, range(len(keys))))
user_dict, users = recnn.data.prepare_dataset(ratings, key_to_id) #, frame_size)

del ratings
gc.collect()
clear_output(True)
clear_output(True)
print('Done!')


class DuelDQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DuelDQN, self).__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.advantage = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                       nn.Linear(128, action_dim))
        self.value = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()


def dqn_update(step, batch, params, learn=True):
    batch = [i.to(device) for i in batch]
    items, next_items, ratings, next_ratings, action, reward, done = batch
    b_size = items.size(0)
    state = torch.cat([embeddings(items).view(b_size, -1), ratings], 1)
    next_state = torch.cat([embeddings(next_items).view(b_size, -1), next_ratings], 1)

    q_values = dqn(state)
    with torch.no_grad():
        next_q_values = target_dqn(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + params['gamma'] * next_q_value * (1 - done)

    loss = (q_value - expected_q_value).pow(2).mean()

    if learn:
        writer.add_scalar('value/train', loss, step)
        embeddings_optimizer.zero_grad()
        value_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dqn.parameters(), -1, 1)
        embeddings_optimizer.step()
        value_optimizer.step()
    else:
        writer.add_histogram('q_values', q_values, step)
        writer.add_scalar('value/test', loss, step)

    return loss.item()


def run_tests():
    test_batch = next(iter(test_dataloader))
    losses = dqn_update(step, test_batch, params, learn=False)
    return losses

def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


# === DQN settings ===

params = {
    'gamma'        : 0.99,
    'value_lr'     : 1e-5,
    'embeddings_lr': 1e-5,
}

# === end ===

dqn = DuelDQN((embed_dim + 1) * frame_size, len(keys)).to(device)
target_dqn = DuelDQN((embed_dim + 1) * frame_size, len(keys)).to(device)
embeddings = nn.Embedding(len(keys), embed_dim).to(device)
embeddings.load_state_dict(torch.load('models/anomaly.pt'))
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()

value_optimizer = optim.Adam(dqn.parameters(),
                              lr=params['value_lr'])
embeddings_optimizer = optim.Adam(embeddings.parameters(),
                              lr=params['embeddings_lr'])
writer = SummaryWriter(log_dir='../../runs')

n_epochs = 100
batch_size = 25

epoch_bar = tqdm(total=n_epochs)
train_users = users[:-5000]
test_users = users[-5000:]

def prepare_batch_wrapper(x):
    batch = recnn.data.prepare_batch_static_size(x, frame_size=frame_size)
    return batch

train_user_dataset = recnn.data.UserDataset(train_users, user_dict)
test_user_dataset = recnn.data.UserDataset(test_users, user_dict)
train_dataloader = DataLoader(train_user_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4,collate_fn=prepare_batch_wrapper)
test_dataloader = DataLoader(test_user_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4,collate_fn=prepare_batch_wrapper)

torch.cuda.empty_cache()

# --- config ---
plot_every = 30
# --- end ---

step = 1

train_loss = []
test_loss = []
test_step = []
mem_usage = []

torch.cuda.reset_max_memory_allocated()
for epoch in range(n_epochs):
    epoch_bar.update(1)
    for batch in tqdm(train_dataloader):
        loss = dqn_update(step, batch, params)
        train_loss.append(loss)
        step += 1
        if step % 30:
            torch.cuda.empty_cache()
            soft_update(dqn, target_dqn)
        if step % plot_every == 0:
            print('step', step)
            mem_usage.append(torch.cuda.max_memory_allocated())
            test_ = run_tests()
            test_step.append(step)
            test_loss.append(test_)
            plt.plot(train_loss)
            plt.plot(test_step, test_loss)
            plt.show()

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
print(embeddings(torch.tensor([[686]]).to(device)).detach().cpu().numpy())

torch.save(embeddings.state_dict(), "dqn.pt")

plt.plot(mem_usage)
