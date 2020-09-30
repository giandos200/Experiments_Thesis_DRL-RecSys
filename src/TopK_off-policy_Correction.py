import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch_optimizer as optim

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from time import gmtime, strftime


import matplotlib.pyplot as plt
#%matplotlib inline


# == recnn ==
import sys
sys.path.append("../../")
import recnn

cuda = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# ---
frame_size = 10
batch_size = 10
n_epochs   = 3
plot_every = 200
num_items    = 10 # n items to recommend. Can be adjusted for your vram
# ---

tqdm.pandas()


def embed_batch(batch, item_embeddings_tensor, *args, **kwargs):
    return recnn.data.batch_contstate_discaction(batch, item_embeddings_tensor,
                                                 frame_size=frame_size, num_items=num_items)


def prepare_dataset(args_mut, kwargs):
    kwargs.set('reduce_items_to', num_items)  # set kwargs for your functions here!
    pipeline = [recnn.data.truncate_dataset, recnn.data.prepare_dataset]
    recnn.data.build_data_pipeline(pipeline, kwargs, args_mut)


# embeddgings: https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL
dirs = recnn.data.env.DataPath(
    base="",
    embeddings="ml20_pca128.pkl",
    ratings="ml-20m/ratings.csv",
    # IMPORTANT! I am using a different name for cache
    # If you change your pipeline, change the name as well!
    # Different pipelines must have different names!
    cache="frame_env_truncated.pkl",
    use_cache=False
)

env = recnn.data.env.FrameEnv(
    dirs, frame_size,
    batch_size,
    embed_batch=embed_batch,
    prepare_dataset=prepare_dataset,
    num_workers=1
)


class Beta(nn.Module):
    def __init__(self):
        super(Beta, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1290, num_items),
            nn.Softmax()
        )
        self.optim = optim.RAdam(self.net.parameters(), lr=1e-5, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, state, action):
        predicted_action = self.net(state)

        loss = self.criterion(predicted_action, action.argmax(1))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return predicted_action.detach()

beta_net   = Beta().to(cuda)
value_net  = recnn.nn.Critic(1290, num_items, 2048, 54e-2).to(cuda)
policy_net = recnn.nn.DiscreteActor(1290, num_items, 2048).to(cuda)
# as miracle24 has suggested https://github.com/awarebayes/RecNN/issues/7
# you can enable this to be more like the paper authors meant it to
policy_net.action_source = {'pi': 'beta', 'beta': 'beta'}

reinforce = recnn.nn.Reinforce(policy_net, value_net)
reinforce = reinforce.to(cuda)

reinforce.writer = SummaryWriter(log_dir='../../runs/ReinforceTopKoffPolicy{}/'.format(strftime("%H_%M", gmtime())))
plotter = recnn.utils.Plotter(reinforce.loss_layout, [['value', 'policy']],)

from recnn.nn import ChooseREINFORCE

def select_action_corr(state, action, K, writer, step, **kwargs):
    # note here I provide beta_net forward in the arguments
    return reinforce.nets['policy_net']._select_action_with_TopK_correction(state, beta_net.forward, action,
                                                                            K=K, writer=writer,
                                                                            step=step)

reinforce.nets['policy_net'].select_action = select_action_corr
reinforce.params['reinforce'] = ChooseREINFORCE(ChooseREINFORCE.reinforce_with_TopK_correction)
reinforce.params['K'] = 10

from tqdm.auto import tqdm
for epoch in range(n_epochs):
    for batch in tqdm(env.train_dataloader):
        loss = reinforce.update(batch)
        reinforce.step()
        if loss:
            plotter.log_losses(loss)
        if reinforce._step % plot_every == 0:
            print('step', reinforce._step)
            plotter.plot_loss()
        # if reinforce._step > 1000:
        #     pass
            # assert False

torch.save(policy_net.state_dict(), "models/topk_ddpg_policy.pt")

torch.save(value_net.state_dict(), "models/topk_ddpg_value.pt")