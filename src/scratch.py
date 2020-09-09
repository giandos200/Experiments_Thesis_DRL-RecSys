import torch
import recnn

env = recnn.data.env.FrameEnv('ml20_pca128.pkl','ml-20m/ratings.csv')

value_net  = recnn.nn.Critic(1290, 128, 256, 54e-2)
policy_net = recnn.nn.Actor(1290, 128, 256, 6e-1)

cuda = torch.device('cuda')
ddpg = recnn.nn.DDPG(policy_net, value_net)
ddpg = ddpg.to(cuda)

for batch in env.train_dataloader:
    ddpg.update(batch, learn=True)
