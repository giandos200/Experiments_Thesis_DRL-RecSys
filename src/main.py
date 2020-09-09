# from quantastica.qiskit_forest import ForestBackend
# from qiskit import QuantumRegister, ClassicalRegister
# from qiskit import QuantumCircuit, execute, Aer
#
# # Import ForestBackend:
# from quantastica.qiskit_forest import ForestBackend
#
# qc = QuantumCircuit()
#
# q = QuantumRegister(2, "q")
# c = ClassicalRegister(2, "c")
#
# qc.add_register(q)
# qc.add_register(c)
#
# qc.h(q[0])
# qc.cx(q[0], q[1])
#
# qc.measure(q[0], c[0])
# qc.measure(q[1], c[1])
#
#
# # Instead:
# #backend = Aer.get_backend("qasm_simulator")
#
# # Use:
# backend = ForestBackend.get_backend("qasm_simulator")
#
# # OR:
# # backend = ForestBackend.get_backend("statevector_simulator")
# # backend = ForestBackend.get_backend("Aspen-7-28Q-A")
# # backend = ForestBackend.get_backend("Aspen-7-28Q-A", as_qvm=True)
# # ...
#
# # To speed things up a little bit qiskit's optimization can be disabled
# # by setting optimization_level to 0 like following:
# # job = execute(qc, backend=backend, optimization_level=0)
# job = execute(qc, backend=backend)
# job_result = job.result()
#
# print(job_result.get_counts(qc))

import recnn

import recnn
import torch
import torch.nn as nn
from recnn.nn.update import ddpg
from tqdm.auto import tqdm

tqdm.pandas()

frame_size = 10
batch_size = 25
# embeddgings: https://drive.google.com/open?id=1EQ_zXBR3DKpmJR3jBgLvt-xoOvArGMsL
dirs = recnn.data.env.DataPath(
    base="",
    embeddings="ml20_pca128.pkl",
    ratings="ml-20m/ratings.csv",
    cache="frame_env.pkl", # cache will generate after you run
    use_cache=True
)
env = recnn.data.env.FrameEnv(dirs, frame_size, batch_size)

train = env.train_batch()
test = env.train_batch()
state, action, reward, next_state, done = recnn.data.get_base_batch(train, device=torch.device('cpu'))

print(state)
value_net  = recnn.nn.Critic(1290, 128, 256, 54e-2)
policy_net = recnn.nn.Actor(1290, 128, 256, 6e-1)

recommendation = policy_net(state)
value = value_net(state, recommendation)
print(recommendation)
print(value)

# ddpg = recnn.nn.DDPG(policy_net, value_net)
# print(ddpg.params)
# ddpg.params['gamma'] = 0.9
# ddpg.params['policy_step'] = 3
# ddpg.optimizers['policy_optimizer'] = torch.optim.Adam(ddpg.nets['policy_net'], your_lr)
# ddpg.writer = torch.utils.tensorboard.SummaryWriter('./runs')
# ddpg = ddpg.to(torch.device('cuda'))

# test function
def run_tests():
    batch = next(iter(env.test_dataloader))
    loss = ddpg.update(batch, learn=False)
    return loss

value_net  = recnn.nn.Critic(1290, 128, 256, 54e-2)
policy_net = recnn.nn.Actor(1290, 128, 256, 6e-1)

cuda = torch.device('cuda')
ddpg = recnn.nn.DDPG(policy_net, value_net)
ddpg = ddpg.to(cuda)
plotter = recnn.utils.Plotter(ddpg.loss_layout, [['value', 'policy']],)

from IPython.display import clear_output
import matplotlib.pyplot as plt
#% matplotlib inline

plot_every = 100
n_epochs = 2


def learn():
    for epoch in range(n_epochs):
        for batch in tqdm(env.train_dataloader):
            loss = ddpg.update(batch, learn=True)
            plotter.log_losses(loss)
            ddpg.step()
            if ddpg._step % plot_every == 0:
                clear_output(True)
                print('step', ddpg._step)
                test_loss = run_tests()
                plotter.log_losses(test_loss, test=True)
                plotter.plot_loss()
            if ddpg._step > 1000:
                return


learn()

value_net  = recnn.nn.Critic(1290, 128, 256, 54e-2)
policy_net = recnn.nn.Actor(1290, 128, 256, 6e-1)
# these are target networks that we need for ddpg algorigm to work
target_value_net = recnn.nn.Critic(1290, 128, 256)
target_policy_net = recnn.nn.Actor(1290, 128, 256)

target_policy_net.eval()
target_value_net.eval()
import torch_optimizer as optim

# soft update
recnn.utils.soft_update(value_net, target_value_net, soft_tau=1.0)
recnn.utils.soft_update(policy_net, target_policy_net, soft_tau=1.0)

# define optimizers
value_optimizer = optim.RAdam(value_net.parameters(),
                              lr=1e-5, weight_decay=1e-2)
policy_optimizer = optim.RAdam(policy_net.parameters(), lr=1e-5 , weight_decay=1e-2)

nets = {
    'value_net': value_net.to(cuda),
    'target_value_net': target_value_net.to(cuda),
    'policy_net': policy_net.to(cuda),
    'target_policy_net': target_policy_net.to(cuda),
}

optimizer = {
    'policy_optimizer': policy_optimizer,
    'value_optimizer':  value_optimizer
}

debug = {}
writer = recnn.utils.misc.DummyWriter()
step = 0
params = {
    'gamma'      : 0.1,
    'min_value'  : -10,
    'max_value'  : 10,
    'policy_step': 10,
    'soft_tau'   : 0.001,
}
batch = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done':done}
loss = recnn.nn.update.ddpg_update(batch, params, nets, optimizer, cuda, debug, writer, step=step)
print(loss)

cuda = torch.device('cuda')
loss = {
    'test': {'value': [], 'policy': [], 'step': []},
    'train': {'value': [], 'policy': [], 'step': []}
    }

plotter = recnn.utils.Plotter(loss, [['value', 'policy']],)
# test function
def run_tests():
    batch = next(iter(env.test_dataloader))
    loss = recnn.nn.ddpg_update(batch, params, nets, optimizer,
                       cuda, debug, writer, step=step, learn=False)
    return loss


import matplotlib.pyplot as plt
#% matplotlib inline

plot_every = 50
n_epochs = 2


def learn():
    step = 0
    for epoch in range(n_epochs):
        for batch in tqdm(env.train_dataloader):
            loss = recnn.nn.ddpg_update(batch, params,
                                        nets, optimizer, cuda, debug,
                                        writer, step=step)
            plotter.log_losses(loss)
            step += 1
            if step % plot_every == 0:
                clear_output(True)
                print('step', step)
                test_loss = run_tests()
                plotter.log_losses(test_loss, test=True)
                plotter.plot_loss()
            if step > 1000:
                return


learn()
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            print('plotting', n)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

plot_grad_flow(ddpg.nets['policy_net'].named_parameters())