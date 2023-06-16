import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable

# Weight initialization function
def fanin_init(size, fanin = None) :
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

# Critic network class
class Critic_d(nn.Module) :
    def __init__(self, state_dim, action_dim, hidden1 = 128, hidden2 = 64, init_w = 3e-4):
        super(Critic_d, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1 + action_dim, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
        #Batch Normalization
        self.BN0 = nn.BatchNorm1d(state_dim)
        self.BN1 = nn.BatchNorm1d(hidden1)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = state
        a = action
        # Batch normalization
        x = self.BN0(x)
        out = self.fc1(x)
        out = self.relu(out)
        # Batch normalization
        out = self.BN1(out)
        out = self.fc2(torch.cat((out,a), dim = 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Actor network class
class Actor_d(nn.Module):
    def __init__(self, state_dim, action_dim, d_max, v_max, hidden1 = 128, hidden2 = 64, init_w = 3e-3):
        super(Actor_d, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weights(init_w)
        self.d_max = d_max
        self.v_max = v_max


    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        state = state.view(-1, self.state_dim)
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        out = out * torch.hstack((torch.ones(len(state), self.action_dim - 1) * self.d_max,
                                  torch.ones(len(state), 1) * torch.minimum(state[:, [0]],
                                                                            torch.Tensor(np.array([self.v_max])))))

        return out
