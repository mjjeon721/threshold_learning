import torch
import numpy as np
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
from utils import *

class Policy_sac(nn.Module) :
    def __init__(self, state_dim, action_dim, hidden_size = 256, init_w = 3e-3, log_std_min = -20, log_std_max = 2):
        super(Policy_sac, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_dim)
        self.fc_log_std = nn.Linear(hidden_size, action_dim)

        self.relu = nn.ReLU()

        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.uniform_(-init_w, init_w)
        self.fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc_log_std.bias.data.uniform_(-init_w, init_w)

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc1.bias.data = fanin_init(self.fc1.bias.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc2.bias.data = fanin_init(self.fc2.bias.data.size())

    def forward(self, state):
        state = state.view(-1, self.state_dim)

        out = self.relu(self.fc1(state))
        out = self.relu(self.fc2(out))
        means = self.fc_mean(out)
        log_stds = torch.clamp(self.fc_log_std(out), self.log_std_min, self.log_std_max)

        return means, log_stds

    def get_actions(self, state):
        means, log_stds = self.forward(state)
        stds = log_stds.exp()

        normal = Normal(0,1)
        z = normal.sample()
        action = torch.sigmoid(means + stds * z)
        return action

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size = 256, init_w = 3e-3):
        super(SoftQNetwork,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc1.bias.data = fanin_init(self.fc1.bias.data.size())
        self.fc2.bias.data = fanin_init(self.fc2.bias.data.size())

    def forward(self, state, action):
        out = self.relu(self.fc1(torch.cat([state, action],1)))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)

        return out

class SoftVNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size = 256, init_w = 3e-3):
        super(SoftVNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc1.bias.data = fanin_init(self.fc1.bias.data.size())
        self.fc2.bias.data = fanin_init(self.fc2.bias.data.size())

        self.relu = nn.ReLU()

    def forward(self, state):
        out = self.relu(self.fc1(state))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)

        return out