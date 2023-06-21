import torch
import torch.nn as nn
import numpy as np

def fanin_init(size):
    '''
        :param size: Neural network size
        :return: Fan-in initialized neural network
        :Explanation: fan-in initialization of the neural network.
    '''
    fanin = size[0]
    v = 1 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Value(nn.Module):
    def __init__(self, state_dim, hidden1 = 100, hidden2 = 64):
        super(Value, self).__init__()
        self.state_dim = state_dim
        self.ut_fc0 = nn.Linear(state_dim - 1, hidden1)
        self.ut_fc1 = nn.Linear(hidden1, hidden2)

        self.zu_fc0 = nn.Linear(state_dim - 1, 1)
        self.yu_fc0 = nn.Linear(state_dim - 1, 1)
        self.u_fc0 = nn.Linear(state_dim - 1, hidden1)
        self.z_fc0 = nn.Linear(1, hidden1)
        self.y_fc0 = nn.Linear(1, hidden1)

        self.zu_fc1 = nn.Linear(hidden1, hidden1)
        self.yu_fc1 = nn.Linear(hidden1, 1)
        self.u_fc1 = nn.Linear(hidden1, hidden2)
        self.z_fc1 = nn.Linear(hidden1, hidden2)
        self.y_fc1 = nn.Linear(1, hidden2)

        self.zu_fc2 = nn.Linear(hidden2, hidden2)
        self.yu_fc2 = nn.Linear(hidden2, 1)
        self.u_fc2 = nn.Linear(hidden2, 1)
        self.z_fc2 = nn.Linear(hidden2, 1)
        self.y_fc2 = nn.Linear(1, 1)

        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        self.init_weights()

    def init_weights(self) :
        for param in self.parameters():
            param.data = fanin_init(param.data.size())

    def forward(self, state):
        state = state.view(-1, self.state_dim)
        u1 = self.relu(self.ut_fc0(state[:,1:]))
        u2 = self.relu(self.ut_fc1(u1))

        z1 = self.relu(self.z_fc0(state[:, [0]] * self.relu(self.zu_fc0(state[:, 1:]))) + self.y_fc0(
            state[:, [0]] * self.yu_fc0(state[:, 1:])) + self.u_fc0(state[:, 1:]))

        z2 = self.relu(self.z_fc1(z1 * self.relu(self.zu_fc1(u1))) + self.y_fc1(
            state[:, [0]] * self.yu_fc1(u1)) + self.u_fc1(u1))

        out = self.leakyrelu(self.z_fc2(z2 * self.relu(self.zu_fc2(u2))) + self.y_fc2(
            state[:, [0]] * self.yu_fc2(u2)) + self.u_fc2(u2))

        return out

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1 = 100, hidden2 = 64):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1 + action_dim, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weight()

    def init_weight(self):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

    def forward(self, state, action):
        state = state.view(-1,self.state_dim)
        action = action.view(-1,self.action_dim)
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(torch.cat((out, action), dim = 1))
        out = self.relu(out)
        out = self.fc3(out)

        return out

class Policy(nn.Module):
    def __init__(self, d_max, v_max, on_hrs, T, state_dim, action_dim, log_std_min, log_std_max, init_w = 3e-3, hidden = 256):
        super(Policy, self).__init__()
        self.d_max = d_max
        self.v_max = v_max
        self.action_dim = action_dim
        self.K = self.action_dim - 1
        self.state_dim = state_dim
        self.on_hrs = on_hrs
        self.off_hrs1 = np.arange(0, on_hrs[0])
        self.off_hrs2 = np.arange(on_hrs[-1], T)
        self.T = T

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, 2 + action_dim)
        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.uniform_(-init_w, init_w)

        self.fc_log_std = nn.Linear(hidden, 2 + action_dim)
        self.fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc_log_std.bias.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.init_weight()

        self.d_plus = 0.5 * self.d_max * np.sort(np.random.rand(2,action_dim - 1), axis = 0)
        self.d_minus = 0.5 * self.d_max * (1 + np.sort(np.random.rand(2, action_dim - 1), axis = 0))


        self.d_th_plus = np.hstack([np.ones((self.K, len(self.off_hrs1))) * self.d_plus[1, :].reshape(self.K, -1),
                                    np.ones((self.K, len(self.on_hrs))) * self.d_plus[0, :].reshape(self.K, -1),
                                    np.ones((self.K, len(self.off_hrs2))) * self.d_plus[1, :].reshape(self.K, -1)])
        self.d_th_minus = np.hstack([np.ones((self.K, len(self.off_hrs1))) * self.d_minus[1, :].reshape(self.K, -1),
                                     np.ones((self.K, len(self.on_hrs))) * self.d_minus[0, :].reshape(self.K, -1),
                                     np.ones((self.K, len(self.off_hrs2))) * self.d_minus[1, :].reshape(self.K, -1)])
    def th_copy(self):
        self.d_th_plus = np.hstack([np.ones((self.K, len(self.off_hrs1))) * self.d_plus[1, :].reshape(self.K, -1),
                                    np.ones((self.K, len(self.on_hrs))) * self.d_plus[0, :].reshape(self.K, -1),
                                    np.ones((self.K, len(self.off_hrs2))) * self.d_plus[1, :].reshape(self.K, -1)])
        self.d_th_minus = np.hstack([np.ones((self.K, len(self.off_hrs1))) * self.d_minus[1, :].reshape(self.K, -1),
                                     np.ones((self.K, len(self.on_hrs))) * self.d_minus[0, :].reshape(self.K, -1),
                                     np.ones((self.K, len(self.off_hrs2))) * self.d_minus[1, :].reshape(self.K, -1)])

    def forward(self, state):
        state = state.reshape(-1, self.state_dim)
        x = torch.FloatTensor(state)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        means = self.fc_mean(x)
        log_stds = self.fc_log_std(x)
        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)

        return means, log_stds

    def action(self, state):
        means, log_stds = self.forward(state)
        stds = log_stds.exp()

        self.sigmoid(torch.Normal(means[:2], stds[:2])) * state[0]

    def action(self, state):
        state = state.reshape(-1, self.state_dim)
        x = torch.FloatTensor(state)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)


        d_cons = np.transpose(self.d_th_plus[:,state[:,-1].astype(int)])
        d_prod = np.transpose(self.d_th_minus[:, state[:, -1].astype(int)])
        v_cons = np.minimum(np.maximum(state[:,0] - self.v_th_plus[state[:,-1].astype(int)], 0), np.minimum(state[:,0],self.v_max)).reshape(-1,1)
        v_prod = np.minimum(np.maximum(state[:,0] - self.v_th_minus[state[:, -1].astype(int)], 0), np.minimum(state[:,0],self.v_max)).reshape(-1, 1)

        th_plus = np.sum(d_cons, axis = 1, keepdims = True) + v_cons
        th_minus = np.sum(d_prod, axis = 1, keepdims = True) + v_prod

        cons_mask = state[:,[1]] < th_plus
        prod_mask = state[:,[1]] > th_minus
        nz_mask = ~cons_mask * ~prod_mask
        a_nz = np.zeros((len(state),self.action_dim))

        x = torch.FloatTensor(state)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        for j in range(self.action_dim):
            ix = torch.argmax(out, dim = 1)
            v_max_ix = (ix == 2).detach().numpy()
            out_i = out[np.arange(0, len(state)), ix]
            a_nz[np.arange(0, len(state)), ix] =  np.minimum(np.minimum((x[:, 1] * out_i).detach().numpy(), state[:, 0]),
                       np.minimum(state[:,0],self.v_max)) * v_max_ix + (
                np.minimum((x[:, 1] * out_i).detach().numpy(), self.d_max)) * ~v_max_ix
            out[np.arange(0, len(state)), ix] = 0
            x[:,1] -= a_nz[np.arange(0, len(state)), ix]
            if j < self.action_dim - 1:
                out = out / torch.sum(out, 1, keepdim = True)

        #a_nz = (x[:,[1]] * out).detach().numpy()

        a_cons = np.hstack([d_cons, v_cons])
        a_prod = np.hstack([d_prod, v_prod])

        action = a_cons * cons_mask + a_prod * prod_mask + a_nz * nz_mask

        return action

    def nz_action(self, state):
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        out = self.softmax(out)
        a_nz = torch.ones((len(state), self.action_dim))
        a_nz[:,:-1] *= self.d_max
        a_nz[:,-1] *= torch.minimum(state[:,0], torch.Tensor(np.array([self.v_max])))
        out = out * a_nz
        return out
        '''
        for j in range(self.action_dim):
            ix = torch.argmax(out, dim=1)
            v_max_ix = (ix == 2)#.detach().numpy()
            out_i = out[torch.arange(0, len(state)), ix]
            a_nz[torch.arange(0, len(state)), ix] = torch.minimum(torch.minimum((state[:, 1] * out_i), state[:, 0]),
                                                            torch.Tensor(np.array([self.v_max]))) * v_max_ix + (
                                                     torch.minimum((state[:, 1] * out_i),
                                                                torch.Tensor(np.array([self.d_max]))) * ~v_max_ix)
            out[torch.arange(0, len(state)), ix] = 0
            state[:, 1] -= a_nz[torch.arange(0, len(state)), ix]
            if j < self.action_dim - 1:
                out = out / torch.sum(out, 1, keepdim=True)
        return a_nz
        '''




