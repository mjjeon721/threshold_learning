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
    def __init__(self, state_dim, hidden = 256):
        super(Value, self).__init__()
        self.state_dim = state_dim
        '''
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
        '''
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

        self.relu = nn.ReLU()
        #self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        self.init_weights()

    def init_weights(self) :
        for param in self.parameters():
            param.data = fanin_init(param.data.size())

    def forward(self, state):
        '''
        state = state.view(-1, self.state_dim)
        u1 = self.relu(self.ut_fc0(state[:,1:]))
        u2 = self.relu(self.ut_fc1(u1))

        z1 = self.relu(self.z_fc0(state[:, [0]] * self.relu(self.zu_fc0(state[:, 1:]))) + self.y_fc0(
            state[:, [0]] * self.yu_fc0(state[:, 1:])) + self.u_fc0(state[:, 1:]))

        z2 = self.relu(self.z_fc1(z1 * self.relu(self.zu_fc1(u1))) + self.y_fc1(
            state[:, [0]] * self.yu_fc1(u1)) + self.u_fc1(u1))

        out = self.leakyrelu(self.z_fc2(z2 * self.relu(self.zu_fc2(u2))) + self.y_fc2(
            state[:, [0]] * self.yu_fc2(u2)) + self.u_fc2(u2))
        '''
        state = state.view(-1, self.state_dim)
        out = self.relu(self.fc1(state))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1 = 256, hidden2 = 256):
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
    def __init__(self, d_max, v_max, on_hrs, T, state_dim, action_dim, thresh_init, hidden1 = 256, hidden2 = 256):
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

        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)

        self.init_weight()

        self.d_plus = 0.5 * self.d_max * np.sort(np.random.rand(2,action_dim - 1), axis = 0)
        self.d_minus = 0.5 * self.d_max * (1 + np.sort(np.random.rand(2, action_dim - 1), axis = 0))


        self.v_plus = thresh_init[0]
        self.v_minus = thresh_init[1]

        self.d_th_plus = np.hstack([np.ones((self.K, len(self.off_hrs1))) * self.d_plus[1, :].reshape(self.K, -1),
                                    np.ones((self.K, len(self.on_hrs))) * self.d_plus[0, :].reshape(self.K, -1),
                                    np.ones((self.K, len(self.off_hrs2))) * self.d_plus[1, :].reshape(self.K, -1)])
        self.d_th_minus = np.hstack([np.ones((self.K, len(self.off_hrs1))) * self.d_minus[1, :].reshape(self.K, -1),
                                     np.ones((self.K, len(self.on_hrs))) * self.d_minus[0, :].reshape(self.K, -1),
                                     np.ones((self.K, len(self.off_hrs2))) * self.d_minus[1, :].reshape(self.K, -1)])
        self.v_th_plus = np.hstack([np.flip(self.off_hrs1) * self.v_max + self.v_plus,
                                    (self.T - self.on_hrs - 1) * self.v_max,
                                    (self.T - self.off_hrs2 - 1) * self.v_max])

        self.v_th_minus = np.hstack([np.zeros(len(self.off_hrs1)),
                                     np.ones(len(self.on_hrs)) * self.v_minus,
                                     np.zeros(len(self.off_hrs2))])
    def th_copy(self):
        self.d_th_plus = np.hstack([np.ones((self.K, len(self.off_hrs1))) * self.d_plus[1, :].reshape(self.K, -1),
                                    np.ones((self.K, len(self.on_hrs))) * self.d_plus[0, :].reshape(self.K, -1),
                                    np.ones((self.K, len(self.off_hrs2))) * self.d_plus[1, :].reshape(self.K, -1)])
        self.d_th_minus = np.hstack([np.ones((self.K, len(self.off_hrs1))) * self.d_minus[1, :].reshape(self.K, -1),
                                     np.ones((self.K, len(self.on_hrs))) * self.d_minus[0, :].reshape(self.K, -1),
                                     np.ones((self.K, len(self.off_hrs2))) * self.d_minus[1, :].reshape(self.K, -1)])
        self.v_th_plus = np.hstack([np.flip(self.off_hrs1) * self.v_max + self.v_plus,
                                    (self.T - self.on_hrs - 1) * self.v_max,
                                    (self.T - self.off_hrs2 - 1) * self.v_max])
        self.v_th_minus = np.hstack([np.zeros(len(self.off_hrs1)),
                                     np.ones(len(self.on_hrs)) * self.v_minus,
                                     np.zeros(len(self.off_hrs2))])
    def init_weight(self):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
    '''
    def zone_ind(self, state):
        state = state.reshape(-1, self.state_dim)
        if np.isin(state[-1], self.off_hrs1):
            d_cons = self.d_plus[0, :]
            v_cons = np.minimum(
                np.maximum(state[0] - self.v_plus - self.v_max * (self.off_hrs1[-1] - state[-1]), 0), self.v_max)
            d_prod = self.d_minus[0,:]
            v_prod = np.minimum(state[0], self.v_max)

        elif np.isin(state[-1], self.on_hrs):
            d_cons = self.d_plus[1,:]
            d_prod = self.d_minus[1,:]
            v_cons = np.minimum(
                np.maximum(state[0] - self.v_max * (self.T - state[-1]), 0), self.v_max)
            v_prod = np.minimum(
                np.maximum(state[0] - self.v_minus , 0), self.v_max)
        else :
            d_cons = self.d_plus[0, :]
            v_cons = np.minimum(
                np.maximum(state[0] - self.v_max * (self.T - state[-1]), 0), self.v_max)
            d_prod = self.d_minus[0, :]
            v_prod = np.minimum(state[0], self.v_max)
        return [d_cons, d_prod, v_cons, v_prod]
    def th_action(self, state):
        d_cons, d_prod, v_cons, v_prod = self.zone_ind(state)

        th_plus = np.sum(d_cons) + v_cons
        th_minus = np.sum(d_prod) + v_prod

        if state[1] < th_plus :
            return np.append(d_cons, v_cons)
        else :
            return np.append(d_prod, v_prod)
'''
    def nz_action(self, state):
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        out = self.softmax(out)
        a_nz = torch.zeros((len(state), self.action_dim))
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



    def action(self, state):
        state = state.reshape(-1, self.state_dim)
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
