from model import *
from utils import *
import torch
import numpy as np
import torch.nn as nn
import itertools
import numpy.random as npr
import torch.optim as optim

class Agent():
    '''
        Learning Agent
    '''
    def __init__(self, d_max, v_max, state_dim, action_dim, TOU_info, env, actor_lr = 5e-4, \
                 critic_lr = 1e-3, value_lr = 1e-3, tau = 0.001, max_memory_size = 10000):
        self.d_max = d_max
        self.v_max = v_max

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.tau = tau

        self.T = TOU_info[0]
        self.on_hrs = TOU_info[1]
        self.off_hrs = np.setdiff1d(np.arange(0, self.T), self.on_hrs)

        v_plus_init = (self.T - self.on_hrs[0]) * v_max * npr.rand()
        v_minus_init = (self.T - self.on_hrs[-1] - 1) * v_max * npr.rand()

        self.actor = Policy(d_max, v_max, self.on_hrs, self.T, self.state_dim, self.action_dim, [v_plus_init, v_minus_init])
        '''
        self.critic1 = Critic(self.state_dim, self.action_dim)
        self.critic2 = Critic(self.state_dim, self.action_dim)
        self.critic_target1 = Critic(self.state_dim, self.action_dim)
        self.critic_target2 = Critic(self.state_dim, self.action_dim)
        '''
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.value = Value(state_dim)
        #self.value_target = Value(state_dim)

        self.env = env

        self.actor_lr = actor_lr

        self.d_update_count = np.zeros(4)
        self.v_update_count = np.zeros(2)
        self.nz_update_count = 0
        hard_updates(self.critic_target, self.critic)
        #hard_updates(self.critic_target1, self.critic1)
        #hard_updates(self.critic_target2, self.critic2)
        #hard_updates(self.value_target, self.value)

        #self.q_params = itertools.chain(self.critic1.parameters(), self.critic2.parameters())
        self.q_params = self.critic.parameters()

        self.thresh_grad_history = np.zeros((4, action_dim-1))
        self.v_th_grad_history = np.zeros(2)
        self.grad_history = []
        for param in self.actor.parameters():
            self.grad_history.append(torch.zeros(param.size()))
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.value_criterion = nn.MSELoss()
        self.critic_optim = optim.Adam(self.q_params, critic_lr)
        self.value_optim = optim.Adam(self.value.parameters(), value_lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), actor_lr)

    def get_action(self, state):
        action = self.actor.action(state).reshape(-1)
        return action

    def d_th_update(self, state, action):
        '''
        Updating consumption threshold values
        :param state: Current state information
        :param action: Current action (for v information)
        :param flag: Indicator of net_cons / net_prod region
        '''
        flag = state[1] < np.sum(action)
        v = action[-1]
        # Update d_plus if flag = True
        # Update d_minus if flag = False
        if flag and np.isin(state[-1], self.off_hrs) :
            a_n = 1e-2 / (1 + self.d_update_count[0]) ** (0.2)
            c_n = 1e-2 / (1 + self.d_update_count[0]) ** (0.1)
            vec = c_n * (npr.binomial(1, 0.5, self.action_dim - 1) * 2 - 1) * (npr.rand(self.action_dim - 1) + 0.5)
            d1 = self.actor.d_plus[1,:] + vec
            d2 = self.actor.d_plus[1,:] - vec
            r1 = self.env.get_reward(state, np.append(d1, v))
            r2 = self.env.get_reward(state, np.append(d2, v))
            grad_est = (r1 - r2) / vec / 2
            self.thresh_grad_history[0, :] = self.thresh_grad_history[0, :] * 0.99 + 0.01 * grad_est
            self.actor.d_plus[1,:] += a_n * self.thresh_grad_history[0, :]
            self.d_update_count[0] += 1
        elif flag and np.isin(state[-1], self.on_hrs) :
            a_n = 1e-2 / (1 + self.d_update_count[1]) ** (0.2)
            c_n = 1e-2 / (1 + self.d_update_count[1]) ** (0.1)
            vec = c_n * (npr.binomial(1, 0.5, self.action_dim - 1) * 2 - 1) * (npr.rand(self.action_dim - 1) + 0.5)
            d1 = self.actor.d_plus[0, :] + vec
            d2 = self.actor.d_plus[0, :] - vec
            r1 = self.env.get_reward(state, np.append(d1, v))
            r2 = self.env.get_reward(state, np.append(d2, v))
            grad_est = (r1 - r2) / vec / 2
            self.thresh_grad_history[1, :] = self.thresh_grad_history[1, :] * 0.99 + 0.01 * grad_est
            self.actor.d_plus[0, :] += a_n * self.thresh_grad_history[1, :]
            self.d_update_count[1] += 1
        elif ~flag and np.isin(state[-1], self.off_hrs) :
            a_n = 1e-2 / (1 + self.d_update_count[2]) ** (0.2)
            c_n = 1e-2 / (1 + self.d_update_count[2]) ** (0.1)
            vec = c_n * (npr.binomial(1, 0.5, self.action_dim - 1) * 2 - 1) * (npr.rand(self.action_dim - 1) + 0.5)
            d1 = self.actor.d_minus[1, :] + vec
            d2 = self.actor.d_minus[1, :] - vec
            r1 = self.env.get_reward(state, np.append(d1, v))
            r2 = self.env.get_reward(state, np.append(d2, v))
            grad_est = (r1 - r2) / vec / 2
            self.thresh_grad_history[2, :] = self.thresh_grad_history[2, :] * 0.99 + 0.01 * grad_est
            self.actor.d_minus[1, :] += a_n * self.thresh_grad_history[2, :]
            self.d_update_count[2] += 1
        else :
            a_n = 1e-2 / (1 + self.d_update_count[3]) ** (0.2)
            c_n = 1e-2 / (1 + self.d_update_count[3]) ** (0.1)
            vec = c_n * (npr.binomial(1, 0.5, self.action_dim - 1) * 2 - 1) * (npr.rand(self.action_dim - 1) + 0.5)
            d1 = self.actor.d_minus[0, :] + vec
            d2 = self.actor.d_minus[0, :] - vec
            r1 = self.env.get_reward(state, np.append(d1, v))
            r2 = self.env.get_reward(state, np.append(d2, v))
            grad_est = (r1 - r2) / vec / 2
            self.thresh_grad_history[3, :] = self.thresh_grad_history[3, :] * 0.99 + 0.01 * grad_est
            self.actor.d_minus[0, :] += a_n * self.thresh_grad_history[3, :]
            self.d_update_count[3] += 1
        self.actor.th_copy()

    def update(self, batch_size) :
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(np.vstack(states))
        actions = torch.FloatTensor(np.vstack(actions))
        rewards = torch.FloatTensor(np.vstack(rewards))
        next_states = np.vstack(next_states)
        dones = torch.FloatTensor(np.vstack(dones))
        states_np = states.detach().clone().numpy()

        # Critic (Q-Value) Update
        #Qvals1 = self.critic1.forward(states, actions)
        #Qvals2 = self.critic2.forward(states, actions)
        Qvals = self.critic.forward(states, actions)
        next_actions = torch.FloatTensor(self.actor.action(next_states))
        next_states = torch.FloatTensor(next_states)
        #Qtarget1 = self.critic_target1.forward(next_states, next_actions)
        #Qtarget2 = self.critic_target2.forward(next_states, next_actions)
        Qtarget = self.critic_target.forward(next_states, next_actions)

        y = -rewards + (1 - dones) * Qtarget#torch.minimum(Qtarget1, Qtarget2)
        #critic_loss1 = self.critic_criterion(Qvals1, y)
        #critic_loss2 = self.critic_criterion(Qvals2, y)
        self.critic_optim.zero_grad()
        critic_loss = self.critic_criterion(Qvals, y)#critic_loss2 + critic_loss1
        critic_loss.backward()
        self.critic_optim.step()
        soft_updates(self.critic_target, self.critic, self.tau)
        #soft_updates(self.critic_target1, self.critic1, self.tau)
        #soft_updates(self.critic_target2, self.critic2, self.tau)

        # Value Update
        Vvals = self.value.forward(states)
        current_policy_actions = self.actor.action(states_np)
        #y1 = self.critic_target1.forward(states, torch.Tensor(current_policy_actions))
        #y2 = self.critic_target2.forward(states, torch.Tensor(current_policy_actions))
        y = self.critic_target.forward(states, torch.Tensor(current_policy_actions))#torch.minimum(y1, y2)
        #current_policy_rewards = - torch.Tensor(self.env.get_reward(states_np, current_policy_actions)).view(-1,1)
        #y = current_policy_rewards + (1 - dones) * self.Q(next_states)
        value_loss = self.value_criterion(Vvals, y)
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        #soft_updates(self.value_target, self.value, self.tau)

        action_sum = torch.sum(actions, dim = 1)
        DERs = states[:,1]
        NETs = action_sum - DERs
        NZ = abs(NETs) <= 1e-6
        #NC = (NETs > 1e-6) * np.isin(states[:,-1].detach().numpy(), np.arange(0, self.on_hrs[0]))
        NC = states[:,-1].detach().numpy() == self.on_hrs[0]
        #NC = NC * (actions[:,-1] < self.v_max) * (actions[:,-1] > 0)
        #NC = NC.bool()
        #NP = (NETs < -1e-6) * np.isin(states[:,-1].detach().numpy(), self.on_hrs)
        NP = states[:,-1].detach().numpy() == (self.on_hrs[-1] + 1)
        #NP = NP * (actions[:,-1] < self.d_max) * (actions[:,-1] > 0)
        #NP = NP.bool()

        # Net-zero zone update
        states_NZ = states[NZ,:].detach().clone()
        a_n = self.actor_lr / (1 + self.nz_update_count) ** 0.05
        c_n = 1e-4 / (1 + self.nz_update_count) ** 0.02
        vecs = []
        for param in self.actor.parameters():
            vec = c_n * (torch.Tensor(npr.binomial(1, 0.5, param.size())) * 2 -1) * (torch.rand(param.size()) + 0.5)
            param.data.add_(vec)
            vecs.append(vec)
        nz_plus = self.actor.nz_action(states_NZ)
        states_NZ = states[NZ,:].detach().clone()
        #Q_plus1 = self.critic1.forward(states_NZ, nz_plus).view(-1)
        #Q_plus2 = self.critic2.forward(states_NZ, nz_plus).view(-1)
        #Q_plus = torch.minimum(Q_plus1, Q_plus2).mean()
        Q_plus = self.critic.forward(states_NZ, nz_plus).mean()
        i = 0
        for param in self.actor.parameters():
            vec = vecs[i]
            param.data.add_(-2 * vec)
            i += 1
        nz_minus = self.actor.nz_action(states_NZ)
        states_NZ = states[NZ,:].detach().clone()
        #Q_minus1 = self.critic1.forward(states_NZ, nz_minus).view(-1)
        #Q_minus2 = self.critic2.forward(states_NZ, nz_minus).view(-1)
        #Q_minus = torch.minimum(Q_minus1, Q_minus2).mean()
        Q_minus = self.critic.forward(states_NZ, nz_minus).mean()
        i = 0
        for param in self.actor.parameters():
            vec = vecs[i]
            grad_est = (Q_plus - Q_minus) / vec / 2
            self.grad_history[i] = 0.9 * self.grad_history[i] + 0.1 * grad_est
            param.data.add_(vec + a_n * grad_est)#self.grad_history[i])
            i += 1

        self.nz_update_count += 1

        # Net Consumption charging threshold update
        if sum(NC) > 0 :
            states_NC = states[NC,1:].detach().clone()
            a_n = 0.01 / (1 + self.v_update_count[0]) ** 0.05
            c_n = self.actor_lr / (1 + self.v_update_count[0]) ** 0.02
            vec = c_n * (npr.binomial(1, 0.5, 1) * 2 - 1) * (npr.rand(1) + 0.5)
            tau_plus = torch.FloatTensor(self.actor.v_plus + vec)
            tau_minus = torch.FloatTensor(self.actor.v_plus - vec)
            x_plus = torch.hstack((torch.ones((sum(NC),1)) * tau_plus, states_NC))
            x_minus = torch.hstack((torch.ones((sum(NC),1)) * tau_minus, states_NC))
            V1 = torch.mean(self.value(x_plus))
            V2 = torch.mean(self.value(x_minus))
            grad_est = (V1 - V2).detach().numpy() / vec / 2 - self.env.pi_p[0]
            self.v_th_grad_history[0] = self.v_th_grad_history[0] * 0.9 + grad_est * 0.1
            self.actor.v_plus -= a_n * self.v_th_grad_history[0]
            self.actor.v_plus = np.maximum(self.actor.v_plus, 0)
            self.v_update_count[0] += 1
            self.actor.th_copy()

        if sum(NP) > 0 :
            states_NP = states[NP,1:].detach().clone()
            a_n = 0.01 / (1 + self.v_update_count[1]) ** 0.05
            c_n = self.actor_lr / (1 + self.v_update_count[1]) ** 0.02
            vec = c_n * (npr.binomial(1, 0.5, 1) * 2 - 1) * (npr.rand(1) + 0.5)
            delta_plus = torch.FloatTensor(self.actor.v_minus + vec)
            delta_minus = torch.FloatTensor(self.actor.v_minus - vec)
            x_plus = torch.hstack((torch.ones((sum(NP),1)) * delta_plus, states_NP))
            x_minus = torch.hstack((torch.ones((sum(NP),1)) * delta_minus, states_NP))
            V1 = torch.mean(self.value(x_plus))
            V2 = torch.mean(self.value(x_minus))
            grad_est = (V1 - V2).detach().numpy() / vec / 2 - self.env.pi_m[1]
            self.v_th_grad_history[1] = self.v_th_grad_history[1] * 0.9 + grad_est * 0.1
            self.actor.v_minus -= a_n * self.v_th_grad_history[1]
            self.actor.v_minus = np.maximum(self.actor.v_minus, 0)
            self.v_update_count[1] += 1
            self.actor.th_copy()

        '''
        if sum(NC) > 0 :
            a_n = self.actor_lr / (1 + self.v_update_count[0]) ** (0.05)
            c_n = self.actor_lr / (1 + self.v_update_count[0]) ** (0.02)
            self.on_hrs[0] 
            states_NC = states[NC,:]
            actions_NC = actions[NC,:]
            vec = c_n * (npr.binomial(1, 0.5, 1) * 2 - 1) * (npr.rand(1) + 0.5)
            actions_NC[:,-1] += vec
            Q1 = torch.mean(self.critic_target.forward(states_NC, actions_NC))
            actions_NC[:,-1] -= 2 * vec
            Q2 = torch.mean(self.critic_target.forward(states_NC, actions_NC))
            grad_est = (Q1 - Q2).detach().numpy() / vec / 2
            self.v_th_grad_history[0] = self.v_th_grad_history[0] * 0.9 + 0.1 * grad_est
            self.actor.v_plus += a_n * grad_est#self.v_th_grad_history[0]
            self.v_update_count[0] += 1
            self.actor.th_copy()
        # Net Production zone charging threshold update
        if sum(NP) > 0 :
            a_n = self.actor_lr / (1 + self.v_update_count[1]) ** (0.05)
            c_n = self.actor_lr / (1 + self.v_update_count[1]) ** (0.02)
            states_NP = states[NP, :]
            actions_NP = actions[NP, :]
            vec = c_n * (npr.binomial(1, 0.5, 1) * 2 - 1) * (npr.rand(1) + 0.5)
            actions_NP[:, -1] += vec
            Q1 = torch.mean(self.critic_target.forward(states_NP, actions_NP))
            actions_NP[:, -1] -= 2 * vec
            Q2 = torch.mean(self.critic_target.forward(states_NP, actions_NP))
            grad_est = (Q1 - Q2).detach().numpy() / vec / 2
            self.v_th_grad_history[1] = self.v_th_grad_history[1] * 0.9 + 0.1 * grad_est
            self.actor.v_minus += a_n * grad_est#self.v_th_grad_history[1]
            self.v_update_count[1] += 1
            self.actor.th_copy()
        '''
