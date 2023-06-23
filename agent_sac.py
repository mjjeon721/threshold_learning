import torch
import torch.optim as optim
from model_sac import *
from utils import *
import itertools

class SACAgent:
    def __init__(self,d_max, v_max, state_dim, action_dim, tau = 0.01, temp = 1, actor_lr = 1e-4,
                 critic_lr = 1e-3, value_lr = 1e-3, replay_buffer_size = 10000):

        self.QNetwork1 = SoftQNetwork(state_dim, action_dim)
        self.QNetwork2 = SoftQNetwork(state_dim, action_dim)

        self.Q_TargetNetwork1 = SoftQNetwork(state_dim, action_dim)
        self.Q_TargetNetwork2 = SoftQNetwork(state_dim, action_dim)

        hard_updates(self.Q_TargetNetwork1, self.QNetwork1)
        hard_updates(self.Q_TargetNetwork2, self.QNetwork2)

        self.Policy = Policy_sac(d_max, v_max, state_dim, action_dim)

        #self.VNetwork = SoftVNetwork(state_dim)
        #self.V_TargetNetwork = SoftVNetwork(state_dim)

        self.alpha = temp
        self.tau = tau

        #hard_updates(self.V_TargetNetwork, self.VNetwork)

        #value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.q_params = itertools.chain(self.QNetwork1.parameters(), self.QNetwork2.parameters())
        self.q_optim = optim.Adam(self.q_params, lr = critic_lr)

        #value_optim = optim.Adam(self.VNetwork.parameters(), lr = value_lr)
        #self.soft_q1_optim = optim.Adam(self.QNetwork1.parameters(), lr = critic_lr)
        #self.soft_q2_optim = optim.Adam(self.QNetwork2.parameters(), lr = critic_lr)
        self.policy_optim = optim.Adam(self.Policy.parameters(), lr = actor_lr)

        self.memory = Memory(max_size=replay_buffer_size)

    def get_action(self, state):
        actions = self.Policy.get_actions(state).detach().numpy()

        return actions

    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(np.vstack(states))
        actions = torch.FloatTensor(np.vstack(actions))
        rewards = torch.FloatTensor(np.vstack(rewards))
        next_states = torch.FloatTensor(np.vstack(next_states))
        dones = torch.Tensor(np.array(dones)).view(-1, 1)

        # Critic update
        next_actions, log_prob, epsilon, mean, log_std = self.Policy.evaluate(next_states)
        predicted_Q_value1 = self.Q_TargetNetwork1.forward(next_states, next_actions)
        predicted_Q_value2 = self.Q_TargetNetwork2.forward(next_states, next_actions)

        target_Q_value = rewards + (1 - dones) * (
                    torch.minimum(predicted_Q_value1, predicted_Q_value2) - self.alpha * log_prob)

        soft_q1_loss = self.soft_q_criterion1(predicted_Q_value1, target_Q_value)
        soft_q2_loss = self.soft_q_criterion2(predicted_Q_value2, target_Q_value)
        self.q_optim.zero_grad()
        soft_q_loss = soft_q1_loss + soft_q2_loss
        soft_q_loss.backward()
        #self.soft_q1_optim.zero_grad()
        #soft_q1_loss.backward()
        #self.soft_q1_optim.step()

        # Freeze Q-networks so that you don't waste computational effort
        # computing gradients for them during the policy learning step

        # Policy update
        new_actions, log_prob, epsilon, mean, log_std = self.Policy.evaluate(states)
        new_predicted_q_vals = torch.minimum(self.QNetwork1.forward(states, new_actions), self.QNetwork2.forward(states, new_actions))
        policy_loss = -(new_predicted_q_vals - self.alpha * log_prob).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        soft_updates(self.Q_TargetNetwork1, self.QNetwork1, self.tau)
        soft_updates(self.Q_TargetNetwork2, self.QNetwork2, self.tau)










