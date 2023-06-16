import numpy as np
from scipy.stats import truncnorm

class Env:
    '''
        Environment model. Modeling customer. Generate next state and compute reward value for current state and action.
        Customer utility function : Quadratic utility function
        Renewable distribution : Truncated normal distribution.
    '''
    def __init__(self, util_param, renewable_param, TOU_param, NEM_param, gamma):
        self.a = util_param[0]
        self.b = util_param[1]

        self.g_mean = renewable_param[0]
        self.g_std = renewable_param[1]
        self.g_low = renewable_param[2]
        self.g_high = renewable_param[3]

        self.on_hrs = TOU_param[0]
        self.off_hrs = TOU_param[1]
        self.T = self.off_hrs[-1]

        self.pi_p = NEM_param[0]
        self.pi_m = NEM_param[1]

        self.gamma = gamma

    def get_next_g(self, size = 1) :
        low = (self.g_low - self.g_mean) / self.g_std
        high = (self.g_high - self.g_mean) / self.g_std
        return truncnorm.rvs(low, high, loc = self.g_mean, scale = self.g_std, size = size).item()

    def get_next_state(self, state, action):
        next_y = state[0] - action[-1]
        next_g = self.get_next_g()
        next_tau = (state[-1] + 1) % 10
        next_pi = np.array([self.pi_p[0], self.pi_m[0]]) if np.isin(next_tau, self.off_hrs) else np.array([self.pi_p[1], self.pi_m[1]])

        return np.hstack([[next_y, next_g], next_pi, [next_tau]])

    def get_reward(self, state, action):
        net_cons = np.sum(action) - state[1]
        reward = np.matmul(self.a, action[:-1]) - 0.5 * np.matmul(self.b, action[:-1] ** 2) - np.max(state[2:4] * net_cons)
        if state[-1] == (self.T - 1) :
            reward -= self.gamma * (state[0] - action[-1])
        return reward
