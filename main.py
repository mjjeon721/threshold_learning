import numpy as np
import copy
import agent
from agent import *
from environment import Env
from scipy.stats import truncnorm
import time
import matplotlib.pyplot as plt

# Utility parameter
a = np.array([2, 1.2])
b = np.array([1,1])

# NEM parameter : Off-peak / On-peak NEM parameters
pi_p = np.array([0.37, 0.58])
pi_m = np.array([0.07, 0.12])

# NEM optimal consumption
opt_d_plus = (a - pi_p.reshape(-1,1)) / b
opt_d_minus = (a - pi_m.reshape(-1,1)) / b

# Consumption parameter
d_max = 3
K = len(a)

# EV charging parameter
#v_max : Maximum charging rate
#T : Charging session length
#gamma : Penalty
v_max = 2
T = 10
gamma = 1

# TOU parameter
on_hrs = np.array([3, 4, 5, 6])
off_hrs = np.setdiff1d(np.arange(0,T), on_hrs)

# Renewable distribution parameter
g_mean = 3
g_std = 2
g_min = 0
g_max = 10

# State and action dimension
# x_t = [y_t, g_t, pi_t, tau_t]
state_dim = 5
action_dim = K + 1

# Environment
env = Env([a,b], [g_mean, g_std, g_min, g_max], [on_hrs, off_hrs], [pi_p, pi_m], gamma)
learning_agent = Agent(d_max, v_max, state_dim, action_dim, [T, on_hrs], env)

episode_len = T
num_epi = 2000

batch_size = 100

trained_reward = []
avg_trained_reward = []

g_0_samples = truncnorm.rvs(-g_mean / g_std, (g_max - g_mean) / g_std, size = num_epi) * g_std + g_mean

interaction = 0
d_off_plus_history = []
d_on_plus_history = []
d_off_minus_history = []
d_on_minus_history = []
tic = time.perf_counter()
for epi in range(num_epi) :
    epi_reward = 0
    state = np.array([0, g_0_samples[epi], pi_p[0], pi_m[0], 0])
    for step in range(episode_len) :
        action = learning_agent.get_action(state)

        next_state = env.get_next_state(state, action)
        reward = env.get_reward(state, action)
        epi_reward += reward

        learning_agent.memory.push(state, action, reward, next_state)

        # Updating consumption thresholds
        if np.abs((np.sum(action) - state[1])) > 1e-6 :
            learning_agent.d_th_update(state,action)

        # Updating Q network and EV charging policy
        '''
        if interaction > 1000 and (interaction % 20 == 1) :
            if np.abs((np.sum(action) - state[1])) > 1e-6 :
                for grad_update in range(20):
                    learning_agent.Q_update(batch_size)
                    learning_agent.v_th_update(state, action)
            else :
                for grad_update in range(20):
                    learning_agent.Q_update(batch_size)
                    learning_agent.nz_update(state)
        '''
        if interaction % 50 == 1 :
            d_off_minus_history.append(copy.copy(learning_agent.actor.d_minus[1,:]))
            d_on_minus_history.append(copy.copy(learning_agent.actor.d_minus[0, :]))
            d_off_plus_history.append(copy.copy(learning_agent.actor.d_plus[1,:]))
            d_on_plus_history.append(copy.copy(learning_agent.actor.d_plus[0,:]))

        interaction += 1
        state = next_state

    trained_reward.append(epi_reward)
    avg_trained_reward.append(np.mean(trained_reward[-100:]))

    if epi % 50 == 49 :
        toc = time.perf_counter()
        print('{0}th Episode, {1:.4f} (s) time elapsed, average reward : {2:.4f}'.format(epi, toc - tic, avg_trained_reward[-1]))
        tic = time.perf_counter()

d_off_minus_history = np.vstack(d_off_minus_history)
d_on_minus_history = np.vstack(d_on_minus_history)
d_off_plus_history = np.vstack(d_off_plus_history)
d_on_plus_history = np.vstack(d_on_plus_history)
'''
plt.plot(np.arange(0,interaction, 50), d_off_minus_history[:,0] )
plt.plot(np.arange(0,interaction, 50), np.ones(int(interaction / 50)) * opt_d_minus[0,0] )
plt.grid()
plt.show()

plt.plot(np.arange(0,interaction, 50), d_off_minus_history[:,1] )
plt.plot(np.arange(0,interaction, 50), np.ones(int(interaction / 50)) * opt_d_minus[0,1] )
plt.grid()
plt.show()

plt.plot(np.arange(0,interaction, 50), d_on_minus_history[:,0] )
plt.plot(np.arange(0,interaction, 50), np.ones(int(interaction / 50)) * opt_d_minus[1,0] )
plt.grid()
plt.show()

plt.plot(np.arange(0,interaction, 50), d_on_minus_history[:,1] )
plt.plot(np.arange(0,interaction, 50), np.ones(int(interaction / 50)) * opt_d_minus[1,1] )
plt.grid()
plt.show()

plt.plot(np.arange(0,interaction, 50), d_off_plus_history[:,0] )
plt.plot(np.arange(0,interaction, 50), np.ones(int(interaction / 50)) * opt_d_plus[0,0] )
plt.grid()
plt.show()

plt.plot(np.arange(0,interaction, 50), d_off_plus_history[:,1] )
plt.plot(np.arange(0,interaction, 50), np.ones(int(interaction / 50)) * opt_d_plus[0,1] )
plt.grid()
plt.show()

plt.plot(np.arange(0,interaction, 50), d_on_plus_history[:,0] )
plt.plot(np.arange(0,interaction, 50), np.ones(int(interaction / 50)) * opt_d_plus[1,0] )
plt.grid()
plt.show()

plt.plot(np.arange(0,interaction, 50), d_on_plus_history[:,1] )
plt.plot(np.arange(0,interaction, 50), np.ones(int(interaction / 50)) * opt_d_plus[1,1] )
plt.grid()
plt.show()
'''