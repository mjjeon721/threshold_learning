import numpy as np
import copy

import torch

import agent
from agent import *
from environment import Env
from scipy.stats import truncnorm
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from agent_ddpg import *


# Utility parameter
a = np.array([2, 1.2])
aa = np.array([a[0] - a[1], a[1] - a[0]])
b = np.array([1,1])

# NEM parameter : Off-peak / On-peak NEM parameters
pi_p = np.array([0.37, 0.58])
pi_m = np.array([0.07, 0.12])

# NEM optimal consumption
opt_d_plus = (a - pi_p.reshape(-1,1)) / b
opt_d_minus = (a - pi_m.reshape(-1,1)) / b

# EV charging parameter
# v_max : Maximum charging rate
# T : Charging session length
# gamma : Penalty
v_max = 4
T = 10
gamma = 1

# TOU parameter
on_hrs = np.array([3, 4, 5, 6])
off_hrs = np.setdiff1d(np.arange(0,T), on_hrs)

# Optimal procrastination threshold
delta = 2.409896891764526
tau = 17.379661111107737

opt_delta = np.concatenate([np.zeros(on_hrs[0]), np.ones(len(on_hrs)) * delta , np.zeros(T - on_hrs[-1] - 1)])
opt_tau = np.concatenate([np.flip(np.arange(0, on_hrs[0])) * v_max + tau,np.flip(np.arange(0, T - on_hrs[0])) * v_max ])

thetas = loadmat('theta.mat')
thetas = thetas['theta_data']
thetas = np.flip(thetas, 1)
thetas = thetas[:,1:]

# Consumption parameter
d_max = 3
K = len(a)

# Renewable distribution parameter
g_mean = 4
g_std = 3
g_min = 0
g_max = 10

# State and action dimension
# x_t = [y_t, g_t, pi_t, tau_t]
state_dim = 5
action_dim = K + 1

# Environment
env = Env([a,b], [g_mean, g_std, g_min, g_max], [on_hrs, off_hrs], [pi_p, pi_m], gamma, state_dim)
learning_agent = Agent(d_max, v_max, state_dim, action_dim, [T, on_hrs], env)
DDPG_agent = DDPGAgent(state_dim, action_dim, d_max, v_max)

episode_len = T
num_epi = 5000

batch_size = 100

trained_reward = []
avg_trained_reward = []

opt_return = []
avg_opt_return = []

DDPG_return = []
avg_DDPG_return = []

low = -g_mean / g_std
high = (g_max - g_mean) / g_std

#g_0_samples = truncnorm.rvs(-g_mean / g_std, (g_max - g_mean) / g_std, size = num_epi) * g_std + g_mean

g_0_samples = truncnorm.rvs(low, high, loc = g_mean, scale = g_std, size = num_epi)
y_0_samples = T * v_max * np.random.rand(num_epi)
#plt.hist(g_0_samples)
#plt.show()

interaction = 0
d_off_plus_history = []
d_on_plus_history = []
d_off_minus_history = []
d_on_minus_history = []

v_plus_history = []
v_minus_history = []

opt_th_return = []
avg_opt_th_return = []

tic = time.perf_counter()
for epi in range(num_epi) :
    epi_reward = 0
    opt_epi_reward = 0
    epi_return_opt_th = 0
    DDPG_epi_return = 0
    state = np.array([y_0_samples[epi], g_0_samples[epi], pi_p[0], pi_m[0], 0])
    state_opt_th = np.array([y_0_samples[epi], g_0_samples[epi], pi_p[0], pi_m[0], 0])
    opt_state = np.array([y_0_samples[epi], g_0_samples[epi], pi_p[0], pi_m[0], 0])
    DDPG_state = np.array([y_0_samples[epi], g_0_samples[epi], pi_p[0], pi_m[0], 0])
    for step in range(episode_len) :
        if interaction < 1000 :
            DDPG_action = DDPG_agent.random_action(DDPG_state)
        else :
            explr_noise_std = 0.4
            DDPG_action = np.clip(DDPG_agent.get_action(DDPG_state) + explr_noise_std * np.random.randn(action_dim),
                                  np.zeros(action_dim),
                                  np.append(d_max * np.ones(action_dim - 1), np.minimum(DDPG_state[0], v_max))).reshape(-1)
        action = learning_agent.get_action(state)

        next_state = env.get_next_state(state, action)
        DDPG_next_state = env.get_next_state(DDPG_state, DDPG_action)
        DDPG_next_state[1] = next_state[1]
        reward = env.get_reward(state, action)
        DDPG_reward = env.get_reward(DDPG_state, DDPG_action)
        epi_reward += reward
        DDPG_epi_return += DDPG_reward

        done = True if step == T-1 else False

        learning_agent.memory.push(state, action, reward, next_state, done)
        DDPG_agent.memory.push(DDPG_state, DDPG_action, DDPG_reward, DDPG_next_state, done)

        v_prod_opt_th = np.minimum(np.maximum(state_opt_th[0] - opt_delta[int(state_opt_th[-1])], 0), v_max)
        v_cons_opt_th = np.minimum(np.maximum(state_opt_th[0] - opt_tau[int(state_opt_th[-1])], 0), v_max)
        d_prod_opt_th = learning_agent.actor.d_th_minus[:, int(state[-1])]
        d_cons_opt_th = learning_agent.actor.d_th_plus[:, int(state[-1])]
        opt_th_cons = np.sum(d_cons_opt_th) + v_cons_opt_th
        opt_th_prod = np.sum(d_prod_opt_th) + v_prod_opt_th
        if state_opt_th[1] < opt_th_cons :
            action_opt_th = np.append(d_cons_opt_th, v_cons_opt_th)
        elif state_opt_th[1] > opt_th_prod :
            action_opt_th = np.append(d_prod_opt_th, v_prod_opt_th)
        else :
            state_tmp = torch.FloatTensor(state_opt_th).view(1,-1)
            action_opt_th = learning_agent.actor.nz_action(state_tmp).view(-1).detach().numpy()

        reward_opt_th = env.get_reward(state_opt_th, action_opt_th)
        next_state_opt_th = env.get_next_state(state_opt_th, action_opt_th)
        next_state_opt_th[1] = next_state[1]
        epi_return_opt_th += reward_opt_th

        # Updating consumption thresholds
        if np.abs((np.sum(action) - state[1])) > 1e-6 :
            learning_agent.d_th_update(state,action)

        # Updating Q network and EV charging policy

        if interaction > 1000 and (interaction % 20 == 1) :
            for grad_update in range(20) :
                learning_agent.update(batch_size)


        # DDPG update
        if interaction > 1000 and (interaction % 20 == 1) :
            for grad_update in range(20):
                # agent_tddpg.update(batch_size, update_count)
                DDPG_agent.update(batch_size)

        if interaction % 50 == 1 :
            d_off_minus_history.append(copy.copy(learning_agent.actor.d_minus[1,:]))
            d_on_minus_history.append(copy.copy(learning_agent.actor.d_minus[0, :]))
            d_off_plus_history.append(copy.copy(learning_agent.actor.d_plus[1,:]))
            d_on_plus_history.append(copy.copy(learning_agent.actor.d_plus[0,:]))

            v_plus_history.append(copy.copy(learning_agent.actor.v_plus))
            v_minus_history.append(copy.copy(learning_agent.actor.v_minus))

        # Computing optimal action / reward for comparison
        d_prod = opt_d_minus[0,:] if np.isin(opt_state[-1], off_hrs) else opt_d_minus[1,:]
        d_cons = opt_d_plus[0,:] if np.isin(opt_state[-1], off_hrs) else opt_d_plus[1,:]
        v_prod = np.minimum(np.maximum(opt_state[0] - opt_delta[int(opt_state[-1])], 0), v_max)
        v_cons = np.minimum(np.maximum(opt_state[0] - opt_tau[int(opt_state[-1])], 0), v_max)
        th_prod = sum(d_prod) + v_prod
        th_cons = sum(d_cons) + v_cons

        if opt_state[1] < th_cons :
            opt_a = np.append(d_cons, v_cons)
        elif opt_state[1] > th_prod :
            opt_a = np.append(d_prod, v_prod)
        elif opt_state[-1] == T-1 :
            opt_v = v_prod
            opt_d = (opt_state[1] - opt_v + aa) * 0.5
            opt_a = np.append(opt_d, opt_v)
        else :
            theta = thetas[:,int(opt_state[-1])]
            grad_coef = np.flip(np.arange(1, len(theta) ))
            grad_coef2 = np.flip(np.arange(1, len(theta) -1) * np.arange(2, len(theta)))
            v_i = np.zeros(11)
            v_i[0] = np.maximum(opt_state[0] - opt_tau[int(opt_state[-1])], 0)
            for i in range(10) :
                yv = opt_state[0] - v_i[i]
                f_val = np.polyval(grad_coef * theta[0:-1], yv) + a[0] - 0.5 * (opt_state[1] - v_i[i] + a[0] - a[1])
                grad_val = -np.polyval(grad_coef2 * theta[0:-2], yv) + 0.5
                v_i[i+1] = v_i[i] - f_val / grad_val
                if abs(v_i[i+1] - v_i[i]) <= 1e-6 :
                    break
            opt_v = np.minimum(np.maximum(v_i[i+1], np.maximum(opt_state[0] - opt_tau[int(opt_state[-1])], 0)), np.minimum(np.maximum(opt_state[0] - opt_delta[int(opt_state[-1])],0), v_max))
            opt_d = 0.5 * (opt_state[1] - opt_v + aa)
            opt_a = np.append(opt_d, opt_v)
        opt_next_state = env.get_next_state(opt_state, opt_a)
        opt_next_state[1] = next_state[1]
        opt_r = env.get_reward(opt_state, opt_a)
        opt_epi_reward += opt_r
        opt_state = opt_next_state

        interaction += 1
        state = next_state
        DDPG_state = DDPG_next_state
        state_opt_th = next_state_opt_th

    opt_return.append(opt_epi_reward)
    avg_opt_return.append(np.mean(opt_return[-100:]))
    trained_reward.append(epi_reward)
    avg_trained_reward.append(np.mean(trained_reward[-100:]))
    DDPG_return.append(DDPG_epi_return)
    avg_DDPG_return.append(np.mean(DDPG_return[-100:]))
    opt_th_return.append(epi_return_opt_th)
    avg_opt_th_return.append(np.mean(opt_th_return[-100:]))

    if epi % 500 == 499 :
        toc = time.perf_counter()
        print(
            '{0}th Episode, {1:.4f} (s) time elapsed, average reward : {2:.4f}, optimal thresh rewward : {3:.4f}, DDPG return : {4:.4f}, opt reward : {5:.4f}'.format(
                epi, toc - tic, avg_trained_reward[-1], avg_opt_th_return[-1], avg_DDPG_return[-1], avg_opt_return[-1]))
        tic = time.perf_counter()
'''
x = np.arange(0, 15, 0.1)
action_test = np.array([])
for i in range(len(x)):
    state = np.array([x[i], 2, pi_p[1], pi_m[1], 5])
    actions = DDPG_agent.get_action(state).reshape(-1)
    action_test = np.append(action_test, actions[-1])

plt.plot(x, action_test)
plt.grid()
plt.show()

d_off_minus_history = np.vstack(d_off_minus_history)
d_on_minus_history = np.vstack(d_on_minus_history)
d_off_plus_history = np.vstack(d_off_plus_history)

plt.plot(np.arange(0, interaction, 50), v_minus_history)
plt.grid()
plt.show()

plt.plot(np.arange(0, interaction, 50), v_plus_history)
plt.grid()
plt.show()


print(learning_agent.v_update_count)

avg_trained_reward = np.array(avg_trained_reward)
avg_opt_return = np.array(avg_opt_return)
avg_DDPG_return = np.array(avg_DDPG_return)
avg_opt_th_return = np.array(avg_opt_th_return)
smoothed_learning_curve = np.array([])
smoothed_opt_return_curve = np.array([])
smoothed_ddpg_learning_curve = np.array([])
smoothed_opt_th_learning_curve = np.array([])
for i in range(num_epi) :
    smoothed_learning_curve = np.append(smoothed_learning_curve, np.mean(avg_trained_reward[np.maximum(i-100, 0):i+1]))
    smoothed_opt_return_curve = np.append(smoothed_opt_return_curve, np.mean(avg_opt_return[np.maximum(i-100, 0):i+1]))
    smoothed_ddpg_learning_curve = np.append(smoothed_ddpg_learning_curve, np.mean(avg_DDPG_return[np.maximum(i-100, 0):i+1]))
    smoothed_opt_th_learning_curve = np.append(smoothed_opt_th_learning_curve, np.mean(avg_opt_th_return[np.maximum(i-100, 0):i+1]))
#plt.plot(np.arange(0, num_epi * T, T), smoothed_learning_curve, label = 'Threshold learning')
plt.plot(np.arange(0, num_epi * T, T), smoothed_opt_return_curve, label = 'Optimal')
plt.plot(np.arange(0, num_epi * T, T), smoothed_ddpg_learning_curve, label = 'DDPG')
plt.plot(np.arange(0, num_epi * T, T), smoothed_opt_th_learning_curve, label = 'Optimal Thresh')
plt.ylabel('Performance')
plt.xlabel('Episodes')
plt.legend()
plt.grid()
plt.show()

regret_th_learning = np.abs(smoothed_opt_return_curve - smoothed_learning_curve) / smoothed_opt_return_curve * 100
regret_DDPG = np.abs(smoothed_opt_return_curve - smoothed_ddpg_learning_curve ) / smoothed_opt_return_curve * 100
regret_opt_th = np.abs(smoothed_opt_return_curve - smoothed_opt_th_learning_curve) / smoothed_opt_return_curve * 100

plt.plot(np.arange(10000, num_epi * T, T), regret_th_learning[1000:], label = 'Threshold learning')
#plt.plot(np.arange(10000, num_epi * T, T), regret_DDPG[1000:], label = 'DDPG')
plt.plot(np.arange(10000, num_epi * T, T), regret_opt_th[1000:], label = 'Optimal Threshold')
#plt.ylim(top = 10, bottom = 0)
plt.grid()
plt.show()


d_on_plus_history = np.vstack(d_on_plus_history)
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


x = np.arange(0, 15, 0.1)
V_test = np.array([])
for i in range(len(x)):
    state = torch.Tensor(np.array([x[i], 3, pi_p[0], pi_m[0], 7]))
    Values = learning_agent.value(state).view(-1).detach().numpy()
    V_test = np.append(V_test, Values.item())

plt.plot(x, V_test)
plt.show()
'''