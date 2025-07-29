import numpy as np
from config import config_dict, step_dict
from env_simulations.env_functions import env_response
from utils import epsilon_greedy
import matplotlib

matplotlib.use('TkAgg')  # or 'Qt5Agg' or another supported GUI backend
import matplotlib.pyplot as plt

step_list = step_dict['steps_param']

action_set = config_dict['action_set']
number_actions = len(action_set)

# Epsilon setting
epsilon = 0.15
learning_rate = 0.3

num_episodes = 10

avg_error = []
avg_rev = []

avg_curve = np.zeros(step_list.shape[1])

q_vec = np.zeros(number_actions)

eps_zero_count = 0
all_avg_error = 0

for episode_idx in range(num_episodes):

    q_vec = np.zeros(number_actions)
    print(f"Episode: {episode_idx + 1}, Epsilon: {epsilon}")

    # Randomly selecting the action and first episode
    action_index = np.random.choice(number_actions)
    config_dict['action_index'] = action_index
    config_dict['num_pilot_block'] = action_set[action_index]

    rnd_step_idx = np.random.choice(step_list.shape[1])
    print(rnd_step_idx)
    config_dict['num_coherence_symbols'] = step_list[0][rnd_step_idx]
    config_dict['snr_jn'] = step_list[1][rnd_step_idx]
    config_dict['snr_tn'] = step_list[2][rnd_step_idx]

    # Initialize the first context
    total_reward, corr_vec, power_jn_db, power_tn_db = env_response(config_dict)

    avg_vec = []
    agg_err = 0
    agg_rev = 0
    for counter, step_params in enumerate(np.array(step_list).T):

        action_index = epsilon_greedy(epsilon, q_vec)
        config_dict['action_idx'] = action_index
        config_dict['num_pilot_block'] = action_set[action_index]
        # print(counter, est_reward_vec,action_index)
        if counter % 20 == 0:
            print(counter, end=', ')
            # print(q_vec , action_index)

        # Observing new env params based on step_params
        config_dict['num_coherence_symbols'] = step_params[0]
        config_dict['snr_jn'] = step_params[1]
        config_dict['snr_tn'] = step_params[2]

        # Taking action in that context
        total_reward, corr_vec, power_jn_db, power_tn_db = env_response(config_dict)

        # Updating q vector q <- q + a ( r - q)
        q_vec[action_index] = q_vec[action_index] + learning_rate * (total_reward - q_vec[action_index])

        agg_err = agg_err + abs(total_reward - q_vec[action_index])
        agg_rev = agg_rev + total_reward
        avg_vec = np.append(avg_vec, total_reward)

    avg_error = np.append(avg_error, agg_err / step_list.shape[1])
    avg_rev = np.append(avg_rev, agg_rev / step_list.shape[1])
    print('\n', f'avg_err: {agg_err / step_list.shape[1]}, avg_rev: {agg_rev / step_list.shape[1]} ')
    print("-" * 50)
    if epsilon <= 1:
        avg_curve = avg_curve + avg_vec
        all_avg_error = all_avg_error + agg_err / step_list.shape[1]
        eps_zero_count += 1

print('total_average_rew: ', np.mean(avg_curve / eps_zero_count))

print('total_average_error: ', all_avg_error / eps_zero_count)
plt.figure(1)
plt.plot(list(range(num_episodes)), avg_error)
plt.xlabel("Episodes")
plt.ylabel("Average Error")
plt.grid(True)

plt.figure(2)
plt.plot(list(range(step_list.shape[1])), avg_curve / eps_zero_count)
plt.xlabel("Steps")
plt.ylabel("Average Error")
plt.grid(True)

plt.figure(3)
plt.plot(list(range(num_episodes)), avg_rev)
plt.xlabel("Episodes")
plt.ylabel("Average Rev")
plt.grid(True)

plt.show()
