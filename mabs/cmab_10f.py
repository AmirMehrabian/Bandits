import numpy as np
from config import config_dict, step_dict
from env_simulations.env_functions import env_response
from utils import epsilon_greedy, context_builder_12features
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

step_list = step_dict['steps_param']

action_set = config_dict['action_set']

number_actions = len(action_set)

# Epsilon setting
epsilon_init = 0.99
epsilon_min = 0
epsilon_decay = 0.1  # 0.025

num_episodes = 20  # 100
avg_error = []
avg_rev = []

num_features = 11
avg_curve = np.zeros(step_list.shape[1])

a_mats = dict()
b_vecs = dict()
theta_vecs = dict()
x_vecs = dict()

for index_action, action in enumerate(action_set):
    a_mats[index_action] = np.eye(num_features)
    b_vecs[index_action] = np.zeros(num_features)
    theta_vecs[index_action] = np.zeros(num_features)

eps_zero_count = 0
all_avg_error = 0

for episode_idx in range(num_episodes):

    epsilon = max(epsilon_min, epsilon_init - (episode_idx * epsilon_decay))

    print(f"Episode: {episode_idx + 1}, Epsilon: {epsilon}")

    # Randomly selecting the action and first episode
    action_index = np.random.choice(number_actions)
    config_dict['action_idx'] = action_index
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

        est_reward_vec = np.zeros(number_actions)
        for index_action, action in enumerate(action_set):
            x_vecs[index_action] = context_builder_12features(corr_vec, power_jn_db, power_tn_db, index_action,
                                                              config_dict)

            theta_vecs[index_action] = np.linalg.pinv(a_mats[index_action]) @ b_vecs[index_action]  # Q = A^(-1) b
            est_reward_vec[index_action] = x_vecs[index_action] @ theta_vecs[index_action]          # r = Q^T x
            est = x_vecs[index_action] @ theta_vecs[index_action]
            est2 = x_vecs[index_action].T @ theta_vecs[index_action]

        action_index = epsilon_greedy(epsilon, est_reward_vec)
        config_dict['action_idx'] = action_index
        config_dict['num_pilot_block'] = action_set[action_index]

        if counter % 20 == 0:
            print(counter, end=', ')

        # Observing new env params based on step_params
        config_dict['num_coherence_symbols'] = step_params[0]
        config_dict['snr_jn'] = step_params[1]
        config_dict['snr_tn'] = step_params[2]

        total_reward, corr_vec, power_jn_db, power_tn_db = env_response(config_dict)
        outer_x = np.outer(x_vecs[action_index], x_vecs[action_index])

        # Updating the A and b
        a_mats[action_index] = a_mats[action_index] + outer_x  # A <- A + xx^T
        b_vecs[action_index] = b_vecs[action_index] + total_reward * x_vecs[action_index]  # b <- b + r x

        agg_err = agg_err + abs(total_reward - est_reward_vec[action_index])
        agg_rev = agg_rev + total_reward
        avg_vec = np.append(avg_vec, total_reward)

    avg_error = np.append(avg_error, agg_err / step_list.shape[1])
    avg_rev = np.append(avg_rev, agg_rev / step_list.shape[1])
    print('\n', f'avg_err: {agg_err / step_list.shape[1]}, avg_rev: {agg_rev / step_list.shape[1]} ')
    print("-" * 50)

    if epsilon <= 0.0001:
        avg_curve = avg_curve + avg_vec
        all_avg_error = all_avg_error + agg_err / step_list.shape[1]
        eps_zero_count += 1

print('total_average_error: ', all_avg_error / eps_zero_count)
print('total_average_reward: ', np.mean(avg_curve / eps_zero_count))


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
