import numpy as np
from config import config_dict, step_dict
from env_simulations.env_functions import env_response
import tqdm
import matplotlib

matplotlib.use('TkAgg')  # or 'Qt5Agg' or another supported GUI backend
import matplotlib.pyplot as plt

PRINT_UPDATE_INTERVAL = 20
NUM_EPISODES = config_dict['num_episode_mab']  # 100
LEARNING_RATE = config_dict['learning_rate_mab']  # 0.3
EPSILON = config_dict['epsilon_mab']  # 0.15

policy = config_dict['policy']
step_list = step_dict['steps_param']
action_set = config_dict['action_set']
optimal_actions_idx_vec = step_dict['optimal_actions_idx_vec']
number_actions = len(action_set)

avg_error = []
avg_rev = []
avg_opt_act = []

avg_curve = np.zeros(step_list.shape[1])
avg_opt_curve = np.zeros(step_list.shape[1])

q_vec = np.zeros(number_actions)

eps_zero_count = 0
all_avg_error = 0

for episode_idx in range(NUM_EPISODES):

    q_vec = np.zeros(number_actions)
    print(f"Episode: {episode_idx + 1}, Epsilon: {EPSILON}")

    # Randomly selecting the action and first episode
    action_index = np.random.choice(number_actions)
    config_dict['action_idx'] = action_index
    config_dict['num_pilot_block'] = action_set[action_index]

    rnd_step_idx = np.random.choice(step_list.shape[1])

    config_dict['num_coherence_symbols'] = step_list[0][rnd_step_idx]
    config_dict['snr_jn'] = step_list[1][rnd_step_idx]
    config_dict['snr_tn'] = step_list[2][rnd_step_idx]

    # Initialize the first context
    total_reward, _, _, _ = env_response(config_dict)

    avg_vec = []
    avg_opt_vec = []
    agg_err = 0
    agg_rev = 0
    agg_optimal_action = 0
    for counter, step_params in enumerate(np.array(step_list).T):

        action_index = policy(EPSILON, q_vec)
        config_dict['action_idx'] = action_index
        config_dict['num_pilot_block'] = action_set[action_index]

        if counter % PRINT_UPDATE_INTERVAL == 0:
            print(counter, end=', ')

        # Observing new env params based on step_params
        config_dict['num_coherence_symbols'] = step_params[0]
        config_dict['snr_jn'] = step_params[1]
        config_dict['snr_tn'] = step_params[2]

        # Taking action in that context
        total_reward, _, _, _ = env_response(config_dict)

        # Updating q vector q <- q + a ( r - q)
        q_vec[action_index] = q_vec[action_index] + LEARNING_RATE * (total_reward - q_vec[action_index])

        is_optimal = 1 if optimal_actions_idx_vec[counter] == action_index else 0

        agg_err = agg_err + abs(total_reward - q_vec[action_index])
        agg_rev = agg_rev + total_reward
        agg_optimal_action = agg_optimal_action + is_optimal

        avg_vec.append(total_reward)
        avg_opt_vec.append(is_optimal)

    avg_error.append(agg_err / step_list.shape[1])
    avg_rev.append(agg_rev / step_list.shape[1])
    avg_opt_act.append(agg_optimal_action / step_list.shape[1])

    print('\n', f'avg_err: {agg_err / step_list.shape[1]} ',
          f'avg_rev: {agg_rev / step_list.shape[1]} ',
          f'avg_opt_act: {agg_optimal_action / step_list.shape[1]} ')

    print("-" * 50)

    if EPSILON <= 1.5:
        avg_curve = avg_curve + avg_vec
        avg_opt_curve = avg_opt_curve + avg_opt_vec
        all_avg_error = all_avg_error + agg_err / step_list.shape[1]
        eps_zero_count += 1

final_avg_rev = np.mean(avg_curve / eps_zero_count)
final_avg_opt_act = np.mean(avg_opt_curve / eps_zero_count)
final_avg_error = all_avg_error / eps_zero_count

print('total_average_rew: ', final_avg_rev)
print('total_average_opt_act: ', final_avg_opt_act)
print('total_average_error: ', final_avg_error)

episode_idx = np.arange(NUM_EPISODES)

plt.figure(1)
plt.plot(episode_idx, avg_error)
plt.xlabel("Episodes")
plt.ylabel("Average Error")
plt.grid(True)

avg_curve = avg_curve / eps_zero_count
step_idx = np.arange(step_list.shape[1])

plt.figure(2)
plt.plot(step_idx, avg_curve)
plt.xlabel("Steps")
plt.ylabel("Average Error")
plt.grid(True)

avg_opt_curve = avg_opt_curve / eps_zero_count

plt.figure(3)
plt.plot(step_idx, 100 * avg_opt_curve)
plt.xlabel("Steps")
plt.ylabel("Optimial_action_Selection (%)")
plt.grid(True)

plt.figure(4)
plt.plot(episode_idx, avg_rev)
plt.xlabel("Episodes")
plt.ylabel("Average Rev")
plt.grid(True)

plt.figure(5)
plt.plot(episode_idx, avg_opt_act)
plt.xlabel("Episodes")
plt.ylabel("Average Rev")
plt.grid(True)

plt.show()
