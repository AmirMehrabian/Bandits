import numpy as np
from scipy.io import savemat

from config import config_dict, step_dict
from env_simulations.env_functions import env_response
from utils import epsilon_greedy, context_builder_10features
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

step_list = step_dict['steps_param']
action_set = config_dict['action_set']
optimal_actions_idx_vec = step_dict['optimal_actions_idx_vec']
number_actions = len(action_set)
NUM_FEATURES = 11
NUM_EXPLOIT_EPISODE = 20
final_avg_error_vec = []
final_avg_opt_act_vec = []
final_avg_rev_vec = []
NUM_ITER = 10
# 100
num_train_vec = [1, 2, 4, 6, 8, 10]

for num_train in num_train_vec:

    temp_save = {'final_avg_rev': final_avg_rev_vec,
    'final_avg_opt_act': final_avg_opt_act_vec,
    'final_avg_error': final_avg_error_vec,
    'num_tain_vec': num_train_vec,
    'num_iter': NUM_ITER,}
    savemat('temp_data_cmab.mat', temp_save, )
    epsilon_vec = np.ones(num_train) - np.arange(num_train) / num_train
    epsilon_vec = np.concatenate([epsilon_vec, np.zeros(1)])
    print('num_train: ', num_train, epsilon_vec)

    PRINT_UPDATE_INTERVAL = 20
    # Number of features in the context vector
    NUM_EPISODES = len(epsilon_vec)
    agg_final_avg_rev = 0
    agg_final_avg_opt_act = 0
    agg_final_avg_error = 0

    for iter_idx in range(NUM_ITER):
        print('Iter: ', iter_idx + 1, end=' - ')
        avg_error = []
        avg_rev = []
        avg_opt_act = []

        avg_curve = np.zeros(step_list.shape[1])
        avg_opt_curve = np.zeros(step_list.shape[1])

        a_mats = dict()
        b_vecs = dict()
        theta_vecs = dict()
        x_vecs = dict()

        for index_action, action in enumerate(action_set):
            a_mats[index_action] = np.eye(NUM_FEATURES)
            b_vecs[index_action] = np.zeros(NUM_FEATURES)
            theta_vecs[index_action] = np.zeros(NUM_FEATURES)

        eps_zero_count = 0
        all_avg_error = 0

        for epsilon_idx, episode_idx in enumerate(range(NUM_EPISODES)):

            epsilon = epsilon_vec[epsilon_idx]

            # print(f"Episode: {episode_idx + 1}, Epsilon: {epsilon}")

            # Randomly selecting the action and first episode
            action_index = np.random.choice(number_actions)
            config_dict['action_idx'] = action_index
            config_dict['num_pilot_block'] = action_set[action_index]

            rnd_step_idx = np.random.choice(step_list.shape[1])

            config_dict['num_coherence_symbols'] = step_list[0][rnd_step_idx]
            config_dict['snr_jn'] = step_list[1][rnd_step_idx]
            config_dict['snr_tn'] = step_list[2][rnd_step_idx]

            # Initialize the first context
            total_reward, corr_vec, power_jn_db, power_tn_db = env_response(config_dict)

            avg_vec = []
            avg_opt_vec = []
            agg_err = 0
            agg_rev = 0
            agg_optimal_action = 0
            for counter, step_params in enumerate(np.array(step_list).T):

                est_reward_vec = np.zeros(number_actions)
                for index_action, action in enumerate(action_set):
                    x_vecs[index_action] = context_builder_10features(corr_vec, power_jn_db, power_tn_db, index_action,
                                                                      config_dict)

                    theta_vecs[index_action] = np.linalg.pinv(a_mats[index_action]) @ b_vecs[
                        index_action]  # Q = A^(-1) b
                    est_reward_vec[index_action] = x_vecs[index_action] @ theta_vecs[index_action]  # r = Q^T x
                    est = x_vecs[index_action] @ theta_vecs[index_action]
                    est2 = x_vecs[index_action].T @ theta_vecs[index_action]

                action_index = epsilon_greedy(epsilon, est_reward_vec)
                config_dict['action_idx'] = action_index
                config_dict['num_pilot_block'] = action_set[action_index]

                # if counter % PRINT_UPDATE_INTERVAL == 0:
                #     print(counter, end=', ')

                # Observing new env params based on step_params
                config_dict['num_coherence_symbols'] = step_params[0]
                config_dict['snr_jn'] = step_params[1]
                config_dict['snr_tn'] = step_params[2]

                total_reward, corr_vec, power_jn_db, power_tn_db = env_response(config_dict)
                outer_x = np.outer(x_vecs[action_index], x_vecs[action_index])

                # Updating the A and b
                if epsilon > 0:
                    a_mats[action_index] = a_mats[action_index] + outer_x  # A <- A + xx^T
                    b_vecs[action_index] = b_vecs[action_index] + total_reward * x_vecs[action_index]  # b <- b + r x

                is_optimal = 1 if optimal_actions_idx_vec[counter] == action_index else 0

                agg_err = agg_err + abs(total_reward - est_reward_vec[action_index])
                agg_rev = agg_rev + total_reward
                agg_optimal_action = agg_optimal_action + is_optimal

                avg_vec.append(total_reward)
                avg_opt_vec.append(is_optimal)

            avg_error.append(agg_err / step_list.shape[1])
            avg_rev.append(agg_rev / step_list.shape[1])
            avg_opt_act.append(agg_optimal_action / step_list.shape[1])

            # print('\n', f'avg_err: {agg_err / step_list.shape[1]} ',
            #       f'avg_rev: {agg_rev / step_list.shape[1]} ',
            #       f'avg_opt_act: {agg_optimal_action / step_list.shape[1]} ')
            #
            # print("-" * 50)

            if epsilon <= 0.0001:
                avg_curve = avg_curve + avg_vec
                avg_opt_curve = avg_opt_curve + avg_opt_vec
                all_avg_error = all_avg_error + agg_err / step_list.shape[1]
                eps_zero_count += 1

        final_avg_rev = np.mean(avg_curve / eps_zero_count)
        agg_final_avg_rev = agg_final_avg_rev + final_avg_rev

        final_avg_opt_act = np.mean(avg_opt_curve / eps_zero_count)
        agg_final_avg_opt_act = agg_final_avg_opt_act + final_avg_opt_act

        final_avg_error = all_avg_error / eps_zero_count
        agg_final_avg_error = agg_final_avg_error + final_avg_error

    final_avg_rev_vec.append(agg_final_avg_rev / NUM_ITER)
    final_avg_opt_act_vec.append(agg_final_avg_opt_act / NUM_ITER)
    final_avg_error_vec.append(agg_final_avg_error / NUM_ITER)

    print('\n', '-' * 100, '\n', 'total_average_rew: ', agg_final_avg_rev / NUM_ITER)
    print('total_average_opt_act: ', agg_final_avg_opt_act / NUM_ITER)
    print('total_average_error: ', agg_final_avg_error / NUM_ITER)

    episode_idx = np.arange(NUM_EPISODES)

plt.figure(1)
plt.plot(num_train_vec, final_avg_rev_vec)
plt.xlabel("Number of Training Episodes")
plt.ylabel("Average Reward")
plt.grid(True)

plt.figure(2)
plt.plot(num_train_vec, final_avg_opt_act_vec)
plt.xlabel("Number of Training Episodes")
plt.ylabel("Optimial_action_Selection (%)")
plt.grid(True)

plt.figure(3)
plt.plot(num_train_vec, final_avg_error_vec)
plt.xlabel("Number of Training Episodes")
plt.ylabel("Average Error")
plt.grid(True)

plt.show()
