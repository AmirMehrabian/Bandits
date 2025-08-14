import numpy as np
from config import config_dict, step_dict
from env_simulations.env_functions import env_response
from mabs.utils import epsilon_greedy, model_builder, model_feeder_no_action, context_builder_10features, ReplayBuffer
import matplotlib.pyplot as plt
from mabs.under_varied_env import new_env_config
# Epsilon setting
NUM_FEATURES = 11

BUFFER_CAPACITY = 5000
BATCH_SIZE = 2500
LEARNING_INTERVAL = 10
EPOCHS = 20

PRINT_UPDATE_INTERVAL = 20

# Epsilon setting
EPSILON_INIT = config_dict['epsilon_initial']  # 0.99
EPSILON_MIN = config_dict['epsilon_min']  # 0
EPSILON_DECAY = config_dict['epsilon_decay']  # 0.03

NUM_EPISODES = config_dict['num_episode_cmab']

step_list = step_dict['steps_param']
action_set = config_dict['action_set']
optimal_actions_idx_vec = step_dict['optimal_actions_idx_vec']
number_actions = len(action_set)

input_model_size = NUM_FEATURES

avg_error = []
avg_rev = []
avg_opt_act = []

avg_curve = np.zeros(new_env_config.step_dict['steps_param'].shape[1])
avg_opt_curve = np.zeros(new_env_config.step_dict['steps_param'].shape[1])

model_list = []
buffers = []
for index_action, action in enumerate(action_set):
    model_list.append(model_builder(NUM_FEATURES, 0))
    model_list[index_action].summary()
    buffers.append(ReplayBuffer(capacity=BUFFER_CAPACITY, input_model_size=input_model_size))

eps_zero_count = 0
all_avg_error = 0
online_learning = True
for episode_index in range(NUM_EPISODES):

    epsilon = max(EPSILON_MIN, EPSILON_INIT - (episode_index * EPSILON_DECAY))
    if epsilon <= 0.0001:
        step_list = new_env_config.step_dict['steps_param']
        online_learning = True #False
        optimal_actions_idx_vec = new_env_config.step_dict['optimal_actions_idx_vec']
        print(new_env_config.coherence_per_part)

    print(f"Episode: {episode_index + 1}, Epsilon: {epsilon}")

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

        est_reward_vec = []

        for index_action, action in enumerate(action_set):
            context = context_builder_10features(corr_vec, power_jn_db, power_tn_db, index_action, config_dict)
            model = model_list[index_action]
            model_input = model_feeder_no_action(context)
            model_output = model.predict(model_input, verbose=False)
            est_reward_vec = np.append(est_reward_vec, model_output.reshape(-1))

        action_index = epsilon_greedy(epsilon, est_reward_vec)
        config_dict['action_idx'] = action_index
        config_dict['num_pilot_block'] = action_set[action_index]

        if counter % PRINT_UPDATE_INTERVAL == 0:
            print(counter, end=', ')

        # Observing new env params based on step_params
        config_dict['num_coherence_symbols'] = step_params[0]
        config_dict['snr_jn'] = step_params[1]
        config_dict['snr_tn'] = step_params[2]

        # taking action in that context
        total_reward, corr_vec, power_jn_db, power_tn_db = env_response(config_dict)

        context = context_builder_10features(corr_vec, power_jn_db, power_tn_db, action_index, config_dict)

        new_input_sample = model_feeder_no_action(context)
        model = model_list[action_index]
        buffer = buffers[action_index]

        is_optimal = 1 if optimal_actions_idx_vec[counter] == action_index else 0

        agg_err = agg_err + abs(total_reward - est_reward_vec[action_index])
        agg_rev = agg_rev + total_reward
        agg_optimal_action = agg_optimal_action + is_optimal

        avg_vec.append(total_reward)
        avg_opt_vec.append(is_optimal)

        # Adding data to buffer
        buffer.add_to_buffer(new_input_sample, total_reward.reshape(-1, 1))

        if counter % LEARNING_INTERVAL == 0 and counter > 0:
            if online_learning:
                batch_input, batch_output = buffer.sample_from_buffer(BATCH_SIZE)
                model.fit(batch_input, batch_output, epochs=EPOCHS, verbose=False)
            else:
                print("Not online learning")

    avg_error.append(agg_err / step_list.shape[1])
    avg_rev.append(agg_rev / step_list.shape[1])
    avg_opt_act.append(agg_optimal_action / step_list.shape[1])

    print('\n', f'avg_err: {agg_err / step_list.shape[1]} ',
          f'avg_rev: {agg_rev / step_list.shape[1]} ',
          f'avg_opt_act: {agg_optimal_action / step_list.shape[1]} ')
    print("-" * 50)

    if epsilon <= 0.0001:
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
plt.ylabel("Optimial_action_Selection")
plt.grid(True)

plt.show()
