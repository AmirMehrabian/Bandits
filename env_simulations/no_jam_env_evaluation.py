import time
from pprint import pformat

import numpy as np
from scipy.io import savemat

from config import config_dict, step_dict
from env_simulations.env_functions import env_response, env_response_with_mitigation
import logging
import matplotlib
import os
matplotlib.use('TkAgg')  # or 'Qt5Agg' or another supported GUI backend
import matplotlib.pyplot as plt
import config

MITIGATION = False
file_base = f'sen1_NoJN'
NUM_EPISODES = 50  # 100

step_list = step_dict['steps_param']
action_set = config_dict['action_set']

number_actions = len(action_set)

avg_rev = []

avg_curve = np.zeros(step_list.shape[1])

q_vec = np.zeros(number_actions)

eps_zero_count = 0

for episode_idx in range(NUM_EPISODES):

    print(f"Episode: {episode_idx + 1}, ")

    avg_vec = []
    agg_rev = 0

    for counter, step_params in enumerate(np.array(step_list).T):
        config_dict['action_idx'] = 0
        config_dict['num_pilot_block'] = 1

        config_dict['num_coherence_symbols'] = step_params[0]
        config_dict['snr_jn'] = -500 #step_params[1]  # -500
        config_dict['snr_tn'] = step_params[2]

        # Taking action in that context
        total_reward, _, _, _ = env_response_with_mitigation(config_dict, mitigation=MITIGATION)

        agg_rev = agg_rev + total_reward

        avg_vec.append(total_reward)

    avg_rev.append(agg_rev / step_list.shape[1])

    print('\n', f'avg_rev: {agg_rev / step_list.shape[1]}')

    print("-" * 50)

    avg_curve = avg_curve + avg_vec

    eps_zero_count += 1

final_avg_rev = np.mean(avg_curve / eps_zero_count)

print('total_average_rew: ', final_avg_rev)

episode_idx = np.arange(NUM_EPISODES)

avg_curve = avg_curve / eps_zero_count
step_idx = np.arange(step_list.shape[1])

plt.figure(2)
plt.plot(step_idx, avg_curve)
plt.xlabel("Steps")
plt.ylabel("Average Error")
plt.grid(True)

plt.figure(4)
plt.plot(episode_idx, avg_rev)
plt.xlabel("Episodes")
plt.ylabel("Average Rev")
plt.grid(True)

plt.show()


output_dir = "NoJN_NoAJ_results"
os.makedirs(output_dir, exist_ok=True)

# === File paths ===

mat_path = os.path.join(output_dir, f"{file_base}.mat")
log_path = os.path.join(output_dir, f"{file_base}.log")

plot_info = {'curves': {
    'title': f'Simulation Results For {file_base}',
    'rev': avg_rev,
    'curve': avg_curve,
    'episodes': episode_idx,
    'steps': step_idx,
    'final_avg_rev': final_avg_rev,
    'mitigation': MITIGATION,
    },
    'config': config.config_dict,
    'episode_param': config.episode_param,
    }

savemat(mat_path, plot_info, )

logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w')

msg = f''' {'=' * 50} {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} {'=' * 50}\n
            Simulation Results For {file_base}\n
            {pformat(config.config_dict)}\n
            {pformat(config.episode_param)}\n
            {'=' * 100} \n '''

logging.info(msg)


print(config.config_dict)