import numpy as np

from mabs.utils import epsilon_greedy

config_dict = {
    # Number of antennas, sensing nodes (SN), and jamming nodes (JN)
    "num_jn": 1,
    "num_sn": 4,
    "num_antennas": 64,
    # Signal-to-noise ratios
    "snr_tn": 10,  # in dB
    "snr_jn": 20,  # in dB
    # Time frames and symbols parameters
    "num_coherence_symbols": 1000,
    "num_pilot_symbols": 20,
    "num_data_symbols": 500,
    # Channel parameter
    "nakagami_shape_param": 2.0,
    # Action parameters
    "action_set": np.array([1, 2, 4, 5, 8]), #np.array([1, 8]),  # np.array([1, 4, 8]) , # np.array([1, 2, 4, 5, 8])# np.array([1, 2, 5]),
    "action_idx": 2,
    "num_pilot_block": 5,
    "epsilon_mab": 0.2,
    "learning_rate_mab": 0.5,  # 0.3,
    "num_episode_mab": 100,
    "policy": epsilon_greedy,  # lambda *x: 2,  # epsilon_greedy, #lambda *x: 2,  # epsilon_greedy
    "num_episode_cmab": 50,
    "epsilon_initial": 0.99,
    "epsilon_min": 0,
    "epsilon_decay": 0.1,
}
print(config_dict)

PART_SIZE = 40  # 100
EPISODE_PARTS = 6

coherence_per_part = [5000, 5000, 3000, 3000, 1000, 1000]  # env_change [1000, 3000, 5000, 1000, 3000, 5000, 1000, 3000,
# 5000]  # training [5000, 3000, 5000]  # [1000, 3000, 5000]  # [5000, 3000, 1000, 5000, 3000]
snr_jn_per_part = [20, 30, 30, 40, 40, 40]
# [40, 40, 40, 40, 40, 40, 40, 40, 40]  # [20, 40, 40]  # [40, 40, 40]  # [20, 40, 40, 40, 40]
snr_tn_per_part = [20, 20, 20, 15, 15, 10]
# [15, 15, 15, 20, 20, 20, 25, 25, 25]  # [10, 5, 20]  # [15, 15, 15]  # [10, 5, 20, 20, 20]
optimal_actions_idx_per_part = [2, 2, 2, 2, 2, 2, 2, 2, 2]  # [0, 2, 1]  # [2, 2, 1]  # [0, 2, 2, 1, 1]

# Initialize empty arrays
num_coherence_symbols_part = []
snr_jn_part = []
snr_tn_part = []
optimal_actions_idx = []  # This variable is not used in the provided code, but initialized for completeness

# Fill arrays using a loop
for i in range(EPISODE_PARTS):
    num_coherence_symbols_part.extend([coherence_per_part[i]] * PART_SIZE)
    snr_jn_part.extend([snr_jn_per_part[i]] * PART_SIZE)
    snr_tn_part.extend([snr_tn_per_part[i]] * PART_SIZE)
    optimal_actions_idx.extend([optimal_actions_idx_per_part[i]] * PART_SIZE)

# Convert to NumPy arrays
num_coherence_symbols_part = np.array(num_coherence_symbols_part)
snr_jn_part = np.array(snr_jn_part)
snr_tn_part = np.array(snr_tn_part)
optimal_actions_idx = np.array(optimal_actions_idx)

episode_param = {
    'coherence_per_part': coherence_per_part,
    'snr_jn_per_part': snr_jn_per_part,
    'snr_tn_per_part': snr_tn_per_part,
    'optimal_actions_idx_per_part': optimal_actions_idx_per_part,
    'part_size': PART_SIZE,
}

step_dict = {
    'steps_param': np.vstack([
        num_coherence_symbols_part,
        snr_jn_part,
        snr_tn_part]),
    'optimal_actions_idx_vec': optimal_actions_idx
}
