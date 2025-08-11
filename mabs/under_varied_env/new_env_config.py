import numpy as np
PART_SIZE = 50  # 100
EPISODE_PARTS = 4

coherence_per_part = [2000, 4000, 2000, 4000]  # [1000, 3000, 5000]  # [5000, 3000, 1000, 5000, 3000]
snr_jn_per_part = [40, 40, 40, 40]  # [40, 40, 40]  # [20, 40, 40, 40, 40]
snr_tn_per_part = [18, 18, 22, 22]  # [15, 15, 15]  # [10, 5, 20, 20, 20]
optimal_actions_idx_per_part = [2, 1, 2, 1]  # [2, 2, 1]  # [0, 2, 2, 1, 1]

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
