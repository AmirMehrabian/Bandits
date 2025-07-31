import numpy as np

from mabs.utils import epsilon_greedy

STEP_REPETITION = 100

config_dict = {
    # Number of antennas, sensors, and jammers
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
    "action_set": np.array([1, 2, 5]),
    "action_idx": 1,
    "num_pilot_block": 4,
    "epsilon_mab": 0.15,
    "policy": lambda *x: 0,  # epsilon_greedy
}

step_dict = {}

num_coherence_symbols_frame = np.concatenate([
    5000 * np.ones(STEP_REPETITION),
    3000 * np.ones(STEP_REPETITION),
    1000 * np.ones(STEP_REPETITION),
    5000 * np.ones(STEP_REPETITION),
    3000 * np.ones(STEP_REPETITION)
])
snr_jn_frame = np.concatenate([
    20 * np.ones(STEP_REPETITION),
    40 * np.ones(STEP_REPETITION),
    40 * np.ones(STEP_REPETITION),
    40 * np.ones(STEP_REPETITION),
    40 * np.ones(STEP_REPETITION)
])

snr_tn_frame = np.concatenate([
    10 * np.ones(STEP_REPETITION),
    5 * np.ones(STEP_REPETITION),
    20 * np.ones(STEP_REPETITION),
    20 * np.ones(STEP_REPETITION),
    20 * np.ones(STEP_REPETITION)
])

step_dict['steps_param'] = np.vstack([
    num_coherence_symbols_frame,
    snr_jn_frame,
    snr_tn_frame])
