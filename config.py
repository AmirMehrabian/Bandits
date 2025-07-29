import numpy as np

config_dict = {
    "num_jn": 1,
    "num_sn": 4,
    "num_antennas": 64,

    "snr_tn": 10,    # in dB
    "snr_jn": 20,     # in dB

    "num_coherence_symbols": 1000,  #N_tc
    "num_pilot_symbols": 20,         #Nt1
    "num_data_symbols": 500,
    # N_d1
    "nakagami_shape_param": 2.0,

    "action_set": np.array([1, 2, 5]),
    "action_idx": 1,
    "num_pilot_block": 4,
}


step_dict = {}

part_repeat = 100

num_coherence_symbols_frame = np.concatenate([
    5000 * np.ones(part_repeat),
    3000 * np.ones(part_repeat),
    1000 * np.ones(part_repeat),
    5000 * np.ones(part_repeat),
    3000 * np.ones(part_repeat)
])
snr_jn_frame = np.concatenate([
    20 * np.ones(part_repeat),
    40 * np.ones(part_repeat),
    40 * np.ones(part_repeat),
    40 * np.ones(part_repeat),
    40 * np.ones(part_repeat)
])

snr_tn_frame = np.concatenate([
    10 * np.ones(part_repeat),
    5 * np.ones(part_repeat),
    20 * np.ones(part_repeat),
    20 * np.ones(part_repeat),
    20 * np.ones(part_repeat)
])

step_dict['steps_param'] = np.vstack([num_coherence_symbols_frame, snr_jn_frame, snr_tn_frame])


