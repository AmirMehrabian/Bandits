import logging
import os
import time
from pprint import pprint, pformat

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat

import config
from config import config_dict
from env_simulations.env_functions import env_response

snr_tn_vec = np.arange(-10, 20, 3)
snr_jn_vec = np.arange(10, 40, 4)
config_dict['num_coherence_symbols'] = 10000
tau_g = config_dict['num_coherence_symbols'] / config_dict['num_data_symbols']
print(f"tau_g: {tau_g}")

num_iter = 5000
prob_sym_error_vec = np.zeros((len(snr_jn_vec), len(snr_tn_vec)), dtype=float)
corr_vec = np.zeros((len(snr_jn_vec), len(snr_tn_vec)), dtype=float)
corr_vec_nb = np.zeros((len(snr_jn_vec), len(snr_tn_vec)), dtype=float)
p_jam_vec = np.zeros((len(snr_jn_vec), len(snr_tn_vec)), dtype=float)
p_signal_vec = np.zeros((len(snr_jn_vec), len(snr_tn_vec)), dtype=float)

nb = config_dict['num_pilot_block']
nd = config_dict['num_data_symbols'] - (config_dict['num_pilot_block'] - 1) * config_dict['num_pilot_symbols']
pprint(config_dict)
for idx_snr_jn, snr_jn in enumerate(snr_jn_vec):
    print(idx_snr_jn)
    for idx_snr_tn, snr_tn in enumerate(snr_tn_vec):
        print(snr_tn, end=', ')
        config_dict['snr_tn'] = snr_tn
        config_dict['snr_jn'] = snr_jn
        print(f'snr jn:{snr_jn} - snr tn:{snr_tn}', end=', ')
        agg_error = 0
        corr_agg = np.zeros(config_dict['num_pilot_block'] + 1)
        p_jam_agg = 0
        p_signal_agg = 0

        for _ in range(num_iter):
            total_rev, r_state, p_jam, p_signal = env_response(config_dict)
            error = 1 - total_rev / nd
            agg_error = agg_error + error
            corr_agg = corr_agg + np.degrees(np.acos(np.abs(r_state))) / 90
            p_jam_agg = p_jam_agg + p_jam
            p_signal_agg = p_signal_agg + p_signal

        prob_sym_error_vec[idx_snr_jn, idx_snr_tn] = agg_error / num_iter
        corr_vec[idx_snr_jn, idx_snr_tn] = corr_agg[0] / num_iter
        corr_vec_nb[idx_snr_jn, idx_snr_tn] = np.sum(corr_agg[1:]) / (config_dict['num_pilot_block'] * num_iter)
        p_jam_vec[idx_snr_jn, idx_snr_tn] = p_jam_agg / num_iter
        p_signal_vec[idx_snr_jn, idx_snr_tn] = p_signal_agg / num_iter

pprint(config_dict)

from matplotlib import pyplot as plt
from matplotlib import ticker

# Prepare the meshgrid
X, Y = np.meshgrid(snr_tn_vec, snr_jn_vec)


def plot_filled_contour(Z, title, xlabel, ylabel, cbar_label):
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=100, cmap='jet')  # Filled contour with 100 levels
    cbar = plt.colorbar(contour)
    cbar.set_label(cbar_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# 1. Symbol Error Probability
plot_filled_contour(np.log10(prob_sym_error_vec), "Symbol Error Probability", "SNR_tn (dB)", "SNR_jn (dB)",
                    "P_sym_error")

# 2. Mean Correlation (nb=1)
plot_filled_contour(corr_vec, "Mean Correlation (block 0)", "SNR_tn (dB)", "SNR_jn (dB)", "Correlation (nb=1)")

# 3. Mean Correlation (nb>1)
plot_filled_contour(corr_vec_nb, f"Mean Correlation (nb={nb})", "SNR_tn (dB)", "SNR_jn (dB)", f"Correlation (nb={nb})")

# 4. Average Signal Power
plot_filled_contour(p_signal_vec, "Average Signal Power", "SNR_tn (dB)", "SNR_jn (dB)", "Signal Power")

# 5. Average Jammer Power
plot_filled_contour(p_jam_vec, "Average Jammer Power", "SNR_tn (dB)", "SNR_jn (dB)", "Jammer Power")

output_dir = "CAJ_Pes_snr_tn_jn"
os.makedirs(output_dir, exist_ok=True)

# === File paths ===
file_base = f'caj_nc_{config_dict["num_coherence_symbols"]}_rerun'
mat_path = os.path.join(output_dir, f"{file_base}.mat")
log_path = os.path.join(output_dir, f"{file_base}.log")

plot_info = {'curves': {
    'title': f'Simulation Results For {file_base}',
    'pes': prob_sym_error_vec,
    'log_pes': np.log10(prob_sym_error_vec),
    'corr': corr_vec,
    'corr_mean': corr_vec_nb,
    'p_sig': p_signal_vec,
    'p_jam': p_jam_vec,
    'snr_tn_vec': snr_tn_vec,
    'X_mesh_tn': X,
    'snr_jn_vec': snr_jn_vec,
    'X_mesh_jn': Y,
},
    'config': config.config_dict,
    'tau': tau_g,
}

savemat(mat_path, plot_info, )

logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w')

msg = f''' {'=' * 50} {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} {'=' * 50}\n
            Simulation Results For {file_base}\n
            {pformat(config.config_dict)}\n
            "tau": {tau_g}\n
            {'=' * 100} \n '''

logging.info(msg)
