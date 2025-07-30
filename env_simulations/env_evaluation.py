import numpy as np
from matplotlib import pyplot as plt
from config import config_dict
from env_simulations.env_functions import env_response

snr_tn_vec = np.arange(-10, 40, 4)
num_iter = 200
prob_sym_error_vec = []
corr_vec = []
corr_vec_nb = []
p_jam_vec = []
p_signal_vec = []

nb = config_dict['num_pilot_block']
nd = config_dict['num_data_symbols'] - (config_dict['num_pilot_block'] - 1) * config_dict['num_pilot_symbols']


for snr_tn in snr_tn_vec:
    print(snr_tn, end=', ')
    config_dict['snr_tn'] = snr_tn
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

    prob_sym_error_vec.append(agg_error / num_iter)
    corr_vec.append(corr_agg[0] / num_iter)
    corr_vec_nb.append(np.sum(corr_agg[1:]) / (config_dict['num_pilot_block'] * num_iter))
    p_jam_vec.append(p_jam_agg / num_iter)
    p_signal_vec.append(p_signal_agg / num_iter)

print(config_dict)

plt.semilogy(snr_tn_vec, prob_sym_error_vec)
plt.xlabel("SNR_s")
plt.ylabel("Average Error")
plt.grid(True, which='both')
plt.show()

plt.plot(snr_tn_vec, corr_vec)
plt.plot(snr_tn_vec, corr_vec_nb)
plt.xlabel("SNR_s")
plt.ylabel("Mean_Corr_vec")
plt.grid(True)
plt.legend(['nb=1', f'nb={nb}'])
plt.show()

plt.plot(snr_tn_vec, p_signal_vec)
plt.xlabel("SNR_s")
plt.ylabel("Mean_SNR_S")
plt.grid(True)
plt.show()

plt.plot(snr_tn_vec, p_jam_vec)
plt.xlabel("SNR_s")
plt.ylabel("Mean_SNR_J")
plt.grid(True)
plt.show()
