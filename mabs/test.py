import os
import time
import config
import logging
from scipy.io import savemat
from pprint import pformat

#from mabs import mab as script
#from mabs import cmab_10f as script
#from mabs import dcmab_5f as script
from mabs import dcmab_10f as script

output_dir = "mabs_outputs"
os.makedirs(output_dir, exist_ok=True)

# === File paths ===
file_base = script.__name__ + f''
mat_path = os.path.join(output_dir, f"{file_base}.mat")
log_path = os.path.join(output_dir, f"{file_base}.log")

plot_info = {'curves': {
    'title': f'Simulation Results For {file_base}',
    'rev': script.avg_rev,
    'err': script.avg_error,
    'curve': script.avg_curve,
    'episodes': script.episode_idx,
    'steps': script.step_idx,
    'opt_act': script.avg_opt_act,
    'final_avg_rev': script.final_avg_rev,
    'final_avg_opt_act': script.final_avg_opt_act,
    'final_avg_error': script.final_avg_error,
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