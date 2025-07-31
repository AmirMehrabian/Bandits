import config
from mabs import mab
import logging
from scipy.io import savemat
from pprint import pformat

script = mab
file_name = f"{script.__name__}.mat"

plot_info = {
    "title": "MAB Simulation Results",
    'rev': script.avg_rev,
    'err': script.avg_error,
    'curve': script.avg_curve,
    'episodes': script.episode_idx,
    'steps': script.step_idx, }

savemat(file_name, plot_info)

logging.basicConfig(filename=f"{script.__name__}.log", level=logging.INFO)

logging.info(f"{script.__name__}\n, pformat(config.config_dict)")



