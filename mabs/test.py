import config
from mabs import mab
import logging
from scipy.io import savemat

script = mab

plot_info = {
            "title": "MAB Simulation Results",
            'rev':mab.avg_rev,
            'err':mab.avg_error,
            'curve':mab.avg_curve,
            'episodes':mab.episode_idx,
            'steps':mab.step_idx,}

savemat(f"output.mat", plot_info)

logging.basicConfig(filename="logfile.log", level=logging.INFO)

logging.info(f"This is the setup:{config.config_dict}")


