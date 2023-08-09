"""Simple profiling for OnePointSpectra Module with config from Calibration tasks"""

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from fracturbulence import OnePointSpectra

CONFIG = {
    "type_EddyLifetime": "tauNet",
    "type_PowerSpectra": "RDT",
    "hidden_layer_size": 2,
    "learn_nu": False,
    "domain": torch.logspace(
        -1, 2, 20
    ),  # NOTE:  Experiment 1: np.logspace(-1, 2, 20), Experiment 2: np.logspace(-2, 2, 40)
}


def driver():
    pass


if __name__ == "__main__":
    driver()