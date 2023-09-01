"""Generating ONNX ports of existing nn.Module classes"""
# SCRAP THIS FOR NOW

import torch
import torch.onnx
<<<<<<< HEAD

from drdmannturb import OnePointSpectra
=======
from fracturbulence import OnePointSpectra
>>>>>>> af53bc6372a2a2bdc7e4ce595385e73ece68a031

OPS_CONFIG = {
    "type_EddyLifetime": "tauNet",
    "type_PowerSpectra": "RDT",
    "hidden_layer_size": 2,
    "learn_nu": False,
    "domain": torch.logspace(
        -1, 2, 20
    ),  # NOTE:  Experiment 1: np.logspace(-1, 2, 20), Experiment 2: np.logspace(-2, 2, 40)
}


if __name__ == "__main__":
    pass
