from math import log
from pathlib import Path

import torch
import torch.nn as nn

from drdmannturb.shared.enums import DataType, EddyLifetimeType, PowerSpectraType
from drdmannturb.shared.parameters import (
    NumericalParameters,
    NNParameters,
    LossParameters,
    PhysicalParameters,
)

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

savedir = Path(__file__).parent / "results"


loss_params = LossParameters()
nn_params = NNParameters()
physics = PhysicalParameters(L=0.59, Gamma=3.9, sigma=3.4)

CONSTANTS_CONFIG = {
    "type_EddyLifetime": EddyLifetimeType.CUSTOMMLP,
    "type_PowerSpectra": PowerSpectraType.RDT,
    "learn_nu": False,  # NOTE: Experiment 1: False, Experiment 2: True
    # "plt_tau": True, NOTE: Deprecated currently
    "hidden_layer_sizes": [10, 10],  # ONLY NEEDED FOR CUSTOMNET OR RESNET
    "curves": [0, 1, 2, 3],
    "data_type": DataType.KAIMAL,  # CALIBRATION: 'Custom', ### 'Kaimal', 'SimiuScanlan', 'SimiuYeo', 'iso'
    "spectra_file": "Spectra.dat",
    "domain": torch.logspace(
        -1, 2, 20
    ),  # np.logspace(-4, 2, 40), ### NOTE: Experiment 1: np.logspace(-1, 2, 20), Experiment 2: np.logspace(-2, 2, 40)
    "noisy_data": 0.0,  # 0*3.e-1, ### level of the data noise  ### NOTE: Experiment 1: zero, Experiment 2: non-zero
    "output_folder": str(savedir),
}
