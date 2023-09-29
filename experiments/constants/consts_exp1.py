from pathlib import Path

import torch

from drdmannturb.shared.parameters import (
    ProblemParameters,
    NNParameters,
    LossParameters,
    PhysicalParameters,
)

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

savedir = Path(__file__).parent / "results"

prob_params = ProblemParameters()
loss_params = LossParameters()
nn_params = NNParameters()
physics = PhysicalParameters(L=0.59, Gamma=3.9, sigma=3.4)

CONSTANTS_CONFIG = {
    "hidden_layer_sizes": [10, 10],  # ONLY NEEDED FOR CUSTOMNET OR RESNET
    "curves": [0, 1, 2, 3],
    "spectra_file": "Spectra.dat",
    # "noisy_data": 0.0,  # 0*3.e-1, ### level of the data noise  ### NOTE: Experiment 1: zero, Experiment 2: non-zero
    "output_folder": str(savedir),
}
