"""# Custom Data Fit"""
# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline

from drdmannturb.calibration import CalibrationProblem
from drdmannturb.data_generator import OnePointSpectraDataGenerator
from drdmannturb.shared.common import MannEddyLifetime
from drdmannturb.shared.enums import DataType, EddyLifetimeType, PowerSpectraType
from drdmannturb.shared.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)

path = Path().resolve()

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

plt.style.use("bmh")

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# %%
spectra_file = (
    path / "../docs/source/data/Spectra_interp.dat"
    if path.name == "examples"
    else path / "../data/Spectra_interp.dat"
)


domain = torch.logspace(-1, 2, 20)

L = 70
GAMMA = 3.7
SIGMA = 0.04

Uref = 21
zref = 1

pb = CalibrationProblem(
    nn_params=NNParameters(
        nlayers=2, hidden_layer_sizes=[10, 10], activations=[nn.GELU(), nn.GELU()]
    ),
    prob_params=ProblemParameters(data_type=DataType.CUSTOM, tol=1e-9, nepochs=5),
    loss_params=LossParameters(alpha_reg=1e-5),
    phys_params=PhysicalParameters(
        L=L,
        Gamma=GAMMA,
        sigma=SIGMA,
        domain=domain,
        Uref=Uref,
        zref=zref,
    ),
    device=device,
)

# %%
CustomData = torch.tensor(np.genfromtxt(spectra_file, skip_header=1, delimiter=","))
f = CustomData[:, 0]
k1_data_pts = 2 * torch.pi * f / Uref

# %%
DataPoints = [(k1, 1) for k1 in k1_data_pts]
Data = OnePointSpectraDataGenerator(
    data_points=DataPoints,
    k1_data_points=k1_data_pts.data.cpu().numpy(),
).Data


# %%
pb.calibrate(data=Data)

# %%
pb.plot(plt_dynamic=False)

# %%
import matplotlib.pyplot as plt

plt.figure()

plt.plot(pb.loss_history_epochs, "o-", label="Epochs Loss History")
plt.legend()
plt.xlabel("Epoch Number")
plt.ylabel("MSE")
plt.yscale("log")

plt.show()
