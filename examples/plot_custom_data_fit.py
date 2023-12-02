"""
===============
Custom Data Fit
===============

In this example, we use ``drdmannturb`` to fit a simple neural network model to real-world data.
"""

##############################################################################
# Import packages
# ---------------
#
# First, we import the packages needed for this example, obtain the current
# working directory and dataset path, and choose to use CUDA if it is available.
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from drdmannturb.enums import DataType
from drdmannturb.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)
from drdmannturb.spectra_fitting import CalibrationProblem, OnePointSpectraDataGenerator

path = Path().resolve()

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


spectra_file = (
    path / "../docs/source/data/Spectra.dat"
    if path.name == "examples"
    else path / "../data/Spectra.dat"
)

##############################################################################
# Setting Physical Parameters
# ---------------------------
# Here, we define our charateristic scales :math:`L, \Gamma, \sigma`, the
# log-scale domain, and the reference height `zref` and velocity `Uref`.

domain = torch.logspace(-1, 2, 20)

L = 70
GAMMA = 3.7
SIGMA = 0.04

Uref = 21
zref = 1

pb = CalibrationProblem(
    nn_params=NNParameters(
        nlayers=2, hidden_layer_sizes=[10, 10], activations=[nn.ReLU(), nn.ReLU()]
    ),
    prob_params=ProblemParameters(data_type=DataType.CUSTOM, tol=1e-9, nepochs=5),
    loss_params=LossParameters(alpha_pen2=1.0, beta_reg=1e-5),
    phys_params=PhysicalParameters(
        L=L,
        Gamma=GAMMA,
        sigma=SIGMA,
        domain=domain,
        Uref=Uref,
        zref=zref,
    ),
    logging_directory="runs/custom_data",
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
    data_type=DataType.CUSTOM,
    spectra_file=spectra_file,
    k1_data_points=k1_data_pts.data.cpu().numpy(),
).Data


# %%
optimal_parameters = pb.calibrate(data=Data)

# %%
pb.plot()

# The training logs can be accessed from the logging directory
# with Tensorboard utilities, but we also provide a simple internal utility for a single
# training log plot.
pb.plot_losses(run_number=0)
