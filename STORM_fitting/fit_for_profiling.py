"""Profiling the data fit for custom data."""

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

device = "cpu"

domain = torch.logspace(-1, 3, 40)

L = 70
Gamma = 3.7
sigma = 0.04
Uref = 21
zref = 1


pb = CalibrationProblem(
    nn_params=NNParameters(nlayers=2, hidden_layer_sizes=[10, 10], activations=[nn.ReLU(), nn.ReLU()]),
    prob_params=ProblemParameters(data_type=DataType.CUSTOM, tol=1e-9, nepochs=5, learn_nu=True),
    loss_params=LossParameters(alpha_pen2=1.0, beta_reg=1e-5),
    phys_params=PhysicalParameters(
        L=L,
        Gamma=Gamma,
        sigma=sigma,
        domain=domain,
        Uref=Uref,
        zref=zref,
    ),
    logging_directory="runs/custom_data",
    device=device,
)

spectra_file = Path("data_cleaned/cleaned_weighted_spectra_quarter.dat")
CustomData = torch.tensor(np.genfromtxt(spectra_file, skip_header=1, delimiter=","))


f = CustomData[:, 0]
k1_data_pts = 2 * torch.pi * f / Uref

gen = OnePointSpectraDataGenerator(
    zref=zref,
    data_points=k1_data_pts,
    data_type=DataType.CUSTOM,
    spectra_file=spectra_file,
    k1_data_points=k1_data_pts.data.cpu().numpy(),
)


Data = gen.Data

optimal_parameters = pb.calibrate(Data)

pb.plot()
