"""
=====================================
Changing MLP Architecture and Fitting
=====================================

"""
import numpy as np
import torch
import torch.nn as nn

from drdmannturb.common import MannEddyLifetime
from drdmannturb.enums import DataType, EddyLifetimeType, PowerSpectraType
from drdmannturb.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)
from drdmannturb.spectra_fitting import CalibrationProblem, OnePointSpectraDataGenerator

# %%


# from drdmannturb.calibration import CalibrationProblem
# from drdmannturb.data_generator import OnePointSpectraDataGenerator
# from drdmannturb.shared.parameters import (
# LossParameters,
# NNParameters,
# PhysicalParameters,
# ProblemParameters,
# )

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

L = 0.59

Gamma = 3.9
sigma = 3.4

domain = torch.logspace(-1, 2, 20)

# %%
pb = CalibrationProblem(
    nn_params=NNParameters(
        nlayers=5,
        hidden_layer_sizes=[5, 10, 20, 10, 5],
        activations=[nn.GELU(), nn.ReLU(), nn.GELU(), nn.ReLU(), nn.GELU()],
    ),
    prob_params=ProblemParameters(nepochs=5, wolfe_iter_count=30),
    loss_params=LossParameters(),
    phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
    device=device,
)

# %%
k1_data_pts = domain
DataPoints = [(k1, 1) for k1 in k1_data_pts]

# %%
Data = OnePointSpectraDataGenerator(data_points=DataPoints).Data

# %%
pb.eval(k1_data_pts)
optimal_parameters = pb.calibrate(data=Data)

# %%
pb.plot(plt_dynamic=False)

# %%
# import matplotlib.pyplot as plt

# plt.figure()

# plt.plot(pb.loss_history_epochs, "o-", label="Epochs Loss History")
# plt.legend()
# plt.xlabel("Epoch Number")
# plt.ylabel("MSE")
# plt.yscale("log")

# plt.show()
