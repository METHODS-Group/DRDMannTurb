"""
==================================================
Adding Regularization and Penalty Terms to Fitting
==================================================

"""
import numpy as np
import torch
import torch.nn as nn

from drdmannturb.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)

# %%
from drdmannturb.spectra_fitting import CalibrationProblem, OnePointSpectraDataGenerator

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
        nlayers=2,
        hidden_layer_sizes=[10, 10],
        activations=[nn.GELU(), nn.GELU()],
    ),
    prob_params=ProblemParameters(nepochs=5),
    loss_params=LossParameters(alpha_pen2=1.0, alpha_pen1=1.0e-5, beta_reg=2e-4),
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
# .. image:: /images/test.png
from pathlib import Path

print(Path().resolve())

# from IPython import get_ipython
# import IPython
# ip = IPython.get_ipython()
#
# if get_ipython():
# if ip:
# ip.magic(u'load_ext tensorboard')
# ip.magic(u'tensorboard --logdir runs')

# %load_ext tensorboard
# %tensorboard --logdir runs
# import matplotlib.pyplot as plt

# plt.figure()

# plt.plot(pb.loss_history_epochs, "o-", label="Epochs Loss History")
# plt.legend()
# plt.xlabel("Epoch Number")
# plt.ylabel("MSE")
# plt.yscale("log")

# plt.show()
