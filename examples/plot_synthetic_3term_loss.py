"""
==================================================
Adding Regularization and Penalty Terms to Fitting
==================================================

This example is nearly identical to the Synthetic Data fit, however we
use a more sophisticated loss function, introducing now a regularization
term.

See again the `original DRD paper <https://arxiv.org/abs/2107.11046>`_.
"""

#######################################################################################
# Import packages
# ---------------
#
# First, we import the packages we need for this example. Additionally, we choose to use
# CUDA if it is available.

import numpy as np
import torch
import torch.nn as nn

from drdmannturb.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)

from drdmannturb.spectra_fitting import CalibrationProblem, OnePointSpectraDataGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

# Scales associated with Kaimal spectrum
L = 0.59
Gamma = 3.9
sigma = 3.2

Uref = 21.0

domain = torch.logspace(-1, 2, 20)

##############################################################################
# %%
# Now, we construct our ``CalibrationProblem``.
#
# Compared to the first Synthetic Fit example, we are instead using GELU
# activations and will train for fewer epochs. The more interesting difference
# is that we will have activated a first order term in the loss function by passing
# ``alpha_pen1`` a value in the ``LossParameters`` constructor.

pb = CalibrationProblem(
    nn_params=NNParameters(
        nlayers=2,
        hidden_layer_sizes=[10, 10],
        activations=[nn.GELU(), nn.GELU()],
    ),
    prob_params=ProblemParameters(nepochs=5),
    loss_params=LossParameters(alpha_pen2=1.0, alpha_pen1=1.0e-5, beta_reg=2e-4),
    phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
    logging_directory="runs/synthetic_3term",
    device=device,
)

##############################################################################
# %%
# In the following cell, we construct our :math:`k_1` data points grid and
# generate the values. ``Data`` will be a tuple ``(<data points>, <data values>)``.
# It is worth noting that the second element of each tuple in ``DataPoints`` is the
# corresponding reference height, which we have chosen to be uniformly :math:`1`.
k1_data_pts = domain
DataPoints = [(k1, 1) for k1 in k1_data_pts]

Data = OnePointSpectraDataGenerator(data_points=DataPoints).Data

##############################################################################
# %%
# Now, we fit our model. ``CalibrationProblem.calibrate()`` takes the tuple ``Data``
# which we just constructed and performs a typical training loop.
optimal_parameters = pb.calibrate(data=Data)

##############################################################################
# %%
# Lastly, we'll used built-in plotting utilities to see the fit result.
pb.plot()

##############################################################################
# %%
# This plots the loss function terms as specified, each multiplied by the
# respective coefficient hyperparameter.
pb.plot_losses(run_number=0)
