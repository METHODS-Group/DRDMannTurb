"""
==================
Synthetic Data Fit
==================

The IEC-recommended spectral tensor model is calibrated to fit the Kaimal spectra. There are three free parameters: :math:`L, T, C`, which have been precomputed in `Mann's original work <https://www.sciencedirect.com/science/article/pii/S0266892097000362>` to be :math:`L=0.59, T=3.9, C=3.2`, which will be used to compare against a DRD model fit. In this example, the exponent :math:`\\nu=-\\frac{1}{3}` is fixed so that :math:`\\tau(\\boldsymbol{k})` matches the slow of :math:`\\tau^{IEC}` for :math:`k \\rightarrow 0`. 

The following example is also discussed in the `original DRD paper <https://arxiv.org/abs/2107.11046>`. 
"""

#######################################################################################
# Import packages
# ---------------
#
# First, we import the packages we need for this example.

import torch
import torch.nn as nn

from drdmannturb import EddyLifetimeType
from drdmannturb.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)
from drdmannturb.spectra_fitting import CalibrationProblem, OnePointSpectraDataGenerator

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

#######################################################################################
# %%
# Scales associated with Kaimal spectrum
L = 0.59
Gamma = 3.9
sigma = 3.2

# Reference velocity
Uref = 21.0
# We consider the range :math:`\mathcal{D} =[0.1, 100]` and sample the data points :math:`f_j \in \mathcal{D}` using a logarithmic grid of :math:`20`` nodes.
domain = torch.logspace(-1, 2, 20)

#######################################################################################
# %%
# We use a neural network consisting of two layers with :math:`10` neurons each, connected by a ReLU activation function. The parameters determining the network architecture can conveniently be set through the ``NNParameters`` dataclass. Using an MSE loss, a second-order derivative penalty term with weight :math:`\alpha_2 = 1`, and a network parameter regularization term with weight :math:`\beta=10^{-5}`, we train the network for :math:`10` epochs.
pb = CalibrationProblem(
    nn_params=NNParameters(
        nlayers=2,
        hidden_layer_sizes=[10, 10],
        activations=[nn.ReLU(), nn.ReLU()],
    ),
    prob_params=ProblemParameters(
        nepochs=10, learn_nu=False, eddy_lifetime=EddyLifetimeType.TAUNET
    ),
    loss_params=LossParameters(alpha_pen2=1.0, beta_reg=1.0e-5),
    phys_params=PhysicalParameters(
        L=L, Gamma=Gamma, sigma=sigma, Uref=Uref, domain=domain
    ),
    logging_directory="runs/synthetic_fit",
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
pb.plot()
# %%

pb.plot_losses(run_number=0)
# %% [markdown]
# ### Save Model with Problem Metadata

# %%
pb.save_model("../results/")

# %% [markdown]
# ### Loading Model and Problem Metadata

# %%
import pickle

# TODO: fix this data load and parameter equivalence check
path_to_parameters = "../results/EddyLifetimeType.TAUNET_DataType.KAIMAL.pkl"

with open(path_to_parameters, "rb") as file:
    (
        nn_params,
        prob_params,
        loss_params,
        phys_params,
        model_params,
    ) = pickle.load(file)

# %% [markdown]
# ### Recovering Old Model Configuration and Old Parameters

# %%
pb_new = CalibrationProblem(
    nn_params=nn_params,
    prob_params=prob_params,
    loss_params=loss_params,
    phys_params=phys_params,
    device=device,
)

pb_new.parameters = model_params

import numpy as np

assert np.ma.allequal(pb.parameters, pb_new.parameters)
