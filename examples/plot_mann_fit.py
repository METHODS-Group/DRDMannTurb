"""
===================
Pure Mann Model Fit
===================

This example demonstrates the usage of ``drdmannturb``'s pure Mann model implementation.
"""

##############################################################################
# Import packages
# ---------------
# First, we import the packages we need for this example. Additionally, we set
# a handful of necessary physical constants, including the length scale and domain.
# Moreover, we choose to use CUDA if it is available.
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

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

# Scales associated with Kaimal spectrum
L = 0.59
Gamma = 3.9
sigma = 3.2

domain = torch.logspace(-1, 2, 20)

##############################################################################
# ``CalibrationProblem`` Construction
# -----------------------------------
# The following cell defines the ``CalibrationProblem`` using default values
# for the ``NNParameters`` and ``LossParameters`` dataclasses. Importantly,
# these lines may be elided, but are included here for clarity.
pb = CalibrationProblem(
    nn_params=NNParameters(),
    prob_params=ProblemParameters(eddy_lifetime=EddyLifetimeType.MANN, nepochs=2),
    loss_params=LossParameters(),
    phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
    device=device,
)

##############################################################################
# Data Generation
# ---------------
# The following cell generates the dataset required for calibration.
#
# The first two lines are required to construct the spatial grid of points.
# Specifically, ``DataPoints`` is a list of tuples of the observed spectra data
# points at each of the :math:`k_1`
# coordinates and the reference height (in our case, this is just :math:`1`).
#
# Lastly, we collect ``Data = (<data points>, <data values>)`` to be used in calibration.

k1_data_pts = domain
DataPoints = [(k1, 1) for k1 in k1_data_pts]

Data = OnePointSpectraDataGenerator(data_points=DataPoints).Data

##############################################################################
# %%
#
# Having the necessary components, the model is "fit" and we conclude with a plot.
optimal_parameters = pb.calibrate(data=Data)

# %%
pb.plot()
