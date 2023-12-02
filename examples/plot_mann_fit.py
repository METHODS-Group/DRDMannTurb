r"""
===================
Pure Mann Model Fit
===================

This example demonstrates the a simple configuration of ``DRDMannTurb`` to spectra fitting while using the Mann model as the eddy lifetime function under the Kaimal one-point spectra.

For reference, the full Mann eddy lifetime function is given by 

.. math::

    \tau^{\mathrm{IEC}}(k)=\frac{(k L)^{-\frac{2}{3}}}{\sqrt{{ }_2 F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-(k L)^{-2}\right)}}

and the Kaimal one-point spectra is defined as in :func:`~drdmannturb.OnePointSpectraDataGenerator.eval_Kaimal`. This set of models has classically been most useful for flat homogeneous terrains. 

Also, the resulting fitting can be used directly to generate a 3D Mann fluctuation field, as demonstrated in our wind generation example. 

"""

#######################################################################################
# Import packages
# ---------------
#
# First, we import the packages we need for this example.

import torch

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

#######################################################################################
# Set up physical parameters and domain associated with the Kaimal spectrum. We perform the spectra fitting over the :math:`k_1` space :math:[10^{{-1}}, 10^2]`
# with 20 points.

# Scales associated with Kaimal spectrum
L = 0.59  # length scale
Gamma = 3.9  # time scale
sigma = 3.2  # energy spectrum scale

domain = torch.logspace(-1, 2, 20)

# %%
pb = CalibrationProblem(
    nn_params=NNParameters(),
    prob_params=ProblemParameters(eddy_lifetime=EddyLifetimeType.MANN, nepochs=2),
    loss_params=LossParameters(),
    phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
    device=device,
)

#%%
k1_data_pts = domain
DataPoints = [(k1, 1) for k1 in k1_data_pts]

# %%
Data = OnePointSpectraDataGenerator(data_points=DataPoints).Data

# %%
pb.eval(k1_data_pts)
optimal_parameters = pb.calibrate(data=Data)

# %%
pb.plot()
