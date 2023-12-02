"""
====================================
Mann Eddy Lifetime Linear Regression
====================================

This example demonstrates the a simple configuration of ``DRDMannTurb`` to spectra fitting while using a linear approximation to the Mann eddy lifetime function under the Kaimal one-point spectra.

For reference, the full Mann eddy lifetime function is given by 

.. math::

    \tau^{\mathrm{IEC}}(k)=\frac{(k L)^{-\frac{2}{3}}}{\sqrt{{ }_2 F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-(k L)^{-2}\right)}}

where the hypergeometric function can only be evaluated on the CPU. The purpose of this example is to show how a GPU kernel of a linear approximation (in log-log space) of the Mann eddy lifetime can be generated automatically to speed up tasks that require the GPU. As before, the Kaimal spectra is used for the one-point-spectra model. 

The external API works the same as with other models, but the following may speed up some tasks that rely exclusively on the Mann eddy lifetime function. 
"""
# %%
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


# %%

# Scales associated with Kaimal spectrum
L = 0.59
Gamma = 3.9
sigma = 3.2

domain = torch.logspace(-1, 2, 20)

# %%
pb = CalibrationProblem(
    nn_params=NNParameters(
        nlayers=2,
        hidden_layer_sizes=[10, 10],
        activations=[nn.GELU(), nn.GELU()],
    ),
    prob_params=ProblemParameters(
        nepochs=10, eddy_lifetime=EddyLifetimeType.MANN_APPROX
    ),
    loss_params=LossParameters(),
    phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
    device=device,
)
# %%

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
