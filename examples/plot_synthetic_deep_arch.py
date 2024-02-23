"""
=====================================
Changing MLP Architecture and Fitting
=====================================

This example is nearly identical to the Synthetic Data fit, however we use
a different neural network architecture in hopes of obtaining a better spectra fitting.
The same set-up using the Mann model under the Kaimal spectra is used here as in other synthetic 
data fitting examples. The only difference here is in the neural network architecture. 
Although certain combinations of activation functions, such as ``GELU`` result in considerably
improved spectra fitting and terminal loss values, the resulting eddy lifetime functions are 
usually non-physical. 

See again the `original DRD paper <https://arxiv.org/abs/2107.11046>`_.
"""


#######################################################################################
# Import packages
# ---------------
#
# First, we import the packages we need for this example. Additionally, we choose to use
# CUDA if it is available.
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

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
#######################################################################################
# The following cell sets the necessary physical constants, including the characteristic
# scales for non-dimensionalization, the reference velocity, and the domain.
# We consider the range :math:`\mathcal{D} = [0.1, 100]` and sample the data points
# :math:`f_j \in \mathcal{D}` using a logarithmic grid of :math:`20` nodes.

# Scales associated with Kaimal spectrum
L = 0.59  # length scale
Gamma = 3.9  # time scale
sigma = 3.2  # energy spectrum scale

Uref = 21.0  # reference velocity
zref = 1  # reference height


domain = torch.logspace(-1, 2, 20)

##############################################################################
# %%
# Now, we construct our ``CalibrationProblem``.
#
# Compared to the first Synthetic Fit example, as noted already, we are using
# a more complicated neural network architecture. This time, specifically, our
# network will have 4 layers of width 10, 20, 20, 10 respectively, and we
# use both ``GELU`` and ``RELU`` activations. We have
# prescribed more Wolfe iterations.
# Finally, this task is considerably more difficult than before since the exponent of
# the eddy lifetime function :math:`\nu` is to be learned. Much more training
# may be necessary to obtain a close fit to the eddy lifetime function. Interestingly,
# learning this parameter results in models that more accurately describe the spectra of
# Mann turbulence than using the Mann model itself.

pb = CalibrationProblem(
    nn_params=NNParameters(
        nlayers=4,
        # Specifying the activations is done similarly.
        hidden_layer_sizes=[10, 20, 20, 10],
        activations=[nn.ReLU(), nn.GELU(), nn.GELU(), nn.ReLU()],
    ),
    prob_params=ProblemParameters(nepochs=25, wolfe_iter_count=20),
    loss_params=LossParameters(alpha_pen2=1.0, beta_reg=1.0e-5),
    phys_params=PhysicalParameters(
        L=L, Gamma=Gamma, sigma=sigma, Uref=Uref, domain=domain
    ),
    logging_directory="runs/synthetic_fit_deep_arch",
    device=device,
)

##############################################################################
# Data Generation
# ---------------
# In the following cell, we construct our :math:`k_1` data points grid and
# generate the values. ``Data`` will be a tuple ``(<data points>, <data values>)``.
# It is worth noting that the second element of each tuple in ``DataPoints`` is the
# corresponding reference height, which we have chosen to be uniformly `zref`.
Data = OnePointSpectraDataGenerator(zref=zref, data_points=domain).Data

##############################################################################
# Training
# --------
# Now, we fit our model. ``CalibrationProblem.calibrate()`` takes the tuple ``Data``
# which we just constructed and performs a typical training loop.
optimal_parameters = pb.calibrate(data=Data)

pb.print_calibrated_params()

##############################################################################
# Plotting
# --------
# Lastly, we'll use built-in plotting utilities to see the fit result.
pb.plot()

##############################################################################
# This plots the loss function terms as specified, each multiplied by the
# respective coefficient hyperparameter. The training logs can be accessed from the logging directory
# with Tensorboard utilities, but we also provide a simple internal utility for a single
# training log plot.
pb.plot_losses(run_number=0)
