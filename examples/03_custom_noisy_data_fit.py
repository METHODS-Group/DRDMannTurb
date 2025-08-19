r"""
==========================
Example 5: Custom Data Fit
==========================

In this example, we use ``drdmannturb`` to fit a simple neural network model to real-world
data without any preprocessing. This involves data that are observed in the real world,
specifically near a North Sea wind turbine farm. The physical parameters are determined
from those measurements. Additionally, the :math:`\nu` parameter is also learned.
is learned for the rational function for :math:`\tau` given by

.. math::
        \tau(\boldsymbol{k})=\frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}{\left(1+|\boldsymbol{a}|^2\right)^{\nu / 2}},
        \quad \boldsymbol{a}=\boldsymbol{a}(\boldsymbol{k}).

"""  # noqa: D205, D400

##############################################################################
# Import packages
# ---------------
#
# First, we import the packages needed for this example, obtain the current
# working directory and dataset path, and choose to use CUDA if it is available.
from pathlib import Path

import torch
import torch.nn as nn

import drdmannturb as drdmt
from drdmannturb.spectra_fitting import CalibrationProblem
from drdmannturb.spectra_fitting import spectral_tensor_models as stm

##############################################################################
# Setting Physical Parameters
# ---------------------------
# Once again, this first block is just setting up several physical parameters we'll need.

path = Path().resolve()
# spectra_file = path / "./inputs/Spectra.dat" if path.name == "examples" else path / "../data/Spectra.dat"

spectra_file = path / "inputs" / "ex03_ops.csv"

domain = torch.logspace(-1, 3, 40)

L = 70  # length scale
Gamma = 3.7  # time scale
sigma = 0.04  # magnitude (σ = αϵ^{2/3})

Uref = 21  # reference velocity
zref = 1  # reference height

#######################################################################################
# Construct the ``CustomDataLoader``
# ------------------------------

data_loader = drdmt.spectra_fitting.data_generator.CustomDataLoader(
    ops_data_file=spectra_file,
)

#######################################################################################
# ``CalibrationProblem`` construction
# -----------------------------------
#
# We'll use a simple neural network consisting of two layers with :math:`10` neurons each,
# connected by a ReLU activation function. The parameters determining the network
# architecture can conveniently be set through the ``NNParameters`` dataclass.
#
# Using the ``ProblemParameters`` dataclass, we indicate the eddy lifetime function
# :math:`\tau` substitution, that we do not intend to learn the exponent :math:`\nu`,
# and that we would like to train for 10 epochs, or until the tolerance ``tol`` loss (0.001 by default),
# whichever is reached first.
#
# Having set our physical parameters above, we need only pass these to the
# ``PhysicalParameters`` dataclass just as is done below.
#
# Lastly, using the ``LossParameters`` dataclass, we introduce a second-order
# derivative penalty term with weight :math:`\alpha_2 = 1` and a
# network parameter regularization term with weight
# :math:`\beta=10^{-5}` to our MSE loss function.
#
# Note that :math:`\nu` is learned here.

model = stm.RDT_SpectralTensor(
    eddy_lifetime_model=stm.TauNet_ELT(
        taunet=drdmt.TauNet(
            n_layers=2,
            hidden_layer_sizes=[10, 10],
            activations=[nn.ReLU(), nn.ReLU()],
            learn_nu=True,  # This makes the rational kernel's :math:`\nu` a learnable parameter.
        )
    ),
    energy_spectrum_model=stm.VonKarman_ESM(),
    L_init=L,
    gamma_init=Gamma,
    sigma_init=sigma,
)


pb = CalibrationProblem(
    data_loader=data_loader,
    model=model,
    loss_params=drdmt.LossParameters(alpha_pen2=1.0, beta_reg=1e-5),
    integration_params=drdmt.IntegrationParameters(),
    logging_directory="runs/custom_data",
    device="cpu",
)


##############################################################################
# Calibration
# -----------
# Now, we fit our model. ``CalibrationProblem.calibrate`` takes the tuple ``Data``
# which we just constructed and performs a typical training loop. The resulting
# fit for :math:`\nu` is close to :math:`\nu \approx - 1/3`, which can be improved
# with further training.

pb.calibrate(
    optimizer_class=torch.optim.LBFGS,
    optimizer_kwargs={
        "line_search_fn": "strong_wolfe",
        "max_iter": 20,
        "history_size": 20,
    },
    lr=1.0,
    max_epochs=5,
    tol=1e-6,
)

##############################################################################
# Plotting
# --------
# ``DRDMannTurb`` offers built-in plotting utilities and Tensorboard integration
# which make visualizing results and various aspects of training performance
# very simple.
#
# The following will plot our fit. As can be seen, the spectra is fairly noisy,
# which suggests that a better fit may be obtained from pre-processing the data, which
# we will explore in the next example.
pb.plot()
