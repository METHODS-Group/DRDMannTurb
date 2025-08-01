r"""
===============================
Example 1: Basic Mann Model Fit
===============================

This example demonstrates fitting the Mann model eddy lifetime function to the Kaimal one-point spectra
and cross-spectra.

For reference, the Mann eddy lifetime function is given by

.. math::

    \tau^{\mathrm{Mann}}(k)=\frac{(k L)^{-\frac{2}{3}}}{\sqrt{{ }_2
    F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-(k L)^{-2}\right)}}\,.

This set of models it widely used for flat, homogeneous terrains.

``drdmannturb`` can also be used directly to generate the corresponding 3D turbulence field, as demonstrated in
Examples 8 and 9.

"""  # noqa: D205, D400

#######################################################################################
# Import packages
# ---------------
#
# First, we import the packages needed for this example.

import torch

import drdmannturb as drdmt
from drdmannturb.spectra_fitting import CalibrationProblem
from drdmannturb.spectra_fitting import spectral_tensor_models as stm

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

#######################################################################################
# We'll use the Kaimal spectrum for this example; this first segment sets up a few
# physical parameters we'll need to specify the problem.

zref = 40  # reference height
ustar = 1.773  # friction velocity

# Intitial parameter guesses for fitting the Mann model
L = 0.59 * zref  # length scale
Gamma = 3.9  # time scale
sigma = 3.2 * ustar**2.0 / zref ** (2.0 / 3.0)  # magnitude (σ = αϵ^{2/3})

print(f"Physical Parameters: {L, Gamma, sigma}")

##############################################################################
# Data Generation
# ---------------
# First, we're going to use the Kaimal spectra data generation function to
# build a synthetic dataset.

k1 = torch.logspace(-1, 2, 20) / zref

kaimal_data = drdmt.spectra_fitting.data_generator.generate_kaimal_spectra(
    k1=k1,
    zref=zref,
    ustar=ustar,
)


##############################################################################
# Model Construction
# ------------------
# Models in ``DRDMannTurb`` are constructed via a ``SpectralTensorModel``, which
# requires an ``EddyLifetimeModel`` and an ``EnergySpectrumModel``.
#
# In this example, we use the ``RDT_SpectralTensor`` (Rapid Distortion Theory) with
# the ``Mann_ELT`` (Mann model eddy lifetime function) and the
# ``VonKarman_ESM`` (standard von Karman energy spectrum).

model = stm.RDT_SpectralTensor(
    eddy_lifetime_model=stm.Mann_ELT(),
    energy_spectrum_model=stm.VonKarman_ESM(),
    L_init=L,
    gamma_init=Gamma,
    sigma_init=sigma,
)

##############################################################################
# ``CalibrationProblem`` Construction
# -----------------------------------
#

pb = CalibrationProblem(
    model=model,
    data=kaimal_data,
    fit_coherence=False,
    loss_params=drdmt.LossParameters(),  # Default values are used here.
    integration_params=drdmt.IntegrationParameters(),
    logging_directory="runs/mann_elt_fit",
    output_directory="outputs",
    device="cpu",
)

##############################################################################
# The model is now fit to the provided spectra given in ``Data``.
#
# Note that the Mann eddy lifetime function relies on evaluating a hypergeometric function,
# which only has a CPU implementation through ``Scipy``; cf. Example 7.
#
# Having the necessary components, the model is "calibrated" (fit) to the provided spectra.

pb.calibrate(
    optimizer_class=torch.optim.LBFGS,
    optimizer_kwargs={
        "line_search_fn": "strong_wolfe",
        "max_iter": 20,
        "history_size": 20,
    },
    lr=1.0,
    max_epochs=20,
    tol=1e-6,
)

##############################################################################
# We conclude by printing the optimized parameters and generating a plot showing the
# fit to the Kaimal spectra.
pb.plot()
