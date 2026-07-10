"""
======================================================
Example 5: Fluctuation Field Generation from DRD Model
======================================================

This example demonstrates generating a fluctuation field from a trained DRD model:
we fit a neural-network-based eddy lifetime model to the Kaimal spectra (as in Example 2),
save the trained model to disk, reload it, and use it to drive fluctuation field generation.
``DRDMannTurb`` provides several utilities for plotting the resulting fields through Plotly
as well as utilities for saving to VTK for downstream analysis (e.g. in ParaView, or as
inlet boundary conditions for CFD solvers).
"""  # noqa

#######################################################################################
#   .. centered::
#       This example may take a few seconds to load. Please be patient if using
#       Plotly, as it requires some time to render 3D graphics.
#

#######################################################################################
# Import packages
# ---------------
#
# First, we import the packages we need for this example.
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import drdmannturb as drdmt
from drdmannturb.fluctuation_generation import (
    FluctuationFieldGenerator,
    NNCovariance,
    plot_velocity_magnitude,
)
from drdmannturb.spectra_fitting import CalibrationProblem
from drdmannturb.spectra_fitting import spectral_tensor_models as stm

path = Path().resolve()

#######################################################################################
# Setting Physical Parameters
# ---------------------------
# Here, we set the physical parameters of the environment in which the synthetic wind
# field is generated: the reference height and velocity, the roughness height, and the
# friction velocity, matching the Kaimal spectrum setting of Example 2.

z0 = 0.02  # roughness height
zref = 90.0  # reference height
uref = 11.4  # reference velocity
ustar = 0.556  # friction velocity
windprofiletype = "LOG"  # choosing log law, use power law with "PL" here instead

# Initial guesses for the model scales, from the IEC-recommended Mann model parameters
L = 0.593 * zref  # length scale
Gamma = 3.89  # time scale
sigma = 3.2 * ustar**2.0 / zref ** (2.0 / 3.0)  # magnitude (σ = αϵ^{2/3})

#######################################################################################
# Fitting a DRD Model
# -------------------
# We generate synthetic Kaimal spectra data and fit a DRD model with a neural-network
# eddy lifetime function, exactly as in Example 2 (see the discussion there for details).

k1 = torch.logspace(-1, 2, 20, dtype=torch.float64) / zref

kaimal_data = drdmt.spectra_fitting.data_generator.generate_kaimal_spectra(
    k1=k1,
    zref=zref,
    ustar=ustar,
)

model = stm.RDT_SpectralTensor(
    eddy_lifetime_model=stm.TauNet_ELT(
        taunet=drdmt.TauNet(n_layers=2, hidden_layer_sizes=[10, 10], activations=[nn.ReLU(), nn.ReLU()])
    ),
    energy_spectrum_model=stm.VonKarman_ESM(),
    L_init=L,
    gamma_init=Gamma,
    sigma_init=sigma,
)

pb = CalibrationProblem(
    model=model,
    data=kaimal_data,
    loss_params=drdmt.LossParameters(alpha_pen2=1.0, beta_reg=1.0e-5),
    output_directory=path / "outputs",
)

pb.calibrate(
    optimizer_class=torch.optim.LBFGS,
    optimizer_kwargs={
        "line_search_fn": "strong_wolfe",
        "max_iter": 20,
        "history_size": 20,
    },
    lr=1.0,
    max_epochs=10,
    tol=1e-6,
)

#######################################################################################
# Saving and Loading the Trained Model
# ------------------------------------
# The trained model is saved to a ``.pt`` checkpoint, which stores the network weights,
# the learned scales, and the calibration scaling factors. It can then be reloaded in a
# different session — the typical workflow separates the (expensive) fitting stage from
# field generation.

checkpoint_path = pb.save_model(path / "outputs" / "kaimal_drd_model.pt")

drd_model = stm.RDT_SpectralTensor.load_model(checkpoint_path)

print(f"Learned scales: L = {torch.exp(drd_model.log_L).item():.3f}, ", end="")
print(f"Gamma = {torch.exp(drd_model.log_gamma).item():.3f}, ", end="")
print(f"sigma = {torch.exp(drd_model.log_sigma).item():.3f}")

#######################################################################################
# Fluctuation Field Generation from the DRD Model
# -----------------------------------------------
# The ``NNCovariance`` kernel evaluates the trained model's spectral tensor during
# field generation. Since the model was fit on a non-dimensional frequency domain
# (:math:`n = k_1 z_{\text{ref}} / 2\pi`, further scaled during calibration), we
# provide the map back to physical wavevectors via ``length_scale``, and back to
# physical spectral amplitudes via ``amplitude``.
#
# The field generation process is summarized by the following diagram of a 2D domain
# (a transversal cross-section of a 3D turbulence block):
#
# .. image:: https://github.com/METHODS-Group/DRDMannTurb/blob/main/paper/fluct_gen_box_by_box.png?raw=true
#
# A continuous wind field is generated block-by-block where noise is copied from the end
# of one block to the start of the next. Turbulent fluctuations are recomputed
# block-by-block using the partially shared noise. This diagram is from
# `Keith, Khristenko, Wohlmuth (2021) <https://arxiv.org/pdf/2107.11046.pdf>`_, please
# see the discussion therein for further details.

scales = drd_model.calibration_scales  # recorded by CalibrationProblem.save_model

cov_drd = NNCovariance(
    spectral_tensor_model=drd_model,
    length_scale=zref / (2 * np.pi * scales["freq_scale"]),
    amplitude=ustar**2 * scales["spectral_scale"],
)

grid_dimensions = np.array([720.0 / 4, 64.0, 64.0])
grid_levels = np.array([6, 4, 4])
nBlocks = 2
seed = None

gen_drd = FluctuationFieldGenerator(
    grid_dimensions,
    grid_levels,
    cov_drd,
    seed=seed,
)

#######################################################################################
# Adding the mean velocity profile
# --------------------------------
# The mean velocity profile follows the logarithmic profile
#
# .. math:: \left\langle U_1(z)\right\rangle= U_{\text{ref}} \frac{\ln \left( \frac{z}{z_0} + 1 \right)}
#                {\ln \left( \frac{z_{\text{ref}}}{z_0} \right)}
#
# where :math:`U_{\text{ref}}` is the reference velocity, :math:`z_0` is the roughness height, and
# :math:`z_{\text{ref}}` is the reference height.

fluctuation_field_drd = gen_drd.generate(nBlocks, zref, uref, z0, windprofiletype)

#######################################################################################
# Evaluating Divergence Properties and Plotting
# ---------------------------------------------
# ``DRDMannTurb`` provides utilities for computing the divergence of the resulting wind
# field as well as visualizing results. At the continuum level, the DRD model should
# yield an approximately divergence-free wind field, which we observe to within a
# reasonable tolerance. Also, the divergence is expected to decrease as the resolution
# of the wind field is improved.
spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))

print(f"Max pointwise divergence: {gen_drd.evaluate_divergence(spacing, fluctuation_field_drd).max():.4f}")

#######################################################################################
# We now visualize the output wind field.
fig_magnitude_drd = plot_velocity_magnitude(spacing, fluctuation_field_drd, transparent=True)

# this is a Plotly figure, which can be visualized with the ``.show()`` method in different contexts.
fig_magnitude_drd  # .show("browser"), or for specific browser, use .show("firefox")

#######################################################################################
# Saving Generated Fluctuation Field as VTK
# -----------------------------------------
# For higher resolution wind fields, we suggest using ParaView. To transfer the generated
# data from our package, we provide the ``.save_to_vtk()`` method.
filename = str(path / "outputs" / "fluctuation_drd")

gen_drd.save_to_vtk(filename)
