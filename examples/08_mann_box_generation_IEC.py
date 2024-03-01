"""
============================
Fluctuation Field Generation
============================

This example demonstrates the utilities for generating fluctuation fields, which can be either from a pre-trained DRD model, or based on some well-known spectra models. ``DRDMannTurb`` provides several utilities for plotting the resulting fields through Plotly, which can be done in several contexts as well as utilities for saving to VTK for downstream analysis.

"""

#######################################################################################
#   .. centered::
#       This example may take a few seconds to load. Please be patient,
#       Plotly requires some time to render 3D graphics.
#

#######################################################################################
# Import packages
# ---------------
#
# First, we import the packages we need for this example.
from pathlib import Path

import numpy as np
import torch

from drdmannturb.fluctuation_generation import (
    plot_velocity_components,  # utility function for plotting each velocity component in the field, not used in this example
)
from drdmannturb.fluctuation_generation import (
    GenerateFluctuationField,
    plot_velocity_magnitude,
)

path = Path().resolve()

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

#######################################################################################
# Setting Physical Parameters
# ---------------------------
# Here, we set the physical parameters of the environment in which the fluctuation field is generated: the friction velocity :math:`u_* = 0.45`, roughness height :math:`z_0=0.0001` and reference height of :math:`180`.
# The physical domain is determined by dimensions in 3D as well as the discretization size (grid levels) in each dimension.
z0 = 0.02
zref = 90
uref = 11.4
ustar = uref * 0.41 / np.log(zref / z0)
plexp = 0.2  # power law exponent
windprofiletype = "PL"  # choosing power law, use log with "LOG" here instead

L = 0.593 * zref
Gamma = 3.89
sigma = 0.052

Lx = 720
Ly = 64
Lz = 64

nBlocks = 3
grid_dimensions = np.array([Lx / 4, Ly, Lz])

grid_levels = np.array([6, 4, 4])

seed = None

#######################################################################################
# Generating Fluctuation Field from Mann Model
# --------------------------------------------
# Fluctuation fields are generated block-by-block, rather than over the domain entirely. Please see section V, B of the original DRD paper for further discussion. Here, we will use 4 blocks.

Type_Model = "Mann"  ### 'Mann', 'VK', 'NN'

#######################################################################################
# Physical Parameters from Kaimal Spectrum
# ----------------------------------------
# The Mann model requires three parameters, length scale, time scale, and spectrum amplitude scale, which we take from the Kaimal spectrum.
#
gen_mann = GenerateFluctuationField(
    ustar,
    zref,
    grid_dimensions,
    grid_levels,
    length_scale=L,
    time_scale=Gamma,
    energy_spectrum_scale=sigma,
    model=Type_Model,
    seed=seed,
)

fluctuation_field_mann = gen_mann.generate(
    nBlocks, zref, uref, z0, windprofiletype, plexp
)

#######################################################################################
# Scaling of the field (normalization)
# ------------------------------------
# The generated fluctuation field is normalized and scaled by the power law profile
#
# .. math:: \left\langle U_1(z)\right\rangle= u_* \left( \frac{z}{z_{\text{ref}}} \right)^\alpha
#
# where :math:`u_*` is the friction velocity and :math:`z_{\text{ref}}` is the reference height.
#

spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))

fig_magnitude_mann = plot_velocity_magnitude(
    spacing, fluctuation_field_mann, transparent=True
)

# this is a Plotly figure, which can be visualized with the ``.show()`` method in different contexts. While these utilities
# may be useful for quick visualization, we recommend using Paraview to visualize higher resolution output. We will cover
# saving to a portable VTK format further in this example.

fig_magnitude_mann  # .show("browser"), or for specific browser, use .show("firefox")


#######################################################################################
# Evaluating Divergence Properties and Plotting
# ---------------------------------------------
# ``DRDMannTurb`` provides utilities for computing the divergence of the resulting fluctuation field as well as
# visualizing results. At the continuum level, the DRD model should yield an approximately divergence-free fluctuation
# field, which we observe to within a reasonable tolerance. Also, the divergence is expected to decrease as the
#  resolution of the fluctuation field is improved.
spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))

gen_mann.evaluate_divergence(spacing, fluctuation_field_mann).max()


#######################################################################################
# Saving Generated Fluctuation Field as VTK
# -----------------------------------------
# For higher resolution fluctuation fields, we suggest using Paraview. To transfer the generated data
# from our package, we provide the ``.save_to_vtk()`` method.
filename = str(
    path / "./outputs/IEC_simple"
    if path.name == "examples"
    else path / "./outputs/IEC_simple"
)

gen_mann.save_to_vtk(filename)
