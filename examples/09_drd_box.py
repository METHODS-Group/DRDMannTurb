"""
===========================================
Fluctuation Field Generation from DRD Model
===========================================

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
# Here, we set the physical parameters of the environment in which the fluctuation field is generated.
# The physical domain is determined by dimensions in 3D as well as the discretization size (grid levels) in each dimension.

z0 = 0.02
zref = 90
uref = 11.4
ustar = uref * 0.41 / np.log(zref / z0)
windprofiletype = "LOG"  # choosing log law, use power law with "PL" here instead

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
# Fluctuation Field Generation from Pre-Trained DRD Model
# -------------------------------------------------------
# We now generate a similar fluctuation field in the same physical setting and domain but using a pre-trained DRD model. This model is the result of
# fitting the Mann model with a Kaimal spectrum, showcased in an earlier example, so we anticipate the resulting fluctuation fields to be similar. Note
# that since DRD models learn the scales, these are taken from the saved object, which has these values as parameters.
# The field generation process can be summarized by the following diagram of a 2D domain (a transversal cross-section of a 3D turbulence block).
#
# .. image:: https://github.com/METHODS-Group/DRDMannTurb/blob/main/paper/fluct_gen_box_by_box.png?raw=true
#
# A continuous wind field is generated block-by-block where noise is being copied from the end of one block to the start of the next block. Turbulent fluctuations are recomputed block-by-block using the partially shared noise. Common Gaussian noise is used in the overlapping domains. This diagram is from `Keith, Khristenko, Wohlmuth (2021) <https://arxiv.org/pdf/2107.11046.pdf>`_, please see the discussion therein for further details.
path_to_parameters = (
    path / "../docs/source/results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
    if path.name == "examples"
    else path / "../results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
)

Type_Model = "NN"  ### 'Mann', 'VK', 'NN'
nBlocks = 2

gen_drd = GenerateFluctuationField(
    ustar,
    zref,
    grid_dimensions,
    grid_levels,
    length_scale=L,
    time_scale=Gamma,
    energy_spectrum_scale=sigma,
    model=Type_Model,
    path_to_parameters=path_to_parameters,
    seed=seed,
)

#######################################################################################
# Adding the mean velocity profile
# ------------------------------------
# The mean velocity profile follows the power law profile
#
# .. math:: \left\langle U_1(z)\right\rangle= U_{\text{ref}} \frac{\ln \left( \frac{z}{z_0} + 1 \right)}{\ln \left( \frac{z_{\text{ref}}}{z_0} \right)}
#
# where :math:`U_{\text{ref}}` is the reference velocity, :math:`z_0` is the roughness height, and :math:`z_{\text{ref}}` is the reference height.
#


fluctuation_field_drd = gen_drd.generate(nBlocks, zref, uref, z0, windprofiletype)


#######################################################################################
# Evaluating Divergence Properties and Plotting
# ---------------------------------------------
# ``DRDMannTurb`` provides utilities for computing the divergence of the resulting fluctuation field as well as
# visualizing results. At the continuum level, the DRD model should yield an approximately divergence-free fluctuation
# field, which we observe to within a reasonable tolerance. Also, the divergence is expected to decrease as the
#  resolution of the fluctuation field is improved.
spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))

gen_drd.evaluate_divergence(spacing, fluctuation_field_drd).max()

#######################################################################################
# We now visualize the output fluctuation field.
fig_magnitude_drd = plot_velocity_magnitude(
    spacing, fluctuation_field_drd, transparent=True
)

# this is a Plotly figure, which can be visualized with the ``.show()`` method in different contexts.
fig_magnitude_drd  # .show("browser"), or for specific browser, use .show("firefox")

#######################################################################################
# Saving Generated Fluctuation Field as VTK
# -----------------------------------------
# For higher resolution fluctuation fields, we suggest using Paraview. To transfer the generated data
# from our package, we provide the ``.save_to_vtk()`` method.
filename = str(
    path / "../docs/source/results/fluctuation_drd"
    if path.name == "examples"
    else path / "../results/fluctuation_drd"
)

gen_drd.save_to_vtk(filename)
