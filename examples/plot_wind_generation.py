"""
============================
Fluctuation Field Generation
============================

This example demonstrates the utilities for generating fluctuation fields, which can be either from a pre-trained DRD model, or based on some well-known spectra models. ``DRDMannTurb`` provides several utilities for plotting the resulting fields through Plotly, which can be done in several contexts as well as utilities for saving to VTK for downstream analysis. 

.. warning:: 
    This example may take a few seconds to load. Please be patient, Plotly requires some time to render 3D graphics. 
"""

#######################################################################################
# Import packages
# ---------------
#
# First, we import the packages we need for this example.
from pathlib import Path

import numpy as np
import torch

from drdmannturb.turbulence_generation import (  # utility function for plotting each velocity component in the field, not used in this example
    GenerateTurbulenceField,
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
friction_velocity = 0.45
reference_height = 180.0
roughness_height = 0.0001

grid_dimensions = np.array([1200.0, 864.0, 576.0])
grid_levels = np.array([5, 3, 5])

seed = None  # 9000

#######################################################################################
# Generating Fluctuation Field from Mann Model
# --------------------------------------------
# Fluctuation fields are generated block-by-block, rather than over the domain entirely. Please see section V, B of the original DRD paper for further discussion. Here, we will use 3 blocks.

Type_Model = "Mann"  ### 'Mann', 'VK', 'NN'
nBlocks = 3

#######################################################################################
# Physical Parameters from Kaimal Spectrum
# ----------------------------------------
# The Mann model requires three parameters, length scale, time scale, and spectrum amplitude scale, which we take from the Kaimal spectrum.
#
gen_mann = GenerateTurbulenceField(
    friction_velocity,
    reference_height,
    grid_dimensions,
    grid_levels,
    length_scale=0.59,
    time_scale=3.9,
    energy_spectrum_scale=3.2,
    model=Type_Model,
    seed=seed,
)

fluctuation_field_mann = gen_mann.generate(nBlocks)

#######################################################################################
# Scaling of the field (normalization)
# ------------------------------------
# We now normalize and scale the generated fluctuation field so that
#
# .. math:: \left\langle U_1(z)\right\rangle=\frac{u_*}{\kappa} \ln \left(\frac{z}{z_0}+1\right)
#
# where :math:`u_*` is the friction velocity and :math:`z_0` is the roughness height.
#
fluctuation_field_mann = gen_mann.normalize(roughness_height, friction_velocity)

spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))

fig_magnitude_mann = plot_velocity_magnitude(spacing, fluctuation_field_mann)

# this is a Plotly figure, which can be visualized with the ``.show()`` method in different contexts.
fig_magnitude_mann  # .show("browser")

#######################################################################################
# Fluctuation Field Generation from Pre-Trained DRD Model
# -------------------------------------------------------
# We now generate a similar fluctuation field in the same physical setting and domain but using a pre-trained DRD model. This model is the result of
# fitting the Mann model with a Kaimal spectrum, showcased in an earlier example, so we anticipate the resulting fluctuation fields to be similar. Note
# that since DRD models learn the scales, these are taken from the saved object, which has these values as parameters.
# sphinx_gallery_start_ignore
path_to_parameters = (
    path / "../docs/source/results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
    if path.name == "examples"
    else path / "../results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
)
# sphinx_gallery_end_ignore
Type_Model = "NN"  ### 'Mann', 'VK', 'NN'
nBlocks = 3

gen_drd = GenerateTurbulenceField(
    friction_velocity,
    reference_height,
    grid_dimensions,
    grid_levels,
    model=Type_Model,
    path_to_parameters=path_to_parameters,
    seed=seed,
)

fluctuation_field_drd = gen_drd.generate(nBlocks)

fluctuation_field_drd = gen_drd.normalize(roughness_height, friction_velocity)


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
fig_magnitude_drd = plot_velocity_magnitude(spacing, fluctuation_field_drd)

# this is a Plotly figure, which can be visualized with the ``.show()`` method in different contexts.
fig_magnitude_drd  # .show("browser")

#######################################################################################
# Saving Generated Fluctuation Field as VTK
# -----------------------------------------
filename = str(
    path / "../docs/source/results/fluctuation_simple"
    if path.name == "examples"
    else path / "../results/fluctuation_simple"
)

gen_drd.save_to_vtk(filename)
