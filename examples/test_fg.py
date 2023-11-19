"""
============================================
Wind Field Generation from Trained DRD Model
============================================

This example demonstrates the utilities for generating wind fields from a pre-trained DRD model. DRDMannTurb provides several utilities for plotting the resulting fields as well, though the resulting fields can be readily saved to VTK to be visualized in Paraview. 

.. warning:: 
    This example may take some time to load. Please be patient, Plotly requires some time to pre-render. 
"""

# %%

# %%
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from drdmannturb.fluctuation_generation import (
    GenerateFluctuationField,
    plot_velocity_components,
    plot_velocity_magnitude,
)

path = Path().resolve()

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


##########################################
# Having set-up the necessary parameters for the domain, we now generate the wind field.
# %%
Type_Model = "Mann"  ### 'FPDE_RDT', 'Mann', 'VK', 'NN'
nBlocks = 3

normalize = True
friction_velocity = 0.45  # 2.683479938442173  # 0.45
reference_height = 180.0
roughness_height = 0.0001  # 0.75  # 0.0001
grid_dimensions = np.array([1200.0, 864.0, 576.0])
grid_levels = np.array([5, 3, 5])
seed = None  # 9000

path_to_parameters = (
    path / "../docs/source/results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
    if path.name == "examples"
    else path / "../results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
)

##########################################
# Wind generation
# ---------------

##########################################
# %%
wind = GenerateFluctuationField(
    friction_velocity,
    reference_height,
    grid_dimensions,
    grid_levels,
    model=Type_Model,
    path_to_parameters=path_to_parameters,
    seed=seed,
    # laplace=True
)
for _ in range(nBlocks):
    wind()
    wind_field = wind.total_wind

# TODO: these should be moved into a GenerateWind method...?
##########################################
# Scaling of the field (normalization)
# ------------------------------------

##########################################
# %%

if normalize == True:
    sd = np.sqrt(np.mean(wind_field**2))
    wind_field = wind_field / sd

log_law = lambda z, z_0, u_ast: u_ast * np.log(z / z_0 + 1.0) / 0.41

z = np.linspace(0.0, grid_dimensions[2], 2 ** (grid_levels[2]) + 1)
mean_profile_z = log_law(z, roughness_height, friction_velocity)

mean_profile = np.zeros_like(wind_field)
mean_profile[..., 0] = np.tile(
    mean_profile_z.T, (mean_profile.shape[0], mean_profile.shape[1], 1)
)

wind_field += mean_profile

# %%
spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))

wind_field_vtk = tuple([np.copy(wind_field[..., i], order="C") for i in range(3)])

cellData = {"grid": np.zeros_like(wind_field[..., 0]), "wind": wind_field_vtk}

# %%
from pyevtk.hl import imageToVTK

filename = str(
    path / "../docs/source/results/fluctuation_simple"
    if path.name == "examples"
    else path / "../results/fluctuation_simple"
)
imageToVTK(filename, cellData=cellData, spacing=spacing)

# %%
# fig_components = plot_velocity_components(spacing, wind_field)

# fig_components

# %%
fig_magnitude = plot_velocity_magnitude(spacing, wind_field)

fig_magnitude.show

# %%
