"""
===================================================
Fluctuation Field Generation from Trained DRD Model
===================================================

This example demonstrates the utilities for generating fluctuation fields from a pre-trained DRD model. DRDMannTurb provides several utilities for plotting the resulting fields as well through Plotly. Moreover, the resulting fields can be readily saved to VTK to be visualized in Paraview. 

.. warning:: 
    This example may take a few seconds to load. Please be patient, Plotly requires some time to render 3D graphics. 
"""

# %%

# %%
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
gen = GenerateFluctuationField(
    friction_velocity,
    reference_height,
    grid_dimensions,
    grid_levels,
    length_scale=0.59,
    time_scale=3.9,
    energy_spectrum_scale=3.2,
    model=Type_Model,
    path_to_parameters=path_to_parameters,
    seed=seed,
)
fluctuation_field = gen.generate(nBlocks)

##########################################
# Scaling of the field (normalization)
# ------------------------------------

##########################################
# %%

fluctuation_field = gen.normalize(roughness_height, friction_velocity)

# %%

filename = str(
    path / "../docs/source/results/fluctuation_simple"
    if path.name == "examples"
    else path / "../results/fluctuation_simple"
)
gen.save_to_vtk(filename)


# %%
spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))

fig_magnitude = plot_velocity_magnitude(spacing, fluctuation_field)

fig_magnitude  # .show("browser")

# %%
