"""
This script is used to generate the vts files to be visualized
in Paraview for wind generation gif.
"""

from pathlib import Path

import numpy as np
import torch

from drdmannturb.turbulence_generation import GenerateTurbulenceField, format_wind_field
from drdmannturb import create_grid
import pyevtk


device = "cuda" if torch.cuda.is_available() else "cpu"
# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


# NOTE: The following must be executed in <a specific location>
path = Path().resolve()


path_to_parameters = (
    path / "../docs/source/results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
    if path.name == "examples"
    else path / "../results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
)

# NOTE: Just setting parameters
friction_velocity = 0.45
reference_height = 180.0
roughness_height = 0.0001

grid_dimensions = np.array([1200.0, 864.0, 576.0]) * 1 / 20  # * 1/10
grid_levels = np.array([5, 3, 5])

seed = None  # 9000
spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))

# TODO: Notice the values are lazily hard-coded!
gen_drd = GenerateTurbulenceField(
    friction_velocity=0.45,
    reference_height=180.0,
    grid_dimensions=np.array([1200.0, 864.0, 576.0]) * 1 / 20,
    grid_levels=np.array([7, 5, 7]),
    model="NN",
    path_to_parameters=path_to_parameters,
    blend_num=20,
)

# NOTE: Main loop
for nBlocks in range(1, 10):
    fluctuation_field_drd = gen_drd.generate(1)
    print("saved")

fluctuation_field_drd = gen_drd.normalize(roughness_height, friction_velocity)

X, Y, Z = create_grid(spacing, fluctuation_field_drd.shape)
formatted_wind_field = format_wind_field(fluctuation_field_drd)

wind_magnitude = np.sqrt(
    formatted_wind_field[0] ** 2
    + formatted_wind_field[1] ** 2
    + formatted_wind_field[2] ** 2
)
fname = "attempts/attempt_" + str(nBlocks)
pyevtk.hl.gridToVTK(fname, X, Y, Z, pointData={"vel": wind_magnitude})
