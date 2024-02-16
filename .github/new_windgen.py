from pathlib import Path

# from plotly.graph_objs import *

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


# friction_velocity = 0.45
# reference_height = 180.0
# roughness_height = 0.0001

friction_velocity = 2.683479938442173
reference_height = 180.0
roughness_height = 0.75

grid_dimensions = np.array([300.0, 864.0, 576.0]) #* 1/20#* 1/10
grid_levels = np.array([6, 6, 8])

seed = None  # 9000
spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))


Type_Model = "NN"  ### 'Mann', 'VK', 'NN'
nBlocks = 16

path_to_parameters = (
    path / "../docs/source/results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
    if path.name == "examples"
    else path / "../results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
)



gen_drd = GenerateFluctuationField(
    friction_velocity,
    reference_height,
    grid_dimensions,
    grid_levels,
    model=Type_Model,
    path_to_parameters=path_to_parameters,
    seed=seed,
)


for nBlocks in range(1, 15+1): 
    # gen_drd = GenerateFluctuationField(
    #     friction_velocity,
    #     reference_height,
    #     grid_dimensions,
    #     grid_levels,
    #     model=Type_Model,
    #     path_to_parameters=path_to_parameters,
    #     seed=seed,
    # )

    fluctuation_field_drd = gen_drd.generate(1)

    sd = np.sqrt(np.mean(fluctuation_field_drd**2))
    fluctuation_field_drd = fluctuation_field_drd / sd 
    fluctuation_field_drd *= 4.26

    log_law = lambda z, z_0, u_ast: u_ast * np.log(z/z_0+1.0)/0.41
    z = np.linspace(0.0,grid_dimensions[2], 2**(grid_levels[2])+1)

    mean_profile_z = log_law(z, roughness_height, friction_velocity)

    mean_profile = np.zeros_like(fluctuation_field_drd)
    mean_profile[...,0] = np.tile(mean_profile_z.T, (mean_profile.shape[0], mean_profile.shape[1], 1))

    # wind_field = mean_profile
    fluctuation_field_drd += mean_profile

    fluctuation_field_drd *= 40/63

    wind_field_vtk = tuple([np.copy(fluctuation_field_drd[...,i], order='C') for i in range(3)])

    cellData = {'grid': np.zeros_like(fluctuation_field_drd[...,0]), 'wind': wind_field_vtk}
    # .hl import imageToVTK

    from pyevtk.hl import imageToVTK

    FileName = f"dat/block_{nBlocks}"

    imageToVTK(FileName, cellData = cellData, spacing=spacing)


    print(f"generated blocks for {nBlocks}")


    print("saved")