"""
================================================
Example 10: 2D Low Frequency + 3D High Frequency
================================================
"""

import numpy as np

from drdmannturb.fluctuation_generation.fluctuation_field_generator import FluctuationFieldGenerator

# from drdmannturb.fluctuation_generation.low_frequency.fluctuation_field_generator import LowFreqGenerator

##################################
# 2d parameters
config_2d = {
    "sigma2": 2.0,
    "L_2d": 15_000.0,
    "z_i": 500.0,
    "psi": np.deg2rad(43.0),
    "L1_factor": 8,
    "L2_factor": 8,
    "N1": 11,
    "N2": 11,
}

##################################
# 3d parameters
z0 = 0.02
zref = 90
uref = 11.4
ustar = uref * 0.41 / np.log(zref / z0)
plexp = 0.2  # power law exponent
windprofiletype = "PL"  # choosing power law, use log with "LOG" here instead

L = 0.593 * zref  # length scale
Gamma = 3.89  # time scale
sigma = 0.052  # magnitude (σ = αϵ^{2/3})

Lx = 720
Ly = 64
Lz = 64

nBlocks = 2
grid_dimensions = np.array([Lx / 4, Ly, Lz])

grid_levels = np.array([6, 4, 4])
Type_Model = "Mann"
######################################

generator_3d = FluctuationFieldGenerator(
    ustar,
    zref,
    grid_dimensions,
    grid_levels,
    model=Type_Model,
    length_scale=L,
    time_scale=Gamma,
    energy_spectrum_scale=sigma,
    seed=1,
    config_2d_model=None,
)


generator_2d3d = FluctuationFieldGenerator(
    ustar,
    zref,
    grid_dimensions,
    grid_levels,
    model=Type_Model,
    length_scale=L,
    time_scale=Gamma,
    energy_spectrum_scale=sigma,
    seed=1,
    config_2d_model=config_2d,
)


box_3d = generator_3d.generate(nBlocks, zref, uref, z0, windprofiletype, plexp)
box_2d3d = generator_2d3d.generate(nBlocks, zref, uref, z0, windprofiletype, plexp)


# ##############################
# # 2d low-frequency parameters
# L2D = 15_000.0
# sigma2 = 0.6
# z_i = 500.0
# psi = np.deg2rad(43.0)

# _grid_dimensions = np.array(
#     [
#         24 * L2D,
#         3 * L2D,
#         z_i,
#     ]
# )

# _grid_levels = np.array(
#     [  # NOTE: isotropic in xy-plane, not z-direction
#         12,
#         9,
#         4,
#     ]
# )

# # Generate 2D low-frequency field
# config_2d = {
#     "sigma2": sigma2,
#     "L_2d": L2D,
#     "z_i": z_i,
#     "psi": psi,
#     "L1_factor": _grid_dimensions[0] / L2D,
#     "L2_factor": _grid_dimensions[1] / L2D,
#     "N1": _grid_levels[0],
#     "N2": _grid_levels[1],
# }

# low_freq_gen = LowFreqGenerator(config_2d)

# lowfreq_u, lowfreq_v = low_freq_gen.generate()


# ###################
# # 3d parameters
# z0 = 0.02
# zref = 90
# uref = 11.4
# ustar = uref * 0.41 / np.log(zref / z0)
# plexp = 0.2
# windprofiletype = "PL"

# # Mann model parameters (still 3d)
# L = 50
# Gamma = 2.5
# sigma = 0.01

# # Generate 3d high-frequency field
# gen_mann = FluctuationFieldGenerator(
#     ustar,
#     z_i,
#     grid_dimensions=_grid_dimensions,
#     grid_levels=_grid_levels,
#     length_scale=L,
#     time_scale=Gamma,
#     energy_spectrum_scale=sigma,
#     model="Mann",
#     seed=None,
# )


# field_3d = gen_mann.generate(1, zref, uref, z0, windprofiletype, plexp)


# spacing = tuple(_grid_dimensions / (2.0**_grid_levels + 1))

# wind_field_vtk = tuple([np.copy(field_3d[..., i], order="C") for i in range(3)])

# cellData = {
#     "grid": np.zeros_like(field_3d[..., 0]),
#     "wind": wind_field_vtk,
# }

# imageToVTK("field_3d", cellData=cellData, spacing=spacing)

# # Below, we construct 2d+3d field

# field_2d3d = field_3d.copy()
# field_2d3d[:-1, :-1, :, 0] += lowfreq_u[..., np.newaxis]
# field_2d3d[:-1, :-1, :, 1] += lowfreq_v[..., np.newaxis]


# spacing = tuple(_grid_dimensions / (2.0**_grid_levels + 1))

# wind_field_vtk = tuple([np.copy(field_3d[..., i], order="C") for i in range(3)])

# cellData = {
#     "grid": np.zeros_like(field_2d3d[..., 0]),
#     "wind": wind_field_vtk,
# }

# imageToVTK("field_3d", cellData=cellData, spacing=spacing)
