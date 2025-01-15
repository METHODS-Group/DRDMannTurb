import numpy as np
from low_freq_prototype import generate_2D_lowfreq_approx

from drdmannturb.fluctuation_generation import FluctuationFieldGenerator

# DRDMT 3D turbulence parameters.
Type_Model = "Mann"

z0 = 0.02
zref = 90
uref = 11.4
ustar = uref * 0.41 / np.log(zref / z0)
plexp = 0.2
windprofiletype = "LOG"

L_3D = 50
Gamma = 2.5
sigma = 0.01

Lx = 60_000  # [m] = 60 km
Ly = 15_000  # [m] = 15 km
Lz = 5_000  # [m]

grid_dimensions = np.array([Lx, Ly, Lz])
grid_levels = np.array([6, 4, 4])

nBlocks = 1

seed = None

# Generate 3d Mann box
gen_mann = FluctuationFieldGenerator(
    ustar,
    zref,
    grid_dimensions,
    grid_levels,
    length_scale=L_3D,
    time_scale=Gamma,
    energy_spectrum_scale=sigma,
    model=Type_Model,
    seed=seed,
)

fluctuation_field = gen_mann.generate(1, zref, uref, z0, windprofiletype)

spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))


# 2D field parameters
L_2D = 15_000.0
sigma2 = 0.6
z_i = 500.0
psi_degs = 43.0

L1, L2 = grid_dimensions[:2]
Nx, Ny = 2 ** grid_levels[:2]

_, _, u_field = generate_2D_lowfreq_approx(Nx, Ny, L1, L2, psi_degs, sigma2, L_2D, z_i)
_, _, v_field = generate_2D_lowfreq_approx(Nx, Ny, L1, L2, psi_degs, sigma2, L_2D, z_i)

# TODO: This is not complete yet.
