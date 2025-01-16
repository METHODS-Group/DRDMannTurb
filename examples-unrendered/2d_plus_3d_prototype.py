import numpy as np
from low_freq_prototype import LowFreq2DFieldGenerator

from drdmannturb import FluctuationFieldGenerator

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

fluctuation_field_mann = gen_mann.generate(nBlocks, zref, uref, z0, windprofiletype, plexp)


# TODO: Why should the user have to do this themselves? This seems like
#       we should handle it ourselves
spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))


# 2D field parameters
L_2D = 15_000.0
sigma2 = 0.6
z_i = 500.0
psi_degs = 43.0

# L1, L2 = grid_dimensions[:2]
# Nx, Ny = 2 ** grid_levels[:2]

# _, _, u_field = generate_2D_lowfreq_approx(Nx, Ny, L1, L2, psi_degs, sigma2, L_2D, z_i)
# _, _, v_field = generate_2D_lowfreq_approx(Nx, Ny, L1, L2, psi_degs, sigma2, L_2D, z_i)
# np_u_field_pad = np.pad(u_field, ((0,1), (0,1)), mode='wrap')
# np_v_field_pad = np.pad(v_field, ((0,1), (0,1)), mode='wrap')

# fluctuation_field_2d3d = fluctuation_field_mann.copy()
# fluctuation_field_2d3d[:, :, :, 0] += np_u_field_pad[:, :, np.newaxis]
# fluctuation_field_2d3d[:, :, :, 1] += np_v_field_pad[:, :, np.newaxis]

## New API

generator = LowFreq2DFieldGenerator(grid_dimensions, grid_levels, L2D=L_2D, sigma2=sigma2, z_i=z_i, psi_degs=psi_degs)

_, _, u_field = generator.generate()
_, _, v_field = generator.generate()

fluctuation_field_2d3d = fluctuation_field_mann.copy()
fluctuation_field_2d3d[:, :, :, 0] += u_field[:, :, np.newaxis]
fluctuation_field_2d3d[:, :, :, 1] += v_field[:, :, np.newaxis]


#########################################################################################
## TESTS
# Helper functions for tests


def analytic_F11_2D(k_1):
    # first_num = (sp.gamma(11/6) * (L_2D**(11/3)))

    pass


def analytic_F22_2D(k_1):
    pass


#########################################################################################
# Test Mean, Std, Dev, Skewness, and Kurtosis

#########################################################################################
# Test energy spectrum
#
# ie,


#########################################################################################
# Test divergence
#
# Flow is assumed to be incompressible, so the divergence of the velocity field should
# be (near) zero.


def calc_div(field, spacing):
    return np.ufunc.reduce(np.add, [np.gradient(field[..., i], spacing[i], axis=i) for i in range(3)])


assert np.isclose(calc_div(fluctuation_field_mann, spacing).mean(), 0.0, atol=1e-6)

avg_div_2d3d = calc_div(fluctuation_field_2d3d, spacing).mean()

print("Divergence of 2d+3d field: ", avg_div_2d3d)
assert np.isclose(avg_div_2d3d, 0.0, atol=1e-4)
