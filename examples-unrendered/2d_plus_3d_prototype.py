import numpy as np
import scipy.special as sp
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

generator = LowFreq2DFieldGenerator(grid_dimensions, grid_levels, L_2D=L_2D, sigma2=sigma2, z_i=z_i, psi_degs=psi_degs)

_, _, u_field = generator.generate()
_, _, v_field = generator.generate()

fluctuation_field_2d3d = fluctuation_field_mann.copy()
fluctuation_field_2d3d[:, :, :, 0] += u_field[:, :, np.newaxis]
fluctuation_field_2d3d[:, :, :, 1] += v_field[:, :, np.newaxis]

#########################################################################################
# Test Mean, Std, Dev, Skewness, and Kurtosis


#########################################################################################
## TESTS
# Helper functions for tests


def analytic_F11_2D(k1):
    """
    Analytic solution for F11(k1) in 2d.
    """
    a = 1 + 2 * (k1 * generator.L_2D * np.cos(generator.psi_rad)) ** 2
    b = 1 + 2 * (k1 * generator.z_i * np.cos(generator.psi_rad)) ** 2
    p = (L_2D**2 * b) / (z_i**2 * a)

    d = 1.0

    first_term_numerator = (sp.gamma(11 / 6) * (L_2D ** (11 / 3))) * (
        -p * sp.hyp2f1(5 / 6, 1, 1 / 2, p) - 7 * sp.hyp2f1(5 / 6) + 2 * sp.hyp2f1(-1 / 6, 1, 1 / 2, p)
    )
    first_term_denominator = (2 * np.pi) ** (11 / 3) * (a ** (11 / 6))

    second_term_numerator = L_2D ** (14 / 3) * np.sqrt(b)
    second_term_denominator = 2 * np.sqrt(2) * d ** (7 / 3) * (generator.z_i * np.sin(generator.psi_rad))

    return generator.c * (
        (first_term_numerator / first_term_denominator) + (second_term_numerator / second_term_denominator)
    )


def analytic_F22_2D(k1):
    a = 1 + 2 * (k1 * generator.L_2D * np.cos(generator.psi_rad)) ** 2
    b = 1 + 2 * (k1 * generator.z_i * np.cos(generator.psi_rad)) ** 2
    p = (L_2D**2 * b) / (z_i**2 * a)

    d = 1.0

    leading_factor_num = generator.z_i**4 * a ** (1 / 6) ** generator.L_2D * sp.gamma(17 / 6)
    leading_factor_denom = (
        55
        * np.sqrt(2 * np.pi)
        * (generator.L_2D**2 - generator.z_i**2) ** 2
        * b
        * sp.gamma(7 / 3)
        * np.sin(generator.psi_rad)
    )
    leading_factor = -leading_factor_num / leading_factor_denom

    line_1 = -9 - 25 * sp.hyp2f1(-1 / 6, 1, 1 / 2, p)
    line_2 = p**2 * (15 - 30 * sp.hyp2f1(-1 / 6, 1, 1 / 2, p) - 59 * sp.hyp2f1(5 / 6, 1, 1 / 2, p))
    line_3 = 35 * sp.hyp2f1(5 / 6, 1, 1 / 2, p) + 15 * p**3 * sp.hyp2f1(5 / 6, 1, 1 / 2, p)
    line_4 = p * (-54 + 88 * sp.hyp2f1(-1 / 6, 1, 1 / 2, p) + 9 * sp.hyp2f1(5 / 6, 1, 1 / 2, p))

    term_1 = leading_factor * (line_1 + line_2 + line_3 + line_4)

    term_2 = (L_2D ** (14 / 3)) / (np.sqrt(2 * b) * d ** (7 / 3) * z_i * np.sin(generator.psi_rad))

    paren = term_1 - term_2

    return generator.c * k1**2 * paren


def estimate_spectra_2d(padded_u_field, padded_v_field) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the spectra of the 2d fields.
    """

    u_fft = np.fft.fft2(padded_u_field)
    v_fft = np.fft.fft2(padded_v_field)

    F_11 = np.abs(u_fft) ** 2
    F_22 = np.abs(v_fft) ** 2

    N = padded_u_field.shape[0]
    F_11 = F_11 / N
    F_22 = F_22 / N

    F_11

    return np.zeros(N), np.zeros(N)


#########################################################################################
# Test energy spectrum


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
