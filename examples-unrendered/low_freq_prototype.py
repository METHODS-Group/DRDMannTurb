from typing import Union

import numpy as np

# For cleanliness
ArrayType = Union[np.ndarray, float]


# So, phi_ij (k1, k2) is given by equation 1.
# phi_ij(k1, k2) = (E(k) / (\pi k)) * (delta_ij - (k_i * k_j) / (k^2))

# Then, we want to substitute kappa in for k in E(k), NB psi is a constant at this point.

# k_bold = (k1, k2)

# NB 0 < psi < pi/2

"""
Input Parameters
"""

# Physical parameters

L_2D = 15000  # Length scale of the domain
z_i = 500  # Height of the domain
c = 1.0  # Scaling factor NOTE: What should this actually be?
psi = 43  # IN DEGREES; anisotropy angle parameter, 0 < psi < 90 in degrees

# Grid parameters

N1 = 1024  # Number of grid points in x direction
N2 = 256  # Number of grid points in y direction

dx = 1 / 2  # Grid spacing in x direction
dy = 1 / 2  # Grid spacing in y direction

"""
Begin calculations
"""

L1 = N1 * dx
L2 = N2 * dy

m1 = np.arange(-N1 // 2, N1 // 2)
m2 = np.arange(-N2 // 2, N2 // 2)

k1 = m1 * 2 * np.pi / L1
k2 = m2 * 2 * np.pi / L2

"""Helper functions"""


def compute_kappa(k1, k2, psi):
    """
    Compute equation 4 from the paper
    """

    return np.sqrt(2 * ((k1**2 * np.cos(psi) ** 2) + (k2**2 * np.sin(psi) ** 2)))


def compute_E_with_attenuation(kappa, L_2D, c, z_i):
    """
    Compute equation 9 from the paper
    """

    energy_spectrum = (c * (kappa**3)) / (((L_2D ** (-2)) + (kappa**2)) ** (7 / 3))

    attenuation_factor = 1 / (1 + (kappa * z_i) ** 2)

    return energy_spectrum * attenuation_factor
