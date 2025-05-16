"""Generate low-frequency fluctuation field, meant as an augmentation to other 3d fields.

The methods implemented here are based on:
[1] A.H. Syed, J. Mann "A Model for Low-Frequency, Anisotropic Wind Fluctuations and Coherences
    in the Marine Atmosphere", Boundary-Layer Meteorology 190:1, 2024 <https://doi.org/10.1007/s10546-023-00850-w>
[2] A.H. Syed, J. Mann "Simulating low-frequency wind fluctuations", Wind Energy Science, 9, 1381-1391, 2024
    <https://doi.org/10.5194/wes-9-1381-2024>
"""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from scipy import integrate

LowFreqConfig = Dict[str, float]


def generate_low_frequency_field(
    sigma2: float,
    L_2d: float,
    psi: float,
    z_i: float,
) -> Float[Array, "N1 N2 2"]:  # noqa: F722
    """Generate a low-frequency fluctuation field.

    This function is not intended for direct use. Instead, use the `FluctuationGenerator` class.

    This implements a low-frequency turbulence model calibrated against marine environments and is
    the 2D component of a so-called 2D+3D turbulence model, which is obtained by generating a 3D
    field and, separately, this 2D field and then adding the 2D field to each vertical level of the 3D field.

    Parameters
    ----------
    sigma2 : float
        Turbulence intensity.
    L_2d : float
        Integral length scale.
    psi : float
        Direction of the mean wind.
    Lx : float
        Length of the domain in the x-direction.
    Ly : float
        Length of the domain in the y-direction.
    Nx : int
        Grid levels in the x-direction.

        This means that there will be (2**Nx + 1) grid points in the x-direction.
    Ny : int
        Grid levels in the y-direction.

        This means that there will be (2**Ny + 1) grid points in the y-direction.

    Returns
    -------
    u_field: Float[Array, "N1 N2 2"]
        Vector field of shape (2**N1 + 1, 2**N2 + 1, 2) representing the low-frequency fluctuations.
    """
