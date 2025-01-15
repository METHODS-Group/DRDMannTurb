"""
This module implements the Syed-Mann (2024) low-frequency wind fluctuation model.
"""

from typing import Optional

import numpy as np
from scipy import integrate


def _compute_kappa(k1: float, k2: float, psi: float) -> float:
    r"""
    Subroutine to compute the horizontal wavevector :math:`\kappa`, defined by

    .. math::
        \kappa = \sqrt{2(k_1^2 \cos^2(\psi) + k_2^2 \sin^2(\psi))}

    Parameters
    ----------
    k1 : float
        Wavenumber k1

    k2 : float
        Wavenumber k2

    psi : float
        "Anisotropy parameter" angle :math:`\psi`, in radians

    Returns
    -------
    float
        Computed kappa value
    """

    return np.sqrt(2.0 * ((k1**2) * np.cos(psi) ** 2 + (k2**2) * np.sin(psi) ** 2))


def _compute_E(kappa: float, c: float, L2D: float, z_i: float) -> float:
    r"""
    Subroutine to compute the energy spectrum :math:`E(\kappa)` with the attenuation factor,
    defined by

    .. math::
        E(\kappa) = \frac{c \kappa^3}{(L_{2\textrm{D}}^{-2} + \kappa^2)^{7/3}} \cdot
        \frac{1}{1 + \kappa^2 z_i^2}

    Parameters
    ----------
    kappa : float
        Replacement "wavenumber" :math:`\kappa`

    c : float
        Scaling factor :math:`c` used to correct the variance

    L2D : float
        Length scale :math:`L_{2\textrm{D}}`
    """
    if np.isclose(kappa, 0.0):
        return 0.0

    denom = (1.0 / (L2D**2) + kappa**2) ** (7.0 / 3.0)
    atten = 1.0 / (1.0 + (kappa * z_i) ** 2)
    return c * (kappa**3) / denom * atten


def _estimate_c(sigma2: float, L2D: float, z_i: float) -> float:
    r"""
    Subroutine to estimate the scaling factor :math:`c` from the target variance :math:`\sigma^2`.

    This is achieved by approximating the integral of :math:`E(\kappa)` from :math:`\kappa=0` to
    :math:`\infty` by quadrature, since
    .. math::
        \int_0^\infty E(\kappa)
        = c \int_0^\infty \frac{\kappa^3}{(L_{2\textrm{D}}^{-2} + \kappa^2)^{7/3}} \cdot
            \frac{1}{1 + \kappa^2 z_i^2}
        = \sigma^2

    Parameters
    ----------
    sigma2 : float
        Target variance :math:`\sigma^2`

    L2D : float
        Length scale :math:`L_{2\textrm{D}}`

    z_i : float
        Height :math:`z_i`
    """

    def integrand(kappa: float) -> float:
        return kappa**3 / ((1.0 / (L2D**2) + kappa**2) ** (7.0 / 3.0)) * (1.0 / (1.0 + (kappa * z_i) ** 2))

    val, err = integrate.quad(integrand, 0, np.inf)

    return sigma2 / val


def generate_2D_lowfreq(
    Nx: int,
    Ny: int,
    L1: float,
    L2: float,
    psi_degs: float,
    sigma2: float,
    L2D: float,
    z_i: float,
    c: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    r"""
    Generates the 2D low-frequency wind fluctuation component of the Syed-Mann (2024) 2D+3D model.

    Parameters
    ----------
    Nx : int
        Number of grid points in the x-direction
    Ny : int
        Number of grid points in the y-direction
    L1 : float
        Length of the domain in the x-direction
    L2 : float
        Length of the domain in the y-direction
    psi_degs : float
        "Anisotropy parameter" angle :math:`\psi`, in degrees
    sigma2 : float
        Target variance :math:`\sigma^2`
    L2D : float
        Length scale :math:`L_{2\textrm{D}}`
    z_i : float
        Height :math:`z_i`
    c : float
        Scaling factor :math:`c` to use for the energy spectrum. If not provided, it is
        estimated by quadrature from the provided target variance :math:`\sigma^2`.

    Returns
    -------
    np.ndarray
        Generated 2D low-frequency wind fluctuation component. This is `Nx` by `Ny` by 2,
        where the third dimension is the u- (longitudinal) and v-components (transverse).

        TODO ^^
    """

    assert 0 < psi_degs and psi_degs < 90, "Anisotropy parameter psi_degs must be between 0 and 90 degrees"

    psi = np.deg2rad(psi_degs)

    if c is None:
        c = _estimate_c(sigma2, L2D, z_i)

    dx = L1 / Nx
    dy = L2 / Ny

    kx_arr = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky_arr = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kx_arr = np.fft.fftshift(kx_arr)
    ky_arr = np.fft.fftshift(ky_arr)

    amp2 = np.zeros((Nx, Ny), dtype=np.float64)

    factor_16 = (2.0 * np.pi**2) / L1

    for ix in range(Nx):
        for iy in range(Ny):
            kx = kx_arr[ix]
            ky = ky_arr[iy]

            kappa = _compute_kappa(kx, ky, psi)
            E_val = _compute_E(kappa, c, L2D, z_i)

            if kappa < 1e-12:
                phi_11 = 0.0
            else:
                phi_11 = E_val / (np.pi * kappa)

            amp2[ix, iy] = factor_16 * phi_11

    Uhat = np.zeros((Nx, Ny), dtype=np.complex128)

    for ix in range(Nx):
        for iy in range(Ny):
            amp = np.sqrt(amp2[ix, iy])
            phase = (np.random.normal() + 1j * np.random.normal()) / np.sqrt(2.0)
            Uhat[ix, iy] = amp * phase

    Uhat_unshift = np.fft.ifftshift(Uhat, axes=(0, 1))
    u_field_complex = np.fft.ifft2(Uhat_unshift, s=(Nx, Ny))
    u_field = np.real(u_field_complex)

    var_now = np.var(u_field)
    if var_now > 1e-12:
        u_field *= np.sqrt(sigma2 / var_now)

    return u_field
