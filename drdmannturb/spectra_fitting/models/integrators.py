"""
Implementation of quantities that are written or defined by integrals of the spectral tensor.

In particular, this implements the integration schemes for calculating the one-point spectra
and spectral coherences.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from .spectral import RDT_spectral_tensor

# ---------------------------------------------------------
# One-point spectra integrator
# ---------------------------------------------------------

class OnePointSpectra(eqx.Module):
    """
    Implements the integration scheme for calculating the one-point spectra of a spectral tensor model.

    Attributes
    ----------
    rdt : RDT_spectral_tensor
        The RDT spectral tensor model.
    _k2 : jnp.ndarray
        The k2 values to integrate over.
    _k3 : jnp.ndarray
        The k3 values to integrate over.

    Methods
    -------
    __call__(phi_component: jnp.ndarray) -> jnp.ndarray
        Integrates the given phi component over the k2 and k3 grids.
    """

    rdt: RDT_spectral_tensor
    _k2: jnp.ndarray = eqx.static_field()
    _k3: jnp.ndarray = eqx.static_field()

    def __init__(self, rdt: RDT_spectral_tensor):
        object.__setattr__(self, "rdt", rdt)
        object.__setattr__(self, "_k2", rdt.k2_grid)
        object.__setattr__(self, "_k3", rdt.k3_grid)

    def __call__(self, phi_component: jnp.ndarray) -> jnp.ndarray:
        r"""
        Calculate the one-point spectra over the given phi component.

        .. math::
            F_{ij}(k_1) = \iint \Phi_{ij}(k_1, k_2, k_3) ~\mathrm{d}k_2 \mathrm{d}k_3

        Parameters
        ----------
        phi_component : jnp.ndarray
            The phi component to integrate.

        Returns
        -------
        jnp.ndarray
            The one-point spectra :math:`F_{ij}(k_1)`.
        """
        tmp = jax.scipy.integrate.trapezoid(phi_component, x=self._k3, axis=2)
        return jax.scipy.integrate.trapezoid(tmp, x=self._k2, axis=1)
