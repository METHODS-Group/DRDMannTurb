"""Spectral-tensor utilities in JAX.

Contains:
    • VKEnergySpectrum - Von Karman curve
    • PowerSpectraRDT - classical RDT model
    • RDT_spectral_tensor - wraps TauNet to produce Φ_ij(k)
    • OnePointSpectra - k₂/k₃ integration helper
"""

from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .taunet import TauNet

__all__ = [
    "VKEnergySpectrum",
    "PowerSpectraRDT",
    "RDT_spectral_tensor",
    "OnePointSpectra",
]


# ---------------------------------------------------------
# Basic spectra functions
# ---------------------------------------------------------

def VKEnergySpectrum(kL: jnp.ndarray) -> jnp.ndarray:
    return kL ** 4 / (1.0 + kL ** 2) ** (17.0 / 6.0)


def PowerSpectraRDT(k: jnp.ndarray, beta: jnp.ndarray, E0: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]

    k30 = k3 + beta * k1
    kk0 = k1 ** 2 + k2 ** 2 + k30 ** 2
    kk = k1 ** 2 + k2 ** 2 + k3 ** 2
    s = k1 ** 2 + k2 ** 2
    s_safe = jnp.where(s == 0, 1e-12, s)

    C1 = beta * k1 ** 2 * (kk0 - 2 * k30 ** 2 + beta * k1 * k30) / (kk * s_safe)
    C2 = k2 * kk0 / jnp.sqrt(s_safe ** 3) * jnp.arctan2(
        beta * k1 * jnp.sqrt(s_safe), kk0 - k30 * k1 * beta
    )

    k1_safe = jnp.where(k1 == 0, 1e-12, k1)
    zeta1 = C1 - k2 / k1_safe * C2
    zeta2 = C1 * k2 / k1_safe + C2

    E0s = E0 / (4.0 * jnp.pi)
    Phi11 = E0s / (kk0 ** 2) * (
        kk0 - k1 ** 2 - 2 * k1 * k30 * zeta1 + (k1 ** 2 + k2 ** 2) * zeta1 ** 2
    )
    Phi22 = E0s / (kk0 ** 2) * (
        kk0 - k2 ** 2 - 2 * k2 * k30 * zeta2 + (k1 ** 2 + k2 ** 2) * zeta2 ** 2
    )
    Phi33 = E0s / (kk ** 2) * (k1 ** 2 + k2 ** 2)
    Phi13 = E0s / (kk * kk0) * (-k1 * k30 + (k1 ** 2 + k2 ** 2) * zeta1)
    Phi12 = E0s / (kk0 ** 2) * (
        -k1 * k2
        - k1 * k30 * zeta2
        - k2 * k30 * zeta1
        + (k1 ** 2 + k2 ** 2) * zeta1 * zeta2
    )
    Phi23 = E0s / (kk * kk0) * (-k2 * k30 + (k1 ** 2 + k2 ** 2) * zeta2)

    return Phi11, Phi22, Phi33, Phi13, Phi12, Phi23


# ---------------------------------------------------------
# Grid helper dataclass
# ---------------------------------------------------------

class k2_k3_parameters(eqx.Module):
    k2_min_p: int = -3
    k2_max_p: int = 3
    k2_points: int = 64  # smaller default for tests
    k3_min_p: int = -3
    k3_max_p: int = 3
    k3_points: int = 64


# ---------------------------------------------------------
# RDT tensor wrapper
# ---------------------------------------------------------

class RDT_spectral_tensor(eqx.Module):
    eddy_lifetime: TauNet
    L: jnp.ndarray
    Gamma: jnp.ndarray
    sigma: jnp.ndarray
    k2_grid: jnp.ndarray = eqx.static_field()
    k3_grid: jnp.ndarray = eqx.static_field()

    def __init__(
        self,
        eddy_lifetime: TauNet,
        *,
        L_init: float = 1.0,
        Gamma_init: float = 1.0,
        sigma_init: float = 1.0,
        k2_k3: Optional[k2_k3_parameters] = None,
    ):
        object.__setattr__(self, "eddy_lifetime", eddy_lifetime)
        self.L = jnp.array(L_init)
        self.Gamma = jnp.array(Gamma_init)
        self.sigma = jnp.array(sigma_init)
        k2_k3 = k2_k3 or k2_k3_parameters()
        k2g = jnp.logspace(k2_k3.k2_min_p, k2_k3.k2_max_p, k2_k3.k2_points)
        k3g = jnp.logspace(k2_k3.k3_min_p, k2_k3.k3_max_p, k2_k3.k3_points)
        object.__setattr__(self, "k2_grid", k2g)
        object.__setattr__(self, "k3_grid", k3g)

    def __call__(self, k1: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        K1, K2, K3 = jnp.meshgrid(k1, self.k2_grid, self.k3_grid, indexing="ij")
        kvec = jnp.stack([K1, K2, K3], -1)
        tau = jax.vmap(self.eddy_lifetime)(kvec.reshape(-1, 3)).reshape(kvec.shape[:-1])
        beta = self.Gamma * tau
        k0 = kvec.at[..., 2].add(beta * kvec[..., 0])
        E0 = self.sigma * VKEnergySpectrum(self.L * jnp.linalg.norm(k0, axis=-1))
        return PowerSpectraRDT(kvec, beta, E0)


# ---------------------------------------------------------
# One-point spectra integrator
# ---------------------------------------------------------

class OnePointSpectra(eqx.Module):
    rdt: RDT_spectral_tensor
    _k2: jnp.ndarray = eqx.static_field()
    _k3: jnp.ndarray = eqx.static_field()

    def __init__(self, rdt: RDT_spectral_tensor):
        object.__setattr__(self, "rdt", rdt)
        object.__setattr__(self, "_k2", rdt.k2_grid)
        object.__setattr__(self, "_k3", rdt.k3_grid)

    def __call__(self, phi_component: jnp.ndarray) -> jnp.ndarray:
        tmp = jax.scipy.integrate.trapezoid(phi_component, x=self._k3, axis=2)
        return jax.scipy.integrate.trapezoid(tmp, x=self._k2, axis=1)
