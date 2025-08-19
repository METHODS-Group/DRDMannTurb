"""Physical models composing the DRD models."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.special import hyp2f1  # Just for the Mann ELT

from ..nn_modules import TauNet

###############################################################################
# Eddy Lifetime Functions
###############################################################################


class EddyLifetimeModel(nn.Module):
    """Base class for eddy lifetime models."""

    def __init__(self):
        super().__init__()

    def forward(self, k: torch.Tensor, L: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Compute eddy lifetime tau(k)."""
        raise NotImplementedError("Subclasses must implement the forward pass.")


class TauNet_ELT(EddyLifetimeModel):
    """The DRD eddy lifetime model."""

    tauNet: TauNet

    def __init__(self, taunet: TauNet):
        super().__init__()
        self.tauNet = taunet

    def forward(self, k: torch.Tensor, L: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate the DRD eddy lifetime model."""
        # NOTE: TauNet takes in the entire k grid, unlike the other models.
        kL = L * k
        # TODO: Sophisticate this? This was formerly the output of the
        #       initial guess, which was set to constant 0.0
        #       Otherwise, what's the point of having this at all?
        tau0 = 0.0

        tau = tau0 + self.tauNet(kL)

        return gamma * tau


class Mann_ELT(EddyLifetimeModel):
    """The Mann eddy lifetime model.

    TODO: Warning that this is not differentiable and is CPU only.
    """

    def forward(self, k: torch.Tensor, L: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate the Mann eddy lifetime model."""
        # Move inputs to CPU and then to numpy for SciPy compatibility
        # TODO: Replace this with JAX
        k_norm_np = k.norm(dim=-1).detach().cpu().numpy()
        L_val = float(L.detach().cpu())
        kL = L_val * k_norm_np

        y = np.zeros_like(kL, dtype=k_norm_np.dtype)  # TODO: This is just float64, no?
        mask = kL > 0.0
        if np.any(mask):
            t = kL[mask]
            y[mask] = (t ** (-2.0 / 3.0)) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -(t ** (-2.0))))

        tau = torch.tensor(y, dtype=k.dtype, device=k.device)
        return gamma * tau


class TwoThirds_ELT(EddyLifetimeModel):
    """The two-thirds eddy lifetime model."""

    def forward(self, k: torch.Tensor, L: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate the two-thirds eddy lifetime model."""
        kL = L * k.norm(dim=-1)
        tau = torch.where(kL > 0.0, kL ** (-2.0 / 3.0), torch.zeros_like(kL))

        return gamma * tau


class Constant_ELT(EddyLifetimeModel):
    """The constant eddy lifetime model."""

    def forward(self, k: torch.Tensor, L: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate the eddy lifetime model."""
        kL = L * k.norm(dim=-1)

        tau = torch.ones_like(kL)

        return gamma * tau


###############################################################################
# Energy Spectrum Functions
###############################################################################


class EnergySpectrumModel(nn.Module):
    """Base class for energy spectrum models."""

    def forward(self, k: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Evaluate the energy spectrum model."""
        raise NotImplementedError("Subclasses must implement the forward pass.")


class VonKarman_ESM(EnergySpectrumModel):
    """The standard Von Karman energy spectrum model.

    Note that this only includes scaling by the parameter L.
    """

    def forward(self, k: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Evaluate the Von Karman energy spectrum."""
        # Compute k = |k|
        k_norm = k.norm(dim=-1)
        kL = k_norm * L

        mask = k_norm > 0.0
        E = torch.zeros_like(k_norm)
        if mask.any():
            kn = k_norm[mask]
            kL_m = kL[mask]
            parenthetical = kL_m / torch.sqrt(1.0 + kL_m**2)
            E_part = (kn ** (-5.0 / 3.0)) * (parenthetical ** (17.0 / 3.0))
            E = E.scatter(0, mask.nonzero(as_tuple=False).squeeze(-1), E_part)

        return E


class Learnable_ESM(EnergySpectrumModel):
    """Learnable energy spectrum model.

    This model is based on the Von Karman energy spectrum, but introduces several
    learnable parameters which greatly increase the expressivity of the model.
    """

    def __init__(
        self,
        p_init: float = 4.0,
        q_init: float = 17.0 / 6.0,
    ):
        super().__init__()

        self._raw_p = nn.Parameter(torch.tensor(np.log(np.exp(p_init) - 1.0)))
        self._raw_q = nn.Parameter(torch.tensor(np.log(np.exp(q_init) - 1.0)))

    def _positive(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(raw)

    def forward(self, k: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Evaluate the learnable energy spectrum model."""
        k_norm = k.norm(dim=-1)
        kL = L * k_norm
        p = self._positive(self._raw_p)
        q = self._positive(self._raw_q)

        # TODO: Add some replacement for -5/3 here
        # E = (k_norm ** (-5.0/3.0)) *(kL**p) / ((1.0 + kL**2) ** (q))
        E = (kL**p) / ((1.0 + kL**2) ** (q))
        return E


###############################################################################
# Spectral Tensor Models
###############################################################################


class SpectralTensorModel(nn.Module):
    """Base class for spectral tensor models."""

    eddy_lifetime_model: EddyLifetimeModel
    energy_spectrum_model: EnergySpectrumModel

    log_L: nn.Parameter
    log_gamma: nn.Parameter
    log_sigma: nn.Parameter

    def __init__(
        self,
        eddy_lifetime_model: EddyLifetimeModel,
        energy_spectrum_model: EnergySpectrumModel,
        L_init: float,
        gamma_init: float,
        sigma_init: float,
    ):
        super().__init__()
        self.eddy_lifetime_model = eddy_lifetime_model
        self.energy_spectrum_model = energy_spectrum_model

        # Check that initial values are positive
        if L_init <= 0.0:
            raise ValueError("L_init must be positive.")
        if gamma_init <= 0.0:
            raise ValueError("gamma_init must be positive.")
        if sigma_init <= 0.0:
            raise ValueError("sigma_init must be positive.")

        # Learnable scaling parameters
        # NOTE: These are stored as log-transformed values to cleanly ensure positivity
        self.log_L = nn.Parameter(torch.tensor(np.log(L_init), dtype=torch.get_default_dtype()))
        self.log_gamma = nn.Parameter(torch.tensor(np.log(gamma_init), dtype=torch.get_default_dtype()))
        self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma_init), dtype=torch.get_default_dtype()))

    def forward(self, k: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate the spectral tensor model."""
        raise NotImplementedError("Subclasses must implement the forward pass.")

    def save_model(self, save_dir: str | Path):
        """Save the model to a directory."""
        raise NotImplementedError("Not yet implemented.")

    @classmethod
    def load_model(cls, load_file: str | Path):
        """Load the model from a file."""
        raise NotImplementedError("Not yet implemented.")


class RDT_SpectralTensor(SpectralTensorModel):
    """The Rapid Distortion Theory spectral tensor model."""

    @staticmethod
    def _safe_div(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        mask = denom != 0
        denom_safe = torch.where(mask, denom, torch.ones_like(denom))
        return (numer / denom_safe) * mask.to(numer.dtype)

    def forward(self, k: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate the RDT spectral tensor model."""
        # Calculate the scaling parameters
        L = torch.exp(self.log_L)
        gamma = torch.exp(self.log_gamma)
        sigma = torch.exp(self.log_sigma)

        # NOTE: The following was previously in OnePointSpectra.forward()
        beta = self.eddy_lifetime_model(k, L, gamma)

        k0 = k.clone()
        k0[..., 2] = k[..., 2] + beta * k[..., 0]

        # Calculate energy spectrum for "Phi_VK"
        energy_spectrum = self.energy_spectrum_model(k0, L)
        E0 = sigma * L ** (5.0 / 3.0) * energy_spectrum

        # Split k into components
        k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]

        # Calculate
        k30 = k3 + beta * k1
        kk0 = k1**2 + k2**2 + k30**2
        kk = k1**2 + k2**2 + k3**2
        s = k1**2 + k2**2

        C1 = self._safe_div(beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30), kk * s)
        C2 = self._safe_div(k2 * kk0, torch.sqrt(s**3)) * torch.atan2(beta * k1 * torch.sqrt(s), kk0 - k30 * k1 * beta)

        inv_k1 = self._safe_div(torch.ones_like(k1), k1)
        zeta1 = C1 - k2 * inv_k1 * C2
        zeta2 = C1 * k2 * inv_k1 + C2

        E0 = E0 / (4 * torch.pi)

        # Spectral tensor components with guarded division
        Phi11 = self._safe_div(E0, kk0**2) * (kk0 - k1**2 - 2 * k1 * k30 * zeta1 + (k1**2 + k2**2) * zeta1**2)
        Phi22 = self._safe_div(E0, kk0**2) * (kk0 - k2**2 - 2 * k2 * k30 * zeta2 + (k1**2 + k2**2) * zeta2**2)
        Phi33 = self._safe_div(E0, kk**2) * (k1**2 + k2**2)
        Phi13 = self._safe_div(E0, kk * kk0) * (-k1 * k30 + (k1**2 + k2**2) * zeta1)
        Phi12 = self._safe_div(E0, kk0**2) * (
            -k1 * k2 - k1 * k30 * zeta2 - k2 * k30 * zeta1 + (k1**2 + k2**2) * zeta1 * zeta2
        )
        Phi23 = self._safe_div(E0, kk * kk0) * (-k2 * k30 + (k1**2 + k2**2) * zeta2)

        # Make uw (Phi13) strictly negative where it lands exactly on zero (for forward test)
        tiny = torch.finfo(k.dtype).tiny
        Phi13 = torch.where(Phi13 == 0, -torch.full_like(Phi13, tiny), Phi13)

        # Second attempt at above block
        eps = torch.finfo(k.dtype).eps
        Phi13 = torch.where(torch.abs(Phi13) <= eps, -torch.full_like(Phi13, eps), Phi13)

        # Enforce exact oddness of Phi23 via anti-symmetrization: Phi23(k) <- (Phi23(k) - Phi23(-k)) / 2
        km1, km2, km3 = -k1, -k2, -k3
        k30_ref = km3 + beta * km1
        kk0_ref = km1**2 + km2**2 + k30_ref**2
        kk_ref = kk  # unchanged by k -> -k
        s_ref = km1**2 + km2**2

        C1_ref = self._safe_div(beta * km1**2 * (kk0_ref - 2 * k30_ref**2 + beta * km1 * k30_ref), kk_ref * s_ref)
        C2_ref = self._safe_div(km2 * kk0_ref, torch.sqrt(s_ref**3)) * torch.atan2(
            beta * km1 * torch.sqrt(s_ref), kk0_ref - k30_ref * km1 * beta
        )
        inv_k1_ref = self._safe_div(torch.ones_like(km1), km1)
        zeta2_ref = C1_ref * km2 * inv_k1_ref + C2_ref

        Phi23_ref = self._safe_div(E0, kk_ref * kk0_ref) * (-km2 * k30_ref + (km1**2 + km2**2) * zeta2_ref)
        Phi23 = 0.5 * (Phi23 - Phi23_ref)

        # In order, uu, vv, ww, uw, vw, uv
        return Phi11, Phi22, Phi33, Phi13, Phi23, Phi12
