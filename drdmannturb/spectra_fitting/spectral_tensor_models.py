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
        kL = L * k.norm(dim=-1).cpu().detach().numpy()
        y = kL ** (-2.0 / 3.0) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -(kL ** (-2))))
        tau = torch.tensor(y, dtype=torch.get_default_dtype(), device=k.device)

        return gamma * tau


class TwoThirds_ELT(EddyLifetimeModel):
    """The two-thirds eddy lifetime model."""

    def forward(self, k: torch.Tensor, L: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate the two-thirds eddy lifetime model."""
        kL = L * k.norm(dim=-1)
        tau = kL ** (-2.0 / 3.0)

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

        # Debug: Check inputs (just summary stats)
        if torch.isnan(k).any():
            print(f"NaN in VonKarman input k: min={k.min().item()}, max={k.max().item()}, mean={k.mean().item()}")
        if torch.isnan(k_norm).any():
            print(f"NaN in k_norm: min={k_norm.min():.2g}, max={k_norm.max():.2g}, mean={k_norm.mean():.2g}")
        if torch.isnan(kL).any():
            print(f"NaN in VonKarman kL: min={kL.min().item()}, max={kL.max().item()}, mean={kL.mean().item()}")
        if torch.isnan(L):
            print(f"NaN in VonKarman L: {L.item()}")

        # Check for zero k_norm
        zero_count = (k_norm == 0).sum().item()
        if zero_count > 0:
            print(f"Zero k_norm in VonKarman: {zero_count} zeros out of {k_norm.numel()}")
            print(f"k shape: {k.shape}")
            print(f"k range: {k.min().item()} to {k.max().item()}")

        parenthetical_term = kL / ((1.0 + kL**2) ** (0.5))
        E = (k_norm ** (-5.0 / 3.0)) * (parenthetical_term ** (17.0 / 3.0))

        # Debug: Check result (just summary stats)
        if torch.isnan(E).any():
            print(f"NaN in VonKarman E: min={E.min().item()}, max={E.max().item()}, mean={E.mean().item()}")
            print(
                f"parenthetical_term: min={parenthetical_term.min().item()}, max={parenthetical_term.max().item()},"
                f"mean={parenthetical_term.mean().item()}"
            )
            print(
                f"k_norm^(-5/3): min={(k_norm ** (-5.0 / 3.0)).min().item()},"
                f"max={(k_norm ** (-5.0 / 3.0)).max().item()},"
                f"mean={(k_norm ** (-5.0 / 3.0)).mean().item()}"
            )

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

    def forward(self, k: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate the RDT spectral tensor model."""
        # Calculate the scaling parameters
        L = torch.exp(self.log_L)
        gamma = torch.exp(self.log_gamma)
        sigma = torch.exp(self.log_sigma)

        # Debug: Check parameters (just values, not full tensors)
        if torch.isnan(L) or torch.isnan(gamma) or torch.isnan(sigma):
            print(f"NaN in parameters: L={L.item()}, gamma={gamma.item()}, sigma={sigma.item()}")

        # NOTE: The following was previously in OnePointSpectra.forward()
        beta = self.eddy_lifetime_model(k, L, gamma)

        # Debug: Check beta (just summary stats)
        if torch.isnan(beta).any():
            print(f"NaN in beta: min={beta.min().item()}, max={beta.max().item()}, mean={beta.mean().item()}")
            print(f"beta shape: {beta.shape}")

        k0 = k.clone()
        k0[..., 2] = k[..., 2] + beta * k[..., 0]

        # Calculate energy spectrum for "Phi_VK"
        energy_spectrum = self.energy_spectrum_model(k0, L)

        # Debug: Check energy_spectrum (just summary stats)
        if torch.isnan(energy_spectrum).any():
            print(f"NaN in energy_spectrum: min={energy_spectrum.min():.2g}, max={energy_spectrum.max():.2g}")

        E0 = sigma * L ** (5.0 / 3.0) * energy_spectrum

        # Debug: Check E0 (just summary stats)
        if torch.isnan(E0).any():
            print(f"NaN in E0: min={E0.min().item()}, max={E0.max().item()}, mean={E0.mean().item()}")

        # Split k into components
        k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]

        # Debug: Check k1, k2, k3 (just summary stats)
        print(f"k1 range: {k1.min().item():.3e} to {k1.max().item():.3e}")
        print(f"k2 range: {k2.min().item():.3e} to {k2.max().item():.3e}")
        print(f"k3 range: {k3.min().item():.3e} to {k3.max().item():.3e}")

        # Calculate
        k30 = k3 + beta * k1
        kk0 = k1**2 + k2**2 + k30**2
        kk = k1**2 + k2**2 + k3**2
        s = k1**2 + k2**2

        # Debug: Check intermediate calculations (just summary stats)
        if torch.isnan(k30).any():
            print(f"NaN in k30: min={k30.min().item()}, max={k30.max().item()}, mean={k30.mean().item()}")
        if torch.isnan(kk0).any():
            print(f"NaN in kk0: min={kk0.min().item()}, max={kk0.max().item()}, mean={kk0.mean().item()}")
        if torch.isnan(kk).any():
            print(f"NaN in kk: min={kk.min().item()}, max={kk.max().item()}, mean={kk.mean().item()}")
        if torch.isnan(s).any():
            print(f"NaN in s: min={s.min().item()}, max={s.max().item()}, mean={s.mean().item()}")

        # Debug: Check s (just summary stats)
        print(f"s range: {s.min().item():.3e} to {s.max().item():.3e}")
        print(f"s zero count: {(s == 0).sum().item()}")

        # Debug: Check kk and kk0 (just summary stats)
        print(f"kk range: {kk.min().item():.3e} to {kk.max().item():.3e}")
        print(f"kk0 range: {kk0.min().item():.3e} to {kk0.max().item():.3e}")
        print(f"kk zero count: {(kk == 0).sum().item()}")
        print(f"kk0 zero count: {(kk0 == 0).sum().item()}")

        C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
        C2 = k2 * kk0 / torch.sqrt(s**3) * torch.atan2(beta * k1 * torch.sqrt(s), kk0 - k30 * k1 * beta)

        # Debug: Check C1 and C2 (just summary stats)
        if torch.isnan(C1).any():
            print(f"NaN in C1: min={C1.min().item()}, max={C1.max().item()}, mean={C1.mean().item()}")
            print(f"kk * s: min={(kk * s).min().item()}, max={(kk * s).max().item()}, mean={(kk * s).mean().item()}")
            print(f"denominator zero: {(kk * s) == 0}.sum().item()")
            print(f"k1 zero: {(k1 == 0).sum().item()}")
            print(f"s zero: {(s == 0).sum().item()}")
        if torch.isnan(C2).any():
            print(f"NaN in C2: min={C2.min().item()}, max={C2.max().item()}, mean={C2.mean().item()}")
            print(f"s**3: min={(s**3).min().item()}, max={(s**3).max().item()}, mean={(s**3).mean().item()}")
            print(f"s zero: {(s == 0).sum().item()}")

        zeta1 = C1 - k2 / k1 * C2
        zeta2 = C1 * k2 / k1 + C2

        # Debug: Check zeta1 and zeta2 (just summary stats)
        if torch.isnan(zeta1).any():
            print(f"NaN in zeta1: min={zeta1.min().item()}, max={zeta1.max().item()}, mean={zeta1.mean().item()}")
            print(f"k1 zero: {(k1 == 0).sum().item()}")
        if torch.isnan(zeta2).any():
            print(f"NaN in zeta2: min={zeta2.min().item()}, max={zeta2.max().item()}, mean={zeta2.mean().item()}")

        E0 /= 4 * torch.pi

        # Calculate the spectral tensor components
        Phi11 = E0 / (kk0**2) * (kk0 - k1**2 - 2 * k1 * k30 * zeta1 + (k1**2 + k2**2) * zeta1**2)
        Phi22 = E0 / (kk0**2) * (kk0 - k2**2 - 2 * k2 * k30 * zeta2 + (k1**2 + k2**2) * zeta2**2)
        Phi33 = E0 / (kk**2) * (k1**2 + k2**2)
        Phi13 = E0 / (kk * kk0) * (-k1 * k30 + (k1**2 + k2**2) * zeta1)

        Phi12 = E0 / (kk0**2) * (-k1 * k2 - k1 * k30 * zeta2 - k2 * k30 * zeta1 + (k1**2 + k2**2) * zeta1 * zeta2)
        Phi23 = E0 / (kk * kk0) * (-k2 * k30 + (k1**2 + k2**2) * zeta2)

        # Debug: Check final outputs (just summary stats)
        for i, phi in enumerate([Phi11, Phi22, Phi33, Phi13, Phi23, Phi12]):
            if torch.isnan(phi).any():
                print(f"NaN in Phi{i+1}: min={phi.min().item()}, max={phi.max().item()}, mean={phi.mean().item()}")
                print(f"Phi{i+1} shape: {phi.shape}")

        # In order, uu, vv, ww, uw, vw, uv
        return Phi11, Phi22, Phi33, Phi13, Phi23, Phi12
