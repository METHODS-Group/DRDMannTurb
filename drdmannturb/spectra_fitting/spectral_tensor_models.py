"""Physical models composing the DRD models."""

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

    length_scale: float
    time_scale: float

    def __init__(self, length_scale: float, time_scale: float):
        super().__init__()

        self.length_scale = length_scale
        self.time_scale = time_scale

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Compute eddy lifetime tau(k)."""
        raise NotImplementedError("Subclasses must implement the forward pass.")


class TauNet_ELT(EddyLifetimeModel):
    """The DRD eddy lifetime model."""

    length_scale: float
    time_scale: float
    tauNet: TauNet

    def __init__(
        self,
        length_scale: float,
        time_scale: float,
        taunet: TauNet,
    ):
        super().__init__(length_scale, time_scale)
        self.tauNet = taunet

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Evaluate the DRD eddy lifetime model."""
        # NOTE: TauNet takes in the entire k grid, unlike the other models.
        kL = self.length_scale * k
        tau0 = 0.0

        tau = tau0 + self.tauNet(kL)

        return self.time_scale * tau


class Mann_ELT(EddyLifetimeModel):
    """The Mann eddy lifetime model.

    TODO: Warning that this is not differentiable and is CPU only.
    """

    length_scale: float
    time_scale: float

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Evaluate the Mann eddy lifetime model."""
        kL = self.length_scale * k.norm(dim=-1).cpu().detach().numpy()
        y = kL ** (-2.0 / 3.0) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -(kL ** (-2))))
        tau = torch.tensor(y, dtype=torch.get_default_dtype(), device=k.device)

        return self.time_scale * tau


class TwoThirds_ELT(EddyLifetimeModel):
    """The two-thirds eddy lifetime model."""

    length_scale: float
    time_scale: float

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Evaluate the two-thirds eddy lifetime model."""
        kL = self.length_scale * k.norm(dim=-1)
        tau = self.time_scale * kL ** (-2.0 / 3.0)

        return tau


class Constant_ELT(EddyLifetimeModel):
    """The constant eddy lifetime model."""

    length_scale: float
    time_scale: float

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Evaluate the eddy lifetime model."""
        kL = self.length_scale * k.norm(dim=-1)

        tau = self.time_scale * torch.ones_like(kL)

        return tau


###############################################################################
# Energy Spectrum Functions
###############################################################################


class EnergySpectrumModel(nn.Module):
    """Base class for energy spectrum models."""

    def forward(self, k: torch.Tensor, L: float) -> torch.Tensor:
        """Evaluate the energy spectrum model."""
        raise NotImplementedError("Subclasses must implement the forward pass.")


class VonKarman_ESM(EnergySpectrumModel):
    """The standard Von Karman energy spectrum model.

    Note that this only includes scaling by the parameter L.
    """

    def forward(self, k: torch.Tensor, L: float) -> torch.Tensor:
        """Evaluate the Von Karman energy spectrum."""
        # Compute k = |k|
        k_norm = k.norm(dim=-1)
        kL = k_norm * L

        parenthetical_term = kL / ((1.0 + kL**2) ** (0.5))
        E = (k_norm ** (-5.0 / 3.0)) * (parenthetical_term ** (17.0 / 3.0))

        return E


class Learnable_ESM(EnergySpectrumModel):
    """Learnable energy spectrum model.

    This model is based on the Von Karman energy spectrum, but introduces several
    learnable parameters which greatly increase the expressivity of the model.
    """

    def __init__(self, p_init: float = 4.0, q_init: float = 17.0 / 6.0):
        super().__init__()

        self._raw_p = nn.Parameter(torch.tensor(np.log(np.exp(p_init) - 1.0)))
        self._raw_q = nn.Parameter(torch.tensor(np.log(np.exp(q_init) - 1.0)))

    def _positive(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(raw)

    def forward(self, k: torch.Tensor, L: float) -> torch.Tensor:
        """Evaluate the learnable energy spectrum model."""
        k_norm = k.norm(dim=-1)
        kL = L * k_norm
        p = self._positive(self._raw_p)
        q = self._positive(self._raw_q)
        return (kL**p) / ((1.0 + kL**2) ** (q))


###############################################################################
# Spectral Tensor Models
###############################################################################


class SpectralTensorModel(nn.Module):
    """Base class for spectral tensor models."""

    def __init__(
        self,
        eddy_lifetime_model: EddyLifetimeModel,
        energy_spectrum_model: EnergySpectrumModel,
        sigma: float,
        L: float,
        gamma: float,
    ):
        super().__init__()
        self.eddy_lifetime_model = eddy_lifetime_model
        self.energy_spectrum_model = energy_spectrum_model

        self.sigma = sigma
        self.L = L
        self.gamma = gamma

    def forward(self, k: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate the spectral tensor model."""
        raise NotImplementedError("Subclasses must implement the forward pass.")


class RDT_SpectralTensor(SpectralTensorModel):
    """The Rapid Distortion Theory spectral tensor model."""

    def forward(self, k: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluate the RDT spectral tensor model."""
        # NOTE: The following was previously in OnePointSpectra.forward()
        beta = self.eddy_lifetime_model(k)

        k0 = k.clone()
        k0[..., 2] = k[..., 2] + beta * k[..., 2]

        # Calculate energy spectrum for "Phi_VK"
        energy_spectrum = self.energy_spectrum_model(k0, self.L)

        E0 = self.sigma * self.L ** (5.0 / 3.0) * energy_spectrum

        # Split k into components
        k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]

        # Calculate
        k30 = k3 + beta * k1
        kk0 = k1**2 + k2**2 + k30**2
        kk = k1**2 + k2**2 + k3**2
        s = k1**2 + k2**2

        C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
        C2 = k2 * kk0 / torch.sqrt(s**3) * torch.atan2(beta * k1 * torch.sqrt(s), kk0 - k30 * k1 * beta)

        zeta1 = C1 - k2 / k1 * C2
        zeta2 = C1 * k2 / k1 + C2
        E0 /= 4 * torch.pi

        # Calculate the spectral tensor components
        Phi11 = E0 / (kk0**2) * (kk0 - k1**2 - 2 * k1 * k30 * zeta1 + (k1**2 + k2**2) * zeta1**2)
        Phi22 = E0 / (kk0**2) * (kk0 - k2**2 - 2 * k2 * k30 * zeta2 + (k1**2 + k2**2) * zeta2**2)
        Phi33 = E0 / (kk**2) * (k1**2 + k2**2)
        Phi13 = E0 / (kk * kk0) * (-k1 * k30 + (k1**2 + k2**2) * zeta1)

        Phi12 = E0 / (kk0**2) * (-k1 * k2 - k1 * k30 * zeta2 - k2 * k30 * zeta1 + (k1**2 + k2**2) * zeta1 * zeta2)
        Phi23 = E0 / (kk * kk0) * (-k2 * k30 + (k1**2 + k2**2) * zeta2)

        # TODO: This ought to go!
        # Clipping to prevent extremely small values
        epsilon = 1e-32
        Phi11 = torch.where(Phi11 < epsilon, epsilon, Phi11)
        Phi22 = torch.where(Phi22 < epsilon, epsilon, Phi22)
        Phi33 = torch.where(Phi33 < epsilon, epsilon, Phi33)
        Phi13 = torch.where(torch.abs(Phi13) < epsilon, epsilon * torch.sign(Phi13), Phi13)
        Phi12 = torch.where(Phi12 < epsilon, epsilon, Phi12)
        Phi23 = torch.where(Phi23 < epsilon, epsilon, Phi23)

        # In order, uu, vv, ww, uw, vw, uv
        return Phi11, Phi22, Phi33, Phi13, Phi23, Phi12
