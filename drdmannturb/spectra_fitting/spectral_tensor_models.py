"""Physical models composing the DRD models."""

import torch
import torch.nn as nn

###############################################################################
# Eddy Lifetime Functions
###############################################################################

class EddyLifetimeModel(nn.Module):
    """Base class for eddy lifetime models."""

    def __init__(self, length_scale: float, time_scale: float):
        super().__init__()

        self.length_scale = length_scale
        self.time_scale = time_scale

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Compute eddy lifetime tau(k)."""
        raise NotImplementedError("Subclasses must implement the forward pass.")


class TauNet_ELT(EddyLifetimeModel):

    def __init__(
        self,
        length_scale: float,
        time_scale: float,
        nlayers: int,
        hidden_layer_size: int,
    ):
        super().__init__(length_scale, time_scale)

        # TODO: Maybe just move this into this file?
        #       Pretty sure TauNet is not directly used elsewhere.
        self.tauNet = TauNet(nlayers, hidden_layer_size)

    def forward(self, k: torch.Tensor) -> torch.Tensor:

        pass








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

        parenthetical_term = kL / ((1.0 + kL**2) **(0.5))
        E = (k_norm ** (-5.0 / 3.0)) * (parenthetical_term ** (17.0 / 3.0))

        return E


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

    def forward(self, k: torch.Tensor) -> tuple[torch.Tensor, ...]:

        raise NotImplementedError("Subclasses must implement the forward pass.")


class RDT_SpectralTensor(SpectralTensorModel):

    def forward(self, k: torch.Tensor) -> tuple[torch.Tensor, ...]:


        pass
