"""
Spectral Coherence module
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from numpy import log

from drdmannturb.common import MannEddyLifetime, VKEnergySpectrum
from drdmannturb.enums import EddyLifetimeType, PowerSpectraType
from drdmannturb.nn_modules import TauNet
from drdmannturb.parameters import NNParameters
from drdmannturb.spectra_fitting.power_spectra_rdt import PowerSpectraRDT


class SpectralCoherence(nn.Module):
    """
    Spectral Coherence class
    """

    def __init__(
        self,
        eddy_lifetime: EddyLifetimeType = EddyLifetimeType.TWOTHIRD,
        power_spectra: PowerSpectraType = PowerSpectraType.RDT,
        nn_params: Optional[NNParameters] = None,
        **kwargs
    ):
        """
        Spectral coherence constructor
        """
        super(SpectralCoherence, self).__init__()

        self.type_EddyLifetime = eddy_lifetime
        self.type_PowerSpectra = power_spectra

        self._init_grids()
        self._init_parameters()

        if self.type_EddyLifetime == EddyLifetimeType.TAUNET:
            self.tauNet = TauNet(**kwargs)

    def forward(
        self,
        k1_input: torch.Tensor,
        Delta_y_input: torch.Tensor,
        Delta_z_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implements the forward method.

        Parameters
        ----------
        k1_input : torch.Tensor
            _description_
        Delta_y_input : torch.Tensor
            _description_
        Delta_z_input : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """

        self._exp_scales()
        self.k = torch.stack(
            torch.meshgrid(k1_input, self.grid_k2, self.grid_k3), dim=-1, indexing="ij"
        )
        self.k123 = self.k[..., 0], self.k[..., 1], self.k[..., 2]
        self.beta = self.EddyLifetime()
        self.k0 = self.k.clone()
        self.k0[..., 2] = self.k[..., 2] + self.beta * self.k[..., 0]
        k0L = self.LengthScale * self.k0.norm(dim=-1)
        self.E0 = self.Magnitude * VKEnergySpectrum(k0L)
        self.Phi = self.PowerSpectra()
        # Phi11, Phi33, Phi13 = self.Phi[0], self.Phi[2], self.Phi[3]
        Phi11, Phi33, Phi13 = self.Phi[0], self.Phi[0], self.Phi[0]
        F1 = self._quad23(Phi11)
        F3 = self._quad23(Phi33)
        k2, k3 = self.k[..., 1], self.k[..., 2]
        Chi = torch.zeros(
            [k1_input.numel(), Delta_y_input.numel(), Delta_z_input.numel()],
            dtype=torch.float64,
        )
        for i, dy in enumerate(Delta_y_input):
            for j, dz in enumerate(Delta_z_input):
                Exponential = torch.exp(1j * (k2 * dy + k3 * dz))
                I = self._quad23(Phi13 * Exponential)
                den = torch.sqrt(F1 * F3)
                Chi[:, i, j] = torch.real(I / den)
        return Chi

    @torch.jit.export
    def EddyLifetime(self, k: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        Eddy Lifetime evaluation conditional branching function to individual implementations

        Parameters
        ----------
        k : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        Exception
            _description_
        """

        if k is None:
            k = self.k
        else:
            self._exp_scales()
        kL = self.LengthScale * k.norm(dim=-1)

        if self.type_EddyLifetime == EddyLifetimeType.CONST:
            tau = torch.ones_like(kL)
        elif (
            self.type_EddyLifetime == EddyLifetimeType.MANN
        ):  # NOTE: uses numpy --> cannot be backpropagated
            tau = MannEddyLifetime(kL)
        elif self.type_EddyLifetime == EddyLifetimeType.TWOTHIRD:
            tau = kL ** (-2 / 3)
        elif self.type_EddyLifetime == EddyLifetimeType.TAUNET:
            tau0 = self.InitialGuess_EddyLifetime(k.norm(dim=-1))
            tau = tau0 + self.tauNet(k)
        else:
            raise ValueError("Provided EddyLifetimeType is not implemented")

        return self.TimeScale * tau

    @torch.jit.export
    def InitialGuess_EddyLifetime(self, k_norm):
        # tau0 = MannEddyLifetime(0.59*k_norm)
        # tau0 = k_norm**(-2/3)
        tau0 = 0
        return tau0

    ###-------------------------------------------

    @torch.jit.export
    def PowerSpectra(self):
        if self.type_PowerSpectra == "RDT":
            Phi = PowerSpectraRDT(self.k, self.beta, self.E0)
        # elif self.type_PowerSpectra == 'Corrector':
        #     Corrector = self.Corrector(k)
        #     Phi = PowerSpectraCorr(self.k, beta, E0, Corrector)
        else:
            raise Exception("Wrong PowerSpectra model !")
        return Phi

    @torch.jit.export
    def _quad23(self, f: torch.Tensor) -> torch.Tensor:
        """
        Integrates f in k2 and k3 using the trapezoid rule.

        Parameters
        ----------
        f : torch.Tensor
            Integrand for trapezoid rule.

        Returns
        -------
        torch.Tensor
            _description_
        """

        # NOTE: Integration in k3
        quad = torch.trapz(f, x=self.k[..., 2], dim=-1)

        # NOTE: Integration in k2. It is sufficient to set k3 = 0, since
        #   slices are identical in meshgrid
        quad = torch.trapz(quad, x=self.k[..., 0, 1], dim=-1)
        return quad

    @torch.jit.export
    def get_div(self, Phi: torch.Tensor) -> torch.Tensor:
        """
        Calculates the divergence of the velocity-spectrum tensor

        Parameters
        ----------
        Phi : torch.Tensor
            Velocity-spectrum tensor

        Returns
        -------
        torch.Tensor
            Divergence of the input Phi
        """

        k1, k2, k3 = self.freq
        Phi11, Phi22, Phi33, Phi13, Phi12, Phi23 = Phi
        div = torch.stack(
            [
                k1 * Phi11 + k2 * Phi12 + k3 * Phi13,
                k1 * Phi12 + k2 * Phi22 + k3 * Phi23,
                k1 * Phi13 + k2 * Phi23 + k3 * Phi33,
            ]
        ) / (1 / 3 * (Phi11 + Phi22 + Phi33))
        return div

    def _init_parameters(self):
        """
        Quick subroutine for the constructor to set up the length scales
        """
        LengthScale = 0.7 * 42
        TimeScale = 3.9
        Magnitude = 1.0
        self.logLengthScale = nn.Parameter(
            torch.tensor(log(LengthScale), dtype=torch.float64)
        )
        self.logTimeScale = nn.Parameter(
            torch.tensor(log(TimeScale), dtype=torch.float64)
        )
        self.logMagnitude = nn.Parameter(
            torch.tensor(log(Magnitude), dtype=torch.float64)
        )

    def _init_grids(self):
        """
        Quick subroutine for the constructor to set up the k2, k3 grids
        """

        # k2 grid
        p1, p2, N = -2, 2, 100
        grid_zero = torch.tensor([0], dtype=torch.float64)
        grid_plus = torch.logspace(p1, p2, N, dtype=torch.float64) ** 2
        grid_minus = -torch.flip(grid_plus, dims=[0])
        self.grid_k2 = torch.cat((grid_minus, grid_zero, grid_plus)).detach()

        # k3 grid
        p1, p2, N = -2, 2, 100
        grid_zero = torch.tensor([0], dtype=torch.float64)
        grid_plus = torch.logspace(p1, p2, N, dtype=torch.float64) ** 2
        grid_minus = -torch.flip(grid_plus, dims=[0])
        self.grid_k3 = torch.cat((grid_minus, grid_zero, grid_plus)).detach()

        self.meshgrid23 = torch.meshgrid(self.grid_k2, self.grid_k3, indexing="ij")

    def _exp_scales(self) -> tuple[float, float, float]:
        """
        Takes exp of each log length scale

        Returns
        -------
        tuple[float, float, float]
            A tuple of the LengthScale, TimeScale, and Magnitude items in that order
        """
        self.LengthScale = torch.exp(self.logLengthScale)
        self.TimeScale = torch.exp(self.logTimeScale)
        self.Magnitude = torch.exp(self.logMagnitude)
        return self.LengthScale.item(), self.TimeScale.item(), self.Magnitude.item()
