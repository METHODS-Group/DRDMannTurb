from typing import Optional

import torch
import torch.nn as nn

from drdmannturb.nn_modules import CustomNet, TauNet, TauResNet
from drdmannturb.power_spectra_rdt import PowerSpectraRDT
from drdmannturb.shared.common import MannEddyLifetime, VKEnergySpectrum
from drdmannturb.shared.enums import EddyLifetimeType, PowerSpectraType
from drdmannturb.shared.parameters import NNParameters


class OnePointSpectra(nn.Module):
    """
    One point spectra implementation
    """

    def __init__(
        self,
        type_eddy_lifetime: EddyLifetimeType = EddyLifetimeType.TWOTHIRD,
        type_power_spectra: PowerSpectraType = PowerSpectraType.RDT,
        nn_parameters: NNParameters = NNParameters(),
        learn_nu: bool = False,
    ):
        super(OnePointSpectra, self).__init__()

        self.type_EddyLifetime = type_eddy_lifetime
        self.type_PowerSpectra = type_power_spectra

        # k2 grid
        p1, p2, N = -3, 3, 100
        grid_zero = torch.tensor([0], dtype=torch.float64)
        grid_plus = torch.logspace(p1, p2, N, dtype=torch.float64)
        grid_minus = -torch.flip(grid_plus, dims=[0])
        self.grid_k2 = torch.cat((grid_minus, grid_zero, grid_plus)).detach()

        # k3 grid
        p1, p2, N = -3, 3, 100
        grid_zero = torch.tensor([0], dtype=torch.float64)
        grid_plus = torch.logspace(p1, p2, N, dtype=torch.float64)
        grid_minus = -torch.flip(grid_plus, dims=[0])
        self.grid_k3 = torch.cat((grid_minus, grid_zero, grid_plus)).detach()

        self.meshgrid23 = torch.meshgrid(self.grid_k2, self.grid_k3, indexing="ij")

        self.logLengthScale = nn.Parameter(torch.tensor(0, dtype=torch.float64))
        self.logTimeScale = nn.Parameter(torch.tensor(0, dtype=torch.float64))
        self.logMagnitude = nn.Parameter(torch.tensor(0, dtype=torch.float64))

        if self.type_EddyLifetime == EddyLifetimeType.TAUNET:
            # TODO -- FIX TAUNET
            self.tauNet = TauNet(nn_parameters)
            # self.tauNet = tauNet(n_layers, hidden_layer_size, n_modes, learn_nu)

        elif self.type_EddyLifetime == EddyLifetimeType.CUSTOMMLP:
            """
            Requires n_layers, activations, n_modes, learn_nu
            """
            self.tauNet = CustomNet(nn_parameters.nlayers, learn_nu=learn_nu)
            # self.tauNet = customNet(n_layers, hidden_layer_size)

        elif self.type_EddyLifetime == EddyLifetimeType.TAURESNET:
            """
            Requires hidden_layer_sizes, n_modes, learn_nu
            """

            self.tauNet = TauResNet(nn_parameters)
            # self.tauNet = TauResNet(hidden_layer_sizes, n_modes, learn_nu)

    def exp_scales(self) -> tuple[float, float, float]:
        """
        Exponentiates the length, time, and magnitude scales

        Returns
        -------
        tuple[float, float, float]
            Returns .item() on each of the length, time, and magnitude scales
        """
        self.LengthScale = torch.exp(self.logLengthScale)  # NOTE: this is L
        self.TimeScale = torch.exp(self.logTimeScale)  # NOTE: this is gamma
        self.Magnitude = torch.exp(self.logMagnitude)  # NOTE: this is sigma
        return self.LengthScale.item(), self.TimeScale.item(), self.Magnitude.item()

    def forward(self, k1_input: torch.Tensor) -> torch.Tensor:
        """
        Forward method implementation

        Parameters
        ----------
        k1_input : torch.Tensor
            Network input

        Returns
        -------
        torch.Tensor
            Network output
        """
        self.exp_scales()
        self.k = torch.stack(
            torch.meshgrid(k1_input, self.grid_k2, self.grid_k3, indexing="ij"), dim=-1
        )
        self.k123 = self.k[..., 0], self.k[..., 1], self.k[..., 2]
        self.beta = self.EddyLifetime()
        self.k0 = self.k.clone()
        self.k0[..., 2] = self.k[..., 2] + self.beta * self.k[..., 0]
        k0L = self.LengthScale * self.k0.norm(dim=-1)
        self.E0 = self.Magnitude * VKEnergySpectrum(k0L)
        self.Phi = self.PowerSpectra()
        kF = torch.stack([k1_input * self.quad23(Phi) for Phi in self.Phi])
        return kF

    @torch.jit.export
    def EddyLifetime(self, k: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Eddy lifetime evaluation

        Parameters
        ----------
        k : Optional[torch.Tensor], optional
            _description_, by default None

        Returns
        -------
        torch.Tensor
            _description_

        Raises
        ------
        Exception
            _description_
        """
        if k is None:
            k = self.k
        else:
            self.exp_scales()

        kL = self.LengthScale * k.norm(dim=-1)

        if self.type_EddyLifetime == EddyLifetimeType.CONST:
            tau = torch.ones_like(kL)
        elif (
            self.type_EddyLifetime == EddyLifetimeType.MANN
        ):  # uses numpy - can not be backpropagated !!
            tau = MannEddyLifetime(kL)
        elif self.type_EddyLifetime == EddyLifetimeType.TWOTHIRD:
            tau = kL ** (-2 / 3)
        # elif self.type_EddyLifetime in ["tauNet", "customMLP", "tauResNet"]:
        elif self.type_EddyLifetime in [
            EddyLifetimeType.TAUNET,
            EddyLifetimeType.CUSTOMMLP,
            EddyLifetimeType.TAURESNET,
        ]:
            tau0 = self.InitialGuess_EddyLifetime(kL)
            tau = tau0 + self.tauNet(
                k * self.LengthScale
            )  # This takes a vector as input
        # elif self.type_EddyLifetime == 'customMLP':
        #    tau0 = self.InitialGuess_EddyLifetime(kL)
        #    tau = tau0 + self.tauNet(k * self.LengthScale)
        # elif self.type_EddyLifetime == 'tauResNet':
        #    tauResNet
        else:
            raise Exception("Wrong EddyLifetime model !")
        return self.TimeScale * tau

    @torch.jit.export
    def InitialGuess_EddyLifetime(self, kL_norm):
        """
        Initial guess implementation

        Parameters
        ----------
        kL_norm : _type_
            _description_

        Returns
        -------
        float
            Initial guess evaluation
        """

        # NOTE: zenodo initial guess suggests 0 as return here
        # tau0 = MannEddyLifetime(kL_norm)
        # return tau0
        tau0 = 0.0
        return tau0

    @torch.jit.export
    def PowerSpectra(self):
        """
        Calls the RDT Power Spectra model

        Returns
        -------
        torch.Tensor
            RDT evaluation

        Raises
        ------
        Exception
            In the case that the Power Spectra is not RDT
            and therefore incorrect
        """

        if self.type_PowerSpectra == PowerSpectraType.RDT:
            return PowerSpectraRDT(self.k, self.beta, self.E0)
        else:
            raise Exception("Incorrect PowerSpectra model !")

    @torch.jit.export
    def quad23(self, f: torch.Tensor) -> torch.Tensor:
        """
        Integrates f in k2 and k3 using torch trapz quadrature

        Parameters
        ----------
        f : torch.Tensor
            Function evaluation (tensor) to integrate

        Returns
        -------
        torch.Tensor
            Evaluated double integral
        """
        # NOTE: Integration in k3
        quad = torch.trapz(f, x=self.k[..., 2], dim=-1)

        # NOTE: Integration in k2 (fixed k3=0, since slices are identical in meshgrid)
        quad = torch.trapz(quad, x=self.k[..., 0, 1], dim=-1)
        return quad

    @torch.jit.export
    def get_div(self, Phi: torch.Tensor) -> torch.Tensor:
        """
        Return divergence

        Parameters
        ----------
        Phi : torch.Tensor
            _description_

        Returns
        -------
        _type_
            _description_
        """
        k1, k2, k3 = self.freq
        Phi11, Phi22, Phi33, Phi13, Phi12, Phi23 = Phi
        div = torch.stack(
            [
                k1 * Phi11 + k2 * Phi12 + k3 * Phi13,
                k1 * Phi12 + k2 * Phi22 + k3 * Phi23,
                k1 * Phi13 + k2 * Phi23 + k3 * Phi33,
            ]
        ) / (1.0 / 3.0 * (Phi11 + Phi22 + Phi33))
        return div
