from typing import Optional, Union

# these imports are only needed for obtaining the exponential approximation of the Mann eddy lifetime function.
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from drdmannturb.common import (
    Mann_linear_exponential_approx,
    MannEddyLifetime,
    VKEnergySpectrum,
)
from drdmannturb.enums import EddyLifetimeType, PowerSpectraType
from drdmannturb.nn_modules import CustomNet, TauNet, TauResNet
from drdmannturb.parameters import NNParameters
from drdmannturb.spectra_fitting.power_spectra_rdt import PowerSpectraRDT


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
        """_summary_

        Parameters
        ----------
        type_eddy_lifetime : EddyLifetimeType, optional
            _description_, by default EddyLifetimeType.TWOTHIRD
        type_power_spectra : PowerSpectraType, optional
            _description_, by default PowerSpectraType.RDT
        nn_parameters : NNParameters, optional
            _description_, by default NNParameters()
        learn_nu : bool, optional
            _description_, by default False
        """
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
            self.tauNet = TauNet(
                nn_parameters.nlayers,
                nn_parameters.hidden_layer_size,
                nn_parameters.n_modes,
                learn_nu=learn_nu,
            )

        elif self.type_EddyLifetime == EddyLifetimeType.CUSTOMMLP:
            """
            Requires n_layers, activations, n_modes, learn_nu
            """
            self.tauNet = CustomNet(
                nn_parameters.nlayers,
                nn_parameters.hidden_layer_sizes,
                nn_parameters.activations,
                nn_parameters.n_modes,
                learn_nu=learn_nu,
            )

        elif self.type_EddyLifetime == EddyLifetimeType.TAURESNET:
            """
            Requires hidden_layer_sizes, n_modes, learn_nu
            """

            self.tauNet = TauResNet(
                nn_parameters.hidden_layer_sizes,
                nn_parameters.n_modes,
                learn_nu=learn_nu,
            )

        elif self.type_EddyLifetime == EddyLifetimeType.MANN_APPROX:
            self.init_mann_linear_approx = False

    def set_scales(
        self, LengthScale: np.float64, TimeScale: np.float64, Magnitude: np.float64
    ):
        self.LengthScale_scalar = LengthScale
        self.TimeScale_scalar = TimeScale
        self.Magnitude_scalar = Magnitude

    def exp_scales(self) -> tuple[float, float, float]:
        """
        Exponentiates the length, time, and spectrum amplitude scales,

        NOTE: The first 3 parameters of self.parameters() are exactly
            - LengthScale
            - TimeScale
            - SpectrumAmplitude

        Returns
        -------
        tuple[float, float, float]
           Scalar values for each of the length, time, and magnitude scales, in that order.
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

    def init_mann_approximation(self):
        """Initializes Mann eddy lifetime function approximation by performing a linear regression in log-log space on
        a given wave space and the true output of

        .. math::
           \frac{x^{-\frac{2}{3}}}{\sqrt{{ }_2 F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-x^{-2}\right)}}

        This operation is performed once on the CPU.
        """

        kL_temp = np.logspace(-3, 3, 50)

        kL_temp = kL_temp.reshape(-1, 1)
        tau_true = np.log(
            self.TimeScale_scalar * MannEddyLifetime(self.LengthScale_scalar * kL_temp)
        )

        kL_temp_log = np.log(kL_temp)

        regressor = LinearRegression()
        # fits in log-log space since tau is nearly linear in log-log
        regressor.fit(kL_temp_log, tau_true)

        print("=" * 40)

        print(
            f"Mann Linear Approximation R2 Score in log-log space: {regressor.score(kL_temp_log, tau_true)}"
        )

        print("=" * 40)

        self.tau_approx_coeff_ = torch.tensor(regressor.coef_.flatten())
        self.tau_approx_intercept_ = torch.tensor(regressor.intercept_)

        self.init_mann_linear_approx = True

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
            Did not specify an admissible EddyLifetime model. Refer to the EddyLifetimeType documentation.
        """
        if k is None:
            k = self.k
        else:
            self.exp_scales()

        kL = self.LengthScale * k.norm(dim=-1)

        if (
            hasattr(self, "init_mann_linear_approx")
            and self.init_mann_linear_approx is False
        ):  # Mann approximation chosen but not initialized
            self.init_mann_approximation()

        if self.type_EddyLifetime == EddyLifetimeType.CONST:
            tau = torch.ones_like(kL)
        elif (
            self.type_EddyLifetime == EddyLifetimeType.MANN
        ):  # uses numpy - can not be backpropagated !!
            tau = MannEddyLifetime(kL)
        elif self.type_EddyLifetime == EddyLifetimeType.MANN_APPROX:
            tau = Mann_linear_exponential_approx(
                kL, self.tau_approx_coeff_, self.tau_approx_intercept_
            )
        elif self.type_EddyLifetime == EddyLifetimeType.TWOTHIRD:
            tau = kL ** (-2 / 3)
        elif self.type_EddyLifetime in [
            EddyLifetimeType.TAUNET,
            EddyLifetimeType.CUSTOMMLP,
            EddyLifetimeType.TAURESNET,
        ]:
            tau0 = self.InitialGuess_EddyLifetime(kL)
            tau = tau0 + self.tauNet(k * self.LengthScale)
        else:
            raise Exception(
                "Did not specify an admissible EddyLifetime model. Refer to the EddyLifetimeType documentation."
            )

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
