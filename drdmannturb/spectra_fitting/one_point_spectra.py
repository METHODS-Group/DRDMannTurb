from typing import Optional

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
    One point spectra implementation, including set-up of eddy lifetime function approximation with DRD models, or several classical eddy lifetime functions.
    """

    def __init__(
        self,
        type_eddy_lifetime: EddyLifetimeType,
        type_power_spectra: PowerSpectraType = PowerSpectraType.RDT,
        nn_parameters: Optional[NNParameters] = None,
        learn_nu: bool = False,
    ):
        r"""Initialization of the one point spectra class. This requires the type of eddy lifetime function to use, the power spectra type (currently only the von Karman spectra is implemented), the neural network parameters to use if a DRD model is selected, and whether or not to learn :math:`\nu` in

        .. math::
            \tau(\boldsymbol{k})=\frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}{\left(1+|\boldsymbol{a}|^2\right)^{\nu / 2}}, \quad \boldsymbol{a}=\boldsymbol{a}(\boldsymbol{k}.

        Here,

        .. math::
            \boldsymbol{a}(\boldsymbol{k}):=\operatorname{abs}(\boldsymbol{k})+\mathrm{NN}(\operatorname{abs}(\boldsymbol{k}))

        if a neural network is used to learn the eddy lifetime function. For a discussion of the details and training, refer to the original DRD paper by Keith et al.

        Non-neural network eddy lifetime functions are provided as well, specifically the Mann model. The default power spectra used is due to von Karman.

        Parameters
        ----------
        type_eddy_lifetime : EddyLifetimeType, optional
            Type of eddy lifetime function :math:`\tau` to use.
        type_power_spectra : PowerSpectraType, optional
            Type of power spectra function to use, by default PowerSpectraType.RDT, the only one currently implemented.
        nn_parameters : NNParameters, optional
            Dataclass containing neural network initialization if a neural network is used to approximate the eddy lifetime function, by default None.
        learn_nu : bool, optional
            Whether or not to learn :math:`\nu`, by default False.

        Raises
        ------
        ValueError
            "Selected neural network-based eddy lifetime function approximation but did not specify neural network parameters. Pass a constructed NNParameters object."
        """
        super(OnePointSpectra, self).__init__()

        if (
            type_eddy_lifetime
            in [
                EddyLifetimeType.TAUNET,
                EddyLifetimeType.CUSTOMMLP,
                EddyLifetimeType.TAURESNET,
            ]
            and nn_parameters is None
        ):
            raise ValueError(
                "Selected neural network-based eddy lifetime function approximation but did not specify neural network parameters. Pass a constructed NNParameters object."
            )

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
        """Sets scalar values for values used in non-dimensionalization.

        Parameters
        ----------
        LengthScale : np.float64
            Length scale.
        TimeScale : np.float64
            Time scale.
        Magnitude : np.float64
            Spectrum amplitude magnitude.
        """
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
        r"""
        Evaluation of one point spectra

        .. math::
            \widetilde{J}_i(f ; \boldsymbol{\theta})=C k_1 \widetilde{F}_{i i}\left(k_1 z ; \boldsymbol{\theta}\right),

        defined by equations 6 (a-d) in the original DRD paper. Here,

        .. math::
            \widetilde{F}_{i j}\left(k_1 ; \boldsymbol{\theta}\right)=\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \Phi_{i j}^{\mathrm{DRD}}(\boldsymbol{k}, \boldsymbol{\theta}) \mathrm{d} k_2 \mathrm{~d} k_3.

        Parameters
        ----------
        k1_input : torch.Tensor
            Discrete :math:`k_1` wavevector domain.

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
        r"""Initializes Mann eddy lifetime function approximation by performing a linear regression in log-log space on
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
        r"""
        Evaluation of eddy lifetime function :math:`\tau` constructed during object initialization. This may be the Mann model or a DRD neural network that learns :math:`\tau`.

        Parameters
        ----------
        k : Optional[torch.Tensor], optional
            Wavevector domain on which to evaluate the eddy lifetime function, by default None, which defaults to grids in logspace(-3, 3).

        Returns
        -------
        torch.Tensor
            Evaluation of current eddylifetime function on provided domain.

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
        ):  # uses numpy - can not be backpropagated, also CPU only.
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
            tau0 = self.InitialGuess_EddyLifetime()
            tau = tau0 + self.tauNet(k * self.LengthScale)
        else:
            raise Exception(
                "Did not specify an admissible EddyLifetime model. Refer to the EddyLifetimeType documentation."
            )

        return self.TimeScale * tau

    @torch.jit.export
    def InitialGuess_EddyLifetime(self):
        r"""
        Initial guess at the eddy lifetime function which the DRD model uses in learning the :math:`\tau` eddy lifetime function. By default, this is just the :math:`0` function, but later functionality may allow this to be dynamically set.

        Returns
        -------
        float
            Initial guess evaluation, presently, the :math:`0` function.
        """

        return 0.0

    @torch.jit.export
    def PowerSpectra(self):
        """
        Calls the RDT Power Spectra model with current approximation of the eddy lifetime function and the energy spectrum.

        Returns
        -------
        torch.Tensor
            RDT power spectra evaluation.

        Raises
        ------
        Exception
            In the case that the Power Spectra is not RDT
            and therefore incorrect.
        """

        if self.type_PowerSpectra == PowerSpectraType.RDT:
            return PowerSpectraRDT(self.k, self.beta, self.E0)
        else:
            raise Exception("Incorrect PowerSpectra model !")

    @torch.jit.export
    def quad23(self, f: torch.Tensor) -> torch.Tensor:
        r"""
        Computes an approximation of the integral of the discretized function :math:`f` in the dimensions defined by :math:`k_2` and :math:`k_3` using the trapezoidal rule:

        .. math::
            \int \int f(k_1 ; \mathbf{\theta}) d k_2 dk_3.

        Parameters
        ----------
        f : torch.Tensor
            Function evaluation (tensor) to integrate over the frequency domain constructed during initialization.

        Returns
        -------
        torch.Tensor
            Evaluated double integral.
        """
        # NOTE: Integration in k3
        quad = torch.trapz(f, x=self.k[..., 2], dim=-1)

        # NOTE: Integration in k2 (fixed k3=0, since slices are identical in meshgrid)
        quad = torch.trapz(quad, x=self.k[..., 0, 1], dim=-1)
        return quad

    @torch.jit.export
    def get_div(self, Phi: torch.Tensor) -> torch.Tensor:
        r"""
        Evaluates the divergence of an evaluated spectral tensor. This is evaluated simply as :math:`\textbf{k} \cdot \Phi_{\textbf{k}}` and normalized by the trace.

        Parameters
        ----------
        Phi : torch.Tensor
            Discrete evaluated spectral tensor :math:`\Phi(\textbf{k}, \tau)`, which may or may not depend on the eddy lifetime function. For instance, if the von Karman model is used, no :math:`\tau` dependence is present.

        Returns
        -------
        torch.Tensor
            Divergence of the spectral tensor in the frequency domain.
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
