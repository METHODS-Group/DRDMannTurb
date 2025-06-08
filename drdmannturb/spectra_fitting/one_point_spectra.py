"""Implements the one point spectra."""

from typing import Optional

# these imports are only needed for obtaining the exponential approximation of the Mann eddy lifetime function.
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from ..common import Mann_linear_exponential_approx, MannEddyLifetime, VKEnergySpectrum
from ..enums import EddyLifetimeType, PowerSpectraType
from ..nn_modules import CustomNet, TauNet
from ..parameters import NNParameters, PhysicalParameters


class OnePointSpectra(nn.Module):
    """One point spectra calculations.

    This includes set-up of eddy lifetime function approximation with DRD models, or several classical eddy lifetime
    functions.
    """

    def __init__(
        self,
        type_eddy_lifetime: EddyLifetimeType,
        physical_params: PhysicalParameters,
        type_power_spectra: PowerSpectraType = PowerSpectraType.RDT,
        nn_parameters: Optional[NNParameters] = None,
        learn_nu: bool = False,
    ):
        r"""Initialize the one point spectra calculator.

        This requires the type of eddy lifetime function to use, the
        power spectra type (currently only the von Karman spectra is implemented), the neural network parameters to use
        if a DRD model is selected, and whether or not to learn :math:`\nu` in

        .. math::
            \tau(\boldsymbol{k})=\frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}
            {\left(1+|\boldsymbol{a}|^2\right)^{\nu / 2}}, \quad \boldsymbol{a}=\boldsymbol{a}(\boldsymbol{k}).

        Here,

        .. math::
            \boldsymbol{a}(\boldsymbol{k}):=\operatorname{abs}(\boldsymbol{k})+
            \mathrm{NN}(\operatorname{abs}(\boldsymbol{k}))

        if a neural network is used to learn the eddy lifetime function. For a discussion of the details and training,
        refer to the original DRD paper by Keith et al.

        Non-neural network eddy lifetime functions are provided as well, specifically the Mann model. The default power
        spectra used is due to von Karman.

        Parameters
        ----------
        type_eddy_lifetime : EddyLifetimeType, optional
            Type of eddy lifetime function :math:`\tau` to use.
        physical_params : PhysicalParameters,
            Object specifying physical parameters of the problem.
        type_power_spectra : PowerSpectraType, optional
            Type of power spectra function to use, by default PowerSpectraType.RDT, the only one currently implemented.
        nn_parameters : NNParameters, optional
            Dataclass containing neural network initialization if a neural network is used to approximate the eddy
            lifetime function, by default None.
        learn_nu : bool, optional
            Whether or not to learn :math:`\nu`, by default False.

        Raises
        ------
        ValueError
            "Selected neural network-based eddy lifetime function approximation but did not specify neural network
            parameters. Pass a constructed NNParameters object."
        """
        super().__init__()

        if type_eddy_lifetime == EddyLifetimeType.TAUNET:
            assert nn_parameters is not None, "TauNet EddyLifetimeType requires NNParameters!"
            self.tauNet = TauNet(
                nn_parameters.nlayers,
                nn_parameters.hidden_layer_size,
                learn_nu=learn_nu,
                k_inf_asymptote=physical_params.k_inf_asymptote,
            )

        elif type_eddy_lifetime == EddyLifetimeType.CUSTOMMLP:
            assert nn_parameters is not None, "Custom MLP EddyLifetimeType requires NNParameters!"
            self.tauNet = CustomNet(
                nn_parameters.nlayers,
                nn_parameters.hidden_layer_sizes,
                nn_parameters.activations,
                learn_nu=learn_nu,
                k_inf_asymptote=physical_params.k_inf_asymptote,
            )

        elif type_eddy_lifetime == EddyLifetimeType.MANN_APPROX:
            self.init_mann_linear_approx = False

        self.type_EddyLifetime = type_eddy_lifetime
        self.type_PowerSpectra = type_power_spectra

        ####
        # OPS grid
        # k2 grid
        p1, p2, N = -3, 3, 100
        grid_zero = torch.tensor([0], dtype=torch.float64)
        grid_plus = torch.logspace(p1, p2, N, dtype=torch.float64)
        grid_minus = -torch.flip(grid_plus, dims=[0])
        self.ops_grid_k2 = torch.cat((grid_minus, grid_zero, grid_plus)).detach()

        # k3 grid
        p1, p2, N = -3, 3, 100
        grid_zero = torch.tensor([0], dtype=torch.float64)
        grid_plus = torch.logspace(p1, p2, N, dtype=torch.float64)
        grid_minus = -torch.flip(grid_plus, dims=[0])
        self.ops_grid_k3 = torch.cat((grid_minus, grid_zero, grid_plus)).detach()

        self.ops_meshgrid23 = torch.meshgrid(self.ops_grid_k2, self.ops_grid_k3, indexing="ij")

        assert physical_params.L > 0, "Length scale L must be positive."
        assert physical_params.Gamma > 0, "Characteristic time scale Gamma must be positive."
        assert physical_params.sigma > 0, "Spectrum amplitude sigma must be positive."

        self.logLengthScale = nn.Parameter(torch.tensor(np.log10(physical_params.L), dtype=torch.float64))
        self.logTimeScale = nn.Parameter(torch.tensor(np.log10(physical_params.Gamma), dtype=torch.float64))
        self.logMagnitude = nn.Parameter(torch.tensor(np.log10(physical_params.sigma), dtype=torch.float64))

    def set_scales(self, LengthScale: np.float64, TimeScale: np.float64, Magnitude: np.float64):
        """Set scalar values for values used in non-dimensionalization.

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
        """Exponentiate the length, time, and spectrum amplitude scales.

        .. note::
            The first 3 parameters of self.parameters() are exactly
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
        r"""Evaluate one point spectra.

        .. math::
            \widetilde{J}_i(f ; \boldsymbol{\theta})=C k_1 \widetilde{F}_{i i}\left(k_1 z ; \boldsymbol{\theta}\right),

        defined by equations 6 (a-d) in the original DRD paper. Here,

        .. math::
            \widetilde{F}_{i j}\left(k_1 ; \boldsymbol{\theta}\right)=\int_{-\infty}^{\infty} \int_{-\infty}^{\infty}
            \Phi_{i j}^{\mathrm{DRD}}(\boldsymbol{k}, \boldsymbol{\theta}) \mathrm{d} k_2 \mathrm{~d} k_3.

        Parameters
        ----------
        k1_input : torch.Tensor
            Discrete :math:`k_1` wavevector domain.

        Returns
        -------
        torch.Tensor
            Network output
        """
        # print(f"\n[DEBUG OPS.forward] Input k1_input shape: {k1_input.shape}")
        # print(f"[DEBUG OPS.forward] k1_input range: [{k1_input.min().item():.3e}, {k1_input.max().item():.3e}]")

        self.exp_scales()
        # print(f"[DEBUG OPS.forward] Scales: L={self.LengthScale}, Gamma={self.TimeScale}, sigma={self.Magnitude}")

        self.k = torch.stack(torch.meshgrid(k1_input, self.ops_grid_k2, self.ops_grid_k3, indexing="ij"), dim=-1)
        self.k123 = self.k[..., 0], self.k[..., 1], self.k[..., 2]
        self.beta = self.EddyLifetime()
        # print(f"[DEBUG OPS.forward] beta range: [{self.beta.min().item():.3e}, {self.beta.max().item():.3e}]")
        # print(f"[DEBUG OPS.forward] Any NaN in beta? {torch.isnan(self.beta).any().item()}")

        self.k0 = self.k.clone()
        self.k0[..., 2] = self.k[..., 2] + self.beta * self.k[..., 0]
        k0L = self.LengthScale * self.k0.norm(dim=-1)
        # print(f"[DEBUG OPS.forward] k0L range: [{k0L.min().item():.3e}, {k0L.max().item():.3e}]")

        self.E0 = self.Magnitude * self.LengthScale ** (5.0 / 3.0) * VKEnergySpectrum(k0L)
        # print(f"[DEBUG OPS.forward] E0 range: [{self.E0.min().item():.3e}, {self.E0.max().item():.3e}]")
        # print(f"[DEBUG OPS.forward] Any NaN in E0? {torch.isnan(self.E0).any().item()}")

        self.Phi = self.PowerSpectra()
        # print(f"[DEBUG OPS.forward] Number of Phi components: {len(self.Phi)}")
        # for i, phi in enumerate(self.Phi):
        #     print(f"[DEBUG OPS.forward] Phi[{i}] range: [{phi.min().item():.3e}, {phi.max().item():.3e}], "
        #           f"NaN? {torch.isnan(phi).any().item()}")

        kF = torch.stack([k1_input * self.quad23(Phi) for Phi in self.Phi])
        # print(f"[DEBUG OPS.forward] Final kF shape: {kF.shape}")
        # print(f"[DEBUG OPS.forward] Final kF range: [{kF.min().item():.3e}, {kF.max().item():.3e}]")
        # print(f"[DEBUG OPS.forward] Any NaN in kF? {torch.isnan(kF).any().item()}")

        return kF

    def SpectralCoherence(
        self,
        k1_input: torch.Tensor,
    ) -> torch.Tensor:
        r"""Evaluate spectral coherence.

        Parameters
        ----------
        k1_input : torch.Tensor
            Discrete :math:`k_1` wavevector domain.

        Returns
        -------
        torch.Tensor
            Spectral coherence.
        """
        self.k = torch.stack(torch.meshgrid(k1_input, self.ops_grid_k2, self.ops_grid_k3, indexing="ij"), dim=-1)
        pass

    def init_mann_approximation(self):
        r"""Initialize Mann eddy lifetime function approximation.

        This is done by performing a linear regression in log-log space on a given wave space and the true output of

        .. math::
           \frac{x^{-\frac{2}{3}}}{\sqrt{{ }_2 F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-x^{-2}\right)}}

        This operation is performed once on the CPU.
        """
        kL_temp = np.logspace(-3, 3, 50)

        kL_temp = kL_temp.reshape(-1, 1)
        tau_true = np.log(self.TimeScale_scalar * MannEddyLifetime(self.LengthScale_scalar * kL_temp))

        kL_temp_log = np.log(kL_temp)

        regressor = LinearRegression()
        # fits in log-log space since tau is nearly linear in log-log
        regressor.fit(kL_temp_log, tau_true)

        print("=" * 40)

        print(f"Mann Linear Approximation R2 Score in log-log space: {regressor.score(kL_temp_log, tau_true)}")

        print("=" * 40)

        self.tau_approx_coeff_ = torch.tensor(regressor.coef_.flatten())
        self.tau_approx_intercept_ = torch.tensor(regressor.intercept_)

        self.init_mann_linear_approx = True

    @torch.jit.export
    def EddyLifetime(self, k: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Evaluate eddy lifetime function :math:`\tau` constructed during object initialization.

        This may be the Mann model or a DRD neural network that learns :math:`\tau`.

        Parameters
        ----------
        k : Optional[torch.Tensor], optional
            Wavevector domain on which to evaluate the eddy lifetime function, by default None, which defaults to grids
            in logspace(-3, 3).

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
            hasattr(self, "init_mann_linear_approx") and self.init_mann_linear_approx is False
        ):  # Mann approximation chosen but not initialized
            self.init_mann_approximation()

        if self.type_EddyLifetime == EddyLifetimeType.CONST:
            tau = torch.ones_like(kL)
        elif self.type_EddyLifetime == EddyLifetimeType.MANN:  # uses numpy - can not be backpropagated, also CPU only.
            tau = MannEddyLifetime(kL)
        elif self.type_EddyLifetime == EddyLifetimeType.MANN_APPROX:
            tau = Mann_linear_exponential_approx(kL, self.tau_approx_coeff_, self.tau_approx_intercept_)
        elif self.type_EddyLifetime == EddyLifetimeType.TWOTHIRD:
            tau = kL ** (-2 / 3)
        elif self.type_EddyLifetime in [
            EddyLifetimeType.TAUNET,
            EddyLifetimeType.CUSTOMMLP,
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
        r"""Set initial guess for the eddy lifetime function.

        This is the initialization for the DRD models for learning the eddy lifetime function :math:`\tau`. Currently,
        this is just the :math:`0` function, but later functionality may allow this to be dynamically set.

        TODO: This should depend on the scale of the data, eg. for Kaimal, 0.0 is better
            but for the STORM data, the initial guess should be higher.

        Returns
        -------
        float
            Initial guess evaluation. (Presently, constant function)
        """
        return 0.0

    @torch.jit.export
    def PowerSpectra(self):
        r"""Classical rapid distortion spectra.

        This is the solution to

        .. math::
            \frac{\bar{D} \mathrm{~d} Z_j(\boldsymbol{k}, t)}{\bar{D} t}=\frac{\partial U_{\ell}}{\partial x_k}
            \left(2 \frac{k_j k_{\ell}}{k^2}-\delta_{j \ell}\right) \mathrm{d} Z_k(\boldsymbol{k}, t)

        given by

        .. math::
            \mathrm{d} \mathbf{Z}(\boldsymbol{k}(t), t)=\boldsymbol{D}_\tau(\boldsymbol{k}) \mathrm{d} \mathbf{Z}
            \left(\boldsymbol{k}_0, 0\right).

        Refer to the original DRD paper, Section III, subsection B for a full expansion.

        Parameters
        ----------
        k : torch.Tensor
            Wave vector domain.
        beta : torch.Tensor
            Evaluated eddy lifetime function.
        E0 : torch.Tensor
            Evaluated and non-dimensionalized von Karman energy spectrum.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
            6-tuple of the components of the velocity-spectrum tensor in the order:
            :math:`\Phi_{11}, \Phi_{22}, \Phi_{33}, \Phi_{13}, \Phi_{12}, \Phi_{23}`.
        """
        k = self.k
        beta = self.beta
        E0 = self.E0

        # BEGIN rdt power spectra calculation.
        k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]

        k30 = k3 + beta * k1
        kk0 = k1**2 + k2**2 + k30**2
        kk = k1**2 + k2**2 + k3**2
        s = k1**2 + k2**2

        # Debug prints (Note: might need to remove @torch.jit.script decorator temporarily)
        # print(f"[DEBUG PowerSpectraRDT] s min: {s.min().item():.3e}, zeros: {(s == 0).sum().item()}")
        # print(f"[DEBUG PowerSpectraRDT] kk min: {kk.min().item():.3e}, zeros: {(kk == 0).sum().item()}")
        # print(f"[DEBUG PowerSpectraRDT] kk0 min: {kk0.min().item():.3e}, zeros: {(kk0 == 0).sum().item()}")

        C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
        C2 = k2 * kk0 / torch.sqrt(s**3) * torch.atan2(beta * k1 * torch.sqrt(s), kk0 - k30 * k1 * beta)

        zeta1 = C1 - k2 / k1 * C2
        zeta2 = C1 * k2 / k1 + C2
        E0 /= 4 * torch.pi
        Phi11 = E0 / (kk0**2) * (kk0 - k1**2 - 2 * k1 * k30 * zeta1 + (k1**2 + k2**2) * zeta1**2)
        Phi22 = E0 / (kk0**2) * (kk0 - k2**2 - 2 * k2 * k30 * zeta2 + (k1**2 + k2**2) * zeta2**2)
        Phi33 = E0 / (kk**2) * (k1**2 + k2**2)
        Phi13 = E0 / (kk * kk0) * (-k1 * k30 + (k1**2 + k2**2) * zeta1)

        Phi12 = E0 / (kk0**2) * (-k1 * k2 - k1 * k30 * zeta2 - k2 * k30 * zeta1 + (k1**2 + k2**2) * zeta1 * zeta2)
        Phi23 = E0 / (kk * kk0) * (-k2 * k30 + (k1**2 + k2**2) * zeta2)

        # DEBUG: add a small epsilon to prevent extremely small values
        epsilon = 1e-12

        # DEBUG: test sign of Phi13 -- are we just resetting it to 1e-12 since it's negative?
        # print(f"[DEBUG PowerSpectraRDT] Phi13 sign: {torch.sign(Phi13.mean()).item()}")

        Phi11 = torch.where(Phi11 < epsilon, epsilon, Phi11)
        Phi22 = torch.where(Phi22 < epsilon, epsilon, Phi22)
        Phi33 = torch.where(Phi33 < epsilon, epsilon, Phi33)
        Phi13 = torch.where(torch.abs(Phi13) < epsilon, epsilon * torch.sign(Phi13), Phi13)
        Phi12 = torch.where(Phi12 < epsilon, epsilon, Phi12)
        Phi23 = torch.where(Phi23 < epsilon, epsilon, Phi23)

        # In order, uu, vv, ww, uw, vw, uv
        return Phi11, Phi22, Phi33, Phi13, Phi23, Phi12

    @torch.jit.export
    def quad23(self, f: torch.Tensor) -> torch.Tensor:
        r"""Approximate integral of discretized :math:`f` over frequency domain.

        This computes an approximation of the integral of :math:`f` in the dimensions defined by
        :math:`k_2` and :math:`k_3` using the trapezoidal rule:

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
        r"""Evaluate the divergence of an evaluated spectral tensor.

        This is evaluated simply as :math:`\textbf{k} \cdot \Phi_{\textbf{k}}` and normalized by the trace.

        Parameters
        ----------
        Phi : torch.Tensor
            Discrete evaluated spectral tensor :math:`\Phi(\textbf{k}, \tau)`, which may or may not depend on the eddy
            lifetime function. For instance, if the von Karman model is used, no :math:`\tau` dependence is present.

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
