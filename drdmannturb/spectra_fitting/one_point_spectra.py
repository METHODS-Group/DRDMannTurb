"""Implements several "integrators" for the spectral tensor.

Several physical quantities of interest in turbulence are computed as integrals of the spectral tensor,
namely the spectral/spatial coherence functions and the one-point spectra functions. This module implements
several integrators for these quantities.
"""

# these imports are only needed for obtaining the exponential approximation of the Mann eddy lifetime function.
import numpy as np
import torch
import torch.nn as nn
from scipy.special import hyp2f1

from ..enums import EddyLifetimeType
from ..nn_modules import TauNet
from ..parameters import IntegrationParameters, NNParameters, PhysicalParameters


def MannEddyLifetime(kL: torch.Tensor | np.ndarray) -> torch.Tensor:
    r"""Evaluate the full Mann eddy lifetime function.

    The full Mann eddy lifetime function has the form

    .. math::
        \tau^{\mathrm{IEC}}(k)=\frac{(k L)^{-\frac{2}{3}}}{\sqrt{{ }_2 F_1\left(1/3, 17/6; 4/3 ;-(kL)^{-2}\right)}}

    This function can execute with input data that are either in Torch or numpy. However,

    .. warning::
        This function depends on SciPy for evaluating the hypergeometric function, meaning a GPU tensor will be returned
        to the CPU for a single evaluation and then converted back to a GPU tensor. This incurs a substantial loss of
        performance.

    Parameters
    ----------
    kL : Union[torch.Tensor, np.ndarray]
        Scaled wave number

    Returns
    -------
    torch.Tensor
        Evaluated Mann eddy lifetime function.
    """
    x = kL.cpu().detach().numpy() if torch.is_tensor(kL) else kL
    y = x ** (-2 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -(x ** (-2))))
    y = torch.tensor(y, dtype=torch.get_default_dtype()) if torch.is_tensor(kL) else y

    return y


@torch.jit.script
def VKEnergySpectrum(kL: torch.Tensor) -> torch.Tensor:
    r"""Evaluate Von Karman energy spectrum without scaling.

    .. math::
        \widetilde{E}(\boldsymbol{k}) = \left(\frac{k L}{\left(1+(k L)^2\right)^{1 / 2}}\right)^{17 / 3}.

    Parameters
    ----------
    kL : torch.Tensor
        Scaled wave number domain.

    Returns
    -------
    torch.Tensor
        Result of the evaluation
    """
    # TODO: is this a bug here? since we introduce extra factors of L
    # NOTE: Originally, this is
    #      k^(-5/3) (kL / (1 + (kL)^2)^(1/2))^(17/3)
    #    = k^(-5/3) (kL)^(17/3) / (1 + (kL)^2)^(17/6))
    # .    -- EXTRA FACTORS OF L INTRODUCED HERE
    # .  = k^(12/3) / (1 + (kL)^2)^(17/6))
    # .  = k^4 / (1 + (kL)^2)^(17/6))

    return kL**4 / (1.0 + kL**2) ** (17.0 / 6.0)


@torch.jit.script
def VKLike_EnergySpectrum(kL: torch.Tensor) -> torch.Tensor:
    r"""Evaluate Von Karman energy spectrum without scaling.

    .. math::
        \widetilde{E}(\boldsymbol{k}) = \left(\frac{k L}{\left(1+(k L)^2\right)^{1 / 2}}\right)^{17 / 3}.

    Parameters
    ----------
    kL : torch.Tensor
        Scaled wave number domain.

    Returns
    -------
    torch.Tensor
        Result of the evaluation
    """
    return kL ** (-5.0 / 3.0) * (kL / torch.sqrt(1.0 + kL**2)) ** (17.0 / 3.0)


@torch.jit.script
def Learnable_EnergySpectrum(kL: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    r"""Parametrizable energy spectrum with learnable exponents, p and q.

    .. math::
        \widetilde{E}(\boldsymbol{k}) = \left(\frac{k L}{\left(1+(k L)^2\right)^{1 / 2}}\right)^{17 / 3}.
    """
    return (kL**p) / ((1.0 + kL**2) ** q)


class OnePointSpectra(nn.Module):
    """One point spectra calculations."""

    def __init__(
        self,
        type_eddy_lifetime: EddyLifetimeType,
        physical_params: PhysicalParameters,
        nn_parameters: NNParameters | None = None,
        learn_nu: bool = False,
        use_coherence: bool = False,
        use_learnable_spectrum: bool = False,
        p_exponent: float = 4.0,
        q_exponent: float = 17.0 / 6.0,
        integration_params: IntegrationParameters | None = None,
    ):
        r"""Initialize the one point spectra calculator."""
        super().__init__()

        assert physical_params.L > 0, "Length scale L must be positive."
        assert physical_params.Gamma > 0, "Characteristic time scale Gamma must be positive."
        assert physical_params.sigma > 0, "Spectrum amplitude sigma must be positive."

        if type_eddy_lifetime == EddyLifetimeType.TAUNET:
            assert nn_parameters is not None, "TauNet EddyLifetimeType requires NNParameters!"
            self.tauNet = TauNet(
                nn_parameters.nlayers,
                nn_parameters.hidden_layer_size,
                learn_nu=learn_nu,
            )

        elif type_eddy_lifetime == EddyLifetimeType.MANN_APPROX:
            self.init_mann_linear_approx = False

        self.type_EddyLifetime = type_eddy_lifetime
        self.use_coherence = use_coherence

        self.use_learnable_spectrum = use_learnable_spectrum

        if integration_params is None:
            integration_params = IntegrationParameters()

        ####
        # OPS grid
        # k2 grid
        p1, p2, N = integration_params.ops_log_min, integration_params.ops_log_max, integration_params.ops_num_points
        grid_zero = torch.tensor([0])
        grid_plus = torch.logspace(p1, p2, N)
        grid_minus = -torch.flip(grid_plus, dims=[0])
        self.ops_grid_k2 = torch.cat((grid_minus, grid_zero, grid_plus)).detach()

        # k3 grid
        p1, p2, N = integration_params.ops_log_min, integration_params.ops_log_max, integration_params.ops_num_points
        grid_zero = torch.tensor([0])
        grid_plus = torch.logspace(p1, p2, N)
        grid_minus = -torch.flip(grid_plus, dims=[0])
        self.ops_grid_k3 = torch.cat((grid_minus, grid_zero, grid_plus)).detach()

        self.ops_meshgrid23 = torch.meshgrid(self.ops_grid_k2, self.ops_grid_k3, indexing="ij")

        ####
        # Separate coherence grid (finer resolution for better accuracy)
        if use_coherence:
            p1, p2 = integration_params.coh_log_min, integration_params.coh_log_max
            N_coh = integration_params.coh_num_points
            grid_zero_coh = torch.tensor([0])
            grid_plus_coh = torch.logspace(p1, p2, N_coh)
            grid_minus_coh = -torch.flip(grid_plus_coh, dims=[0])
            self.coh_grid_k2 = torch.cat((grid_minus_coh, grid_zero_coh, grid_plus_coh)).detach() / physical_params.L
            self.coh_grid_k3 = torch.cat((grid_minus_coh, grid_zero_coh, grid_plus_coh)).detach() / physical_params.L

            self.coh_meshgrid23 = torch.meshgrid(self.coh_grid_k2, self.coh_grid_k3, indexing="ij")

        self.logLengthScale = nn.Parameter(torch.tensor(np.log10(physical_params.L), dtype=torch.get_default_dtype()))
        self.logTimeScale = nn.Parameter(torch.tensor(np.log10(physical_params.Gamma), dtype=torch.get_default_dtype()))
        self.logMagnitude = nn.Parameter(torch.tensor(np.log10(physical_params.sigma), dtype=torch.get_default_dtype()))

        if use_learnable_spectrum:
            # raw (unconstrained) tensors
            self._raw_p = nn.Parameter(torch.tensor(np.log(np.exp(p_exponent) - 1.0)))
            self._raw_q = nn.Parameter(torch.tensor(np.log(np.exp(q_exponent) - 1.0)))

        self.LengthScale_scalar = physical_params.L
        self.TimeScale_scalar = physical_params.Gamma
        self.Magnitude_scalar = physical_params.sigma

    def set_scales(self, LengthScale: float, TimeScale: float, Magnitude: float):
        """Set scalar values for values used in non-dimensionalization.

        Parameters
        ----------
        LengthScale : float
            Length scale.
        TimeScale : float
            Time scale.
        Magnitude : float
            Spectrum amplitude magnitude.
        """
        self.LengthScale_scalar = LengthScale
        self.TimeScale_scalar = TimeScale
        self.Magnitude_scalar = Magnitude

        if self.use_learnable_spectrum:
            self.p_exponent = self._positive(self._raw_p)  # 0.3 ≤ p ≤ 6
            self.q_exponent = self._positive(self._raw_q)  # 0.3 ≤ q ≤ 6

    def exp_scales(self) -> tuple[float, float, float]:
        """Exponentiate the length, time, and spectrum amplitude scales.

        .. note::
            The first 3 parameters of self.parameters() are exactly
                - LengthScale
                - TimeScale
                - SpectrumAmplitude

        Additionally, if use_learnable_spectrum is True, the next two parameters are
            - p_low
            - q_high

        Returns
        -------
        tuple[float, float, float]
           Scalar values for each of the length, time, and magnitude scales, in that order.
        """
        self.LengthScale = torch.pow(10, self.logLengthScale)  # NOTE: this is L
        self.TimeScale = torch.pow(10, self.logTimeScale)  # NOTE: this is gamma
        self.Magnitude = torch.pow(10, self.logMagnitude)  # NOTE: this is sigma
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
        self.exp_scales()

        self.k = torch.stack(torch.meshgrid(k1_input, self.ops_grid_k2, self.ops_grid_k3, indexing="ij"), dim=-1)
        self.k123 = self.k[..., 0], self.k[..., 1], self.k[..., 2]
        self.beta = self.EddyLifetime()

        self.k0 = self.k.clone()
        self.k0[..., 2] = self.k[..., 2] + self.beta * self.k[..., 0]
        k0L = self.LengthScale * self.k0.norm(dim=-1)

        # Choose energy spectrum based on parameters
        if self.use_learnable_spectrum:
            p = self._positive(self._raw_p)  # 0.3 ≤ p ≤ 6
            q = self._positive(self._raw_q)  # 0.3 ≤ q ≤ 6
            energy_spectrum = Learnable_EnergySpectrum(k0L, p, q)
        else:
            energy_spectrum = VKLike_EnergySpectrum(k0L)

        self.E0 = self.Magnitude * self.LengthScale ** (5.0 / 3.0) * energy_spectrum

        self.Phi = self.PowerSpectra()

        kF = torch.stack([k1_input * self.quad23(Phi) for Phi in self.Phi])

        return kF

    def SpectralCoherence(
        self,
        k1_input: torch.Tensor,
        spatial_separations: torch.Tensor,
    ) -> torch.Tensor:
        r"""Evaluate spectral auto-coherence.

        Computes auto-coherence C_{ii}(r,f) = |S_{ii}(r,f)|²/|S_{ii}(0,f)|²
        where S_{ii}(r,f) = ∬ Φ_{ii}(f,k₂,k₃)e^{i·k₂·Δy} dk₂dk₃

        Parameters
        ----------
        k1_input : torch.Tensor
            Discrete k₁ wavevector domain (frequencies).
        spatial_separations : torch.Tensor
            Spatial separation values (Δy) in cross-stream direction.

        Returns
        -------
        torch.Tensor
            Auto-coherence values for u, v, w components.
            Shape: (3, n_separations, n_frequencies)
        """
        self.exp_scales()
        if not hasattr(self, "coh_grid_k2") or not hasattr(self, "coh_grid_k3"):
            p1, p2, N_coh = -3, 3, 100
            grid_zero_coh = torch.tensor([0], dtype=torch.get_default_dtype())
            grid_plus_coh = torch.logspace(p1, p2, N_coh)
            grid_minus_coh = -torch.flip(grid_plus_coh, dims=[0])

            # Scale the grid by 1/L to get proper physical units
            scale_factor = 1.0 / self.LengthScale

            self.coh_grid_k2 = torch.cat((grid_minus_coh, grid_zero_coh, grid_plus_coh)).detach() * scale_factor
            self.coh_grid_k3 = torch.cat((grid_minus_coh, grid_zero_coh, grid_plus_coh)).detach() * scale_factor

            # self.coh_grid_k2 = torch.cat((grid_minus_coh, grid_zero_coh, grid_plus_coh)).detach()
            # self.coh_grid_k3 = torch.cat((grid_minus_coh, grid_zero_coh, grid_plus_coh)).detach()

        # Create coherence-specific k-space grid
        coh_k = torch.stack(torch.meshgrid(k1_input, self.coh_grid_k2, self.coh_grid_k3, indexing="ij"), dim=-1)

        # Calculate eddy lifetime on coherence grid
        beta_coh = self.EddyLifetime(coh_k)

        # Set up distorted k-space for coherence calculation
        coh_k0 = coh_k.clone()
        coh_k0[..., 2] = coh_k[..., 2] + beta_coh * coh_k[..., 0]
        k0L_coh = self.LengthScale * coh_k0.norm(dim=-1)

        # Energy spectrum on coherence grid
        if self.use_learnable_spectrum:
            energy_spectrum_coh = Learnable_EnergySpectrum(
                k0L_coh,
                self.p_exponent,
                self.q_exponent,
            )
        else:
            energy_spectrum_coh = VKLike_EnergySpectrum(k0L_coh)

        E0_coh = self.Magnitude * self.LengthScale ** (5.0 / 3.0) * energy_spectrum_coh

        # Calculate power spectra on coherence grid
        Phi11_coh, Phi22_coh, Phi33_coh, Phi13_coh, Phi23_coh, Phi12_coh = self._PowerSpectra_coherence(
            coh_k,
            beta_coh,
            E0_coh,
        )

        # Extract k2 grid for phase calculation
        k2_grid = coh_k[..., 1]  # Shape: (n_freq, n_k2, n_k3)

        # Compute auto-spectra at zero separation (r=0)
        S11_0 = self._quad23_coherence(Phi11_coh, coh_k)  # S_uu(r=0)
        S22_0 = self._quad23_coherence(Phi22_coh, coh_k)  # S_vv(r=0)
        S33_0 = self._quad23_coherence(Phi33_coh, coh_k)  # S_ww(r=0)

        # Initialize coherence arrays
        n_seps = len(spatial_separations)
        n_freqs = len(k1_input)

        coherence_u = torch.zeros(n_seps, n_freqs)  # Auto-coherence of u
        coherence_v = torch.zeros(n_seps, n_freqs)  # Auto-coherence of v
        coherence_w = torch.zeros(n_seps, n_freqs)  # Auto-coherence of w

        for i, r in enumerate(spatial_separations):
            # Complex exponential phase factor: exp(i * k2 * r)
            phase = 1j * k2_grid * (r)
            exp_phase = torch.exp(phase)

            # Auto-power spectra with spatial separation
            S11_r = self._quad23_coherence(Phi11_coh * exp_phase, coh_k)  # S_uu(r)
            S22_r = self._quad23_coherence(Phi22_coh * exp_phase, coh_k)  # S_vv(r)
            S33_r = self._quad23_coherence(Phi33_coh * exp_phase, coh_k)  # S_ww(r)

            # Auto-coherence: |S_ii(r)|² / |S_ii(0)|²
            coherence_u[i, :] = torch.abs(S11_r) ** 2 / torch.abs(S11_0) ** 2  # C_11
            coherence_v[i, :] = torch.abs(S22_r) ** 2 / torch.abs(S22_0) ** 2  # C_22
            coherence_w[i, :] = torch.abs(S33_r) ** 2 / torch.abs(S33_0) ** 2  # C_33

        return torch.stack([coherence_u, coherence_v, coherence_w])

    def _PowerSpectra_coherence(self, k_coh: torch.Tensor, beta_coh: torch.Tensor, E0_coh: torch.Tensor):
        """Power spectra calculation on coherence grid (copy of PowerSpectra but with coherence grid)."""
        k1, k2, k3 = k_coh[..., 0], k_coh[..., 1], k_coh[..., 2]

        k30 = k3 + beta_coh * k1
        kk0 = k1**2 + k2**2 + k30**2
        kk = k1**2 + k2**2 + k3**2
        s = k1**2 + k2**2

        C1 = beta_coh * k1**2 * (kk0 - 2 * k30**2 + beta_coh * k1 * k30) / (kk * s)
        C2 = k2 * kk0 / torch.sqrt(s**3) * torch.atan2(beta_coh * k1 * torch.sqrt(s), kk0 - k30 * k1 * beta_coh)

        zeta1 = C1 - k2 / k1 * C2
        zeta2 = C1 * k2 / k1 + C2
        E0_coh /= 4 * torch.pi

        Phi11 = E0_coh / (kk0**2) * (kk0 - k1**2 - 2 * k1 * k30 * zeta1 + (k1**2 + k2**2) * zeta1**2)
        Phi22 = E0_coh / (kk0**2) * (kk0 - k2**2 - 2 * k2 * k30 * zeta2 + (k1**2 + k2**2) * zeta2**2)
        Phi33 = E0_coh / (kk**2) * (k1**2 + k2**2)
        Phi13 = E0_coh / (kk * kk0) * (-k1 * k30 + (k1**2 + k2**2) * zeta1)
        Phi12 = E0_coh / (kk0**2) * (-k1 * k2 - k1 * k30 * zeta2 - k2 * k30 * zeta1 + (k1**2 + k2**2) * zeta1 * zeta2)
        Phi23 = E0_coh / (kk * kk0) * (-k2 * k30 + (k1**2 + k2**2) * zeta2)

        # TODO: Remove this... recall bug with ``self.curves``
        # Regularization
        epsilon = 1e-32
        Phi11 = torch.where(Phi11 < epsilon, epsilon, Phi11)
        Phi22 = torch.where(Phi22 < epsilon, epsilon, Phi22)
        Phi33 = torch.where(Phi33 < epsilon, epsilon, Phi33)
        Phi13 = torch.where(torch.abs(Phi13) < epsilon, epsilon * torch.sign(Phi13), Phi13)
        Phi12 = torch.where(Phi12 < epsilon, epsilon, Phi12)
        Phi23 = torch.where(Phi23 < epsilon, epsilon, Phi23)

        return Phi11, Phi22, Phi33, Phi13, Phi23, Phi12

    def _quad23_coherence(self, f: torch.Tensor, k_coh: torch.Tensor) -> torch.Tensor:
        """Integration routine for coherence calculation using coherence grid."""
        # Integration in k3
        quad = torch.trapz(f, x=k_coh[..., 2], dim=-1)

        # Integration in k2
        quad = torch.trapz(quad, x=k_coh[..., 0, 1], dim=-1)
        return quad

    @torch.jit.export
    def quad23_complex(self, f: torch.Tensor) -> torch.Tensor:
        r"""Approximate integral of complex discretized f over frequency domain.

        This computes an approximation of the integral of complex f in the dimensions
        defined by k₂ and k₃ using the trapezoidal rule.

        Parameters
        ----------
        f : torch.Tensor
            Complex function evaluation (tensor) to integrate over the frequency domain.

        Returns
        -------
        torch.Tensor
            Evaluated complex double integral.
        """
        # NOTE: Integration in k3
        quad = torch.trapz(f, x=self.k[..., 2], dim=-1)

        # NOTE: Integration in k2
        quad = torch.trapz(quad, x=self.k[..., 0, 1], dim=-1)
        return quad

    @torch.jit.export
    def EddyLifetime(self, k: torch.Tensor | None = None) -> torch.Tensor:
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
        elif self.type_EddyLifetime == EddyLifetimeType.TWOTHIRD:
            tau = kL ** (-2 / 3)
        elif self.type_EddyLifetime == EddyLifetimeType.TAUNET:
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

    def _positive(self, raw, lower=0.3, upper=6.0):
        # softplus keeps it >0; clamp keeps it inside a safe range
        return torch.clamp(torch.nn.functional.softplus(raw), min=lower, max=upper)
