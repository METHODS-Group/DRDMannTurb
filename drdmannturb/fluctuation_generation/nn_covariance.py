"""Covariance kernel implementation for trained DRD spectral tensor models."""

import numpy as np
import torch

from ..spectra_fitting.spectral_tensor_models import SpectralTensorModel
from .covariance_kernels import Covariance, _assemble_rdt_sqrt_tensor


class NNCovariance(Covariance):
    r"""
    Covariance kernel built from a trained (DRD) spectral tensor model.

    Like other covariance kernel implementations, this evaluates the
    :math:`G(\boldsymbol{k})` which satisfies :math:`G(\boldsymbol{k}) G^*(\boldsymbol{k})=\Phi(\boldsymbol{k},
    \tau(\boldsymbol{k}))`, where the spectral tensor is defined through the eddy lifetime and energy spectrum
    models of a trained :py:class:`SpectralTensorModel`. In particular, the eddy lifetime function may be the
    neural network (``TauNet_ELT``) learned during spectra fitting.

    Since spectra fitting operates on a non-dimensional frequency domain while field generation operates on a
    physical (dimensional) domain, this kernel requires the affine map between the two:

    .. math::
        \boldsymbol{k}_{\mathrm{model}} = \ell \, \boldsymbol{k}_{\mathrm{phys}}, \qquad
        (k_1 F_{ij})_{\mathrm{phys}}(k_{\mathrm{phys}}) = A \, (k_1 F_{ij})_{\mathrm{model}}(\ell k_{\mathrm{phys}}),

    where :math:`\ell` (``length_scale``, units of length) maps physical wavevectors onto the domain the model
    was trained on and :math:`A` (``amplitude``, units of velocity squared) maps the model's non-dimensional
    frequency-weighted spectra back to physical units. The full spectral tensor then satisfies
    :math:`\Phi_{\mathrm{phys}}(\boldsymbol{k}) = A \ell^3 \, \Phi_{\mathrm{model}}(\ell \boldsymbol{k})`.

    For example, for data non-dimensionalized as :math:`n = k_1 z_{\mathrm{ref}} / 2\pi` and
    :math:`k_1 F_{ij} / u_*^2` (the convention of ``generate_kaimal_spectra``), and calibrated with
    ``auto_scale=True``,

    .. code-block:: python

        length_scale = zref / (2 * np.pi * pb.freq_scale)
        amplitude = ustar**2 * pb.spectral_scale

    .. note::
        ``FluctuationFieldGenerator`` currently normalizes each generated block to unit RMS before applying the
        mean profile, so ``amplitude`` does not affect the final field there; it is applied for correctness of
        the returned spectral tensor itself.
    """

    def __init__(
        self,
        spectral_tensor_model: SpectralTensorModel,
        length_scale: float,
        amplitude: float = 1.0,
        ndim: int = 3,
    ):
        r"""Initialize the covariance kernel from a trained spectral tensor model.

        Parameters
        ----------
        spectral_tensor_model : SpectralTensorModel
            Trained spectral tensor model (e.g. from ``CalibrationProblem`` or ``load_model``). Its eddy
            lifetime model, energy spectrum model, and learned scales ``L``, ``gamma``, ``sigma`` fully
            determine the non-dimensional spectral tensor.
        length_scale : float
            Length :math:`\ell` mapping physical wavevectors to the model's non-dimensional frequency
            domain, ``k_model = length_scale * k_phys``. See the class docstring for how to compute this
            from the data non-dimensionalization and calibration scaling.
        amplitude : float, optional
            Velocity-squared scale :math:`A` mapping the model's frequency-weighted spectra to physical
            units, by default 1.0.
        ndim : int, optional
            Number of dimensions for the kernel to operate over, by default 3.

        Raises
        ------
        ValueError
            ndim must be 3 for NN covariance; length_scale and amplitude must be positive.
        """
        super().__init__(ndim=ndim)

        if length_scale <= 0.0:
            raise ValueError("length_scale must be positive.")
        if amplitude <= 0.0:
            raise ValueError("amplitude must be positive.")

        self.spectral_tensor_model = spectral_tensor_model
        self.length_scale = length_scale
        self.amplitude = amplitude

        # Dimensional parameters read by FluctuationFieldGenerator for buffer/margin sizing.
        with torch.no_grad():
            self.L = float(torch.exp(spectral_tensor_model.log_L)) * length_scale
            self.Gamma = float(torch.exp(spectral_tensor_model.log_gamma))
        self.E0 = amplitude

    def precompute_spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """Pre-compute the square-root of the associated spectral tensor in the complex domain.

        Parameters
        ----------
        frequencies : np.ndarray
            Physical frequency domain in 3D over which to compute the square-root of the spectral tensor.

        Returns
        -------
        np.ndarray
            Square-root of the spectral tensor evaluated in the frequency domain; note that these are
            complex values.
        """
        model = self.spectral_tensor_model
        param = next(model.parameters())
        dtype, device = param.dtype, param.device

        # Map the physical wavevector grid onto the model's non-dimensional domain.
        k = np.array(list(np.meshgrid(*frequencies, indexing="ij"))) * self.length_scale
        kk = np.sum(k**2, axis=0)

        with torch.no_grad():
            L = torch.exp(model.log_L)
            gamma = torch.exp(model.log_gamma)
            sigma = torch.exp(model.log_sigma)

            k_torch = torch.as_tensor(np.moveaxis(k, 0, -1), dtype=dtype, device=device)
            beta = model.eddy_lifetime_model(k_torch, L, gamma).cpu().numpy().astype(np.float64)

        # The eddy lifetime diverges as k -> 0; the amplitude there is zero anyway.
        beta[np.where(kk == 0)] = 0.0

        def const_fn(k30: np.ndarray, kk0: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                k0_torch = torch.as_tensor(np.stack([k[0, ...], k[1, ...], k30], axis=-1), dtype=dtype, device=device)
                energy = sigma * L ** (5.0 / 3.0) * model.energy_spectrum_model(k0_torch, L)
            energy_np = energy.cpu().numpy().astype(np.float64)

            with np.errstate(divide="ignore", invalid="ignore"):
                const = np.sqrt(energy_np / (4 * np.pi * kk0**2))
            return np.nan_to_num(const, nan=0.0, posinf=0.0)

        sqrt_spectral_tensor = _assemble_rdt_sqrt_tensor(k[0, ...], k[1, ...], k[2, ...], beta, const_fn)

        # Physical rescaling: Phi_phys(k) = amplitude * length_scale^3 * Phi_model(length_scale * k).
        return sqrt_spectral_tensor * np.sqrt(self.amplitude * self.length_scale**3)
