"""Includes covariance kernels used in generating wind.

Specifically, in evaluating the square roots of spectral tensors used in various models.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import hyp2f1


class Covariance(ABC):
    r"""
    Generic covariance kernel metaclass.

    Every subclass implements the computation of a matrix :math:`G(\boldsymbol{k})` such that

     .. math::
         G(\boldsymbol{k}) G^*(\boldsymbol{k})=\Phi(\boldsymbol{k}, \tau(\boldsymbol{k}))

    for different choices of the spectral tensor :math:`\Phi(\boldsymbol{k})`

    Subclasses must implement:
    - `precompute_spectrum()`: Compute the square-root of the spectral tensor.
    - `__init__()`: Initialize with required parameters.
    """

    ndim: int

    def __init__(self, ndim: int = 3, **kwargs):
        """Initialize the covariance kernel.

        Parameters
        ----------
        ndim : int
            Number of dimensions for the kernel to operate over.
        **kwargs
            Additional parameters specific to each covariance type.
        """
        self.ndim = ndim
        self._validate_ndim()

    def _validate_ndim(self):
        """Validate that ndim is appropriate for this covariance type."""
        if self.ndim != 3:
            raise ValueError(f"ndim must be 3 for {self.__class__.__name__}.")

    @abstractmethod
    def precompute_spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Pre-compute the square-root of the associated spectral tensor in the frequency domain.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequency domain in 3D over which to compute the square-root of the spectral tensor.

        Returns
        -------
        np.ndarray
            Square-root of the spectral tensor evaluated in the frequency domain;
            note that these are complex values.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class VonKarmanCovariance(Covariance):
    r"""
    Von Karman covariance kernel.

    Evaluates the :math:`G(\boldsymbol{k})` which satisfies
    :math:`G(\boldsymbol{k}) G^*(\boldsymbol{k})=\Phi(\boldsymbol{k})` where

    .. math::
            \Phi_{i j}^{\mathrm{VK}}(\boldsymbol{k})=\frac{E(k)}{4 \pi k^2}
            \left(\delta_{i j}-\frac{k_i k_j}{k^2}\right)

    which utilizes the energy spectrum function

    .. math::
        E(k)=c_0^2 \varepsilon^{2 / 3} k^{-5 / 3}
        \left(\frac{k L}{\left(1+(k L)^2\right)^{1 / 2}}\right)^{17 / 3},

    where :math:`\varepsilon` is the viscous dissipation of the turbulent kinetic energy,
    :math:`L` is the length scale parameter and :math:`c_0^2 \approx 1.7` is an empirical constant.
    """

    def __init__(self, ndim: int = 3, length_scale: float = 1.0, E0: float = 1.0, **kwargs):
        """
        Initialize the Von Karman covariance kernel.

        Parameters
        ----------
        ndim : int
            Number of dimensions over which to apply the covariance kernel.
        length_scale : float
            Length scale non-dimensionalizing constant.
        E0 : float
            Energy spectrum.

        Raises
        ------
        ValueError
            ndim must be 3 for Von Karman covariance.
        """
        super().__init__(ndim=ndim, **kwargs)

        self.L = length_scale
        self.E0 = E0
        # NOTE: VK does not use Gamma, but this is set for compatibility.
        self.Gamma = 1.0

    def precompute_spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Pre-compute the square-root of the associated spectral tensor in the frequency domain.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequency domain in 3D over which to compute the square-root of the spectral tensor.

        Returns
        -------
        np.ndarray
            Square-root of the spectral tensor evaluated in the frequency domain; note
            that these are complex values.
        """
        ndim = self.ndim
        Nd = [frequencies[j].size for j in range(ndim)]
        sqrt_spectral_tensor = np.tile(np.zeros(Nd), (ndim, ndim, 1, 1, 1))

        k = np.array(list(np.meshgrid(*frequencies, indexing="ij")))
        kk = np.sum(k**2, axis=0)

        const = self.E0 * (self.L ** (17 / 3)) / (4 * np.pi)
        const = np.sqrt(const / (1 + (self.L**2) * kk) ** (17 / 6))

        sqrt_spectral_tensor[0, 1, ...] = -const * k[2, ...]
        sqrt_spectral_tensor[0, 2, ...] = const * k[1, ...]
        sqrt_spectral_tensor[1, 0, ...] = const * k[2, ...]
        sqrt_spectral_tensor[1, 2, ...] = -const * k[0, ...]
        sqrt_spectral_tensor[2, 0, ...] = -const * k[1, ...]
        sqrt_spectral_tensor[2, 1, ...] = const * k[0, ...]

        return sqrt_spectral_tensor * 1j


class MannCovariance(Covariance):
    r"""
    Mann covariance kernel implementation.

    Evaluates the :math:`G(\boldsymbol{k})` which satisfies
    :math:`G(\boldsymbol{k}) G^*(\boldsymbol{k})=\Phi(\boldsymbol{k}, \tau(\boldsymbol{k}))` where

    .. math::
            \tau^{\mathrm{IEC}}(k)=\frac{T B^{-1}(k L)^{-\frac{2}{3}}}
            {\sqrt{{ }_2 F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-(k L)^{-2}\right)}}

    and the full spectral tensor can be found in the following reference:
        J. Mann, "The spatial structure of neutral atmospheric surface layer turbulence,"
        Journal of fluid mechanics 273, 141-168 (1994)
    """

    def __init__(self, ndim: int = 3, length_scale: float = 1.0, E0: float = 1.0, Gamma: float = 1.0, **kwargs):
        """
        Initialize the Mann covariance kernel.

        Parameters
        ----------
        ndim : int, optional
            Number of dimensions for kernel to operate over, by default 3
        length_scale : float, optional
            Length scale non-dimensionalizing constant, by default 1.0
        E0 : float, optional
            Energy spectrum, by default 1.0
        Gamma : float, optional
            Time scale, by default 1.0

        Raises
        ------
        ValueError
            ndim must be 3 for Mann covariance.
        """
        super().__init__(ndim=ndim, **kwargs)

        self.L = length_scale
        self.E0 = E0
        self.Gamma = Gamma

    def precompute_spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Pre-compute the square-root of the associated spectral tensor in the frequency domain.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequency domain in 3D over which to compute the square-root of the spectral tensor.

        Returns
        -------
        np.ndarray
            Square-root of the spectral tensor evaluated in the frequency domain;
            note that these are complex values.
        """
        ndim = self.ndim
        Nd = [frequencies[j].size for j in range(ndim)]
        sqrt_spectral_tensor = np.tile(np.zeros(Nd), (ndim, ndim, 1, 1, 1))
        tmp_tensor = np.tile(np.zeros(Nd), (ndim, ndim, 1, 1, 1))

        k = np.array(list(np.meshgrid(*frequencies, indexing="ij")))
        kk = np.sum(k**2, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            beta = (
                self.Gamma * (kk * self.L**2) ** (-1 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -1 / (kk * self.L**2)))
            )
            beta[np.where(kk == 0)] = 0

            k1 = k[0, ...]
            k2 = k[1, ...]
            k3 = k[2, ...]
            k30 = k3 + beta * k1

            kk0 = k1**2 + k2**2 + k30**2

            #### Isotropic with k0

            const = self.E0 * (self.L ** (17 / 3)) / (4 * np.pi)
            const = np.sqrt(const / (1 + (self.L**2) * kk0) ** (17 / 6))

            # to enforce zero mean in the x-direction:
            # const[k1 == 0] = 0.0

            tmp_tensor[0, 1, ...] = -const * k30
            tmp_tensor[0, 2, ...] = const * k2
            tmp_tensor[1, 0, ...] = const * k30
            tmp_tensor[1, 2, ...] = -const * k1
            tmp_tensor[2, 0, ...] = -const * k2
            tmp_tensor[2, 1, ...] = const * k1

            #### RDT

            s = k1**2 + k2**2
            C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
            numerator = beta * k1 * np.sqrt(s)
            denominator = kk0 - k30 * k1 * beta
            C2 = k2 * kk0 / s ** (3 / 2) * np.arctan2(numerator, denominator)

            zeta1 = C1 - k2 / k1 * C2
            zeta2 = k2 / k1 * C1 + C2
            zeta3 = kk0 / kk

            # deal with divisions by zero
            zeta1 = np.nan_to_num(zeta1)
            zeta2 = np.nan_to_num(zeta2)
            zeta3 = np.nan_to_num(zeta3)

            sqrt_spectral_tensor[0, 0, ...] = tmp_tensor[0, 0, ...] + zeta1 * tmp_tensor[2, 0, ...]
            sqrt_spectral_tensor[0, 1, ...] = tmp_tensor[0, 1, ...] + zeta1 * tmp_tensor[2, 1, ...]
            sqrt_spectral_tensor[0, 2, ...] = tmp_tensor[0, 2, ...] + zeta1 * tmp_tensor[2, 2, ...]
            sqrt_spectral_tensor[1, 0, ...] = tmp_tensor[1, 0, ...] + zeta2 * tmp_tensor[2, 0, ...]
            sqrt_spectral_tensor[1, 1, ...] = tmp_tensor[1, 1, ...] + zeta2 * tmp_tensor[2, 1, ...]
            sqrt_spectral_tensor[1, 2, ...] = tmp_tensor[1, 2, ...] + zeta2 * tmp_tensor[2, 2, ...]
            sqrt_spectral_tensor[2, 0, ...] = zeta3 * tmp_tensor[2, 0, ...]
            sqrt_spectral_tensor[2, 1, ...] = zeta3 * tmp_tensor[2, 1, ...]
            sqrt_spectral_tensor[2, 2, ...] = zeta3 * tmp_tensor[2, 2, ...]

            return sqrt_spectral_tensor * 1j
