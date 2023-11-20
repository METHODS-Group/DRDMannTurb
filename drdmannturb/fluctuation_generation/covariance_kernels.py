"""Includes covariance kernels used in generating wind, specifically in evaluating the square roots of spectral tensors used in various models."""

import numpy as np
from scipy.special import hyp2f1


class Covariance:
    r"""
    Generic Covariance kernel metaclass. In particular, every subclass involves the computation of :math:`G(\boldsymbol{k})` where

    .. math::
        G(\boldsymbol{k}) G^*(\boldsymbol{k})=\Phi(\boldsymbol{k}, \tau(\boldsymbol{k}))

    for different choices of the spectral tensor :math:`\Phi(\boldsymbol{k})`.

    Subclasses only require one field ``ndim``, which specifies the number of dimensions in which the generated kernels operate, respectively. For now, all fields and kernels operate in 3D, though functionality for 2D may be added readily. Finally, if a generic evaluation function is desired for a subclass, it may be set with ``eval_func`` in the constructor, as well as any associated arguments.
    """

    def __init__(self, ndim=2, **kwargs):
        self.ndim = ndim  # dimension 2D or 3D

        if "func" in kwargs:
            self.eval_func = kwargs["func"]

    def eval(self, *argv, **kwargs):
        self.eval_func(*argv, **kwargs)


class VonKarmanCovariance(Covariance):
    r"""
    Von Karman covariance kernel. Evaluates the :math:`G(\boldsymbol{k})` which satisfies :math:`G(\boldsymbol{k}) G^*(\boldsymbol{k})=\Phi(\boldsymbol{k})` where

    .. math::
            \Phi_{i j}^{\mathrm{VK}}(\boldsymbol{k})=\frac{E(k)}{4 \pi k^2}\left(\delta_{i j}-\frac{k_i k_j}{k^2}\right)

    which utilizes the energy spectrum function

    .. math::
        E(k)=c_0^2 \varepsilon^{2 / 3} k^{-5 / 3}\left(\frac{k L}{\left(1+(k L)^2\right)^{1 / 2}}\right)^{17 / 3},

    where :math:`\varepsilon` is the viscous dissipation of the turbulent kinetic energy, :math:`L` is the length scale parameter and :math:`c_0^2 \approx 1.7` is an empirical constant.
    """

    def __init__(self, ndim: int, length_scale: float, E0: float, **kwargs):
        """
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
        super().__init__(**kwargs)

        if ndim != 3:
            raise ValueError("ndim must be 3 for Von Karman covariance.")

        self.ndim = 3

        self.L = length_scale
        self.E0 = E0

    def precompute_Spectrum(self, Frequencies: np.ndarray) -> np.ndarray:
        """Evaluation method which pre-computes the square-root of the associated spectral tensor in the frequency domain.

        Parameters
        ----------
        Frequencies : np.ndarray
            Frequency domain in 3D over which to compute the square-root of the spectral tensor.

        Returns
        -------
        np.ndarray
            Square-root of the spectral tensor evaluated in the frequency domain; note that these are complex values.
        """
        Nd = [Frequencies[j].size for j in range(self.ndim)]
        SqrtSpectralTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))

        k = np.array(list(np.meshgrid(*Frequencies, indexing="ij")))
        kk = np.sum(k**2, axis=0)

        const = self.E0 * (self.L ** (17 / 3)) / (4 * np.pi)
        const = np.sqrt(const / (1 + (self.L**2) * kk) ** (17 / 6))

        SqrtSpectralTens[0, 1, ...] = -const * k[2, ...]
        SqrtSpectralTens[0, 2, ...] = const * k[1, ...]
        SqrtSpectralTens[1, 0, ...] = const * k[2, ...]
        SqrtSpectralTens[1, 2, ...] = -const * k[0, ...]
        SqrtSpectralTens[2, 0, ...] = -const * k[1, ...]
        SqrtSpectralTens[2, 1, ...] = const * k[0, ...]

        return SqrtSpectralTens * 1j


class MannCovariance(Covariance):
    r"""
    Mann covariance kernel implementation. Evaluates the :math:`G(\boldsymbol{k})` which satisfies :math:`G(\boldsymbol{k}) G^*(\boldsymbol{k})=\Phi(\boldsymbol{k}, \tau(\boldsymbol{k}))` where

    .. math::
            \tau^{\mathrm{IEC}}(k)=\frac{T B^{-1}(k L)^{-\frac{2}{3}}}{\sqrt{{ }_2 F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-(k L)^{-2}\right)}}

    and the full spectral tensor can be found in the following reference:
        J. Mann, “The spatial structure of neutral atmospheric surface layer turbulence,” Journal of fluid mechanics 273, 141-168 (1994)
    """

    def __init__(
        self,
        ndim: int = 3,
        length_scale: float = 1.0,
        E0: float = 1.0,
        Gamma: float = 1.0,
        **kwargs
    ):
        """
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
        super().__init__(**kwargs)

        ### Spatial dimensions
        if ndim != 3:
            raise ValueError("ndim must be 3 for Mann covariance.")
        self.ndim = 3

        self.L = length_scale
        self.E0 = E0
        self.Gamma = Gamma

    def precompute_Spectrum(self, Frequencies: np.ndarray) -> np.ndarray:
        """Evaluation method which pre-computes the square-root of the associated spectral tensor in the frequency domain.

        Parameters
        ----------
        Frequencies : np.ndarray
            Frequency domain in 3D over which to compute the square-root of the spectral tensor.

        Returns
        -------
        np.ndarray
            Square-root of the spectral tensor evaluated in the frequency domain; note that these are complex values.
        """
        Nd = [Frequencies[j].size for j in range(self.ndim)]
        SqrtSpectralTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))
        tmpTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))

        k = np.array(list(np.meshgrid(*Frequencies, indexing="ij")))
        kk = np.sum(k**2, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            beta = (
                self.Gamma
                * (kk * self.L**2) ** (-1 / 3)
                / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -1 / (kk * self.L**2)))
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

            tmpTens[0, 1, ...] = -const * k30
            tmpTens[0, 2, ...] = const * k2
            tmpTens[1, 0, ...] = const * k30
            tmpTens[1, 2, ...] = -const * k1
            tmpTens[2, 0, ...] = -const * k2
            tmpTens[2, 1, ...] = const * k1

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

            SqrtSpectralTens[0, 0, ...] = (
                tmpTens[0, 0, ...] + zeta1 * tmpTens[2, 0, ...]
            )
            SqrtSpectralTens[0, 1, ...] = (
                tmpTens[0, 1, ...] + zeta1 * tmpTens[2, 1, ...]
            )
            SqrtSpectralTens[0, 2, ...] = (
                tmpTens[0, 2, ...] + zeta1 * tmpTens[2, 2, ...]
            )
            SqrtSpectralTens[1, 0, ...] = (
                tmpTens[1, 0, ...] + zeta2 * tmpTens[2, 0, ...]
            )
            SqrtSpectralTens[1, 1, ...] = (
                tmpTens[1, 1, ...] + zeta2 * tmpTens[2, 1, ...]
            )
            SqrtSpectralTens[1, 2, ...] = (
                tmpTens[1, 2, ...] + zeta2 * tmpTens[2, 2, ...]
            )
            SqrtSpectralTens[2, 0, ...] = zeta3 * tmpTens[2, 0, ...]
            SqrtSpectralTens[2, 1, ...] = zeta3 * tmpTens[2, 1, ...]
            SqrtSpectralTens[2, 2, ...] = zeta3 * tmpTens[2, 2, ...]

            return SqrtSpectralTens * 1j
