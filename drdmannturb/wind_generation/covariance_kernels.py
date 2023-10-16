"""Includes covariance kernels used in generating wind."""

import numpy as np
from scipy.special import hyp2f1


class Covariance:
    """
    Generic Covariance implementation class
    """
    def __init__(self, ndim=2, verbose=0, **kwargs):
        self.verbose = verbose

        self.ndim = ndim  # dimension 2D or 3D

        if "func" in kwargs:
            self.eval_func = kwargs["func"]

    def eval(self, *argv, **kwargs):
        self.eval_func(*argv, **kwargs)


class VonKarmanCovariance(Covariance):
    """
    Von Karman covariance class subclassing Covariance
    """
    def __init__(self, ndim, length_scale, E0, **kwargs):
        super().__init__(**kwargs)

        ### Spatial dimensions
        if ndim != 3:
            raise ValueError("ndim must be 3 for Von Karman covariance.")
        self.ndim = 3

        self.L = length_scale
        self.E0 = E0

    def precompute_Spectrum(self, Frequencies):
        Nd = [Frequencies[j].size for j in range(self.ndim)]
        SqrtSpectralTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))

        k = np.array(list(np.meshgrid(*Frequencies, indexing="ij")))
        kk = np.sum(k**2, axis=0)

        const = self.E0 * (self.L ** (17 / 3)) / (4 * np.pi)
        const = np.sqrt(const / (1 + (self.L**2) * kk) ** (17 / 6))

        # beta = 0.1
        # const = np.exp(-beta*k) * const

        SqrtSpectralTens[0, 1, ...] = -const * k[2, ...]
        SqrtSpectralTens[0, 2, ...] = const * k[1, ...]
        SqrtSpectralTens[1, 0, ...] = const * k[2, ...]
        SqrtSpectralTens[1, 2, ...] = -const * k[0, ...]
        SqrtSpectralTens[2, 0, ...] = -const * k[1, ...]
        SqrtSpectralTens[2, 1, ...] = const * k[0, ...]

        return SqrtSpectralTens * 1j


class MannCovariance(Covariance):
    """
    Mann covariance implementation
    """   

    def __init__(self, ndim: int = 3, length_scale: float = 1.0, E0: float = 1.0, Gamma: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        ### Spatial dimensions
        if ndim != 3:
            print("ndim MUST BE 3")
            raise
        self.ndim = 3

        self.L = length_scale
        self.E0 = E0
        self.Gamma = Gamma

    def precompute_Spectrum(self, Frequences):
        """
        Compute the power spectrum
        """
        Nd = [Frequences[j].size for j in range(self.ndim)]
        SqrtSpectralTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))
        tmpTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))

        k = np.array(list(np.meshgrid(*Frequences, indexing="ij")))
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
