from math import *
import numpy as np
from scipy.special import kv as Kv
from scipy.special import hyp2f1
from itertools import product
from time import time
from tqdm import tqdm

import scipy.fftpack as fft
import matplotlib.pyplot as plt
from scipy.special import hyp2f1

from .utilities import Matern_kernel, GM_kernel, EP_kernel
from .CovarianceKernels import Covariance, set_ShapeOperator


#######################################################################################################
# 	Neural Net Covariance class
#######################################################################################################


class NNCovariance(Covariance):
    def __init__(self, ndim, **kwargs):
        super().__init__(**kwargs)

        ### Spatial dimensions
        if ndim is not 3:
            print("ndim MUST BE 3")
            raise
        self.ndim = 3

        ### Correlation length
        self.corrlen = kwargs.get("length_scale", None)
        ### Spectrum maginitude
        self.E0 = kwargs.get("E0", None)
        ### Viscous_dissipation_rate
        # self.epsilon = kwargs['viscous_dissipation_rate']
        ### Kolmogorov constant
        # self.alpha = kwargs['kolmogorov_constant']
        ### Time scale
        self.Gamma = kwargs.get("Gamma", None)

        self.OPS = kwargs.get("OnePointSpetra", None)

    # --------------------------------------------------------------------------
    #   Compute the power spectrum
    # --------------------------------------------------------------------------

    def precompute_Spectrum(self, Frequences):
        Nd = [Frequences[j].size for j in range(self.ndim)]
        SqrtSpectralTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))
        tmpTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))

        k = np.array(list(np.meshgrid(*Frequences, indexing="ij")))
        kk = np.sum(k**2, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            beta = self.Gamma * self.OPS.tauNet(k)
            beta[np.where(kk == 0)] = 0

            k1 = k[0, ...]
            k2 = k[1, ...]
            k3 = k[2, ...]
            # k30  = k3 + k1
            k30 = k3 + beta * k1

            kk0 = k1**2 + k2**2 + k30**2

            # deal with divisions by zero (1/2)
            # np.seterr(divide='ignore', invalid='ignore')

            #### Isotropic with k0

            const = self.E0 * (self.corrlen ** (17 / 3)) / (4 * np.pi)
            const = np.sqrt(const / (1 + (self.corrlen**2) * kk0) ** (17 / 6))

            ##NOTE: Applying the curl second

            #### RDT

            s = k1**2 + k2**2
            C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
            tmp = beta * k1 * np.sqrt(s) / (kk0 - k30 * k1 * beta)
            C2 = k2 * kk0 / s ** (3 / 2) * np.arctan(tmp)

            zeta1 = C1 - k2 / k1 * C2
            zeta2 = k2 / k1 * C1 + C2
            zeta3 = kk0 / kk

            # deal with divisions by zero (2/2)
            zeta1 = np.nan_to_num(zeta1)
            zeta2 = np.nan_to_num(zeta2)
            zeta3 = np.nan_to_num(zeta3)

            tmpTens[0, 0, ...] = const * zeta3
            tmpTens[1, 1, ...] = const * zeta3
            tmpTens[2, 0, ...] = -const * zeta1
            tmpTens[2, 1, ...] = -const * zeta2
            tmpTens[2, 2, ...] = const

            SqrtSpectralTens[0, 0, ...] = (
                -k3 * tmpTens[1, 0, ...] + k2 * tmpTens[2, 0, ...]
            )
            SqrtSpectralTens[0, 1, ...] = (
                -k3 * tmpTens[1, 1, ...] + k2 * tmpTens[2, 1, ...]
            )
            SqrtSpectralTens[0, 2, ...] = (
                -k3 * tmpTens[1, 2, ...] + k2 * tmpTens[2, 2, ...]
            )
            SqrtSpectralTens[1, 0, ...] = (
                k3 * tmpTens[0, 0, ...] - k1 * tmpTens[2, 0, ...]
            )
            SqrtSpectralTens[1, 1, ...] = (
                k3 * tmpTens[0, 1, ...] - k1 * tmpTens[2, 1, ...]
            )
            SqrtSpectralTens[1, 2, ...] = (
                k3 * tmpTens[0, 2, ...] - k1 * tmpTens[2, 2, ...]
            )
            SqrtSpectralTens[2, 0, ...] = (
                -k2 * tmpTens[0, 0, ...] + k1 * tmpTens[1, 0, ...]
            )
            SqrtSpectralTens[2, 1, ...] = (
                -k2 * tmpTens[0, 1, ...] + k1 * tmpTens[1, 1, ...]
            )
            SqrtSpectralTens[2, 2, ...] = (
                -k2 * tmpTens[0, 2, ...] + k1 * tmpTens[1, 2, ...]
            )

            return SqrtSpectralTens * 1j

    # --------------------------------------------------------------------------
    #   Compute the power spectrum
    # --------------------------------------------------------------------------

    # def factor(self, SpectralTensor):

    #     w, v = np.linalg.eig(SpectralTensor)

    #     w[w.real < 1e-15] = 0.0

    #     w = np.sqrt(np.maximum(w,0.0))
    #     wdiag = np.diag(w)

    #     # if any(isinstance(v_, complex) for v_ in list(v.flatten())):
    #     #     print('v = ', v)

    #     return np.dot(wdiag,np.transpose(v)).real

    # --------------------------------------------------------------------------
    #   Evaluate covariance function
    # --------------------------------------------------------------------------

    def eval(self, *args):
        print("eval function is not supported")
        raise

    def eval_sqrt(self, *args, nu=None, corrlen=None):
        print("eval_sqrt function is not supported")
        raise
