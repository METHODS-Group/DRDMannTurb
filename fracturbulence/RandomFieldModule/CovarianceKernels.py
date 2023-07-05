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


# --------------------------------------------------------------------------
#   Set the shape operator
# --------------------------------------------------------------------------


def set_ShapeOperator(length, angle=None, ndim=2):
    if np.isscalar(length):
        l = [length] * ndim
    else:
        l = length[:ndim]
    l = np.array(l)

    if angle is None:
        angle = 0
        l[:] = l.mean()

    if np.isscalar(angle):
        Theta = np.zeros([ndim, ndim])
    else:
        Theta = np.zeros(list(angle.shape) + [ndim, ndim])

    if ndim == 2:
        c, s = np.cos(angle), np.sin(angle)
        L0, L1 = l[0] ** 2, l[1] ** 2
        detTheta = L0 * L1

        Theta[..., 0, 0] = L0 * (s * s) + L1 * (c * c)
        Theta[..., 1, 1] = L0 * (c * c) + L1 * (s * s)
        Theta[..., 0, 1] = L0 * (c * s) - L1 * (s * c)
        Theta[..., 1, 0] = Theta[..., 0, 1]

    else:
        detTheta = 1
        for j in range(ndim):
            Theta[..., j, j] = l[j] ** 2
            detTheta *= Theta[j, j]

    yield Theta
    yield detTheta


#######################################################################################################
# 	Generic Covariance class
#######################################################################################################


class Covariance:
    def __init__(self, ndim=2, verbose=0, **kwargs):
        self.verbose = verbose

        self.ndim = ndim  # dimension 2D or 3D

        if "func" in kwargs:
            self.eval_func = kwargs["func"]

    # --------------------------------------------------------------------------
    #   Evaluate covariance function
    # --------------------------------------------------------------------------

    def eval(self, *argv, **kwargs):
        self.eval_func(*argv, **kwargs)


#######################################################################################################
# 	Matérn Covariance class
#######################################################################################################


class MaternCovariance(Covariance):
    def __init__(self, nu, corrlen, angle_anis=None, marg_var=1, **kwargs):
        super().__init__(**kwargs)

        self.nu = nu  # Regularity of the Matérn covariance kernel
        self.marg_var = marg_var  # Marginal (pointwise) variance of the field

        ### Correlation length
        self.corrlen = np.zeros(self.ndim)
        if np.isscalar(corrlen):
            self.corrlen[:] = corrlen
        else:
            self.corrlen[:] = corrlen[: self.ndim]

        if all([self.corrlen[0] == self.corrlen[i] for i in range(self.ndim)]):
            self.isotropic = True
        else:
            self.isotropic = False

        ### Anisotropy
        self.angle_anis = angle_anis

        ### Operator Theta
        self.Theta, self.detTheta = set_ShapeOperator(
            self.corrlen, self.angle_anis, self.ndim
        )

        ### Normalizing coefficient
        self.eta = self.set_NormalizingCoefficient()

    # --------------------------------------------------------------------------
    #   Set Normalizing coefficient
    # --------------------------------------------------------------------------

    def set_NormalizingCoefficient(self):
        nu, d = self.nu, self.ndim
        alpha = nu + d / 2
        try:
            default_variance = (
                gamma(nu)
                / gamma(alpha)
                * (2 * nu / (4 * pi)) ** (d / 2)
                / sqrt(self.detTheta)
            )
        except:
            default_variance = (2 * pi) ** (-d / 2) / sqrt(self.detTheta)
        return sqrt(self.marg_var / default_variance)

    # --------------------------------------------------------------------------
    #   Compute the power spectrum
    # --------------------------------------------------------------------------

    def precompute_Spectrum(self, Frequences):
        nu, d = self.nu, self.ndim

        alpha = nu + d / 2
        eta = self.eta

        Nd = [Frequences[j].size for j in range(d)]
        Spectrum = np.zeros(Nd)

        w = np.array(list(np.meshgrid(*Frequences, indexing="ij")))
        Tw = np.einsum("kl,l...->k...", self.Theta, w)
        wTw = np.einsum("k...,k...->...", w, Tw)
        if nu < 1000:
            Spectrum = eta * (1 + wTw / (2 * nu)) ** (-0.5 * alpha)
        else:  # Squared-Exponential
            Spectrum = eta * np.exp(-1 * wTw / 4)

        return Spectrum

    # --------------------------------------------------------------------------
    #   Evaluate covariance function
    # --------------------------------------------------------------------------

    def eval(self, *args, nu=None, corrlen=None):
        if nu is None:
            nu = self.nu
        if corrlen is None:
            corrlen = self.corrlen[0]
        if len(args) == 1:
            r = args[0]
        elif len(args) == 2:
            x = args[0]
            y = args[1]
            r = np.linalg.norm(x - y)
        return Matern_kernel(r, nu, corrlen)
        # if self.isotropic:
        #     # r = np.linalg.norm(x-y, axis=0)
        #     r = x-y
        #     r = np.sqrt(np.sum(r.T.dot(r), axis=0))
        #     return Matern_kernel(r, nu, corrlen[0])
        # else:
        #     r = x-y
        #     r = np.sqrt(np.sum(self.Theta.dot(r).T.dot(r), axis=0))
        #     return Matern_kernel(r, nu, 1)

    def eval_sqrt(self, *args, nu=None, corrlen=None):
        if nu is None:
            nu = self.nu
        if corrlen is None:
            corrlen = self.corrlen[0]
        if len(args) == 1:
            r = args[0]
        elif len(args) == 2:
            x = args[0]
            y = args[1]
            r = np.linalg.norm(x - y)
        d = self.ndim
        nu_mod = (nu - d / 2) / 2
        sigma = sqrt(
            gamma(nu + d / 2) / gamma(nu) * (nu / (2 * pi) / self.detTheta) ** (d / 2)
        )
        sigma *= gamma(nu_mod) / gamma(nu_mod + d / 2)
        return sigma * Matern_kernel(r, nu_mod, corrlen * sqrt(nu_mod / nu))


###################################################################


#######################################################################################################
# 	Von Karman Covariance class
#######################################################################################################


class VonKarmanCovariance(Covariance):
    def __init__(
        self,
        correlation_length,
        viscous_dissipation_rate,
        kolmogorov_constant,
        ndim=3,
        **kwargs
    ):
        super().__init__(**kwargs)

        ### Spatial dimensions
        if ndim is not 3:
            print("ndim MUST BE 3")
            raise
        self.ndim = 3

        self.name = "Von Karman"

        ### Correlation length
        self.corrlen = correlation_length
        ### Viscous_dissipation_rate
        self.epsilon = viscous_dissipation_rate
        ### Kolmogorov constant
        self.alpha = kolmogorov_constant

    # --------------------------------------------------------------------------
    #   Compute the power spectrum
    # --------------------------------------------------------------------------

    def precompute_Spectrum(self, Frequences):
        Nd = [Frequences[j].size for j in range(self.ndim)]
        SqrtSpectralTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))

        k = np.array(list(np.meshgrid(*Frequences, indexing="ij")))
        kk = np.sum(k**2, axis=0)

        const = (
            self.alpha
            * (self.epsilon ** (2 / 3))
            * (self.corrlen ** (17 / 3))
            / (4 * np.pi)
        )
        const = np.sqrt(const / (1 + (self.corrlen**2) * kk) ** (17 / 6))

        # beta = 0.1
        # const = np.exp(-beta*k) * const

        SqrtSpectralTens[0, 0, ...] = const
        SqrtSpectralTens[1, 1, ...] = const
        SqrtSpectralTens[2, 2, ...] = const
        return SqrtSpectralTens

        # SqrtSpectralTens[0,1,...] = -const * k[2,...]
        # SqrtSpectralTens[0,2,...] =  const * k[1,...]
        # SqrtSpectralTens[1,0,...] =  const * k[2,...]
        # SqrtSpectralTens[1,2,...] = -const * k[0,...]
        # SqrtSpectralTens[2,0,...] = -const * k[1,...]
        # SqrtSpectralTens[2,1,...] =  const * k[0,...]

        # return SqrtSpectralTens*1j

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


###################################################################


#######################################################################################################
# 	Mann Covariance class
#######################################################################################################


class MannCovariance(Covariance):
    def __init__(self, ndim, **kwargs):
        super().__init__(**kwargs)

        ### Spatial dimensions
        if ndim is not 3:
            print("ndim MUST BE 3")
            raise
        self.ndim = 3

        ### Correlation length
        self.corrlen = kwargs["length_scale"]
        ### Viscous_dissipation_rate
        self.epsilon = kwargs["viscous_dissipation_rate"]
        ### Kolmogorov constant
        self.alpha = kwargs["kolmogorov_constant"]
        ### Kolmogorov constant
        self.Gamma = kwargs["Gamma"]

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
            beta = (
                self.Gamma
                * (kk * self.corrlen**2) ** (-1 / 3)
                / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -1 / (kk * self.corrlen**2)))
            )
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

            const = (
                self.alpha
                * (self.epsilon ** (2 / 3))
                * (self.corrlen ** (17 / 3))
                / (4 * np.pi)
            )
            const = np.sqrt(const / (1 + (self.corrlen**2) * kk0) ** (17 / 6))

            # tmpTens[0,1,...] = -const * k30
            # tmpTens[0,2,...] =  const * k2
            # tmpTens[1,0,...] =  const * k30
            # tmpTens[1,2,...] = -const * k1
            # tmpTens[2,0,...] = -const * k2
            # tmpTens[2,1,...] =  const * k1

            # #### RDT

            # s = k1**2 + k2**2
            # C1  =  beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
            # tmp =  beta * k1 * np.sqrt(s) / (kk0 - k30 * k1 * beta)
            # C2  =  k2 * kk0 / s**(3/2) * np.arctan (tmp)

            # zeta1 =  C1 - k2/k1 * C2
            # zeta2 =  k2/k1 *C1 + C2
            # zeta3 =  kk0/kk

            # # deal with divisions by zero (2/2)
            # zeta1 = np.nan_to_num(zeta1)
            # zeta2 = np.nan_to_num(zeta2)
            # zeta3 = np.nan_to_num(zeta3)

            # SqrtSpectralTens[0,0,...] = tmpTens[0,0,...] + zeta1 * tmpTens[2,0,...]
            # SqrtSpectralTens[0,1,...] = tmpTens[0,1,...] + zeta1 * tmpTens[2,1,...]
            # SqrtSpectralTens[0,2,...] = tmpTens[0,2,...] + zeta1 * tmpTens[2,2,...]
            # SqrtSpectralTens[1,0,...] = tmpTens[1,0,...] + zeta2 * tmpTens[2,0,...]
            # SqrtSpectralTens[1,1,...] = tmpTens[1,1,...] + zeta2 * tmpTens[2,1,...]
            # SqrtSpectralTens[1,2,...] = tmpTens[1,2,...] + zeta2 * tmpTens[2,2,...]
            # SqrtSpectralTens[2,0,...] = zeta3 * tmpTens[2,0,...]
            # SqrtSpectralTens[2,1,...] = zeta3 * tmpTens[2,1,...]
            # SqrtSpectralTens[2,2,...] = zeta3 * tmpTens[2,2,...]

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

            # SqrtSpectralTens[0,0,...] =  const * zeta3
            # SqrtSpectralTens[1,1,...] =  const * zeta3
            # SqrtSpectralTens[2,0,...] = -const * zeta1
            # SqrtSpectralTens[2,1,...] = -const * zeta2
            # SqrtSpectralTens[2,2,...] =  const

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


###################################################################

#######################################################################################################
# 	Uniform Shear Covariance class
#######################################################################################################


class UniformShearCovariance(Covariance):
    def __init__(self, ndim, **kwargs):
        super().__init__(**kwargs)

        ### Spatial dimensions
        if ndim is not 3:
            print("ndim MUST BE 3")
            raise
        self.ndim = 3

        ### Correlation length
        self.corrlen = kwargs["correlation_length"]
        ### Viscous_dissipation_rate
        self.epsilon = kwargs["viscous_dissipation_rate"]
        ### Kolmogorov constant
        self.alpha = kwargs["kolmogorov_constant"]
        ### Kolmogorov constant
        self.Gamma = kwargs["Gamma"]

    # --------------------------------------------------------------------------
    #   Compute the power spectrum
    # --------------------------------------------------------------------------

    def precompute_Spectrum(self, Frequences):
        Nd = [Frequences[j].size for j in range(self.ndim)]
        SqrtSpectralTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))
        tmpTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))

        k = np.array(list(np.meshgrid(*Frequences, indexing="ij")))
        kk = np.sum(k**2, axis=0)

        const = (
            self.alpha
            * (self.epsilon ** (2 / 3))
            * (self.corrlen**17 / 3)
            / (4 * np.pi)
        )
        const = np.sqrt(const / (1 + (self.corrlen**2) * kk) ** (17 / 6))

        tmpTens[0, 1, ...] = -const * k[2, ...]
        tmpTens[0, 2, ...] = const * k[1, ...]
        tmpTens[1, 0, ...] = const * k[2, ...]
        tmpTens[1, 2, ...] = -const * k[0, ...]
        tmpTens[2, 0, ...] = -const * k[1, ...]
        tmpTens[2, 1, ...] = const * k[0, ...]

        # deal with divisions by zero (1/2)
        np.seterr(divide="ignore", invalid="ignore")

        # NOTE: Hard-coded for now
        beta = 1.0

        k1 = k[0, ...]
        k2 = k[1, ...]
        k30 = k[2, ...]
        k3 = k30 - beta * k1

        kkt = k1**2 + k2**2 + k3**2

        C1 = (
            beta
            * k1**2
            * (kk - 2 * k30**2 + beta * k1 * k30)
            / (kkt * (k1**2 + k2**2))
        )
        tmp = beta * k1 * np.sqrt(k1**2 + k2**2) / (kk - k30 * k1 * beta)
        C2 = k2 * kk / (k1**2 + k2**2) ** (3 / 2) * np.arctan(tmp)

        zeta1 = C1 - k2 / k1 * C2
        zeta2 = k2 / k1 * C1 + C2
        zeta3 = kk / kkt

        # deal with divisions by zero (2/2)
        zeta1 = np.nan_to_num(zeta1)
        zeta2 = np.nan_to_num(zeta2)
        zeta3 = np.nan_to_num(zeta3)

        SqrtSpectralTens[0, 0, ...] = tmpTens[0, 0, ...] + zeta1 * tmpTens[2, 0, ...]
        SqrtSpectralTens[0, 1, ...] = tmpTens[0, 1, ...] + zeta1 * tmpTens[2, 1, ...]
        SqrtSpectralTens[0, 2, ...] = tmpTens[0, 2, ...] + zeta1 * tmpTens[2, 2, ...]
        SqrtSpectralTens[1, 0, ...] = tmpTens[1, 0, ...] + zeta2 * tmpTens[2, 0, ...]
        SqrtSpectralTens[1, 1, ...] = tmpTens[1, 1, ...] + zeta2 * tmpTens[2, 1, ...]
        SqrtSpectralTens[1, 2, ...] = tmpTens[1, 2, ...] + zeta2 * tmpTens[2, 2, ...]
        SqrtSpectralTens[2, 0, ...] = zeta3 * tmpTens[2, 0, ...]
        SqrtSpectralTens[2, 1, ...] = zeta3 * tmpTens[2, 1, ...]
        SqrtSpectralTens[2, 2, ...] = zeta3 * tmpTens[2, 2, ...]

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


#######################################################################################################
# Covariance class for computing the vector potential
#######################################################################################################


class VectorPotentialCovariance(Covariance):
    def __init__(self, ndim, **kwargs):
        super().__init__(**kwargs)

        ### Spatial dimensions
        if ndim is not 3:
            print("ndim MUST BE 3")
            raise
        self.ndim = 3

        ### Correlation length
        self.corrlen = kwargs["correlation_length"]
        ### Viscous_dissipation_rate
        self.epsilon = kwargs["viscous_dissipation_rate"]
        ### Kolmogorov constant
        self.alpha = kwargs["kolmogorov_constant"]
        ### Time scale
        self.Gamma = kwargs["Gamma"]

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
            beta = (
                self.Gamma
                * (kk * self.corrlen**2) ** (-1 / 3)
                / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -1 / (kk * self.corrlen**2)))
            )
            beta[np.where(kk == 0)] = 0

            k1 = k[0, ...]
            k2 = k[1, ...]
            k3 = k[2, ...]
            k30 = k3 + beta * k1

            kk0 = k1**2 + k2**2 + k30**2
            self.kk0 = kk0

            #### Isotropic with k0

            const = (
                self.alpha
                * (self.epsilon ** (2 / 3))
                * (self.corrlen ** (17 / 3))
                / (4 * np.pi)
            )
            const = np.sqrt(const / (1 + (self.corrlen**2) * kk0) ** (17 / 6))

            # #### RDT

            s = k1**2 + k2**2
            C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
            tmp = beta * k1 * np.sqrt(s) / (kk0 - k30 * k1 * beta)
            C2 = k2 * kk0 / s ** (3 / 2) * np.arctan(tmp)

            zeta1 = C1 - k2 / k1 * C2
            zeta2 = k2 / k1 * C1 + C2
            zeta3 = kk0 / kk

            # deal with divisions by zero
            zeta1 = np.nan_to_num(zeta1)
            zeta2 = np.nan_to_num(zeta2)
            zeta3 = np.nan_to_num(zeta3)

            SqrtSpectralTens[0, 0, ...] = const * zeta3
            SqrtSpectralTens[1, 1, ...] = const * zeta3
            SqrtSpectralTens[2, 0, ...] = -const * zeta1
            SqrtSpectralTens[2, 1, ...] = -const * zeta2
            SqrtSpectralTens[2, 2, ...] = const

            return SqrtSpectralTens

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


###################################################################

#######################################################################################################
# Covariance class for computing the generalized vorticity
#######################################################################################################


class GeneralizedVorticityCovariance(VectorPotentialCovariance):
    def __init__(self, ndim, **kwargs):
        super().__init__(**kwargs)

    # --------------------------------------------------------------------------
    #   Compute the power spectrum
    # --------------------------------------------------------------------------

    def precompute_Spectrum(self, Frequences):
        SqrtSpectralTens = super().__call__(Frequences)
        kk0 = self.kk0

        SqrtSpectralTens[0, 0, ...] *= kk0
        SqrtSpectralTens[1, 1, ...] *= kk0
        SqrtSpectralTens[2, 0, ...] *= kk0
        SqrtSpectralTens[2, 1, ...] *= kk0
        SqrtSpectralTens[2, 2, ...] *= kk0

        return SqrtSpectralTens

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


###################################################################
