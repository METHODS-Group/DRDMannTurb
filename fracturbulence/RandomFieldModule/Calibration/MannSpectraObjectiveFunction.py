from math import *
import numpy as np
import scipy.optimize
from scipy.special import hyp2f1
from collections.abc import Iterable
from pylab import *
from time import time, sleep
from multiprocessing import Process
from matplotlib.animation import FuncAnimation
import scipy.fftpack as fft


# sys.path.append("/Users/bk/Work/Papers/Collaborations/2020_inletgeneration/code/source/")
# sys.path.append("/home/bkeith/Work/Papers/2020_inletgeneration/code/source/")
# sys.path.append("/home/khristen/Projects/Brendan/2019_inletgeneration/code/source")

from .EddyLifetime2 import EddyLifetime
from .GenericObjectiveFunction import GenericObjectiveFunction

# from RandomFieldModule.PowerSpectra import StdEddyLifetime, MannEddyLifetime, EnergySpectrum

###################################################################################################
# Mann Power Spectrum
###################################################################################################


def EnergySpectrum(kL, nu=17 / 6, p=4):
    E = kL**p / (1 + kL**2) ** nu
    return E


###################################################################################################
# Eddy Lifetime
###################################################################################################


### Standard Eddy Liftime
def StdEddyLifetime(kL):
    return (kL) ** (-2 / 3)


### Mann's Eddy Liftime
def MannEddyLifetime(kL):
    return (kL) ** (-2 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -((kL) ** (-2))))
    # return (kL)**(-2/3) / np.sqrt( hyp2f1(4/3, 17/6, 7/3, -(kL)**(-2) ) )


###################################################################################################
#   Rapid distortion one-point spectra
###################################################################################################


class MannSpectraObjectiveFunction(GenericObjectiveFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ### Eddy Life Time (redefinition)
        self.EddyLifetime = EddyLifetime(**kwargs)
        self.EddyLifetimeFunc = self.EddyLifetime
        # self.EddyLifetimeFunc = MannEddyLifetime
        # self.EddyLifetimeFunc = StdEddyLifetime
        # self.EddyLifetimeFunc = lambda k: 0

        ### Dimensions and grids
        # DP = np.array(self.DataPoints)
        # self.grid_k1 = DP[:,0] if DP.ndim > 1 else DP
        # self.grid_k2 = (2*pi)*np.r_[-np.logspace(-2,5,100)[::-1], 0, np.logspace(-2,5,100)]
        N = 200
        p0, p1, p2 = -2, 3, 3
        # self.grid_k2 = np.r_[-np.logspace(p0,p1,N)[::-1], 0, np.logspace(p0,p1,N)]
        # g = 10**p1 * np.linspace(0,1,N)**5
        g = np.r_[0, np.logspace(p0, p1, N)]
        # g = np.logspace(p0,p1,N)
        # g = np.linspace(0,100,N)
        self.grid_k2 = np.r_[-g[:0:-1], g]
        # self.grid_k2 = np.r_[-g[::-1], g]
        # self.grid_k2 = N*fft.fftfreq(N)
        g = np.r_[0, np.logspace(p0, p2, N)]
        self.grid_k3 = self.grid_k2  # np.r_[-g[:0:-1], g]
        # self.grid_k3 = np.r_[-np.logspace(p0,p1,N)[::-1], 0, np.logspace(p0,p1,N)]
        self.dk2 = np.diff(self.grid_k2)
        self.dk3 = np.diff(self.grid_k3)

        self.z = 1

    ###------------------------------------------------
    ### Parameters
    ###------------------------------------------------

    @property
    def Common_parameters(self):
        return np.array([self.lengthscale, self.scalingfactor, self.Gamma])

    @Common_parameters.setter
    def Common_parameters(self, params):
        self.lengthscale, self.scalingfactor, self.Gamma = params

    def init_Common_parameters(self, **kwargs):
        self.lengthscale = 0.59 * self.z
        self.scalingfactor = 3.2 / self.z ** (2 / 3)
        self.Gamma = 3.9
        # self.lengthscale     = 0.79
        # self.scalingfactor   = 2.8
        # self.Gamma           = 3.8
        # self.lengthscale     = 5.76216787e-01
        # self.scalingfactor   = 3.07287639e+00
        # self.Gamma           = 1.23773881e+00

    # NOTE: NN parameters are defined in the parent class !

    # ------------------------------------------------------------------------------------------
    # Computational part
    # ------------------------------------------------------------------------------------------

    def initialize(self, **kwargs):
        self.init_Common_parameters(**kwargs)

        ### Init EddyLifeTime
        configEddyLifeTime = kwargs.get("configEddyLifeTime", {})
        if configEddyLifeTime:
            self.EddyLifetime.Initialize(**configEddyLifeTime)

        ### Sizes
        self.nParams = len(self.All_parameters)

        ### Arrays
        self.SP = np.zeros_like(self.DataValues)
        self.DSP = np.zeros([self.nParams, self.nDataPoints, self.vdim, self.vdim])

        if self.verbose:
            print("Initialization complete.")

        ### TODO: add an option for Reynolds stress computation

    # ------------------------------------------------------------------------------------------

    def update(self, theta=None, jac=False):
        if theta is not None:
            self.All_parameters = theta

        J = 0
        if jac:
            DJ = np.zeros([len(self.All_parameters)])

        FullSpectrum, gradFullSpectrum = self.compute_spectrum(jac=jac)

        self.SP = self.Quad_k3(self.Quad_k2(FullSpectrum))
        # self.SP = fft.fftn(FullSpectrum, axes=(1,2))[:,0,0,...]
        if jac:
            self.DSP = self.Quad_k2(self.Quad_k3(gradFullSpectrum))

        for l in range(self.nDataPoints):
            for i in range(self.vdim):
                for j in range(
                    self.vdim
                ):  ### TODO: accelerate due to symmetry of Reynolds stress
                    if i == j:  ### NOTE: Only dioganal of Reynolds stress
                        eps = (
                            self.SP[l, i, j] - self.DataValues[l, i, j]
                        ) / self.DataValues[l, i, j]
                        if i == 1:  ### fit only specific components
                            J += eps**2
                            if jac:
                                DJ += eps * self.DSP[l, i, j] / self.DataValues[l, i, j]

        self.J = 0.5 * J
        if jac:
            self.DJ = DJ

        if self.verbose:
            print("Updated.")

    # ------------------------------------------------------------------------------------------

    def compute_spectrum(self, jac=False):
        # TODO: gradient computation

        L = self.lengthscale
        factor = self.scalingfactor
        Gamma = self.Gamma

        # ==========================================
        ### data for the plot
        grid_kL = self.grid_k1 * L
        # self.tau_model = Gamma * self.EddyLifetimeFunc(grid_kL)
        self.tau_model = MannEddyLifetime(grid_kL) + 0 * self.EddyLifetimeFunc(grid_kL)
        # ==========================================

        L1 = L * np.array([1, 1, 1])

        # with np.errstate(divide='ignore', invalid='ignore'):

        k = np.meshgrid(self.grid_k1, self.grid_k2, self.grid_k3, indexing="ij")
        k1 = k[0]
        k2 = k[1]
        k3 = k[2]

        kk = k1**2 + k2**2 + k3**2
        kL = np.sqrt(
            L1[0] ** 2 * k1**2 + L1[1] ** 2 * k2**2 + L1[2] ** 2 * k3**2
        )  # *L

        beta = Gamma * (
            MannEddyLifetime(kL)
        )  # + 0*self.EddyLifetimeFunc(k1*L, k2*L, k3*L) )
        # beta = Gamma * ( StdEddyLifetime(kL) + 0*self.EddyLifetimeFunc(k1*L, k2*L, k3*L) )

        # if jac:
        #     dbeta_dL = Gamma * dtau_dL(kL)
        #     dbeta_dG = MannEddyLifetime(kL)

        k30 = k3 + beta * k1
        kk0 = k1**2 + k2**2 + k30**2
        k0 = np.sqrt(kk0)
        k0L = k0 * L  # np.sqrt(L1[0]**2*k1**2 + L1[1]**2*k2**2 + L1[2]**2*k30**2)

        # if jac:
        #     dk30_dL = k1*dbeta_dL
        #     dbeta_dG = MannEddyLifetime(kL)

        E0 = L ** (5 / 3) * EnergySpectrum(k0L)

        s = k1**2 + k2**2
        C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
        arg = beta * k1 * np.sqrt(s) / (kk0 - k30 * k1 * beta)
        C2 = k2 * kk0 / s ** (3 / 2) * np.arctan(arg)

        zeta1 = C1 - k2 / k1 * C2
        zeta2 = k2 / k1 * C1 + C2
        zeta3 = kk0 / kk

        ### deal with divisions by zero
        zeta1 = np.nan_to_num(zeta1)
        zeta2 = np.nan_to_num(zeta2)
        zeta3 = np.nan_to_num(zeta3)

        Phi = np.tile(np.zeros_like(kk), (3, 3, 1, 1, 1))
        Phi[0, 0] = (
            E0
            / (4 * pi * kk0**2)
            * (
                k0**2
                - k1**2
                - 2 * k1 * k30 * zeta1
                + (k1**2 + k2**2) * zeta1**2
            )
        )
        Phi[1, 1] = (
            E0
            / (4 * pi * kk0**2)
            * (
                k0**2
                - k2**2
                - 2 * k2 * k30 * zeta2
                + (k1**2 + k2**2) * zeta2**2
            )
        )
        Phi[2, 2] = E0 / (4 * pi * kk**2) * (k1**2 + k2**2)
        Phi[0, 1] = (
            E0
            / (4 * pi * kk0**2)
            * (
                -k1 * k2
                - k1 * k30 * zeta2
                - k2 * k30 * zeta1
                + (k1**2 + k2**2) * zeta1 * zeta2
            )
        )
        Phi[0, 2] = E0 / (4 * pi * kk0 * kk) * (-k1 * k30 + (k1**2 + k2**2) * zeta1)
        Phi[1, 2] = E0 / (4 * pi * kk0 * kk) * (-k2 * k30 + (k1**2 + k2**2) * zeta2)
        Phi[1, 0] = Phi[0, 1]
        Phi[2, 0] = Phi[0, 2]
        Phi[2, 1] = Phi[1, 2]

        F = k1 * Phi

        # ==============

        if jac:
            grafF = np.tile(
                np.zeros_like(kk), (3, 3, 1, 1, 1, len(self.All_parameters))
            )
            grafF[..., 1] = F
        else:
            gradF = None

        F = factor * F

        ### Bring to the right format
        F = np.transpose(F, axes=[2, 3, 4, 0, 1])
        if jac:
            grafF = np.transpose(grafF, axes=[2, 3, 4, 0, 1, 5])

        gradF = None

        return F, gradF

    ###------------------------------------------------
    ### Tools
    ###------------------------------------------------

    def Quad_k2(self, f):
        return np.sum(
            0.5
            * (f[:, 1:, ...] + f[:, :-1, ...])
            * self.dk2[None, :, None, None, None],
            axis=1,
        )

    def Quad_k3(self, f):
        return np.sum(
            0.5 * (f[:, 1:, ...] + f[:, :-1, ...]) * self.dk3[None, :, None, None],
            axis=1,
        )


###################################################################################################
###################################################################################################

if __name__ == "__main__":
    """TEST by method of manufactured solutions"""

    pass
