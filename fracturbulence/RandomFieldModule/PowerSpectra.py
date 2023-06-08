
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

from RandomFieldModule.utilities import Matern_kernel, GM_kernel, EP_kernel


###################################################################################################
# Mann Power Spectrum
###################################################################################################

def EnergySpectrum(kL, nu=17/6):
    E = kL**4 / (1+kL**2)**nu
    return E






###################################################################################################
# Eddy Liftime
###################################################################################################

### Standard Eddy Liftime
def StdEddyLifetime(kL):
    return (kL)**(-2/3)

### Mann's Eddy Liftime
def MannEddyLifetime(kL):
    return (kL)**(-2/3) / np.sqrt( hyp2f1(1/3, 17/6, 4/3, -(kL)**(-2) ) )


###################################################################################################
# Mann's Power Spectrum
###################################################################################################

def MannPowerSpectrum(k, **kwargs):

    L = kwargs.get('L', 1)
    Gamma = kwargs.get('Gamma', 1)
    factor = kwargs.get('factor', None)

    if factor is None:
        alpha = kwargs.get('KolmogorovConst', 1)
        epsilon = kwargs.get('DissipationRate', 1)
        factor = alpha * epsilon**(2/3)

    # with np.errstate(divide='ignore', invalid='ignore'):

    k1  = k[0,...]
    k2  = k[1,...]
    k3  = k[2,...]

    kk = k1**2 + k2**2 + k3**2
    kL = np.sqrt(kk)*L

    beta = Gamma * MannEddyLifetime(kL)
    # beta = Gamma * StdEddyLifetime(kL)

    k30 = k3 + beta*k1
    kk0 = k1**2 + k2**2 + k30**2
    k0  = np.sqrt(kk0)
    k0L = k0*L

    E0 = factor* L**(5/3) * EnergySpectrum(k0L) 

    s = k1**2 + k2**2
    C1  =  beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
    arg =  beta * k1 * np.sqrt(s) / (kk0 - k30 * k1 * beta)
    C2  =  k2 * kk0 / s**(3/2) * np.arctan(arg)

    zeta1 =  C1 - k2/k1 * C2
    zeta2 =  k2/k1 *C1 + C2
    zeta3 =  kk0/kk

    # deal with divisions by zero
    # zeta1 = np.nan_to_num(zeta1)
    # zeta2 = np.nan_to_num(zeta2)
    # zeta3 = np.nan_to_num(zeta3)

    Phi = np.tile(np.zeros_like(kk), (3,3,1,1,1))
    Phi[0,0] = E0/(4*pi*kk0**2) * (k0**2 - k1**2 - 2*k1*k30*zeta1 + (k1**2+k2**2)*zeta1**2)
    Phi[1,1] = E0/(4*pi*kk0**2) * (k0**2 - k2**2 - 2*k2*k30*zeta2 + (k1**2+k2**2)*zeta2**2)
    Phi[2,2] = E0/(4*pi*kk**2)  * (k1**2+k2**2)
    Phi[0,1] = E0/(4*pi*kk0**2) * (-k1*k2 - k1*k30*zeta2 - k2*k30*zeta1 + (k1**2+k2**2)*zeta1*zeta2)
    Phi[0,2] = E0/(4*pi*kk0*kk) * (-k1*k30 + (k1**2+k2**2)*zeta1)
    Phi[1,2] = E0/(4*pi*kk0*kk) * (-k2*k30 + (k1**2+k2**2)*zeta2)

    return Phi



###################################################################################################
###################################################################################################

if __name__ == "__main__":

    '''TEST by method of manufactured solutions'''
    
    pass