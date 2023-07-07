
import pyfftw
from math import *
import numpy as np
from scipy.special import kv as Kv
from scipy.linalg import sqrtm
from itertools import product
from time import time
from tqdm import tqdm

import scipy.fftpack as fft
import matplotlib.pyplot as plt

from . import CovarianceKernels
from .utilities.common import FourierOfGaussian, SpacialCovariance, autocorrelation
from .Sampling_Methods import *


#######################################################################################################
#	Gaussian Random Field generator class
#######################################################################################################

class GaussianRandomField:

    def __init__(self, grid_level, grid_shape=None, ndim=3, window_margin=0, sampling_method='fft', verbose=0, nodes=None, seed=None, **kwargs):
        self.verbose = verbose
        self.ndim = ndim     # dimension 2D or 3D
        self.all_axes = np.arange(self.ndim)

        N = 2**grid_level + 1
        h = 1/N
        
        if grid_shape is None: ### Unit square (cube)
            self.grid_shape = [N] * ndim
        elif np.isscalar(grid_shape):
            self.grid_shape = [grid_shape]*ndim
        else:
            self.grid_shape = grid_shape[:ndim]
        self.grid_shape = np.array(self.grid_shape)
        self.L = np.array([1] * ndim)
        # self.L = h * self.grid_shape

        ### only for nfft
        self.nodes = nodes

        ### Extended window
        N_margin = ceil(window_margin/h)
        self.ext_grid_shape = self.grid_shape + 2*N_margin
        self.nvoxels = self.ext_grid_shape.prod()
        self.DomainSlice = tuple([slice(N_margin, N_margin + self.grid_shape[j]) for j in range(self.ndim)])

        ### Covariance
        self.Covariance = kwargs['Covariance']

        ### Gamma
        self.Gamma = kwargs.get('Gamma',  None)

        ### Sampling method
        t = time()
        self.setSamplingMethod(sampling_method, **kwargs)
        if self.verbose:
            print('Init method {0:s}, time {1}'.format(self.method, time()-t))

        # Pseudo-random number generator
        if seed is None:
            self.prng = np.random.RandomState()
        else:
            self.prng = np.random.RandomState(seed=seed)
        self.noise_std = np.sqrt(np.prod(self.L/self.grid_shape))


    #--------------------------------------------------------------------------
    #   Initialize sampling method
    #--------------------------------------------------------------------------

    def setSamplingMethod(self, method, **kwargs):
        self.method = method

        if method == METHOD_FFT:
            self.Correlate = Sampling_FFT(self)

        elif method == METHOD_DST:
            self.Correlate = Sampling_DST(self)

        elif method == METHOD_DCT:
            self.Correlate = Sampling_DCT(self)

        elif method == METHOD_NFFT:
            self.Correlate = Sampling_NFFT(self)

        elif method == METHOD_FFTW:
            self.Correlate = Sampling_FFTW(self)

        elif method == METHOD_VF_FFTW:
            self.Correlate = Sampling_VF_FFTW(self)

        elif method == METHOD_VF_FFT_HALFSPACE:
            self.Correlate = Sampling_VF_Halfspace(self)

        elif method == METHOD_VF_FFT_HALFSPACE_SPDE:
            self.Correlate = Sampling_VF_Halfspace_SPDE(self)

        elif method == METHOD_VF_NFFT:
            self.Correlate = Sampling_VF_NFFT(self)

        elif method in (METHOD_H2, METHOD_H2_hlibpro, METHOD_H2_h2tools):
            self.Correlate = Sampling_H2(self, **kwargs)

        elif method in (METHOD_ODE,):
            self.Correlate = Sampling_ODE(self, **kwargs)

        elif method in (METHOD_RAT,):
            self.Correlate = Sampling_Rational(self, **kwargs)

        elif method == METHOD_VF_RAT_HALFSPACE_VK:
            self.Correlate = Sampling_Rational_VK_Wind_Blocking(self, **kwargs)
        
        elif method == METHOD_VF_RAT_HALFSPACE_GEN_VK:
            self.Correlate = Sampling_Rational_Generalized_VK_Wind_Blocking(self, **kwargs)

        elif method == METHOD_VF_RAT_HALFSPACE_RAPID_DISTORTION:
            self.Correlate = Sampling_Rational_Rapid_Distortion_Wind_Blocking(self, **kwargs)
            
        else:
            raise Exception('Unknown sampling method "{0}".'.format(method))


    #--------------------------------------------------------------------------
    #   Sample a realization
    #--------------------------------------------------------------------------

    ### Reseed pseudo-random number generator
    def reseed(self, seed=None):
        if seed is not None:
            self.prng.seed(seed)
        else:
            self.prng.seed()

    ### Sample noise
    def sample_noise(self):
        noise = self.prng.normal(0, 1, self.ext_grid_shape)
        noise *= self.noise_std
        return noise

    ### Sample GRF
    def sample(self, noise=None, **kwargs):

        if noise is None:
            noise = self.sample_noise()

        t0 = time()
        field = self.Correlate(noise, **kwargs)
        if self.verbose>=2: print('Convolution time: ', time() - t0)

        return field


    #--------------------------------------------------------------------------
    #   TESTS
    #--------------------------------------------------------------------------

    def test_Covariance(self, nsamples=1000):

        C = np.zeros(self.grid_shape)
        for isample in tqdm(range(nsamples)):
            X = self.sample()
            C += autocorrelation(X)
        C /= nsamples

        R = np.zeros(self.grid_shape)
        if self.ndim==2:
            n, m = self.grid_shape
            for i in range(n):
                for j in range(m):
                    R[i,j] = sqrt((i/n)**2 + (j/m)**2)
        elif self.ndim==3:
            n, m, l = self.grid_shape
            for i in range(n):
                for j in range(m):
                    for k in range(l):
                        R[i,j,k] = sqrt((i/n)**2 + (j/m)**2 + (k/l)**2)

        C0 = self.Covariance.eval(R.flatten())
        C0 = C0.reshape(self.grid_shape)

        error = np.linalg.norm(C-C0, np.inf)
        print("Covariance error = ", error)

        if self.ndim==2:
            plt.plot(C[:,0], color='blue')
            plt.plot(C0[:,0], color='red')
        elif self.ndim==3:
            plt.plot(C[:,0,0], color='blue')
            plt.plot(C0[:,0,0], color='red')
        plt.legend(["Sampled", "Analythic"])
        plt.show()
        return error


#######################################################################################################
#	Gaussian Random Vector Field generator class
#######################################################################################################

class VectorGaussianRandomField(GaussianRandomField):

    def __init__(self, vdim=3, **kwargs):

        super().__init__(**kwargs)
        self.vdim = vdim
        self.DomainSlice = tuple(list(self.DomainSlice) + [slice(None,None,None)])

        ### Sampling method
        self.Correlate.DomainSlice = self.DomainSlice

    ### Sample noise
    def sample_noise(self):
        noise = np.stack([self.prng.normal(0, 1, self.ext_grid_shape) for _ in range(self.vdim)], axis=-1)
        return noise