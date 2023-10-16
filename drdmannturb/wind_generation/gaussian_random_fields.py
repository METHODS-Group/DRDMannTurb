"""
This module implements and exposes a Gaussian random field generator
"""

import numpy as np
from drdmannturb.wind_generation.sampling_methods import *

import drdmannturb.loggers as lgg


class GaussianRandomField:
    """
    Gaussian Random Field generator class
    """

    def __init__(
        self,
        grid_level,
        grid_shape=None,
        grid_dimensions=[1.0, 1.0, 1.0],
        ndim=2,
        window_margin=0,
        sampling_method="fft",
        verbose=0,
        **kwargs
    ):
        self.verbose = verbose
        self.ndim = ndim  # dimension 2D or 3D
        self.all_axes = np.arange(self.ndim)

        if np.isscalar(grid_level):
            if not np.isscalar(grid_shape):
                print("grid_level and grid_shape must have the same dimensions")
            h = 1 / 2**grid_level
            self.grid_shape = np.array([grid_shape] * ndim)
        else:
            h = np.array(
                [
                    grid_dimensions[0] / (2 ** grid_level[0] + 1),
                    grid_dimensions[1] / (2 ** grid_level[1] + 1),
                    grid_dimensions[2] / (2 ** grid_level[2] + 1),
                ]
            )
            self.grid_shape = np.array(grid_shape[:ndim])
        self.L = h * self.grid_shape

        ### Extended window (NOTE: extension is done outside)
        N_margin = 0
        self.ext_grid_shape = self.grid_shape
        self.nvoxels = self.ext_grid_shape.prod()
        self.DomainSlice = tuple(
            [slice(N_margin, N_margin + self.grid_shape[j]) for j in range(self.ndim)]
        )

        ### Covariance
        self.Covariance = kwargs["Covariance"]

        ### Sampling method
        t = time()
        self.setSamplingMethod(sampling_method, **kwargs)
        if self.verbose:
            print("Init method {0:s}, time {1}".format(self.method, time() - t))

        # Pseudo-random number generator
        self.prng = np.random.RandomState()
        self.noise_std = np.sqrt(np.prod(h))

    def setSamplingMethod(self, method, **kwargs):
        """
        Initialize the sampling method
        """
        self.method = method

        if method == METHOD_FFT:
            self.Correlate = Sampling_FFT(self)

        elif method == METHOD_DST:
            self.Correlate = Sampling_DST(self)

        elif method == METHOD_DCT:
            self.Correlate = Sampling_DCT(self)

        elif method == METHOD_FFTW:
            self.Correlate = Sampling_FFTW(self)

        elif method == METHOD_VF_FFTW:
            self.Correlate = Sampling_VF_FFTW(self)

        else:
            raise Exception('Unknown sampling method "{0}".'.format(method))

    def reseed(self, seed=None):
        """
        Quick routine to seed the PRNG
        """
        if seed is not None:
            self.prng.seed(seed)
        else:
            self.prng.seed()

    ### Sample noise
    def sample_noise(self, grid_shape=None):
        if grid_shape is None:
            noise = self.prng.normal(0, 1, self.ext_grid_shape)
        else:
            noise = self.prng.normal(0, 1, grid_shape)
        noise *= self.noise_std
        return noise

    ### Sample GRF
    def sample(self, noise=None):
        if noise is None:
            noise = self.sample_noise()

        # noise *= self.noise_std

        t0 = time()
        field = self.Correlate(noise)
        if self.verbose >= 2:
            print("Convolution time: ", time() - t0)

        return field


class VectorGaussianRandomField(GaussianRandomField):
    """
    Gaussian random vector field generator
    """

    def __init__(self, vdim=3, **kwargs):
        super().__init__(**kwargs)
        self.vdim = vdim
        self.DomainSlice = tuple(list(self.DomainSlice) + [slice(None, None, None)])

        self.Correlate.DomainSlice = self.DomainSlice
