"""
This module implements a random field generator.

Notes
-----

This should not be directly used but is needed by ``GenerateFluctuationField``.
"""

import numpy as np

from drdmannturb.fluctuation_generation.sampling_methods import *


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
        laplace: bool = False,
        **kwargs,
    ):
        """Constructor for Gaussian Random Field generator

        Parameters
        ----------
        grid_level : np.ndarray

        grid_shape : _type_, optional
            _description_, by default ``None``
        grid_dimensions : list, optional
            _description_, by default ``[1.0, 1.0, 1.0]``
        ndim : int, optional
            Denotes either 2d or 3d, by default 2
        window_margin : int, optional
            _description_, by default 0
        sampling_method : str, optional
            _description_, by default ``"fft"``
        verbose : int, optional
            , by default 0
        laplace : bool, optional
            If true, uses noise from a Laplace distribution instead of
            Gaussian, by default ``False``.

        Warning
        -------
        The Laplace distribution option ``laplace`` is not supported
        and does not lead to physically sound results.

        Raises
        ------
        ValueError
            ``grid_level`` and ``grid_shape`` must have the same dimensions.
        """
        self.verbose = verbose
        self.ndim = ndim  # dimension 2D or 3D
        self.all_axes = np.arange(self.ndim)

        if np.isscalar(grid_level):
            if not np.isscalar(grid_shape):
                raise ValueError(
                    "grid_level and grid_shape must have the same dimensions."
                )
            h = 1 / 2**grid_level
            self.grid_shape = np.array([grid_shape] * ndim)
        else:
            assert len(grid_dimensions) == 3
            assert len(grid_level) == 3

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
            print(f"Init method {self.method}, time {time()-t}")

        # Pseudo-random number generator
        self.prng = np.random.RandomState()
        self.noise_std = np.sqrt(np.prod(h))
        self.distribution = self.prng.laplace if laplace else self.prng.normal

    def setSamplingMethod(self, method, **kwargs):
        """Initialize the sampling method

        Parameters
        ----------
        method : str
            One of "fft", "dst", "dct", "fftw", "vf_fftw"; see Sampling methods.

        Raises
        ------
        Exception
            If method is not one of the above
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
            raise Exception(f'Unknown sampling method "{method}".')

    def reseed(self, seed=None):
        """Sets a new seed for the class's PRNG.

        Parameters
        ----------
        seed : Seed value for PRNG, following np.RandomState() conventions,
            by default None
        """
        if seed is not None:
            self.prng.seed(seed)
        else:
            self.prng.seed()

    ### Sample noise
    def sample_noise(self, grid_shape=None) -> np.ndarray:
        """
        Sample

        Args:
            grid_shape (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        if grid_shape is None:
            # noise = self.prng.normal(0, 1, self.ext_grid_shape)
            noise = self.distribution(0, 1, self.ext_grid_shape)
        else:
            # noise = self.prng.normal(0, 1, grid_shape)]
            noise = self.distribution(0, 1, grid_shape)
        noise *= self.noise_std
        return noise

    ### Sample GRF
    def sample(self, noise=None):
        """
        Gaussian Random Field sampler
        """
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
        """Constructor for vector of GRFs.

        Parameters
        ----------
        vdim : int, optional
            Dimension count of vector of GRF, by default 3
        """
        super().__init__(**kwargs)
        self.vdim = vdim
        self.DomainSlice = tuple(list(self.DomainSlice) + [slice(None, None, None)])

        self.Correlate.DomainSlice = self.DomainSlice
