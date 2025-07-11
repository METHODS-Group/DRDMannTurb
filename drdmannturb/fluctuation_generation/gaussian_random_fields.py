"""
Implements several random field generators.

The main class is `GaussianRandomField`, which generates
a Gaussian random field from a given covariance kernel.

The `VectorGaussianRandomField` class generates a vector of
Gaussian random fields in a specified number of dimensions.
"""

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .covariance_kernels import Covariance as covariance_metaclass
from .sampling_methods import (
    METHOD_DCT,
    METHOD_DST,
    METHOD_FFT,
    METHOD_FFTW,
    METHOD_VF_FFTW,
    Sampling_DCT,
    Sampling_DST,
    Sampling_FFT,
    Sampling_FFTW,
    Sampling_VF_FFTW,
)


@runtime_checkable
class SamplingMethod(Protocol):
    """Interface for sampling methods."""

    DomainSlice: tuple[slice, ...]

    def __call__(self, noise: np.ndarray) -> np.ndarray:
        """
        Correlate the noise with the covariance kernel.

        Parameters
        ----------
        noise : np.ndarray
            Noise to be correlated with the covariance kernel.
        """
        ...


class GaussianRandomField:
    r"""
    Generator for a discrete Gaussian random field.

    Several sampling methods are provided by default:

    #. fft - SciPy Fast Fourier Transform (default)
    #. fftw - "Fastest Fourier Transform in the West"
    #. vf_fftw - vector field version of Fastest Fourier Transform in the West
    #. dst - Discrete Sine Transform
    #. dct - Discrete Cosine Transform

    Please refer to the sampling methods documentation for more details on each of these methods.
    """

    def __init__(
        self,
        Covariance: covariance_metaclass,
        grid_level: np.ndarray,
        grid_shape: np.ndarray = None,
        grid_dimensions: np.ndarray = [1.0, 1.0, 1.0],
        ndim: int = 3,
        sampling_method: str = "fft",
    ):
        """
        Initialize the Gaussian random field generator.

        Parameters
        ----------
        Covariance : Covariance
            An instantiation of one of the Covariance classes (VonKarman, Mann, or NN) provided in the package that
            subclass from the Covariance metaclass.
        grid_level : np.ndarray
            Numpy array denoting the grid levels; number of discretization points used in each dimension, which
            evaluates as 2^k for each dimension for FFT-based sampling methods.
        grid_shape : np.ndarray, optional
            _description_, by default None
        grid_dimensions : np.ndarray, optional
            Numpy array denoting the grid size; the real dimensions of the domain of interest,
            by default [1.0, 1.0, 1.0]
        ndim : int, optional
           Number of dimensions of the random field, by default 3, which is currently forced (only 3D field
           generation is implemented).
        sampling_method : str, optional
            Sampling method to be used in correlating with field generated from underlying distribution
            (Gaussian), by default "fft"

        Raises
        ------
        ValueError
            ``grid_level`` and ``grid_shape`` must have the same dimensions.
        """
        self.ndim = ndim  # dimension 2D or 3D
        self.all_axes = np.arange(self.ndim)

        # TODO: This is a strange block of code
        if np.isscalar(grid_level):
            if not np.isscalar(grid_shape):
                raise ValueError("grid_level is scalar, so grid_shape should be as well.")

            h = 1 / 2**grid_level
            self.grid_shape = np.array([grid_shape] * ndim)
        else:
            assert len(grid_dimensions) == 3
            assert len(grid_level) == 3

            h = grid_dimensions / (2**grid_level + 1)
            self.grid_shape = np.array(grid_shape[:ndim])

        self.L = h * self.grid_shape

        ### Extended window (NOTE: extension is done outside)
        N_margin = 0
        self.ext_grid_shape = self.grid_shape
        self.nvoxels = self.ext_grid_shape.prod()
        self.DomainSlice = tuple([slice(N_margin, N_margin + self.grid_shape[j]) for j in range(self.ndim)])

        ### Covariance kernel
        self.Covariance = Covariance

        # Pseudo-random number generator
        self.prng = np.random.RandomState()
        self.noise_std = np.sqrt(np.prod(h))
        self.distribution = self.prng.normal

        # Annotate self.Correlate with the protocol
        self.Correlate: SamplingMethod
        self.setSamplingMethod(sampling_method)

    def setSamplingMethod(self, method: str):
        """
        Initialize the sampling method.

        Parameters
        ----------
        method : str
            One of "fft", "dst", "dct", "fftw", "vf_fftw"; see Sampling methods.

        Raises
        ------
        Exception
            If method is not one of the above.
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
        """
        Set a new seed for the class's PRNG.

        Parameters
        ----------
        seed : Seed value for PRNG, following ``np.RandomState()`` conventions,
            by default None
        """
        if seed is not None:
            self.prng.seed(seed)
        else:
            self.prng.seed()

    ### Sample noise
    def sample_noise(self, grid_shape: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample random grid from specified distribution.

        Parameters
        ----------
        grid_shape : Optional[np.ndarray], optional
            Number of grid points in each dimension, by default None, which results in a field over the grid
            determined during construction of this object.

        Returns
        -------
        np.ndarray
            Sampled random values from underlying distribution of size matching the given grid shape.
        """
        if grid_shape is None:
            noise = self.distribution(0, 1, self.ext_grid_shape)
        else:
            noise = self.distribution(0, 1, grid_shape)

        noise *= self.noise_std
        return noise

    ### Sample GRF
    def sample(self, noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample the Gaussian random field with specified sampling method.

        Note this is not just what is sampled from the distribution;
        this field is also correlated with the covariance kernel provided
        during construction.

        Parameters
        ----------
        noise : Optional[np.ndarray], optional
            Random field from underlying distribution, by default None, which results in an additional sampling of
            the domain from the distribution.

        Returns
        -------
        np.ndarray
            Sampled grid from distribution.

        Raises
        ------
        Exception
            If Sampling method has not been set already.
        """
        if not hasattr(self, "Correlate"):
            raise Exception("Sampling method not set.")

        if noise is None:
            noise = self.sample_noise()

        field = self.Correlate(noise)

        return field


class VectorGaussianRandomField(GaussianRandomField):
    """
    Gaussian random vector field generator.

    This class generates a vector of Gaussian random fields in a specified number of dimensions.
    """

    def __init__(self, vdim: int = 3, **kwargs):
        """
        Initialize a vector GRF generator in specified number of dimensions.

        ``kwargs`` must contain all information required by :py:class:`GaussianRandomField`.

        Parameters
        ----------
        vdim : int, optional
            Dimension count of vector of GRF, by default 3
        """
        super().__init__(**kwargs)
        self.vdim = vdim
        self.DomainSlice = tuple(list(self.DomainSlice) + [slice(None, None, None)])

        self.Correlate.DomainSlice = self.DomainSlice
