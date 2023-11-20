"""
This module implements and exposes the various sampling methods required
"""

import os
from math import *

import numpy as np
import scipy.fftpack as fft

METHOD_DST = "dst"
METHOD_DCT = "dct"
METHOD_FFT = "fft"
METHOD_FFTW = "fftw"
METHOD_VF_FFTW = "vf_fftw"


class Sampling_method_base:
    """Meta class for different sampling methods. Each of these requires a ``RandomField`` object, which is a subclass of :py:class:``GaussianRandomField``."""

    def __init__(self, RandomField):
        """
        Parameters
        ----------
        RandomField : GaussianRandomField
            The random field from which to sample from. This object also determines all of the physical quantities and domain partitioning.
        """
        self.L, self.Nd, self.ndim = (
            RandomField.L,
            RandomField.ext_grid_shape,
            RandomField.ndim,
        )
        self.DomainSlice = RandomField.DomainSlice


class Sampling_method_freq(Sampling_method_base):
    """Sampling method specifically in the frequency domain. This metaclass involves a single precomputation of the covariance spectrum of the underlying ``GaussianRandomField``. Refer to specific subclasses for details on what each of these entails, but generally, the approximate square-root of each associated spectral tensor is computed and transformed into the frequency domain.

    The norm of the transform is defined as the square-root of the length-scale.
    """

    def __init__(self, RandomField):
        super().__init__(RandomField)
        L, Nd, d = self.L, self.Nd, self.ndim
        self.Frequencies = [
            (2 * pi / L[j]) * (Nd[j] * fft.fftfreq(Nd[j])) for j in range(d)
        ]
        self.TransformNorm = np.sqrt(L.prod())
        self.Spectrum = RandomField.Covariance.precompute_Spectrum(self.Frequencies)


#######################################################################################################
# 	Fourier Transform (FFTW)
#######################################################################################################
### - Only stationary covariance
### - Uses the Fastest Fourier Transform in the West


class Sampling_FFTW(Sampling_method_freq):
    """Sampling with FFTW. Two stencils for the forward and inverse FFTs are generated using the following FFTW flags: ``"FFTW_MEASURE", "FFTW_DESTROY_INPUT", "FFTW_UNALIGNED"``.

    Due to properties of the FFT, only stationary covariances are admissible.
    """

    def __init__(self, RandomField):
        super().__init__(RandomField)

        import pyfftw

        shpR = RandomField.ext_grid_shape
        shpC = shpR.copy()
        shpC[-1] = int(shpC[-1] // 2) + 1
        axes = np.arange(self.ndim)
        flags = ("FFTW_MEASURE", "FFTW_DESTROY_INPUT", "FFTW_UNALIGNED")
        self.fft_x = pyfftw.empty_aligned(shpR, dtype="float64")
        self.fft_y = pyfftw.empty_aligned(shpC, dtype="complex128")
        self.fft_plan = pyfftw.FFTW(
            self.fft_x, self.fft_y, axes=axes, direction="FFTW_FORWARD", flags=flags
        )
        self.ifft_plan = pyfftw.FFTW(
            self.fft_y, self.fft_x, axes=axes, direction="FFTW_BACKWARD", flags=flags
        )
        self.Spectrum_half = self.Spectrum[..., : shpC[-1]] * np.sqrt(self.Nd.prod())

    def __call__(self, noise):
        self.fft_x[:] = noise
        self.fft_plan()
        self.fft_y[:] *= self.Spectrum_half
        self.ifft_plan()
        return self.fft_x[self.DomainSlice] / self.TransformNorm


#######################################################################################################
# 	Vector Field Fourier Transform (VF_FFTW)
#######################################################################################################
### - Random vector fields
### - Only stationary covariance
### - Uses the Fastest Fourier Transform in the West


class Sampling_VF_FFTW(Sampling_method_freq):
    """FFTW applied to a vector field. This should be used in conjunction with :py:class:`VectorGaussianRandomField`. This sampling method is also multi-threaded across 4 threads, or else the maximum allowed by the environment. As in :py:class:`Sampling_FFTW`, the following FFTW flags are used: ``"FFTW_MEASURE", "FFTW_DESTROY_INPUT", "FFTW_UNALIGNED"``."""

    def __init__(self, RandomField):
        super().__init__(RandomField)

        import pyfftw

        n_cpu = 4
        try:
            n_cpu = int(os.environ["OMP_NUM_THREADS"])
        except:
            pass
        shpR = RandomField.ext_grid_shape
        shpC = shpR.copy()
        shpC[-1] = int(shpC[-1] // 2) + 1
        axes = np.arange(self.ndim)
        flags = ("FFTW_MEASURE", "FFTW_DESTROY_INPUT", "FFTW_UNALIGNED")
        self.fft_x = pyfftw.empty_aligned(shpR, dtype="float32")
        self.fft_y = pyfftw.empty_aligned(shpC, dtype="complex64")
        self.fft_plan = pyfftw.FFTW(
            self.fft_x,
            self.fft_y,
            axes=axes,
            direction="FFTW_FORWARD",
            flags=flags,
            threads=n_cpu,
        )
        self.ifft_plan = pyfftw.FFTW(
            self.fft_y,
            self.fft_x,
            axes=axes,
            direction="FFTW_BACKWARD",
            flags=flags,
            threads=n_cpu,
        )
        self.Spectrum_half = self.Spectrum[..., : shpC[-1]] * np.sqrt(self.Nd.prod())
        self.hat_noise = np.stack(
            [np.zeros(shpC, dtype="complex64") for _ in range(3)], axis=-1
        )
        self.shpC = shpC

    def __call__(self, noise):
        tmp = np.zeros(noise.shape)
        for i in range(noise.shape[-1]):
            self.fft_x[:] = noise[..., i]
            self.fft_plan()
            self.hat_noise[..., i] = self.fft_y[:]
        self.hat_noise = np.einsum(
            "kl...,...l->...k", self.Spectrum_half, self.hat_noise
        )
        for i in range(noise.shape[-1]):
            self.fft_y[:] = self.hat_noise[..., i]
            self.ifft_plan()
            tmp[..., i] = self.fft_x[:]
        return tmp[self.DomainSlice] / self.TransformNorm


#######################################################################################################
# 	Fourier Transform
#######################################################################################################
### - Only stationary covariance
### - Uses sscipy.fftpack (slower solution)


class Sampling_FFT(Sampling_method_freq):
    """Sampling using ``scipy.fftpack``, which is considerably slower than with FFTW but is a simpler interface."""

    def __init__(self, RandomField):
        super().__init__(RandomField)

    def __call__(self, noise):
        noise_hat = fft.ifftn(noise)
        y = self.Spectrum * noise_hat
        y = fft.fftn(y)
        return y.real[self.DomainSlice] / self.TransformNorm


#######################################################################################################
# 	Sine Transform
#######################################################################################################
### - Only stationary covariance
### - Uses sscipy.fftpack (non the fastest solution)


class Sampling_DST(Sampling_method_freq):
    """Sampling using the discrete sine transform from ``scipy.fftpack``, with all other operations being identical as other sampling methods."""

    def __init__(self, RandomField):
        super().__init__(RandomField)

    def __call__(self, noise):
        y = self.Spectrum * noise
        for j in range(self.ndim):
            y = fft.dst(y, axis=j, type=1)
        return y[self.DomainSlice] / self.TransformNorm


#######################################################################################################
# 	Cosine Transform
#######################################################################################################
### - Only stationary covariance
### - Uses sscipy.fftpack (non the fastest solution)


class Sampling_DCT(Sampling_method_freq):
    """Sampling using the discrete cosine transform from ``scipy.fftpack``, with all other operations being identical as other sampling methods."""

    def __init__(self, RandomField):
        super().__init__(RandomField)

    def __call__(self, noise):
        y = self.Spectrum * noise
        for j in range(self.ndim):
            y = fft.dct(y, axis=j, type=2)
        return y[self.DomainSlice] / self.TransformNorm
