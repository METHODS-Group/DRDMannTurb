import os
import sys
from collections.abc import Callable, Iterable

# from pathos.multiprocessing import ProcessingPool as Pool
# from multiprocessing import Process, Value, Array
# from multiprocessing import Pool
from functools import partial
from math import *
from multiprocessing import Pool
from time import time

import numpy as np
import scipy.fftpack as fft
from scipy.special import hyp2f1

from .utilities.common import FourierOfGaussian, SpacialCovariance, autocorrelation
from .utilities.fde_solve import fde_solve

METHOD_DST = "dst"
METHOD_DCT = "dct"
METHOD_FFT = "fft"
METHOD_FFTW = "fftw"
METHOD_VF_FFTW = "vf_fftw"
METHOD_VF_FFT_HALFSPACE = "vf_fft_halfspace"
METHOD_VF_FFT_HALFSPACE_SPDE = "vf_fft_halfspace_spde"
METHOD_NFFT = "nfft"
METHOD_VF_NFFT = "vf_nfft"
METHOD_H2 = "H2"
METHOD_H2_hlibpro = "H2_hlibpro"
METHOD_H2_h2tools = "H2_h2tools"
METHOD_ODE = "ODE"
METHOD_RAT = "Rational"
METHOD_VF_RAT_HALFSPACE_VK = "vf_rat_halfspace_VK"
METHOD_VF_RAT_HALFSPACE_GEN_VK = "vf_rat_halfspace_gen_VK"
METHOD_VF_RAT_HALFSPACE_RAPID_DISTORTION = "vf_rat_halfspace_rapid_distortion"


#######################################################################################################


class Sampling_method_base:
    def __init__(self, RandomField):
        self.verbose = RandomField.verbose
        self.L, self.Nd, self.ndim = (
            RandomField.L,
            RandomField.ext_grid_shape,
            RandomField.ndim,
        )
        self.DomainSlice = RandomField.DomainSlice


class Sampling_method_freq(Sampling_method_base):
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
### - Uses the Fastest Fourier Transform on the West


class Sampling_FFTW(Sampling_method_freq):
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
    def __init__(self, RandomField):
        super().__init__(RandomField)

        import pyfftw

        try:
            n_cpu = int(os.environ["OMP_NUM_THREADS"])
        except:
            n_cpu = 1
        shpR = RandomField.ext_grid_shape
        shpC = shpR.copy()
        shpC[-1] = int(shpC[-1] // 2) + 1
        axes = np.arange(self.ndim)
        flags = ("FFTW_MEASURE", "FFTW_DESTROY_INPUT", "FFTW_UNALIGNED")
        self.fft_x = pyfftw.empty_aligned(shpR, dtype="float64")
        self.fft_y = pyfftw.empty_aligned(shpC, dtype="complex128")
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
        self.Spectrum_half = self.Spectrum[..., : shpC[-1]]
        self.hat_noise = np.stack(
            [np.zeros(shpC, dtype="complex128") for _ in range(3)], axis=-1
        )
        # self.TransformNorm = self.TransformNorm / np.sqrt(self.Nd.prod())
        self.shpC = shpC

    def __call__(self, noise):
        tmp = np.zeros(noise.shape)
        # tmp2 = np.zeros(noise.shape, dtype='complex64')
        for i in range(noise.shape[-1]):
            self.fft_x[:] = noise[..., i]
            self.fft_plan()
            self.hat_noise[..., i] = self.fft_y[:]
            # tmp2[...,i] = FourierOfGaussian(noise[...,i]) * np.sqrt(self.Nd.prod())
        # TODO use an anisotropic TranformOfGaussian function
        # self.hat_noise  = np.einsum('kl...,...l->...k' , self.Spectrum, self.hat_noise)
        self.hat_noise = np.einsum(
            "kl...,...l->...k", self.Spectrum_half, self.hat_noise
        )

        # div=0

        # k = np.array(list(np.meshgrid(*self.Frequencies, indexing='ij')))[...,:self.shpC[-1]]

        for i in range(noise.shape[-1]):
            self.fft_y[:] = self.hat_noise[..., i]
            # div += self.hat_noise[...,i] * 1j * k[i,...]
            self.ifft_plan()
            tmp[..., i] = self.fft_x[:]

        # self.fft_y[:] = div
        # self.ifft_plan()
        # div2 = self.fft_x[:]

        return tmp[self.DomainSlice] / self.TransformNorm


#######################################################################################################
#######################################################################################################
# 	Parallel computations
#######################################################################################################
#######################################################################################################


def func_parallel_polar(arg, rhs, **kwargs):
    r, solve = arg
    # def func_parallel_polar(args):
    # fde_solve, r, rhs, Robin_const, component = args
    # print(fde_solve)
    arr = np.zeros([r.size, rhs.size], dtype=np.complex)
    for i in range(r.size):
        arr[i, :] = solve(rhs, r[i], 0, **kwargs)
    return arr


def func_parallel_polar2(solve_split, r_split, **kwargs):
    nproc = len(r_split)
    pool = Pool(nproc)
    # solve_split = [deepcopy(solve) for i in range(nproc)]
    func = partial(func_parallel_polar, **kwargs)
    return list(pool.map(func, zip(r_split, solve_split)))
