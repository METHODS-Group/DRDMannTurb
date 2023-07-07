import os
import sys
from collections.abc import Callable, Iterable
from copy import deepcopy
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

from .utilities.common import (FourierOfGaussian, SpacialCovariance,
                               autocorrelation)
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
# 	Fourier Transform for synthetic wind fields with blocking by the ground
#######################################################################################################
### - Only stationary covariance
### - Uses sscipy.fftpack (non the fastest solution)


# class Sampling_VF_Halfspace(Sampling_method_freq):
class Sampling_VF_Halfspace(Sampling_VF_FFTW):
    def __init__(self, RandomField):
        super().__init__(RandomField)

        self.Gamma = RandomField.Gamma

    def __call__(self, noise):
        fftImplementation = True

        if fftImplementation:
            f = super().__call__(noise)
            # return f
            np.seterr(divide="ignore", invalid="ignore")

            # Now f corresponds to the solution without blocking
            # Let us apply respectively homogeneous Dirichlet on z=0 and Neumann on z=z_max

            # apply free space Laplacian in frequency domain
            k = np.array(list(np.meshgrid(*self.Frequencies, indexing="ij")))
            kk = np.sum(k**2, axis=0)

            # kk = np.sum(k**2,axis=0)
            # for i in range(3):
            #     f[...,i] = fft.ifftn( fft.fftn(f[...,i]) * kk ).real
            i = 2
            f[..., i] = fft.ifftn(fft.fftn(f[..., i]) * kk).real

            # double domain w.r.t z
            L, Nd, d = np.copy(self.L), np.copy(self.Nd), self.ndim
            Nz = Nd[2]
            Nd[2] = 2 * Nd[2]
            L[2] *= 2
            Frequencies2 = [
                (2 * pi / L[j]) * (Nd[j] * fft.fftfreq(Nd[j])) for j in range(d)
            ]
            k = np.array(list(np.meshgrid(*Frequencies2, indexing="ij")))
            kk = np.sum(k**2, axis=0)

            kk[np.where(kk == 0)] = 1
            shape2 = list(noise.shape)
            shape2[2] = Nd[2]
            f_hat2 = np.zeros(shape2, dtype="complex64")
            f_hat2[..., :Nz, :] = f
            # for i in range(3):
            #     f_hat2[...,i] = fft.fft(f_hat2[...,i], axis = 2)
            #     if i==2: f_hat2[...,i] = 2j*f_hat2[...,i].imag
            #     f_hat2[...,i] = fft.fft(fft.fft(f_hat2[...,i], axis = 1), axis = 0)

            #     ### Laplace
            #     f_hat2[...,i] = f_hat2[...,i] / kk
            #     f_hat2[...,i] = fft.ifftn(f_hat2[...,i])
            i = 2
            f_hat2[..., i] = fft.fft(f_hat2[..., i], axis=2)
            f_hat2[..., i] = 2j * f_hat2[..., i].imag
            f_hat2[..., i] = fft.fft(fft.fft(f_hat2[..., i], axis=1), axis=0)
            f_hat2[..., i] = f_hat2[..., i] / kk
            f_hat2[..., i] = fft.ifftn(f_hat2[..., i])

            return f_hat2[..., :Nz, :].real

        else:  # slow implimentation
            ## Construct RHS in Fourier space
            f_hat = np.zeros(noise.shape, dtype="complex64")
            for i in range(noise.shape[-1]):
                f_hat[..., i] = fft.fftn(noise[..., i])
            f_hat = np.einsum("kl...,...l->...k", self.Spectrum, f_hat)

            # apply free space Laplacian in frequency domain
            k = np.array(list(np.meshgrid(*self.Frequencies, indexing="ij")))
            kk = np.sum(k**2, axis=0)
            for i in range(3):
                f_hat[..., i] = f_hat[..., i] * kk

            # Inverse Fourier transform of RHS in z-coordinate
            f_hat = fft.ifft(f_hat, axis=2)

            # Define the Green's function
            # green = lambda c, z1, z2 : np.pi * np.divide(1,c) * ( np.exp(-c*np.abs(z1-z2)) ) if (c > 0) else -np.pi * np.abs(z1-z2) # this is the free space Green's function (for testing)
            green = (
                lambda c, z1, z2: np.pi
                * np.divide(1, c)
                * (np.exp(-c * np.abs(z1 - z2)) - np.exp(-c * np.abs(z1 + z2)))
                if (c > 0)
                else -np.pi * (np.abs(z1 - z2) - np.abs(z1 + z2))
            )
            # green = lambda c, z1, z2 : 1/(2*c) *( np.exp(-c*np.abs(z1-z2)) - np.exp(-c*np.abs(z1+z2)) )
            k12 = np.array(list(np.meshgrid(*self.Frequencies[:2], indexing="ij")))
            k12_norm = np.sqrt(np.sum(k12**2, axis=0))

            G_size = (
                self.Frequencies[0].size,
                self.Frequencies[1].size,
                self.Frequencies[2].size,
                self.Frequencies[2].size,
            )
            k12_norm = np.tile(
                k12_norm[..., np.newaxis, np.newaxis], (1, 1, G_size[2], G_size[3])
            )
            h3 = self.L[2] / self.Nd[2]

            z = np.array(range(G_size[2])) * h3
            zz1, zz2 = np.meshgrid(z, z, indexing="ij")
            zz1 = np.tile(
                zz1[np.newaxis, np.newaxis, ...], (G_size[0], G_size[1], 1, 1)
            )
            zz2 = np.tile(
                zz2[np.newaxis, np.newaxis, ...], (G_size[0], G_size[1], 1, 1)
            )

            # deal with divisions by zero (1/2)
            np.seterr(divide="ignore", invalid="ignore")

            G = np.vectorize(green)(k12_norm, zz1, zz2)

            # deal with divisions by zero (2/2)
            G = np.nan_to_num(G)

            for i in range(noise.shape[-1]):
                # Convolve with Green's function
                f_hat[..., i] = np.einsum("ijkl,ijl->ijk", G, f_hat[..., i])

            # Inverse Fourier transform of RHS in x- and y-coordinates
            f_hat = fft.ifft(fft.ifft(f_hat, axis=1), axis=0)

            return f_hat.real[self.DomainSlice] / self.TransformNorm


#######################################################################################################
# 	Fourier Transform for synthetic wind fields with blocking by the ground
#######################################################################################################
### - Only stationary covariance
### - Uses the Fastest Fourier Transform in the West


# class Sampling_VF_Halfspace_SPDE(Sampling_VF_FFTW):
class Sampling_VF_Halfspace_SPDE(Sampling_method_base):
    def __init__(self, RandomField):
        super().__init__(RandomField)

        L, Nd, d = np.copy(self.L), np.copy(self.Nd), self.ndim
        self.Frequencies = [
            (2 * pi / L[j]) * (Nd[j] * fft.fftfreq(Nd[j])) for j in range(d)
        ]
        self.TransformNorm = np.sqrt(L.prod())

        # double domain w.r.t. z
        Nd[2] *= 2
        L[2] *= 2
        Frequencies2 = [
            (2 * pi / L[j]) * (Nd[j] * fft.fftfreq(Nd[j])) for j in range(d)
        ]
        self.Spectrum = RandomField.Covariance.precompute_Spectrum(Frequencies2)

        import pyfftw

        n_cpu = 1
        shpR = RandomField.ext_grid_shape.copy()
        # double domain w.r.t. z
        shpR[2] *= 2
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
        self.TransformNorm = self.TransformNorm / np.sqrt(self.Nd.prod())
        self.shpR = shpR
        self.shpC = shpC

    def __call__(self, noise):
        ### STEP 1

        # create symmetric noise
        shape2 = list(noise.shape)
        Nz = shape2[2]
        shape2[2] *= 2
        noise2 = np.zeros(shape2, dtype="float64")
        noise2[..., :Nz, :] = noise
        # noise2[...,Nz:,:] = -np.flip(noise, axis=2)
        noise2[..., Nz:, 0] = -np.flip(noise[..., 0], axis=2)
        noise2[..., Nz:, 1] = -np.flip(noise[..., 1], axis=2)
        noise2[..., Nz:, 2] = np.flip(noise[..., 2], axis=2)

        # FFTW
        for i in range(noise2.shape[-1]):
            self.fft_x[:] = noise2[..., i]
            self.fft_plan()
            self.hat_noise[..., i] = self.fft_y[:]

        # correlate noise
        self.hat_noise = np.einsum(
            "kl...,...l->...k", self.Spectrum_half, self.hat_noise
        )

        # IFFTW
        for i in range(noise.shape[-1]):
            self.fft_y[:] = self.hat_noise[..., i]
            self.ifft_plan()
            noise2[..., i] = self.fft_x[:]

        # restrict solution
        psi = noise2.real[..., :Nz, :] / self.TransformNorm

        ### STEP 2

        # # Apply curl
        # k = np.array(list(np.meshgrid(*self.Frequencies, indexing='ij')))
        # k1  = k[0,...]
        # k2  = k[1,...]
        # k3  = k[2,...]

        # u_hat = np.zeros(list(noise.shape), dtype='complex64')
        # for i in range(3):
        #     u_hat[...,i] = fft.fftn(psi[...,i])

        # tmp1 = u_hat[...,0].copy()
        # tmp2 = u_hat[...,1].copy()
        # tmp3 = u_hat[...,2].copy()
        # u_hat[...,0] = k2*tmp3 - k3*tmp2
        # u_hat[...,1] = k3*tmp1 - k1*tmp3
        # u_hat[...,2] = k1*tmp2 - k2*tmp1

        # for i in range(3):
        #     psi[...,i] = fft.ifftn(1j*u_hat[...,i]).real

        return psi


#######################################################################################################
# 	Fourier Transform
#######################################################################################################
### - Only stationary covariance
### - Uses sscipy.fftpack (non the fastest solution)


class Sampling_FFT(Sampling_method_freq):
    def __init__(self, RandomField):
        super().__init__(RandomField)

    def __call__(self, noise):
        noise_hat = FourierOfGaussian(noise)
        # noise_hat = fft.ifftn(noise)
        y = self.Spectrum * noise_hat
        y = fft.fftn(y)
        return y.real[self.DomainSlice] / self.TransformNorm


#######################################################################################################
# 	Sine Transform
#######################################################################################################
### - Only stationary covariance
### - Uses sscipy.fftpack (non the fastest solution)


class Sampling_DST(Sampling_method_freq):
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
    def __init__(self, RandomField):
        super().__init__(RandomField)

    def __call__(self, noise):
        y = self.Spectrum * noise
        for j in range(self.ndim):
            y = fft.dct(y, axis=j, type=2)
        return y[self.DomainSlice] / self.TransformNorm


#######################################################################################################
# 	Non-Uniform FFT
#######################################################################################################
### - Only stationary covariance
### - Non-regular grid


class Sampling_NFFT(Sampling_method_freq):
    def __init__(self, RandomField):
        super().__init__(RandomField)

        # Prepare NFFT objects
        from pynfft.nfft import NFFT

        x = RandomField.nodes
        M = x.shape[1]
        self.nfft_obj = NFFT(self.Nd, M)  # , n=self.Nd, m=1)
        self.nfft_obj.x = (x - 0.5) / np.tile(self.L, [M, 1]).T
        self.nfft_obj.precompute()

    def __call__(self, noise):
        self.nfft_obj.f_hat = self.Spectrum * FourierOfGaussian(noise)
        y = self.nfft_obj.trafo()
        # assert (abs(y.imag) < 1.e-8).all(), np.amax(abs(y.imag))
        y = np.array(y.real, dtype=np.float) / self.TransformNorm
        return y


#######################################################################################################
# 	Vector Field Non-Uniform FFT
#######################################################################################################
### - Only stationary covariance
### - Non-regular grid


class Sampling_VF_NFFT(Sampling_method_freq):
    def __init__(self, RandomField):
        super().__init__(RandomField)

        # Prepare NFFT objects
        from pynfft.nfft import NFFT

        # import pyfftw
        x = RandomField.nodes
        M = x.shape[0]
        self.nfft_obj = NFFT(self.Nd, M, n=self.Nd, m=1)
        self.nfft_obj.x = (x - 0.5) / np.tile(self.L, [M, 1])
        self.nfft_obj.precompute()
        self.hat_noise = np.stack(
            [
                np.zeros(RandomField.ext_grid_shape, dtype="complex128")
                for _ in range(3)
            ],
            axis=-1,
        )

        # n_cpu = 1
        # shpR = RandomField.ext_grid_shape
        # shpC = shpR.copy()
        # shpC[-1] = int(shpC[-1] // 2)+1
        # axes = np.arange(self.ndim)
        # flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT', 'FFTW_UNALIGNED')
        # self.fft_x     = pyfftw.empty_aligned(shpR, dtype='float32')
        # self.fft_y 	   = pyfftw.empty_aligned(shpC, dtype='complex64')
        # self.fft_plan  = pyfftw.FFTW(self.fft_x, self.fft_y, axes=axes, direction='FFTW_FORWARD',  flags=flags, threads=n_cpu)
        # self.ifft_plan = pyfftw.FFTW(self.fft_y, self.fft_x, axes=axes, direction='FFTW_BACKWARD', flags=flags, threads=n_cpu)
        # self.hat_noise = np.stack([np.zeros(shpC, dtype='complex64') for _ in range(3)], axis=-1)
        # self.Spectrum_half = self.Spectrum[...,:shpC[-1]]

    def __call__(self, noise):
        tmp = np.zeros(self.nfft_obj.x.shape)
        for i in range(noise.shape[-1]):
            self.hat_noise[..., i] = fft.fftn(noise[..., i])
        self.hat_noise = np.einsum("kl...,...l->...k", self.Spectrum, self.hat_noise)
        # for i in range(self.ndim):
        #     self.hat_noise[...,i] = self.Spectrum[i,i,...] * self.hat_noise[...,i]
        # for i in range(noise.shape[-1]):
        # self.fft_x[:] = noise[...,i]
        # self.fft_plan()
        # self.hat_noise[...,i] = self.fft_y[:]
        # self.hat_noise  = np.einsum('kl...,...l->...k' , self.Spectrum_half, self.hat_noise)
        for i in range(self.ndim):
            self.hat_noise = np.roll(self.hat_noise, int(self.Nd[i] / 2), i)
        for i in range(noise.shape[-1]):
            # self.nfft_obj.f_hat[:] = 1.0
            self.nfft_obj.f_hat = self.hat_noise[..., i]
            y = self.nfft_obj.trafo()
            # y = fft.ifftn(self.hat_noise[...,i])
            tmp[..., i] = np.array(y.real, dtype=np.float)
        # tmp = tmp.reshape(self.field_shape)
        return tmp / self.TransformNorm


#  import pyfftw
#         n_cpu = 1
#         shpR = RandomField.ext_grid_shape
#         shpC = shpR.copy()
#         shpC[-1] = int(shpC[-1] // 2)+1
#         axes = np.arange(self.ndim)
#         flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT', 'FFTW_UNALIGNED')
#         self.fft_x     = pyfftw.empty_aligned(shpR, dtype='complex64')
#         self.fft_y 	   = pyfftw.empty_aligned(shpR, dtype='complex64')
#         self.fft_plan  = pyfftw.FFTW(self.fft_x, self.fft_y, axes=axes, direction='FFTW_FORWARD',  flags=flags, threads=n_cpu)
#         self.ifft_plan = pyfftw.FFTW(self.fft_y, self.fft_x, axes=axes, direction='FFTW_BACKWARD', flags=flags, threads=n_cpu)
#         self.Spectrum_half = self.Spectrum[...,:shpC[-1]]
#         self.hat_noise = np.stack([np.zeros(shpR, dtype='complex64') for _ in range(3)], axis=-1)
#         self.TransformNorm = self.TransformNorm / np.sqrt(self.Nd.prod())

#     def __call__(self, noise):
#         tmp = np.zeros(noise.shape)
#         # tmp2 = np.zeros(noise.shape, dtype='complex64')
#         for i in range(noise.shape[-1]):
#             self.fft_x[:] = noise[...,i]
#             self.fft_plan()
#             self.hat_noise[...,i] = self.fft_y[:]
#             # tmp2[...,i] = FourierOfGaussian(noise[...,i]) * np.sqrt(self.Nd.prod())
#         #TODO use an anisotropic TranformOfGaussian function
#         self.hat_noise  = np.einsum('kl...,...l->...k' , self.Spectrum, self.hat_noise)
#         # self.hat_noise  = np.einsum('kl...,...l->...k' , self.Spectrum_half, self.hat_noise)
#         for i in range(noise.shape[-1]):
#             self.fft_y[:] = self.hat_noise[...,i]
#             self.ifft_plan()
#             tmp[...,i] = self.fft_x[:]
#         return tmp[self.DomainSlice] / self.TransformNorm


#######################################################################################################
# 	Hierarchical matrix H2 (Fast Multipole)
#######################################################################################################
### - Non-stationnary covariances
### - Non-regular grid


class Sampling_H2(Sampling_method_base):
    def __init__(self, RandomField, lib=METHOD_H2, **kwargs):
        super().__init__(RandomField)
        L, Nd, d = self.L, self.Nd, self.ndim
        if self.verbose:
            print("\nSetting up H2-matrix...\n")

        t0 = time()

        axes = (np.arange(N).astype(np.float) / N,) * d
        position = np.meshgrid(*axes)
        position = np.vstack(list(map(np.ravel, position)))
        nvoxels = position.shape[1]

        if lib in (METHOD_H2_hlibpro,):
            import os

            os.system(
                "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/khristen/ThirdPartyCode/hlibpro-2.7.2/lib"
            )
            os.system(
                "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/khristen/ThirdPartyCode/hlibpro-2.7.2/aux/lib"
            )
            from fracturbulence.hlibpro.wrapper_Covariance import H2matrix

            self.M_H2 = H2matrix(position)

        elif lib in (METHOD_H2_h2tools, METHOD_H2):
            from h2tools import ClusterTree, Problem
            from h2tools.collections import particles
            from h2tools.mcbh import mcbh

            Covariance = RandomField.Covariance
            nu, corrlen = Covariance.nu, Covariance.corrlen
            nu_mod = (nu - d / 2) / 2
            sigma = sqrt(
                gamma(nu + d / 2)
                / gamma(nu)
                * (nu / (2 * pi)) ** (d / 2)
                / np.prod(corrlen)
            )
            sigma *= gamma(nu_mod) / gamma(nu_mod + d / 2)

            if "angle_field" in kwargs.keys():
                anis_angle = kwargs["angle_field"]
            else:
                anis_angle = Covariance.angle_anis

            from fracturbulence.Covariance.wrapper_Covariance import \
                py_Matern_block_func

            def block_func(row_data, rows, col_data, cols):
                submatrix = (
                    sigma
                    * py_Matern_block_func(
                        row_data,
                        rows,
                        col_data,
                        cols,
                        nu_mod,
                        corrlen * sqrt(nu_mod / nu),
                        anis_angle,
                        d,
                    )
                    / N ** (d / 2)
                )
                return submatrix

            data = particles.Particles(d, nvoxels, position)
            tree = ClusterTree(data, block_size=10)
            problem = Problem(block_func, tree, tree, symmetric=1, verbose=self.verbose)
            self.M_H2 = mcbh(
                problem, tau=1e-1, iters=0, verbose=self.verbose, random_init=0
            )

        print("Build H2:", time() - t0)

        ### Test
        if self.verbose:
            import matplotlib.pyplot as plt

            N = self.Nd[0]
            k = N
            r = np.arange(k) / N
            x = np.zeros(nvoxels)
            # x[0] = 1
            x[:k] = Covariance.eval_sqrt(r) / N ** (d / 2)
            t0 = time()
            Mx = self.M_H2.dot(x)
            print("dot", time() - t0)
            plt.figure()
            plt.plot(r, Mx[:k], "o-")
            plt.plot(r, Covariance.eval(r), "o--")

    def __call__(self, noise):
        z = noise.flatten()
        y = self.M_H2.dot(z)
        y = y.reshape(self.Nd)
        return y[self.DomainSlice]


#######################################################################################################
# 	ODE-basd method
#######################################################################################################
### - Non-stationnary covariances
### - Non-regular grid


class Sampling_ODE(Sampling_method_base):
    def __init__(self, RandomField, lib="h2tools", **kwargs):
        super().__init__(RandomField)
        L, Nd, d = self.L, self.Nd, self.ndim

        from fracturbulence.CovarianceKernels import set_ShapeOperator
        from fracturbulence.ODE_based.TransientPower import (Problem,
                                                             TransientPower)

        Covariance = RandomField.Covariance
        nu, corrlen = Covariance.nu, Covariance.corrlen

        if "angle_field" in kwargs.keys():
            anis_angle = kwargs["angle_field"]
        else:
            anis_angle = Covariance.angle_anis

        coef, detTheta = set_ShapeOperator(corrlen, anis_angle, ndim=self.ndim)
        coef /= 2 * nu

        t0 = time()
        self.pb = Problem(N, d, coef)
        if self.verbose:
            print("Build problem time", time() - t0)
        self.tpow = TransientPower(self.pb)
        alpha = (nu + d / 2) / 2
        eta = sqrt(
            gamma(nu + d / 2)
            / gamma(nu)
            * ((4 * pi) / (2 * nu)) ** (d / 2)
            * sqrt(detTheta)
        )
        slope = 1.0e6 * detTheta ** (1 / d) / (2 * nu)
        self.apply = (
            lambda x: self.tpow(alpha, x, nts=10, theta=4, slope=40)
            * eta
            * np.sqrt(self.Nd.prod())
        )

    def __call__(self, noise):
        y = self.apply(noise.flatten())
        y = y.reshape(self.Nd)
        return y[self.DomainSlice]


#######################################################################################################
# 	Best Rational Approximation
#######################################################################################################
### - Non-stationnary covariances
### - Non-regular grid
### - Opimal method


class Sampling_Rational(Sampling_method_base):
    def __init__(self, RandomField, **kwargs):
        super().__init__(RandomField)
        L, Nd, d = self.L, self.Nd, self.ndim

        from fracturbulence.CovarianceKernels import set_ShapeOperator
        from fracturbulence.ODE_based.TransientPower import Problem
        from fracturbulence.RationalApproximation import RationalApproximation

        Covariance = RandomField.Covariance
        nu, corrlen = Covariance.nu, Covariance.corrlen

        if "angle_field" in kwargs.keys():
            anis_angle = kwargs["angle_field"]
        else:
            anis_angle = Covariance.angle_anis

        coef, detTheta = set_ShapeOperator(corrlen, anis_angle, ndim=self.ndim)
        coef /= 2 * nu

        t0 = time()
        self.pb = Problem(N, d, coef)
        if self.verbose:
            print("Assemble problem time", time() - t0)
        alpha = (nu + d / 2) / 2
        eta = sqrt(
            gamma(nu + d / 2)
            / gamma(nu)
            * ((4 * pi) / (2 * nu)) ** (d / 2)
            * sqrt(detTheta)
        )
        self.RA = RationalApproximation(self.pb, alpha, niter=4)
        self.apply = lambda x: self.RA(x) * eta * np.sqrt(self.Nd.prod())

    def __call__(self, noise):
        y = self.apply(noise.flatten())
        y = y.reshape(self.Nd)
        return y[self.DomainSlice]


#######################################################################################################


#######################################################################################################
# 	Best Rational Approximation for Von Karman wind field with blocking                                                                 (MAIN METHOD)
#######################################################################################################
### - Non-stationary covariances
### - Optimal method


class Sampling_Rational_VK_Wind_Blocking(Sampling_method_base):
    """
    This function assumes \mcL u = -\nabla\cdot ( L(x)^2 \nabla u )
    """

    def __init__(self, RandomField, **kwargs):
        super().__init__(RandomField)

        from RandomFieldModule.utilities.fde_solve import fde_solve

        corrlen = kwargs["correlation_length"]
        eps = kwargs["viscous_dissipation_rate"]
        C = kwargs["kolmogorov_constant"]
        self.corrlen = corrlen

        self.z_grid = kwargs.get("z_grid", None)
        if self.z_grid is not None:
            self.h = np.diff(self.z_grid)
        else:
            self.h = self.L[2] / self.Nd[2] * np.ones(self.Nd[2] - 1)

        L, Nd = self.L, self.Nd
        self.Frequencies = [
            (2 * pi / L[j]) * (Nd[j] * fft.fftfreq(Nd[j])) for j in range(3)
        ]
        self.TransformNorm = np.sqrt(L[0] * L[1]) / (2 * pi)

        # NOTE: the following values are fixed for the VK model
        self.corr_len_fun = lambda z: corrlen**2  # * np.tanh(z)
        self.alpha = 17 / 12

        # self.factor = np.sqrt( C * (eps**(2/3)) * (corrlen**(17/3)) / (4*np.pi) )
        self.factor = np.sqrt(C * (eps ** (2 / 3)) / (4 * np.pi))

        # instantiate rational approximation object
        self.fde_solve = fde_solve(
            Nd[2], self.alpha, self.corr_len_fun, domain_height=L[2], z_grid=self.z_grid
        )
        # self.fde_solver = np.vectorize(self.rat_approx_inst.__call__, signature='(n),(),()->(n)')

        ### Prepare structures for Jacobian (optimization)
        # self.fde_solve_grad = fde_solve(Nd[2], 1, self.corr_len_fun, domain_height=L[2])
        # self.fde_solve_grad_aux = fde_solve(Nd[2], 1, self.corr_len_fun, domain_height=L[2])
        # self.grid = self.fde_solve_grad.ode_solve.grid[:]

        self.isParallel = kwargs.get("isParallel", False)
        if self.isParallel:
            nproc = kwargs.get("nproc", 3)
            self.nproc = nproc
            self.parallel_pool = Pool(nproc)
            self.fde_solve_parallel = [deepcopy(self.fde_solve) for i in range(nproc)]
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"

    ### z-derivative
    def Dz(self, f, adjoint=False):
        h = self.h
        dzf = np.zeros_like(f)
        if adjoint:
            dzf[..., 0] = 0
            dzf[..., 1] = f[..., 0] / h[0] - f[..., 2] / (h[1] + h[2])
            dzf[..., -2] = f[..., -3] / (h[-2] + h[-3])
            dzf[..., -1] = f[..., -2] / (h[-1] + h[-2])
            # dzf[:,:,0] = -(f[:,:,0]/h[0] + f[:,:,1]/(h[0] + h[1]))
            # dzf[:,:,1] = (f[:,:,0]/h[0] - f[:,:,2]/(h[1] + h[2]))
            # dzf[:,:,-2] = (f[:,:,-3]/(h[-2] + h[-3]) - f[:,:,-1]/ h[-1])
            # dzf[:,:,-1] = (f[:,:,-2]/(h[-1] + h[-2]) + f[:,:,-1]/ h[-1])
            dzf[..., 2:-2] = -(
                f[..., 3:-1] / (h[2:-1] + h[3:]) - f[..., 1:-3] / (h[:-3] + h[1:-2])
            )
        else:
            dzf[:, :, 2:-1] = (f[:, :, 3:] - f[:, :, 1:-2]) / (h[2:] + h[1:-1])
            dzf[:, :, 0] = f[:, :, 1] / h[0]
            dzf[:, :, 1] = f[:, :, 2] / (h[1] + h[0])
            dzf[:, :, -1] = 0
            # dzf[:,:,1:-1] = (f[:,:,2:] - f[:,:,:-2]) / (h[1:] + h[:-1])
            # dzf[:,:,1:-1] = (f[:,:,2:] - f[:,:,:-2]) / (h[1:] + h[:-1])
            # dzf[:,:,0] = (f[:,:,1] - f[:,:,0]) / h[0]
            # dzf[:,:,-1] = (f[:,:,-1] - f[:,:,-2]) / h[-1]
            # dzf[:,:,1:] = (f[:,:,1:] - f[:,:,:-1]) / h[:]
            # dzf[:,:,0] = (f[:,:,1] - f[:,:,0]) / h[0]
            # dzf[:,:,:-1] = (f[:,:,1:] - f[:,:,:-1]) / h[:]
            # dzf[:,:,-1] = (f[:,:,-1] - f[:,:,-2]) / h[-1]
        return dzf

    ### Apply curl
    def curl(self, f_hat, adjoint=False):
        k1, k2, _ = np.meshgrid(*self.Frequencies, indexing="ij")
        # k1, k2 = np.meshgrid(*self.Frequencies[:2], indexing='ij')

        tmp1 = f_hat[..., 0].copy()
        tmp2 = f_hat[..., 1].copy()
        tmp3 = f_hat[..., 2].copy()
        if not adjoint:
            dzf1 = self.Dz(f_hat[..., 0], adjoint=False)
            dzf2 = self.Dz(f_hat[..., 1], adjoint=False)
            f_hat[..., 0] = 1j * k2 * tmp3 - dzf2
            f_hat[..., 1] = dzf1 - 1j * k1 * tmp3
            f_hat[..., 2] = 1j * k1 * tmp2 - 1j * k2 * tmp1
        else:
            dzf1 = self.Dz(f_hat[..., 0], adjoint=True)
            dzf2 = self.Dz(f_hat[..., 1], adjoint=True)
            # f_hat[...,0] = 0
            # f_hat[...,1] = dzf1
            # f_hat[...,2] = 0
            f_hat[..., 0] = 1j * k2 * tmp3 + dzf2
            f_hat[..., 1] = -dzf1 - 1j * k1 * tmp3
            f_hat[..., 2] = 1j * k1 * tmp2 - 1j * k2 * tmp1

        return f_hat

    def __call__(self, noise, **kwargs):
        Robin_const = [
            np.infty,
            np.infty,
            kwargs.get("Robin_const"),
        ]  ### 3rd component is None if is not in kwargs
        adjoint = kwargs.get("adjoint", False)
        jac = kwargs.get("jac", False)
        grad_coef = kwargs.get("grad_coef")
        if grad_coef:
            jac = True
        mode = kwargs.get("mode", None)

        if not adjoint:
            f_hat = noise / np.sqrt(self.L[2] / self.Nd[2])

            ## Fourier transform of RHS in x- and y-coordinates
            f_hat = fft.fft(fft.fft(f_hat, axis=1), axis=0)

            ## Apply SqrtMass in z
            f_hat = np.apply_along_axis(self.fde_solve.apply_sqrtMass, -1, f_hat)
        else:
            # f_hat = noise[:]
            pass

        ### STEP 1

        if jac and adjoint:
            nPar = len(grad_coef) + 1
            # grad = np.zeros(list(f_hat.shape) + [nPar], dtype=f_hat.dtype)
            grad = np.zeros(list(f_hat.shape[0:2]) + [3, nPar], dtype=f_hat.dtype)

        # Apply curl
        if adjoint:
            # f_hat = self.curl(noise, adjoint=True)
            f_hat = noise

        # define frequencies in 2D domain (x & y)
        k1, k2 = np.meshgrid(*self.Frequencies[:2], indexing="ij")

        # solve rational approx problem (in z) for each k1, k2
        t0 = time()

        # func = lambda arg: self.fde_solve(*arg)
        # for l in range(3):
        #     # f_hat[:,:,:,l] = self.fde_solver(f_hat[:,:,:,l],k1,k2)
        #     iter_arg = [ (f_hat[i,j,:,l], k1[i,j], k2[i,j], Robin_const[l], adjoint, jac, grad_coef,0,l) for i in range(self.Nd[0]) for j in range(self.Nd[1]) ]
        #     t0 = time()
        #     if jac and adjoint:
        #         out = list(zip(*map(func, iter_arg)))
        #         f_hat[:,:,:,l] = np.array(out[0]).reshape(self.Nd)
        #         grad[:,:,l,:] = np.array(out[1]).reshape(list(self.Nd[0:2]) + [-1])
        #         # print('Time (with Jac): ', time()-t0)
        #     else:
        #         f_hat[:,:,:,l] = np.array(list(map(func, iter_arg))).reshape(self.Nd)
        #         # print('Time (func only): ', time()-t0)
        # # print('Time  RA:', time()-t0)

        time_start = time()
        if self.isParallel:
            ### PARALLEL
            func = lambda arg: self.fde_solve(*arg)
            for l in range(3):
                iter_arg = [
                    (
                        f_hat[i, j, :, l],
                        k1[i, j],
                        k2[i, j],
                        Robin_const[l],
                        adjoint,
                        jac,
                        grad_coef,
                        0,
                        l,
                    )
                    for i in range(self.Nd[0])
                    for j in range(self.Nd[1])
                ]
                f_hat[:, :, :, l] = np.array(
                    list(self.parallel_pool.map(func, iter_arg))
                ).reshape(self.Nd)
            print("Runtime (Parallel): ", time() - time_start)
        else:
            ### SEQUENTIAL
            rhs = np.zeros(f_hat.shape[2], dtype=f_hat.dtype)
            for l in range(3):
                for i in range(self.Nd[0]):
                    for j in range(self.Nd[1]):
                        rhs[:] = f_hat[i, j, :, l]
                        f_hat[i, j, :, l] = self.fde_solve(
                            rhs,
                            k1[i, j],
                            k2[i, j],
                            Robin_const=Robin_const[l],
                            component=l,
                            t=0,
                            adjoint=adjoint,
                            jac=jac,
                            grad_coef=grad_coef,
                            mode=mode,
                        )
            # print('Runtime (Sequential): ', time()-time_start)

        # Apply curl
        if not adjoint:
            f_hat = self.curl(f_hat, adjoint=False)

            # Inverse Fourier transform of RHS in x- and y-coordinates
            f = fft.ifft(fft.ifft(f_hat, axis=1), axis=0)
            f = self.factor * f.real[self.DomainSlice]  # / self.TransformNorm
        else:
            f = self.factor * f_hat[self.DomainSlice]
            raise Exception("Never applied")  ### we don't use it

        if jac and adjoint:
            # grad = self.factor**2 * fft.ifft(fft.ifft(grad, axis = 1), axis = 0).real
            grad_sum = np.sum(grad, axis=(0, 1, 2))
            assert np.all(np.isclose(np.imag(grad_sum), 0))
            grad_sum = 2 * self.factor**2 * grad_sum.real / self.Nd[0:2].prod()
            return f, grad_sum
        else:
            return f

    ###
    def compute_Adjoint(self, stress_comp, z, **kwargs):
        Robin_const = [
            np.infty,
            np.infty,
            kwargs.get("Robin_const"),
        ]  ### 3rd component is None if is not in kwargs
        mode = kwargs.get("mode", None)

        # define frequencies in 2D domain (x & y)
        k1, k2 = np.meshgrid(*self.Frequencies[:2], indexing="ij")

        # solve rational approx problem (in z) for each k1, k2
        f = np.zeros(list(self.Nd) + [3], dtype=np.complex)
        for l in range(3):
            for i in range(self.Nd[0]):
                for j in range(self.Nd[1]):
                    f[i, j, :, l] = self.fde_solve(
                        None,
                        k1[i, j],
                        k2[i, j],
                        Robin_const=Robin_const[l],
                        adjoint=True,
                        component=l,
                        z=z,
                        stress_comp=stress_comp,
                        mode=mode,
                    )

        # f = fft.ifft(fft.ifft(f, axis = 1), axis = 0)
        f = self.factor * f[self.DomainSlice]  # / self.TransformNorm
        return f

    def compute_FFT_Adjoint(self, noise, **kwargs):
        # if kwargs.get('Robin_const') is not np.infty:
        # raise Exception("incompatible Robin_const: {}".format(kwargs.get('Robin_const')))
        # Robin_const = [np.infty, np.infty, kwargs.get('Robin_const')] ### 3rd component is None if is not in kwargs

        ### STEP 1

        # Apply curl
        noise = fft.fftn(noise, axes=(0, 1, 2))
        # k = np.array(list(np.meshgrid(*self.Frequencies, indexing='ij')))
        # k1  = k[0,...]
        # k2  = k[1,...]
        # k3  = k[2,...]

        # tmp1 = noise[...,0].copy()
        # tmp2 = noise[...,1].copy()
        # tmp3 = noise[...,2].copy()
        # noise[...,0] = k2*tmp3 - k3*tmp2
        # noise[...,1] = k3*tmp1 - k1*tmp3
        # noise[...,2] = k1*tmp2 - k2*tmp1
        # noise = 1j*noise

        ### STEP 2

        # double domain w.r.t. z and symmetrize (for BC trick)
        noise = fft.ifft(noise, axis=2)

        shape2 = list(noise.shape)
        Nz = shape2[2]
        shape2[2] = 2 * shape2[2] - 1
        noise2 = np.zeros(shape2, dtype="complex128")
        # noise2[...,Nz-1:,:] = noise
        # noise2[...,:Nz,:] -= np.flip(noise, axis=2)
        noise2[..., :Nz, :] = noise
        # noise2[...,Nz-1:,:] -= np.flip(noise, axis=2)
        if kwargs.get("Robin_const") is np.infty:
            noise2[..., Nz - 1 :, :] -= np.flip(noise, axis=2)
        else:
            noise2[..., Nz - 1 :, 0] -= np.flip(noise[..., 0], axis=2)
            noise2[..., Nz - 1 :, 1] -= np.flip(noise[..., 1], axis=2)
            noise2[..., Nz - 1 :, 2] += np.flip(noise[..., 2], axis=2)
        noise2 /= 2

        ### STEP 3

        # correlate noise
        noise2 = fft.fft(noise2, axis=2)

        L, Nd, d = np.copy(self.L), np.copy(self.Nd), self.ndim
        Nd[2] = 2 * Nd[2] - 1
        L[2] = 2
        Frequencies2 = [
            (2 * pi / L[j]) * (Nd[j] * fft.fftfreq(Nd[j])) for j in range(d)
        ]
        k = np.array(list(np.meshgrid(*Frequencies2, indexing="ij")))
        kk = np.sum(k**2, axis=0)

        G = self.corrlen ** (17 / 3) / (1 + (self.corrlen**2) * kk) ** (17 / 12)

        noise2[..., 0] *= G
        noise2[..., 1] *= G
        noise2[..., 2] *= G

        ### STEP 4

        # restrict solution
        f = fft.ifftn(noise2, axes=(0, 1, 2)).real[..., :Nz, :]
        f = self.factor * f[self.DomainSlice]  # / self.TransformNorm
        return f

    def compute_vector_potential_nonuniform_Fourier_polar(
        self, delta, r, component=0, **kwargs
    ):
        Robin_const = kwargs.get("Robin_const", None)
        if Robin_const is np.infty:
            Robin_const = "infty"
        Robin_const = [
            "infty",
            "infty",
            Robin_const,
        ]  ### 3rd component is None if is not in kwargs
        # Robin_const = [np.infty, np.infty, kwargs.get('Robin_const', None)] ### 3rd component is None if is not in kwargs
        mode = kwargs.get("mode", None)

        if not self.isParallel:
            ### SEQUENTIAL
            time_start = time()
            psi = np.zeros((len(r), len(delta)), dtype=np.complex)
            for i in range(len(r)):
                psi[i, :] = self.fde_solve(
                    delta,
                    r[i],
                    0,
                    Robin_const=Robin_const[component],
                    component=component,
                    t=0,
                    adjoint=True,
                    jac=False,
                    grad_coef=False,
                    mode=mode,
                )
            # print('Runtime (Sequential): ', time()-time_start)

        if self.isParallel:
            ### PARALLEL
            r_split = np.array_split(r, self.nproc)
            solve_split = [deepcopy(self.fde_solve) for i in range(self.nproc)]
            time_start = time()
            func = partial(
                func_parallel_polar,
                rhs=delta,
                Robin_const=Robin_const[component],
                component=component,
                mode=mode,
            )
            psi_list = list(self.parallel_pool.map(func, zip(r_split, solve_split)))
            # psi_list = func_parallel_polar2(solve_split, r_split, rhs=delta, Robin_const=Robin_const[component], component=component)
            psi_par = np.vstack(psi_list)
            psi = psi_par
            print("Runtime (Parallel):   ", time() - time_start)

        return psi

    ### Gradient (for optimization)
    def Gradient(self, f_hat, coef1, coef2_list, nPar, Robin_const=None):
        Robin_const = [None, None, Robin_const]
        k1, k2 = np.meshgrid(*self.Frequencies[:2], indexing="ij")
        f_hat = fft.fft(fft.fft(f_hat, axis=1), axis=0)

        grad = np.zeros(list(f_hat.shape) + [nPar], dtype=f_hat.dtype)

        self.fde_solve_grad.reset_parameters(coef=coef1)

        for comp in range(nPar):
            if comp == nPar - 1:
                ### derivative wrt Robin const
                def func(args):
                    x, k1, k2, Robin_const = args
                    e1 = np.zeros_like(x)
                    e1[0] = 1
                    y = -self.fde_solve(
                        e1, k1, k2, Robin_const=Robin_const, adjoint=True
                    )
                    y *= x[0]
                    return y

            else:
                ### derivative wrt expansion parameters
                coef2 = lambda z: coef2_list(z, comp=comp)
                self.fde_solve_grad_aux.reset_parameters(coef=coef2)

                def func(args):
                    x, k1, k2, Robin_const = args
                    y = -self.fde_solve_grad_aux.ode_solve.apply_matvec(-1, x, k1, k2)
                    y = self.fde_solve_grad(
                        y, k1, k2, Robin_const=Robin_const, adjoint=True
                    )
                    y += coef2(self.grid) * x
                    y *= self.alpha / 2 * coef1(self.grid) ** (self.alpha / 2 - 1)
                    return y

            for l in range(3):
                iter_arg = [
                    (f_hat[i, j, :, l], k1[i, j], k2[i, j], Robin_const[l])
                    for i in range(self.Nd[0])
                    for j in range(self.Nd[1])
                ]
                grad[:, :, :, l, comp] = np.array(
                    list(map(func, iter_arg)), dtype=grad.dtype
                ).reshape(self.Nd)

        grad = fft.ifft(fft.ifft(grad, axis=1), axis=0).real

        return grad


#######################################################################################################
# 	Best Rational Approximation for generalized VK spectrum wind field with blocking
#######################################################################################################
### - Non-stationary covariances
### - Optimal method


class Sampling_Rational_Generalized_VK_Wind_Blocking(
    Sampling_Rational_VK_Wind_Blocking
):
    """
    Solves (1 -\nabla\cdot ( L(x)^2 \nabla ))^\alpha (-\nabla\cdot ( L(x)^2 \nabla u ))^\beta u = f
    """

    def __init__(self, RandomField, **kwargs):
        super().__init__(RandomField, **kwargs)

        from RandomFieldModule.utilities.fde_solve import fde_solve

        self.alpha = 11 / 12
        self.beta = 1 / 2
        self.c = 0.00333333
        # self.c = 1.e-2

        # instantiate rational approximation object
        L, Nd = self.L, self.Nd
        self.fde_solve = fde_solve(
            Nd[2], self.alpha, self.corr_len_fun, domain_height=L[2], beta=self.beta
        )

    def __call__(self, noise, **kwargs):
        Robin_const = [
            np.infty,
            np.infty,
            kwargs.get("Robin_const"),
        ]  ### 3rd component is None if is not in kwargs
        adjoint = kwargs.get("adjoint", False)

        ## Fourier transform of RHS in x- and y-coordinates
        f_hat = noise
        f_hat = fft.fft(fft.fft(f_hat, axis=1), axis=0)

        if not adjoint:
            ## Correlate noise
            f_hat = noise / np.sqrt(self.L[2] / self.Nd[2])
            f_hat = np.apply_along_axis(self.fde_solve.apply_sqrtMass, -1, f_hat)
            f_hat = self.smoothen_noise(f_hat)
        else:
            # Apply curl
            f_hat = self.curl(f_hat, adjoint=True)
            raise Exception("Never applied")  ### we don't use it

        # define frequencies in 2D domain (x & y)
        k1, k2 = np.meshgrid(*self.Frequencies[:2], indexing="ij")

        # solve rational approx problem (in z) for each k1, k2
        t0 = time()

        # func = lambda arg: self.fde_solve(*arg)
        # for l in range(3):
        #     iter_arg = [ (f_hat[i,j,:,l], k1[i,j], k2[i,j], Robin_const[l], adjoint) for i in range(self.Nd[0]) for j in range(self.Nd[1]) ]
        #     t0 = time()
        #     f_hat[:,:,:,l] = np.array(list(map(func, iter_arg))).reshape(self.Nd)
        # print('Time  RA:', time()-t0)

        rhs = np.zeros(f_hat.shape[2], dtype=f_hat.dtype)
        for l in range(3):
            for i in range(self.Nd[0]):
                for j in range(self.Nd[1]):
                    rhs[:] = f_hat[i, j, :, l]
                    f_hat[i, j, :, l] = self.fde_solve(
                        rhs,
                        k1[i, j],
                        k2[i, j],
                        Robin_const=Robin_const[l],
                        component=l,
                        t=0,
                        adjoint=adjoint,
                        jac=False,
                        grad_coef=False,
                    )

        if not adjoint:
            # Apply curl
            f_hat = self.curl(f_hat, adjoint=False)
            # f_hat = self.curl(f_hat, adjoint=False)
        else:
            ## Correlate noise
            f_hat = self.smoothen_noise(f_hat)
            raise Exception("Never applied")  ### we don't use it

        # Inverse Fourier transform of RHS in x- and y-coordinates
        f = fft.ifft(fft.ifft(f_hat, axis=1), axis=0)
        f = self.factor * f.real[self.DomainSlice]  # / self.TransformNorm

        return f

    def smoothen_noise(self, f_hat):
        k = np.array(list(np.meshgrid(*self.Frequencies, indexing="ij")))
        kk = np.sum(k**2, axis=0)

        f_hat = fft.fft(f_hat, axis=2)
        f_hat[..., 0] = f_hat[..., 0] * np.exp(-self.c * np.sqrt(kk))
        f_hat[..., 1] = f_hat[..., 1] * np.exp(-self.c * np.sqrt(kk))
        f_hat[..., 2] = f_hat[..., 2] * np.exp(-self.c * np.sqrt(kk))
        return fft.ifft(f_hat, axis=2)


#######################################################################################################
# 	Best Rational Approximation for rapid distortion wind field with blocking
#######################################################################################################
### - Non-stationary covariances
### - Optimal method


class Sampling_Rational_Rapid_Distortion_Wind_Blocking(
    Sampling_Rational_VK_Wind_Blocking
):
    """
    Solves (1 -\nabla\cdot ( \Theta_t \nabla ))^\alpha u = f_t
    """

    def __init__(self, RandomField, **kwargs):
        super().__init__(RandomField, **kwargs)

        from RandomFieldModule.utilities.fde_solve import fde_solve

        self.alpha = 17 / 12
        self.t = 1.0

        # instantiate rational approximation object
        L, Nd = self.L, self.Nd
        self.fde_solve = fde_solve(
            Nd[2], self.alpha, self.corr_len_fun, domain_height=L[2], t=self.t
        )

    def __call__(self, noise, **kwargs):
        Robin_const = [
            np.infty,
            np.infty,
            kwargs.get("Robin_const"),
        ]  ### 3rd component is None if is not in kwargs
        t = kwargs.get("t", 1.0)
        adjoint = kwargs.get("adjoint", False)

        ## Fourier transform of RHS in x- and y-coordinates
        f_hat = noise
        f_hat = fft.fft(fft.fft(f_hat, axis=1), axis=0)

        if not adjoint:
            ## Correlate noise
            f_hat = noise / np.sqrt(self.L[2] / self.Nd[2])
            f_hat = np.apply_along_axis(self.fde_solve.apply_sqrtMass, -1, f_hat)
            f_hat = self.distort_noise(f_hat, t)
        else:
            # Apply curl
            f_hat = self.curl(f_hat, adjoint=True)
            raise Exception("Never applied")  ### we don't use it

        # define frequencies in 2D domain (x & y)
        k1, k2 = np.meshgrid(*self.Frequencies[:2], indexing="ij")

        # solve rational approx problem (in z) for each k1, k2
        t0 = time()

        # func = lambda arg: self.fde_solve(*arg)
        # for l in range(3):
        #     iter_arg = [ (f_hat[i,j,:,l], k1[i,j], k2[i,j], Robin_const[l], adjoint, False, None, t) for i in range(self.Nd[0]) for j in range(self.Nd[1]) ]
        #     t0 = time()
        #     f_hat[:,:,:,l] = np.array(list(map(func, iter_arg))).reshape(self.Nd)
        # print('Time  RA:', time()-t0)

        rhs = np.zeros(f_hat.shape[2], dtype=f_hat.dtype)
        for l in range(3):
            for i in range(self.Nd[0]):
                for j in range(self.Nd[1]):
                    rhs[:] = f_hat[i, j, :, l]
                    f_hat[i, j, :, l] = self.fde_solve(
                        rhs,
                        k1[i, j],
                        k2[i, j],
                        Robin_const=Robin_const[l],
                        component=l,
                        t=t,
                        adjoint=adjoint,
                        jac=False,
                        grad_coef=False,
                    )

        if not adjoint:
            # Apply curl
            f_hat = self.curl(f_hat, adjoint=False)
        else:
            raise Exception("Never applied")  ### we don't use it

        # Inverse Fourier transform of RHS in x- and y-coordinates
        f = fft.ifft(fft.ifft(f_hat, axis=1), axis=0)
        f = self.factor * f.real[self.DomainSlice]  # / self.TransformNorm

        return f

    def distort_noise(self, f_hat, beta):
        f_hat = fft.fft(f_hat, axis=2)

        # L, Nd, d = self.L, self.Nd, self.ndim
        # Frequencies = [(2*pi/L[j])*(Nd[j]*fft.fftfreq(Nd[j])) for j in range(d)]

        k = np.array(list(np.meshgrid(*self.Frequencies, indexing="ij")))

        with np.errstate(divide="ignore", invalid="ignore"):
            # beta = self.Gamma * (kk * self.corrlen**2)**(-1/3) / np.sqrt( hyp2f1(1/3, 17/6, 4/3, -1/(kk*self.corrlen**2)) )
            # beta[np.where(kk==0)] = 0

            k1 = k[0, ...]
            k2 = k[1, ...]
            k3 = k[2, ...]
            k30 = k3 + beta * k1

            kk = k1**2 + k2**2 + k3**2
            kk0 = k1**2 + k2**2 + k30**2

            #### RDT

            s = k1**2 + k2**2
            C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
            tmp = beta * k1 * np.sqrt(s) / (kk0 - k30 * k1 * beta)
            C2 = k2 * kk0 / s ** (3 / 2) * np.arctan(tmp)

            zeta1_by_zeta3 = (C1 - k2 / k1 * C2) * kk / kk0
            zeta2_by_zeta3 = (k2 / k1 * C1 + C2) * kk / kk0
            one_by_zeta3 = kk / kk0

            # deal with divisions by zero (2/2)
            zeta1_by_zeta3 = np.nan_to_num(zeta1_by_zeta3)
            zeta2_by_zeta3 = np.nan_to_num(zeta2_by_zeta3)
            one_by_zeta3 = np.nan_to_num(one_by_zeta3)

            f_hat[..., 2] = (
                -f_hat[..., 0] * zeta1_by_zeta3
                - f_hat[..., 1] * zeta2_by_zeta3
                + f_hat[..., 2] * one_by_zeta3
            )

            return fft.ifft(f_hat, axis=2)

    def compute_vector_potential_nonuniform_Fourier(
        self, delta, k1, k2, component=0, **kwargs
    ):
        Robin_const = kwargs.get("Robin_const", None)
        tau = kwargs.get("tau", 0)
        if Robin_const is np.infty:
            Robin_const = "infty"
        Robin_const = [
            "infty",
            "infty",
            Robin_const,
        ]  ### 3rd component is None if is not in kwargs
        # Robin_const = [np.infty, np.infty, kwargs.get('Robin_const', None)] ### 3rd component is None if is not in kwargs
        mode = kwargs.get("mode", None)
        noFactor = kwargs.get("noFactor", False)

        if not self.isParallel:
            ### SEQUENTIAL
            # time_start = time()
            psi = np.zeros((len(k1), len(k2), len(delta)), dtype=np.complex)
            for i in range(len(k1)):
                for j in range(len(k2)):
                    psi[i, j, :] = self.fde_solve(
                        delta,
                        k1[i],
                        k2[j],
                        Robin_const=Robin_const[component],
                        component=component,
                        t=tau,
                        adjoint=True,
                        mode=mode,
                        noL2factor=noFactor,
                    )
            # print('Runtime (Sequential): ', time()-time_start)

        if self.isParallel:
            ### PARALLEL
            raise Exception("Not implemented !")

        if not noFactor:
            psi *= self.factor

        return psi

    def distorted_noise_covariance(self, k1, k2, k3, tau_func):
        with np.errstate(divide="ignore", invalid="ignore"):
            # k = np.array(list(np.meshgrid(k1, k2, k3, indexing='ij')))
            # k1  = k[0,...]
            # k2  = k[1,...]
            # k3  = k[2,...]
            k1, k2, k3 = np.meshgrid(k1, k2, k3, indexing="ij")

            if isinstance(tau_func, Callable):
                tau = tau_func(k1, k2, k3)
            else:
                tau = tau_func

            k30 = k3 + tau * k1

            kk = k1**2 + k2**2 + k3**2
            kk0 = k1**2 + k2**2 + k30**2

            s = k1**2 + k2**2
            C1 = tau * k1**2 * (kk0 - 2 * k30**2 + tau * k1 * k30) / (kk * s)
            tmp = tau * k1 * np.sqrt(s) / (kk0 - k30 * k1 * tau)
            C2 = k2 * kk0 / s ** (3 / 2) * np.arctan(tmp)

            zeta1_by_zeta3 = (C1 - k2 / k1 * C2) * kk / kk0
            zeta2_by_zeta3 = (k2 / k1 * C1 + C2) * kk / kk0
            one_by_zeta3 = kk / kk0
            term33 = ((C1**2 + C2**2) * (k1**2 + k2**2) / k1**2 + 1) * (
                kk / kk0
            ) ** 2

            zeta1_by_zeta3[np.isnan(zeta1_by_zeta3)] = 0
            zeta2_by_zeta3[np.isnan(zeta2_by_zeta3)] = 0
            one_by_zeta3[np.isnan(zeta2_by_zeta3)] = 0
            term33[np.isnan(term33)] = 0

            # Column3 = [-zeta1_by_zeta3, -zeta2_by_zeta3, term33]
            Column3 = [-zeta1_by_zeta3, -zeta2_by_zeta3, one_by_zeta3]

            return Column3

    ### z-derivative
    def Dz(self, f, adjoint=False):
        h = self.h
        dzf = np.zeros_like(f)
        if adjoint:
            dzf[..., 0] = 0
            dzf[..., 1] = f[..., 0] / h[0] - f[..., 2] / (h[1] + h[2])
            dzf[..., -2] = f[..., -3] / (h[-2] + h[-3])
            dzf[..., -1] = f[..., -2] / (h[-1] + h[-2])
            dzf[..., 2:-2] = -(
                f[..., 3:-1] / (h[2:-1] + h[3:]) - f[..., 1:-3] / (h[:-3] + h[1:-2])
            )
        else:
            dzf[:, :, 2:-1] = (f[:, :, 3:] - f[:, :, 1:-2]) / (h[2:] + h[1:-1])
            dzf[:, :, 0] = f[:, :, 1] / h[0]
            dzf[:, :, 1] = f[:, :, 2] / (h[1] + h[0])
            dzf[:, :, -1] = 0
        return dzf


#######################################################################################################
#######################################################################################################
# 	Parallel cumputations
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
