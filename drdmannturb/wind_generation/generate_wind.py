"""
This module implements the wind generation
"""


import pickle
from os import PathLike
from time import time
from typing import Union

import numpy as np
from torch.cuda import is_available

from drdmannturb.spectra_fitting import CalibrationProblem
from drdmannturb.wind_generation.covariance_kernels import (
    MannCovariance,
    VonKarmanCovariance,
)
from drdmannturb.wind_generation.gaussian_random_fields import *
from drdmannturb.wind_generation.nn_covariance import NNCovariance


class GenerateFluctuationField:
    r"""
    Class for generating a fluctuation field either from a Mann model or a pre-fit DRD model that generates the field spectra.

    Turbulent fluctuations can be formally written as a convolution of a covariance kernel with Gaussian noise :math:`\boldsymbol{\xi}` in the physical domain:

    .. math::
        \mathbf{u}=\mathcal{F}^{-1} \mathcal{G} \widehat{\boldsymbol{\xi}}=\mathcal{F}^{-1} \mathcal{G} \mathcal{F} \boldsymbol{\xi},

    where :math:`\mathcal{F}` is the Fourier transform and the operator :math:`\mathcal{G}` is the point-wise multiplication by :math:G(\boldsymbol{k})`, which is any positive-definite "square-root" of the spectral tensor and satisfies :math:`G(\boldsymbol{k}) G^*(\boldsymbol{k})=\Phi(\boldsymbol{k})`.

    This is determined by which :math:`\Phi(\boldsymbol{k}, \tau(\boldsymbol{k}))` is used. The following are provided:

    #. Mann, which utilizes the Mann eddy lifetime function

        .. math::
            \tau^{\mathrm{IEC}}(k)=\frac{T B^{-1}(k L)^{-\frac{2}{3}}}{\sqrt{{ }_2 F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-(k L)^{-2}\right)}}

    and the full spectral tensor can be found in the following reference:
        J. Mann, “The spatial structure of neutral atmospheric surfacelayer turbulence,” Journal of fluid mechanics 273, 141-168 (1994)

    #. DRD model, which utilizes a learned eddy lifetime function and requires a pre-trained DRD model. The eddy lifetime function is given by

        .. math::
            \tau(\boldsymbol{k})=\frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}{\left(1+|\boldsymbol{a}|^2\right)^{\nu / 2}}, \quad \boldsymbol{a}=\boldsymbol{a}(\boldsymbol{k})

    #. Von Karman model,

        .. math::
            \Phi_{i j}^{\mathrm{VK}}(\boldsymbol{k})=\frac{E(k)}{4 \pi k^2}\left(\delta_{i j}-\frac{k_i k_j}{k^2}\right)

        which utilizes the energy spectrum function

        .. math::
            E(k)=c_0^2 \varepsilon^{2 / 3} k^{-5 / 3}\left(\frac{k L}{\left(1+(k L)^2\right)^{1 / 2}}\right)^{17 / 3},

        where :math:`\varepsilon` is the viscous dissipation of the turbulent kinetic energy, :math:`L` is the length scale parameter and :math:`c_0^2 \approx 1.7` is an empirical constant.
    """

    def __init__(
        self,
        friction_velocity: float,
        reference_height: float,
        grid_dimensions,
        grid_levels,
        model: str,
        path_to_parameters: Union[str, PathLike],
        seed: int = None,
        blend_num=10,
    ):
        r"""
        Parameters
        ----------
        friction_velocity : float
            The reference wind friction velocity :math:`u_*`
        reference_height : float
            Reference height :math:`L`
        grid_dimensions : np.ndarray
            Numpy array denoting the grid size
        grid_levels : np.ndarray
            Numpy array denoting the grid levels
        model : str
            One of ``"NN"``, ``"VK"``, or ``"Mann"`` denoting
            "Neural Network," "Von Karman," and "Mann model".
        path_to_parameters : Union[str, PathLike]
            File path (string or ``Pathlib.Path()``)
        seed : int, optional
            Pseudo-random number generator seed, by default None. See ``np.random.RandomState``.
        blend_num : int, optional
           Number of grid points in the y-z plane to use as buffer regions between successive blocks of fluctuation; see figures 7 and 8 of the original DRD paper, by default 10. Note that at the boundary of each block, points are often correlated, so if the resulting field has undesirably high correlation, increasing this number may mitigate some of these effects.

        Raises
        ------
        ValueError
            If ``model`` doesn't match one of the 3 available models: NN, VK and Mann.
        """

        if model not in ["NN", "VK", "Mann"]:
            raise ValueError(
                "Provided model type not supported, must be one of NN, VK, Mann"
            )

        if model == "NN":
            with open(path_to_parameters, "rb") as file:
                (
                    nn_params,
                    prob_params,
                    loss_params,
                    phys_params,
                    model_params,
                ) = pickle.load(file)

            device = "cuda" if is_available() else "cpu"

            pb = CalibrationProblem(
                nn_params=nn_params,
                prob_params=prob_params,
                loss_params=loss_params,
                phys_params=phys_params,
                device="cuda",
            )
            pb.parameters = model_params
            L, T, M = pb.OPS.exp_scales()

            M = (4 * np.pi) * L ** (-5 / 3) * M
            print("Scales: ", [L, T, M])
            E0 = M * friction_velocity**2 * reference_height ** (-2 / 3)
            L = L * reference_height
            Gamma = T
        else:
            E0 = 3.2 * friction_velocity**2 * reference_height ** (-2 / 3)
            L = 0.59 * reference_height
            # L = 95 # why should the length scale depend on the reference height???????
            Gamma = 3.9

        # define margins and buffer
        time_buffer = 3 * Gamma * L
        spatial_margin = 1 * L

        try:
            grid_levels = [grid_levels[i].GetInt() for i in range(3)]
        except:
            pass
        Nx = 2 ** grid_levels[0] + 1
        Ny = 2 ** grid_levels[1] + 1
        Nz = 2 ** grid_levels[2] + 1
        hx = grid_dimensions[0] / Nx
        hy = grid_dimensions[1] / Ny
        hz = grid_dimensions[2] / Nz

        n_buffer = ceil(time_buffer / hx)
        n_marginy = ceil(spatial_margin / hy)
        n_marginz = ceil(spatial_margin / hz)

        wind_shape = [0] + [Ny] + [Nz] + [3]
        if blend_num > 0:
            noise_shape = (
                [Nx + 2 * n_buffer + (blend_num - 1)]
                + [Ny + 2 * n_marginy]
                + [Nz + 2 * n_marginz]
                + [3]
            )
        else:
            noise_shape = (
                [Nx + 2 * n_buffer] + [Ny + 2 * n_marginy] + [Nz + 2 * n_marginz] + [3]
            )
        new_part_shape = [Nx] + [Ny + 2 * n_marginy] + [Nz + 2 * n_marginz] + [3]

        central_part = [
            slice(None, None),
        ] * 4
        new_part = central_part.copy()
        central_part[0] = slice(n_buffer, -n_buffer)
        central_part[1] = slice(n_marginy, -n_marginy)
        central_part[2] = slice(0, -2 * n_marginz)
        if blend_num > 0:
            new_part[0] = slice(2 * n_buffer + (blend_num - 1), None)
        else:
            new_part[0] = slice(2 * n_buffer, None)

        self.new_part = new_part
        self.Nx = Nx
        self.blend_num = blend_num
        self.central_part = central_part
        self.new_part_shape = new_part_shape
        self.noise_shape = noise_shape
        self.n_buffer = n_buffer
        self.n_marginy = n_marginy
        self.n_marginz = n_marginz
        self.seed = seed
        self.noise = None
        self.total_wind = np.zeros(wind_shape)

        ### Random field object

        if model == "VK":
            self.Covariance = VonKarmanCovariance(ndim=3, length_scale=L, E0=E0)
            self.RF = VectorGaussianRandomField(
                # **kwargs,
                ndim=3,
                grid_level=grid_levels,
                grid_dimensions=grid_dimensions,
                sampling_method="vf_fftw",
                grid_shape=self.noise_shape[:-1],
                Covariance=self.Covariance,
                # laplace=True
            )
        elif model == "Mann":
            self.Covariance = MannCovariance(ndim=3, length_scale=L, E0=E0, Gamma=Gamma)
            self.RF = VectorGaussianRandomField(
                # **kwargs,
                ndim=3,
                grid_level=grid_levels,
                grid_dimensions=grid_dimensions,
                sampling_method="vf_fftw",
                grid_shape=self.noise_shape[:-1],
                Covariance=self.Covariance,
                # laplace=True
            )
        elif model == "NN":
            self.Covariance = NNCovariance(
                ndim=3,
                length_scale=L,
                E0=E0,
                Gamma=Gamma,
                OnePointSpectra=pb.OPS,
                h_ref=reference_height,
            )
            self.RF = VectorGaussianRandomField(
                # **kwargs,
                ndim=3,
                grid_level=grid_levels,
                grid_dimensions=grid_dimensions,
                sampling_method="vf_fftw",
                grid_shape=self.noise_shape[:-1],
                Covariance=self.Covariance,
                # laplace=True
            )

        self.RF.reseed(self.seed)

    def __call__(self):
        noise_shape = self.noise_shape
        central_part = self.central_part
        new_part = self.new_part
        new_part_shape = self.new_part_shape
        Nx = self.Nx

        ### update noise
        if self.noise is None:
            noise = self.RF.sample_noise(noise_shape)
        else:
            noise = np.roll(self.noise, -Nx, axis=0)
            noise[tuple(new_part)] = self.RF.sample_noise(new_part_shape)
        self.noise = noise

        wind_block = self.RF.sample(noise)
        wind = wind_block[tuple(central_part)]
        if self.blend_num > 0:
            self.blend_region = wind[-self.blend_num :, ...].copy()
        else:
            self.blend_region = None
        if self.blend_num > 1:
            wind = wind[: -(self.blend_num - 1), ...]

        self.total_wind = np.concatenate((self.total_wind, wind), axis=0)

        return wind
