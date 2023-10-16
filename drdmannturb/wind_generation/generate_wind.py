import pickle
from time import time

import numpy as np

from drdmannturb.spectra_fitting.calibration import CalibrationProblem

from drdmannturb.wind_generation.covariance_kernels import MannCovariance, VonKarmanCovariance
from drdmannturb.wind_generation.gaussian_random_fields import *
from drdmannturb.wind_generation.nn_covariance import NNCovariance


class GenerateWind:
    """
    Wind generation class
    """

    def __init__(
        self,
        friction_velocity,
        reference_height,
        grid_dimensions,
        grid_levels,
        seed=None,
        blend_num=10,
        **kwargs
    ):
        model = kwargs.get("model", "Mann")  ### 'FPDE_RDT', 'Mann', 'VK', 'NN'

        # # Parameters taken from pg 13 of M. Andre's dissertation
        # # model = 'FPDE_RDT'
        # model = 'Mann'
        # # model = 'VK'
        # model = 'NN'

        if model == "NN":
            path_to_parameters = kwargs.get("path_to_parameters", None)
            with open(path_to_parameters, "rb") as file:
                (
                    nn_params,
                    prob_params,
                    loss_params,
                    phys_params,
                    model_params,
                    loss_history_total,
                    loss_history_epochs,
                ) = pickle.load(file)

            pb = CalibrationProblem(
                nn_params=nn_params,
                prob_params=prob_params,
                loss_params=loss_params,
                phys_params=phys_params,
                device="cuda",
            )
            pb.parameters = model_params
            L, T, M = pb.OPS.exp_scales()

            # L, T, M = pb.OPS.update_scales()
            M = (4 * np.pi) * L ** (-5 / 3) * M
            # print("Scales: ", [L, T, M])
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
                **kwargs,
                ndim=3,
                grid_level=grid_levels,
                grid_dimensions=grid_dimensions,
                sampling_method="vf_fftw",
                grid_shape=self.noise_shape[:-1],
                Covariance=self.Covariance
            )
        elif model == "Mann":
            self.Covariance = MannCovariance(ndim=3, length_scale=L, E0=E0, Gamma=Gamma)
            self.RF = VectorGaussianRandomField(
                **kwargs,
                ndim=3,
                grid_level=grid_levels,
                grid_dimensions=grid_dimensions,
                sampling_method="vf_fftw",
                grid_shape=self.noise_shape[:-1],
                Covariance=self.Covariance
            )
        elif model == "FPDE_RDT":
            self.Covariance = None
            kwargs = {"correlation_length": L, "E0": E0}
            self.RF = VectorGaussianRandomField(
                **kwargs,
                ndim=3,
                grid_level=grid_levels,
                grid_dimensions=grid_dimensions,
                sampling_method="vf_rat_halfspace_rapid_distortion",
                grid_shape=self.noise_shape[:-1],
                Covariance=self.Covariance
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
                **kwargs,
                ndim=3,
                grid_level=grid_levels,
                grid_dimensions=grid_dimensions,
                sampling_method="vf_fftw",
                grid_shape=self.noise_shape[:-1],
                Covariance=self.Covariance
            )

        self.RF.reseed(self.seed)
        # self.RS = np.random.RandomState(seed=self.seed)

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

        t = time()
        wind_block = self.RF.sample(noise)
        # print("block computation:", time() - t)
        wind = wind_block[tuple(central_part)]
        if self.blend_num > 0:
            self.blend_region = wind[-self.blend_num :, ...].copy()
        else:
            self.blend_region = None
        if self.blend_num > 1:
            wind = wind[: -(self.blend_num - 1), ...]

        # NOTE: COMMENT THIS LINE TO SAVE MEMORY  -- THAT'S FUNNY )
        self.total_wind = np.concatenate((self.total_wind, wind), axis=0)

        return wind