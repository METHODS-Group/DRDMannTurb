"""Generates a uniform shear wind field."""

import pickle
from pathlib import Path
from time import time

import numpy as np
from pyevtk.hl import imageToVTK

import drdmannturb.fluctuation_generation.gaussian_random_fields as grf
from drdmannturb.fluctuation_generation.covariance_kernels import (
    MannCovariance,
    VonKarmanCovariance,
)
from drdmannturb.fluctuation_generation.nn_covariance import NNCovariance
from drdmannturb.spectra_fitting.calibration import CalibrationProblem

resdir = Path(__file__).parent / "data"


class GenerateWind:
    """
    Generate a uniform shear wind field.

    This class generates a uniform shear wind field by sampling a
    random field and then applying a mean profile to it.
    """

    def __init__(
        self,
        friction_velocity,
        reference_height,
        grid_dimensions,
        grid_levels,
        seed=None,
        blend_num=10,
        **kwargs,
    ):
        model = kwargs.get("model", "NN")  ### 'FPDE_RDT', 'Mann', 'VK', 'NN'
        print(model)

        if model == "NN":
            path_to_parameters = kwargs.get("path_to_parameters", None)
            with open(path_to_parameters, "rb") as file:
                config, parameters, Data, _, loss_history = pickle.load(file)

            pb = CalibrationProblem(**config)
            pb.parameters = parameters
            L, T, M = pb.OPS.update_scales()
            M = L ** (-5 / 3) * M
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

        grid_levels = [int(grid_levels[i]) for i in range(3)]

        Nx = 2 ** grid_levels[0] + 1
        Ny = 2 ** grid_levels[1] + 1
        Nz = 2 ** grid_levels[2] + 1
        hx = grid_dimensions[0] / Nx
        hy = grid_dimensions[1] / Ny
        hz = grid_dimensions[2] / Nz

        n_buffer = int(np.ceil(time_buffer / hx))
        n_marginy = int(np.ceil(spatial_margin / hy))
        n_marginz = int(np.ceil(spatial_margin / hz))

        wind_shape = [0] + [Ny] + [Nz] + [3]
        if blend_num > 0:
            noise_shape = [Nx + 2 * n_buffer + (blend_num - 1)] + [Ny + 2 * n_marginy] + [Nz + 2 * n_marginz] + [3]
        else:
            noise_shape = [Nx + 2 * n_buffer] + [Ny + 2 * n_marginy] + [Nz + 2 * n_marginz] + [3]
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
        print(model)
        if model == "VK":
            self.Covariance = VonKarmanCovariance(ndim=3, length_scale=L, E0=E0)
            self.RF = grf.VectorGaussianRandomField(
                **kwargs,
                ndim=3,
                grid_level=grid_levels,
                grid_dimensions=grid_dimensions,
                sampling_method="vf_fftw",
                grid_shape=self.noise_shape[:-1],
                Covariance=self.Covariance,
            )
        elif model == "Mann":
            self.Covariance = MannCovariance(ndim=3, length_scale=L, E0=E0, Gamma=Gamma)
            self.RF = grf.VectorGaussianRandomField(
                **kwargs,
                ndim=3,
                grid_level=grid_levels,
                grid_dimensions=grid_dimensions,
                sampling_method="vf_fftw",
                grid_shape=self.noise_shape[:-1],
                Covariance=self.Covariance,
            )
        elif model == "FPDE_RDT":
            self.Covariance = None
            kwargs = {"correlation_length": L, "E0": E0}
            self.RF = grf.VectorGaussianRandomField(
                **kwargs,
                ndim=3,
                grid_level=grid_levels,
                grid_dimensions=grid_dimensions,
                sampling_method="vf_rat_halfspace_rapid_distortion",
                grid_shape=self.noise_shape[:-1],
                Covariance=self.Covariance,
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
            self.RF = grf.VectorGaussianRandomField(
                **kwargs,
                ndim=3,
                grid_level=grid_levels,
                grid_dimensions=grid_dimensions,
                sampling_method="vf_fftw",
                grid_shape=self.noise_shape[:-1],
                Covariance=self.Covariance,
            )

        self.RF.reseed(self.seed)
        # self.RS = np.random.RandomState(seed=self.seed)

    def __call__(self):
        """
        Generate a uniform shear wind field.

        This method generates a uniform shear wind field by sampling a
        random field and then applying a mean profile to it.
        """
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
        print("block computation:", time() - t)
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


############################################################################
############################################################################

if __name__ == "__main__":
    normalize = False
    friction_velocity = 0.46
    reference_height = 100.0
    roughness_height = 0.01
    grid_dimensions = np.array([1000.0, 1000, 1000])
    grid_levels = np.array([5, 5, 5])
    seed = None  # 9000

    # path_to_parameters = '../data2/tauNet_Custom.pkl'
    path_to_parameters = resdir / "resultstauNet_Custom.pkl"

    wind = GenerateWind(
        friction_velocity,
        reference_height,
        grid_dimensions,
        grid_levels,
        seed,
        model="NN",
        path_to_parameters=path_to_parameters,
    )
    # wind = GenerateWind(friction_velocity, reference_height, grid_dimensions, grid_levels,
    # seed, model='NN', path_to_parameters=path_to_parameters)
    # wind = GenerateWind(friction_velocity, reference_height, grid_dimensions, grid_levels,
    # seed, model='Mann', path_to_parameters=path_to_parameters)
    for _ in range(4):
        wind()
    wind_field = wind.total_wind

    if normalize:
        # h = np.array(grid_dimensions/wind_field.shape[0:-1])
        # h = np.array(1/wind_field.shape[0],1/wind_field.shape[1],1/wind_field.shape[2])
        sd = np.sqrt(np.mean(wind_field**2))
        wind_field = wind_field / sd
        wind_field *= 4.26  # rescale to match Mann model

    # plt.imshow(wind_field[:,0,:,0])
    # plt.show()

    # # total_wind = wind.total_wind
    # # plt.imshow(total_wind[:,0,:,0])
    # # plt.show()

    def JCSS_law(z, z_0, delta, u_ast):
        """Calculate wind speed using JCSS law.

        Parameters
        ----------
        z : float or array-like
            Height above ground
        z_0 : float
            Roughness height
        delta : float
            Boundary layer thickness
        u_ast : float
            Friction velocity

        Returns
        -------
        float or array-like
            Wind speed at height z
        """
        return (
            u_ast
            / 0.41
            * (
                np.log(z / z_0 + 1.0)
                + 5.57 * z / delta
                - 1.87 * (z / delta) ** 2
                - 1.33 * (z / delta) ** 3
                + 0.25 * (z / delta) ** 4
            )
        )

    def log_law(z, z_0, u_ast):
        """Calculate wind speed using logarithmic law.

        Parameters
        ----------
        z : float or array-like
            Height above ground
        z_0 : float
            Roughness height
        u_ast : float
            Friction velocity

        Returns
        -------
        float or array-like
            Wind speed at height z
        """
        return u_ast * np.log(z / z_0 + 1.0) / 0.41

    z = np.linspace(0.0, grid_dimensions[2], 2 ** (grid_levels[2]) + 1)
    # mean_profile_z = JCSS_law(z, roughness_height, 10.0, friction_velocity)
    mean_profile_z = log_law(z, roughness_height, friction_velocity)

    mean_profile = np.zeros_like(wind_field)
    mean_profile[..., 0] = np.tile(mean_profile_z.T, (mean_profile.shape[0], mean_profile.shape[1], 1))

    # wind_field = mean_profile
    wind_field += mean_profile

    # wind_field *= 40/63

    ###################
    ## Export to vtk
    print("=" * 30)
    FileName = str(resdir / "OntheFlyWindField")
    print("SAVING ON THE FLY WIND FIELDS VTK TO " + f"{FileName}")

    # FileName = '../data2/WindField/OntheFlyWindField'
    spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))

    wind_field_vtk = tuple([np.copy(wind_field[..., i], order="C") for i in range(3)])

    cellData = {"grid": np.zeros_like(wind_field[..., 0]), "wind": wind_field_vtk}
    imageToVTK(FileName, cellData=cellData, spacing=spacing)
