"""Generates an isotropic turbulence field."""

from time import time

import numpy as np
from pyevtk.hl import imageToVTK

import drdmannturb.fluctuation_generation.gaussian_random_fields as grf

# from drdmannturb.spectra_fitting import CalibrationProblem, OnePointSpectra
from drdmannturb.fluctuation_generation.covariance_kernels import (
    MannCovariance,
    VonKarmanCovariance,
)


class GenerateWindTurbulence:
    """
    Generate an isotropic turbulence field.

    This class generates an isotropic turbulence field by sampling a
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
        model = kwargs.get("model", "VK")
        print(model)

        E0 = 3.2 * friction_velocity**2 * reference_height ** (-2 / 3)
        L = 0.59 * reference_height
        Gamma = 0.0

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

        self.RF.reseed(self.seed)
        # self.RS = np.random.RandomState(seed=self.seed)

    def __call__(self):
        """
        Generate a wind field.

        This method generates a wind field by sampling a random field
        and then applying a mean profile to it.
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
    friction_velocity = 0.45
    reference_height = 100.0
    roughness_height = 0.01
    grid_dimensions = np.array([1000.0, 1000, 1000])
    grid_levels = np.array([5, 5, 5])
    seed = None  # 9000

    wind_turbulence = GenerateWindTurbulence(
        friction_velocity,
        reference_height,
        grid_dimensions,
        grid_levels,
        seed,
        model="Mann",
    )

    for _ in range(4):
        wind_turbulence()
    wind_field = wind_turbulence.total_wind

    if normalize:
        sd = np.sqrt(np.mean(wind_field**2))
        wind_field = wind_field / sd
        wind_field *= 4.26  # rescale to match Mann model

    def JCSS_law(z, z_0, delta, u_ast):
        """
        Calculate wind speed profile using JCSS (Joint Committee on Structural Safety) law.

        Parameters
        ----------
        z: height above ground
        z_0: roughness height
        delta: boundary layer height
        u_ast: friction velocity

        Returns
        -------
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
        """
        Calculate wind speed profile using logarithmic law.

        Parameters
        ----------
        z: height above ground
        z_0: roughness height
        u_ast: friction velocity

        Returns
        -------
        Wind speed at height z
        """
        return u_ast * np.log(z / z_0 + 1.0) / 0.41

    z = np.linspace(0.0, grid_dimensions[2], 2 ** (grid_levels[2]) + 1)
    # mean_profile_z = JCSS_law(z, roughness_height, 10.0, friction_velocity)
    mean_profile_z = log_law(z, roughness_height, friction_velocity)

    mean_profile = np.zeros_like(wind_field)
    mean_profile[..., 0] = np.tile(mean_profile_z.T, (mean_profile.shape[0], mean_profile.shape[1], 1))

    print(mean_profile)
    # wind_field = mean_profile
    wind_field += mean_profile

    # wind_field *= 40/63

    ###################
    ## Export to vtk
    print("=" * 30)
    FileName = str("WindField")
    print("SAVING ON THE FLY WIND FIELDS VTK TO " + f"{FileName}")

    # FileName = '../data2/WindField/OntheFlyWindField'
    spacing = tuple(grid_dimensions / (2.0**grid_levels + 1))

    wind_field_vtk = tuple([np.copy(wind_field[..., i], order="C") for i in range(3)])

    cellData = {"grid": np.zeros_like(wind_field[..., 0]), "wind": wind_field_vtk}
    imageToVTK(FileName, cellData=cellData, spacing=spacing)
