"""
This module implements the wind generation functionality forward facing API
"""

import pickle
from math import ceil
from os import PathLike
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch

from ..common import CPU_Unpickler
from ..spectra_fitting import CalibrationProblem
from .covariance_kernels import MannCovariance, VonKarmanCovariance
from .gaussian_random_fields import VectorGaussianRandomField
from .nn_covariance import NNCovariance


class FluctuationFieldGenerator:
    r"""
    .. _generate-fluctuation-field-reference:
    Class for generating a fluctuation field either from a Mann model or a pre-fit DRD model that generates
    the field spectra.

    Turbulent fluctuations can be formally written as a convolution of a covariance kernel with Gaussian noise
    :math:`\boldsymbol{\xi}` in the physical domain:

    .. math::
        \mathbf{u}=\mathcal{F}^{-1} \mathcal{G} \widehat{\boldsymbol{\xi}}=\mathcal{F}^{-1} \mathcal{G}
        \mathcal{F} \boldsymbol{\xi},

    where :math:`\mathcal{F}` is the Fourier transform and the operator :math:`\mathcal{G}` is the point-wise
    multiplication by :math:`G(\boldsymbol{k})`, which is any positive-definite "square-root" of the spectral
    tensor and satisfies :math:`G(\boldsymbol{k}) G^*(\boldsymbol{k})=\Phi(\boldsymbol{k})`.

    This is determined by which :math:`\Phi(\boldsymbol{k}, \tau(\boldsymbol{k}))` is used.
    The following are provided:

    #. Mann, which utilizes the Mann eddy lifetime function

        .. math::
            \tau^{\mathrm{IEC}}(k)=\frac{T B^{-1}(k L)^{-\frac{2}{3}}}
            {\sqrt{{ }_2 F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-(k L)^{-2}\right)}}

    and the full spectral tensor can be found in the following reference:
        J. Mann, "The spatial structure of neutral atmospheric surfacelayer turbulence,"
        Journal of Fluid Mechanics 273, 141-168 (1994)

    #. DRD model, which utilizes a learned eddy lifetime function and requires a pre-trained DRD model.
       The eddy lifetime function is given by

        .. math::
            \tau(\boldsymbol{k})=\frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}
            {\left(1+|\boldsymbol{a}|^2\right)^{\nu / 2}}, \quad \boldsymbol{a}=\boldsymbol{a}(\boldsymbol{k})

    #. Von Karman model,

        .. math::
            \Phi_{i j}^{\mathrm{VK}}(\boldsymbol{k})=\frac{E(k)}{4 \pi k^2}\left(\delta_{i j}-\frac{k_i k_j}{k^2}\right)

        which utilizes the energy spectrum function

        .. math::
            E(k)=c_0^2 \varepsilon^{2 / 3} k^{-5 / 3}\left(\frac{k L}{\left(1+(k L)^2\right)^{1 / 2}}\right)^{17 / 3},

        where :math:`\varepsilon` is the viscous dissipation of the turbulent kinetic energy, :math:`L` is the length
        scale parameter and :math:`c_0^2 \approx 1.7` is an empirical constant.
    """

    def __init__(
        self,
        friction_velocity: float,
        reference_height: float,
        grid_dimensions: np.ndarray,
        grid_levels: np.ndarray,
        model: str,
        length_scale: Optional[float] = None,
        time_scale: Optional[float] = None,
        energy_spectrum_scale: Optional[float] = None,
        path_to_parameters: Optional[Union[str, PathLike]] = None,
        lowfreq_params: Optional[dict] = None,
        seed: Optional[int] = None,
        blend_num=10,
    ):
        r"""
        Parameters
        ----------
        friction_velocity : float
            The reference wind friction velocity :math:`u_*`
        reference_height : float
            Reference height :math:`z_{\text{ref}}`
        grid_dimensions : np.ndarray
            Numpy array denoting the grid size; the real dimensions of the domain of interest.
        grid_levels : np.ndarray
            Numpy array denoting the grid levels; number of discretization points used in each dimension, which
            evaluates as 2^k for each dimension for FFT-based sampling methods.
        model : str
            One of ``"DRD"``, ``"VK"``, or ``"Mann"`` denoting
            "Neural Network," "Von Karman," and "Mann model".
        length_scale : Optional[float]
            The length scale :math:`L:`, used only if non-DRD model is used. By default, None.
        time_scale : Optional[float]
            The time scale :math:`T`, used only if non-DRD model is used. By default, None.
        energy_spectrum_scale : Optional[float]
            Scaling of energy spectrum, used only if non-DRD model is used. By default, None.
        path_to_parameters : Union[str, PathLike]
            File path (string or ``Pathlib.Path()``)
        seed : int, optional
            Pseudo-random number generator seed, by default None. See ``np.random.RandomState``.
        blend_num : int, optional
           Number of grid points in the y-z plane to use as buffer regions between successive blocks of fluctuation;
           see figures 7 and 8 of the original DRD paper, by default 10. Note that at the boundary of each block,
           points are often correlated, so if the resulting field has undesirably high correlation, increasing this
           number may mitigate some of these effects.

        Raises
        ------
        ValueError
            If ``model`` doesn't match one of the 3 available models: DRD, VK and Mann.
        """

        # Validate model type and required parameters
        if model not in ["DRD", "VK", "Mann"]:
            raise ValueError("Model must be one of: DRD, VK, Mann")

        if model == "DRD":
            if path_to_parameters is None:
                raise ValueError("DRD model requires path to pre-trained parameters")

            # Load DRD model parameters
            device = "cuda" if torch.cuda.is_available() else "cpu"
            with open(path_to_parameters, "rb") as file:
                params = pickle.load(file) if device == "cpu" else CPU_Unpickler(file).load()
            nn_params, prob_params, loss_params, phys_params, model_params = params

            # Initialize calibration problem and get scales
            pb = CalibrationProblem(
                nn_params=nn_params,
                prob_params=prob_params,
                loss_params=loss_params,
                phys_params=phys_params,
                device=device,
            )
            pb.parameters = model_params
            L, T, M = pb.OPS.exp_scales()

            # Calculate final parameters
            M = (4 * np.pi) * L ** (-5 / 3) * M
            print("Scales: ", [L, T, M])
            E0 = M * friction_velocity**2 * reference_height ** (-2 / 3)
            L *= reference_height
            Gamma = T

        else:  # VK or Mann model case
            assert length_scale is not None, "VK/Mann models require length scale"  # for type checker
            assert time_scale is not None, "VK/Mann models require time scale"
            assert energy_spectrum_scale is not None, "VK/Mann models require energy spectrum scale"

            E0 = energy_spectrum_scale * friction_velocity**2 * reference_height ** (-2 / 3)
            L = length_scale
            Gamma = time_scale

        # define margins and buffer
        time_buffer = 3 * Gamma * L
        spatial_margin = 1 * L  # TODO: where is this used?

        # TODO: this should not be needed
        # grid_levels = [level.GetInt() for level in grid_levels]

        # Expand on grid_levels parameter to get grid node counts in each direction
        grid_node_counts = 2**grid_levels + 1
        Nx, Ny, Nz = grid_node_counts

        # Use grid_dimensions and grid_node_counts to get grid spacings in each direction
        grid_spacings = [dim / size for dim, size in zip(grid_dimensions, grid_node_counts)]
        hx, hy, hz = grid_spacings

        # Calculate buffer and margin sizes
        n_buffer = ceil(time_buffer / hx)
        n_margins = [
            ceil(spatial_margin / hy),  # y margin
            ceil(spatial_margin / hz),  # z margin
        ]

        # Calculate shapes
        wind_shape = [0, Ny, Nz, 3]

        buffer_extension = 2 * n_buffer + (blend_num - 1 if blend_num > 0 else 0)
        margin_extension = [2 * margin for margin in n_margins]

        noise_shape = [
            Nx + buffer_extension,
            Ny + margin_extension[0],
            Nz + margin_extension[1],
            3,
        ]

        new_part_shape = [Nx] + [size + margin_ext for size, margin_ext in zip([Ny, Nz], margin_extension)] + [3]

        # Calculate slices for different parts
        def make_slice(start=None, end=None):
            return slice(start, end)

        central_part = [make_slice()] * 4  # Initially all are full slices
        central_part[0] = make_slice(n_buffer, -n_buffer)
        central_part[1] = make_slice(n_margins[0], -n_margins[0])
        central_part[2] = make_slice(0, -2 * n_margins[1])

        new_part = [make_slice()] * 4
        if blend_num > 0:
            new_part[0] = make_slice(2 * n_buffer + (blend_num - 1), None)
        else:
            new_part[0] = make_slice(2 * n_buffer, None)

        self.grid_dimensions = grid_dimensions
        self.grid_levels = grid_levels

        self.new_part = new_part
        self.Nx = Nx
        self.blend_num = blend_num
        self.central_part = central_part
        self.new_part_shape = new_part_shape
        self.noise_shape = noise_shape
        self.n_buffer = n_buffer
        self.n_marginy = n_margins[0]
        self.n_marginz = n_margins[1]
        self.seed = seed
        self.noise = None
        self.total_fluctuation = np.zeros(wind_shape)

        CovarianceType = Union[
            type[VonKarmanCovariance],
            type[MannCovariance],
            Callable[..., NNCovariance],
        ]

        ### Random field object
        covariance_map: dict[str, CovarianceType] = {  # type: ignore
            "VK": VonKarmanCovariance,
            "Mann": MannCovariance,
            "DRD": lambda **kwargs: NNCovariance(**kwargs, ops=pb.OPS, h_ref=reference_height),
        }

        # Initialize covariance based on model type
        covariance_params = {"ndim": 3, "length_scale": L, "E0": E0}
        if model in ["Mann", "DRD"]:
            covariance_params["Gamma"] = Gamma

        self.Covariance = covariance_map[model](**covariance_params)

        # Initialize random field generator
        self.RF = VectorGaussianRandomField(
            ndim=3,
            grid_level=grid_levels,
            grid_dimensions=grid_dimensions,
            sampling_method="vf_fftw",
            grid_shape=self.noise_shape[:-1],
            Covariance=self.Covariance,
        )

        self.RF.reseed(self.seed)

    def _generate_block(self) -> np.ndarray:
        """Generates a single block of the fluctuation field.

        Returns
        -------
        np.ndarray
            A single block of the fluctuation field, to be concatenated with the total field.
        """
        if self.noise is None:
            noise = self.RF.sample_noise(self.noise_shape)
        else:
            noise = np.roll(self.noise, -self.Nx, axis=0)
            noise[tuple(self.new_part)] = self.RF.sample_noise(self.new_part_shape)
        self.noise = noise

        wind_block = self.RF.sample(noise)
        wind = wind_block[tuple(self.central_part)]
        if self.blend_num > 0:
            self.blend_region = wind[-self.blend_num :, ...].copy()
        else:
            self.blend_region = None
        if self.blend_num > 1:
            wind = wind[: -(self.blend_num - 1), ...]

        return wind

    def _normalize_block(
        self,
        curr_block: np.ndarray,
        zref: float,
        uref: float,
        z0: float,
        windprofiletype: str,
        plexp: Optional[float] = None,
    ) -> np.ndarray:
        r"""Normalizes an individual block of wind under the given profile and physical parameters.

        Parameters
        ----------
        curr_block : np.ndarray
            _description_

        zref : float
            Reference height.
        uref : float
            Reference velocity.
        z0 : float
            Roughness height.
        windprofiletype : str
            Type of wind profile by which to normalize, either ``"LOG"`` for logarithmic scaling
            or ``"PL"`` for power law scaling: for ``"LOG"``,

            .. math::
                \left\langle U_1(z)\right\rangle= U_{\text{ref}} \frac{\ln \left( \frac{z}{z_0} + 1 \right)}
                {\ln \left( \frac{z_{\text{ref}}}{z_0} \right)}

            or for ``"PL"``,

            .. math::
                \left\langle U_1(z)\right\rangle= u_* \left( \frac{z}{z_{\text{ref}}} \right)^\alpha

            where :math:`u_*` is the friction velocity and :math:`z_{\text{ref}}` is the reference height.

        plexp : Optional[float], optional
            Power law exponent :math:`\alpha`, by default None.

        Returns
        -------
        np.ndarray
            Fluctuation field normalized by the logarithmic profile.

        Raises
        ------
        ValueError
            In the case that curr_block does not satisfy not np.any (ie, it is empty, or all zeros).
            "No fluctuation field has been generated, call the .generate() method first."

        ValueError
            If windprofiletype is not one of "LOG" or "PL".

        ValueError
            In the case that any of the parameters zref, uref, or z0 are not positive.
        """
        if not np.any(curr_block):
            raise ValueError("No fluctuation field has been generated. The .generate() method must be called first.")

        if windprofiletype not in ["LOG", "PL"]:
            raise ValueError('windprofiletype must be either "LOG" or "PL"')

        if any(param <= 0 for param in [zref, uref, z0]):
            raise ValueError("zref, uref, and z0 must all be positive")

        if windprofiletype == "PL" and plexp is None:
            raise ValueError("Power law exponent (plexp) is required when using power law profile")

        sd = np.sqrt(np.mean(curr_block**2))
        curr_block /= sd

        z_space = np.linspace(0.0, self.grid_dimensions[2], 2 ** (self.grid_levels[2]) + 1)
        if windprofiletype == "LOG":
            mean_profile_z = self.log_law(z_space, z0, zref, uref)
        else:
            assert plexp is not None, "Power law exponent (plexp) is required when using power law mean profile."
            mean_profile_z = self.power_law(z_space, zref, uref, plexp)

        mean_profile = np.zeros_like(curr_block)
        # NOTE: (Leaving for now) this *was* mean_profile_z.T, but that's a float? so there's no transpose.
        mean_profile[..., 0] = np.tile(mean_profile_z, (mean_profile.shape[0], mean_profile.shape[1], 1))

        return curr_block + mean_profile

    def generate(
        self,
        num_blocks: int,
        zref: float,
        uref: float,
        z0: float,
        windprofiletype: str,
        plexp: Optional[float] = None,
    ) -> np.ndarray:
        r"""Generates the full fluctuation field in blocks. The resulting field is stored as the ``total_fluctuation``
        field of this object, allowing for all metadata of the object to be stored safely with the fluctuation field,
        and also reducing data duplication for post-processing; all operations can be performed on this public
        variable.

        .. warning::
            If this method is called twice in the same object, additional fluctuation field blocks will be appended
            to the field generated from the first call. If this is undesirable behavior, instantiate a new object.

        Parameters
        ----------
        num_blocks : int
            Number of blocks to use in fluctuation field generation.
        zref : float
            Reference height.
        uref : float
            Reference velocity.
        z0 : float
            Roughness height.
        windprofiletype : str
            Type of wind profile by which to normalize, either ``"LOG"`` for logarithmic scaling or ``"PL"`` for
            power law scaling: for ``"LOG"``,

            .. math::
                \left\langle U_1(z)\right\rangle=\frac{u_*}{\kappa} \ln \left(\frac{z}{z_0}+1\right)

            or for ``"PL"``,

            .. math::
                \left\langle U_1(z)\right\rangle= u_* \left( \frac{z}{z_{\text{ref}}} \right)^\alpha

            where :math:`u_*` is the friction velocity and :math:`z_{\text{ref}}` is the reference height.

        plexp : Optional[float], optional
            Power law exponent :math:`\alpha`, by default None.

        Returns
        -------
        np.ndarray
            The full fluctuation field, which is also stored as the ``total_fluctuation`` field.
        """
        if np.any(self.total_fluctuation):
            import warnings

            warnings.warn(
                "Fluctuation field has already been generated, additional blocks will be appended to existing field.\
                If this is undesirable behavior, instantiate a new object."
            )

        for _ in range(num_blocks):
            t_block = self._generate_block()

            normed_block = self._normalize_block(
                curr_block=t_block,
                zref=zref,
                uref=uref,
                z0=z0,
                windprofiletype=windprofiletype,
                plexp=plexp,
            )

            self.total_fluctuation = np.concatenate((self.total_fluctuation, normed_block), axis=0)

        return self.total_fluctuation

    def save_to_vtk(self, filepath: Union[str, Path] = "./"):
        """Saves generated fluctuation field in VTK format to specified filepath.

        Parameters
        ----------
        filepath : Union[str, Path]
           Filepath to which to save generated fluctuation field.
        """
        from pyevtk.hl import imageToVTK

        spacing = tuple(self.grid_dimensions / (2.0**self.grid_levels + 1))

        wind_field_vtk = tuple([np.copy(self.total_fluctuation[..., i], order="C") for i in range(3)])

        cellData = {
            "grid": np.zeros_like(self.total_fluctuation[..., 0]),
            "wind": wind_field_vtk,
        }

        imageToVTK(filepath, cellData=cellData, spacing=spacing)

    def evaluate_divergence(self, spacing: Union[tuple, np.ndarray], field: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Evaluates the point-wise divergence of a generated fluctuation vector (!) field on a given grid. The
        underlying method is numpy's ``gradient`` function, which is computed with second-order central
        difference methods.

        .. note::

            If the generated field has been normalized with ``.normalize()``, it must be passed into this method as
            the ``field`` argument. The default evaluation of this method is on the ``total_fluctuation`` attribute
            of this object.

        This method will approximate

        .. math::
            \operatorname{div} \boldsymbol{F} = \frac{\partial \boldsymbol{F}_x}{\partial x} +
            \frac{\partial \boldsymbol{F}_y}{\partial y} + \frac{\partial \boldsymbol{F}_z}{\partial z}.

        Note that the vector field is assumed to be 3D.

        Parameters
        ----------
        spacing : Union[tuple, np.ndarray]
            The spacing of the grid on which the fluctuation field has been generated. This is necessary for
            derivatives to be computed properly.
        field : Optional[np.ndarray], optional
            The fluctuation field containing all field components, of the shape :math:`(x, y, z, 3)`, by default
            None, which evaluates the divergence of the non-normalized field stored in ``total_fluctuation``.

        Returns
        -------
        np.ndarray
            Point-wise divergence of the vector field, this will be of shape (x, y, z). To gather further information
            about the divergence, consider using ``.max()``, ``.sum()`` or ``.mean()`` to determine the maximum,
            total, or average point-wise divergence of the generated field.

        Raises
        ------
        ValueError
            Spacing must contain 3 scalars determining the spacing of evaluation points of the field for
            each dimension.
        ValueError
            Last dimension of vector field must be 3, consider reshaping your vector field.
        """
        if len(spacing) != 3:
            raise ValueError(
                "Spacing must contain 3 scalars determining the spacing of evaluation points of the field for \
                each dimension."
            )

        if field is None:
            field = self.total_fluctuation

        if field.shape[-1] != 3:
            raise ValueError("Last dimension of vector field must be 3, consider reshaping your vector field.")

        return np.ufunc.reduce(np.add, [np.gradient(field[..., i], spacing[i], axis=i) for i in range(3)])

    @staticmethod
    def log_law(z: float, z0: float, zref: float, uref: float) -> float:
        """Calculate wind speed using logarithmic law."""
        return uref * np.log(z / z0 + 1.0) / np.log(zref / z0)

    @staticmethod
    def power_law(z: float, zref: float, Uref: float, a: float) -> float:
        """Calculate wind speed using power law."""
        return Uref * (z / zref) ** a
