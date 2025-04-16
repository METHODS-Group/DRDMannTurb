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
from .low_frequency.fluctuation_field_generator import LowFreqGenerator
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
        seed: Optional[int] = None,
        blend_num: int = 10,
        config_2d_model: Optional[dict] = None,
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

        self.grid_dimensions = grid_dimensions
        self.grid_levels = grid_levels
        self.blend_num = blend_num

        # Expand on grid_levels parameter to get grid node counts in each direction
        grid_node_counts = 2**grid_levels + 1

        # Obtain spacing between grid points, split node counts into Nx, Ny, Nz
        dx, dy, dz = (L_i / N_i for L_i, N_i in zip(self.grid_dimensions, grid_node_counts))
        Nx, Ny, Nz = grid_node_counts
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        del grid_node_counts

        # Calculate buffer and margin sizes
        # NOTE: buffer scale 3 * Gamma * L is arbitrary. Could/should be tunable param?
        self.n_buffer = ceil((3 * Gamma * L) / dx)

        self.n_margin_y, self.n_margin_z = ceil(L / dy), ceil(L / dz)

        ## Spatial margin is just the length scale L

        # Calculate shapes
        buffer_extension = 2 * self.n_buffer + (self.blend_num - 1 if self.blend_num > 0 else 0)
        margin_extension = [
            2 * self.n_margin_y,
            2 * self.n_margin_z,
        ]

        self.noise_shape = [
            Nx + buffer_extension,
            Ny + margin_extension[0],
            Nz + margin_extension[1],
            3,
        ]
        self.new_part_shape = [Nx, Ny + margin_extension[0], Nz + margin_extension[1], 3]
        self.central_part = [
            slice(self.n_buffer, -self.n_buffer),
            slice(self.n_margin_y, -self.n_margin_y),
            slice(0, -2 * self.n_margin_z),
            slice(None),
        ]
        self.new_part = [
            slice(2 * self.n_buffer + max(0, self.blend_num - 1), None),
            slice(None),
            slice(None),
            slice(None),
        ]

        self.Nx = Nx
        self.seed = seed
        # NOTE: self.noise is a placeholder for now
        #   - Also, total_fluctuation is the "accumulator" for generated blocks
        self.noise = None
        self.total_fluctuation = np.zeros([0, Ny, Nz, 3])

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

        self.low_freq_gen: Optional[LowFreqGenerator] = None
        if config_2d_model is not None:
            self.low_freq_gen = LowFreqGenerator(config_2d_model)

            print("Generating low-frequency field...")

            self.low_freq_gen.generate()

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
        r"""Normalize an individual block of wind under the given profile and physical parameters.

        Parameters
        ----------
        curr_block : np.ndarray
            The block of wind to normalize.
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
        r"""Generate the full fluctuation field block by block.

        The resulting field is stored as the ``total_fluctuation`` field of this object, allowing for all metadata of
        the object to be stored safely with the fluctuation field, and also reducing data duplication for
        post-processing; all operations can be performed on this public variable.

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
        suppress_warning : bool, optional
            Suppress warning about existing fluctuation field, by default False.

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

        for i in range(num_blocks):
            # --- Generate and normalize block ---
            t_block = self._generate_block()

            normed_block = self._normalize_block(
                curr_block=t_block,
                zref=zref,
                uref=uref,
                z0=z0,
                windprofiletype=windprofiletype,
                plexp=plexp,
            )

            # --- Interpolate and add low-frequency component ---
            if self.low_freq_gen is not None:
                # Ensure low-freq field exists
                if self.low_freq_gen.u1 is None or self.low_freq_gen.u2 is None:
                    raise RuntimeError("LowFreqGenerator.generate() must be called before combining fields.")

                # --- X-Coordinates (blocks progress in x-direction
                current_block_nx = normed_block.shape[0]
                block_length_x = self.grid_dimensions[0] * (current_block_nx / self.Nx)
                x_start = i * self.grid_dimensions[0]  # TODO: Revisit if blending affects block start/end precisely
                x_end = x_start + block_length_x
                x_coords_block_target = np.linspace(x_start, x_end, current_block_nx, endpoint=False)

                # --- Y-Coordinates (where we want to center; same with each block)
                L_3d_y = self.grid_dimensions[1]  # Physical width of the 3D domain
                L_2d_y = self.low_freq_gen.user_L2  # Physical width of the 2D domain

                if L_3d_y > L_2d_y:
                    import warnings

                    warnings.warn(
                        f"3D domain width ({L_3d_y}m) is larger than 2D domain width ({L_2d_y}m). "
                        "Interpolation will use edge values (fill_value) of the 2D field."
                        "Additionally, this is likely to produce non-physical results."
                    )
                    # Center as best as possible, but coordinates will extend beyond 2D bounds

                y_center_2d = L_2d_y / 2.0
                y_half_width_3d = L_3d_y / 2.0
                y_start_target = y_center_2d - y_half_width_3d
                y_end_target = y_center_2d + y_half_width_3d

                # Generate Ny points within the calculated centered range [y_start_target, y_end_target)
                y_coords_centered_target = np.linspace(y_start_target, y_end_target, self.Ny, endpoint=False)

                # Interpolate the 2D field (u1, u2) onto this block's centered XY coordinates
                print(f"Interpolating 2D field onto centered 3D block {i+1}/{num_blocks}...")
                u1_interp, u2_interp = self.low_freq_gen.interp_slice(
                    x_coords_block_target,  # Target X coords for this sequential block
                    y_coords_centered_target,  # Target Y coords centered in 2D domain
                )
                print("Interpolation done.")

                # Verify shape
                expected_shape = (current_block_nx, self.Ny)
                if u1_interp.shape != expected_shape or u2_interp.shape != expected_shape:
                    raise ValueError(
                        f"Interpolated slice shape mismatch. Got {u1_interp.shape}, expected {expected_shape}"
                    )

                # Add the interpolated 2D components (u1, u2) to the 3D block's components
                # Use broadcasting across the Z dimension (axis 2)
                print("Adding interpolated fields...")
                normed_block[..., 0] += u1_interp[..., np.newaxis]  # Add u1_interp to u component
                normed_block[..., 1] += u2_interp[..., np.newaxis]  # Add u2_interp to v component
                print("Addition done.")

            # --- Concatenate block ---
            self.total_fluctuation = np.concatenate((self.total_fluctuation, normed_block), axis=0)

        return self.total_fluctuation

    def save_to_vtk(self, filepath: Union[str, Path] = "./") -> None:
        """Save generated fluctuation field in VTK format to specified filepath.

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
        r"""Evaluate the point-wise divergence of a generated fluctuation vector field.

        Evaluates the point-wise divergence of a generated fluctuation vector (!) field on a given grid. The
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
