"""
Implements a 2D low-frequency turbulence model.

This module provides the `LowFreqGenerator` class for generating 2D low-frequency
fluctuational fields, which demonstrate much larger scale coherent structures
on the scale of kilometers, rather than meters as is the case for the
small-scale turbulence controlled elsewhere.
"""

import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator


class LowFreqGenerator:
    r"""Generator class for a 2d low-frequency fluctuational field model.

    This class implements a 2d low-frequency fluctuational field model based on the following references:
    #. [1] `A.H. Syed, J. Mann "A Model for Low-Frequency, Anisotropic Wind Fluctuations and Coherences
        in the Marine Atmosphere", Boundary-Layer Meteorology 190:1, 2024 <https://doi.org/10.1007/s10546-023-00850-w>`_.
    #. [2] `A.H. Syed, J. Mann "Simulating low-frequency wind fluctuations", Wind Energy Science, 9, 1381-1391, 2024
        <https://doi.org/10.5194/wes-9-1381-2024>`_.

    In summary, the model has 4 parameters:
    - :math:`\sigma^2` - variance exhibited by the low-frequency fluctuation field (:math:`m^2/s^2`),
    - :math:`L_{2d}` - length scale corresponding to the peak of mesoscale turbulence (:math:`m`,
        typically several kilometers),
    - :math:`\psi` - anisotropy parameter (:math:`rad`), :math:`0 < \psi < \pi/2`,
    - and :math:`z_i` - height of the inertial sublayer (:math:`m`, assume to be the boundary layer height).

    The shape of the energy spectrum is inspired by the Von Karman spectrum and is tuned to marine
    atmospheres:

    .. math::
        E(\kappa) = \frac{c \kappa^3}{(L_{2d}^{-2} + \kappa^2) ^ {7/3}} \frac{1}{1 + \kappa^2 z_i^2}

    where

    .. math::
        \kappa = \sqrt{2 (\cos(\psi) k_1)^2 + (\sin(\psi) k_2)^2}

    and :math:`c` is a scaling parameter that normalizes the field to have the correct variance :math:`\sigma^2`. The
    second factor in the denominator is there to attenuate the energy at high wavenumbers, specifically
    :math:`k > 1/z_i`.

    The outputs of this generator form :math:`xy`-planes parallel to the mean wind direction.

    We assume that these fluctuations
    are vertically homogeneous and statistically independent of the small-scale fluctuations
    that this model is to be coupled with to construct the complete :math:`2\rm{d}+3\rm{d}` model.

    Parameters
    ----------
    config : dict
        Configuration dictionary for the low-frequency generator.
        Requires the following keys:
        - ``"sigma2"`` : float
            Variance of the low-frequency fluctuation field (:math:`m^2/s^2`).
        - ``"L_2d"`` : float
            Length scale corresponding to the peak of mesoscale turbulence (:math:`m`).
        - ``"psi"`` : float
            Anisotropy parameter (:math:`rad`), :math:`0 < \psi < \pi/2`.
        - ``"z_i"`` : float
            Height of the inertial sublayer (:math:`m`).
        - ``"L1_factor"`` : float
            Factor determining user-specified domain length in x-direction relative to ``L_2d``.
        - ``"L2_factor"`` : float
            Factor determining user-specified domain length in y-direction relative to ``L_2d``.
        - ``"exp1"`` : int
            Exponent for the number of points in the x-direction (``user_N1 = 2**exp1``).
        - ``"exp2"`` : int
            Exponent for the number of points in the y-direction (``user_N2 = 2**exp2``).



    Notes
    -----
    The computational domain size and resolution are calculated internally to ensure
    the domain is at least 5 times the length scale :math:`L_{2d}` and encompasses the
    user's requested domain, while maintaining an isotropic grid resolution based on the
    finer of the user's requested spacings.

    """

    def __init__(self, config: dict):
        """
        Initialize the LowFreqGenerator instance.

        Sets up physical parameters, user-specified domain parameters, calculates
        internal computational domain parameters, and computes wavenumber grids.

        Parameters
        ----------
        config : dict
            Configuration dictionary. See class docstring for required keys.
        """
        # Physical parameters
        self.sigma2 = config["sigma2"]
        self.L_2d = config["L_2d"]
        self.psi = config["psi"]
        self.z_i = config["z_i"]

        self.c = self._compute_c()

        # User-specified domain (store these)
        self.user_L1 = config["L1_factor"] * self.L_2d
        self.user_L2 = config["L2_factor"] * self.L_2d
        self.user_N1 = 2 ** config["exp1"]
        self.user_N2 = 2 ** config["exp2"]
        self.user_dx = self.user_L1 / self.user_N1
        self.user_dy = self.user_L2 / self.user_N2

        # Store 1D user coordinates explicitly
        self.user_x_coords = np.linspace(0, self.user_L1, self.user_N1, endpoint=False)
        self.user_y_coords = np.linspace(0, self.user_L2, self.user_N2, endpoint=False)
        # Create user meshgrid for plotting/output if needed elsewhere
        self.X, self.Y = np.meshgrid(self.user_x_coords, self.user_y_coords, indexing="ij")

        # Calculate computational domain sizes (isotropic, >= 5*L_2d, encompasses user grid)
        self.comp_L1, self.comp_L2, self.comp_N1, self.comp_N2 = self._calculate_buffer_sizes()
        self.comp_dx = self.comp_L1 / self.comp_N1
        self.comp_dy = self.comp_L2 / self.comp_N2

        # Verify computational grid is isotropic
        if not np.isclose(self.comp_dx, self.comp_dy):
            raise ValueError(f"Computational grid is not isotropic! dx={self.comp_dx}, dy={self.comp_dy}")

        # Create wavenumber arrays for the *computational* grid
        self.k1_fft = 2 * np.pi * np.fft.fftfreq(self.comp_N1, self.comp_dx)
        self.k2_fft = 2 * np.pi * np.fft.fftfreq(self.comp_N2, self.comp_dy)
        self.k1, self.k2 = np.meshgrid(self.k1_fft, self.k2_fft, indexing="ij")

        self.config = config

        # Initialize fields to None until generate() is called
        self.u1 = None
        self.u2 = None
        self.u1_full = None
        self.u2_full = None

    @staticmethod
    def _check_isotropic_grid(Lx: float, Ly: float, Nx: int, Ny: int) -> bool:
        """Check if grid parameters result in an isotropic grid."""
        return Lx / Nx == Ly / Ny

    def _calculate_buffer_sizes(self) -> tuple[float, float, int, int]:
        """Calculate the required internal computational grid sizes.

        This function determines the size and resolution of the internal grid
        used for field generation to ensure statistical properties are met.

        The computational grid is designed to be:
        *   Isotropic (``comp_dx == comp_dy``).
        *   At least 5 times the length scale :math:`L_{2d}` in each dimension.
        *   Large enough to fully contain the user's requested domain (``user_L1``, ``user_L2``).
        *   Resolution is determined by the finer of the user's requested spacings (``user_dx``, ``user_dy``).
        *   Number of points in each dimension (``comp_N1``, ``comp_N2``) is even.

        Returns
        -------
        comp_L1 : float
            Length of the computational domain in the x-direction (m).
        comp_L2 : float
            Length of the computational domain in the y-direction (m).
        comp_N1 : int
            Number of grid points in the computational domain in the x-direction.
        comp_N2 : int
            Number of grid points in the computational domain in the y-direction.

        Raises
        ------
        RuntimeError
            If the internal calculation fails to produce a computational grid
            that meets the size requirements (>= 5*L_2d and >= user domain).
        UserWarning
            If the user's requested domain is smaller than the recommended minimum
            size of 5*L_2d x 5*L_2d.
        """
        # Determine the isotropic grid spacing for the computational grid
        comp_d = min(self.user_dx, self.user_dy)

        # Target minimum lengths based on 5*L_2d and user request
        L1_target = max(self.user_L1, 5 * self.L_2d)
        L2_target = max(self.user_L2, 5 * self.L_2d)

        # Warn if user domain is smaller than recommended minimum
        if self.user_L1 < 5 * self.L_2d or self.user_L2 < 5 * self.L_2d:
            import warnings

            warnings.warn(
                f"User requested domain ({self.user_L1:.1f}x{self.user_L2:.1f}) is smaller than the "
                f"recommended minimum size ({5*self.L_2d:.1f}x{5*self.L_2d}). "
                f"Computational domain will be enlarged to ensure minimum size.",
                UserWarning,
            )

        # Calculate ideal minimum number of points for target lengths and computed spacing
        n1_min_ideal = int(np.ceil(L1_target / comp_d))
        n2_min_ideal = int(np.ceil(L2_target / comp_d))

        # Ensure number of points is even (often beneficial for FFT)
        # Ensure N=0 if ideal is 0, otherwise ensure N>=2 if ideal > 0
        comp_N1 = n1_min_ideal + (n1_min_ideal % 2) if n1_min_ideal > 0 else 0
        comp_N2 = n2_min_ideal + (n2_min_ideal % 2) if n2_min_ideal > 0 else 0
        comp_N1 = max(comp_N1, 2) if n1_min_ideal > 0 else 0
        comp_N2 = max(comp_N2, 2) if n2_min_ideal > 0 else 0

        # Calculate final computational domain lengths based on N and d
        comp_L1 = comp_N1 * comp_d
        comp_L2 = comp_N2 * comp_d

        # Verify that the calculated lengths meet the target requirements
        tolerance = 1e-9
        if not (comp_L1 >= L1_target - tolerance and comp_L2 >= L2_target - tolerance):
            raise RuntimeError(
                "Internal calculation error: Final computational grid does not meet size requirements. "
                f"Needed ({L1_target:.4f}, {L2_target:.4f}), got ({comp_L1:.4f}, {comp_L2:.4f})"
            )

        return comp_L1, comp_L2, comp_N1, comp_N2

    def _compute_c(self) -> float:
        r"""Compute the scaling factor c for the energy spectrum.

        The scaling factor :math:`c` ensures that the integral of the energy spectrum
        over all wavenumbers equals the desired variance :math:`\sigma^2`.

        Returns
        -------
        float
            The scaling factor :math:`c`.

        Notes
        -----
        The scaling factor is determined by solving:

        .. math::
            \sigma^2 = 2 \int_0^\infty E_{shape}(\kappa) d\kappa

        where :math:`E_{shape}(\kappa)` is the shape of the energy spectrum without the
        scaling factor :math:`c`:

        .. math::
           E_{shape}(\kappa) = \frac{\kappa^3}{(L_{2d}^{-2} + \kappa^2) ^ {7/3}} \frac{1}{1 + \kappa^2 z_i^2}

        Thus, :math:`c = \sigma^2 / (2 \int_0^\infty E_{shape}(\kappa) d\kappa)`.
        The factor of 2 arises from integrating over the radial coordinate in 2D wavenumber space.
        """
        L_2d = self.L_2d
        z_i = self.z_i

        def integrand_shape(kappa):
            """Define the shape of the energy spectrum E(kappa) / c."""
            # Avoid potential division by zero or log(0) issues at kappa=0
            kappa_sq = kappa**2
            if kappa_sq < 1e-30:
                return 0.0

            denom1 = (L_2d**-2 + kappa_sq) ** (7 / 3)
            denom2 = 1.0 + kappa_sq * z_i**2
            denominator = denom1 * denom2

            # Avoid division by zero if denominator is extremely small
            if denominator < 1e-30:
                return 0.0

            # kappa^3 term can be zero if kappa is zero
            if kappa < 1e-30:
                return 0.0

            return (kappa**3) / denominator

        integral_val, integral_err = integrate.quad(integrand_shape, 0, np.inf)

        if integral_val < 1e-16:
            raise ValueError(
                f"Integral of spectrum shape is near zero ({integral_val:.2e}). "
                "Check physical parameters L_2d and z_i."
            )

        return self.sigma2 / (2 * integral_val)

    # ------------------------------------------------------------------------------------------------ #
    @staticmethod
    @numba.njit(parallel=True, fastmath=True)
    def _generate_numba_helper(
        k1: np.ndarray,
        k2: np.ndarray,
        c: float,
        L_2d: float,
        psi: float,
        z_i: float,
        dx: float,
        dy: float,
        N1: int,
        N2: int,
        noise_hat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute Fourier components of velocity fluctuations using Numba.

        Calculates the Fourier space components :math:`\hat{u}_1` and :math:`\hat{u}_2`
        on the computational grid.

        Parameters
        ----------
        k1 : np.ndarray
            2D array of wavenumbers in the x-direction (computational grid).
        k2 : np.ndarray
            2D array of wavenumbers in the y-direction (computational grid).
        c : float
            Scaling factor for the energy spectrum.
        L_2d : float
            Length scale corresponding to the peak of mesoscale turbulence.
        psi : float
            Anisotropy parameter (:math:`rad`).
        z_i : float
            Height of the inertial sublayer (:math:`m`).
        dx : float
            Grid spacing in the x-direction (computational grid, :math:`m`).
        dy : float
            Grid spacing in the y-direction (computational grid, :math:`m`).
        N1 : int
            Number of points in the x-direction (computational grid).
        N2 : int
            Number of points in the y-direction (computational grid).
        noise_hat : np.ndarray
            Complex Fourier transform of the initial random noise field.

        Returns
        -------
        u1_freq : np.ndarray
            Fourier components of the u1 velocity field.
        u2_freq : np.ndarray
            Fourier components of the u2 velocity field.

        Notes
        -----
        The calculation follows the formulation for simulating 2D turbulence,
        where the Fourier components are related to the energy spectrum :math:`E(\kappa)`
        and the wavenumbers :math:`k_1, k_2`.

        .. math::
            \hat{u}_1(\mathbf{k}) = i k_2 \sqrt{\frac{E(\kappa)}{\pi k^3}}
                \left( \frac{2\pi}{\sqrt{dx dy}} \right) \hat{n}(\mathbf{k}) \\
            \hat{u}_2(\mathbf{k}) = -i k_1 \sqrt{\frac{E(\kappa)}{\pi k^3}}
                \left( \frac{2\pi}{\sqrt{dx dy}} \right) \hat{n}(\mathbf{k})

        where :math:`\mathbf{k} = (k_1, k_2)`, :math:`k = |\mathbf{k}|`, and :math:`\hat{n}(\mathbf{k})`
        is the Fourier transform of the input noise field. Small values are handled
        to avoid division by zero.
        """
        # Calculate magnitudes of k and kappa vectors
        k_mag_sq = k1**2 + k2**2
        kappa_sq = 2 * ((k1 * np.cos(psi)) ** 2 + (k2 * np.sin(psi)) ** 2)

        phi_ = np.empty_like(k_mag_sq, dtype=np.float64)

        # Grid scaling factor for discrete Fourier transform normalization
        grid_scale = (2 * np.pi) / np.sqrt(max(dx * dy, 1e-16))

        for i in numba.prange(N1):
            for j in numba.prange(N2):
                _kappa_sq = kappa_sq[i, j]
                _k_sq = k_mag_sq[i, j]

                # Handle zero wavenumber case (DC component)
                if _k_sq < 1e-20:
                    phi_[i, j] = 0.0
                    continue

                _k = np.sqrt(_k_sq)
                _kappa = np.sqrt(max(_kappa_sq, 1e-20))  # Ensure kappa is non-negative

                # Calculate energy spectrum E(kappa)
                denom1 = (L_2d**-2 + _kappa_sq) ** (7 / 3)
                denom2 = 1 + _kappa_sq * z_i**2
                denominator = denom1 * denom2

                # Avoid division by zero if denominator is extremely small
                if denominator < 1e-30:
                    energy = 0.0
                else:
                    energy = c * (_kappa**3) / denominator

                # Calculate sqrt(E(kappa) / (pi * k^3)) factor
                # Note: k^3 in denominator
                phi_term = energy / (np.pi * _k**3)
                phi_[i, j] = np.sqrt(max(phi_term, 0.0)) * grid_scale

        # Calculate Fourier components using the stream function relation
        Q1 = 1j * phi_ * k2
        Q2 = 1j * phi_ * (-1 * k1)

        # Check for NaNs which might indicate numerical issues
        if np.isnan(Q1).any() or np.isnan(Q2).any():
            # Numba doesn't support warnings, potential alternative is to return an error flag
            pass  # Consider how to handle potential NaNs if they occur

        return Q1 * noise_hat, Q2 * noise_hat

    def _extract_field_portion(self, field: np.ndarray) -> np.ndarray:
        """Extract the central portion of a field matching user dimensions.

        Takes a field computed on the internal computational grid and extracts
        the central portion corresponding to the user's requested number of
        points (``user_N1``, ``user_N2``).

        Parameters
        ----------
        field : np.ndarray
            The 2D velocity field computed on the computational grid.

        Returns
        -------
        np.ndarray
            The extracted 2D velocity field with dimensions ``(user_N1, user_N2)``.

        Raises
        ------
        ValueError
            If the shape of the extracted field does not match the requested shape.
        """
        # Get user-requested number of points
        nx_requested = self.user_N1
        ny_requested = self.user_N2

        # Get computational grid dimensions
        nx_computed, ny_computed = field.shape

        # If grids match, no extraction needed
        if nx_computed == nx_requested and ny_computed == ny_requested:
            return field

        # Calculate start/end indices for a centered extraction
        start_x = (nx_computed - nx_requested) // 2
        start_y = (ny_computed - ny_requested) // 2

        end_x = start_x + nx_requested
        end_y = start_y + ny_requested

        # Extract the portion using slicing
        extracted_field = field[start_x:end_x, start_y:end_y]

        # Sanity check the shape of the extracted field
        if extracted_field.shape != (nx_requested, ny_requested):
            raise ValueError(
                f"Extracted field shape {extracted_field.shape} doesn't match "
                f"requested shape {(nx_requested, ny_requested)}"
            )

        return extracted_field

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate the 2D low-frequency velocity fluctuation fields.

        This method performs the core generation process:
        1. Generates a Gaussian white noise field on the computational grid.
        2. Computes its Fourier transform.
        3. Calculates the Fourier components of the velocity fields (u1, u2)
           using the energy spectrum and the noise field.
        4. Transforms the velocity components back to physical space via inverse FFT.
        5. Stores the full fields (on the computational grid) in ``u1_full``, ``u2_full``.
        6. Extracts the central portion corresponding to the user's requested dimensions
           and stores them in ``u1``, ``u2``.

        Attributes Set
        --------------
        u1_full : np.ndarray
            Velocity field u1 on the full internal, computational grid.
        u2_full : np.ndarray
            Velocity field u2 on the full internal, computational grid.
        u1 : np.ndarray
            Velocity field u1 extracted to match user's requested dimensions (N1, N2).
        u2 : np.ndarray
            Velocity field u2 extracted to match user's requested dimensions (N1, N2).

        Returns
        -------
        u1 : np.ndarray
            The u1 component of the velocity field, extracted to match user dimensions.
        u2 : np.ndarray
            The u2 component of the velocity field, extracted to match user dimensions.
        """
        # 1. Generate noise on the computational grid
        noise = np.random.normal(0, 1, size=(self.comp_N1, self.comp_N2))
        # 2. Compute its Fourier transform
        noise_hat = np.fft.fft2(noise)

        # 3. Calculate Fourier components of velocity fields
        u1_freq, u2_freq = self._generate_numba_helper(
            self.k1,  # Wavenumbers (comp grid)
            self.k2,  # Wavenumbers (comp grid)
            self.c,  # Spectrum scaling factor
            self.L_2d,  # Length scale
            self.psi,  # Anisotropy parameter
            self.z_i,  # Inertial sublayer height
            self.comp_dx,  # Grid spacing (comp grid)
            self.comp_dy,  # Grid spacing (comp grid)
            self.comp_N1,  # Grid points (comp grid)
            self.comp_N2,  # Grid points (comp grid)
            noise_hat,  # Fourier transformed noise
        )

        # 4. Transform back to physical space
        u1_full = np.real(np.fft.ifft2(u1_freq))
        u2_full = np.real(np.fft.ifft2(u2_freq))

        # 5. Store the full fields
        self.u1_full = u1_full
        self.u2_full = u2_full

        # 6. Extract the portion matching user's N1, N2
        u1 = self._extract_field_portion(u1_full)
        u2 = self._extract_field_portion(u2_full)

        # Store the *extracted* fields
        self.u1 = u1
        self.u2 = u2

        return u1, u2

    # ------------------------------------------------------------------------------------------------ #
    # Below are member functions for interpolating the low-frequency fields to the 3d grid

    def interp_slice(self, x_coords_target: np.ndarray, y_coords_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate the generated low-frequency fields onto a target grid slice.

        Uses linear interpolation (`scipy.interpolate.RegularGridInterpolator`)
        to map the generated 2D fields (``u1``, ``u2``, defined on the user's
        x/y coordinates) onto a target 2D grid defined by ``x_coords_target``
        and ``y_coords_target``.

        Parameters
        ----------
        x_coords_target : np.ndarray
            1D array of target x-coordinates for the slice.
        y_coords_target : np.ndarray
            1D array of target y-coordinates for the slice.

        Returns
        -------
        u1_interp : np.ndarray
            Interpolated u1 field on the target grid slice. Shape matches the
            meshgrid created from ``x_coords_target`` and ``y_coords_target``.
        u2_interp : np.ndarray
            Interpolated u2 field on the target grid slice. Shape matches the
            meshgrid created from ``x_coords_target`` and ``y_coords_target``.

        Raises
        ------
        RuntimeError
            If ``generate()`` has not been called before this method, or if
            required internal attributes (``user_x_coords``, ``user_y_coords``)
            are missing.
        """
        # Check if source data exists
        if self.u1 is None or self.u2 is None:
            raise RuntimeError("LowFreqGenerator.generate() must be called before interp_slice()")
        if not hasattr(self, "user_x_coords") or not hasattr(self, "user_y_coords"):
            # Should not happen if __init__ completed successfully
            raise RuntimeError("LowFreqGenerator needs user_x_coords and user_y_coords attributes.")

        # Create interpolators for u1 and u2 fields defined on the user grid
        # Using linear interpolation, allow extrapolation (fill_value=0.0)
        interpolator_2d_u1 = RegularGridInterpolator(
            (self.user_x_coords, self.user_y_coords),  # Source grid coordinates
            self.u1,  # Source data
            method="linear",
            bounds_error=False,  # Allow extrapolation
            fill_value=0.0,  # Value for points outside source grid
        )

        interpolator_2d_u2 = RegularGridInterpolator(
            (self.user_x_coords, self.user_y_coords),
            self.u2,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        # Create a meshgrid of the target coordinates
        X_target, Y_target = np.meshgrid(x_coords_target, y_coords_target, indexing="ij")

        # Prepare target points for the interpolator (needs NxM, 2 shape)
        target_points_xy = np.stack([X_target.ravel(), Y_target.ravel()], axis=-1)

        # Perform interpolation
        u1_interp_flat = interpolator_2d_u1(target_points_xy)
        u2_interp_flat = interpolator_2d_u2(target_points_xy)

        # Reshape flat interpolated data back to the target grid shape
        u1_interp = u1_interp_flat.reshape(X_target.shape)
        u2_interp = u2_interp_flat.reshape(X_target.shape)

        return u1_interp, u2_interp

    # ------------------------------------------------------------------------------------------------ #
    # Below are member functions for estimating the 1d spectra of the generated fields, F11 and F22.

    @staticmethod
    @numba.njit(parallel=True, fastmath=True)
    def _compute_spectrum_numba_helper(
        power_u1: np.ndarray,
        power_u2: np.ndarray,
        k1_grid: np.ndarray,
        k1_pos: np.ndarray,
        scaling_factor: float,
        k_tol: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1D spectra F11 and F22 from 2D power spectra (Numba helper).

        Integrates the 2D power spectra ``power_u1`` and ``power_u2`` along the k2
        direction for each positive k1 value specified in ``k1_pos``.

        Parameters
        ----------
        power_u1 : np.ndarray
            2D array of power spectrum |FFT(u1)|^2.
        power_u2 : np.ndarray
            2D array of power spectrum |FFT(u2)|^2.
        k1_grid : np.ndarray
            2D array of k1 wavenumbers corresponding to the power spectra arrays.
        k1_pos : np.ndarray
            1D array of positive k1 values for which to compute the 1D spectrum.
        scaling_factor : float
            Normalization factor for the spectrum calculation.
        k_tol : float
            Tolerance used to identify wavenumbers matching ``k1_pos`` values.

        Returns
        -------
        F11 : np.ndarray
            1D spectrum F11(k1) corresponding to ``k1_pos``.
        F22 : np.ndarray
            1D spectrum F22(k1) corresponding to ``k1_pos``.
        """
        F11 = np.empty_like(k1_pos, dtype=np.float64)
        F22 = np.empty_like(k1_pos, dtype=np.float64)
        N1, N2 = k1_grid.shape

        # Iterate through each target positive k1 value
        for i in numba.prange(len(k1_pos)):
            k1_val = k1_pos[i]

            summed_power_u1 = 0.0
            summed_power_u2 = 0.0

            # Sum power spectrum contributions along the k2 axis for the current k1
            for r in range(N1):
                for c in range(N2):
                    # Check if the grid k1 value is close to the target k1 value
                    if np.abs(k1_grid[r, c] - k1_val) < k_tol:
                        summed_power_u1 += power_u1[r, c]
                        summed_power_u2 += power_u2[r, c]

            # Apply scaling factor
            F11[i] = summed_power_u1 * scaling_factor
            F22[i] = summed_power_u2 * scaling_factor

        return F11, F22

    def compute_spectrum(self, k_tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate the 1D spectra F11(k1) and F22(k1) from the generated fields.

        Calculates the 1D spectra by integrating the 2D power spectra of the
        generated velocity fields (``u1``, ``u2`` on the user's grid) along the
        k2 direction.

        Parameters
        ----------
        k_tol : float, optional
            Tolerance for matching k1 wavenumbers during integration.
            Defaults to 1e-9.

        Returns
        -------
        k1_pos : np.ndarray
            1D array of positive k1 wavenumbers for which the spectra are calculated.
        F11 : np.ndarray
            Estimated 1D spectrum F11(k1).
        F22 : np.ndarray
            Estimated 1D spectrum F22(k1).

        Raises
        ------
        RuntimeError
            If ``generate()`` has not been called before this method.
        UserWarning
            If NaNs are detected in the intermediate power spectra calculation.

        Notes
        -----
        The spectra are computed based on the fields extracted to the user's grid
        dimensions (``self.u1``, ``self.u2``).
        The scaling factor accounts for the discrete Fourier transform normalization
        and the integration along k2.
        """
        if self.u1 is None or self.u2 is None:
            raise RuntimeError("Call generate() before compute_spectrum()")

        # Compute FFT of the extracted fields (on user grid)
        u1_fft_extracted = np.fft.fft2(self.u1)
        u2_fft_extracted = np.fft.fft2(self.u2)

        # Define wavenumbers corresponding to the user grid
        k1_fft_user = 2 * np.pi * np.fft.fftfreq(self.user_N1, self.user_dx)
        k2_fft_user = 2 * np.pi * np.fft.fftfreq(self.user_N2, self.user_dy)
        k1_user, k2_user = np.meshgrid(k1_fft_user, k2_fft_user, indexing="ij")

        # Get unique positive k1 values (excluding near-zero)
        k1_pos = np.unique(np.abs(k1_fft_user))
        k1_pos = k1_pos[k1_pos > k_tol]
        k1_pos = np.sort(k1_pos)  # Ensure sorted order

        # Compute power spectra |FFT|^2
        power_u1 = (np.abs(u1_fft_extracted)) ** 2
        power_u2 = (np.abs(u2_fft_extracted)) ** 2

        # Check for potential issues
        if np.isnan(power_u1).any() or np.isnan(power_u2).any():
            import warnings

            warnings.warn("NaN detected in power spectra!", UserWarning)

        # Define scaling factor for DFT normalization and integration
        # Factor incorporates L1, N1*N2 grid size, and 2*pi from integration convention
        scaling_factor = self.user_L1 / ((self.user_N1 * self.user_N2) ** 2 * (2 * np.pi))

        # Compute 1D spectra using the Numba helper
        F11, F22 = self._compute_spectrum_numba_helper(
            power_u1,
            power_u2,
            k1_user,  # The k1 grid corresponding to power_u1/u2
            k1_pos,  # The target positive k1 values
            scaling_factor,
            k_tol,
        )

        return k1_pos, F11, F22

    # ------------------------------------------------------------------------------------------------ #
    def analytical_spectrum(
        self, k1_arr: np.ndarray, k2_limit: float = 5.0, warn: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute the analytical 1D spectra F11(k1) and F22(k1).

        Calculates the theoretical 1D spectra by numerically integrating the 2D
        energy spectrum :math:`E(\kappa)` along the :math:`k_2` direction for each
        given :math:`k_1` value in ``k1_arr``.

        Parameters
        ----------
        k1_arr : np.ndarray
            1D array of :math:`k_1` wavenumbers for which to compute the analytical spectra.
        k2_limit : float, optional
            Integration limit for :math:`k_2`. The integration is performed from
            ``-k2_limit`` to ``+k2_limit``. Defaults to 5.0.
        warn : bool, optional
            If ``True``, print warnings about integration errors and maximum errors.
            Defaults to ``False``.

        Returns
        -------
        F11_analytical : np.ndarray
            Analytical 1D spectrum F11(k1) corresponding to ``k1_arr``.
        F22_analytical : np.ndarray
            Analytical 1D spectrum F22(k1) corresponding to ``k1_arr``.

        Notes
        -----
        The calculation involves numerical integration using `scipy.integrate.quad`.
        The integrands are derived from the relationship between the 2D spectrum
        :math:`E(\kappa)` and the 1D spectra:

        .. math::
            F_{11}(k_1) = \int_{-\infty}^{\infty} \frac{E(\kappa)}{\pi k}
                \frac{k_2^2}{k^2} dk_2 \\
            F_{22}(k_1) = \int_{-\infty}^{\infty} \frac{E(\kappa)}{\pi k}
                \frac{k_1^2}{k^2} dk_2

        where :math:`k = \sqrt{k_1^2 + k_2^2}`. Integration limits are truncated to
        ``[-k2_limit, k2_limit]``. Potential integration errors or warnings from
        `scipy.integrate.quad` are caught and reported if ``warn=True``.
        """
        F11_res_arr = np.zeros_like(k1_arr)
        F11_err_arr = np.zeros_like(k1_arr)
        F22_res_arr = np.zeros_like(k1_arr)
        F22_err_arr = np.zeros_like(k1_arr)

        # Store class attributes locally for potential performance/clarity inside helpers
        _psi = self.psi
        _L_2d = self.L_2d
        _z_i = self.z_i
        _c = self.c

        def _E_kappa(k1: float, k2: float) -> float:
            """Calculate E(kappa) for given k1, k2."""
            kappa_squared = 2 * ((k1 * np.cos(_psi)) ** 2 + (k2 * np.sin(_psi)) ** 2)
            # Avoid issues with exactly zero kappa
            kappa_squared = max(kappa_squared, 1e-24)
            _kappa = np.sqrt(kappa_squared)

            denom_term_1 = (_L_2d**-2 + kappa_squared) ** (7 / 3)
            denom_term_2 = 1 + kappa_squared * _z_i**2
            denominator = denom_term_1 * denom_term_2

            if denominator < 1e-30:  # Avoid division by zero
                return 0.0
            Ekappa = _c * (_kappa**3) / denominator
            if not np.isfinite(Ekappa):  # Catch potential overflows/NaNs
                return 0.0
            return Ekappa

        def _integrand11(k2: float, k1: float) -> float:
            """Integrand for F11."""
            k_mag_sq = k1**2 + k2**2
            if k_mag_sq < 1e-24:  # Avoid division by zero if k1=k2=0
                return 0.0
            k_mag = np.sqrt(k_mag_sq)

            Ekappa = _E_kappa(k1, k2)

            # Term E(kappa) / (pi * k^3) * k2^2
            integrand = (Ekappa / (np.pi * k_mag)) * (k2**2 / k_mag_sq)
            return integrand if np.isfinite(integrand) else 0.0

        def _integrand22(k2: float, k1: float) -> float:
            """Integrand for F22."""
            k_mag_sq = k1**2 + k2**2
            if k_mag_sq < 1e-24:  # Avoid division by zero if k1=k2=0
                return 0.0
            k_mag = np.sqrt(k_mag_sq)

            Ekappa = _E_kappa(k1, k2)

            # Term E(kappa) / (pi * k^3) * k1^2
            integrand = (Ekappa / (np.pi * k_mag)) * (k1**2 / k_mag_sq)
            return integrand if np.isfinite(integrand) else 0.0

        # Set integration parameters
        integration_limit = abs(k2_limit)  # Ensure positive limit
        quad_kwargs = {"limit": 100, "epsabs": 1.49e-08, "epsrel": 1.49e-08}

        if warn:
            print(f"Using integration limits for k2: [-{integration_limit:.2e}, {integration_limit:.2e}]")

        # Integrate for each k1 value
        for i, k1_val in enumerate(k1_arr):
            # Skip integration if k1 is effectively zero (F22 integrand undefined)
            # F11 should be zero here anyway as k2^2 / k^2 -> 1 but E(kappa)->0 faster
            if abs(k1_val) < 1e-15:
                F11_res_arr[i], F11_err_arr[i] = 0.0, 0.0
                F22_res_arr[i], F22_err_arr[i] = 0.0, 0.0
                continue

            try:
                F11_res_arr[i], F11_err_arr[i] = integrate.quad(
                    _integrand11, -integration_limit, integration_limit, args=(k1_val,), **quad_kwargs
                )
                # Check for large relative error estimate
                if warn and abs(F11_res_arr[i]) > 1e-12 and F11_err_arr[i] > 0.1 * abs(F11_res_arr[i]):
                    rel_err = F11_err_arr[i] / F11_res_arr[i]
                    print(f"Warning: High relative error ({rel_err:.1%}) for F11 at k1={k1_val:.4e}")

            except Exception as e:
                if warn:
                    print(f"Warning: Integration failed for F11 at k1={k1_val:.4e}: {e}")
                F11_res_arr[i], F11_err_arr[i] = np.nan, np.nan

            try:
                F22_res_arr[i], F22_err_arr[i] = integrate.quad(
                    _integrand22, -integration_limit, integration_limit, args=(k1_val,), **quad_kwargs
                )
                # Check for large relative error estimate
                if warn and abs(F22_res_arr[i]) > 1e-12 and F22_err_arr[i] > 0.1 * abs(F22_res_arr[i]):
                    rel_err = F22_err_arr[i] / F22_res_arr[i]
                    print(f"Warning: High relative error ({rel_err:.1%}) for F22 at k1={k1_val:.4e}")

            except Exception as e:
                if warn:
                    print(f"Warning: Integration failed for F22 at k1={k1_val:.4e}: {e}")
                F22_res_arr[i], F22_err_arr[i] = np.nan, np.nan

        if warn:
            max_err_f11 = np.nanmax(F11_err_arr) if np.any(np.isfinite(F11_err_arr)) else 0.0
            max_err_f22 = np.nanmax(F22_err_arr) if np.any(np.isfinite(F22_err_arr)) else 0.0
            print(f"Max absolute integration error estimate on F11: {max_err_f11:.2e}")
            print(f"Max absolute integration error estimate on F22: {max_err_f22:.2e}")

        return F11_res_arr, F22_res_arr

    # ------------------------------------------------------------------------------------------------ #

    def plot_velocity_fields(self) -> None:
        """Plot the generated velocity fields u1 and u2.

        Creates a two-panel plot showing the u1 and u2 fields (extracted to the
        user's grid dimensions) using ``matplotlib.pyplot.imshow``.

        Raises
        ------
        RuntimeError
            If ``generate()`` has not been called before plotting.
        """
        if self.u1 is None or self.u2 is None:
            raise RuntimeError("LowFreqGenerator.generate() must be called before plot_velocity_fields()")

        # Prepare coordinates in kilometers for plotting
        x_km = self.X / 1000
        y_km = self.Y / 1000

        # Determine symmetric color limits based on data range
        vmin = min(np.min(self.u1), np.min(self.u2))
        vmax = max(np.max(self.u1), np.max(self.u2))
        vlim = max(abs(vmin), abs(vmax))  # Symmetric limit around zero
        vmin, vmax = -vlim, vlim

        # Define the extent for imshow based on coordinate edges
        # Assuming user_x_coords and user_y_coords represent cell centers,
        # calculate edges. If they represent edges, adjust accordingly.
        dx_plot = self.user_dx / 1000
        dy_plot = self.user_dy / 1000
        x_min_plot, x_max_plot = x_km[0, 0] - dx_plot / 2, x_km[-1, 0] + dx_plot / 2
        y_min_plot, y_max_plot = y_km[0, 0] - dy_plot / 2, y_km[0, -1] + dy_plot / 2
        extent = [x_min_plot, x_max_plot, y_min_plot, y_max_plot]

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # Share x-axis

        # Plot u1
        im1 = ax1.imshow(
            self.u1.T,
            extent=extent,
            origin="lower",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        fig.colorbar(im1, ax=ax1, label="[m s$^{-1}$]")
        ax1.set_ylabel("y [km]")
        ax1.set_title("(a) u")

        # Plot u2
        im2 = ax2.imshow(
            self.u2.T,
            extent=extent,
            origin="lower",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        fig.colorbar(im2, ax=ax2, label="[m s$^{-1}$]")
        ax2.set_xlabel("x [km]")
        ax2.set_ylabel("y [km]")
        ax2.set_title("(b) v")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    cfg_fig3 = {
        "sigma2": 0.6,
        "L_2d": 15_000.0,
        "psi": np.deg2rad(43.0),
        "z_i": 500.0,
        "L1_factor": 16,
        "L2_factor": 4,
        "exp1": 12,
        "exp2": 10,
    }

    cfg_a = {
        "sigma2": 2.0,
        "L_2d": 15_000.0,
        "psi": np.deg2rad(45.0),
        "z_i": 500.0,
        "L1_factor": 40,  # For case (a): 40L_2D × 5L_2D
        "L2_factor": 5,
        "exp1": 12,
        "exp2": 9,
    }

    cfg_b = {
        "sigma2": 2.0,
        "L_2d": 15_000.0,
        "psi": np.deg2rad(45.0),
        "z_i": 500.0,
        "L1_factor": 1,  # For case (b): L_2D × 0.125L_2D
        "L2_factor": 0.125,
        "exp1": 12,
        "exp2": 9,
    }

    # lp.domain_size_study()
    # lp.length_AND_grid_size_study(cfg_a, do_plot = True)

    gen_a = LowFreqGenerator(cfg_a)
    gen_b = LowFreqGenerator(cfg_b)

    # lp.recreate_fig2(gen_a, gen_b)

    # lp.rectangular_domain_study(cfg_a)
