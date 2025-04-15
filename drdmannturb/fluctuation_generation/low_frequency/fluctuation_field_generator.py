"""
This module implements a 2d low-frequency model, which demonstrates much larger scale coherent structures
on the scale of kilometers, rather than meters as is the case for the small-scale turbulence controlled elsewhere.

TODO: Get rid of print statements
"""

import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator


class LowFreqGenerator:
    r"""This class implements a 2d low-frequency fluctuational field model based on the following references:
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

    TODO: WARN USER THAT THEY NEED TO BE CAREFUL WITH PHYSICAL GRID SIZE
    """

    def __init__(self, config: dict):
        """
        Constructor for the LowFreqGenerator class.

        TODO: We're going to clean this interface up...

        Parameters
        ----------
            config: dict
                A dictionary containing the configuration for the low-frequency generator.
                The dictionary must contain the following keys:
                - "sigma2": float
                    The variance of the low-frequency fluctuation field.
                - "L_2d": float
                    The length scale corresponding to the peak of mesoscale turbulence.
                - "psi": float
                    The anisotropy parameter.
                - "z_i": float
                    The height of the inertial sublayer.
                - "L1_factor": float
                    The factor by which the computational domain is larger than the user-specified domain
                    in the x-direction.
                - "L2_factor": float
                    The factor by which the computational domain is larger than the user-specified domain
                    in the y-direction.
                - "exp1": int
                    The exponent for the number of points in the x-direction.
                - "exp2": int
                    The exponent for the number of points in the y-direction.
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
        return Lx / Nx == Ly / Ny

    def _calculate_buffer_sizes(self):
        """
        Calculates the buffer sizes for the computational grid, ensuring it's the
        smallest isotropic grid meeting resolution, size, and even number of points requirements.

        Requirements:
        1. Grid must be isotropic (comp_dx == comp_dy).
        2. Computational domain size (comp_L1, comp_L2) must be >= 5 * L_2d.
        3. Computational domain must encompass the user's requested domain (user_L1, user_L2).
        4. Computational grid resolution (comp_d) must be <= min(user_dx, user_dy).
        5. Number of computational grid points (comp_N1, comp_N2) must be EVEN.

        Returns
        -------
            tuple: (comp_L1, comp_L2, comp_N1, comp_N2) - Computed lengths and grid points.
        """
        # print("Calculating computational grid size...")

        # TODO: Maybe we just increase the number of points to whatever even number approx gets us dx = dy

        comp_d = min(self.user_dx, self.user_dy)
        # print(f"  Target isotropic spacing (comp_d): {comp_d:.4f} m")

        L1_target = max(self.user_L1, 5 * self.L_2d)
        L2_target = max(self.user_L2, 5 * self.L_2d)
        # print(f"  Target physical lengths (L1_target, L2_target): ({L1_target:.1f}, {L2_target:.1f}) m")

        if self.user_L1 < 5 * self.L_2d or self.user_L2 < 5 * self.L_2d:
            import warnings

            warnings.warn(
                f"User requested domain ({self.user_L1:.1f}x{self.user_L2:.1f}) is smaller than the "
                f"recommended minimum size ({5*self.L_2d:.1f}x{5*self.L_2d}). "
                f"Computational domain will be enlarged to ensure minimum size.",
                UserWarning,
            )

        n1_min_ideal = int(np.ceil(L1_target / comp_d))
        n2_min_ideal = int(np.ceil(L2_target / comp_d))
        # print(f"  Minimum points needed (n1_min_ideal, n2_min_ideal): ({n1_min_ideal}, {n2_min_ideal})")

        comp_N1 = n1_min_ideal + (n1_min_ideal % 2)
        comp_N2 = n2_min_ideal + (n2_min_ideal % 2)

        # NOTE: Should never happen, but jic
        if comp_N1 == 0 and n1_min_ideal > 0:
            comp_N1 = 2
        if comp_N2 == 0 and n2_min_ideal > 0:
            comp_N2 = 2
        # print(f"  Even computational points (comp_N1, comp_N2): ({comp_N1}, {comp_N2})")

        comp_L1 = comp_N1 * comp_d
        comp_L2 = comp_N2 * comp_d
        # print(f"  Final computational lengths (comp_L1, comp_L2): ({comp_L1:.1f}, {comp_L2:.1f}) m")

        # print("-" * 30)
        # print("Grid Summary:")
        # print(
        #     f"  User Requested: L1={self.user_L1:.1f}, L2={self.user_L2:.1f}, "
        #     f"N1={self.user_N1}, N2={self.user_N2}, dx={self.user_dx:.4f}, dy={self.user_dy:.4f}"
        # )
        # print(f"  Computed:       L1={comp_L1:.1f}, L2={comp_L2:.1f}, N1={comp_N1}, N2={comp_N2}, d={comp_d:.4f}")
        # print("-" * 30)

        tolerance = 1e-9
        if not (comp_L1 >= L1_target - tolerance and comp_L2 >= L2_target - tolerance):
            raise RuntimeError(
                "Internal calculation error: Final computational grid does not meet size requirements. "
                f"Needed ({L1_target:.4f}, {L2_target:.4f}), got ({comp_L1:.4f}, {comp_L2:.4f})"
            )

        return comp_L1, comp_L2, comp_N1, comp_N2

    def _compute_c(self):
        """
        TODO:
        """
        L_2d = self.L_2d
        z_i = self.z_i

        def integrand(kappa):
            denom1 = (L_2d**-2 + kappa**2) ** (7 / 3)
            denom2 = 1.0 + (kappa * z_i) ** 2
            denominator = denom1 * denom2

            shape_kappa = (kappa**3) / denominator

            return shape_kappa

        return self.sigma2 / (2 * integrate.quad(integrand, 0, np.inf)[0])

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
        """
        Numba-accelerated helper function for the generation of the low-frequency fluctuation
        fields on the computational grid.

        Parameters
        ----------
        TODO:


        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The u1 and u2 components of the velocity field on the internal, computational grid.
        """
        k_mag = np.sqrt(k1**2 + k2**2)
        kappa = np.sqrt(2 * ((k1 * np.cos(psi)) ** 2 + (k2 * np.sin(psi)) ** 2))

        phi_ = np.empty_like(k_mag)

        grid_scale = (2 * np.pi) / np.sqrt(max(dx * dy, 1e-16))
        # grid_scale = np.pi / np.sqrt(max(dx * dy, 1e-16))

        for i in numba.prange(N1):
            for j in numba.prange(N2):
                # Avoid division by zero with kappa and k
                _kappa = max(kappa[i, j], 1e-10)
                _k = k_mag[i, j]

                if _k < 1e-10:
                    phi_[i, j] = 0.0
                    continue

                else:
                    energy = c * _kappa**3 / ((L_2d**-2 + _kappa**2) ** (7 / 3))
                    energy /= 1 + (_kappa * z_i) ** 2

                    # NOTE: k**-3 is due to formula with pi * k in denominator; then, k**-2
                    #       comes out of the factorization of Q below.
                    phi_[i, j] = np.sqrt(energy / (np.pi * _k**3)) * grid_scale

        Q1 = 1j * phi_ * k2
        Q2 = 1j * phi_ * (-1 * k1)

        if np.isnan(Q1).any() or np.isnan(Q2).any():
            pass

        return Q1 * noise_hat, Q2 * noise_hat

    def _extract_field_portion(self, field: np.ndarray) -> np.ndarray:
        """
        Extracts the portion of the field that matches the user's requested dimensions (N1, N2).
        Assumes the extraction is centered within the larger computed field.

        Parameters
        ----------
        field : np.ndarray
            The computed field on the potentially larger computational grid.

        Returns
        -------
        np.ndarray
            The extracted portion of the field matching the user's requested dimensions.
        """
        # Get user-requested number of points
        nx_requested = self.user_N1
        ny_requested = self.user_N2

        # Get computational grid dimensions
        nx_computed = self.comp_N1
        ny_computed = self.comp_N2

        # Check if extraction is necessary
        if nx_computed == nx_requested and ny_computed == ny_requested:
            return field

        # Calculate indices for centered extraction
        start_x = (nx_computed - nx_requested) // 2
        start_y = (ny_computed - ny_requested) // 2

        end_x = start_x + nx_requested
        end_y = start_y + ny_requested

        # Extract the portion
        extracted_field = field[start_x:end_x, start_y:end_y]

        # Verify the extracted field has the correct dimensions
        if extracted_field.shape != (nx_requested, ny_requested):
            raise ValueError(
                f"Extracted field shape {extracted_field.shape} doesn't match "
                f"requested shape {(nx_requested, ny_requested)}"
            )

        return extracted_field

    def generate(self):
        """
        Generates the velocity field on the computational grid, stores the full and
        extracted fields, and returns the portion matching the user's requested dimensions.

        TODO: How to correctly "denote" this in the docstring?
        Stores:
            self.u1_full, self.u2_full : Fields on the full computational grid.
            self.u1, self.u2 : Fields extracted to user's requested dimensions.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The u1 and u2 components of the velocity field, extracted to match user's N1, N2.
        """
        # Generate noise on the computational grid
        noise = np.random.normal(0, 1, size=(self.comp_N1, self.comp_N2))
        noise_hat = np.fft.fft2(noise)

        # Generate frequency components using computational grid parameters
        u1_freq, u2_freq = self._generate_numba_helper(
            self.k1,  # Based on comp grid
            self.k2,  # Based on comp grid
            self.c,
            self.L_2d,
            self.psi,
            self.z_i,
            self.comp_dx,  # Computational spacing
            self.comp_dy,  # Computational spacing
            self.comp_N1,  # Computational points
            self.comp_N2,  # Computational points
            noise_hat,
        )

        # Convert full field to physical space
        u1_full = np.real(np.fft.ifft2(u1_freq))
        u2_full = np.real(np.fft.ifft2(u2_freq))

        # Store the full fields
        self.u1_full = u1_full
        self.u2_full = u2_full

        # Extract the portion matching user's N1, N2
        u1 = self._extract_field_portion(u1_full)
        u2 = self._extract_field_portion(u2_full)

        # Store the *extracted* fields
        self.u1 = u1
        self.u2 = u2

        return u1, u2

    # ------------------------------------------------------------------------------------------------ #
    # Below are member functions for interpolating the low-frequency fields to the 3d grid

    def interp_slice(self, x_coords_target: np.ndarray, y_coords_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Given the 1D x and y coordinates defining a target grid slice, interpolates the
        generated low-frequency fields (self.u1, self.u2) onto this target grid.

        Assumes self.u1 and self.u2 (the extracted fields) and self.user_x_coords,
        self.user_y_coords (defining the source grid) exist.

        Parameters
        ----------
        x_coords_target : np.ndarray
            1D array of target x-coordinates for the slice.
        y_coords_target : np.ndarray
            1D array of target y-coordinates for the slice.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Interpolated u1 and u2 fields on the target grid slice (shape matching meshgrid of targets).
        """
        if self.u1 is None or self.u2 is None:
            raise RuntimeError("LowFreqGenerator.generate() must be called before interp_slice()")
        if not hasattr(self, "user_x_coords") or not hasattr(self, "user_y_coords"):
            raise RuntimeError("LowFreqGenerator needs user_x_coords and user_y_coords attributes.")

        interpolator_2d_u1 = RegularGridInterpolator(
            (self.user_x_coords, self.user_y_coords),
            self.u1,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        interpolator_2d_u2 = RegularGridInterpolator(
            (self.user_x_coords, self.user_y_coords),
            self.u2,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        X_target, Y_target = np.meshgrid(x_coords_target, y_coords_target, indexing="ij")

        target_points_xy = np.stack([X_target.ravel(), Y_target.ravel()], axis=-1)

        u1_3d_flat = interpolator_2d_u1(target_points_xy)
        u2_3d_flat = interpolator_2d_u2(target_points_xy)

        u1_3d = u1_3d_flat.reshape(X_target.shape)
        u2_3d = u2_3d_flat.reshape(X_target.shape)

        return u1_3d, u2_3d

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
        """
        Numba-accelerated helper function for the estimation of the 1d spectra F_11 and F_22.

        Parameters
        ----------
        power_u1 : np.ndarray
            The power spectrum of u1.
        power_u2 : np.ndarray
            The power spectrum of u2.
        k1_grid : np.ndarray
            The grid of k1 values.
        k1_pos : np.ndarray
            Positive k1 values.
        scaling_factor : float
            The scaling factor for the spectra.
        k_tol : float
            The tolerance for the k1 values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The 1d spectra F11 and F22.
        """
        F11 = np.empty_like(k1_pos)
        F22 = np.empty_like(k1_pos)
        N1, N2 = k1_grid.shape

        for i in numba.prange(len(k1_pos)):
            k1_val = k1_pos[i]

            summed_power_u1 = 0.0
            summed_power_u2 = 0.0

            for r in range(N1):
                for c in range(N2):
                    if np.abs(k1_grid[r, c] - k1_val) < k_tol:
                        summed_power_u1 += power_u1[r, c]
                        summed_power_u2 += power_u2[r, c]

            F11[i] = summed_power_u1 * scaling_factor
            F22[i] = summed_power_u2 * scaling_factor

        return F11, F22

    def compute_spectrum(self, k_tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimates the 1d spectra F11 and F22 of the velocity fields u1 and u2
        computed on the **computational** grid before extraction.
        """
        if not hasattr(self, "u1") or not hasattr(self, "u2"):
            raise RuntimeError("Call generate() before compute_spectrum()")

        # TODO: Why the fuck is this SO slow all of a sudden? It's more than 2x slower than it was before,
        # .    so it's not only the up-scaled grid/domain for the L x 0.125L case
        u1_fft_extracted = np.fft.fft2(self.u1)
        u2_fft_extracted = np.fft.fft2(self.u2)

        k1_fft_user = 2 * np.pi * np.fft.fftfreq(self.user_N1, self.user_dx)
        k2_fft_user = 2 * np.pi * np.fft.fftfreq(self.user_N2, self.user_dy)
        k1_user, k2_user = np.meshgrid(k1_fft_user, k2_fft_user, indexing="ij")

        k1_pos = np.unique(np.abs(k1_fft_user))
        k1_pos = k1_pos[k1_pos > k_tol]
        k1_pos = np.sort(k1_pos)

        power_u1 = (np.abs(u1_fft_extracted)) ** 2
        power_u2 = (np.abs(u2_fft_extracted)) ** 2

        if np.isnan(power_u1).any() or np.isnan(power_u2).any():
            import warnings

            warnings.warn("NaN detected in power spectra!")

        scaling_factor = self.user_L1 / ((self.user_N1 * self.user_N2) ** 2 * (2 * np.pi))

        F11, F22 = self._compute_spectrum_numba_helper(
            power_u1,
            power_u2,
            k1_user,
            k1_pos,
            scaling_factor,
            k_tol,
        )

        return k1_pos, F11, F22

    # ------------------------------------------------------------------------------------------------ #

    def analytical_spectrum(self, k1_arr: np.ndarray, warn: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the analytical spectrum of the Mann Syed 2d model

        Parameters
        ----------
        k1_arr: np.ndarray
            Wavenumber array

        Returns
        -------
        F11: np.ndarray
            Analytical spectrum of u1
        F22: np.ndarray
            Analytical spectrum of u2
        """
        F11_res_arr = np.zeros_like(k1_arr)
        F11_err_arr = np.zeros_like(k1_arr)
        F22_res_arr = np.zeros_like(k1_arr)
        F22_err_arr = np.zeros_like(k1_arr)

        _psi = self.psi

        _L_2d = self.L_2d
        _z_i = self.z_i
        _c = self.c

        def _E_kappa(k1: float, k2: float) -> float:
            """Calculate E(kappa)"""
            kappa_squared = 2 * ((k1 * np.cos(_psi)) ** 2 + (k2 * np.sin(_psi)) ** 2)
            kappa_squared = max(kappa_squared, 1e-24)
            _kappa = np.sqrt(kappa_squared)

            denom_term_1 = (_L_2d**-2 + kappa_squared) ** (7 / 3)
            denom_term_2 = 1 + kappa_squared * _z_i**2

            if denom_term_1 * denom_term_2 < 1e-30:
                return 0.0
            Ekappa = _c * (_kappa**3) / (denom_term_1 * denom_term_2)
            if not np.isfinite(Ekappa):
                return 0.0
            return Ekappa

        def _integrand11(k2: float, k1: float, eps: float = 1e-20) -> float:
            """
            Integrand for F11
            Uses (E(kappa) / (pi * k)) * (k2^2 / k^2) simplification.
            """
            k_mag_sq = k1**2 + k2**2
            k_mag = np.sqrt(k_mag_sq)

            Ekappa = _E_kappa(k1, k2)

            integrand = (Ekappa / (np.pi * k_mag)) * (k2**2 / k_mag_sq)
            return integrand

        def _integrand22(k2: float, k1: float, eps: float = 1e-20) -> float:
            """
            Integrand for F22
            Uses (E(kappa) / (pi * k)) * (k1^2 / k^2) simplification.
            """
            k_mag_sq = k1**2 + k2**2
            k_mag = np.sqrt(k_mag_sq)

            Ekappa = _E_kappa(k1, k2)

            integrand = (Ekappa / (np.pi * k_mag)) * (k1**2 / k_mag_sq)
            return integrand

        k2_limit = 5
        print(f"Using integration limits for k2: [-{k2_limit:.2e}, {k2_limit:.2e}]")

        for i, k1_val in enumerate(k1_arr):
            try:
                F11_res_arr[i], F11_err_arr[i] = integrate.quad(
                    _integrand11,
                    -k2_limit,
                    k2_limit,
                    args=(k1_val,),
                    limit=100,
                    epsabs=1.49e-08,
                    epsrel=1.49e-08,
                )
                # Check for large error estimate
                if F11_err_arr[i] > 0.1 * abs(F11_res_arr[i]):
                    rel_err = F11_err_arr[i] / F11_res_arr[i]
                    print(f"Warning: High relative error ({rel_err:.1%}) for F11 at k1={k1_val:.4e}")

            except Exception as e:
                print(f"Warning: Integration failed for F11 at k1={k1_val:.4e}: {e}")
                F11_res_arr[i], F11_err_arr[i] = np.nan, np.nan

            try:
                F22_res_arr[i], F22_err_arr[i] = integrate.quad(
                    _integrand22,
                    -k2_limit,
                    k2_limit,
                    args=(k1_val,),
                    limit=100,
                    epsabs=1.49e-08,
                    epsrel=1.49e-08,
                )
                # Check for large error estimate
                if F22_err_arr[i] > 0.1 * abs(F22_res_arr[i]):
                    rel_err = F22_err_arr[i] / F22_res_arr[i]
                    print(f"Warning: High relative error ({rel_err:.1%}) for F22 at k1={k1_val:.4e}")

            except Exception as e:
                print(f"Warning: Integration failed for F22 at k1={k1_val:.4e}: {e}")
                F22_res_arr[i], F22_err_arr[i] = np.nan, np.nan

        if warn:
            print("Max error on F11: ", np.max(F11_err_arr))
            print("Max error on F22: ", np.max(F22_err_arr))

        return F11_res_arr, F22_res_arr

    # ------------------------------------------------------------------------------------------------ #

    def plot_velocity_fields(self) -> None:
        """
        Plots the generated fields u1 and u2 on the user's grid.
        """
        print("=" * 80)
        print("VELOCITY FIELD PLOT")
        print("=" * 80)

        if self.u1 is None or self.u2 is None:
            raise RuntimeError("LowFreqGenerator.generate() must be called before plot_velocity_fields()")

        print("u1 stats")
        print(f"min: {np.min(self.u1)}", f"max: {np.max(self.u1)}")
        print(f"mean: {np.mean(self.u1)}", f"variance: {np.var(self.u1)}")
        print(f"Any nan: {np.isnan(self.u1).any()}")

        print("u2 stats")
        print(f"min: {np.min(self.u2)}", f"max: {np.max(self.u2)}")
        print(f"mean: {np.mean(self.u2)}", f"variance: {np.var(self.u2)}")
        print(f"Any nan: {np.isnan(self.u2).any()}")

        x_km = self.X / 1000
        y_km = self.Y / 1000

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        vmin = min(np.min(self.u1), np.min(self.u2))
        vmax = max(np.max(self.u1), np.max(self.u2))
        vlim = max(abs(vmin), abs(vmax))
        vmin, vmax = -vlim, vlim

        extent = [x_km[0, 0], x_km[-1, -1], y_km[0, 0], y_km[-1, -1]]

        im1 = ax1.imshow(self.u1.T, extent=extent, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label("[m s$^{-1}$]")
        ax1.set_ylabel("y [km]")
        ax1.set_title("(a) u")

        im2 = ax2.imshow(self.u2.T, extent=extent, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label("[m s$^{-1}$]")
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
