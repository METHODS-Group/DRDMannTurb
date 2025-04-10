import concurrent.futures

import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy import integrate
from scipy.integrate import dblquad

"""
Notes:
- I think we can use either the 1d or 2d integration for c.
- We want isotropic grids as usual.
- We basically want to make sure that the domain size is at least 4 x L_2d in each direction.
"""


class analytical_Fij:
    def __init__(self, config: dict):
        self.L_2d = config["L_2d"]
        self.psi = config["psi"]
        self.sigma2 = config["sigma2"]
        self.z_i = config["z_i"]

        # Obtain scaling constant c from integration
        def integrand_c(k):
            denominator1 = (self.L_2d**-2 + k**2) ** (7 / 3)
            denominator2 = 1 + k**2 * self.z_i**2
            return (k**3) / (denominator1 * denominator2)

        c_int = integrate.quad(integrand_c, 0, np.infty)

        self.c = self.sigma2 / c_int[0]

        print(f"Using c (from integration):        {self.c:.6e}")

    def _E_kappa(self, k1: float, k2: float) -> float:
        """Calculate E(kappa)"""
        kappa_squared = 2 * ((k1 * np.cos(self.psi)) ** 2 + (k2 * np.sin(self.psi)) ** 2)
        kappa_squared = max(kappa_squared, 1e-24)
        _kappa = np.sqrt(kappa_squared)

        denom_term_1 = (self.L_2d**-2 + kappa_squared) ** (7 / 3)
        denom_term_2 = 1 + kappa_squared * self.z_i**2

        if denom_term_1 * denom_term_2 < 1e-30:
            return 0.0
        Ekappa = self.c * (_kappa**3) / (denom_term_1 * denom_term_2)
        if not np.isfinite(Ekappa):
            return 0.0
        return Ekappa

    def _integrand11(self, k2: float, k1: float, eps: float = 1e-20) -> float:
        """
        Integrand matching the SHAPE of the original code, but maybe numerically stabler.
        Uses (E(kappa) / (pi * k)) * (k2^2 / k^2) form.
        """
        k_mag_sq = k1**2 + k2**2
        k_mag = np.sqrt(k_mag_sq)

        Ekappa = self._E_kappa(k1, k2)

        integrand = (Ekappa / (np.pi * k_mag)) * (k2**2 / k_mag_sq)
        return integrand

    def _integrand22(self, k2: float, k1: float, eps: float = 1e-20) -> float:
        """
        Integrand matching the SHAPE of the original code.
        Uses (E(kappa) / (pi * k)) * (k1^2 / k^2) form.
        """
        k_mag_sq = k1**2 + k2**2
        k_mag = np.sqrt(k_mag_sq)

        Ekappa = self._E_kappa(k1, k2)

        integrand = (Ekappa / (np.pi * k_mag)) * (k1**2 / k_mag_sq)
        return integrand

    def generate(self, k1_arr):
        """
        Generate F11, F22 over given k1_arr using the ORIGINAL integrand SHAPE
        but with potentially more stable integration settings.
        """
        F11_res_arr = np.zeros_like(k1_arr)
        F11_err_arr = np.zeros_like(k1_arr)
        F22_res_arr = np.zeros_like(k1_arr)
        F22_err_arr = np.zeros_like(k1_arr)

        # Use large, but finite, limits for better numerical stability
        # Choose a limit based on where E(kappa) becomes negligible
        # Example: Limit based on many L_2d or related to z_i if high-k decay is strong
        k2_limit = 5  # Increase this if needed
        print(f"Using integration limits for k2: [-{k2_limit:.2e}, {k2_limit:.2e}]")

        for i, k1_val in enumerate(k1_arr):
            try:
                F11_res_arr[i], F11_err_arr[i] = integrate.quad(
                    self._integrand11,  # Use integrand with original 1/k factor
                    -k2_limit,
                    k2_limit,  # Finite limits
                    args=(k1_val,),
                    limit=100,
                    epsabs=1.49e-08,
                    epsrel=1.49e-08,  # Standard tolerance
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
                    self._integrand22,  # Use integrand with original 1/k factor
                    -k2_limit,
                    k2_limit,  # Finite limits
                    args=(k1_val,),
                    limit=100,
                    epsabs=1.49e-08,
                    epsrel=1.49e-08,  # Standard tolerance
                )
                # Check for large error estimate
                if F22_err_arr[i] > 0.1 * abs(F22_res_arr[i]):
                    rel_err = F22_err_arr[i] / F22_res_arr[i]
                    print(f"Warning: High relative error ({rel_err:.1%}) for F22 at k1={k1_val:.4e}")

            except Exception as e:
                print(f"Warning: Integration failed for F22 at k1={k1_val:.4e}: {e}")
                F22_res_arr[i], F22_err_arr[i] = np.nan, np.nan

        return F11_res_arr, F11_err_arr, F22_res_arr, F22_err_arr


class generator:
    def __init__(self, config):
        # Physical parameters
        self.sigma2 = config["sigma2"]
        self.L_2d = config["L_2d"]
        self.psi = config["psi"]
        self.z_i = config["z_i"]

        self.c = self._compute_c_2d()

        self.L1 = config["L1_factor"] * self.L_2d
        self.L2 = config["L2_factor"] * self.L_2d

        self.N1 = 2 ** config["N1"]
        self.N2 = 2 ** config["N2"]

        if not np.isclose(self.N1 / self.N2, self.L1 / self.L2):
            print("WARNING: It is suggested that dx is approximately equal to dy.")

        self.dx = self.L1 / self.N1
        self.dy = self.L2 / self.N2

        x = np.linspace(0, self.L1, self.N1, endpoint=False)
        y = np.linspace(0, self.L2, self.N2, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")

        self.k1_fft = 2 * np.pi * np.fft.fftfreq(self.N1, self.dx)
        self.k2_fft = 2 * np.pi * np.fft.fftfreq(self.N2, self.dy)
        self.k1, self.k2 = np.meshgrid(self.k1_fft, self.k2_fft, indexing="ij")

        self.config = config

    def _compute_c_1d(self):
        # Obtain scaling constant c from integration
        def integrand_c(k):
            denominator1 = (self.L_2d**-2 + k**2) ** (7 / 3)
            denominator2 = 1 + k**2 * self.z_i**2
            return (k**3) / (denominator1 * denominator2)

        c_int_1d = integrate.quad(integrand_c, 0, np.infty)
        c_1d = self.sigma2 / c_int_1d[0]
        print(f"Using c (from 1D integration): {c_1d:.6e}")  # Note this is printed for info
        return c_1d

    def _compute_c_2d(self):
        """Computes normalization constant c using 2D integration in polar coordinates"""
        print("Calculating 'c' using 2D integral (polar coordinates)...")

        L_2d = self.L_2d
        psi = self.psi
        z_i = self.z_i

        # Define the new integrand: Shape(kappa) / pi
        # Inner integral is over k (0 to inf), outer is over theta (0 to 2pi)
        def polar_integrand(k, theta):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            cos_psi = np.cos(psi)
            sin_psi = np.sin(psi)
            kappa_sq = 2 * (k**2) * ((cos_psi * cos_theta) ** 2 + (sin_psi * sin_theta) ** 2)

            if kappa_sq < 1e-15:
                return 0.0
            kappa = np.sqrt(kappa_sq)

            denom1_base = L_2d**-2 + kappa_sq

            denom1 = denom1_base ** (7 / 3)
            denom2 = 1.0 + kappa_sq * z_i**2

            denominator = denom1 * denom2

            shape_kappa = (kappa**3) / denominator

            integrand_val = shape_kappa / np.pi

            return integrand_val

        limit_factor = 5000
        char_len = L_2d if z_i <= 0 else min(L_2d, z_i)
        if char_len <= 0:
            raise ValueError("Characteristic length scale must be positive.")
        k_upper_limit = limit_factor / char_len
        k_lower_limit = 0.0

        print(f"  Using dblquad limits: k in [{k_lower_limit:.1e}, {k_upper_limit:.1e}], theta in [0, 2*pi]")

        try:
            # Integrate k inner (0 to k_upper_limit), theta outer (0 to 2*pi)
            integral_2d, abserr = dblquad(
                polar_integrand,
                0,
                2 * np.pi,  # Outer integral limits (theta)
                lambda theta: k_lower_limit,  # Inner integral lower limit (k)
                lambda theta: k_upper_limit,  # Inner integral upper limit (k)
                epsabs=1.49e-9,
                epsrel=1.49e-9,  # Maybe slightly tighter tolerance
            )
        except Exception as e:
            print(f"ERROR during polar dblquad: {e}")
            raise

        print(f"  2D Integral (polar) result: {integral_2d:.6e}, Est. Error: {abserr:.2e}")

        if integral_2d <= 1e-15:  # Check if integral is essentially zero or negative
            raise ValueError(f"2D Polar Integration for 'c' failed or invalid result: {integral_2d}")
        # Increase error tolerance slightly, as 2D integration can be tricky
        if abserr > 0.1 * abs(integral_2d):  # Check 10% relative error
            print(f"Warning: High relative error in 2D polar integration for 'c': {abserr/integral_2d:.1%}")

        c_2d = self.sigma2 / integral_2d
        print(f"Using c (from 2D polar integration): {c_2d:.6e}")
        return c_2d

    # ------------------------------------------------------------------------------------------------ #

    @staticmethod
    @numba.njit(parallel=True)
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
        k_mag = np.sqrt(k1**2 + k2**2)
        kappa = np.sqrt(2 * ((k1 * np.cos(psi)) ** 2 + (k2 * np.sin(psi)) ** 2))

        phi_ = np.empty_like(k_mag)

        grid_scale = (2 * np.pi) / np.sqrt(max(dx * dy, 1e-16))

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

    def generate(self):
        # Obtain random noise
        noise = np.random.normal(0, 1, size=(self.N1, self.N2))
        noise_hat = np.fft.fft2(noise)

        u1_freq, u2_freq = self._generate_numba_helper(
            self.k1, self.k2, self.c, self.L_2d, self.psi, self.z_i, self.dx, self.dy, self.N1, self.N2, noise_hat
        )

        u1 = np.real(np.fft.ifft2(u1_freq))
        u2 = np.real(np.fft.ifft2(u2_freq))

        self.u1 = u1
        self.u2 = u2

        return u1, u2

    # ------------------------------------------------------------------------------------------------ #

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
        Numba-accelerated helper function for the estimation
        of the 1d spectra F_11 and F_22.
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

    def compute_spectrum(self, k_tol=1e-9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimates the 1d spectra F11 and F22 of the velocity fields u1 and u2"""

        u1_fft = np.fft.fft2(self.u1)
        u2_fft = np.fft.fft2(self.u2)

        k1_pos = np.unique(np.abs(self.k1_fft))
        k1_pos = k1_pos[k1_pos > k_tol]
        k1_pos = np.sort(k1_pos)

        power_u1 = (np.abs(u1_fft)) ** 2
        power_u2 = (np.abs(u2_fft)) ** 2

        if np.isnan(power_u1).any() or np.isnan(power_u2).any():
            import warnings

            warnings.warn("NaN detected in power spectra!")

        scaling_factor = self.L1 / ((self.N1 * self.N2) ** 2 * np.pi)

        F11, F22 = self._compute_spectrum_numba_helper(power_u1, power_u2, self.k1, k1_pos, scaling_factor, k_tol)

        return k1_pos, F11, F22

    # ------------------------------------------------------------------------------------------------ #

    def analytical_spectrum(self, k1_arr: np.ndarray, warn=False) -> tuple[np.ndarray, np.ndarray]:
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
        Fij_gen = analytical_Fij(self.config)

        F11, F11_err, F22, F22_err = Fij_gen.generate(k1_arr)

        if warn:
            print("Max error on F11: ", np.max(F11_err))
            print("Max error on F22: ", np.max(F22_err))

        return F11, F22

    # ------------------------------------------------------------------------------------------------ #

    def plot_velocity_fields(self):
        print("=" * 80)
        print("VELOCITY FIELD PLOT")
        print("=" * 80)

        # Print statistics for debugging
        print("u1 stats")
        print(f"min: {np.min(self.u1)}", f"max: {np.max(self.u1)}")
        print(f"mean: {np.mean(self.u1)}", f"variance: {np.var(self.u1)}")
        print(f"Any nan: {np.isnan(self.u1).any()}")

        print("u2 stats")
        print(f"min: {np.min(self.u2)}", f"max: {np.max(self.u2)}")
        print(f"mean: {np.mean(self.u2)}", f"variance: {np.var(self.u2)}")
        print(f"Any nan: {np.isnan(self.u2).any()}")

        # Convert coordinates to kilometers
        x_km = self.X / 1000
        y_km = self.Y / 1000

        # Create figure with two subplots stacked vertically
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Calculate common vmin/vmax for both plots to share the same scale
        vmin = min(np.min(self.u1), np.min(self.u2))
        vmax = max(np.max(self.u1), np.max(self.u2))
        # Make symmetric around zero
        vlim = max(abs(vmin), abs(vmax))
        vmin, vmax = -vlim, vlim

        extent = [x_km[0, 0], x_km[-1, -1], y_km[0, 0], y_km[-1, -1]]

        # Plot u1 (longitudinal component)
        im1 = ax1.imshow(self.u1.T, extent=extent, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label("[m s$^{-1}$]")
        ax1.set_ylabel("y [km]")
        ax1.set_title("(a) u")

        # Plot u2 (transverse component)
        im2 = ax2.imshow(self.u2.T, extent=extent, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label("[m s$^{-1}$]")
        ax2.set_xlabel("x [km]")
        ax2.set_ylabel("y [km]")
        ax2.set_title("(b) v")

        # Adjust layout
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------------------------------------ #
def _run_single_mesh(exponent, config, num_realizations=5):
    """
    Complete a single run with a config and exponent, return
    exponent itself (to be used for sorting), the u1 norm, and
    the u2 norm
    """
    local_config = config.copy()
    local_config["N1"] = exponent
    local_config["N2"] = exponent

    gen = generator(local_config)
    u1 = np.zeros_like(gen.k1)
    u2 = np.zeros_like(gen.k2)

    for _ in range(num_realizations):
        curr_u1, curr_u2 = gen.generate()
        u1 += curr_u1
        u2 += curr_u2

    u1 /= num_realizations
    u2 /= num_realizations

    # Calculate the discrete approximation of the integral of u^2 dA
    u1_norm = np.sum(u1**2) * gen.dx * gen.dy
    u2_norm = np.sum(u2**2) * gen.dx * gen.dy

    u1_var = np.var(u1)
    u2_var = np.var(u2)

    print(f"Completed mesh size 2^{exponent}")
    print(f"\tu1_var: {u1_var}, u2_var: {u2_var}")
    print(f"\tu1 mean: {np.mean(u1)}, u2 mean: {np.mean(u2)}")
    return exponent, u1_norm, u2_norm, u1_var, u2_var


def mesh_independence_study(low=4, high=12):
    """
    Compute norm of field over several grid sizes.
    """
    cfg_mesh_base = {
        "sigma2": 2.0,
        "L_2d": 15_000.0,
        "psi": np.deg2rad(45.0),
        "z_i": 500.0,
        "L1_factor": 4,
        "L2_factor": 4,
        "N1": 10,
        "N2": 10,
    }

    exponents = np.arange(low, high + 1)

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_single_mesh, exp, cfg_mesh_base) for exp in exponents]

        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error: {e}")

    if results:
        results.sort(key=lambda x: x[0])
        u1_norms = np.array([r[1] for r in results])
        u2_norms = np.array([r[2] for r in results])
        u1_vars = np.array([r[3] for r in results])
        u2_vars = np.array([r[4] for r in results])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot absolute norms
        ax1.plot(exponents, u1_norms, label="u1")
        ax1.plot(exponents, u2_norms, label="u2")
        ax1.set_title("Norm squared times volume element")
        ax1.legend()

        # Plot relative changes
        ax2.semilogy(exponents, u1_vars, label="u1")
        ax2.semilogy(exponents, u2_vars, label="u2")
        ax2.set_title("Variance of u1 and u2")
        ax2.legend()

        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------------------------------------ #


def _run_single_domain_size(domain_factor, config, num_realizations=5):
    """
    Complete a single run with a config and domain size factor, return
    the factor itself (to be used for sorting), the u1 norm, the u2 norm,
    and variances
    """
    local_config = config.copy()
    local_config["L1_factor"] = domain_factor
    local_config["L2_factor"] = domain_factor

    gen = generator(local_config)
    u1 = np.zeros_like(gen.k1)
    u2 = np.zeros_like(gen.k2)

    for _ in range(num_realizations):
        curr_u1, curr_u2 = gen.generate()
        u1 += curr_u1
        u2 += curr_u2

    u1 /= num_realizations
    u2 /= num_realizations

    # Calculate physical domain size
    domain_size = local_config["L_2d"] * domain_factor

    u1_norm = np.sum(u1**2) * gen.dx * gen.dy
    u2_norm = np.sum(u2**2) * gen.dx * gen.dy

    u1_var = np.var(u1)
    u2_var = np.var(u2)

    print(f"Completed domain size {domain_size/1000:.1f} km (factor {domain_factor})")
    print(f"\tu1_var: {u1_var}, u2_var: {u2_var}")
    print(f"\tu1 mean: {np.mean(u1)}, u2 mean: {np.mean(u2)}")
    return domain_factor, u1_norm, u2_norm, u1_var, u2_var, domain_size


def domain_size_study(factors=None):
    """
    Compute norm and variance of field over several domain sizes.
    """
    if factors is None:
        factors = np.arange(1, 17)

    cfg_base = {
        "sigma2": 2.0,
        "L_2d": 15_000.0,
        "psi": np.deg2rad(45.0),
        "z_i": 500.0,
        "L1_factor": 4,  # Will be overridden
        "L2_factor": 4,  # Will be overridden
        "N1": 9,
        "N2": 9,
    }

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_single_domain_size, factor, cfg_base) for factor in factors]

        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error: {e}")

    if results:
        results.sort(key=lambda x: x[0])
        domain_sizes = np.array([r[5] / 1000 for r in results])  # Convert to km
        u1_norms = np.array([r[1] for r in results])
        u2_norms = np.array([r[2] for r in results])
        u1_vars = np.array([r[3] for r in results])
        u2_vars = np.array([r[4] for r in results])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot norms
        ax1.plot(domain_sizes, u1_norms, "o-", label="u1")
        ax1.plot(domain_sizes, u2_norms, "o-", label="u2")
        ax1.set_xlabel("Domain size (km)")
        ax1.set_title("Norm squared times volume element")
        ax1.legend()

        # Plot variances
        ax2.plot(domain_sizes, u1_vars, "o-", label="u1")
        ax2.plot(domain_sizes, u2_vars, "o-", label="u2")
        ax2.set_xlabel("Domain size (km)")
        ax2.set_title("Variance of u1 and u2")
        ax2.axhline(y=cfg_base["sigma2"], color="r", linestyle="--", label="Target σ²")
        ax2.legend()

        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------------------------------------ #


def plot_spectra_comparison(gen: generator):
    """
    Computes and plots numerical vs analytical spectra F11 and F22.
    Plots k1*F vs k1*L_2d on log-log axes.

    Parameters
    ----------
    gen : generator
        An instance of the generator class that has already run gen.generate().
    """
    print("=" * 80)
    print("SPECTRA COMPARISON PLOT")
    print("=" * 80)

    # Compute numerical spectrum
    k1_pos, F11_numerical, F22_numerical = gen.compute_spectrum()

    # Compute analytical spectrum
    F11_analytical, F22_analytical = gen.analytical_spectrum(k1_pos)

    # Non-dimensional wavenumber
    k1_L_2d = k1_pos * gen.L_2d

    # Pre-multiply spectra by k1
    k1_F11_numerical = k1_pos * F11_numerical
    k1_F11_analytical = k1_pos * F11_analytical
    k1_F22_numerical = k1_pos * F22_numerical
    k1_F22_analytical = k1_pos * F22_analytical

    # Print variances for comparison
    dk1 = k1_pos[1] - k1_pos[0] if len(k1_pos) > 1 else 0
    var_u1_from_spectrum = np.sum(F11_numerical) * dk1 if dk1 > 0 else 0
    var_u2_from_spectrum = np.sum(F22_numerical) * dk1 if dk1 > 0 else 0
    var_u1_actual = np.var(gen.u1)
    var_u2_actual = np.var(gen.u2)

    print(f"Variance u1 (Actual)      : {var_u1_actual:.4f}")
    print(f"Variance u1 (From F11_num): {var_u1_from_spectrum:.4f}")
    print(f"Variance u2 (Actual)      : {var_u2_actual:.4f}")
    print(f"Variance u2 (From F22_num): {var_u2_from_spectrum:.4f}")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # F11 spectrum
    ax1.loglog(k1_L_2d, k1_F11_numerical, "bo-", label="Numerical k1*F11", markersize=3, linewidth=1)
    ax1.loglog(k1_L_2d, k1_F11_analytical, "k--", label="Analytical k1*F11")
    ax1.set_ylabel("$k_1 F_{11}$ [m$^2$/s$^2$]")
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()
    ax1.set_title("Spectra Comparison")

    # F22 spectrum
    ax2.loglog(k1_L_2d, k1_F22_numerical, "ro-", label="Numerical k1*F22", markersize=3, linewidth=1)
    ax2.loglog(k1_L_2d, k1_F22_analytical, "k--", label="Analytical k1*F22")
    ax2.set_xlabel("$k_1 L_{2d}$")
    ax2.set_ylabel("$k_1 F_{22}$ [m$^2$/s$^2$]")
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def recreate_fig2(gen_a, gen_b, num_realizations=10, do_plot=True):
    """
    Computes and plots numerical vs analytical spectra F11 and F22.

    Plots k1*F vs k1*L_2d on semilogx axes, recreating figure 2 from the Mann-Syed paper
    """
    print("=" * 80)
    print("RECREATE FIGURE 2")
    print("=" * 80)
    # --- Analytical / Target ---
    k1_a_custom = np.logspace(-1, 3, 500) / gen_a.L_2d
    F11_a_target, F22_a_target = gen_a.analytical_spectrum(k1_a_custom)
    k1_b_custom = np.logspace(0.5, 4.5, 500) / gen_b.L_2d
    F11_b_target, F22_b_target = gen_b.analytical_spectrum(k1_b_custom)

    # --- Numerical (Averaged) ---
    f11_a_list = []
    f22_a_list = []
    f11_b_list = []
    f22_b_list = []
    k1_a_pos = None  # Get k1_pos only once
    k1_b_pos = None  # Get k1_pos only once

    print(f"Running {num_realizations} realizations for numerical spectrum...")
    for i in range(num_realizations):
        print(f"  Realization {i+1}/{num_realizations}")
        # Generate new random fields for this realization
        gen_a.generate()
        gen_b.generate()

        # Compute spectrum for this realization
        _k1_a_pos, _F11_a, _F22_a = gen_a.compute_spectrum()
        _k1_b_pos, _F11_b, _F22_b = gen_b.compute_spectrum()

        # Store results
        f11_a_list.append(_F11_a)
        f22_a_list.append(_F22_a)
        f11_b_list.append(_F11_b)
        f22_b_list.append(_F22_b)

        # Store k1_pos on the first iteration
        if k1_a_pos is None:
            k1_a_pos = _k1_a_pos
        if k1_b_pos is None:
            k1_b_pos = _k1_b_pos
    print("...Averaging complete.")

    # Average the spectra over realizations
    F11_a_numerical = np.mean(np.array(f11_a_list), axis=0)
    F22_a_numerical = np.mean(np.array(f22_a_list), axis=0)
    F11_b_numerical = np.mean(np.array(f11_b_list), axis=0)
    F22_b_numerical = np.mean(np.array(f22_b_list), axis=0)

    # --- Post Processing ---
    # Non-dimensional wavenumber for TARGET/ANALYTICAL plots
    k1_L_2d_a_target = k1_a_custom * gen_a.L_2d
    k1_L_2d_b_target = k1_b_custom * gen_b.L_2d

    # Check if numerical k values were obtained
    if k1_a_pos is None or k1_b_pos is None:
        print("ERROR: Numerical k values not obtained. Cannot proceed.")
        return

    # Non-dimensional wavenumber for NUMERICAL plots (using the 1D k1_pos)
    k1_L_2d_a_numerical = k1_a_pos * gen_a.L_2d
    k1_L_2d_b_numerical = k1_b_pos * gen_b.L_2d

    # Pre-multiply spectra
    k1F11_a_target = k1_a_custom * F11_a_target
    k1F22_a_target = k1_a_custom * F22_a_target
    k1F11_a_numerical = k1_a_pos * F11_a_numerical  # Now uses 1D k1_a_pos
    k1F22_a_numerical = k1_a_pos * F22_a_numerical  # Now uses 1D k1_a_pos

    k1F11_b_target = k1_b_custom * F11_b_target
    k1F22_b_target = k1_b_custom * F22_b_target
    k1F11_b_numerical = k1_b_pos * F11_b_numerical  # Now uses 1D k1_b_pos
    k1F22_b_numerical = k1_b_pos * F22_b_numerical  # Now uses 1D k1_b_pos

    # --- Print Values ---
    print("\n--- CASE (a) ---")
    comparison_k1L_a = [0.1, 1.0, 10.0]

    for k1L_val in comparison_k1L_a:
        print(f"  Comparison near k1*L_2d = {k1L_val:.1f}")
        # Target
        idx_target = np.argmin(np.abs(k1_L_2d_a_target - k1L_val))
        y_val = k1F11_a_target[idx_target]
        x_val = k1_L_2d_a_target[idx_target]
        print(f"    Target    (k1L={x_val:.3f}): k1F11={y_val:.4f}, k1F22={k1F22_a_target[idx_target]:.4f}")
        # Numerical
        idx_numerical = np.argmin(np.abs(k1_L_2d_a_numerical - k1L_val))
        # Check if numerical index is valid before accessing
        if idx_numerical < len(k1F11_a_numerical):
            y_val = k1F11_a_numerical[idx_numerical]
            x_val = k1_L_2d_a_numerical[idx_numerical]
            print(f"    Numerical (k1L={x_val:.3f}): k1F11={y_val:.4f}, k1F22={k1F22_a_numerical[idx_numerical]:.4f}")
        else:
            print("    Numerical: Could not find close k1L value.")

    # Peak Values (Case a)
    peak_idx_target_f11 = np.argmax(k1F11_a_target)
    peak_idx_numerical_f11 = np.argmax(k1F11_a_numerical)
    print("  Peak k1F11 (a):")

    y_val = k1F11_a_target[peak_idx_target_f11]
    x_val = k1_L_2d_a_target[peak_idx_target_f11]
    print(f"    Target   : {y_val:.4f} at k1L={x_val:.3f}")
    if len(k1F11_a_numerical) > 0:
        y_val = k1F11_a_numerical[peak_idx_numerical_f11]
        x_val = k1_L_2d_a_numerical[peak_idx_numerical_f11]
        print(f"    Numerical: {y_val:.4f} at k1L={x_val:.3f}")
    else:
        print("    Numerical: No numerical data.")

    peak_idx_target_f22 = np.argmax(k1F22_a_target)
    peak_idx_numerical_f22 = np.argmax(k1F22_a_numerical)
    print("  Peak k1F22 (a):")
    y_val = k1F22_a_target[peak_idx_target_f22]
    x_val = k1_L_2d_a_target[peak_idx_target_f22]
    print(f"    Target   : {y_val:.4f} at k1L={x_val:.3f}")

    if len(k1F22_a_numerical) > 0:
        y_val = k1F22_a_numerical[peak_idx_numerical_f22]
        x_val = k1_L_2d_a_numerical[peak_idx_numerical_f22]
        print(f"    Numerical: {y_val:.4f} at k1L={x_val:.3f}")
    else:
        print("    Numerical: No numerical data.")

    print("\n--- CASE (b) ---")
    comparison_k1L_b = [10.0, 100.0, 1000.0]

    for k1L_val in comparison_k1L_b:
        print(f"  Comparison near k1*L_2d = {k1L_val:.1f}")
        # Target
        idx_target = np.argmin(np.abs(k1_L_2d_b_target - k1L_val))
        y_val = k1F11_b_target[idx_target]
        x_val = k1_L_2d_b_target[idx_target]
        print(f"    Target    (k1L={x_val:.3f}): k1F11={y_val:.4f}, k1F22={k1F22_b_target[idx_target]:.4f}")
        # Numerical
        idx_numerical = np.argmin(np.abs(k1_L_2d_b_numerical - k1L_val))
        # Check if numerical index is valid before accessing
        if idx_numerical < len(k1F11_b_numerical):
            y_val = k1F11_b_numerical[idx_numerical]
            x_val = k1_L_2d_b_numerical[idx_numerical]
            print(f"    Numerical (k1L={x_val:.3f}): k1F11={y_val:.4f}, k1F22={k1F22_b_numerical[idx_numerical]:.4f}")
        else:
            print("    Numerical: Could not find close k1L value.")

    # Peak Values (Case b) - May not be a clear peak in log scale, but find max value
    peak_idx_target_f11 = np.argmax(k1F11_b_target)
    peak_idx_numerical_f11 = np.argmax(k1F11_b_numerical)
    print("  Peak k1F11 (b):")
    y_val = k1F11_b_target[peak_idx_target_f11]
    x_val = k1_L_2d_b_target[peak_idx_target_f11]
    print(f"    Target   : {y_val:.4f} at k1L={x_val:.3f}")
    if len(k1F11_b_numerical) > 0:
        y_val = k1F11_b_numerical[peak_idx_numerical_f11]
        x_val = k1_L_2d_b_numerical[peak_idx_numerical_f11]
        print(f"    Numerical: {y_val:.4f} at k1L={x_val:.3f}")
    else:
        print("    Numerical: No numerical data.")

    peak_idx_target_f22 = np.argmax(k1F22_b_target)
    peak_idx_numerical_f22 = np.argmax(k1F22_b_numerical)
    print("  Peak k1F22 (b):")
    y_val = k1F22_b_target[peak_idx_target_f22]
    x_val = k1_L_2d_b_target[peak_idx_target_f22]
    print(f"    Target   : {y_val:.4f} at k1L={x_val:.3f}")
    if len(k1F22_b_numerical) > 0:
        y_val = k1F22_b_numerical[peak_idx_numerical_f22]
        x_val = k1_L_2d_b_numerical[peak_idx_numerical_f22]
        print(f"    Numerical: {y_val:.4f} at k1L={x_val:.3f}")
    else:
        print("    Numerical: No numerical data.")

    print("-" * 50)

    if do_plot:
        # --- Plotting ---
        # Create figure with 2 rows and 2 columns
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        ########################################################################
        # Top row: Case (a) 40L_2D × 5L_2D
        # Plot Target
        axs[0, 0].loglog(k1_L_2d_a_target, k1F11_a_target, "k-", label="target")
        # Plot Numerical using its own k values
        axs[0, 0].loglog(k1_L_2d_a_numerical, k1F11_a_numerical, "r-", label="numerical")
        axs[0, 0].set_ylabel("$k_1 F_{11}(k_1)$ [m$^2$s$^{-2}$]")
        axs[0, 0].grid(True, which="both", ls="-", alpha=0.2)
        axs[0, 0].legend()

        ########################################################################
        # F22 spectrum (top right)
        # Plot Target
        axs[0, 1].loglog(k1_L_2d_a_target, k1F22_a_target, "k-", label="target")
        # Plot Numerical using its own k values
        axs[0, 1].loglog(k1_L_2d_a_numerical, k1F22_a_numerical, "r-", label="numerical")
        axs[0, 1].set_ylabel("$k_1 F_{22}(k_1)$ [m$^2$s$^{-2}$]")
        axs[0, 1].grid(True, which="both", ls="-", alpha=0.2)

        ########################################################################
        # Bottom row: Case (b) L_2D × 0.125L_2D
        # Plot Target
        axs[1, 0].loglog(k1_L_2d_b_target, k1F11_b_target, "k-", label="target")
        # Plot Numerical using its own k values
        axs[1, 0].loglog(k1_L_2d_b_numerical, k1F11_b_numerical, "r-", label="numerical")
        axs[1, 0].set_xlabel("$k_1 L_{2D}$ [-]")
        axs[1, 0].set_ylabel("$k_1 F_{11}(k_1)$ [m$^2$s$^{-2}$]")
        axs[1, 0].grid(True, which="both", ls="-", alpha=0.2)

        ########################################################################
        # Plot Target
        axs[1, 1].loglog(k1_L_2d_b_target, k1F22_b_target, "k-", label="target")
        # Plot Numerical using its own k values
        axs[1, 1].loglog(k1_L_2d_b_numerical, k1F22_b_numerical, "r-", label="numerical")
        axs[1, 1].set_xlabel("$k_1 L_{2D}$ [-]")
        axs[1, 1].set_ylabel("$k_1 F_{22}(k_1)$ [m$^2$s$^{-2}$]")
        axs[1, 1].grid(True, which="both", ls="-", alpha=0.2)

        ########################################################################
        fig.text(0.5, 0.98, "(a) $40L_{2D} \\times 5L_{2D}$", ha="center", va="top")
        fig.text(0.5, 0.48, "(b) $L_{2D} \\times 0.125L_{2D}$", ha="center", va="top")

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.3)
        plt.show()


def length_AND_grid_size_study(base_config, do_plot=False, num_realizations=10):
    config = base_config.copy()

    grid_exponents = [7, 8, 9, 10, 11]
    domain_factors = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    print(f"Grid exponents: {grid_exponents}")
    print(f"Domain factors: {domain_factors}")
    print(f"Num Realizations per Combo: {num_realizations}")

    n_exp = len(grid_exponents)
    n_factors = len(domain_factors)

    total_combinations = n_exp * n_factors

    u1_vars_avg = np.zeros((n_exp, n_factors))  # Store average variance
    u2_vars_avg = np.zeros((n_exp, n_factors))  # Store average variance

    current_combo = 0

    for i, exp in enumerate(grid_exponents):
        for j, factor in enumerate(domain_factors):
            current_combo += 1
            grid_size = 2**exp

            print(f"\n{'-'*60}")
            print(
                f"Combination {current_combo}/{total_combinations}: "
                + f"Grid={grid_size}x{grid_size}, Domain factor={factor}"
            )

            config["N1"] = exp
            config["N2"] = exp
            config["L1_factor"] = factor
            config["L2_factor"] = factor

            gen = generator(config)  # Create generator instance

            # Calculate variance for each realization and average the variance values
            realization_u1_vars = []
            realization_u2_vars = []
            for r_idx in range(num_realizations):
                u1_real, u2_real = gen.generate()
                realization_u1_vars.append(np.var(u1_real))
                realization_u2_vars.append(np.var(u2_real))

            # Average the variances
            u1_vars_avg[i, j] = np.mean(realization_u1_vars)
            u2_vars_avg[i, j] = np.mean(realization_u2_vars)

            print(f"\t Avg u1 variance: {u1_vars_avg[i,j]:.6f}")
            print(f"\t Avg u2 variance: {u2_vars_avg[i,j]:.6f}")
            print(f"\t Avg Total variance: {u1_vars_avg[i,j] + u2_vars_avg[i,j]:.6f}")

    figs = []

    def create_heatmap(data, title, cmap="viridis", logscale=False):
        fig, ax = plt.subplots(figsize=(10, 8))

        x_labels = [str(f) for f in domain_factors]
        y_labels = [f"2^{e} ({2**e})" for e in grid_exponents]

        finite_data = np.isfinite(data)
        if logscale and np.any(data[finite_data] <= 0):
            print(f"Warning: Non-positive values encountered in '{title}'. Cannot use log scale.")
            logscale = False

        plot_data = np.log10(data) if logscale else data

        im = ax.imshow(plot_data, cmap=cmap, aspect="auto")

        cbar = ax.figure.colorbar(im, ax=ax)

        if logscale:
            cbar.set_label(f"{title} (log10 scale)")
        else:
            cbar.set_label(title)

        ax.set_xticks(np.arange(len(domain_factors)))
        ax.set_yticks(np.arange(len(grid_exponents)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Domain Factor (L/L_2d)")
        ax.set_ylabel("Grid Size (N)")

        cmap_obj = plt.get_cmap(cmap)
        valid_plot_data = plot_data[np.isfinite(plot_data)]
        norm_obj = plt.Normalize(
            vmin=np.min(valid_plot_data) if len(valid_plot_data) > 0 else 0,
            vmax=np.max(valid_plot_data) if len(valid_plot_data) > 0 else 1,
        )

        for r in range(len(grid_exponents)):
            for c_idx in range(len(domain_factors)):
                val = data[r, c_idx]
                plot_val = plot_data[r, c_idx]

                if not np.isfinite(val) or not np.isfinite(plot_val):
                    continue

                text: str = ""
                if logscale:
                    text = f"{val:.2e}"
                elif abs(val) < 0.001 and val != 0:
                    text = f"{val:.2e}"
                else:
                    text = f"{val:.3f}"

                bg_color = cmap_obj(norm_obj(plot_val))
                luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                text_color = "white" if luminance < 0.5 else "black"

                ax.text(c_idx, r, text, ha="center", va="center", color=text_color, fontsize=8)

        ax.set_title(title)
        plt.tight_layout()
        return fig, im

    # Create a heatmap of the variances
    if do_plot:
        total_vars_avg = u1_vars_avg + u2_vars_avg
        target_sigma2 = base_config.get("sigma2", 1.0)
        vmin_var = 0
        vmax_var = target_sigma2 * 1.5  # Keep color range, adjust if needed
        cmap_var = "RdBu_r"

        fig1, im1 = create_heatmap(u1_vars_avg, "Average u1 variance", cmap=cmap_var, logscale=False)
        if im1:
            im1.set_clim(vmin_var, vmax_var)

        fig2, im2 = create_heatmap(u2_vars_avg, "Average u2 variance", cmap=cmap_var, logscale=False)
        if im2:
            im2.set_clim(vmin_var, vmax_var)

        fig_total, im_total = create_heatmap(
            total_vars_avg, f"Average Total variance (Target={target_sigma2:.2f})", cmap=cmap_var, logscale=False
        )
        if im_total:
            im_total.set_clim(vmin_var, vmax_var)

        figs.extend([fig1, fig2, fig_total])

        for fig in figs:
            if fig:
                plt.figure(fig.number)
                plt.show()

    return


def rectangular_domain_study(base_config, num_realizations=10, do_plot=True):
    """
    Same thing as above but we want to try large rectangles wrt. L_2d
    """

    L1_L2_factor_pairs = [
        (16, 8),
        (16, 4),
        (8, 4),
        (8, 2),
        (4, 2),
        (2, 4),
        (2, 8),
        (4, 8),
        (4, 16),
        (8, 16),
    ]

    N1_N2_pairs = [
        (11, 10),
        (12, 10),
        (11, 10),
        (12, 10),
        (11, 10),
        (10, 11),
        (10, 12),
        (10, 11),
        (10, 12),
        (10, 11),
    ]

    assert len(L1_L2_factor_pairs) == len(N1_N2_pairs)

    for L1_L2_factor, N1_N2_pair in zip(L1_L2_factor_pairs, N1_N2_pairs):
        print(f"{'='*60}")
        print(f"L1_L2_factor: {L1_L2_factor}")
        print(f"N1_N2_pair: {N1_N2_pair}")

        local_config = base_config.copy()
        local_config["L1_factor"] = L1_L2_factor[0]
        local_config["L2_factor"] = L1_L2_factor[1]
        local_config["N1"] = N1_N2_pair[0]
        local_config["N2"] = N1_N2_pair[1]

        gen = generator(local_config)
        avg_u1_var = []
        avg_u2_var = []
        avg_total_var = []

        for r in range(num_realizations):
            u1, u2 = gen.generate()
            avg_u1_var.append(np.var(u1))
            avg_u2_var.append(np.var(u2))
            avg_total_var.append(np.var(u1) + np.var(u2))

        print(f"\tAvg var u1: {np.mean(avg_u1_var)}")
        print(f"\tAvg var u2: {np.mean(avg_u2_var)}")
        print(f"\tAvg Total var: {np.mean(avg_total_var)}")
        print(f"\tTarget sigma2: {gen.sigma2}")
        print(f"{'='*60}\n")

    return


def anisotropy_study(base_config, psi_degrees, num_realizations=10, do_plot=True):
    """
    Computes statistics (variances) and plots a representative generated
    velocity field (u1, u2) for different anisotropy angles (psi).
    """
    print("=" * 80)
    print("RUNNING ANISOTROPY STUDY (Plotting Fields)")
    print(f"Psi angles: {psi_degrees} degrees")
    print(f"Num Realizations for Variance: {num_realizations}")
    print("=" * 80)

    n_angles = len(psi_degrees)
    results = {}
    generated_fields = {}  # To store one field per angle for plotting

    # --- Generate fields once per angle for plotting and calculate overall range ---
    print("Generating representative fields for plotting and range calculation...")
    all_u1 = []
    all_u2 = []
    for i, psi_deg in enumerate(psi_degrees):
        local_config = base_config.copy()
        local_config["psi"] = np.deg2rad(psi_deg)
        gen = generator(local_config)
        print(f"  Generating field for psi={psi_deg}...")
        u1_plot, u2_plot = gen.generate()  # Generate one field for plotting
        generated_fields[psi_deg] = (gen.X, gen.Y, u1_plot, u2_plot)  # Store X,Y too
        all_u1.append(u1_plot)
        all_u2.append(u2_plot)

    # Determine common color limits across all fields
    global_min = min(np.min(np.array(all_u1)), np.min(np.array(all_u2)))
    global_max = max(np.max(np.array(all_u1)), np.max(np.array(all_u2)))
    vlim = max(abs(global_min), abs(global_max))
    vmin, vmax = -vlim, vlim
    print(f"Global velocity range for color scale: [{vmin:.2f}, {vmax:.2f}] m/s")

    # --- Setup Figure ---
    fig, axs = None, None
    if do_plot:
        fig, axs = plt.subplots(n_angles, 2, figsize=(9, 3.5 * n_angles), sharex=True, sharey=True)
        fig.suptitle(r"Generated Velocity Fields for different $\psi$ angles", fontsize=14)
    elif not do_plot:
        print("Plotting disabled.")

    # --- Loop through angles again for variance calculation and plotting ---
    for i, psi_deg in enumerate(psi_degrees):
        print(f"\n--- Processing Psi = {psi_deg} degrees (Variance Calculation) ---")
        local_config = base_config.copy()
        local_config["psi"] = np.deg2rad(psi_deg)
        gen = generator(local_config)

        # --- Calculate Average Variances ---
        u1_vars = []
        u2_vars = []
        print(f"  Running {num_realizations} realizations for variance...")
        for r in range(num_realizations):
            u1_realization, u2_realization = gen.generate()
            u1_vars.append(np.var(u1_realization))
            u2_vars.append(np.var(u2_realization))
        print("  ...Realizations complete.")

        avg_u1_var = np.mean(u1_vars)
        avg_u2_var = np.mean(u2_vars)
        avg_total_var = avg_u1_var + avg_u2_var

        print(f"  Average var(u1)    : {avg_u1_var:.4f}")
        print(f"  Average var(u2)    : {avg_u2_var:.4f}")
        print(f"  Average var(Total) : {avg_total_var:.4f} (Target sigma2={gen.sigma2})")

        results[psi_deg] = {"avg_u1_var": avg_u1_var, "avg_u2_var": avg_u2_var, "avg_total_var": avg_total_var}

        # --- Plotting (if enabled) ---
        if do_plot and axs is not None:
            X, Y, u1_plot, u2_plot = generated_fields[psi_deg]
            x_km = X / 1000
            y_km = Y / 1000
            extent = [x_km[0, 0], x_km[-1, -1], y_km[0, 0], y_km[-1, -1]]
            row_axs = axs[i]

            # Plot u1 (left column)
            imshow_kwargs = dict(extent=extent, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")

            row_axs[0].imshow(u1_plot.T, **imshow_kwargs)
            row_axs[0].set_ylabel(f"$\\psi={psi_deg}^\\circ$\ny [km]")
            if i == 0:
                row_axs[0].set_title("u1 field")
            if i == n_angles - 1:
                row_axs[0].set_xlabel("x [km]")

            # Plot u2 (right column)
            row_axs[1].imshow(u2_plot.T, **imshow_kwargs)
            if i == 0:
                row_axs[1].set_title("u2 field")
            if i == n_angles - 1:
                row_axs[1].set_xlabel("x [km]")

            # Add a single colorbar for the whole figure
            if i == n_angles - 1:  # Add after the last row is plotted
                fig.colorbar(row_axs[1].images[0], ax=axs[:, 1], shrink=0.8, label="Velocity [m/s]")

    print("\n...Processing complete.")

    if do_plot and fig is not None:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return results


# ------------------------------------------------------------------------------------------------ #

if __name__ == "__main__":
    ###############################################
    # NOTE: Check norm against several grid resolutions. Always square

    # mesh_independence_study()

    ###############################################
    # NOTE: Check norm against domain sizes. Also always square

    # domain_size_study()

    ###############################################
    # Recreate figure 3
    cfg_fig3 = {
        "sigma2": 0.6,
        "L_2d": 15_000.0,
        "psi": np.deg2rad(43.0),
        "z_i": 500.0,
        "L1_factor": 16,
        "L2_factor": 4,
        "N1": 14,
        "N2": 12,
    }

    # cfg_fig3_sq = {
    #     "sigma2": 0.6,
    #     "L_2d": 15_000.0,
    #     "psi": np.deg2rad(45.0),
    #     "z_i": 500.0,
    #     "L1_factor": 5,
    #     "L2_factor": 5,
    #     "N1": 10,
    #     "N2": 10,
    # }
    # gen = generator(cfg_fig3)
    # gen.generate()
    # gen.plot_velocity_fields()

    cfg_a = {
        "sigma2": 2.0,
        "L_2d": 15_000.0,
        "psi": np.deg2rad(45.0),
        "z_i": 500.0,
        "L1_factor": 40,  # For case (a): 40L_2D × 5L_2D
        "L2_factor": 5,
        "N1": 13,
        "N2": 10,
    }

    cfg_b = {
        "sigma2": 2.0,
        "L_2d": 15_000.0,
        "psi": np.deg2rad(45.0),
        "z_i": 500.0,
        "L1_factor": 1,  # For case (b): L_2D × 0.125L_2D
        "L2_factor": 0.125,
        "N1": 13,
        "N2": 10,
    }

    gen_a = generator(cfg_a)
    gen_b = generator(cfg_b)

    # generate the fields first
    gen_a.generate()
    gen_b.generate()

    # # NOTE: This one attempts to recreate figure 2 as closely as possible.
    recreate_fig2(gen_a, gen_b)

    ##############################################
    # NOTE: Isotropic grid/domain study (psi=45) with updated 'c' calc
    # cfg_iso_study = {
    #     "sigma2": 1.0,
    #     "L_2d": 5_000.0,
    #     "psi": np.deg2rad(45.0),
    #     "z_i": 500.0,
    #     "L1_factor": 4,
    #     "L2_factor": 4,
    #     "N1": 10,
    #     "N2": 10,
    # }
    # print("\n" + "="*80)
    # print("RUNNING ISOTROPIC DOMAIN/GRID STUDY (psi=45) w/ Polar 'c'")
    # print("Target sigma2 =", cfg_iso_study["sigma2"])
    # print("Averaging Variance over Realizations")
    # print("="*80 + "\n")
    # length_AND_grid_size_study(cfg_iso_study, do_plot = True, num_realizations=10)

    rectangular_domain_study(cfg_fig3, num_realizations=10, do_plot=True)
