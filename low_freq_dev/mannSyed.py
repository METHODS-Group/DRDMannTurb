import concurrent.futures

import matplotlib.pyplot as plt
import numba
import numpy as np
import redo_num_int as Fij

"""
- Mesh independence study
- Scale independence study
- Plot
- Match spectrum
"""


class generator:
    def __init__(self, config):
        # Physical parameters
        self.sigma2 = config["sigma2"]
        self.L_2d = config["L_2d"]
        self.psi = config["psi"]
        self.z_i = config["z_i"]

        self.c = (8.0 * self.sigma2) / (9.0 * (self.L_2d ** (2 / 3)))

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

        for i in numba.prange(N1):
            for j in numba.prange(N2):
                # Avoid division by zero with kappa and k
                _kappa = max(kappa[i,j], 1e-10)
                _k = k_mag[i,j]

                if _k < 1e-10:
                    phi_[i, j] = 0.0
                    continue

                else:
                    energy = c * _kappa**3 / ((L_2d**-2 + _kappa**2) ** (7 / 3))
                    energy /= (1 + (_kappa * z_i)**2)

                    # NOTE: k**-3 is due to formula with pi * k in denominator; then, k**-2
                    #       comes out of the factorization of Q below.
                    phi_[i,j] = np.sqrt(energy / (np.pi * _k**3))

        Q1 = 1j * phi_ * k2
        Q2 = 1j * phi_ * (-1 * k1)

        if np.isnan(Q1).any() or np.isnan(Q2).any():
            pass

        return Q1 * noise_hat, Q2 * noise_hat

    def generate(self):
        # Obtain random noise
        noise = np.random.normal(0, 1, size=(self.N1, self.N2))
        # noise_hat = np.fft.fft2(noise)
        noise_hat = np.fft.fft2(noise)

        u1_freq, u2_freq = self._generate_numba_helper(
            self.k1, self.k2, self.c, self.L_2d, self.psi, self.z_i,
            self.dx, self.dy, self.N1, self.N2, noise_hat
        )

        # NOTE: transform below is to control spatial white noise variance
        # transform = 1 if eta_ones else np.sqrt(self.N1 * self.N2)
        # u1 = np.real(np.fft.ifft2(u1_freq) / transform)
        # u2 = np.real(np.fft.ifft2(u2_freq) / transform)

        # Proper normalization that preserves variance across grid sizes

        u1 = np.real(np.fft.ifft2(u1_freq)) * np.sqrt(self.N1 * self.N2)
        u2 = np.real(np.fft.ifft2(u2_freq)) * np.sqrt(self.N1 * self.N2)

        # u1 = np.real(np.fft.ifft2(u1_freq)) / np.sqrt(self.dx * self.dy)
        # u2 = np.real(np.fft.ifft2(u2_freq)) / np.sqrt(self.dx * self.dy)

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

    def compute_spectrum(self, k_tol = 1e-9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        scaling_factor = self.L1 / ((self.N1 * self.N2)**2 * np.pi)

        F11, F22 = self._compute_spectrum_numba_helper(
            power_u1,
            power_u2,
            self.k1,
            k1_pos,
            scaling_factor,
            k_tol
        )

        return k1_pos, F11, F22

    # ------------------------------------------------------------------------------------------------ #

    def analytical_spectrum(self, k1_arr: np.ndarray, warn = False) -> tuple[np.ndarray, np.ndarray]:
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

        Fij_gen = Fij.analytical_Fij(self.config)

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

    u1_norm = np.linalg.norm(u1 * gen.dx * gen.dy) ** 2
    u2_norm = np.linalg.norm(u2 * gen.dx * gen.dy) ** 2

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
    ax1.loglog(k1_L_2d, k1_F11_numerical, 'bo-', label='Numerical k1*F11', markersize=3, linewidth=1)
    ax1.loglog(k1_L_2d, k1_F11_analytical, 'k--', label='Analytical k1*F11')
    ax1.set_ylabel('$k_1 F_{11}$ [m$^2$/s$^2$]')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()
    ax1.set_title('Spectra Comparison')

    # F22 spectrum
    ax2.loglog(k1_L_2d, k1_F22_numerical, 'ro-', label='Numerical k1*F22', markersize=3, linewidth=1)
    ax2.loglog(k1_L_2d, k1_F22_analytical, 'k--', label='Analytical k1*F22')
    ax2.set_xlabel('$k_1 L_{2d}$')
    ax2.set_ylabel('$k_1 F_{22}$ [m$^2$/s$^2$]')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def recreate_fig2(gen_a, gen_b, num_realizations = 10, do_plot = True):
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
    k1_a_pos = None # Get k1_pos only once
    k1_b_pos = None # Get k1_pos only once

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
    k1F11_a_numerical = k1_a_pos * F11_a_numerical # Now uses 1D k1_a_pos
    k1F22_a_numerical = k1_a_pos * F22_a_numerical # Now uses 1D k1_a_pos

    k1F11_b_target = k1_b_custom * F11_b_target
    k1F22_b_target = k1_b_custom * F22_b_target
    k1F11_b_numerical = k1_b_pos * F11_b_numerical # Now uses 1D k1_b_pos
    k1F22_b_numerical = k1_b_pos * F22_b_numerical # Now uses 1D k1_b_pos

    # --- Print Values ---
    print("\n--- CASE (a) ---")
    comparison_k1L_a = [0.1, 1.0, 10.0]

    for k1L_val in comparison_k1L_a:
        print(f"  Comparison near k1*L_2d = {k1L_val:.1f}")
        # Target
        idx_target = np.argmin(np.abs(k1_L_2d_a_target - k1L_val))
        print(f"    Target    (k1L={k1_L_2d_a_target[idx_target]:.3f}): k1F11={k1F11_a_target[idx_target]:.4f}, k1F22={k1F22_a_target[idx_target]:.4f}")
        # Numerical
        idx_numerical = np.argmin(np.abs(k1_L_2d_a_numerical - k1L_val))
        # Check if numerical index is valid before accessing
        if idx_numerical < len(k1F11_a_numerical):
            print(f"    Numerical (k1L={k1_L_2d_a_numerical[idx_numerical]:.3f}): k1F11={k1F11_a_numerical[idx_numerical]:.4f}, k1F22={k1F22_a_numerical[idx_numerical]:.4f}")
        else:
             print("    Numerical: Could not find close k1L value.")

    # Peak Values (Case a)
    peak_idx_target_f11 = np.argmax(k1F11_a_target)
    peak_idx_numerical_f11 = np.argmax(k1F11_a_numerical)
    print("  Peak k1F11 (a):")
    print(f"    Target   : {k1F11_a_target[peak_idx_target_f11]:.4f} at k1L={k1_L_2d_a_target[peak_idx_target_f11]:.3f}")
    if len(k1F11_a_numerical) > 0:
        print(f"    Numerical: {k1F11_a_numerical[peak_idx_numerical_f11]:.4f} at k1L={k1_L_2d_a_numerical[peak_idx_numerical_f11]:.3f}")
    else:
         print("    Numerical: No numerical data.")

    peak_idx_target_f22 = np.argmax(k1F22_a_target)
    peak_idx_numerical_f22 = np.argmax(k1F22_a_numerical)
    print("  Peak k1F22 (a):")
    print(f"    Target   : {k1F22_a_target[peak_idx_target_f22]:.4f} at k1L={k1_L_2d_a_target[peak_idx_target_f22]:.3f}")
    if len(k1F22_a_numerical) > 0:
        print(f"    Numerical: {k1F22_a_numerical[peak_idx_numerical_f22]:.4f} at k1L={k1_L_2d_a_numerical[peak_idx_numerical_f22]:.3f}")
    else:
         print("    Numerical: No numerical data.")

    print("\n--- CASE (b) ---")
    comparison_k1L_b = [10.0, 100.0, 1000.0]

    for k1L_val in comparison_k1L_b:
        print(f"  Comparison near k1*L_2d = {k1L_val:.1f}")
        # Target
        idx_target = np.argmin(np.abs(k1_L_2d_b_target - k1L_val))
        print(f"    Target    (k1L={k1_L_2d_b_target[idx_target]:.3f}): k1F11={k1F11_b_target[idx_target]:.4f}, k1F22={k1F22_b_target[idx_target]:.4f}")
        # Numerical
        idx_numerical = np.argmin(np.abs(k1_L_2d_b_numerical - k1L_val))
        # Check if numerical index is valid before accessing
        if idx_numerical < len(k1F11_b_numerical):
            print(f"    Numerical (k1L={k1_L_2d_b_numerical[idx_numerical]:.3f}): k1F11={k1F11_b_numerical[idx_numerical]:.4f}, k1F22={k1F22_b_numerical[idx_numerical]:.4f}")
        else:
             print("    Numerical: Could not find close k1L value.")

    # Peak Values (Case b) - May not be a clear peak in log scale, but find max value
    peak_idx_target_f11 = np.argmax(k1F11_b_target)
    peak_idx_numerical_f11 = np.argmax(k1F11_b_numerical)
    print("  Peak k1F11 (b):")
    print(f"    Target   : {k1F11_b_target[peak_idx_target_f11]:.4f} at k1L={k1_L_2d_b_target[peak_idx_target_f11]:.3f}")
    if len(k1F11_b_numerical) > 0:
        print(f"    Numerical: {k1F11_b_numerical[peak_idx_numerical_f11]:.4f} at k1L={k1_L_2d_b_numerical[peak_idx_numerical_f11]:.3f}")
    else:
        print("    Numerical: No numerical data.")

    peak_idx_target_f22 = np.argmax(k1F22_b_target)
    peak_idx_numerical_f22 = np.argmax(k1F22_b_numerical)
    print("  Peak k1F22 (b):")
    print(f"    Target   : {k1F22_b_target[peak_idx_target_f22]:.4f} at k1L={k1_L_2d_b_target[peak_idx_target_f22]:.3f}")
    if len(k1F22_b_numerical) > 0:
        print(f"    Numerical: {k1F22_b_numerical[peak_idx_numerical_f22]:.4f} at k1L={k1_L_2d_b_numerical[peak_idx_numerical_f22]:.3f}")
    else:
        print("    Numerical: No numerical data.")

    print("-" * 50)
    # --- End Print Values ---


    if do_plot:
        # --- Plotting ---
        # Create figure with 2 rows and 2 columns
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        ########################################################################
        # Top row: Case (a) 40L_2D × 5L_2D
        # Plot Target
        axs[0, 0].loglog(k1_L_2d_a_target, k1F11_a_target, 'k-', label='target')
        # Plot Numerical using its own k values
        axs[0, 0].loglog(k1_L_2d_a_numerical, k1F11_a_numerical, 'r-', label='numerical')
        axs[0, 0].set_ylabel('$k_1 F_{11}(k_1)$ [m$^2$s$^{-2}$]')
        axs[0, 0].grid(True, which="both", ls="-", alpha=0.2)
        axs[0, 0].legend()

        ########################################################################
        # F22 spectrum (top right)
        # Plot Target
        axs[0, 1].loglog(k1_L_2d_a_target, k1F22_a_target, 'k-', label='target')
        # Plot Numerical using its own k values
        axs[0, 1].loglog(k1_L_2d_a_numerical, k1F22_a_numerical, 'r-', label='numerical')
        axs[0, 1].set_ylabel('$k_1 F_{22}(k_1)$ [m$^2$s$^{-2}$]')
        axs[0, 1].grid(True, which="both", ls="-", alpha=0.2)

        ########################################################################
        # Bottom row: Case (b) L_2D × 0.125L_2D
        # Plot Target
        axs[1, 0].loglog(k1_L_2d_b_target, k1F11_b_target, 'k-', label='target')
        # Plot Numerical using its own k values
        axs[1, 0].loglog(k1_L_2d_b_numerical, k1F11_b_numerical, 'r-', label='numerical')
        axs[1, 0].set_xlabel('$k_1 L_{2D}$ [-]')
        axs[1, 0].set_ylabel('$k_1 F_{11}(k_1)$ [m$^2$s$^{-2}$]')
        axs[1, 0].grid(True, which="both", ls="-", alpha=0.2)

        ########################################################################
        # Plot Target
        axs[1, 1].loglog(k1_L_2d_b_target, k1F22_b_target, 'k-', label='target')
        # Plot Numerical using its own k values
        axs[1, 1].loglog(k1_L_2d_b_numerical, k1F22_b_numerical, 'r-', label='numerical')
        axs[1, 1].set_xlabel('$k_1 L_{2D}$ [-]')
        axs[1, 1].set_ylabel('$k_1 F_{22}(k_1)$ [m$^2$s$^{-2}$]')
        axs[1, 1].grid(True, which="both", ls="-", alpha=0.2)

        ########################################################################
        fig.text(0.5, 0.98, '(a) $40L_{2D} \\times 5L_{2D}$', ha='center', va='top')
        fig.text(0.5, 0.48, '(b) $L_{2D} \\times 0.125L_{2D}$', ha='center', va='top')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.3)
        plt.show()


# ------------------------------------------------------------------------------------------------ #

if __name__ == "__main__":
    ###############################################
    # Mesh independence study

    # mesh_independence_study()

    ###############################################
    # Domain size study

    # domain_size_study()

    ###############################################
    # Recreate figure 3

    cfg_fig3 = {
        "sigma2": 0.6,
        "L_2d": 15_000.0,
        "psi": np.deg2rad(43.0),
        "z_i": 500.0,
        "L1_factor": 4,
        "L2_factor": 1,
        "N1": 14,
        "N2": 12,
    }
    gen = generator(cfg_fig3)
    gen.generate()
    gen.plot_velocity_fields()

    # # Call the new function to plot spectra
    # plot_spectra_comparison(gen)

    ###############################################
    # Recreate Figure 2
    # plot_spectra_comparison()

    # ratio_F11, ratio_F22 = analyze_spectrum_vs_theory()

    # # Example usage in your main:
    # cfg_a = {
    #     "sigma2": 0.6,  # Adjust as needed
    #     "L_2d": 15_000.0,
    #     "psi": np.deg2rad(43.0),
    #     "z_i": 500.0,
    #     "L1_factor": 40,  # For case (a): 40L_2D × 5L_2D
    #     "L2_factor": 5,
    #     "N1": 13,
    #     "N2": 10,
    # }

    # cfg_b = {
    #     "sigma2": 0.6,  # Adjust as needed
    #     "L_2d": 15_000.0,
    #     "psi": np.deg2rad(43.0),
    #     "z_i": 500.0,
    #     "L1_factor": 1,  # For case (b): L_2D × 0.125L_2D
    #     "L2_factor": 0.125,
    #     "N1": 13,
    #     "N2": 10,
    # }

    # gen_a = generator(cfg_a)
    # gen_b = generator(cfg_b)

    # # You may need to generate the fields first
    # gen_a.generate()
    # gen_b.generate()

    # # Call the function to create the plot
    # recreate_fig2(gen_a, gen_b)
