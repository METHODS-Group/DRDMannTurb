import concurrent.futures

import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy

"""
- [X] Mesh independence study
- [X] Scale independence study
- [X] Plot velocity fields
- [X] Match spectrum


1. Why is L_factor = 1000 for F11 not lining up
2. Why does F11 drop off at end of "stitch"
3. Why is it not so noisy at higher frequencies kL?

"""


class generator:
    def __init__(self, config):
        self.c0 = 1.7
        self.L = config["L"]
        self.epsilon = config["epsilon"]

        self.L1 = config["L1_factor"] * self.L
        self.L2 = config["L2_factor"] * self.L

        self.N1 = 2 ** config["N1"]
        self.N2 = 2 ** config["N2"]

        self.dx = self.L1 / self.N1
        self.dy = self.L2 / self.N2

        x = np.linspace(0, self.L1, self.N1, endpoint=False)
        y = np.linspace(0, self.L2, self.N2, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")

        self.k1_fft = 2 * np.pi * np.fft.fftfreq(self.N1, self.dx)
        self.k2_fft = 2 * np.pi * np.fft.fftfreq(self.N2, self.dy)
        self.k1, self.k2 = np.meshgrid(self.k1_fft, self.k2_fft, indexing="ij")

        self.k_mag = np.sqrt(self.k1**2 + self.k2**2)

        self.k1L = self.k1 * self.L
        self.k2L = self.k2 * self.L
        self.kL = self.k_mag * self.L

    def generate(self, eta_ones=False):
        """
        Generate turbulent velocity fields using the von Karman spectrum
        """

        noise_hat: np.ndarray
        if eta_ones:
            noise_hat = np.ones_like(self.k1, dtype=complex)
        else:
            noise = np.random.normal(0, 1, size=(self.N1, self.N2))
            noise_hat = np.fft.fft2(noise)

        u1_freq, u2_freq = self._generate_numba_helper(
            self.k1,
            self.k2,
            self.c0,
            self.epsilon,
            self.L,
            self.N1,
            self.N2,
            noise_hat,
        )

        ###########

        # NOTE: transform_norm was important for spatial white noise for correcting variance
        # transform = 1 if eta_ones else np.sqrt(self.dx * self.dy)
        transform = 1 if eta_ones else np.sqrt(self.N1 * self.N2)

        u1 = np.real(np.fft.ifft2(u1_freq) / transform)
        u2 = np.real(np.fft.ifft2(u2_freq) / transform)

        self.u1 = u1
        self.u2 = u2

        return u1, u2

    @staticmethod
    @numba.njit(parallel=True)
    def _generate_numba_helper(k1, k2, c0, epsilon, L, N1, N2, eta):
        k_mag_sq = k1**2 + k2**2

        #########################################################
        # Compute sqrt(Phi) leading factors
        phi_ = np.empty_like(k_mag_sq)

        for i in numba.prange(N1):
            for j in numba.prange(N2):
                if k_mag_sq[i, j] < 1e-10:
                    k_sq = 1e-10
                else:
                    k_sq = k_mag_sq[i, j]

                phi_[i, j] = (c0 * L ** (17 / 6) * np.cbrt(epsilon)) / (
                    2 * np.sqrt(np.pi) * (k_sq * L**2 + 1) ** (17 / 12)
                )

        #########################################################
        # Compute frequency components
        C1 = 1j * phi_ * k2
        C2 = 1j * phi_ * (-1 * k1)

        # Convolve
        u1_freq_complex = C1 * eta
        u2_freq_complex = C2 * eta

        return u1_freq_complex, u2_freq_complex

    # ------------------------------------------------------------------------------------------------ #

    @staticmethod
    @numba.njit(parallel=True)
    def _compute_spectrum_numba_helper(k1_flat, k1_pos, power_u1_flat, power_u2_flat, L2):
        """
        Numba-accelerated helper function for spectrum computation
        """
        F11 = np.zeros_like(k1_pos)
        F22 = np.zeros_like(k1_pos)

        dk2 = 2 * np.pi / L2

        for i in numba.prange(len(k1_pos)):
            k1_val = k1_pos[i]
            indices = np.where(k1_flat == k1_val)[0]

            if len(indices) > 0:
                F11[i] = np.sum(power_u1_flat[indices]) * dk2
                F22[i] = np.sum(power_u2_flat[indices]) * dk2

        return F11, F22

    def compute_spectrum(self, u1=None, u2=None):
        """
        Numba-accelerated version of compute_spectrum

        Parameters
        ----------
        u1: np.ndarray, optional
            x- or longitudinal component of velocity field
        u2: np.ndarray, optional
            y- or transversal component of velocity field
        """
        if u1 is None and u2 is None:
            u1 = self.u1
            u2 = self.u2

        # Compute FFTs
        u1_fft = np.fft.fft2(u1)
        u2_fft = np.fft.fft2(u2)

        k1_pos = np.abs(self.k1_fft)
        k1_pos = k1_pos[np.argsort(k1_pos)]
        k1_pos = k1_pos[1:-1]

        power_u1 = (np.abs(u1_fft)) ** 2
        power_u2 = (np.abs(u2_fft)) ** 2

        if np.isnan(power_u1).any() or np.isnan(power_u2).any():
            print("WARNING: NaN detected in power spectra!")

        k1_flat = self.k1.flatten()
        power_u1_flat = power_u1.flatten()
        power_u2_flat = power_u2.flatten()

        F11, F22 = self._compute_spectrum_numba_helper(
            k1_flat,
            k1_pos,
            power_u1_flat,
            power_u2_flat,
            self.L2,
        )

        return k1_pos, F11, F22

    # ------------------------------------------------------------------------------------------------ #

    def analytical_spectrum(self, k1_arr):
        """
        Compute the analytical spectrum of the von Karman spectrum

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
        F11_constant = (self.c0**2 * self.epsilon ** (2 / 3) * self.L ** (8 / 3) * scipy.special.gamma(4 / 3)) / (
            8 * np.sqrt(np.pi) * scipy.special.gamma(17 / 6)
        )
        F11 = F11_constant / (1 + (self.L * k1_arr) ** 2) ** (4 / 3)

        F22_constant = (self.c0**2 * self.epsilon ** (2 / 3) * self.L ** (14 / 3) * scipy.special.gamma(7 / 3)) / (
            4 * np.sqrt(np.pi) * scipy.special.gamma(17 / 6)
        )
        F22 = F22_constant * k1_arr**2 / ((1 + (self.L * k1_arr) ** 2) ** (7 / 3))

        return F11, F22


# ------------------------------------------------------------------------------------------------ #


def _run_single_mesh(exponent, config):
    local_config = config.copy()
    local_config["N1"] = exponent
    local_config["N2"] = exponent

    gen = generator(local_config)
    u1, u2 = gen.generate(eta_ones=True)

    u1_norm = np.linalg.norm(u1) * gen.dx * gen.dy
    u2_norm = np.linalg.norm(u2) * gen.dx * gen.dy

    print(f"Completed mesh size 2^{exponent}")
    return exponent, u1_norm, u2_norm


def mesh_independence_study(low=4, high=15):
    """
    Mesh independence study. Runs a number of simulations with different square mesh
    sizes.

    Parameters
    ----------
    low: int, optional
        Lowest exponent to consider
    high: int, optional
        Highest exponent to consider
    """
    print("=" * 80)
    print("MESH INDEPENDENCE STUDY")
    print("=" * 80)

    config = {
        "L": 500,
        "epsilon": 0.01,
        "L1_factor": 2,
        "L2_factor": 2,
        "N1": 9,
        "N2": 9,
    }

    exponents = np.arange(low, high + 1)

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_single_mesh, exp, config) for exp in exponents]

        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error: {e}")

    # Initialize empty lists in case no results were collected
    u1_norms = []
    u2_norms = []

    if results:
        results.sort(key=lambda x: x[0])
        u1_norms = [r[1] for r in results]
        u2_norms = [r[2] for r in results]

    print("\t u1 norm variance: ", np.var(u1_norms))
    print("\t u2 norm variance: ", np.var(u2_norms))

    plt.plot(exponents, u1_norms, label="u1")
    plt.plot(exponents, u2_norms, label="u2")
    plt.title("Mesh Independence Study")
    plt.legend()
    plt.show()


# ------------------------------------------------------------------------------------------------ #


def _run_single_scale(scale, config):
    local_config = config.copy()
    local_config["L1_factor"] = scale
    local_config["L2_factor"] = scale

    gen = generator(local_config)
    u1, u2 = gen.generate(eta_ones=True)

    # L = length scale param
    # L1 = length in x direction
    # L2 = length in y direction

    # L1 = L * L1_factor

    # dx = L1 / N1
    # dy = L2 / N2
    u1_norm = np.linalg.norm(u1) * gen.dx * gen.dy
    u2_norm = np.linalg.norm(u2) * gen.dx * gen.dy

    print(f"Completed scale {scale}")
    return scale, u1_norm, u2_norm


def scale_independence_study(low=0.5, high=40, step=0.5):
    """
    Scale independence study. Run a number of simulations with different scales
    relative to the length scale parameter L.

    Parameters
    ----------
    low: float, optional
        Lowest scale to consider
    high: float, optional
        Highest scale to consider
    step: float, optional
        Step size between scales
    """
    print("=" * 80)
    print("SCALE INDEPENDENCE STUDY")
    print("=" * 80)

    config = {
        "L": 500,
        "epsilon": 0.01,
        "L1_factor": 2,
        "L2_factor": 2,
        "N1": 9,
        "N2": 9,
    }
    factors = np.arange(low, high, step)

    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_single_scale, factor, config) for factor in factors]

        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error: {e}")

    u1_norms = []
    u2_norms = []

    if results:
        results.sort(key=lambda x: x[0])
        u1_norms = [r[1] for r in results]
        u2_norms = [r[2] for r in results]

    print("\t u1 norm variance: ", np.var(u1_norms))
    print("\t u2 norm variance: ", np.var(u2_norms))

    plt.plot(factors, u1_norms, label="u1")
    plt.plot(factors, u2_norms, label="u2")
    plt.title("Scale Independence Study")
    plt.legend()
    plt.show()


# ------------------------------------------------------------------------------------------------ #


def plot_velocity_fields(u1, u2, x_coords, y_coords, title="Velocity Fields"):
    """
    Plot two 2D velocity fields stacked vertically with proper aspect ratio.

    Parameters:
    - u1, u2: 2D numpy arrays of the same shape (velocity components)
    - x_coords, y_coords: 1D arrays with the physical coordinates
    - title: Plot title
    """
    # Increase top margin by adjusting figure size or using subplots_adjust
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Add more space at the top for the title
    plt.subplots_adjust(top=0.85)  # Adjust this value to create more space

    # Calculate aspect ratio to preserve the physical dimensions
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    aspect_ratio = dx / dy

    # Find global min/max for consistent colormap
    vmin = min(np.min(u1), np.min(u2))
    vmax = max(np.max(u1), np.max(u2))

    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

    # Plot u1 (top)
    axs[0].pcolormesh(X, Y, u1, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axs[0].set_aspect(aspect_ratio)
    axs[0].set_ylabel("y [km]")
    axs[0].set_title("(a) u")

    # Plot u2 (bottom)
    im2 = axs[1].pcolormesh(X, Y, u2, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axs[1].set_aspect(aspect_ratio)
    axs[1].set_xlabel("x [km]")
    axs[1].set_ylabel("y [km]")
    axs[1].set_title("(b) v")

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label(r"$[m\,s^{-1}]$")

    # Add main title with more space and larger font
    plt.suptitle(title, y=0.95, fontsize=16)

    # Use tight_layout but preserve the space we created for the title
    plt.tight_layout(rect=[0, 0, 0.9, 0.88])  # Adjust the top value (0.88) to match subplots_adjust

    return fig, axs


def diagnostics(gen: generator, plot: bool = True):
    assert np.any(gen.u1) and np.any(gen.u2), "Generator has not been run"

    print("=" * 80)
    print("DIAGNOSTICS")
    print("=" * 80)

    print(f"u1 min: {np.min(gen.u1)}, u1 max: {np.max(gen.u1)}")
    print(f"u2 min: {np.min(gen.u2)}, u2 max: {np.max(gen.u2)}")
    print(f"u1 mean: {np.mean(gen.u1)}, u1 variance: {np.var(gen.u1)}")
    print(f"u2 mean: {np.mean(gen.u2)}, u2 variance: {np.var(gen.u2)}")

    x_coords = np.linspace(0, gen.L1, gen.N1)
    y_coords = np.linspace(0, gen.L2, gen.N2)

    if plot:
        plot_velocity_fields(gen.u1, gen.u2, x_coords, y_coords, title="Von Karman Velocity field")
        plt.show()


# ------------------------------------------------------------------------------------------------ #
# Spectrum plot


def _compute_single_realization(config: dict, k1_max: float):
    gen = generator(config)
    gen.generate()
    k1_sim, F11, F22 = gen.compute_spectrum(k1_max=k1_max)
    return k1_sim, F11, F22


def plot_spectrum_comparison(config: dict, num_realizations: int = 10):
    """
    Generate velocity fields and plot their spectra compared to analytical spectrum
    on entire domain

    Parameters
    ----------
    config: dict
        Configuration dictionary seen everywhere in this file
    num_realizations: int, optional
        Number of realizations to use in ensemble average for estimated spectrum
    """
    print(f"\n{'='*50}")
    print(f"SPECTRUM COMPARISON - {num_realizations} realizations")
    print(f"{'='*50}")

    gen = generator(config)
    print(f"Generator initialized with N1={gen.N1}, N2={gen.N2}")

    # Custom wavenumber array. Only used for analytical spectrum.
    k1_custom = np.logspace(-3, 3, 1000) / config["L"]
    print("Computing analytical spectrum...")
    F11_analytical, F22_analytical = gen.analytical_spectrum(k1_custom)

    k1_max = np.max(k1_custom)

    results = []
    print(f"\nSpawning {num_realizations} worker processes...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []

        for i in range(num_realizations):
            print(f"\tSubmitting job {i+1}/{num_realizations}...")
            futures.append(executor.submit(_compute_single_realization, config, k1_max))

        print(f"\nAll {num_realizations} jobs submitted, waiting for completion...")

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
                completed += 1
                print(f"\tProcess {completed}/{num_realizations} completed")
            except Exception as e:
                print(f"ERROR in worker process: {e}")

    if not results:
        raise RuntimeError("No results were collected")

    print("\nAll processes completed. Processing results...")

    k1_pos = results[0][0]
    F11_avg = np.zeros_like(k1_pos)
    F22_avg = np.zeros_like(k1_pos)

    for _, F11, F22 in results:
        F11_avg += F11
        F22_avg += F22

    lr = len(results)
    F11_avg /= lr
    F22_avg /= lr

    print("\nCreating plots...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot F11 estimated and analytical
    axs[0].loglog(k1_custom * config["L"], k1_custom * F11_analytical, "k-", label="Analytical F11")
    axs[0].loglog(k1_pos * config["L"], k1_pos * F11_avg, "r--", label="Simulated F11")
    axs[0].set_xlabel(r"$k_1 L$ [-]")
    axs[0].set_ylabel(r"$k_1 F_{11}(k_1)$ [m$^2$s$^{-2}$]")
    axs[0].set_title("F11 spectrum")
    axs[0].grid(True, which="both", ls="-", alpha=0.2)
    axs[0].legend()

    # Plot F22 estimated and analytical
    axs[1].loglog(k1_custom * config["L"], k1_custom * F22_analytical, "k-", label="Analytical F22")
    axs[1].loglog(k1_pos * config["L"], k1_pos * F22_avg, "r--", label="Simulated F22")
    axs[1].set_xlabel(r"$k_1 L$ [-]")
    axs[1].set_ylabel(r"$k_1 F_{22}(k_1)$ [m$^2$s$^{-2}$]")
    axs[1].set_title("F22 spectrum")
    axs[1].grid(True, which="both", ls="-", alpha=0.2)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return fig, axs


# ------------------------------------------------------------------------------------------------ #


def compare_spectra_across_scales(base_config: dict, scale_factors=None, num_realizations=5, autoscale=False):
    """
    Compare analytical spectrum with estimated spectra from multiple domain scale factors.
    Each scale factor will have its own line on the plot.

    Parameters
    ----------
    base_config : dict
        Base configuration dictionary
    scale_factors : list, optional
        List of domain scale factors to test. If None, defaults to [1, 5, 10, 50, 100]
    num_realizations : int, optional
        Number of realizations for each domain size
    autoscale : bool, optional
        Whether to automatically scale the estimated spectra to match the analytical peak

    Returns
    -------
    fig, axs : matplotlib figure and axes
        The generated plot showing analytical and multiple estimated spectra
    """
    if scale_factors is None:
        scale_factors = [1, 5, 10, 50, 100]

    print(f"\n{'='*60}")
    print("COMPARING SPECTRA ACROSS DOMAIN SCALES")
    print(f"Scale factors: {scale_factors}, Realizations per scale: {num_realizations}")
    print(f"Auto-scaling: {'Enabled' if autoscale else 'Disabled'}")
    print(f"{'='*60}")

    # Create custom wavenumber array for analytical spectrum
    k1_custom = np.logspace(-3, 3, 1000) / base_config["L"]

    # Create base generator for computing analytical spectrum
    base_gen = generator(base_config)
    F11_analytical, F22_analytical = base_gen.analytical_spectrum(k1_custom)

    # Store results for each scale factor
    all_results = {}
    scaling_factors = {}  # Store scaling factors for each scale

    # Create a distinct color palette for better visibility
    # Using distinct colors that pop and are easy to distinguish
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]

    # Ensure we have enough colors for all scale factors
    if len(scale_factors) > len(colors):
        from matplotlib.cm import get_cmap

        colors = get_cmap("tab20").colors

    for i, scale in enumerate(scale_factors):
        print(f"\n{'-'*50}")
        print(f"Testing domain scale factor: {scale}")

        # Create modified config with current scale factor
        config = base_config.copy()
        config["L1_factor"] = scale
        config["L2_factor"] = scale  # Keep domain square

        # Initialize generator with this scale
        gen = generator(config)
        print(f"Domain size: {gen.L1} x {gen.L2} units")
        print(f"Resolution: {gen.N1} x {gen.N2} points")
        print(f"Min resolvable k1L: {2*np.pi/gen.L1*gen.L:.4f}")

        # Run simulations and collect results for this scale
        results = []

        for j in range(num_realizations):
            print(f"  Realization {j+1}/{num_realizations}...")
            gen.generate()
            k1_sim, F11, F22 = gen.compute_spectrum()
            results.append((k1_sim, F11, F22))

        # Average the results for this scale
        k1_pos = results[0][0]
        F11_avg = np.zeros_like(k1_pos)
        F22_avg = np.zeros_like(k1_pos)

        for _, F11, F22 in results:
            F11_avg += F11
            F22_avg += F22

        F11_avg /= num_realizations
        F22_avg /= num_realizations

        # Calculate scaling factor even if we're not using it (for reporting)
        max_analytical_F11 = np.max(k1_custom * F11_analytical)
        max_simulated_F11 = np.max(k1_pos * F11_avg)
        auto_scale = max_analytical_F11 / max_simulated_F11 if max_simulated_F11 > 0 else 1
        scaling_factors[scale] = auto_scale

        # Only apply scaling if autoscale is True
        if autoscale:
            print(f"  Auto-scaling factor: {auto_scale:.2e}")
            F11_avg *= auto_scale
            F22_avg *= auto_scale
        else:
            print(f"  Auto-scaling disabled. Scale factor would be: {auto_scale:.2e}")

        F11_analytical_interp = np.interp(k1_pos, k1_custom, F11_analytical)
        F22_analytical_interp = np.interp(k1_pos, k1_custom, F22_analytical)

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_error_F11 = np.abs(F11_avg - F11_analytical_interp) / F11_analytical_interp
            rel_error_F22 = np.abs(F22_avg - F22_analytical_interp) / F22_analytical_interp
            valid_indices = ~np.isnan(rel_error_F11) & ~np.isinf(rel_error_F11)
            if np.any(valid_indices):
                print(f"  Mean relative error F11: {np.mean(rel_error_F11[valid_indices]):.4e}")
            valid_indices = ~np.isnan(rel_error_F22) & ~np.isinf(rel_error_F22)
            if np.any(valid_indices):
                print(f"  Mean relative error F22: {np.mean(rel_error_F22[valid_indices]):.4e}")

        # Store the averaged, (potentially) scaled results for this scale factor
        all_results[scale] = (k1_pos, F11_avg, F22_avg)

    if autoscale:
        fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot analytical spectrum on both spectrum plots
    axs[0].loglog(k1_custom * base_config["L"], k1_custom * F11_analytical, "k-", label="Analytical F11", linewidth=2.5)
    axs[1].loglog(k1_custom * base_config["L"], k1_custom * F22_analytical, "k-", label="Analytical F22", linewidth=2.5)

    # Plot estimated spectra for each scale factor
    for i, scale in enumerate(scale_factors):
        k1_pos, F11_avg, F22_avg = all_results[scale]
        label = f"L factor = {scale}"
        axs[0].loglog(
            k1_pos * base_config["L"], k1_pos * F11_avg, "--", color=colors[i % len(colors)], label=label, linewidth=1.5
        )
        axs[1].loglog(
            k1_pos * base_config["L"], k1_pos * F22_avg, "--", color=colors[i % len(colors)], label=label, linewidth=1.5
        )

    # Set axis labels and titles for spectrum plots
    for i in range(2):
        comp = "11" if i == 0 else "22"
        axs[i].set_xlabel(r"$k_1 L$ [-]")
        axs[i].set_ylabel(r"$k_1 F_{" + comp + r"}(k_1)$ [m$^2$s$^{-2}$]")
        axs[i].set_title(f"F{comp} Spectrum Comparison")
        axs[i].grid(True, which="both", ls="-", alpha=0.2)
        axs[i].legend()
        axs[i].set_xlim(1e-3, 1e3)

    # If autoscale is enabled, add a plot of scale factors vs scaling factors
    if autoscale:
        scales = list(scaling_factors.keys())
        scalings = [scaling_factors[s] for s in scales]

        axs[2].plot(scales, scalings, "ko-", linewidth=2, markersize=8)
        for i, scale in enumerate(scales):
            axs[2].annotate(f"{scalings[i]:.1e}", (scale, scalings[i]), xytext=(5, 5), textcoords="offset points")

        axs[2].set_xlabel("Domain Scale Factor")
        axs[2].set_ylabel("Auto-Scaling Factor")
        axs[2].set_title("Auto-Scaling Factors vs. Domain Scale")
        axs[2].grid(True)

        # Set x-axis to use exact scale values
        axs[2].set_xticks(scales)
        axs[2].set_xticklabels([str(s) for s in scales])

        # Use log scale for y-axis if values span multiple orders of magnitude
        if max(scalings) / min(scalings) > 100:
            axs[2].set_yscale("log")

    plt.tight_layout()
    plt.show()

    # Print summary of minimum k1L values
    print("\nSUMMARY OF MINIMUM k1L VALUES:")
    print("-" * 40)
    for scale in scale_factors:
        k1_min = min(all_results[scale][0]) * base_config["L"]
        k1_max = max(all_results[scale][0]) * base_config["L"]
        print(f"Scale factor {scale:3d}: k1L range [{k1_min:.5f}, {k1_max:.5f}]")

    # Print summary of scaling factors
    print("\nSUMMARY OF AUTO-SCALING FACTORS:")
    print("-" * 40)
    for scale in scale_factors:
        print(f"Scale factor {scale:3d}: {scaling_factors[scale]:.2e}")

    return fig, axs


# ------------------------------------------------------------------------------------------------ #


def study_grid_and_domain_effects(base_config, grid_exponents=None, domain_factors=None, num_realizations=3):
    """
    Study the combined effects of grid resolution and domain size on spectrum accuracy.
    Creates heatmaps to visualize various metrics across the parameter space.

    Parameters
    ----------
    base_config : dict
        Base configuration dictionary
    grid_exponents : list, optional
        List of exponents for grid size where N1=N2=2^exponent
    domain_factors : list, optional
        List of domain size factors where L1_factor=L2_factor
    num_realizations : int, optional
        Number of realizations for each parameter combination

    Returns
    -------
    figs : list
        List of matplotlib figures with different metric visualizations
    """
    if grid_exponents is None:
        grid_exponents = [7, 8, 9, 10]  # 128, 256, 512, 1024 grid points

    if domain_factors is None:
        domain_factors = [1, 5, 10, 50, 100, 500]

    print(f"\n{'='*70}")
    print("GRID AND DOMAIN SIZE EFFECT STUDY")
    print(f"Grid exponents: {grid_exponents} (grid sizes: {[2**n for n in grid_exponents]})")
    print(f"Domain factors: {domain_factors}")
    print(f"Realizations per combination: {num_realizations}")
    print(f"{'='*70}")

    # Initialize data storage
    n_exp = len(grid_exponents)
    n_factors = len(domain_factors)

    # Keep only two metrics: auto-scale factors and u1 norms
    auto_scale_factors = np.zeros((n_exp, n_factors))
    u1_norms = np.zeros((n_exp, n_factors))

    # Create reference analytical spectrum
    k1_ref = np.logspace(-3, 3, 1000) / base_config["L"]
    # Using base generator just for analytical spectrum
    base_gen = generator(base_config)
    F11_analytical, F22_analytical = base_gen.analytical_spectrum(k1_ref)

    # Iterate through all combinations
    total_combinations = n_exp * n_factors
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

            # Configure this specific test case
            config = base_config.copy()
            config["N1"] = exp
            config["N2"] = exp
            config["L1_factor"] = factor
            config["L2_factor"] = factor

            # Initialize generator
            gen = generator(config)

            # Run simulations for this combination
            all_F11 = []

            u1, u2 = gen.generate(eta_ones=True)

            u1_norms[i, j] = np.linalg.norm(u1) * np.sqrt(gen.dx * gen.dy)

            for r in range(num_realizations):
                print(f"  Running realization {r+1}/{num_realizations}...")
                # Generate velocity fields
                gen.generate(eta_ones=True)

                # Get spectrum
                k1_sim, F11, F22 = gen.compute_spectrum()

                if r == 0:
                    all_F11 = np.zeros((num_realizations, len(F11)))

                all_F11[r, :] = F11

            # Average the spectra and norms
            F11_avg = np.mean(all_F11, axis=0)

            # Auto-scaling factor
            max_analytical_F11 = np.max(k1_ref * F11_analytical)
            max_simulated_F11 = np.max(k1_sim * F11_avg)
            auto_scale_factors[i, j] = max_analytical_F11 / max_simulated_F11 if max_simulated_F11 > 0 else np.nan

            print(f"  Average u1 norm: {u1_norms[i, j]:.6e}")
            print(f"  Auto-scaling factor: {auto_scale_factors[i, j]:.2e}")

    # Create visualizations (heatmaps)
    figs = []

    # Common function for creating heatmaps
    def create_heatmap(data, title, cmap="viridis", logscale=False):
        fig, ax = plt.subplots(figsize=(10, 8))

        x_labels = [str(f) for f in domain_factors]
        y_labels = [f"2^{e} ({2**e})" for e in grid_exponents]

        # Log transform for certain metrics if needed
        plot_data = np.log10(data) if logscale else data

        # Create heatmap
        im = ax.imshow(plot_data, cmap=cmap, aspect="auto")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        if logscale:
            cbar.set_label(f"{title} (log10 scale)")
        else:
            cbar.set_label(title)

        # Configure axes
        ax.set_xticks(np.arange(len(domain_factors)))
        ax.set_yticks(np.arange(len(grid_exponents)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        # Label axes
        ax.set_xlabel("Domain Scale Factor (L1_factor = L2_factor)")
        ax.set_ylabel("Grid Size Exponent (N1 = N2 = 2^exponent)")
        ax.set_title(title)

        # Add text annotations with values
        for i in range(len(grid_exponents)):
            for j in range(len(domain_factors)):
                if logscale:
                    text = f"{data[i, j]:.2e}"
                elif data[i, j] < 0.01:
                    text = f"{data[i, j]:.2e}"
                else:
                    text = f"{data[i, j]:.4f}"

                # More readable text color logic
                if cmap in ["RdBu", "RdBu_r", "coolwarm"]:
                    # For diverging colormaps, use threshold at middle of scale
                    text_color = (
                        "white"
                        if abs(plot_data[i, j] - np.mean(plot_data)) > (np.max(plot_data) - np.min(plot_data)) / 4
                        else "black"
                    )
                else:
                    text_color = "white" if plot_data[i, j] > np.mean(plot_data) else "black"

                ax.text(j, i, text, ha="center", va="center", color=text_color)

        plt.tight_layout()
        return fig

    # 1. Auto-scale factors heatmap with red-blue colormap
    fig1 = create_heatmap(auto_scale_factors, "Auto-Scaling Factor", cmap="RdBu_r", logscale=True)
    figs.append(fig1)

    # 2. Velocity field norm (||u1||) heatmap
    fig2 = create_heatmap(u1_norms, "Velocity Field Norm (||u1||)", cmap="viridis", logscale=True)
    figs.append(fig2)

    # Display all plots
    for fig in figs:
        plt.figure(fig.number)
        plt.show()

    # Save data to a file for later reference
    results_dict = {
        "grid_exponents": grid_exponents,
        "domain_factors": domain_factors,
        "auto_scale_factors": auto_scale_factors,
        "u1_norms": u1_norms,
    }

    return figs, results_dict


# ------------------------------------------------------------------------------------------------ #


def study_length_scale_effect(L_values=None, num_realizations=3):
    """
    Study how the length scale L affects the scaling factors and field statistics.

    Parameters
    ----------
    L_values : list, optional
        List of length scale values to test. Default is [125, 250, 500, 1000, 2000]
    num_realizations : int, optional
        Number of realizations for each length scale value

    Returns
    -------
    fig : matplotlib figure
        Figure showing the results
    results : dict
        Dictionary containing detailed results
    """
    if L_values is None:
        L_values = [125, 250, 500, 1000, 2000]

    print(f"\n{'='*60}")
    print("LENGTH SCALE (L) EFFECT STUDY")
    print(f"Testing L values: {L_values}")
    print(f"Realizations per value: {num_realizations}")
    print(f"{'='*60}")

    # Fixed parameters
    L1_factor = 10
    L2_factor = 10
    grid_exponent = 9  # 2^9 = 512 grid points
    epsilon = 0.01

    # Store results
    scaling_factors = []
    stats_dict = {}

    # Reference analytical spectrum (using first L value as reference)
    base_config = {
        "L": L_values[0],
        "epsilon": epsilon,
        "L1_factor": L1_factor,
        "L2_factor": L2_factor,
        "N1": grid_exponent,
        "N2": grid_exponent,
    }

    ref_gen = generator(base_config)
    # Custom wavenumber array for analytical spectrum
    k1_custom = np.logspace(-3, 3, 1000) / L_values[0]
    F11_analytical, F22_analytical = ref_gen.analytical_spectrum(k1_custom)

    # Test each L value
    for L in L_values:
        print(f"\n{'-'*50}")
        print(f"Testing length scale L = {L}")

        config = {
            "L": L,
            "epsilon": epsilon,
            "L1_factor": L1_factor,
            "L2_factor": L2_factor,
            "N1": grid_exponent,
            "N2": grid_exponent,
        }

        gen = generator(config)
        domain_size = f"{gen.L1} x {gen.L2}"
        grid_size = f"{gen.N1} x {gen.N2}"
        print(f"Domain: {domain_size} units, Grid: {grid_size}, Min k1L: {2*np.pi/gen.L1*gen.L:.4f}")

        # Initialize accumulators for statistics
        u1_mins, u1_maxs, u1_means, u1_vars = [], [], [], []
        u2_mins, u2_maxs, u2_means, u2_vars = [], [], [], []
        F11_avgs = []

        for i in range(num_realizations):
            print(f"  Realization {i+1}/{num_realizations}...")
            u1, u2 = gen.generate()

            # Collect statistics
            u1_mins.append(np.min(u1))
            u1_maxs.append(np.max(u1))
            u1_means.append(np.mean(u1))
            u1_vars.append(np.var(u1))

            u2_mins.append(np.min(u2))
            u2_maxs.append(np.max(u2))
            u2_means.append(np.mean(u2))
            u2_vars.append(np.var(u2))

            # Compute spectrum
            k1_sim, F11, F22 = gen.compute_spectrum()
            F11_avgs.append(F11)

        # Average the F11 spectra across realizations
        F11_avg = np.mean(np.array(F11_avgs), axis=0)

        # Calculate auto-scaling factor
        k1_this_analytical = np.logspace(-3, 3, 1000) / L
        F11_this_analytical, _ = gen.analytical_spectrum(k1_this_analytical)

        max_analytical = np.max(k1_this_analytical * F11_this_analytical)
        max_simulated = np.max(k1_sim * F11_avg)
        auto_scale = max_analytical / max_simulated if max_simulated > 0 else np.nan

        scaling_factors.append(auto_scale)

        # Collect statistics for this L value
        stats_dict[L] = {
            "u1_min": np.mean(u1_mins),
            "u1_max": np.mean(u1_maxs),
            "u1_mean": np.mean(u1_means),
            "u1_var": np.mean(u1_vars),
            "u2_min": np.mean(u2_mins),
            "u2_max": np.mean(u2_maxs),
            "u2_mean": np.mean(u2_means),
            "u2_var": np.mean(u2_vars),
            "auto_scale": auto_scale,
            "dx": gen.dx,
            "dy": gen.dy,
            "min_k1L": 2 * np.pi / gen.L1 * gen.L,
        }

        # Print summary for this L value
        print(f"\nResults for L = {L}:")
        print(f"  Auto-scaling factor: {auto_scale:.2e}")
        print(
            f"  u1: min={stats_dict[L]['u1_min']:.6f}, max={stats_dict[L]['u1_max']:.6f}, "
            f"mean={stats_dict[L]['u1_mean']:.6f}, var={stats_dict[L]['u1_var']:.6f}"
        )
        print(
            f"  u2: min={stats_dict[L]['u2_min']:.6f}, max={stats_dict[L]['u2_max']:.6f}, "
            f"mean={stats_dict[L]['u2_mean']:.6f}, var={stats_dict[L]['u2_var']:.6f}"
        )

    # Print summary table
    print("\nSUMMARY OF RESULTS ACROSS LENGTH SCALES:")
    print("-" * 70)
    print(f"{'L':>8} | {'dx':>10} | {'dy':>10} | {'min_k1L':>10} | {'Auto-scale':>15} | {'u1_var':>15}")
    print("-" * 70)
    for L in L_values:
        s = stats_dict[L]
        print(
            f"{L:>8} | {s['dx']:>10.4f} | {s['dy']:>10.4f} | {s['min_k1L']:>10.4f} | "
            f"{s['auto_scale']:>15.4e} | {s['u1_var']:>15.4e}"
        )

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot scaling factors vs L
    ax1.plot(L_values, scaling_factors, "bo-", linewidth=2, markersize=8)
    for i, L in enumerate(L_values):
        ax1.annotate(f"{scaling_factors[i]:.1e}", (L, scaling_factors[i]), xytext=(5, 5), textcoords="offset points")

    ax1.set_xlabel("Length Scale L")
    ax1.set_ylabel("Auto-Scaling Factor")
    ax1.set_title("Auto-Scaling Factors vs. Length Scale L")
    ax1.grid(True)

    # Check if we need log scale
    if max(scaling_factors) / min(scaling_factors) > 100:
        ax1.set_yscale("log")

    # Second plot: Variance vs L
    u1_vars = [stats_dict[L]["u1_var"] for L in L_values]
    ax2.plot(L_values, u1_vars, "ro-", linewidth=2, markersize=8)
    for i, L in enumerate(L_values):
        ax2.annotate(f"{u1_vars[i]:.1e}", (L, u1_vars[i]), xytext=(5, 5), textcoords="offset points")

    ax2.set_xlabel("Length Scale L")
    ax2.set_ylabel("u1 Variance")
    ax2.set_title("Velocity Field Variance vs. Length Scale L")
    ax2.grid(True)

    # Check if we need log scale
    if max(u1_vars) / min(u1_vars) > 100:
        ax2.set_yscale("log")

    plt.tight_layout()
    plt.show()

    # Return results
    results = {"L_values": L_values, "scaling_factors": scaling_factors, "stats": stats_dict}

    return fig, results


# ------------------------------------------------------------------------------------------------ #

if __name__ == "__main__":
    # mesh_independence_study(high=14)
    # scale_independence_study(high=20)

    config = {
        "L": 500,
        "epsilon": 0.01,
        "L1_factor": 2,
        "L2_factor": 2,
        "N1": 9,
        "N2": 9,
    }

    FINE_CONFIG = {
        "L": 500,  # [m]
        "epsilon": 0.01,
        "L1_factor": 10,
        "L2_factor": 10,
        "N1": 9,
        "N2": 9,
    }

    # gen = generator(FINE_CONFIG)
    # gen.generate()
    # diagnostics(gen, plot=True)

    # plot_spectrum_comparison(FINE_CONFIG)

    # compare_analytical_vs_estimated(FINE_CONFIG)

    # Base configuration
    base_config = {
        "L": 500,
        "epsilon": 0.01,
        "L1_factor": 1,
        "L2_factor": 1,
        "N1": 10,
        "N2": 10,
    }

    # # Compare spectra across different domain scales
    # compare_spectra_across_scales(
    #     base_config, scale_factors=[10, 20, 50, 100, 500, 1000], num_realizations=5, autoscale=True
    # )

    # compare_spectra_across_scales(
    #     base_config, scale_factors=[10, 20, 50, 100, 500, 1000], num_realizations=10, autoscale=False
    # )

    study_grid_and_domain_effects(
        base_config, grid_exponents=[6, 7, 8, 9, 10, 11], domain_factors=[10, 20, 40, 80, 100], num_realizations=2
    )

    # Add this line to run the length scale study
    # study_length_scale_effect()
