import concurrent.futures
import time

import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy

"""
- [X] Mesh independence study
- [X] Scale independence study
- [ ] Plot velocity fields
- [ ] Match spectrum
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

        eta: np.ndarray
        if eta_ones:
            eta = np.ones_like(self.k1, dtype=complex)
        else:
            noise = np.random.normal(0, 1, size=(self.N1, self.N2))
            eta = np.fft.fft2(noise)

        u1_freq_complex, u2_freq_complex = self._generate_numba_helper(
            self.k1,
            self.k2,
            self.c0,
            self.epsilon,
            self.L,
            self.N1,
            self.N2,
            self.dx,
            self.dy,
            eta,
        )

        transform_norm = np.sqrt(self.dx * self.dy)
        normalization = 1 / (self.dx * self.dy)

        u1 = np.real(np.fft.ifft2(u1_freq_complex) / transform_norm) * normalization
        u2 = np.real(np.fft.ifft2(u2_freq_complex) / transform_norm) * normalization

        self.u1 = u1
        self.u2 = u2

        return u1, u2

    @staticmethod
    @numba.njit(parallel=True)
    def _generate_numba_helper(k1, k2, c0, epsilon, L, N1, N2, dx, dy, eta):
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

                phi_[i, j] = c0 * np.cbrt(epsilon) * np.sqrt((L / np.sqrt(1 + (k_sq * L**2))) ** (17 / 3) / (4 * np.pi))

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
    def _compute_spectrum_numba_helper(k1_flat, k1_pos, power_u1_flat, power_u2_flat, dy):
        """
        Numba-accelerated helper function for spectrum computation
        """
        F11 = np.zeros_like(k1_pos)
        F22 = np.zeros_like(k1_pos)

        for i in numba.prange(len(k1_pos)):
            k1_val = k1_pos[i]
            indices = np.where(k1_flat == k1_val)[0]

            if len(indices) > 0:
                # Calculate mean of power values at these indices
                F11[i] = np.mean(power_u1_flat[indices]) * dy
                F22[i] = np.mean(power_u2_flat[indices]) * dy

        return F11, F22

    def compute_spectrum(self, u1=None, u2=None, k1_max=None):
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

        # Get positive wavenumbers
        k1_pos_mask = self.k1_fft > 0
        k1_pos = self.k1_fft[k1_pos_mask]

        # Compute power spectra
        power_u1 = (np.abs(u1_fft) / (self.N1 * self.N2)) ** 2
        power_u2 = (np.abs(u2_fft) / (self.N1 * self.N2)) ** 2

        # Flatten k1 for faster processing
        k1_flat = self.k1.flatten()
        power_u1_flat = power_u1.flatten()
        power_u2_flat = power_u2.flatten()

        # Call the Numba-accelerated helper function
        F11, F22 = self._compute_spectrum_numba_helper(k1_flat, k1_pos, power_u1_flat, power_u2_flat, self.dy)

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


def diagnostic_plot(gen: generator):
    assert np.any(gen.u1) and np.any(gen.u2), "Generator has not been run"

    x_coords = np.linspace(0, gen.L1, gen.N1)
    y_coords = np.linspace(0, gen.L2, gen.N2)

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

    gen = generator(config)

    # Custom wavenumber array. Only used for analytical spectrum.
    k1_custom = np.logspace(-3, 3, 1000) / config["L"]
    F11_analytical, F22_analytical = gen.analytical_spectrum(k1_custom)

    k1_max = np.max(k1_custom)

    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_compute_single_realization, config, k1_max) for _ in range(num_realizations)]

        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error: {e}")

    if not results:
        raise RuntimeError("No results were collected")

    k1_pos = results[0][0]
    F11_avg = np.zeros_like(k1_pos)
    F22_avg = np.zeros_like(k1_pos)

    for _, F11, F22 in results:
        F11_avg += F11
        F22_avg += F22

    lr = len(results)
    F11_avg /= lr
    F22_avg /= lr

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


def compare_generation_methods(config=None, num_runs=5):
    """
    Compare the performance and results of the standard and Numba-accelerated generation methods.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary for the generator. If None, a default config is used.
    num_runs : int, optional
        Number of runs to average timing results over.

    Returns
    -------
    dict
        Dictionary containing timing results and error metrics.
    """
    if config is None:
        config = {
            "L": 500,
            "epsilon": 0.01,
            "L1_factor": 2,
            "L2_factor": 2,
            "N1": 9,
            "N2": 9,
        }

    gen = generator(config)

    # Warm-up run for Numba (first run includes compilation time)
    print("Performing Numba warm-up run...")
    gen.generate_numba()

    # Timing and comparison
    standard_times = []
    numba_times = []
    rel_errors_u1 = []
    rel_errors_u2 = []

    print(f"Running {num_runs} comparison tests...")
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}")

        # Time standard method
        start_time = time.time()
        u1_std, u2_std = gen.generate(eta_ones=True)
        standard_time = time.time() - start_time
        standard_times.append(standard_time)

        # Time Numba method
        start_time = time.time()
        u1_numba, u2_numba = gen.generate_numba(eta_ones=True)
        numba_time = time.time() - start_time
        numba_times.append(numba_time)

        # Calculate relative error
        # u1_error = np.linalg.norm(u1_std - u1_numba) / np.linalg.norm(u1_std)
        # u2_error = np.linalg.norm(u2_std - u2_numba) / np.linalg.norm(u2_std)

        u1_error = np.linalg.norm(u1_std - u1_numba)
        u2_error = np.linalg.norm(u2_std - u2_numba)

        assert np.allclose(u1_std, u1_numba, atol=1e-10)
        assert np.allclose(u2_std, u2_numba, atol=1e-10)

        rel_errors_u1.append(u1_error)
        rel_errors_u2.append(u2_error)

    diagnostic_plot(u1_std, u2_std)
    diagnostic_plot(u1_numba, u2_numba)

    # Calculate statistics
    avg_standard_time = np.mean(standard_times)
    avg_numba_time = np.mean(numba_times)
    speedup = avg_standard_time / avg_numba_time
    avg_u1_error = np.mean(rel_errors_u1)
    avg_u2_error = np.mean(rel_errors_u2)

    # Print results
    print("\nPerformance Comparison:")
    print(f"Average standard method time: {avg_standard_time:.4f} seconds")
    print(f"Average Numba method time: {avg_numba_time:.4f} seconds")
    print(f"Speedup factor: {speedup:.2f}x")

    print("\nAccuracy Comparison:")
    print(f"Average relative error in u1: {avg_u1_error:.8e}")
    print(f"Average relative error in u2: {avg_u2_error:.8e}")

    # Test with different mesh sizes
    print("\nTesting scaling with mesh size:")
    mesh_sizes = [7, 8, 9, 10, 11, 12, 13]
    std_times = []
    numba_times = []

    for n in mesh_sizes:
        test_config = config.copy()
        test_config["N1"] = n
        test_config["N2"] = n
        test_gen = generator(test_config)

        # Standard method
        start_time = time.time()
        test_gen.generate()
        std_time = time.time() - start_time
        std_times.append(std_time)

        # Numba method
        start_time = time.time()
        test_gen.generate_numba()
        nb_time = time.time() - start_time
        numba_times.append(nb_time)

        print(f"Mesh size 2^{n}: Standard={std_time:.4f}s, Numba={nb_time:.4f}s, Speedup={std_time/nb_time:.2f}x")

    # Plot scaling results
    plt.figure(figsize=(10, 6))
    mesh_points = [2**n for n in mesh_sizes]
    plt.loglog(mesh_points, std_times, "o-", label="Standard Method")
    plt.loglog(mesh_points, numba_times, "s-", label="Numba Method")
    plt.xlabel("Mesh Size (N)")
    plt.ylabel("Execution Time (s)")
    plt.title("Performance Scaling with Mesh Size")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------ #


if __name__ == "__main__":
    # mesh_independence_study(high=12)
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
        "L": 500,
        "epsilon": 0.01,
        "L1_factor": 10,
        "L2_factor": 10,
        "N1": 15,
        "N2": 7,
    }

    print("\n" + "=" * 80)
    print("GENERATION METHOD COMPARISON")
    print("=" * 80)
    compare_generation_methods(config)

    # gen = generator(config)
    # gen.generate()
    # diagnostic_plot(gen.u1, gen.u2)

    # plot_spectrum_comparison(FINE_CONFIG)
