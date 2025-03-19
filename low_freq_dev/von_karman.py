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

        # Wavenumber generation
        self.k1_fft = 2 * np.pi * np.fft.fftfreq(self.N1, self.dx)
        self.k2_fft = 2 * np.pi * np.fft.fftfreq(self.N2, self.dy)
        self.k1, self.k2 = np.meshgrid(self.k1_fft, self.k2_fft, indexing="ij")

        # Useful for calculations
        self.k_mag = np.sqrt(self.k1**2 + self.k2**2)

        # Useful for plotting
        self.k1L = self.k1 * self.L
        self.k2L = self.k2 * self.L
        self.kL = self.k_mag * self.L

    def generate(self, eta_ones=False):
        """
        Generate turbulent velocity fields using the von Karman spectrum

        Parameters
        ----------
        eta_ones: bool, optional
            If True, replaces the white noise with a unit field (all ones)

        Returns
        -------
        u1: np.ndarray
            x- or longitudinal component of velocity field
        u2: np.ndarray
            y- or transversal component of velocity field
        """
        k_mag_sq = self.k1**2 + self.k2**2

        k_mag_sq_safe = np.copy(k_mag_sq)
        k_mag_sq_safe[k_mag_sq_safe == 0] = 1e-10

        # Sqrt of all leading factors, does NOT include sqrt P(k)
        phi_ = (
            self.c0
            * np.cbrt(self.epsilon)
            * np.sqrt((self.L / np.sqrt(1 + (k_mag_sq * self.L**2))) ** (17 / 3) / (4 * np.pi))
        )

        # TODO: OLD
        # C1 = 1j * phi_ * self.k2
        # C2 = 1j * phi_ * (-1 * self.k1)

        # TODO: NEW
        C1 = 1j * phi_ * self.k2  # / k_mag_sq_safe
        C2 = 1j * phi_ * (-1 * self.k1)  # / k_mag_sq_safe

        eta: np.ndarray
        if eta_ones:
            eta = np.ones_like(self.k1)
        else:
            noise = np.random.normal(0, 1, size=(self.N1, self.N2))
            eta = np.fft.fft2(noise)

        u1_freq = C1 * eta
        u2_freq = C2 * eta

        transform_norm = np.sqrt(self.dx * self.dy)
        normalization = 1 / (self.dx * self.dy)

        u1 = np.real(np.fft.ifft2(u1_freq) / transform_norm) * normalization
        u2 = np.real(np.fft.ifft2(u2_freq) / transform_norm) * normalization

        self.u1 = u1
        self.u2 = u2

        return u1, u2

    # ------------------------------------------------------------------------------------------------ #

    def compute_spectrum(self, u1=None, u2=None):
        """
        Compute the spectrum of generated velocity fields. If no fields are provided, checks
        class attributes for self.u1 and self.u2

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

        # u1_fft = np.fft.fft2(u1) / (self.dx * self.dy)
        # u2_fft = np.fft.fft2(u2) / (self.dx * self.dy)
        u1_fft = np.fft.fft2(u1)
        u2_fft = np.fft.fft2(u2)

        k1_pos_mask = self.k1_fft > 0
        k1_pos = self.k1_fft[k1_pos_mask]

        power_u1 = (np.abs(u1_fft) / (self.N1 * self.N2)) ** 2
        power_u2 = (np.abs(u2_fft) / (self.N1 * self.N2)) ** 2

        F11 = np.zeros_like(k1_pos)
        F22 = np.zeros_like(k1_pos)

        for i, k1_val in enumerate(k1_pos):
            mask = self.k1 == k1_val
            F11[i] = np.mean(power_u1[mask]) * self.dy
            F22[i] = np.mean(power_u2[mask]) * self.dy

        return k1_pos, F11, F22

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

    def compute_spectrum_numba(self, u1=None, u2=None):
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

    def compute_spectrum_numpy_fast(self, u1=None, u2=None):
        """
        Fast NumPy implementation of spectrum computation without Numba

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

        # Create result arrays
        F11 = np.zeros_like(k1_pos)
        F22 = np.zeros_like(k1_pos)

        # Use a vectorized approach with unique k1 values
        unique_k1 = np.unique(self.k1)
        unique_k1_pos = unique_k1[unique_k1 > 0]

        # Ensure we're using exactly the same k1_pos values as the original method
        # This fixes the shape mismatch
        k1_pos_set = set(k1_pos)
        unique_k1_pos = np.array([k for k in unique_k1_pos if k in k1_pos_set])

        # Create mapping from k1 values to indices in result arrays
        k1_to_idx = {k: i for i, k in enumerate(k1_pos)}

        for k1_val in unique_k1_pos:
            if k1_val in k1_to_idx:
                idx = k1_to_idx[k1_val]
                mask = self.k1 == k1_val
                F11[idx] = np.mean(power_u1[mask]) * self.dy
                F22[idx] = np.mean(power_u2[mask]) * self.dy

        return k1_pos, F11, F22

    def test_spectrum_computation(self, num_tests=3):
        """
        Test and compare different spectrum computation methods
        """
        # Generate velocity fields if not already present
        if not hasattr(self, "u1") or not hasattr(self, "u2"):
            self.generate()

        _ = self.compute_spectrum_numba()

        print("Original method")
        start_time = time.time()
        for _ in range(num_tests):
            k1_pos, F11, F22 = self.compute_spectrum()
        orig_time = (time.time() - start_time) / num_tests

        print("Numba method")
        start_time = time.time()
        for _ in range(num_tests):
            k1_pos_numba, F11_numba, F22_numba = self.compute_spectrum_numba()
        numba_time = (time.time() - start_time) / num_tests

        print("Numpy fast method")
        start_time = time.time()
        for _ in range(num_tests):
            k1_pos_np_fast, F11_np_fast, F22_np_fast = self.compute_spectrum_numpy_fast()
        np_fast_time = (time.time() - start_time) / num_tests

        # Verify results match
        np.testing.assert_allclose(k1_pos, k1_pos_numba, rtol=1e-7)
        np.testing.assert_allclose(F11, F11_numba, rtol=1e-7)
        np.testing.assert_allclose(F22, F22_numba, rtol=1e-7)

        np.testing.assert_allclose(k1_pos, k1_pos_np_fast, rtol=1e-7)
        np.testing.assert_allclose(F11, F11_np_fast, rtol=1e-7)
        np.testing.assert_allclose(F22, F22_np_fast, rtol=1e-7)

        print(f"Original method: {orig_time:.4f} seconds")
        print(f"Numba method: {numba_time:.4f} seconds (speedup: {orig_time/numba_time:.2f}x)")
        print(f"NumPy fast method: {np_fast_time:.4f} seconds (speedup: {orig_time/np_fast_time:.2f}x)")

        return {"original": orig_time, "numba": numba_time, "numpy_fast": np_fast_time}

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


def run_single_mesh(exponent, config):
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

    exponents = np.arange(low, high)

    # Square mesh

    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Pass config as an additional argument
        futures = [executor.submit(run_single_mesh, exp, config) for exp in exponents]

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


def run_single_scale(scale, config):
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

    factors = np.arange(low, high, 0.5)

    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_scale, factor, config) for factor in factors]

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


def diagnostic_plot(u1, u2):
    x_coords = np.linspace(0, 60, u1.shape[1])
    y_coords = np.linspace(0, 15, u1.shape[0])

    plot_velocity_fields(u1, u2, x_coords, y_coords, title="Von Karman Velocity field")
    plt.show()


# ------------------------------------------------------------------------------------------------ #
# Spectrum plot


def _compute_single_realization(config):
    gen = generator(config)
    u1, u2 = gen.generate()
    k1_sim, F11, F22 = gen.compute_spectrum(u1, u2)
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

    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_compute_single_realization, config) for _ in range(num_realizations)]

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


def plot_spectrum(config: dict, num_realizations=10):
    """
    Generate and plot spectra.

    Parameters:
    - config: Configuration dictionary
    - num_realizations: Number of realizations to average
    """
    # Create generator
    gen = generator(config)

    # Initialize arrays for averaging
    F11_avg = None
    F22_avg = None
    k1_pos = None

    # Generate and average multiple realizations
    for _ in range(num_realizations):
        u1, u2 = gen.generate()

        k1_sim, F11, F22 = gen.compute_spectrum(u1, u2)

        if F11_avg is None:
            k1_pos = k1_sim
            F11_avg = F11
            F22_avg = F22
        else:
            F11_avg += F11
            F22_avg += F22

    F11_avg /= num_realizations
    F22_avg /= num_realizations

    # Compute analytical spectrum
    F11_analytical, F22_analytical = gen.analytical_spectrum(k1_pos)

    # Plot spectra
    plt.figure(figsize=(10, 6))

    # Plot k1*F11 and k1*F22 for easier comparison with literature
    plt.loglog(k1_pos * gen.L, k1_pos * F11_avg, "r--", label="Estimated F11")
    plt.loglog(k1_pos * gen.L, k1_pos * F22_avg, "b--", label="Estimated F22")
    plt.loglog(k1_pos * gen.L, k1_pos * F11_analytical, "r-", label="Analytical F11")
    plt.loglog(k1_pos * gen.L, k1_pos * F22_analytical, "b-", label="Analytical F22")

    plt.xlabel("$k_1 L$ [-]")
    plt.ylabel("$k_1 F(k_1)$ [$m^2s^{-2}$]")
    plt.title("Von Karman Spectra")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.show()


# ------------------------------------------------------------------------------------------------ #
# Driver

if __name__ == "__main__":
    mesh_independence_study(high=12)
    scale_independence_study(high=20)

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
        "L1_factor": 2,
        "L2_factor": 2,
        "N1": 10,
        "N2": 10,
    }

    gen = generator(FINE_CONFIG)
    gen.generate()
    gen.test_spectrum_computation()

    # plot_spectrum(config)
    # plot_spectrum_comparison(FINE_CONFIG)

    # gen = generator(config)
    # u1, u2 = gen.generate()
    # diagnostic_plot(u1, u2)
