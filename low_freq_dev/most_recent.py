import concurrent.futures

import Fij_numerical_integrator as Fij
import matplotlib.pyplot as plt
import numba
import numpy as np

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

        self.L1 = config["L1_factor"] * self.L_2d
        self.L2 = config["L2_factor"] * self.L_2d

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

        self.k1L = self.k1 * self.L_2d
        self.k2L = self.k2 * self.L_2d
        self.kL = self.k_mag * self.L_2d

    # ------------------------------------------------------------------------------------------------ #

    # TODO: This assumes isotropic, so we need to add kappa back in
    @staticmethod
    @numba.njit(parallel=True)
    def _generate_numba_helper(
        k1: np.ndarray,
        k2: np.ndarray,
        c: float,
        L_2d: float,
        psi: float,
        z_i: float,
        N1: int,
        N2: int,
        noise_hat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        k_mag_sq = k1**2 + k2**2

        phi_ = np.empty_like(k_mag_sq)
        for i in numba.prange(N1):
            for j in numba.prange(N2):
                k_sq: float
                if k_mag_sq[i, j] < 1e-10:
                    k_sq = 1e-10
                else:
                    k_sq = k_mag_sq[i, j]

                phi_[i,j] = np.sqrt(
                    c / (
                        np.pi * (L_2d**-2 + k_sq)**(7/3) *\
                        (1 + k_sq * z_i**2)
                    )
                )

        C1 = 1j * phi_ * k2
        C2 = 1j * phi_ * (-1 * k1)

        return C1 * noise_hat, C2 * noise_hat

    def generate(self, eta_ones=False):
        # TODO: Numba numba numba!

        # Calculate scaling constant c
        self.c = (8.0 * self.sigma2) / (9.0 * (self.L_2d ** (2 / 3)))

        # Random noise
        noise_hat: np.ndarray
        if eta_ones:
            noise_hat = np.ones_like(self.k1, dtype=complex)
        else:
            noise = np.random.normal(0, 1, size=(self.N1, self.N2))
            noise_hat = np.fft.fft2(noise)
        
        # Fourier space
        u1_freq, u2_freq = self._generate_numba_helper(
            self.k1,
            self.k2,
            self.c,
            self.L_2d,
            self.psi,
            self.z_i,
            self.N1,
            self.N2,
            noise_hat
        )

        transform = 1 if eta_ones else np.sqrt(self.dx * self.dy)

        u1 = np.real(np.fft.ifft2(u1_freq) / transform)
        u2 = np.real(np.fft.ifft2(u2_freq) / transform)

        self.u1 = u1
        self.u2 = u2

        return u1, u2

    # ------------------------------------------------------------------------------------------------ #

    @staticmethod
    @numba.njit(parallel=True)
    def _compute_spectrum_numba_helper(
        k1_flat: np.ndarray,
        k1_pos: np.ndarray,
        power_u1_flat: np.ndarray,
        power_u2_flat: np.ndarray,
        L2: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Numba-accelerated helper function for the estimation
        of the 1d spectra F_11 and F_22.
        """

        F11 = np.empty_like(k1_pos)
        F22 = np.empty_like(k1_pos)

        dk2 = 2 * np.pi / L2

        for i in numba.prange(len(k1_pos)):
            k1_val = k1_pos[i]
            indices = np.where(k1_flat == k1_val)[0]

            if len(indices) > 0:
                F11[i] = np.sum(power_u1_flat[indices]) * dk2
                F22[i] = np.sum(power_u2_flat[indices]) * dk2

        return F11, F22

    def compute_spectrum(
        self
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimates the 1d spectra F11 and F22 of the velocity fields u1 and u2"""

        u1_fft = np.fft.fft2(self.u1)
        u2_fft = np.fft.fft2(self.u2)

        k1_pos = np.abs(self.k1_fft)
        k1_pos = k1_pos[np.argsort(k1_pos)]
        k1_pos = k1_pos[1:-1]

        power_u1 = (np.abs(u1_fft)) ** 2
        power_u2 = (np.abs(u2_fft)) ** 2

        if np.isnan(power_u1).any() or np.isnan(power_u2).any():
            import warnings
            warnings.warn("NaN detected in power spectra!")

        k1_flat = self.k1.flatten()
        power_u1_flat = power_u1.flatten()
        power_u2_flat = power_u2.flatten()

        F11, F22 = self._compute_spectrum_numba_helper(
            k1_flat,
            k1_pos, 
            power_u1_flat, 
            power_u2_flat, 
            self.L2
        )

        return k1_pos, F11, F22

    # ------------------------------------------------------------------------------------------------ #

    def analytical_spectrum(self, k1_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        # Initialize output arrays
        F11 = np.zeros_like(k1_arr)
        F22 = np.zeros_like(k1_arr)
        
        # Loop through each k1 value and compute spectrum point by point
        for i in range(len(k1_arr)):
            k1 = k1_arr[i]
            # Extract just the result value (first element of the tuple)
            F11[i] = Fij.eq6_numerical_F11_2D(k1, self.psi, self.L_2d, self.z_i, self.c)
            F22[i] = Fij.eq6_numerical_F22_2D(k1, self.psi, self.L_2d, self.z_i, self.c)
        
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
        "sigma2": 2.0,
        "L_2d": 15_000.0,
        "psi": np.deg2rad(45.0),
        "z_i": 500.0,
        
        "L1_factor": 4,
        "L2_factor": 1,
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

if __name__ == "__main__":
    base_config = {
        "sigma2": 2.0,
        "L_2d": 15_000.0,
        "psi": np.deg2rad(45.0),
        "z_i": 500.0,
        "L1_factor": 4,
        "L2_factor": 1,
        "N1": 10,
        "N2": 10,
    }

    gen = generator(base_config)
    gen.generate()
    diagnostics(gen, plot=True)
