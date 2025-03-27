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

        self.c = (8.0 * self.sigma2) / (9.0 * (self.L_2d ** (2 / 3)))

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
        N1: int,
        N2: int,
        noise_hat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        k_mag = np.sqrt(k1**2 + k2**2)
        kappa = np.sqrt(2 * ((k1 * np.cos(psi)) ** 2 + (k2 * np.sin(psi)) ** 2))

        phi_ = np.empty_like(k_mag)
        for i in numba.prange(N1):
            for j in numba.prange(N2):
                # Avoid division by zero with kappa
                _kappa: float
                if kappa[i, j] < 1e-10:
                    _kappa = 1e-15
                else:
                    _kappa = kappa[i, j]

                # Avoid division by zero with k
                _k: float
                if k_mag[i, j] < 1e-10:
                    _k = 1e-15
                else:
                    _k = k_mag[i, j]

                # Energy spectrum in terms of kappa
                energy = c * _kappa**3 / ((L_2d**-2 + _kappa**2) ** (7 / 3))

                # Attenuation factor
                energy /= 1 + (_kappa * z_i) ** 2

                phi_[i, j] = np.sqrt(energy / (np.pi * _k**3))

        C1 = 1j * phi_ * k2
        C2 = 1j * phi_ * (-1 * k1)

        return C1 * noise_hat, C2 * noise_hat

    def generate(self):
        # Obtain random noise
        noise = np.random.normal(0, 1, size=(self.N1, self.N2))
        # noise_hat = np.fft.fft2(noise)
        noise_hat = np.fft.fft2(noise)

        u1_freq, u2_freq = self._generate_numba_helper(
            self.k1, self.k2, self.c, self.L_2d, self.psi, self.z_i, self.N1, self.N2, noise_hat
        )

        # NOTE: transform below is to control spatial white noise variance
        # transform = 1 if eta_ones else np.sqrt(self.N1 * self.N2)
        # u1 = np.real(np.fft.ifft2(u1_freq) / transform)
        # u2 = np.real(np.fft.ifft2(u2_freq) / transform)

        # Proper normalization that preserves variance across grid sizes

        u1 = np.real(np.fft.ifft2(u1_freq)) * np.sqrt(self.N1 * self.N2)
        u2 = np.real(np.fft.ifft2(u2_freq)) * np.sqrt(self.N1 * self.N2)

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

    def compute_spectrum(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        F11, F22 = self._compute_spectrum_numba_helper(k1_flat, k1_pos, power_u1_flat, power_u2_flat, self.L2)

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

    def plot_velocity_fields(self):
        print("=" * 80)
        print("VELOCITY FIELD PLOT")
        print("=" * 80)

        # Print statistics for debugging
        print("u1 stats")
        print(f"min: {np.min(self.u1)}", f"max: {np.max(self.u1)}")
        print(f"mean: {np.mean(self.u1)}", f"std: {np.std(self.u1)}")
        print(f"Any nan: {np.isnan(self.u1).any()}")

        print("u2 stats")
        print(f"min: {np.min(self.u2)}", f"max: {np.max(self.u2)}")
        print(f"mean: {np.mean(self.u2)}", f"std: {np.std(self.u2)}")
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
NUM_REALIZATIONS = 5


def _run_single_mesh(exponent, config):
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

    for _ in range(NUM_REALIZATIONS):
        curr_u1, curr_u2 = gen.generate()
        u1 += curr_u1
        u2 += curr_u2

    u1 /= NUM_REALIZATIONS
    u2 /= NUM_REALIZATIONS

    u1_norm = np.linalg.norm(u1 * gen.dx * gen.dy) ** 2
    u2_norm = np.linalg.norm(u2 * gen.dx * gen.dy) ** 2

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

if __name__ == "__main__":
    ###############################################
    # Mesh independence study

    mesh_independence_study()

    ###############################################
    # Length scale study

    ###############################################
    # Recreate figure 3

    cfg_fig3 = {
        "sigma2": 0.6,
        "L_2d": 15_000.0,
        "psi": np.deg2rad(43.0),
        "z_i": 500.0,
        "L1_factor": 4,
        "L2_factor": 1,
        "N1": 10,
        "N2": 10,
    }
    # gen = generator(cfg_fig3)
    # gen.generate()
    # gen.plot_velocity_fields()

    ###############################################
    # Recreate Figure 2

    # plot_spectra_comparison()
