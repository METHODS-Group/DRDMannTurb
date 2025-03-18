import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

#########################################################################


class Mann2DWindField:
    def __init__(self, config):
        """
        Initialize the 2D Mann model for low-frequency wind fluctuations.

        Parameters:
        -----------
        config : dict
            Configuration dictionary with the following keys:
            - sigma2: Variance of wind fluctuations [m²/s²]
            - L_2d: Length scale of mesoscale turbulence [m]
            - psi: Anisotropy parameter [rad]
            - z_i: Attenuation length/boundary layer height [m]
            - L1_factor: Domain length in x direction as multiple of L_2d
            - L2_factor: Domain length in y direction as multiple of L_2d
            - N1: Number of grid points in x direction
            - N2: Number of grid points in y direction
            - equation: Which equation to use ('eq14', 'eq15', 'eq16')
        """
        # Physical parameters
        self.sigma2 = config.get("sigma2")
        self.L_2d = config.get("L_2d")
        self.psi = config.get("psi", np.pi / 4)  # 45 degrees by default
        self.z_i = config.get("z_i")

        # Grid parameters
        self.L1 = config.get("L1_factor") * self.L_2d
        self.L2 = config.get("L2_factor") * self.L_2d
        self.N1 = config.get("N1")
        self.N2 = config.get("N2")
        self.equation = config.get("equation", "eq15")

        # Calculate grid spacing
        self.dx = self.L1 / self.N1
        self.dy = self.L2 / self.N2

        # Calculate c coefficient from sigma2 and L_2d (equation 3)
        self.c = (8.0 * self.sigma2) / (9.0 * self.L_2d ** (2.0 / 3.0))

        # Initialize grid
        self._initialize_grid()

    def _initialize_grid(self):
        """Initialize the computational grid in physical and wavenumber space."""
        # Physical grid
        self.x = np.linspace(0, self.L1, self.N1)
        self.y = np.linspace(0, self.L2, self.N2)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Wavenumber grid
        # TODO: check against Frequencies in sampling_methods
        self.k1_fft = 2 * np.pi * np.fft.fftfreq(self.N1, self.dx)
        self.k2_fft = 2 * np.pi * np.fft.fftfreq(self.N2, self.dy)

        self.k1, self.k2 = np.meshgrid(self.k1_fft, self.k2_fft, indexing="ij")

    def sqrtSpectralTensor_isotropic(self):
        """

        """
        common_factor = self.c / (
                np.pi *
                (self.L_2d**-2 + self.kappa**2) ** (7 / 3) *
                (1 + (self.kappa * self.z_i)**2)
        )

        # print("<isotropic_attenuated_spectral_tensor> common_factor: ", common_factor)
        phi_11 = common_factor * (self.k_mag**2 - self.k1**2)
        phi_12 = common_factor * (-1 * self.k1 * self.k2)
        phi_22 = common_factor * (self.k_mag**2 - self.k2**2)

        return phi_11, phi_12, phi_22

    # def _calculate_energy_spectrum(self):
    #     """Calculate the energy spectrum with anisotropy parameter."""
    #     E_kappa = self.c * (self.kappa**3) / ((self.L_2d**-2 + self.kappa**2) ** (7 / 3))
    #     E_kappa_attenuated = E_kappa / (1 + (self.kappa * self.z_i) ** 2)

    #     return E_kappa_attenuated

    # def _calculate_spectral_tensor(self, E_kappa):
    #     """Calculate the spectral tensor components (equation 1)."""

    #     # TODO: move k^3 from energy spectrum into parenthetical and div by zero goes away
    #     phi_common = np.zeros_like(self.kappa)
    #     idx = ~self.k_mag_mask
    #     phi_common[idx] = E_kappa[idx] / (np.pi * self.k_mag[idx])

    #     phi_11 = np.zeros_like(phi_common)
    #     phi_12 = np.zeros_like(phi_common)
    #     phi_22 = np.zeros_like(phi_common)

    #     idx = ~self.k_mag_mask
    #     phi_11[idx] = phi_common[idx] * (1 - (self.k1[idx] / self.k_mag[idx]) ** 2)
    #     phi_12[idx] = phi_common[idx] * (-1 * self.k1[idx] * self.k2[idx] / self.k_mag[idx] ** 2)
    #     phi_22[idx] = phi_common[idx] * (1 - (self.k2[idx] / self.k_mag[idx]) ** 2)

    #     return phi_11, phi_12, phi_22

    def _calculate_fourier_coefficients(self, phi_11, phi_12, phi_22):
        """Calculate Fourier coefficients based on spectral tensor."""
        C1= 1j


        return C_11, C_12, C_22

    def _generate_wind_field(self, C_11, C_12, C_22):
        """
        Generate the wind field using Fourier synthesis with random phases.

        Parameters
        ----------
        C_11 : np.ndarray
            Fourier coefficients for the 11 component of the spectral tensor
        C_12 : np.ndarray
            Fourier coefficients for the 12 component of the spectral tensor
        C_22 : np.ndarray
            Fourier coefficients for the 22 component of the spectral tensor

        Returns
        -------
        u : np.ndarray
            Generated wind field with shape (N1, N2, 2), where [:,:,i] represents u_i
        """

        # TODO: multiply by prod h analogue from gaussian_random_fields.py

        eta = np.random.normal(0, 1, size=(self.N1, self.N2, 2))

        # eta *= vol_scale

        # NOTE: Should be FT of real-valued noise
        # TODO: (1) check convolutions against dirac delta, ones, etc. which are nicer on paper
        # TODO: also try no spectral tensor/ ones'ing it out; then try to recreate covariance (identity)
        eta_freq = np.zeros_like(eta, dtype=complex)

        eta_freq[:, :, 0] = np.fft.fft2(eta[:, :, 0])
        eta_freq[:, :, 1] = np.fft.fft2(eta[:, :, 1])

        # Dirac delta
        ss = eta_freq.shape
        eta_freq = np.zeros_like(eta_freq)
        eta_freq[ss[0] // 2, ss[1] // 2, :] = 1

        # Ones
        # eta_freq = np.ones_like(eta_freq)

        u1_freq = (C_11 * eta_freq[:, :, 0]) + (C_12 * eta_freq[:, :, 1])
        u2_freq = (C_12 * eta_freq[:, :, 0]) + (C_22 * eta_freq[:, :, 1])

        # u1_freq = eta_freq
        # u2_freq = eta_freq

        # TODO: Maybe check the FFT outputs with deterministic/easier things instead of white noise
        # NOTE: numpy's ifft2 includes 1/(N1*N2) normalization
        print("u1 complex part", np.mean(np.imag(u1_freq)))
        print("u2 complex part", np.mean(np.imag(u2_freq)), "\n\n")

        u1 = np.real(np.fft.ifft2(u1_freq) * self.N1 * self.N2)
        u2 = np.real(np.fft.ifft2(u2_freq) * self.N1 * self.N2)

        u = np.stack([u1, u2], axis=-1) * vol_scale

        """
        self.fft_x[:] = noise
        self.fft_plan()
        self.fft_y[:] *= self.Spectrum_half
        self.ifft_plan()
        return self.fft_x[self.DomainSlice] / self.TransformNorm
        """

        # concatenate u1 and u2 for a [N1, N2, 2] array
        self.u_freq = np.stack([u1_freq, u2_freq], axis=-1)
        u = np.stack([u1, u2], axis=-1)
        self.u = u

        return u

    def generate(self):
        """Generate the 2D wind field and return the velocity components."""

        # E_kappa = self._calculate_energy_spectrum()
        # phi_11, phi_12, phi_22 = self._calculate_spectral_tensor(E_kappa)

        phi_11, phi_12, phi_22 = self.isotropic_attenuated_spectral_tensor()

        # Calculate Fourier coefficients
        C_11, C_12, C_22 = self._calculate_fourier_coefficients(phi_11, phi_12, phi_22)

        # Generate wind field
        self._generate_wind_field(C_11, C_12, C_22)

        # Verify variance
        var_u1 = np.var(self.u[:, :, 0])
        var_u2 = np.var(self.u[:, :, 1])

        print(f"Generation completed in {time.time() - t_start:.2f} seconds")
        print(f"Variance u1: {var_u1:.6f} m²/s² (target: {self.sigma2:.6f} m²/s²)")
        print(f"Variance u2: {var_u2:.6f} m²/s² (target: {self.sigma2:.6f} m²/s²)")

        return self.u

    def plot_field(self):
        """Plot the generated wind field."""
        if self.u is None:
            print("No wind field generated yet. Call generate() first.")
            return

        plt.figure(figsize=(12, 10))

        x_km = self.x / 1000
        y_km = self.y / 1000
        X_km, Y_km = np.meshgrid(x_km, y_km, indexing="ij")

        u1_plot = self.u[:, :, 0]
        u2_plot = self.u[:, :, 1]

        plt.subplot(2, 1, 1)
        im1 = plt.pcolormesh(
            X_km, Y_km, u1_plot, cmap="RdBu_r", shading="auto", vmin=-3 * np.std(u1_plot), vmax=3 * np.std(u1_plot)
        )
        plt.colorbar(im1, label="u [m/s]")
        plt.xlabel("x [km]")
        plt.ylabel("y [km]")
        plt.title("Longitudinal component (u)")

        # Plot u2 component
        plt.subplot(2, 1, 2)
        im2 = plt.pcolormesh(
            X_km, Y_km, u2_plot, cmap="RdBu_r", shading="auto", vmin=-3 * np.std(u2_plot), vmax=3 * np.std(u2_plot)
        )
        plt.colorbar(im2, label="v [m/s]")
        plt.xlabel("x [km]")
        plt.ylabel("y [km]")
        plt.title("Transverse component (v)")

        plt.tight_layout()
        plt.show()

    def plot_spectrum(self, plot_ratio=True):
        """Plot the energy spectrum."""
        if self.u is None:
            raise ValueError("No wind field generated yet. Call generate() first.")

        # Calculate 1D spectra
        k1_fft_pos_mask = self.k1_fft > 0
        k1 = self.k1_fft[k1_fft_pos_mask]

        u_freq_pos = self.u_freq[k1_fft_pos_mask]

        # Calculate power spectrum (average over k2)
        power_u = (np.abs(u_freq_pos) / (self.N1 * self.N2)) ** 2

        # Average over k2 to get 1D spectrum
        spectrum_u = np.mean(power_u, axis=1) * self.dy

        plt.figure(figsize=(10, 6))
        plt.semilogx(self.L_2d * k1, k1 * spectrum_u[:, 0], "b-", label="$k_1 F_{11}(k_1)$ (simulated)")
        plt.semilogx(self.L_2d * k1, k1 * spectrum_u[:, 1], "r-", label="$k_1 F_{22}(k_1)$ (simulated)")

        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel("$L_{2d}k_1$ [rad/m]")
        plt.ylabel("$k_1 F(k_1)$ [m²/s²]")
        plt.title("1D Energy Spectrum")
        plt.legend()
        plt.tight_layout()
        plt.show()

        if plot_ratio:
            actual_ratio = spectrum_u[:, 0] / (spectrum_u[:, 1] + 1e-16)
            expected_ratio = 3 * (np.reciprocal(np.tan(self.psi)) ** 2) / 5

            plt.figure(figsize=(10, 6))
            plt.semilogx(self.L_2d * k1, actual_ratio, "b-", label="$F_{11}/F_{22}$ (simulated)")
            plt.axhline(expected_ratio, color="r", linestyle="--", label=f"Expected ratio: {expected_ratio:.3f}")
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.xlabel("$L_{2d}k_1$ [rad/m]")
            plt.ylabel("$F_{11}/F_{22}$")
            plt.title("Anisotropy Ratio")
            plt.legend()
            plt.ylim(0, 2 * expected_ratio)  # Limit y-axis for better visualization
            plt.tight_layout()
            plt.show()


########################################################################################
# END class definition
# BEGIN figure recreation code


def figure_3p1():
    "Recreates figure 3.1 from the simulation paper"

    config_base = {
        "sigma2": 2.0,  # m²/s²
        "L_2d": 15_000.0,  # m
        "psi": np.deg2rad(45.0),  # radians
        "z_i": 500.0,  # m
        "N1": 2**10,
        "N2": 2**7,
    }

    config_a = config_base.copy()
    config_a["L1_factor"] = 40
    config_a["L2_factor"] = 5

    config_b = config_base.copy()
    config_b["L1_factor"] = 40
    config_b["L2_factor"] = 10

    accumulator_eq15_a = np.zeros((2**10, 2**7, 2))
    accumulator_eq16_a = np.zeros((2**10, 2**7, 2))
    accumulator_eq15_b = np.zeros((2**10, 2**7, 2))
    accumulator_eq16_b = np.zeros((2**10, 2**7, 2))

    # Build 10-realization average of config a for eq15
    config_a["equation"] = "eq15"

    model = Mann2DWindField(config_a)
    for _ in range(10):
        u = model.generate()
        accumulator_eq15_a += u
    accumulator_eq15_a /= 10

    # Build 10-realization average of config a for eq16
    config_a["equation"] = "eq16"

    model = Mann2DWindField(config_a)
    for _ in range(10):
        u = model.generate()
        accumulator_eq16_a += u
    accumulator_eq16_a /= 10

    # Build 10-realization average of config b for eq15
    config_b["equation"] = "eq15"

    model = Mann2DWindField(config_b)
    for _ in range(10):
        u = model.generate()
        accumulator_eq15_b += u
    accumulator_eq15_b /= 10

    # Build 10-realization average of config b for eq16
    config_b["equation"] = "eq16"

    model = Mann2DWindField(config_b)
    for _ in range(10):
        u = model.generate()
        accumulator_eq16_b += u
    accumulator_eq16_b /= 10

    # Plot the results

########################################################################################
# BEGIN tests


def test_spectral_tensor_isotropicSimp():
    config = {
        "sigma2": 2.0,  # m²/s²
        "L_2d": 5.0,  # m
        "psi": np.deg2rad(45.0),  # radians
        "z_i": 5.0,  # m
        "L1_factor": 40,  # Domain length = L1_factor * L_2d
        "L2_factor": 5,  # Domain length = L2_factor * L_2d
        "N1": 2**0,  # Grid points in x direction
        "N2": 2**0,  # Grid points in y direction
        "equation": "eq15",  # Which equation to use
    }
    model = Mann2DWindField(config)

    model.k_mag = 0
    model.kappa = 0
    model.k1 = 1
    model.k2 = -1

    p11, p12, p22 = model.isotropic_attenuated_spectral_tensor()

    expected_common_factor = model.c * (model.L_2d**(14 / 3)) / np.pi

    print("\n")
    print("EXPECTED common_factor = ", expected_common_factor)
    print("EXPECTED P11 = ", -1 * (model.k1**2))
    print("EXPECTED P12 = ", -1 * (model.k1 * model.k2))
    print("EXPECTED P22 = ", -1 * (model.k2**2))
    print("<isotropic_attenuated_spectral_tensor> Phi11 ", p11)
    print("<isotropic_attenuated_spectral_tensor> Phi12 ", p12)
    print("<isotropic_attenuated_spectral_tensor> Phi22 ", p22)
    print("\n\n")


# TODO: Get outputs from Mann-Syed code via Docker/vm
########################################################################################
# END figure recreation code
# BEGIN driver
if __name__ == "__main__":
    # Configuration for Figure 2a from the paper

    config = {
        "sigma2": 2.0,  # m²/s²
        "L_2d": 15_000.0,  # m
        "psi": np.deg2rad(45.0),  # radians
        "z_i": 500.0,  # m
        "L1_factor": 40,  # Domain length = L1_factor * L_2d
        "L2_factor": 5,  # Domain length = L2_factor * L_2d
        "N1": 2**8,  # Grid points in x direction
        "N2": 2**5,  # Grid points in y direction
        # "N1": 2**10,  # Grid points in x direction
        # "N2": 2**7,  # Grid points in y direction
        "equation": "eq15",  # Which equation to use
    }

    # Create model
    model = Mann2DWindField(config)

    # Generate wind field
    t_start = time.time()
    u = model.generate()
    print(f"Generation completed in {time.time() - t_start:.2f} seconds")

    # test_spectral_tensor_isotropicSimp()

    # Plot results
    model.plot_field()
    # model.plot_field(smooth=True, sigma=15)
    # model.plot_spectrum()
    # figure_3p1()
    # model.plot_visualization_panel()
