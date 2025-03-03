import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


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
        self.sigma2 = config.get("sigma2", 2.0)
        self.L_2d = config.get("L_2d", 15000.0)
        self.psi = config.get("psi", np.pi / 4)  # 45 degrees by default
        self.z_i = config.get("z_i", 500.0)

        # Computational parameters
        self.L1 = config.get("L1_factor", 40) * self.L_2d
        self.L2 = config.get("L2_factor", 5) * self.L_2d
        self.N1 = config.get("N1", 2**10)
        self.N2 = config.get("N2", 2**7)
        self.equation = config.get("equation", "eq15")

        # Calculate grid spacing
        self.dx = self.L1 / self.N1
        self.dy = self.L2 / self.N2

        # Calculate c coefficient from sigma2 and L_2d (equation 3)
        self.c = (8.0 * self.sigma2) / (9.0 * self.L_2d ** (2.0 / 3.0))

        # Initialize grid
        self._initialize_grid()

        # Initialize results
        self.u1 = None
        self.u2 = None
        self.u1_freq = None
        self.u2_freq = None

    def _initialize_grid(self):
        """Initialize the computational grid in physical and wavenumber space."""
        # Physical grid
        self.x = np.linspace(0, self.L1, self.N1)
        self.y = np.linspace(0, self.L2, self.N2)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Wavenumber grid
        k1_fft = 2 * np.pi * np.fft.fftfreq(self.N1, self.dx)
        k2_fft = 2 * np.pi * np.fft.fftfreq(self.N2, self.dy)
        self.k1, self.k2 = np.meshgrid(k1_fft, k2_fft, indexing="ij")

        self.k_mag = np.sqrt(self.k1**2 + self.k2**2)

        self.kappa = np.sqrt(2 * ((self.k1 * np.cos(self.psi)) ** 2 + (self.k2 * np.sin(self.psi)) ** 2))

        self.k_mag_mask = np.isclose(self.k_mag, 0.0)
        self.kappa_mask = np.isclose(self.kappa, 0.0)

    def _calculate_energy_spectrum(self):
        """Calculate the energy spectrum with anisotropy parameter (equation 5)."""
        E_kappa = self.c * (self.kappa**3) / ((self.L_2d**-2 + self.kappa**2) ** (7 / 3))
        E_kappa_attenuated = E_kappa / (1 + (self.kappa * self.z_i) ** 2)

        # E_kappa_attenuated[self.kappa_mask] = 0.0

        return E_kappa_attenuated

    def _calculate_spectral_tensor(self, E_kappa):
        """Calculate the spectral tensor components (equation 1)."""
        phi_common = np.zeros_like(self.kappa)
        idx = ~self.k_mag_mask
        phi_common[idx] = E_kappa[idx] / (np.pi * self.k_mag[idx])

        phi_11 = np.zeros_like(phi_common)
        phi_12 = np.zeros_like(phi_common)
        phi_22 = np.zeros_like(phi_common)

        idx = ~self.k_mag_mask
        phi_11[idx] = phi_common[idx] * (1 - (self.k1[idx] / self.k_mag[idx]) ** 2)
        phi_12[idx] = phi_common[idx] * (-1 * self.k1[idx] * self.k2[idx] / self.k_mag[idx] ** 2)
        phi_22[idx] = phi_common[idx] * (1 - (self.k2[idx] / self.k_mag[idx]) ** 2)

        return phi_11, phi_12, phi_22

    def _calculate_fourier_coefficients(self, phi_11, phi_12, phi_22):
        """Calculate Fourier coefficients based on spectral tensor."""
        C_11 = np.zeros_like(phi_11, dtype=complex)
        C_22 = np.zeros_like(phi_22, dtype=complex)
        C_12 = np.zeros_like(phi_12, dtype=complex)

        if self.equation == "eq14":
            # TODO:
            raise NotImplementedError("Equation 14 not yet implemented")

        elif self.equation == "eq15":
            """
            Simplified approximation (equation 15) for L_i >> L_2d
            """
            norm_factor = (2 * np.pi) ** 2 / (self.L1 * self.L2)
            C_11 = np.sqrt(norm_factor * phi_11 + 0j)
            C_22 = np.sqrt(norm_factor * phi_22 + 0j)

            idx = ~self.k_mag_mask
            C_12[idx] = np.sign(phi_12[idx]) * np.sqrt(abs(norm_factor * phi_12[idx]) + 0j)

        elif self.equation == "eq16":
            # TODO:
            raise NotImplementedError("Equation 16 not yet implemented")

        return C_11, C_12, C_22

    def _generate_wind_field(self, C_11, C_12, C_22):
        """Generate the wind field using Fourier synthesis with random phases."""

        # TODO: should be normalized by some spatial factor, won't fix things totally, but may help
        #       Convince myself of the need

        eta_1 = np.random.normal(0, 1, size=(self.N1, self.N2)) + 1j * np.random.normal(0, 1, size=(self.N1, self.N2))
        eta_2 = np.random.normal(0, 1, size=(self.N1, self.N2)) + 1j * np.random.normal(0, 1, size=(self.N1, self.N2))
        eta_1 /= np.sqrt(2)  # Divide by sqrt(2) to get unit variance for complex numbers
        eta_2 /= np.sqrt(2)

        # NOTE:

        u1_freq = (C_11 * eta_1) + (C_12 * eta_2)
        u2_freq = (C_12 * eta_1) + (C_22 * eta_2)

        # NOTE: numpy's ifft2 includes 1/(N1*N2) normalization
        u1 = np.real(np.fft.ifft2(u1_freq)) * self.N1 * self.N2
        u2 = np.real(np.fft.ifft2(u2_freq)) * self.N1 * self.N2

        return u1, u2, u1_freq, u2_freq

    def generate(self):
        """Generate the 2D wind field and return the velocity components."""
        t_start = time.time()
        print(f"Generating 2D wind field using {self.equation}...")

        # Calculate energy spectrum
        E_kappa = self._calculate_energy_spectrum()

        # Calculate spectral tensor components
        phi_11, phi_12, phi_22 = self._calculate_spectral_tensor(E_kappa)

        # Calculate Fourier coefficients
        C_11, C_12, C_22 = self._calculate_fourier_coefficients(phi_11, phi_12, phi_22)

        # Generate wind field
        self.u1, self.u2, self.u1_freq, self.u2_freq = self._generate_wind_field(C_11, C_12, C_22)

        # Verify variance
        var_u1 = np.var(self.u1)
        var_u2 = np.var(self.u2)

        print(f"Generation completed in {time.time() - t_start:.2f} seconds")
        print(f"Variance u1: {var_u1:.6f} m²/s² (target: {self.sigma2:.6f} m²/s²)")
        print(f"Variance u2: {var_u2:.6f} m²/s² (target: {self.sigma2:.6f} m²/s²)")

        # Scale to match target variance
        # if abs(var_u1 - self.sigma2) / self.sigma2 > 0.05 or abs(var_u2 - self.sigma2) / self.sigma2 > 0.05:
        #     print("Applying variance correction...")
        #     scale_u1 = np.sqrt(self.sigma2 / var_u1) if var_u1 > 0 else 1.0
        #     scale_u2 = np.sqrt(self.sigma2 / var_u2) if var_u2 > 0 else 1.0

        #     self.u1 *= scale_u1
        #     self.u2 *= scale_u2

        #     print(f"Scaling factors: u1={scale_u1:.4f}, u2={scale_u2:.4f}")
        #     print(f"Corrected variance u1: {np.var(self.u1):.6f} m²/s²")
        #     print(f"Corrected variance u2: {np.var(self.u2):.6f} m²/s²")

        return self.u1, self.u2

    def plot_field(self, smooth=False, sigma=10):
        """Plot the generated wind field."""
        if self.u1 is None or self.u2 is None:
            print("No wind field generated yet. Call generate() first.")
            return

        plt.figure(figsize=(12, 10))

        x_km = self.x / 1000
        y_km = self.y / 1000
        X_km, Y_km = np.meshgrid(x_km, y_km, indexing="ij")

        if smooth:
            u1_plot = gaussian_filter(self.u1, sigma=sigma)
            u2_plot = gaussian_filter(self.u2, sigma=sigma)
            title_suffix = f" (Smoothed σ={sigma})"
        else:
            u1_plot = self.u1
            u2_plot = self.u2
            title_suffix = ""

        plt.subplot(2, 1, 1)
        im1 = plt.pcolormesh(
            X_km, Y_km, u1_plot, cmap="RdBu_r", shading="auto", vmin=-3 * np.std(self.u1), vmax=3 * np.std(self.u1)
        )
        plt.colorbar(im1, label="u [m/s]")
        plt.xlabel("x [km]")
        plt.ylabel("y [km]")
        plt.title(f"Longitudinal component (u){title_suffix}")

        # Plot u2 component
        plt.subplot(2, 1, 2)
        im2 = plt.pcolormesh(
            X_km, Y_km, u2_plot, cmap="RdBu_r", shading="auto", vmin=-3 * np.std(self.u2), vmax=3 * np.std(self.u2)
        )
        plt.colorbar(im2, label="v [m/s]")
        plt.xlabel("x [km]")
        plt.ylabel("y [km]")
        plt.title(f"Transverse component (v){title_suffix}")

        plt.tight_layout()
        plt.show()

    def plot_spectrum(self):
        """Plot the energy spectrum."""
        if self.u1_freq is None:
            print("No wind field generated yet. Call generate() first.")
            return

        # Calculate 1D spectra
        k1_1d = 2 * np.pi * np.fft.fftfreq(self.N1, self.dx)
        positive_idx = k1_1d > 0
        k1_1d = k1_1d[positive_idx]

        # Calculate power spectrum (average over k2)
        power_u1 = np.abs(self.u1_freq) ** 2 / (self.N1 * self.N2) ** 2
        power_u2 = np.abs(self.u2_freq) ** 2 / (self.N1 * self.N2) ** 2

        # Average over k2 to get 1D spectrum
        spectrum_u1 = np.mean(power_u1, axis=1)[positive_idx] * self.dy
        spectrum_u2 = np.mean(power_u2, axis=1)[positive_idx] * self.dy

        # Theoretical spectrum
        # E_kappa = self._calculate_energy_spectrum()
        # phi_11, phi_12, phi_22 = self._calculate_spectral_tensor(E_kappa)

        # # Extract 1D theoretical spectrum (central values)
        # mid_idx = self.N2 // 2
        # theo_idx = np.logical_and(positive_idx, ~self.kappa_mask[0:self.N1, mid_idx])
        # k1_theo = k1_1d[theo_idx[positive_idx]]
        # spectrum_theo = phi_11[0:self.N1, mid_idx][theo_idx[0:self.N1]]

        plt.figure(figsize=(10, 6))
        plt.loglog(k1_1d, k1_1d * spectrum_u1, "b-", label="$k_1 F_{11}(k_1)$ (simulated)")
        plt.loglog(k1_1d, k1_1d * spectrum_u2, "r-", label="$k_1 F_{22}(k_1)$ (simulated)")

        # if len(k1_theo) > 0:
        #     plt.loglog(k1_theo, k1_theo * spectrum_theo, 'k--', label='Theory')

        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel("$k_1$ [rad/m]")
        plt.ylabel("$k_1 F(k_1)$ [m²/s²]")
        plt.title("1D Energy Spectrum")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Also plot the ratio F11/F22 to check anisotropy
        ratio = spectrum_u1 / (spectrum_u2 + 1e-10)
        expected_ratio = 3 / 5 * (1 / np.cos(self.psi) ** 2)  # For psi=45°, should be 3/5

        plt.figure(figsize=(10, 6))
        plt.semilogx(k1_1d, ratio, "b-", label="$F_{11}/F_{22}$ (simulated)")
        plt.axhline(expected_ratio, color="r", linestyle="--", label=f"Expected ratio: {expected_ratio:.3f}")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel("$k_1$ [rad/m]")
        plt.ylabel("$F_{11}/F_{22}$")
        plt.title("Anisotropy Ratio")
        plt.legend()
        plt.ylim(0, 5)  # Limit y-axis for better visualization
        plt.tight_layout()
        plt.show()

    def plot_visualization_panel(self):
        """Create a panel of visualizations with different smoothing levels."""
        if self.u1 is None or self.u2 is None:
            print("No wind field generated yet. Call generate() first.")
            return

        plt.figure(figsize=(15, 10))

        # Convert to km for plotting
        x_km = self.x / 1000
        y_km = self.y / 1000
        X_km, Y_km = np.meshgrid(x_km, y_km, indexing="ij")

        # Apply different levels of smoothing
        smoothing_levels = [0, 5, 15, 30]  # No smoothing, light, medium, heavy
        for i, sigma in enumerate(smoothing_levels):
            plt.subplot(2, 4, i + 1)
            if sigma == 0:
                field_smooth = self.u1
                title = "u - Raw"
            else:
                field_smooth = gaussian_filter(self.u1, sigma=sigma)
                title = f"u - sigma={sigma}"

            im = plt.pcolormesh(
                X_km,
                Y_km,
                field_smooth,
                cmap="RdBu_r",
                shading="auto",
                vmin=-3 * np.std(self.u1),
                vmax=3 * np.std(self.u1),
            )
            plt.colorbar(im, label="[m/s]")
            plt.xlabel("x [km]")
            plt.ylabel("y [km]")
            plt.title(title)

            plt.subplot(2, 4, i + 5)
            if sigma == 0:
                field_smooth = self.u2
                title = "v - Raw"
            else:
                field_smooth = gaussian_filter(self.u2, sigma=sigma)
                title = f"v - sigma={sigma}"

            im = plt.pcolormesh(
                X_km,
                Y_km,
                field_smooth,
                cmap="RdBu_r",
                shading="auto",
                vmin=-3 * np.std(self.u2),
                vmax=3 * np.std(self.u2),
            )
            plt.colorbar(im, label="[m/s]")
            plt.xlabel("x [km]")
            plt.ylabel("y [km]")
            plt.title(title)

        plt.tight_layout()
        plt.show()


# Example usage
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
        # "N1": 2**10,             # Grid points in x direction
        # "N2": 2**7,              # Grid points in y direction
        "equation": "eq15",  # Which equation to use
    }

    # Create model
    model = Mann2DWindField(config)

    # Generate wind field
    u1, u2 = model.generate()

    # Plot results
    model.plot_field()
    model.plot_field(smooth=True, sigma=15)
    model.plot_spectrum()
    model.plot_visualization_panel()
