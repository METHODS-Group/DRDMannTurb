import concurrent.futures

import matplotlib.pyplot as plt
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

        self.N1 = 2**config["N1"]
        self.N2 = 2**config["N2"]

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
        phi_ = self.c0 * np.cbrt(self.epsilon) * \
            np.sqrt((self.L / np.sqrt(1 + (k_mag_sq * self.L**2)))**(17/3) / (4 * np.pi))

        # TODO: OLD
        # C1 = 1j * phi_ * self.k2
        # C2 = 1j * phi_ * (-1 * self.k1)

        # TODO: NEW
        C1 = 1j * phi_ * self.k2 #/ k_mag_sq_safe
        C2 = 1j * phi_ * (-1 * self.k1) #/ k_mag_sq_safe

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

    def compute_spectrum(self, u1 = None, u2 = None):
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

        u1_fft = np.fft.fft2(u1) / (self.dx * self.dy)
        u2_fft = np.fft.fft2(u2) / (self.dx * self.dy)

        k1_pos_mask = self.k1_fft > 0
        k1_pos = self.k1_fft[k1_pos_mask]

        power_u1 = (np.abs(u1_fft) / (self.N1 * self.N2))**2
        power_u2 = (np.abs(u2_fft) / (self.N1 * self.N2))**2

        F11 = np.zeros_like(k1_pos)
        F22 = np.zeros_like(k1_pos)

        for i, k1_val in enumerate(k1_pos):
            mask = (self.k1 == k1_val)
            F11[i] = np.mean(power_u1[mask]) * self.dy
            F22[i] = np.mean(power_u2[mask]) * self.dy

        return k1_pos, F11, F22


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
        F11_constant = (self.c0**2 * self.epsilon**(2/3) * self.L**(8/3) * scipy.special.gamma(4/3))\
            / (8 * np.sqrt(np.pi) * scipy.special.gamma(17/6))

        F11 = F11_constant / (1 + (self.L * k1_arr)**2)**(4/3)

        F22_constant = (self.c0**2 * self.epsilon**(2/3) * self.L**(14/3) * scipy.special.gamma(7/3))\
            / (4 * np.sqrt(np.pi) * scipy.special.gamma(17/6))

        F22 = F22_constant * k1_arr**2 / ((1 + (self.L * k1_arr)**2)**(7/3))

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


def mesh_independence_study(low = 4, high = 15):

    print("="*80)
    print("MESH INDEPENDENCE STUDY")
    print("="*80)

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


def scale_independence_study(low = 0.5, high = 40, step = 0.5):

    print("="*80)
    print("SCALE INDEPENDENCE STUDY")
    print("="*80)

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
    aspect_ratio = dx/dy

    # Find global min/max for consistent colormap
    vmin = min(np.min(u1), np.min(u2))
    vmax = max(np.max(u1), np.max(u2))

    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Plot u1 (top)
    axs[0].pcolormesh(X, Y, u1, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axs[0].set_aspect(aspect_ratio)
    axs[0].set_ylabel('y [km]')
    axs[0].set_title('(a) u')

    # Plot u2 (bottom)
    im2 = axs[1].pcolormesh(X, Y, u2, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axs[1].set_aspect(aspect_ratio)
    axs[1].set_xlabel('x [km]')
    axs[1].set_ylabel('y [km]')
    axs[1].set_title('(b) v')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('$[m\,s^{-1}]$')

    # Add main title with more space and larger font
    plt.suptitle(title, y=0.95, fontsize=16)

    # Use tight_layout but preserve the space we created for the title
    plt.tight_layout(rect=[0, 0, 0.9, 0.88])  # Adjust the top value (0.88) to match subplots_adjust

    return fig, axs

def diagnostic_plot(u1, u2):
    x_coords = np.linspace(0, 60, u1.shape[1])
    y_coords = np.linspace(0, 15, u1.shape[0])

    plot_velocity_fields(u1, u2, x_coords, y_coords,
                                    title = "Von Karman Velocity field")
    plt.show()


# ------------------------------------------------------------------------------------------------ #
# Spectrum plot

def plot_spectrum_comparison(config: dict, num_realizations: int = 10):
    """
    Generate velocity fields and plot their spectra compared to analytical spectrum

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

    k1_pos = None
    F11_avg = None
    F22_avg = None

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

    fig, axs = plt.subplots(1, 2, figsize = (12, 5))

    # Plot F11 estimated and analytical
    axs[0].loglog(k1_custom * config["L"], k1_custom * F11_analytical, 'k-', label = "Analytical F11")
    axs[0].loglog(k1_pos * config["L"], k1_pos * F11_avg, 'r--', label = "Simulated F11")
    axs[0].set_xlabel(r"$k_1 L$ [-]")
    axs[0].set_ylabel(r"$k_1 F_{11}(k_1)$ [m$^2$s$^{-2}$]")
    axs[0].set_title("F11 spectrum")
    axs[0].grid(True, which='both', ls='-', alpha=0.2)
    axs[0].legend()

    # Plot F22 estimated and analytical
    axs[1].loglog(k1_custom * config["L"], k1_custom * F22_analytical, 'k-', label = "Analytical F22")
    axs[1].loglog(k1_pos * config["L"], k1_pos * F22_avg, 'r--', label = "Simulated F22")
    axs[1].set_xlabel(r"$k_1 L$ [-]")
    axs[1].set_ylabel(r"$k_1 F_{22}(k_1)$ [m$^2$s$^{-2}$]")
    axs[1].set_title("F22 spectrum")
    axs[1].grid(True, which='both', ls='-', alpha=0.2)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return fig, axs


def plot_spectrum(config: dict, num_realizations=10):
    """
    Generate and plot spectra with better handling of log scales.

    Parameters:
    - config: Configuration dictionary
    - num_realizations: Number of realizations to average
    """
    # Create generator
    gen = generator(config)

    # Initialize arrays for averaging
    u1_avg_freq = None
    u2_avg_freq = None

    # Generate and average multiple realizations
    for _ in range(num_realizations):
        u1, u2 = gen.generate()

        # Compute FFT
        u1_freq = np.fft.fft2(u1) * gen.dx * gen.dy
        u2_freq = np.fft.fft2(u2) * gen.dx * gen.dy

        if u1_avg_freq is None:
            u1_avg_freq = u1_freq
            u2_avg_freq = u2_freq
        else:
            u1_avg_freq += u1_freq
            u2_avg_freq += u2_freq

    u1_avg_freq /= num_realizations
    u2_avg_freq /= num_realizations

    # Obtain estimated spectrum
    k1_fft_pos_mask = gen.k1_fft > 0
    k1_ = gen.k1_fft[k1_fft_pos_mask]

    # Compute power spectra
    power_u1 = np.zeros((len(k1_), gen.N2))
    power_u2 = np.zeros((len(k1_), gen.N2))

    for i, k1_val in enumerate(k1_):
        indices = np.where(gen.k1 == k1_val)
        power_u1[i] = np.abs(u1_avg_freq[indices])**2 / (gen.N1 * gen.N2)**2
        power_u2[i] = np.abs(u2_avg_freq[indices])**2 / (gen.N1 * gen.N2)**2

    # Average along k2 direction
    F11_approx = np.mean(power_u1, axis=1) * gen.dy
    F22_approx = np.mean(power_u2, axis=1) * gen.dy

    # Compute analytical spectrum
    F11_analytical, F22_analytical = gen.analytical_spectrum(k1_)

    # Plot spectra
    plt.figure(figsize=(10, 6))

    # Plot k1*F11 and k1*F22 for easier comparison with literature
    plt.loglog(k1_ * gen.L, k1_ * F11_approx, 'r--', label='Estimated F11')
    plt.loglog(k1_ * gen.L, k1_ * F22_approx, 'b--', label='Estimated F22')
    plt.loglog(k1_ * gen.L, k1_ * F11_analytical, 'r-', label='Analytical F11')
    plt.loglog(k1_ * gen.L, k1_ * F22_analytical, 'b-', label='Analytical F22')

    plt.xlabel('$k_1 L$ [-]')
    plt.ylabel('$k_1 F(k_1)$ [$m^2s^{-2}$]')
    plt.title('Von Karman Spectra')
    plt.grid(True, which='both', ls='-', alpha=0.2)
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

    plot_spectrum(config)

    gen = generator(config)
    u1, u2 = gen.generate()
    diagnostic_plot(u1, u2)
    # plot_spectrum_comparison(config)

    # gen = generator(config)
    # u1, u2 = gen.generate()

    # k1_fft_pos_mask = gen.k1_fft > 0
    # k1_ = gen.k1_fft[k1_fft_pos_mask]

    # u1_freq_pos = u1[k1_fft_pos_mask]
    # u2_freq_pos = u2[k1_fft_pos_mask]

    # power_u1 = (np.abs(u1_freq_pos) / (gen.N1 * gen.N2))**2
    # power_u2 = (np.abs(u2_freq_pos) / (gen.N1 * gen.N2))**2

    # F11_est = np.mean(power_u1, axis = 1) * gen.dy
    # F22_est = np.mean(power_u2, axis = 1) * gen.dy

    # # Plot spectrum
    # plt.figure(figsize = (10, 6))
    # plt.semilogx(k1_, k1_ * F11_est, label = "Estimated F11")
    # plt.semilogx(k1_, k1_ * F22_est, label = "Estimated F22")
    # plt.legend()
    # plt.show()
