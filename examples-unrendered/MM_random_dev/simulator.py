import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore
from pretty_print import arr_debug, print_header, print_param, print_section
from scipy.fft import fftfreq, ifft2

"""
TRACKING:
- [ ] Check if dy is correct; we are integrating in fourier space
- [ ] Implement eq16 again
- [ ] Set up the comparison numerical integration plot
- [ ] Implement 10 realizations and plot average for F11, F22


For N1 = 2**10, N2 = 2**7,
    Field min is
    F11 values are
"""

################################
# Type
config_type = "figure2_a"

# Flags
plot_field = True
plot_spectra = True

param_sets = {
    "figure3_standard_eq14": {
        "sigma2": 0.6,  # m2/s2
        "L_2d": 15_000.0,  # m
        "psi": np.deg2rad(43.0),  # degrees
        "z_i": 500.0,  # m
        "L1_factor": 4,
        "L2_factor": 1,
        "N1": 2**10,
        "N2": 2**7,
        "equation": "eq14",
    },
    "figure3_fast_eq14": {
        "sigma2": 0.6,  # m2/s2
        "L_2d": 15_000.0,  # m
        "psi": np.deg2rad(43.0),  # degrees
        "z_i": 500.0,  # m
        "L1_factor": 4,
        "L2_factor": 1,
        "N1": 2**8,
        "N2": 2**5,
        "equation": "eq14",
    },
    "figure2_a": {
        "sigma2": 2.0,  # m2/s2
        "L_2d": 15_000.0,  # m
        "psi": np.deg2rad(45.0),  # degrees
        "z_i": 500.0,  # m
        "L1_factor": 40,
        "L2_factor": 5,
        "N1": 2**10,
        "N2": 2**7,
        "equation": "eq15",
    },
    "figure2_b": {
        "sigma2": 2.0,  # m2/s2
        "L_2d": 15_000.0,  # m
        "psi": np.deg2rad(45.0),  # degrees
        "z_i": 500.0,  # m
        "L1_factor": 1,
        "L2_factor": 0.125,
        "N1": 2**10,
        "N2": 2**7,
        "equation": "eq16",
    },
}

params = param_sets[config_type]

# Physical params
sigma2 = params["sigma2"]
L_2d = params["L_2d"]
psi = params["psi"]
z_i = params["z_i"]

c = (8 * sigma2) / (9 * L_2d ** (2 / 3))

# Domain params
L1 = params["L1_factor"] * L_2d
L2 = params["L2_factor"] * L_2d
N1 = params["N1"]
N2 = params["N2"]

dx = L1 / N1
dy = L2 / N2

equation = params["equation"]

# Replace your print statements with these prettier versions
print_header("WIND FIELD SIMULATOR")

print_section("Physical Parameters")
print_param("config_type", config_type)
print_param("sigma2", f"{sigma2:.2f}", "m²/s²")
print_param("L_2d", f"{L_2d:.2f}", "m")
print_param("psi", f"{np.rad2deg(psi):.2f}", "degrees")
print_param("z_i", f"{z_i:.2f}", "m")
print_param("c", f"{c:.4f}")

print_section("Domain Parameters")
print_param("L1", f"{L1:.2f}", "m")
print_param("L2", f"{L2:.2f}", "m")
print_param("N1", N1)
print_param("N2", N2)
print_param("dx", f"{dx:.2f}", "m")
print_param("dy", f"{dy:.2f}", "m")

print_section("Problem Configuration")
print_param("Using equation", f"{equation}")
print_param("Plot field", f"{Fore.GREEN if plot_field else Fore.RED}{plot_field}")
print_param("Plot spectra", f"{Fore.GREEN if plot_spectra else Fore.RED}{plot_spectra}")

###############################################################################################################
###############################################################################################################
# Script begin

#########################################
# Simulate field
#########################################
# Create wavenumber arrays

k1_arr = fftfreq(N1, dx)
k2_arr = fftfreq(N2, dy)

k1, k2 = np.meshgrid(k1_arr, k2_arr, indexing="ij")

# Calculate kappa and k_mag
kappa = np.sqrt(2 * ((k1 * np.cos(psi)) ** 2 + (k2 * np.sin(psi)) ** 2))
k_mag = np.sqrt(k1**2 + k2**2)

# Calculate E_kappa_attenuated
E_kappa_attenuated = c * (kappa**3) / (((L_2d**-2) + kappa**2) ** (7 / 3) * (1 + (kappa * z_i) ** 2))

# Calculate spectral tensor common factor
kappa_mask = np.isclose(kappa, 0.0)
k_mag_mask = np.isclose(k_mag, 0.0)

phi_common = np.zeros_like(kappa, dtype=float)

for i in range(N1):
    for j in range(N2):
        if not k_mag_mask[i, j]:
            phi_common[i, j] = E_kappa_attenuated[i, j] / (np.pi * k_mag[i, j])

# Calculate spectral tensor
phi_11 = np.zeros_like(phi_common, dtype=float)
phi_12 = np.zeros_like(phi_common, dtype=float)
phi_22 = np.zeros_like(phi_common, dtype=float)

for i in range(N1):
    for j in range(N2):
        if not k_mag_mask[i, j]:  # Only calculate when k_mag is not zero
            phi_11[i, j] = phi_common[i, j] * (1 - (k1[i, j] / k_mag[i, j]) ** 2)
            phi_12[i, j] = phi_common[i, j] * (-1 * (k1[i, j] * k2[i, j]) / k_mag[i, j] ** 2)
            phi_22[i, j] = phi_common[i, j] * (1 - (k2[i, j] / k_mag[i, j]) ** 2)

# Obtain Fourier coefficients
C_11 = np.zeros_like(phi_11, dtype=complex)
C_22 = np.zeros_like(phi_22, dtype=complex)
C_12 = np.zeros_like(phi_12, dtype=complex)

########################################################################################################
# Begin eq14
if equation == "eq14":
    # TODO: implement eq14 CORRECTLY

    def sinc2(x: float) -> float:
        if np.isclose(x, 0.0):
            return 1.0
        else:
            return np.sin(x) ** 2 / x**2

    n_int_k1 = 21
    n_int_k2 = 21

    for i in range(N1):
        for j in range(N2):
            if i % 50 == 0 and j == 0:
                print(f"Processing wavenumber {i}/{N1}, {j}/{N2}")

            k1_target = k1[i, j]
            k2_target = k2[i, j]

            # Skip if k_mag is zero (DC component)
            if k_mag_mask[i, j]:
                continue

            # Define integration range centered on the target wavenumber
            # The range should cover the main lobe of the sinc² function
            dk1_range = 2 * np.pi / L1 * 4  # Cover 4 periods
            dk2_range = 2 * np.pi / L2 * 4

            k1_min = k1_target - dk1_range / 2
            k1_max = k1_target + dk1_range / 2
            k2_min = k2_target - dk2_range / 2
            k2_max = k2_target + dk2_range / 2

            # Create integration grid
            k1_int = np.linspace(k1_min, k1_max, n_int_k1)
            k2_int = np.linspace(k2_min, k2_max, n_int_k2)
            dk1_int = (k1_max - k1_min) / (n_int_k1 - 1)
            dk2_int = (k2_max - k2_min) / (n_int_k2 - 1)

            ###################################
            # Integrate
            integral_11 = 0.0
            integral_22 = 0.0
            integral_12 = 0.0

            for k1_prime_idx, k1_prime in enumerate(k1_int):
                for k2_prime_idx, k2_prime in enumerate(k2_int):
                    i_prime = np.argmin(np.abs(k1_arr - k1_prime))
                    j_prime = np.argmin(np.abs(k2_arr - k2_prime))

                    if 0 <= i_prime < N1 and 0 <= j_prime < N2:
                        phi_11_val = phi_11[i_prime, j_prime]
                        phi_22_val = phi_22[i_prime, j_prime]
                        phi_12_val = phi_12[i_prime, j_prime]

                        sinc2_k1 = sinc2((k1_target - k1_prime) * L1 / 2)
                        sinc2_k2 = sinc2((k2_target - k2_prime) * L2 / 2)

                        sinc2_product = sinc2_k1 * sinc2_k2

                        integral_11 += phi_11_val * sinc2_product * dk1_int * dk2_int
                        integral_22 += phi_22_val * sinc2_product * dk1_int * dk2_int
                        integral_12 += phi_12_val * sinc2_product * dk1_int * dk2_int

            # Calculate C_ij values
            C_11[i, j] = np.sqrt(integral_11 + 0j)
            C_22[i, j] = np.sqrt(integral_22 + 0j)

            if integral_12 != 0:
                # sign = np.sign(integral_12)
                sign = 1
                C_12[i, j] = sign * np.sqrt(abs(integral_12) + 0j)


########################################################################################################
# Begin eq15
elif equation == "eq15":
    C_11 = np.sqrt((2 * np.pi) ** 2 / (L1 * L2) * phi_11 + 0j)
    C_22 = np.sqrt((2 * np.pi) ** 2 / (L1 * L2) * phi_22 + 0j)

    for i in range(N1):
        for j in range(N2):
            if not k_mag_mask[i, j]:
                sign = np.sign(phi_12[i, j])
                C_12[i, j] = sign * np.sqrt(abs((2 * np.pi) ** 2 / (L1 * L2) * phi_12[i, j]) + 0j)

########################################################################################################
# Begin eq16
elif equation == "eq16":
    # TODO: implement eq16 CORRECTLY
    pass


# Generate Gaussian white noise
eta_1 = np.random.normal(0, 1, size=(N1, N2)) + 1j * np.random.normal(0, 1, size=(N1, N2))
eta_2 = np.random.normal(0, 1, size=(N1, N2)) + 1j * np.random.normal(0, 1, size=(N1, N2))
eta_1 /= np.sqrt(2)
eta_2 /= np.sqrt(2)


# Calculate field
u1 = np.real(ifft2((C_11 * eta_1) + (C_12 * eta_2)))  # Longitudinal component
u2 = np.real(ifft2((C_12 * eta_1) + (C_22 * eta_2)))  # Transverse component

arr_debug(u1, "u1", plot_heatmap=False)
arr_debug(u2, "u2", plot_heatmap=False)


# Verify total variance (should match sigma2)
var_u1 = np.var(u1)
var_u2 = np.var(u2)

print_section("Variance Verification")
print_param("Variance of u1", f"{var_u1:.8f}", "m²/s²")
print_param("Variance of u2", f"{var_u2:.8f}", "m²/s²")
print_param("Target variance", f"{sigma2:.4f}", "m²/s²")
print_param("Ratio u1/target", f"{var_u1/sigma2:.4f}")
print_param("Ratio u2/target", f"{var_u2/sigma2:.4f}")

# Now plot
if plot_field:
    plt.figure(figsize=(10, 6), dpi=100)

    x_km = np.linspace(0, L1 / 1000, N1)
    y_km = np.linspace(0, L2 / 1000, N2)
    X_km, Y_km = np.meshgrid(x_km, y_km, indexing="ij")

    plt.subplot(211)
    im1 = plt.pcolormesh(X_km.T, Y_km.T, u1.T, cmap="RdBu_r", shading="auto")
    cbar1 = plt.colorbar(im1, label="[m s$^{-1}$]")
    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    plt.title("(a) u")

    plt.subplot(212)
    im2 = plt.pcolormesh(X_km.T, Y_km.T, u2.T, cmap="RdBu_r", shading="auto")
    cbar2 = plt.colorbar(im2, label="[m s$^{-1}$]")
    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    plt.title("(b) v")

    plt.tight_layout()
    plt.show()

exit()

#####################################################################################################
#####################################################################################################
# Calculate spectra

# FT u1 and u2
u1_fft = np.fft.fft2(u1)
u2_fft = np.fft.fft2(u2)

# 2d psd and normalize
psd_u1 = np.abs(u1_fft) ** 2 / (N1 * N2)
psd_u2 = np.abs(u2_fft) ** 2 / (N1 * N2)

# Integrate over k2
F11 = np.mean(psd_u1, axis=1) * dy  # TODO: Check if dy is correct; we are integrating in fourier space
F22 = np.mean(psd_u2, axis=1) * dy

positive_mask = k1_arr >= 0
k1_positive = k1_arr[positive_mask]  # Positive wavenumbers

F11_positive = F11[positive_mask]
F22_positive = F22[positive_mask]

k1_F11 = k1_positive * F11_positive
k1_F22 = k1_positive * F22_positive

if plot_spectra:
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(L_2d * k1_positive, k1_F11, "b-", linewidth=1.5, label="$k_1 F_{11}(k_1)$")
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_xlabel("$k_1$ (rad/m)")
    ax1.set_ylabel("$k_1 F_{11}(k_1)$ (m$^2$/s$^2$)")
    ax1.set_title("F11")
    ax1.legend()

    ax2.loglog(L_2d * k1_positive, k1_F22, "b-", linewidth=1.5, label="$k_1 F_{22}(k_1)$")
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.set_xlabel("$k_1$ (rad/m)")
    ax2.set_ylabel("$k_1 F_{22}(k_1)$ (m$^2$/s$^2$)")
    ax2.set_title("F22")
    ax2.legend()

    plt.tight_layout()
    plt.show()
