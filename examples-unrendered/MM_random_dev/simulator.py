import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftfreq, ifft2

# Physical params
sigma2 = 2.0
L_2d = 15000.0
psi = np.deg2rad(45.0)
z_i = 500.0

c = (8 * sigma2) / (9 * L_2d ** (2 / 3))

# Domain params
L1 = 40 * L_2d
L2 = 5 * L_2d
N1 = 2**10
N2 = 2**7

dx = L1 / N1
dy = L2 / N2


################################
# Flags
plot_field = True
plot_spectra = True

use_eq15 = True  # If False, use eq16


print("Physical parameters:")
print(f"\t sigma2 = {sigma2:.2f}")
print(f"\t L_2d = {L_2d:.2f}")
print(f"\t psi = {psi:.2f}")
print(f"\t z_i = {z_i:.2f}")

print(f"Obtained c: {c:.2f}")

print("Domain parameters:")
print(f"\t L1 = {L1:.2f}")
print(f"\t L2 = {L2:.2f}")
print(f"\t N1 = {N1}")
print(f"\t N2 = {N2}")

print("Problem parameters:")
print(f"\t Use Equation 15 = {use_eq15}")
print(f"\t Plot field = {plot_field}")
print(f"\t Plot spectra = {plot_spectra}")


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

_intermediate = E_kappa_attenuated / np.pi
phi_common = np.zeros_like(kappa, dtype=float)

for i in range(N1):
    for j in range(N2):
        if not k_mag_mask[i, j]:
            phi_common[i, j] = _intermediate[i, j] / k_mag[i, j]

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

if use_eq15:
    C_11 = np.sqrt((2 * np.pi) ** 2 / (L1 * L2) * phi_11 + 0j)
    C_22 = np.sqrt((2 * np.pi) ** 2 / (L1 * L2) * phi_22 + 0j)

    for i in range(N1):
        for j in range(N2):
            if not k_mag_mask[i, j]:
                sign = np.sign(phi_12[i, j])
                C_12[i, j] = sign * np.sqrt(abs((2 * np.pi) ** 2 / (L1 * L2) * phi_12[i, j]) + 0j)

else:
    # TODO: Implement this
    C_11 = np.sqrt((2 * np.pi) ** 2 / (L1 * L2) * phi_11 + 0j)
    C_22 = np.sqrt((2 * np.pi) ** 2 / (L1 * L2) * phi_22 + 0j)
    C_12 = np.zeros_like(phi_12)

    for i in range(N1):
        for j in range(N2):
            if k_mag_mask[i, j]:
                # C_12[i, j] = ((2 * np.pi)**2 / (L1 * L2) * phi_12[i, j])
                C_12[i, j] = 1

exit()

# Generate Gaussian white noise
eta_1 = np.random.normal(0, 1, size=(N1, N2)) + 1j * np.random.normal(0, 1, size=(N1, N2))
eta_2 = np.random.normal(0, 1, size=(N1, N2)) + 1j * np.random.normal(0, 1, size=(N1, N2))
eta_1 /= np.sqrt(2)
eta_2 /= np.sqrt(2)


# Calculate field
u1 = np.real(ifft2((C_11 * eta_1) + (C_12 * eta_2)))  # Longitudinal component
u2 = np.real(ifft2((C_12 * eta_1) + (C_22 * eta_2)))  # Transverse component

# Verify total variance (should match sigma2)
var_u1 = np.var(u1)
var_u2 = np.var(u2)
print(f"Variance of u1: {var_u1:.8f}, Variance of u2: {var_u2:.8f}")
print(f"Target variance: {sigma2:.4f}")

# Now plot
if plot_field:
    plt.figure(figsize=(12, 5))

    # Longitudinal (u) wind field - subplot (a)
    plt.subplot(211)  # 1 row, 2 columns, 1st plot
    plt.imshow(u1.T, cmap="coolwarm")
    plt.colorbar(label="m/s")
    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    plt.title("(a) u")

    # Transverse (v) wind field - subplot (b)
    plt.subplot(212)  # 1 row, 2 columns, 2nd plot
    plt.imshow(u2.T, cmap="coolwarm")
    plt.colorbar(label="m/s")
    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    plt.title("(b) v")

    plt.tight_layout()
    plt.show()


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

print("k1_arr.shape: ", k1_arr.shape)
print("k1.shape: ", k1.shape)

print("u1_fft.shape: ", u1_fft.shape)
print("psd_u1.shape: ", psd_u1.shape)
print("F11.shape: ", F11.shape)

print("k1_positive.shape: ", k1_positive.shape)
print("positive_mask.shape: ", positive_mask.shape)

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
