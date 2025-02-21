import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftfreq, ifft2
from seaborn import heatmap

# Parameters from the model
sigma2 = 2.0  # Variance of low-frequency fluctuations
L_2d = 15_000.0    # Mesoscale length scale (meters)
psi = np.deg2rad(45)       # Anisotropy parameter (angle in radians or as described)
z_i = 500.0    # Attenuation length (boundary-layer height, meters)

c = (8 * sigma2) / (9 * L_2d**(2/3))

# Domain parameters
L1, L2 = 40 * L_2d, 5 * L_2d
N1, N2 = 2**10, 2**7  # Grid points in longitudinal and transverse directions
dx, dy = L1 / N1, L2 / N2  # Grid spacings (meters)

print(f"Physical parameters: \n\t sigma2 = {sigma2:.2f},\n\t L_2d = {L_2d:.2f},\n\t psi = {psi:.2f},\n\t z_i = {z_i:.2f}")
print(f"Obtained c: {c:.2f}")
print(f"Domain parameters: \n\t L1 = {L1:.2f},\n\t L2 = {L2:.2f},\n\t N1 = {N1},\n\t N2 = {N2},\n\t dx = {dx:.2f},\n\t dy = {dy:.2f}")



def numerical_debug(some_arr, heat = False):
    """
    Debug-ish function; prints heatmap if input is 2d array
    """
    print("Max: ", np.max(some_arr))
    print("Min: ", np.min(some_arr))
    print("Mean: ", np.mean(some_arr))
    print("Median: ", np.median(some_arr))
    print("Shape: ", some_arr.shape)

    if len(some_arr.shape) == 2 and heat:
        heatmap(some_arr)
    print()



# k1_arr = fftshift(2 * np.pi * fftfreq(N1, d=dx))
# k2_arr = fftshift(2 * np.pi * fftfreq(N2, d=dy))

k1_arr = (2 * np.pi * fftfreq(N1, d=dx))
k2_arr = (2 * np.pi * fftfreq(N2, d=dy))

print(k1_arr.shape)
print(k2_arr.shape)


# TODO: check these shapes
k1, k2 = np.meshgrid(k1_arr, k2_arr)


kappa = np.sqrt(2 * ((k1 * np.cos(psi)) ** 2 + (k2 * np.sin(psi)) ** 2))
k_mag = np.sqrt(k1**2 + k2**2)  # Magnitude of wavenumber

numerical_debug(k_mag)
numerical_debug(kappa)

E_kappa_attenuated = (
    c * (kappa**3)
) / (
    (((L_2d**-2) + kappa**2)**(7/3)) * (1 + (kappa * z_i)**2)
)

numerical_debug(E_kappa_attenuated)

mask = np.isclose(kappa, 0.0)

_intermediate = E_kappa_attenuated / np.pi

# phi_common = np.where(mask,  _intermediate / kappa, 0.0)

phi_common = np.zeros_like(kappa, dtype=float)

for i in range(N2):
    for j in range(N1):
        if np.isclose(kappa[i, j], 0.0):
            phi_common[i, j] = 0.0
        else:
            phi_common[i, j] = _intermediate[i, j] / kappa[i, j]


# NOTE: Here we are currently leaving k in, only using kappa in the energy spectrum
mask = k_mag > 0

phi_11 = np.zeros_like(phi_common, dtype=float)
phi_12 = np.zeros_like(phi_common, dtype=float)
phi_22 = np.zeros_like(phi_common, dtype=float)

# phi_11 = np.where(mask, phi_common * (1 - (k1 / k_mag)**2), 0.0)
# phi_12 = np.where(mask, phi_common * (-1 * (k1 * k2) / k_mag**2), 0.0)
# phi_21 = phi_12
# phi_22 = np.where(mask, phi_common * (1 - (k2 / k_mag)**2), 0.0)

for i in range(N2):
    for j in range(N1):
        if mask[i, j]:
            phi_11[i, j] = phi_common[i, j] * (1 - (k1[i, j] / k_mag[i, j])**2)
            phi_12[i, j] = phi_common[i, j] * (-1 * (k1[i, j] * k2[i, j]) / k_mag[i, j]**2)
            phi_22[i, j] = phi_common[i, j] * (1 - (k2[i, j] / k_mag[i, j])**2)


C_11 = np.sqrt((2 * np.pi)**2 / (L1 * L2) * phi_11)
C_22 = np.sqrt((2 * np.pi)**2 / (L1 * L2) * phi_22)

numerical_debug(C_11)

C_12 = np.zeros_like(phi_12)

for i in range(N2):
    for j in range(N1):
        if mask[i, j]:
            C_12[i, j] = phi_12[i, j]


numerical_debug(C_12)

# eta_1 = np.random.normal(0, 1, size=(N1, N2)) + 1j * np.random.normal(0, 1, size=(N1, N2))
# eta_2 = np.random.normal(0, 1, size=(N1, N2)) + 1j * np.random.normal(0, 1, size=(N1, N2))
eta_1 = np.random.normal(0, 1, size=(N2, N1)) + 1j * np.random.normal(0, 1, size=(N2, N1))
eta_2 = np.random.normal(0, 1, size=(N2, N1)) + 1j * np.random.normal(0, 1, size=(N2, N1))

u1 = np.real(ifft2((C_11 * eta_1) + (C_12 * eta_2)))  # Longitudinal component
u2 = np.real(ifft2((C_12 * eta_1) + (C_22 * eta_2)))  # Transverse component

# Verify total variance (should match sigma_2d^2)
var_u1 = np.var(u1)
var_u2 = np.var(u2)
print(f"Variance of u1: {var_u1:.4f}, Variance of u2: {var_u2:.4f}")
print(f"Target variance: {sigma2:.4f}")

# Plot the wind field
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(u1, cmap='viridis')
plt.title('Longitudinal Wind Field (u1)')
plt.colorbar(label='Velocity (m/s)')

plt.subplot(122)
plt.imshow(u2, cmap='viridis')
plt.title('Transverse Wind Field (u2)')
plt.colorbar(label='Velocity (m/s)')
plt.show()
