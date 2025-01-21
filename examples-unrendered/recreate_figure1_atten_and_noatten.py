import matplotlib.pyplot as plt
import numpy as np
from analytic_f11_f22 import analytic_F11_2d

L_2D = 20_000.0  # [m]
z_i = 500.0  # [m]
psi_rad = np.deg2rad(43.0)

# So, this is k1 * F11(k1) while x axis is k1 * L2D
k1_times_L2D = np.logspace(-4, 4, 100)
k1 = k1_times_L2D / L_2D

k1F11_values = k1 * np.array([analytic_F11_2d(k, L_2D=L_2D, z_i=z_i, psi_rad=psi_rad) for k in k1])
# End, before plotting

plt.figure(figsize=(8, 6))
plt.plot(k1_times_L2D, k1F11_values, "-", label="Without attenuation?")
plt.xscale("log")
plt.grid(True)
plt.xlabel(r"$k_1 L_{2D}$ [-]")
plt.ylabel(r"$F_{11}(k_1)$ [m^2s^{-2}]")
plt.legend()
plt.show()
