import numpy as np
from low_freq_prototype import LowFreq2DFieldGenerator

####
# Figure 2a recreation
#
# sigma^2 = 2 (m/s)^2
# z_i = 500 m
# psi = 45 degrees
#
# Grid dimensions: 40 L_2D x 5 L_2D
#
# We'll say L_2D = 15km

sigma2 = 2
z_i = 500.0
psi_deg = 45.0

L_2D = 15_000.0

# L_1, L_2 =
grid_dimensions = [40 * L_2D, 5 * L_2D]
grid_levels = [6, 4]

generator = LowFreq2DFieldGenerator(
    grid_dimensions,
    grid_levels,
    L_2D=L_2D,
    sigma2=sigma2,
    z_i=500.0,
    psi_degs=45.0,
    c=None,
)

###############################################################################

sum_sim_F11 = None
sum_sim_F22 = None

for _ in range(10):
    u_field = generator.generate()
    v_field = generator.generate()

    F11 = np.fft.fft2(u_field)
    F22 = np.fft.fft2(v_field)

    sum_sim_F11 += F11
    sum_sim_F22 += F22
