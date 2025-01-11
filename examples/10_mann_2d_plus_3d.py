"""
================================================
Example 10: 2D Low Frequency + 3D High Frequency
================================================

This example demonstrates the addition of the low-frequency wind fluctuation field to a complete
3D wind field, following Syed-Mann (2024). ``DRDMannTurb`` includes this functionality through
an instance of the ``GenerateFluctuationField`` class. This example uses the Mann model for the
"high-frequency" component, though any other model could be used.
"""

L1 = 60_000.0
L2 = 15_000.0
Nx = 1024
Ny = 256

# Figure 3 parameters
L2D = 15000.0  # [m]
sigma2 = 0.6  # [m^2/s^2]
z_i = 500.0  # [m]
psi_degs = 43.0  # anisotropy angle
