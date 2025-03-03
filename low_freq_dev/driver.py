"""
Driver for low-frequency model testing
"""

import numpy as np
from pretty_print import print_header, print_section

config_type = "figure2_a_eq15"

# Flags
plot_field = True
plot_spectra = True

param_sets = {
    "figure2_a_eq15": {
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
    "figure2_b_eq16": {
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
    "figure3_standard_eq14": {
        "sigma2": 0.6,  # m2/s2
        "L_2d": 15_000.0,  # m
        "psi": np.deg2rad(45.0),  # degrees
        "z_i": 500.0,  # m
        "L1_factor": 4,
        "L2_factor": 1,
        "N1": 2**10,
        "N2": 2**7,
        "equation": "eq14",
    },
    "figure3_standard_eq15": {
        "sigma2": 0.6,  # m2/s2
        "L_2d": 15_000.0,  # m
        "psi": np.deg2rad(45.0),  # degrees
        "z_i": 500.0,  # m
        "L1_factor": 4,
        "L2_factor": 1,
        "N1": 2**10,
        "N2": 2**7,
        "equation": "eq15",
    },
}

config = param_sets[config_type]


print_header("2D Wind Field Simulator")

print_section("Physical Parameters")
# print_param("config_type", config_type)
# print_param("sigma2", f"{sim.sigma2:.2f}", "m²/s²")
# print_param("L_2d", f"{sim.L_2d:.2f}", "m")
# print_param("psi (in degrees)", f"{np.rad2deg(sim.psi):.2f}", "degrees")
# print_param("z_i", f"{sim.z_i:.2f}", "m")
# print_param("c", f"{sim.c:.4f}")

# print_section("Domain Parameters")
# print_param("L1", f"{sim.L1:.2f}", "m")
# print_param("L2", f"{sim.L2:.2f}", "m")
# print_param("N1", sim.N1)
# print_param("N2", sim.N2)
# print_param("dx", f"{sim.dx:.2f}", "m")
# print_param("dy", f"{sim.dy:.2f}", "m")

# print_section("Problem Configuration")
# print_param("Configuration", config_type)
# print_param("Using equation", f"{sim.equation}")
# print_param("Plot field", f"{Fore.GREEN if plot_field else Fore.RED}{plot_field}")
# print_param("Plot spectra", f"{Fore.GREEN if plot_spectra else Fore.RED}{plot_spectra}")


# u1, u2 = sim.gen()


# arr_debug(u1, "u1", plot_heatmap=False)
# arr_debug(u2, "u2", plot_heatmap=False)

# var_u1 = np.var(u1)
# var_u2 = np.var(u2)

# print_param("Variance of u1", f"{var_u1:.8f}", "m²/s²")
# print_param("Variance of u2", f"{var_u2:.8f}", "m²/s²")
