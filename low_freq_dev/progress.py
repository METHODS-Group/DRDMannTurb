import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

"""
- Mesh independence study
- Scale independence study
- Plot
- Match spectrum
"""


class generator:

    def __init__(self, config):

        # Physical parameters
        self.sigma2 = config["sigma2"]
        self.L_2d = config["L_2d"]
        self.psi = config["psi"]
        self.z_i = config["z_i"]

        # Grid parameters
        self.L1 = config["L1_factor"] * self.L_2d
        self.L2 = config["L2_factor"] * self.L_2d
        self.N1 = 2**config["N1"]
        self.N2 = 2**config["N2"]

        # Grid spacing
        self.dx = self.L1 / self.N1
        self.dy = self.L2 / self.N2

        # Domain
        x = np.linspace(0, self.N1)
        y = np.linspace(0, self.N2)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")

        # Frequency set up
        k1_fft = 2 * np.pi * np.fft.fftfreq(self.N1, self.dx)
        k2_fft = 2 * np.pi * np.fft.fftfreq(self.N2, self.dy)

        self.k1, self.k2 = np.meshgrid(k1_fft, k2_fft, indexing="ij")

        self.c = (8.0 * self.sigma2) / (9.0 * (self.L_2d**(2 / 3)))


    def generate(self):

        # Fourier space
        k_mag = self.k1**2 + self.k2**2
        phi_ = np.sqrt(self.c / (np.pi * (self.L_2d**-2 + k_mag)**(7/3) * (1 + k_mag * self.z_i**2)))

        C1 = 1j * phi_ * self.k2
        C2 = 1j * phi_ * (-1 * self.k1)

        # Random noise
        eta = np.fft.fft2(np.random.normal(0, 1, size=(self.N1, self.N2)))

        u1_freq = C1 * eta
        u2_freq = C2 * eta

        # TODO: TUNE
        transform_norm = np.sqrt(self.dx * self.dy)
        normalization = 1 / (self.dx * self.dy)

        # TODO: TUNE
        u1 = np.real(np.fft.ifft2(u1_freq) / transform_norm) * normalization
        u2 = np.real(np.fft.ifft2(u2_freq) / transform_norm) * normalization

        return u1, u2


def mesh_study():

    config = {
        "sigma2": 2.0,  # m²/s²
        "L_2d": 1.0,  # m
        "psi": np.deg2rad(45.0),  # radians
        "z_i": 1.0,  # m
        "L1_factor": 1,  # Domain length = L1_factor * L_2d
        "L2_factor": 1,  # Domain length = L2_factor * L_2d

        "N1": 9,  # Grid points in x direction
        "N2": 9,  # Grid points in y direction
    }


    gen = generator(config)

    # Ok, x-axis = exponent of 2 for N1 and N2

    exponents = np.arange(4, 12)

    u1_norms = []
    u2_norms = []

    for x in exponents:

        config["N1"] = x
        config["N2"] = x

        gen = generator(config)

        u1, u2 = gen.generate()

        u1_norm = np.linalg.norm(u1) * gen.dx * gen.dy
        u2_norm = np.linalg.norm(u2) * gen.dx * gen.dy


        u1_norms.append(u1_norm)
        u2_norms.append(u2_norm)


    plt.plot(exponents, u1_norms, label="u1")
    plt.plot(exponents, u2_norms, label="u2")
    plt.legend()
    plt.show()


def debug_plot(u1, u2):

    pass



if __name__ == "__main__":

    fig2a_full_dom = {
        "sigma2": 2.0,  # m²/s²
        "L_2d": 15.0,  # m
        "psi": np.deg2rad(45.0),  # radians
        "z_i": 500.0,  # m
        "L1_factor": 1,  # Domain length = L1_factor * L_2d
        "L2_factor": 1,  # Domain length = L2_factor * L_2d
        "N1": 9,  # Grid points in x direction
        "N2": 9,  # Grid points in y direction
    }

    gen = generator(fig2a_full_dom)

    u1, u2 = gen.generate()

    mesh_study()


    # for _ in range(10):



