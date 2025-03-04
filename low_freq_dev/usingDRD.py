import numpy as np

from drdmannturb.fluctuation_generation.covariance_kernels import Covariance

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
config = param_sets["figure2_a_eq15"]

_sigma2 = config["sigma2"]
_L_2d = config["L_2d"]
_psi = config["psi"]
_z_i = config["z_i"]
_L1_factor = config["L1_factor"]
_L2_factor = config["L2_factor"]
_N1 = config["N1"]
_N2 = config["N2"]

grid_dimensions = np.array([_L_2d * _L1_factor, _L_2d * _L2_factor, _z_i])
grid_levels = np.array([_N1, _N2, 1])


#########################################################################################################
# Implementation of Covariance


class LowFreqCovariance(Covariance):
    def __init__(self, L_2d: float, psi: float, z_i: float, sigma2: float, L1: float, L2: float):
        super().__init__()

        self.ndim = 2
        self.L_2d = L_2d
        self.psi = psi
        self.z_i = z_i
        self.sigma2 = sigma2
        self.L1 = L1
        self.L2 = L2

    def precompute_Spectrum(self, Frequencies: np.ndarray) -> np.ndarray:
        sqrtSpectralTensor = np.zeros((2, 2, Frequencies[0].size, Frequencies[1].size), dtype=np.complex128)

        # k = np.array(list(np.meshgrid(*Frequencies, indexing="ij")))

        # kk = np.sum(k**2, axis=0)
        # kappa = np.sqrt(2 * ((k[0, ...] * np.cos(self.psi)) ** 2 + (k[1, ...] * np.sin(self.psi)) ** 2))

        return sqrtSpectralTensor


# Need to create a random field object
# fft_sampler = Sampling_FFT()
