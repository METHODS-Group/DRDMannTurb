import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


class analytical_Fij:

    def __init__(self, config: dict):

        self.L_2d = config["L_2d"]
        self.psi = config["psi"]
        self.sigma2 = config["sigma2"]
        self.z_i = config["z_i"]

        self.c = (8 * self.sigma2) / (9 * (self.L_2d**(2/3)))
        print("c: ", self.c)

    def _phi_leading(self, k1: float, k2: float) -> float:
        """
        Common leading factor E(kappa) / pi * k
        """

        kappa = np.sqrt(
            2 * ((k1 * np.cos(self.psi))**2 + (k2 * np.sin(self.psi))**2)
        )
        k_mag = np.sqrt(k1**2 + k2**2)

        # Ekappa calculation
        denom_term_1 = (self.L_2d**-2 + kappa**2)**(7/3)
        denom_term_2 = (1 + (kappa * self.z_i)**2)

        Ekappa = self.c * (kappa**3) / (denom_term_1 * denom_term_2)

        return Ekappa / (np.pi * k_mag)

    def phi11(self, k1: float, k2: float, eps: float = 1e-20) -> float:
        """
        Returns Phi_11(k1, k2)
        """
        _common = self._phi_leading(k1, k2)

        k_mag_sq = k1**2 + k2**2
        # Use original formulation matching the generation code's implication
        _P_11 = 1.0 - (k1**2 / max(k_mag_sq, eps))

        return _common * _P_11

    def phi22(self, k1: float, k2: float, eps: float = 1e-20) -> float:
        """
        Returns Phi_22(k1, k2)
        """
        _common = self._phi_leading(k1, k2)

        k_mag_sq = k1**2 + k2**2
        # Use original formulation matching the generation code's implication
        _P_22 = 1.0 - (k2**2 / max(k_mag_sq, eps))

        return _common * _P_22


    def generate(self, k1_arr):
        """
        Generate F11 over given k1_arr
        """

        F11_res_arr = np.zeros_like(k1_arr)
        F11_err_arr = np.zeros_like(k1_arr)

        F22_res_arr = np.zeros_like(k1_arr)
        F22_err_arr = np.zeros_like(k1_arr)

        limit_factor = 1000.0 # Adjust this factor as needed
        k2_limit = limit_factor / self.L_2d

        for i, k1 in enumerate(k1_arr):
            F11_res_arr[i], F11_err_arr[i] = integrate.quad(
                lambda _k2: self.phi11(
                    k1, _k2
                ),
                -k2_limit, k2_limit # Use finite limits
                # Optional: Increase points or change tolerance if warning persists
                # limit=100, epsabs=1.49e-09, epsrel=1.49e-09
            )

            F22_res_arr[i], F22_err_arr[i] = integrate.quad(
                lambda _k2: self.phi22(
                    k1, _k2
                ),
                -k2_limit, k2_limit # Use finite limits
                # Optional: Increase points or change tolerance if warning persists
                # limit=100, epsabs=1.49e-09, epsrel=1.49e-09
            )

        return F11_res_arr, F11_err_arr, F22_res_arr, F22_err_arr



if __name__ == "__main__":

    do_plot = True

    phys_config = {
        "L_2d": 15_000.0, # [m]
        "psi": np.deg2rad(45.0),
        "sigma2": 2.0,
        "z_i": 500.0, # [m]
    }

    analytical_gen = analytical_Fij(phys_config)

    k1_arr_a = np.logspace(-1, 3, 500) / phys_config["L_2d"]
    k1_arr_b = np.logspace(0.5, 4.5, 500) / phys_config["L_2d"]

    F11_a, F11_err_a, F22_a, F22_err_a = analytical_gen.generate(k1_arr_a)
    F11_b, F11_err_b, F22_b, F22_err_b = analytical_gen.generate(k1_arr_b)

    # Print "grid" values

    print("\n\n---------------------------------------------------")
    print("EXPECTED F11 A @ k1L = 10^-1: approx 0.07")
    print("ACTUAL: ", (k1_arr_a[0] * F11_a[0]), " \n")

    print("EXPECTED MAX k1 * F11 A: approx 0.325")
    print("ACTUAL: ", np.max(k1_arr_a * F11_a))



    if do_plot:
        plt.semilogx(k1_arr_a * phys_config["L_2d"], k1_arr_a * F11_a)
        plt.grid(which="both", alpha=0.2)
        plt.title("F11 top row")
        plt.xlabel(r"$k_1 L_{2d}$")
        plt.ylabel(r"$k_1 F_{11}(k_1)$")
        plt.show()
