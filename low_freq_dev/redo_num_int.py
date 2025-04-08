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
        print(f"Using approximate c:  {self.c:.6e}")

    def _E_kappa(self, k1: float, k2: float) -> float:
        """ Calculate E(kappa) """
        kappa_squared = 2 * ((k1 * np.cos(self.psi))**2 + (k2 * np.sin(self.psi))**2)
        kappa_squared = max(kappa_squared, 1e-24)
        _kappa = np.sqrt(kappa_squared)

        denom_term_1 = (self.L_2d**-2 + kappa_squared)**(7/3)
        denom_term_2 = (1 + kappa_squared * self.z_i**2)

        if denom_term_1 * denom_term_2 < 1e-30:
            return 0.0
        Ekappa = self.c * (_kappa**3) / (denom_term_1 * denom_term_2)
        if not np.isfinite(Ekappa):
            return 0.0
        return Ekappa

    def _integrand11(self, k2: float, k1: float, eps: float = 1e-20) -> float:
        """
        Integrand matching the SHAPE of the original code, but maybe numerically stabler.
        Uses (E(kappa) / (pi * k)) * (k2^2 / k^2) form.
        """
        k_mag_sq = k1**2 + k2**2
        k_mag = np.sqrt(k_mag_sq)

        Ekappa = self._E_kappa(k1, k2)

        integrand = (Ekappa / (np.pi * k_mag)) * (k2**2 / k_mag_sq)
        return integrand

    def _integrand22(self, k2: float, k1: float, eps: float = 1e-20) -> float:
        """
        Integrand matching the SHAPE of the original code.
        Uses (E(kappa) / (pi * k)) * (k1^2 / k^2) form.
        """
        k_mag_sq = k1**2 + k2**2
        k_mag = np.sqrt(k_mag_sq)

        Ekappa = self._E_kappa(k1, k2)

        integrand = (Ekappa / (np.pi * k_mag)) * (k1**2 / k_mag_sq)
        return integrand


    def generate(self, k1_arr):
        """
        Generate F11, F22 over given k1_arr using the ORIGINAL integrand SHAPE
        but with potentially more stable integration settings.
        """
        F11_res_arr = np.zeros_like(k1_arr)
        F11_err_arr = np.zeros_like(k1_arr)
        F22_res_arr = np.zeros_like(k1_arr)
        F22_err_arr = np.zeros_like(k1_arr)

        # Use large, but finite, limits for better numerical stability
        # Choose a limit based on where E(kappa) becomes negligible
        # Example: Limit based on many L_2d or related to z_i if high-k decay is strong
        k2_limit_factor = 100 # Increase this if needed
        k2_limit = k2_limit_factor / min(self.L_2d, self.z_i) if self.z_i > 0 else k2_limit_factor / self.L_2d
        print(f"Using integration limits for k2: [-{k2_limit:.2e}, {k2_limit:.2e}]")


        for i, k1_val in enumerate(k1_arr):
            try:
                 F11_res_arr[i], F11_err_arr[i] = integrate.quad(
                     self._integrand11, # Use integrand with original 1/k factor
                     -k2_limit, k2_limit,          # Finite limits
                     args=(k1_val,),
                     limit=100, epsabs=1.49e-08, epsrel=1.49e-08 # Standard tolerance
                 )
                 # Check for large error estimate
                 if F11_err_arr[i] > 0.1 * abs(F11_res_arr[i]):
                      print(f"Warning: High relative error ({F11_err_arr[i]/F11_res_arr[i]:.1%}) for F11 at k1={k1_val:.4e}")

            except Exception as e:
                 print(f"Warning: Integration failed for F11 at k1={k1_val:.4e}: {e}")
                 F11_res_arr[i], F11_err_arr[i] = np.nan, np.nan

            try:
                 F22_res_arr[i], F22_err_arr[i] = integrate.quad(
                     self._integrand22, # Use integrand with original 1/k factor
                     -k2_limit, k2_limit,          # Finite limits
                     args=(k1_val,),
                     limit=100, epsabs=1.49e-08, epsrel=1.49e-08 # Standard tolerance
                 )
                 # Check for large error estimate
                 if F22_err_arr[i] > 0.1 * abs(F22_res_arr[i]):
                      print(f"Warning: High relative error ({F22_err_arr[i]/F22_res_arr[i]:.1%}) for F22 at k1={k1_val:.4e}")

            except Exception as e:
                 print(f"Warning: Integration failed for F22 at k1={k1_val:.4e}: {e}")
                 F22_res_arr[i], F22_err_arr[i] = np.nan, np.nan

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

    print("EXPECTED F22 A @ k1L = 10^-1: approx 0.07")
    print("ACTUAL: ", (k1_arr_a[0] * F22_a[0]), " \n")

    print("EXPECTED MAX k1 * F22 A: approx 0.410")
    print("ACTUAL: ", np.max(k1_arr_a * F22_a))



    if do_plot:
        plt.semilogx(k1_arr_a * phys_config["L_2d"], k1_arr_a * F11_a)
        plt.grid(which="both", alpha=0.2)
        plt.title("F11 top row")
        plt.xlabel(r"$k_1 L_{2d}$")
        plt.ylabel(r"$k_1 F_{11}(k_1)$")
        plt.show()
