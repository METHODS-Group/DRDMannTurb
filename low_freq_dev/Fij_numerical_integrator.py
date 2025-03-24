import numpy as np
import scipy.integrate as integrate


def _safe_division(numerator: float, denominator: float, safe_value=0.0):
    """
    Safe division
    """
    return np.where(np.isclose(denominator, 0.0), safe_value, numerator / denominator)


def _calculate_kappa(k1: float, k2: float, psi: float) -> float:
    """
    Calculate the "augmented wavenumber" kappa
    """

    return np.sqrt(2 * ((k1 * np.cos(psi)) ** 2 + (k2 * np.sin(psi)) ** 2))


def _E_kappa(kappa: float, L2D: float, c: float) -> float:
    """
    Calculates the energy spectrum in terms of kappa = sqrt(2(k1^2cos^2psi + k2^2sin^2psi))
    """
    numerator = c * (kappa**3)
    denominator = (L2D ** (-2) + kappa**2) ** (7 / 3)

    return _safe_division(numerator, denominator)


def _E_kappa_attenuated(kappa: float, L2D: float, z_i: float, c: float) -> float:
    """
    Attenuated energy spectrum in terms of kappa
    """

    return _safe_division(_E_kappa(kappa, L2D, c), 1 + (kappa * z_i) ** 2)


def _spectral_tensor_common(kappa: float, k: float, L2D: float, z_i: float, c: float) -> float:
    """
    Common spectral tensor calculations
    """

    energy = _E_kappa_attenuated(kappa, L2D, z_i, c)

    denom = np.pi * kappa

    return _safe_division(energy, denom)


def spectral_tensor_11(k1: float, k2: float, psi: float, L2D: float, z_i: float, c: float) -> float:
    """
    Simulated spectral tensor i = j = 1 component

    Note that all that varies between this and the 22 method is the "parenthetical"
    """
    kappa = _calculate_kappa(k1, k2, psi)
    k = np.sqrt(k1**2 + k2**2)

    leading_factor = _spectral_tensor_common(kappa, k, L2D, z_i, c)

    parenthetical = 1 - (_safe_division(k1, k) ** 2)

    return leading_factor * parenthetical


def spectral_tensor_12(k1: float, k2: float, psi: float, L2D: float, z_i: float, c: float) -> float:
    """
    Simulated spectral tensor i = 1, j = 2 component
    """
    kappa = _calculate_kappa(k1, k2, psi)
    k = np.sqrt(k1**2 + k2**2)

    leading_factor = _spectral_tensor_common(kappa, k, L2D, z_i, c)

    parenthetical = -1 * _safe_division(k1 * k2, k**2)

    return leading_factor * parenthetical


def spectral_tensor_21(k1: float, k2: float, psi: float, L2D: float, z_i: float, c: float) -> float:
    """
    Simulated spectral tensor i = 2, j = 1 component
    """
    return spectral_tensor_12(k1, k2, psi, L2D, z_i, c)


def spectral_tensor_22(k1: float, k2: float, psi: float, L2D: float, z_i: float, c: float) -> float:
    """
    Simulated spectral tensor i = j = 2 component

    Note that all that varies between this and the 11 method is the "parenthetical"
    """
    kappa = _calculate_kappa(k1, k2, psi)
    k = np.sqrt(k1**2 + k2**2)

    leading_factor = _spectral_tensor_common(kappa, k, L2D, z_i, c)

    parenthetical = 1 - (_safe_division(k2, k) ** 2)

    return leading_factor * parenthetical


######################################################################################################################
# "Public" integrator methods


def eq6_numerical_F11_2D(k1: float, psi: float, L2D: float, z_i: float, c: float, warn: bool = False) -> tuple[float, float]:
    """
    By numerical integration, provides an "analytical" solution for F_11 2-dimensional spectrum
    """
    res, err = integrate.quad(lambda k2: spectral_tensor_11(k1, k2, psi, L2D, z_i, c), -np.inf, np.inf)

    if warn:
        print(f"Warning: Numerical integration error for F11: {err}")

    return res


def eq6_numerical_F22_2D(k1: float, psi: float, L2D: float, z_i: float, c: float, warn: bool = False) -> tuple[float, float]:
    """
    By numerical integration, provides an "analytical" solution for F_22 2-dimensional spectrum
    """
    res, err = integrate.quad(lambda k2: spectral_tensor_22(k1, k2, psi, L2D, z_i, c), -np.inf, np.inf)

    if warn:
        print(f"Warning: Numerical integration error for F22: {err}")

    return res
