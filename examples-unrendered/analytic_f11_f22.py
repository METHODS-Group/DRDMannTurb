import warnings

import numpy as np
import scipy.special as sp

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"  # Reset to default color


def _compute_a_b_p_d(
    k1: float,
    L_2D: float,
    z_i: float,
    psi_rad: float,
) -> tuple[float, float, float, float]:
    # TODO: All of these evaluations can be simplified and sped up
    #       We are doing the same things several times.

    # Calculate a, b
    common = 2 * (k1 * np.cos(psi_rad)) ** 2

    square_L2D = L_2D**2
    square_z_i = z_i**2

    a = 1 + (common * square_L2D)
    b = 1 + (common * square_z_i)

    # Calculate p, d
    p = (square_L2D * b) / (square_z_i * a)
    d = (square_L2D / square_z_i) - 1

    if p > 1:
        warnings.warn(f"p > 1, p: {p}")

    return a, b, p, d


def analytic_F11_2d(
    k1: float,
    c: float = 1.0,
    L_2D: float = 15_000.0,
    z_i: float = 500.0,
    psi_rad: float = np.pi / 4,
) -> float:
    """
    Analytic solution for F11(k1)

    Default values are to replicate Fig 2 in the simulation paper
    """

    print(f"{BLUE}")
    print(" F11 inputs, k1: ", k1, " L_2D: ", L_2D, " z_i: ", z_i, " psi_rad: ", psi_rad)

    # L_2D /= 1000.0
    # z_i /= 1000.0

    #############################
    # CONSTANTS
    a, b, p, d = _compute_a_b_p_d(k1, L_2D, z_i, psi_rad)

    print(" F11 a: ", a)
    print(" F11 b: ", b)
    print(" F11 p: ", p)
    print(" F11 d: ", d)

    #############################
    # Precompute hyp2f1's
    alpha = sp.hyp2f1(-1 / 6, 1, 1 / 2, p)
    print(" hyp2f1(-1/6, 1, 1/2, p): ", alpha)
    beta = sp.hyp2f1(5 / 6, 1, 1 / 2, p)
    print(" hyp2f1(5/6, 1, 1/2, p): ", beta)

    #############################
    # First term
    first_term_factor = (sp.gamma(11 / 6) * (L_2D ** (11 / 3))) / (
        10 * np.sqrt(2 * np.pi) * sp.gamma(7 / 3) * (L_2D**2 - z_i**2) * np.sin(psi_rad) ** 3 * a ** (5 / 6)
    )
    first_term = (2 * alpha) - ((p + 7) * beta)
    first_term *= first_term_factor
    print(" first_term: ", first_term)

    second_term = (L_2D ** (14 / 3) * np.sqrt(b)) / (2 * np.sqrt(2) * d ** (7 / 3) * (z_i * np.sin(psi_rad)) ** 3)

    print(" second_term: ", second_term)

    print(f"{RESET}")

    return c * (first_term + second_term)


def analytic_F22_2d(
    k1: float,
    c: float = 1.0,
    L_2D: float = 15_000.0,
    z_i: float = 500.0,
    psi_rad: float = np.pi / 4,
) -> float:
    """
    Analytic solution for F22(k1)

    Default values are to replicate Fig 2 in the simulation paper
    """

    print(f"{BLUE}")
    print(" F22 inputs, k1: ", k1, " L_2D: ", L_2D, " z_i: ", z_i, " psi_rad: ", psi_rad)

    # TODO: Note that, without this, we get all the NaNs
    L_2D /= 1000.0

    a, b, p, d = _compute_a_b_p_d(k1, L_2D, z_i, psi_rad)

    print(f"{RESET}")
    return 0.0


if __name__ == "__main__":
    # NOTE: don't forget these when checking F22!
    do_F11 = True
    do_F22 = False

    test_a, test_b, test_p, test_d = _compute_a_b_p_d(
        1.0,  # k1
        5.0,  # L_2D
        2.0,  # z_i
        0,  # psi_rad
    )

    assert test_a == 51
    assert test_b == 9
    assert test_p == (225 / 204)
    assert test_d == (21 / 4)

    # TODO: Try everything in [km] instead of [m]
    c = 1.0
    L_2D = 15_000.0
    z_i = 500.0
    psi_rad = np.pi / 4

    k1_times_L2D = np.array([10, 100, 1000])
    k1 = k1_times_L2D / L_2D

    if do_F11:
        print(f"{RED}k1*F11(k1) @ k1*L_2D = 10: {RESET}", k1[0] * analytic_F11_2d(k1[0], c, L_2D, z_i, psi_rad))
        print(f"{RED}k1*F11 @ k1*L_2D = 100: {RESET}", analytic_F11_2d(k1[1], c, L_2D, z_i, psi_rad))
        print(f"{RED}k1*F11 @ k1*L_2D = 1000: {RESET}", analytic_F11_2d(k1[2], c, L_2D, z_i, psi_rad))

    if do_F22:
        print(f"{RED}k1*F22 @ k1*L_2D = 10: {RESET}", analytic_F22_2d(10, c))
        print(f"{RED}k1*F22 @ k1*L_2D = 100: {RESET}", analytic_F22_2d(100, c))
        print(f"{RED}k1*F22 @ k1*L_2D = 1000: {RESET}", analytic_F22_2d(1000, c))
