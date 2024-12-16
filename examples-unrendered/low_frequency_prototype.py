import numpy as np
import matplotlib.pyplot as plt
import torch

from drdmannturb.fluctuation_generation import (
    GenerateFluctuationField,
    plot_velocity_magnitude,
    plot_velocity_components,
)

from scipy.special import gamma, hyp2f1

sigma2_2d = 0 # variance of the 2D spectrum, excluding attentuation
L_2d = 0 # length scale of the 2D spectrum, corresponding to
         # peak of mesoscale turbulence
psi = 0 # anisotropy parameter
z_i = 0 # Attenutation parameter

# NOTE: WE ARE GOING TO ASSUME THAT PSI IS GIVEN IN DEGREES FOR PROTOTYPE

def deg_to_rad(deg):
    """
    Convert degrees to radians

    NOTE: the psi anisotropy parameter is in degrees but
    numpy functions expect radians.
    """
    return deg * np.pi / 180

def compute_kappa(k1, k2, psi):
    """
    Compute kappa, how the scale-independent anisotropy
    is introduced to the energy spectrum.
    """
    return np.sqrt(
        (k1**2 * np.cos(psi)**2)
        + (k2**2 * np.sin(psi)**2)
    )

def compute_Ekappa(kappa, L_2D, c=1.0):
    """
    Compute the energy spectrum E(kappa) with attenuation.
    Following equation (2) in the paper.
    """
    energy_spectrum = (c * (kappa**3) / (L_2D**(-2) + kappa**2)**(7/3))
    attenuation = 1 / (1 + (kappa * z_i)**2)

    return energy_spectrum * attenuation

# A few helper functions to compute intermediates

def _compute_a(k1, L_2D, psi):
    return 1 + (2 * (k1**2) * (L_2D**2) * (np.cos(psi)**2))

def _compute_b(k1, z_i, psi):
    return 1 + (2 * (k1**2) * (z_i**2) * (np.cos(psi)**2))

def _compute_p(L_2D, z_i, a, b):
    return (L_2D**2 * b) / (z_i**2 * a)


def compute_F11(k1, L_2D, z_i, c=1.0):
    """
    Compute the F11 component of the spectral tensor.
    """
    # Convert psi to radians for numpy functions
    psi_rad = deg_to_rad(psi)
    
    # Compute intermediate values with checks
    a = _compute_a(k1, L_2D, psi_rad)
    b = _compute_b(k1, z_i, psi_rad)
    p = _compute_p(L_2D, z_i, a, b)
    
    print(f"\nDebug for k1 = {k1}:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"p = {p}")
    
    # Check for valid inputs to hypergeometric functions
    if p >= 1:
        print(f"Warning: p = {p} >= 1, hypergeometric function may not converge")
    
    try:
        term1 = hyp2f1(5/6, 1, 1/2, p)
        term2 = hyp2f1(-1/6, 1, 1/2, p)
        print(f"hyp2f1 terms: {term1}, {term2}")
    except Exception as e:
        print(f"Error in hypergeometric calculation: {e}")
        return np.nan
    
    # Calculate denominator terms separately to check for division by zero
    denom1 = ((L_2D**2) - (z_i**2))
    denom2 = np.sin(psi_rad)**3
    denom3 = a**(5/6)

    # print(f"Denominators: {denom1}, {denom2}, {denom3}")
    
    if abs(denom1) < 1e-10 or abs(denom2) < 1e-10 or abs(denom3) < 1e-10:
        print("Warning: Near-zero denominator detected")
    
    hyp2f1_sum = (-1 * p * term1) + (-7 * term1) + (2 * term2)
    
    first_term = (
        gamma(11/6) * (L_2D ** (11/3))
    ) / (
        10 * np.sqrt(2 * np.pi) * gamma(7/3) * denom1 * denom2 * denom3
    ) * hyp2f1_sum
    
    second_term = (
        (L_2D**(14/3)) * np.sqrt(b)
    ) / (
        2 * np.sqrt(2) * (a**(7/3)) * (z_i**3) * np.sin(psi_rad)**3
    )
    
    result = c * (first_term + second_term)
    
    # print(f"First term: {first_term}")
    # print(f"Second term: {second_term}")
    # print(f"Final result: {result}")
    
    return result

def compute_F22(k1, L_2D, z_i, c=1.0):
    """
    Compute the F22 component of the spectral tensor.
    """
    a = _compute_a(k1, L_2D, psi)
    b = _compute_b(k1, z_i, psi)
    p = _compute_p(L_2D, z_i, a, b)

    # First major term
    prefactor = (z_i**4 * a**(1/6) * L_2D**(11/3) * gamma(17/6)) / (
        55 * np.sqrt(2*np.pi) * (L_2D**2 - z_i**2)**2 * b * gamma(7/3) * np.sin(deg_to_rad(psi))
    )

    # All the hypergeometric function terms
    hyp2f1_sum = (
        -9 - (26 * hyp2f1(1/6, 1, 1/2, p))
        + (p**2) * (15 - (30*hyp2f1(-1/6, 1, 1/2, p)) - (59*hyp2f1(5/6, 1, 1/2, p)))
        + (35*hyp2f1(5/6, 1, 1/2, p)) + 15*(p**3)*hyp2f1(5/6, 1, 1/2, p)
        + p*(-54 + 88*hyp2f1(-1/6, 1, 1/2, p) + 9*hyp2f1(5/6, 1, 1/2, p))
    )

    first_term = prefactor * hyp2f1_sum

    # Second term
    second_term = -L_2D**(14/3) / (np.sqrt(2*b) * a**(7/3) * z_i * np.sin(deg_to_rad(psi)))

    return c * (k1**2) * (first_term + second_term)


import matplotlib.pyplot as plt

sigma2_2d = 1.0
L_2D = 20 # 20000
z_i = 500
psi = 43

import sys

# sys.exit()

k1 = np.logspace(-3, 3, 7)
# k1 = np.logspace(-3, 3, 1400)
F11 = np.array([k * compute_F11(k, L_2D, z_i, c=1.0) for k in k1])

# Print some debug info
print("Min F11:", np.min(F11))
print("Max F11:", np.max(F11))
print("Any NaN:", np.any(np.isnan(F11)))
print("Any inf:", np.any(np.isinf(F11)))

# Plot with error checking
if np.any(F11 <= 0):
    print("Warning: Some y-values are <= 0, which won't work with log scale")
    print("Number of non-positive values:", np.sum(F11 <= 0))

plt.figure(figsize=(8, 6))
plt.plot(k1 * L_2D, F11, '--', label='...')
plt.xscale('log')
if np.all(F11 > 0):
    plt.yscale('log')

plt.xlabel('k₁L₂D [-]')
plt.ylabel('k₁F₁₁(k₁) [m²/s²]')
plt.grid(True)
plt.legend()
plt.show()
