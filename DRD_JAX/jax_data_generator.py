"""Routines for generating toy data for validation purposes."""


import jax
import jax.numpy as jnp


def generate_kaimal_data(k1_arr, zref, ustar) -> dict:
    """Generate frequency-weighted Kaimal data for an array of k1 values.

    Parameters
    ----------
    k1_arr: np.ndarray | jax.numpy.ndarray
        Array of k1 values

    Returns
    -------
    data : dict
        Dictionary containing the following keys:
        - k1: One-dimensional JAX array of k1 values
    """
    n_k1 = len(k1_arr)

    ############################################################
    # Generate spectral tensor
    #
    # Per k1 value, we have a 3x3 tensor.
    phi = jnp.zeros((n_k1, 3, 3))

    # F_ij is a function of k_1*z where z is the reference height (zref here)
    nondim_freq = 1 / (2 * jnp.pi) * k1_arr * zref

    spectra_uu_formula = jax.vmap(
        lambda f: 52.5 * f / (1 + 33 * f) ** (5/3)
    )
    spectra_vv_formula = jax.vmap(
        lambda f: 8.5 * f / (1 + 9.5 * f) ** (5/3)
    )
    spectra_ww_formula = jax.vmap(
        lambda f: 1.05 * f / (1 + 5.3 * f ** (5/3))
    )
    spectra_uw_formula = jax.vmap(
        lambda f: -7 * f / (1 + 9.6 * f) ** (12/5)
    )

    phi = phi.at[:, 0, 0].set(
        spectra_uu_formula(nondim_freq)
    )
    phi = phi.at[:, 1, 1].set(
        spectra_vv_formula(nondim_freq)
    )
    phi = phi.at[:, 2, 2].set(
        spectra_ww_formula(nondim_freq)
    )
    # NOTE: Symmetry
    spectra_uw_wu = spectra_uw_formula(nondim_freq)
    phi = phi.at[:, 0, 2].set(
        spectra_uw_wu
    )
    phi = phi.at[:, 2, 0].set(
        spectra_uw_wu
    )
    phi *= ustar**2

    ############################################################
    # Generate coherences
    #
    # Per k1 value, we have [uu, vv, ww] coherence.
    coherence = jnp.zeros((n_k1, 3))

    # TODO: Implement coherence generation

    return {
        "k1": k1_arr,
        "phi": phi,
        "coherence": coherence,
    }

def generate_von_karman_data(k1_arr, zref) -> dict:
    """Generate frequency-weighted data from the Von Karman spectra.

    Parameters
    ----------
    k1_arr: np.ndarray | jax.numpy.ndarray
        Array of k1 values
    zref: float
        Reference height

    Returns
    -------
    data : dict
        Dictionary containing the following keys:
    """
    pass
