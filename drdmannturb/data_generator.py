"""Routines for generating toy data for validation purposes."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


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
    F_ij = jnp.zeros((n_k1, 3, 3))

    # F_ij is a function of k_1*z where z is the reference height (zref here)
    nondim_freq = 1 / (2 * jnp.pi) * k1_arr * zref

    spectra_uu_formula = jax.vmap(lambda f: 52.5 * f / (1 + 33 * f) ** (5 / 3))
    spectra_vv_formula = jax.vmap(lambda f: 8.5 * f / (1 + 9.5 * f) ** (5 / 3))
    spectra_ww_formula = jax.vmap(lambda f: 1.05 * f / (1 + 5.3 * f ** (5 / 3)))
    spectra_uw_formula = jax.vmap(lambda f: -7 * f / (1 + 9.6 * f) ** (12 / 5))

    F_ij = F_ij.at[:, 0, 0].set(spectra_uu_formula(nondim_freq))
    F_ij = F_ij.at[:, 1, 1].set(spectra_vv_formula(nondim_freq))
    F_ij = F_ij.at[:, 2, 2].set(spectra_ww_formula(nondim_freq))
    # NOTE: Symmetry
    spectra_uw_wu = spectra_uw_formula(nondim_freq)
    F_ij = F_ij.at[:, 0, 2].set(spectra_uw_wu)
    F_ij = F_ij.at[:, 2, 0].set(spectra_uw_wu)
    F_ij *= ustar**2

    ############################################################
    # Generate coherences
    #
    # Per k1 value, we have [uu, vv, ww] coherence.
    coherence = jnp.zeros((n_k1, 3))

    # TODO: Implement coherence generation

    return {
        "k1z": k1_arr * zref,
        "spectra": F_ij,
        "coherence": coherence,
    }


def generate_von_karman_data(k1_arr) -> dict:
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
    n_k1 = len(k1_arr)

    ############################################################
    # Generate spectral tensor
    #
    # Per k1 value, we have a 3x3 tensor.
    C = 3.2
    L = 0.59
    phi = jnp.zeros((n_k1, 3, 3))

    spectra_uu_formula = jax.vmap(lambda k1: 9 / 55 * C / (L ** (-2) + k1**2) ** (5 / 6))
    spectra_vv_formula = jax.vmap(
        lambda k1: 3 / 110 * C * (3 * L ** (-2) + 8 * k1**2) / (L ** (-2) + k1**2) ** (11 / 6)
    )
    spectra_ww_formula = jax.vmap(
        lambda k1: 3 / 110 * C * (3 * L ** (-2) + 8 * k1**2) / (L ** (-2) + k1**2) ** (11 / 6)
    )

    phi = phi.at[:, 0, 0].set(spectra_uu_formula(k1_arr))
    phi = phi.at[:, 1, 1].set(spectra_vv_formula(k1_arr))
    phi = phi.at[:, 2, 2].set(spectra_ww_formula(k1_arr))
    phi = k1_arr * phi

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



def plot_generated_data(dataset: dict):
    """Plot raw output from the data_generator functions.

    Handles loglog plotting of potentially negative or zero spectral components.
    """
    k1z_orig = dataset["k1z"]
    spectra_orig = dataset["spectra"]
    coherence_orig = dataset["coherence"]

    k1z = jnp.asarray(k1z_orig)
    spectra = jnp.asarray(spectra_orig)
    coherence = jnp.asarray(coherence_orig)

    fig = plt.figure(figsize=(12, 14))
    # Increased hspace for more vertical separation between plot rows.
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.3)

    axes_phi = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)])
    axes_coherence = [fig.add_subplot(gs[3, j]) for j in range(3)]

    # Plot each component of the spectral tensor
    for i in range(3):
        for j in range(3):
            ax = axes_phi[i, j]
            component_data = spectra[:, i, j]
            # Corrected how idx_label is formed for direct use in f-string
            idx_label = f"{i+1}{j+1}"

            plot_data_for_loglog = component_data
            prefix_label = ""
            suffix_label = ""

            effectively_zero_threshold = 1e-16
            is_effectively_zero = jnp.all(jnp.abs(component_data) < effectively_zero_threshold)

            if i == j:
                plot_data_for_loglog = component_data
            elif (i == 0 and j == 2) or (i == 2 and j == 0):
                if jnp.all(component_data <= effectively_zero_threshold) and jnp.any(
                    component_data < -effectively_zero_threshold
                ):
                    plot_data_for_loglog = -component_data
                    prefix_label = "-"
                else:
                    plot_data_for_loglog = jnp.abs(component_data)
                    if not is_effectively_zero:
                        prefix_label = "|"
                        suffix_label = "|"
            else:
                plot_data_for_loglog = jnp.abs(component_data)
                if not is_effectively_zero:
                    prefix_label = "|"
                    suffix_label = "|"

            positive_mask = plot_data_for_loglog > effectively_zero_threshold

            # Apply grid to all phi subplots before plotting data or text
            ax.grid(True, which="both", ls=":", alpha=0.7)

            if jnp.any(positive_mask):
                ax.loglog(k1z[positive_mask], plot_data_for_loglog[positive_mask])
            else:
                # If no positive data to plot, display text and set scales manually
                # This helps the grid to be drawn even if no data lines are present.
                ax.text(
                    0.5,
                    0.5,
                    "Non-positive data\nor effectively zero",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                # Set some default limits if k1 is available, otherwise generic log limits
                if k1z.size > 0 and jnp.max(k1z) > 0 and jnp.min(k1z) > 0:
                    ax.set_xlim(jnp.min(k1z) * 0.5, jnp.max(k1z) * 2)
                else:  # Fallback if k1 is empty or non-positive
                    ax.set_xlim(1e-3, 1e3)
                ax.set_ylim(1e-6, 1e1)  # Arbitrary log range for y

            ax.set_xlabel("k1z")
            # Ensured idx_label is correctly interpolated into the f-string
            ax.set_ylabel(rf"{prefix_label}$k_1 F_{idx_label}${suffix_label}")

    labels_coh = ["uu", "vv", "ww"]
    for i_coh, ax_coh in enumerate(axes_coherence):
        ax_coh.semilogx(k1z, coherence[:, i_coh])
        ax_coh.set_xlabel("k1z")
        ax_coh.set_ylabel(f"Coherence ({labels_coh[i_coh]})")
        ax_coh.grid(True, which="both", ls=":", alpha=0.7)
        ax_coh.set_ylim([-1.1, 1.1])

    fig.suptitle("Spectral Data and Coherence", fontsize=16)
    # Adjusted rect for tight_layout: [left, bottom, right, top]
    # This gives a little space at the bottom (0.03) and top (1-0.97=0.03 for title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    plt.show()


if __name__ == "__main__":

    zref = 40.0
    ustar = 1.773

    # Generate Kaimal data
    k1 = jnp.logspace(-3, 1, 20) / zref
    data = generate_kaimal_data(k1, zref, ustar)

    plot_generated_data(data)
