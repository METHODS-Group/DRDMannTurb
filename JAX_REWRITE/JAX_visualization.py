"""Visualization utilities for the spectra_fitting module."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_generated_data(dataset: dict):
    """Plot raw output from the data_generator functions.

    Handles loglog plotting of potentially negative or zero spectral components.
    """
    k1_orig = dataset["k1"]
    phi_orig = dataset["phi"]
    coherence_orig = dataset["coherence"]

    k1 = jnp.asarray(k1_orig)
    phi = jnp.asarray(phi_orig)
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
            component_data = phi[:, i, j]
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
                ax.loglog(k1[positive_mask], plot_data_for_loglog[positive_mask])
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
                if k1.size > 0 and jnp.max(k1) > 0 and jnp.min(k1) > 0:
                    ax.set_xlim(jnp.min(k1) * 0.5, jnp.max(k1) * 2)
                else:  # Fallback if k1 is empty or non-positive
                    ax.set_xlim(1e-3, 1e3)
                ax.set_ylim(1e-6, 1e1)  # Arbitrary log range for y

            ax.set_xlabel("k1")
            # Ensured idx_label is correctly interpolated into the f-string
            ax.set_ylabel(rf"{prefix_label}$\Phi_{idx_label}${suffix_label}")

    labels_coh = ["uu", "vv", "ww"]
    for i_coh, ax_coh in enumerate(axes_coherence):
        ax_coh.semilogx(k1, coherence[:, i_coh])
        ax_coh.set_xlabel("k1")
        ax_coh.set_ylabel(f"Coherence ({labels_coh[i_coh]})")
        ax_coh.grid(True, which="both", ls=":", alpha=0.7)
        ax_coh.set_ylim([-1.1, 1.1])

    fig.suptitle("Spectral Data and Coherence", fontsize=16)
    # Adjusted rect for tight_layout: [left, bottom, right, top]
    # This gives a little space at the bottom (0.03) and top (1-0.97=0.03 for title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    plt.show()
