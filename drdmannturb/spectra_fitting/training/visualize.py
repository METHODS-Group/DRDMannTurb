"""Visualization utilities for DRD training."""

from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .. import data_generator as dg
from ..models.integrators import OnePointSpectra
from ..models.spectral import RDT_spectral_tensor
from ..models.taunet import TauNet


def plot_spectra(
    model: RDT_spectral_tensor,
    k1: jnp.ndarray,
    zref: float = 40.0,
    ustar: float = 1.773,
    save: Optional[Path] = None,
):
    """Plot learned one-point spectra against target Kaimal spectra."""
    data = dg.generate_kaimal_data(k1 * zref, zref, ustar)
    target_kF = jnp.stack(
        [
            k1 * data["phi"][:, 0, 0] / ustar**2,
            k1 * data["phi"][:, 1, 1] / ustar**2,
            k1 * data["phi"][:, 2, 2] / ustar**2,
            -1 * k1 * data["phi"][:, 0, 2] / ustar**2,
        ]
    )

    integrator = OnePointSpectra(model)
    pred_kF = compute_kF(model, integrator, k1)

    labels = ["uu", "vv", "ww", "-uw"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # distinct colors
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, color) in enumerate(zip(labels, colors)):
        if label == "-uw":
            ax.loglog(k1, -target_kF[i], color=color, linestyle='-', marker='o', label=f"{label} (target)")
            ax.loglog(k1, -pred_kF[i], color=color, linestyle='--', label=f"{label} (learned)")
        else:
            ax.loglog(k1, target_kF[i], color=color, linestyle='-', marker='o', label=f"{label} (target)")
            ax.loglog(k1, pred_kF[i], color=color, linestyle='--', label=f"{label} (learned)")
    ax.set_xlabel("k1")
    ax.set_ylabel("k1 * F")
    ax.legend()
    ax.grid(True)
    if save is not None:
        fig.savefig(save)
    plt.show()


def compute_kF(rdt: RDT_spectral_tensor, integrator: OnePointSpectra, k1: jnp.ndarray) -> jnp.ndarray:
    phi = rdt(k1)
    comps = []
    for idx in (0, 1, 2, 3):
        comps.append(k1 * integrator(phi[idx]))
    return jnp.stack(comps)  # (4, Nk1)


if __name__ == "__main__":
    # Example usage: load saved parameters and visualize
    key = jax.random.PRNGKey(0)
    taunet = TauNet(key=key)
    rdt = RDT_spectral_tensor(taunet)
    # Load parameters if available
    try:
        rdt = eqx.tree_deserialise_leaves("model.eqx", rdt)
    except FileNotFoundError:
        print("No saved parameters found. Using untrained model.")
    k1 = jnp.logspace(-1, 2, 60) / 40.0
    plot_spectra(rdt, k1)
    eqx.tree_serialise_leaves("model.eqx", rdt)
