"""Visualization utilities for DRD training."""

from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .. import data_generator as dg
from ..models.spectral import OnePointSpectra, RDT_spectral_tensor
from ..models.taunet import TauNet
from .train import compute_kF


def plot_spectra(
    model: RDT_spectral_tensor,
    k1: jnp.ndarray,
    zref: float = 40.0,
    ustar: float = 1.773,
    save: Optional[Path] = None,
):
    """Plot learned one-point spectra against target Kaimal spectra."""
    data = dg.generate_kaimal_data(k1, zref, ustar)
    target_kF = jnp.stack(
        [
            k1 * data["phi"][:, 0, 0],
            k1 * data["phi"][:, 1, 1],
            k1 * data["phi"][:, 2, 2],
            k1 * data["phi"][:, 0, 2],
        ]
    )

    integrator = OnePointSpectra(model)
    pred_kF = compute_kF(model, integrator, k1)

    labels = ["uu", "vv", "ww", "uw"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        ax.loglog(k1, target_kF[i], "k-", label=f"{label} (target)")
        ax.loglog(k1, pred_kF[i], "r--", label=f"{label} (learned)")
    ax.set_xlabel("k1")
    ax.set_ylabel("k1 * F")
    ax.legend()
    ax.grid(True)
    if save is not None:
        fig.savefig(save)
    plt.show()


if __name__ == "__main__":
    # Example usage: load saved parameters and visualize
    key = jax.random.PRNGKey(0)
    taunet = TauNet(key=key)
    rdt = RDT_spectral_tensor(taunet)
    # Load parameters if available
    try:
        params = np.load("model_params.npz")
        leaves = [params[f"p{i}"] for i in range(len(params.files))]
        rdt_params, rdt_static = eqx.partition(rdt, eqx.is_array)
        rdt = eqx.combine(rdt_static, rdt_params)
    except FileNotFoundError:
        print("No saved parameters found. Using untrained model.")
    k1 = jnp.logspace(-1, 2, 60) / 40.0
    plot_spectra(rdt, k1)
