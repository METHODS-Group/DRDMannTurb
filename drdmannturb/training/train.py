"""Simple JAX/Optax training script for DRD calibration.

Run:  python -m drdmannturb.training.train
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .. import data_generator as dg
from ..models.spectral import OnePointSpectra, RDT_spectral_tensor
from ..models.taunet import TauNet
from .losses import log_mse

DEFAULT_SAVE = Path("model_params.npz")


def compute_kF(rdt: RDT_spectral_tensor, integrator: OnePointSpectra, k1: jnp.ndarray) -> jnp.ndarray:
    phi = rdt(k1)
    comps = []
    for idx in (0, 1, 2, 3):
        comps.append(k1 * integrator(phi[idx]))
    return jnp.stack(comps)  # (4, Nk1)


def train(num_epochs: int = 1000, lr: float = 5e-4, *, seed: int = 0, save: Optional[Path] = DEFAULT_SAVE):
    key = jax.random.PRNGKey(seed)

    # --- data
    zref = 40.0
    ustar = 1.773
    k1 = jnp.logspace(-1, 2, 60) / zref
    data = dg.generate_kaimal_data(k1, zref, ustar)
    target_kF = jnp.stack(
        [
            k1 * data["phi"][:, 0, 0],
            k1 * data["phi"][:, 1, 1],
            k1 * data["phi"][:, 2, 2],
            k1 * data["phi"][:, 0, 2],
        ]
    )

    # --- model
    taunet = TauNet(key=key)
    rdt = RDT_spectral_tensor(taunet)

    params, static = eqx.partition(rdt, eqx.is_array)

    # optimizer
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        def loss_fn(p):
            model = eqx.combine(static, p)
            integ = OnePointSpectra(model)
            pred = compute_kF(model, integ, k1)
            return log_mse(pred, target_kF)

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = opt.update(grads, opt_state)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss_val

    for epoch in range(num_epochs):
        params, opt_state, loss_val = step(params, opt_state)
        if epoch % 100 == 0:
            print(f"epoch {epoch:4d}  loss {loss_val:.3e}")

    model_final = eqx.combine(static, params)

    if save is not None:
        import numpy as np

        leaves = jax.tree_util.tree_leaves(params)
        np.savez_compressed(save, **{f"p{i}": np.asarray(ell) for i, ell in enumerate(leaves)})
        print("Saved parameters to", save)

    return model_final


if __name__ == "__main__":
    train()
