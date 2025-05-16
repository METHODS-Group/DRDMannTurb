"""Loss utilities for DRD calibration."""

import jax
import jax.numpy as jnp


@jax.jit
def log_mse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Log–MSE used in original DRD calibration."""
    eps = 1e-12
    return jnp.mean(jnp.square(jnp.log(jnp.abs(y_pred) + eps) - jnp.log(jnp.abs(y_true) + eps)))
