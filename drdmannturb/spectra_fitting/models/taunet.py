"""TauNet model: neural-network approximation of eddy-lifetime τ(k)."""

from typing import Callable, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


class RationalKernel(eqx.Module):
    r"""Fixed-form rational function ensuring correct asymptotics.

    τ(k) ∝ |k|^{\nu-2/3} / (1+|k|²)^{\nu/2}
    """

    nu: jnp.ndarray = eqx.static_field()  # filled at runtime

    def __init__(self, nu_value: float = -1 / 3, learn_nu: bool = True):
        if learn_nu:
            object.__setattr__(self, "nu", jnp.array(nu_value))
        else:
            # store as static python float wrapped in jnp.array
            object.__setattr__(self, "nu", jnp.array(nu_value))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        a = self.nu - 2 / 3
        b = self.nu
        return jnp.abs(x) ** a / (1.0 + jnp.abs(x) ** 2) ** (b / 2)


class TauNet(eqx.Module):
    """Multi-layer perceptron + rational kernel."""

    layers: list[eqx.nn.Linear]
    activations: list[Callable]
    kernel: RationalKernel

    def __init__(
        self,
        hidden_layer_sizes: Sequence[Tuple[int, Callable]] = (
            (16, jax.nn.relu),
            (16, jax.nn.relu),
        ),
        learn_nu: bool = True,
        nu_value: float = -1 / 3,
        *,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        key = jax.random.PRNGKey(0) if key is None else key

        sizes = [s for s, _ in hidden_layer_sizes]
        self.activations = [act for _, act in hidden_layer_sizes]

        keys = jax.random.split(key, len(sizes) + 2)
        self.layers = [eqx.nn.Linear(3, sizes[0], use_bias=False, key=keys[0])]
        for i in range(len(sizes) - 1):
            self.layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], use_bias=False, key=keys[i + 1]))
        self.layers.append(eqx.nn.Linear(sizes[-1], 3, use_bias=False, key=keys[-1]))

        # small random noise so weight=0 isn't pathological
        for i, lyr in enumerate(self.layers):
            subkey = jax.random.split(key)[0]
            noise = 1e-3 * jax.random.normal(subkey, lyr.weight.shape)
            self.layers[i] = eqx.tree_at(lambda ell: ell.weight, lyr, lyr.weight + noise)

        self.kernel = RationalKernel(nu_value, learn_nu)

    def _forward_vec(self, vec: jnp.ndarray) -> jnp.ndarray:
        """Forward pass for a single 3-vector."""
        h = jnp.abs(vec)
        for lyr, act in zip(self.layers[:-1], self.activations):
            h = act(lyr(h))
        h = self.layers[-1](h)
        return self.kernel(jnp.linalg.norm(h))

    def __call__(self, k: jnp.ndarray) -> jnp.ndarray:
        """Supports input shape (...,3)."""
        if k.ndim == 1:
            return self._forward_vec(k)
        else:
            return jax.vmap(self._forward_vec)(k)
