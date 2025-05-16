import jax.numpy as jnp

from drdmannturb import data_generator as dg


def test_kaimal_shapes():
    k1 = jnp.logspace(-1, 1, 5)
    data = dg.generate_kaimal_data(k1, zref=40.0, ustar=1.0)
    phi = data["phi"]
    assert phi.shape == (5, 3, 3)
    # symmetry
    assert jnp.allclose(phi[:, 0, 2], phi[:, 2, 0])


def test_vk_positive():
    k1 = jnp.array([0.1, 1.0, 10.0])
    data = dg.generate_von_karman_data(k1)
    phi = data["phi"]
    assert phi.shape == (3, 3, 3)
    # Only diagonal components are strictly positive; off-diagonals may be negative.
    assert jnp.all(phi[:, jnp.arange(3), jnp.arange(3)] > 0)
