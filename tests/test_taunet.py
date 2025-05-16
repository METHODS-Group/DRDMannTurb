import jax.numpy as jnp

from drdmannturb.models.taunet import RationalKernel, TauNet


def test_rational_kernel_asymptotics():
    rk = RationalKernel(learn_nu=False)
    small = jnp.array([1e-6])
    big = jnp.array([1e6])
    val_small = rk(small)[0]
    val_big = rk(big)[0]
    # k^-1 scaling at small k → multiply by k gives ~ const
    assert jnp.isfinite(val_small)
    assert jnp.isfinite(val_big)


def test_taunet_forward_shape():
    net = TauNet()
    k = jnp.ones((5, 3))
    out = net(k)
    assert out.shape == (5,)
    assert jnp.all(out > 0)
