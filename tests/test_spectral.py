import jax.numpy as jnp

from drdmannturb.models.spectral import OnePointSpectra, PowerSpectraRDT, RDT_spectral_tensor, VKEnergySpectrum
from drdmannturb.models.taunet import TauNet


def test_vk_positive():
    kL = jnp.logspace(-3, 3, 10)
    spec = VKEnergySpectrum(kL)
    assert jnp.all(spec > 0)


def test_power_spectra_shapes():
    k = jnp.zeros((2, 3))
    beta = jnp.ones(2)
    E0 = jnp.ones(2)
    Phi = PowerSpectraRDT(k, beta, E0)
    assert len(Phi) == 6
    for comp in Phi:
        assert comp.shape == (2,)


def test_rdt_forward_small():
    net = TauNet()
    rdt = RDT_spectral_tensor(net)
    k1 = jnp.array([0.1, 1.0])
    Phi = rdt(k1)
    assert len(Phi) == 6
    assert Phi[0].shape == (2, rdt.k2_grid.size, rdt.k3_grid.size)


def test_one_point_spectra_integration():
    net = TauNet()
    rdt = RDT_spectral_tensor(net, k2_k3=None)
    ops = OnePointSpectra(rdt)
    k1 = jnp.array([0.5])
    Phi = rdt(k1)
    result = ops(Phi[0])
    assert result.shape == (1,)
