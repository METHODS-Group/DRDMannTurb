"""Tests for the NNCovariance kernel bridging trained spectral tensor models to field generation."""

import numpy as np
import pytest
import torch

from drdmannturb.fluctuation_generation import (
    FluctuationFieldGenerator,
    MannCovariance,
    NNCovariance,
)
from drdmannturb.nn_modules import TauNet
from drdmannturb.spectra_fitting import spectral_tensor_models as stm

pytestmark = pytest.mark.unit


@pytest.fixture
def fft_frequencies():
    """Physical FFT frequency grid including the zero mode."""
    return [2 * np.pi * np.fft.fftfreq(8, d=0.5) for _ in range(3)]


def _mann_model(L: float = 0.75, gamma: float = 3.9, sigma: float = 3.2) -> stm.RDT_SpectralTensor:
    return stm.RDT_SpectralTensor(
        eddy_lifetime_model=stm.Mann_ELT(),
        energy_spectrum_model=stm.VonKarman_ESM(),
        L_init=L,
        gamma_init=gamma,
        sigma_init=sigma,
    ).double()


class TestNNCovariance:
    """Test the NNCovariance kernel."""

    def test_matches_mann_covariance(self, fft_frequencies):
        """A Mann_ELT + VonKarman_ESM model must reproduce MannCovariance exactly.

        MannCovariance hardcodes the Mann eddy lifetime and von Karman energy spectrum in
        numpy; NNCovariance evaluates the model's torch submodules. With equivalent
        parameters the sqrt spectral tensors must agree to machine precision.
        """
        model = _mann_model()

        # Read back the model's actual parameter values (float32-rounded at construction).
        L = float(torch.exp(model.log_L))
        gamma = float(torch.exp(model.log_gamma))
        sigma = float(torch.exp(model.log_sigma))

        nn_cov = NNCovariance(model, length_scale=1.0, amplitude=1.0)
        # MannCovariance's E0 is sigma * L^{5/3} in the model's parametrization.
        mann_cov = MannCovariance(length_scale=L, E0=sigma * L ** (5.0 / 3.0), Gamma=gamma)

        G_nn = nn_cov.precompute_spectrum(fft_frequencies)
        G_mann = mann_cov.precompute_spectrum(fft_frequencies)

        assert G_nn.shape == G_mann.shape
        assert np.allclose(G_nn, G_mann, rtol=1e-10, atol=1e-13)

    def test_length_scale_and_amplitude_rescaling(self, fft_frequencies):
        """Phi_phys(k) = amplitude * length_scale^3 * Phi_model(length_scale * k)."""
        model = _mann_model()
        amplitude, ell = 2.5, 3.0

        G_ref = NNCovariance(model, length_scale=1.0).precompute_spectrum(fft_frequencies)
        G_scaled = NNCovariance(model, length_scale=ell, amplitude=amplitude).precompute_spectrum(
            [f / ell for f in fft_frequencies]
        )

        assert np.allclose(G_scaled, np.sqrt(amplitude * ell**3) * G_ref, rtol=1e-10, atol=1e-13)

    def test_taunet_model_produces_finite_spectrum(self, fft_frequencies):
        """An (untrained) TauNet-based DRD model yields a finite complex sqrt tensor."""
        torch.manual_seed(0)
        model = stm.RDT_SpectralTensor(
            eddy_lifetime_model=stm.TauNet_ELT(taunet=TauNet(n_layers=2, hidden_layer_sizes=[10, 10])),
            energy_spectrum_model=stm.VonKarman_ESM(),
            L_init=0.59,
            gamma_init=3.9,
            sigma_init=3.2,
        )
        cov = NNCovariance(model, length_scale=90.0 / (2 * np.pi), amplitude=0.556**2)

        G = cov.precompute_spectrum(fft_frequencies)

        assert G.shape == (3, 3, 8, 8, 8)
        assert np.iscomplexobj(G)
        assert np.isfinite(G).all()
        assert np.any(G != 0)

    def test_learnable_esm_supported(self, fft_frequencies):
        """The kernel is generic in the energy spectrum model, not hardcoded to von Karman."""
        model = stm.RDT_SpectralTensor(
            eddy_lifetime_model=stm.Mann_ELT(),
            energy_spectrum_model=stm.Learnable_ESM(),
            L_init=0.75,
            gamma_init=3.9,
            sigma_init=3.2,
        )
        G = NNCovariance(model, length_scale=1.0).precompute_spectrum(fft_frequencies)

        assert np.isfinite(G).all()
        assert np.any(G != 0)

    def test_exposes_dimensional_parameters(self):
        """FluctuationFieldGenerator reads L and Gamma off the covariance for buffer sizing."""
        model = _mann_model(L=0.75)
        ell = 3.0
        cov = NNCovariance(model, length_scale=ell)

        assert cov.L == pytest.approx(float(torch.exp(model.log_L)) * ell)
        assert cov.Gamma == pytest.approx(float(torch.exp(model.log_gamma)))

    def test_invalid_arguments_raise(self):
        """Non-positive scales and unsupported ndim are rejected."""
        model = _mann_model()
        with pytest.raises(ValueError):
            NNCovariance(model, length_scale=0.0)
        with pytest.raises(ValueError):
            NNCovariance(model, length_scale=1.0, amplitude=-1.0)
        with pytest.raises(ValueError):
            NNCovariance(model, length_scale=1.0, ndim=2)

    def test_end_to_end_field_generation(self):
        """A DRD model drives FluctuationFieldGenerator to a finite non-trivial field."""
        torch.manual_seed(0)
        model = stm.RDT_SpectralTensor(
            eddy_lifetime_model=stm.TauNet_ELT(taunet=TauNet(n_layers=2, hidden_layer_sizes=10)),
            energy_spectrum_model=stm.VonKarman_ESM(),
            L_init=0.59,
            gamma_init=3.9,
            sigma_init=3.2,
        )
        cov = NNCovariance(model, length_scale=90.0 / (2 * np.pi), amplitude=0.556**2)

        gen = FluctuationFieldGenerator(
            grid_dimensions=np.array([300.0, 150.0, 150.0]),
            grid_levels=np.array([5, 4, 4]),
            covariance=cov,
            seed=42,
        )
        field = gen.generate(num_blocks=1, zref=90.0, uref=11.4, z0=0.02, windprofiletype="LOG")

        assert field.shape[-1] == 3
        assert np.isfinite(field).all()
        assert np.any(field != 0)
