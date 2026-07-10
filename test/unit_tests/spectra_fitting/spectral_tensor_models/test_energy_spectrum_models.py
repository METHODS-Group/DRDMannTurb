"""Test energy spectrum models."""

import pytest
import torch

from drdmannturb.spectra_fitting.spectral_tensor_models import (
    EnergySpectrumModel,
    Learnable_ESM,
    VonKarman_ESM,
)


@pytest.mark.unit
@pytest.mark.spectra_fitting
class TestEnergySpectrumModels:
    """Test the energy spectrum models."""

    @pytest.fixture
    def sample_k(self):
        """Sample wavevector tensor."""
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    @pytest.fixture
    def sample_L(self):
        """Sample length scale."""
        return torch.tensor(10.0)

    def test_energy_spectrum_model_base(self, sample_k, sample_L):
        """Test that the base class raises NotImplementedError."""
        model = EnergySpectrumModel()

        with pytest.raises(NotImplementedError):
            model(sample_k, sample_L)

    def test_von_karman_esm(self, sample_k, sample_L):
        """Test the Von Karman ESM model."""
        model = VonKarman_ESM()

        result = model(sample_k, sample_L)

        assert result.shape == sample_k.shape[:-1]
        assert torch.all(result >= 0.0)  # Energy should be non-negative
        assert not torch.isnan(result).any()

    def test_von_karman_esm_zero_k(self, sample_L):
        """Test the Von Karman ESM at zero wavevector."""
        model = VonKarman_ESM()
        k_zero = torch.tensor([[0.0, 0.0, 0.0]])

        result = model(k_zero, sample_L)
        assert not torch.isnan(result).any()

    def test_learnable_esm(self, sample_k, sample_L):
        """Test the learnable energy spectrum model."""
        model = Learnable_ESM()

        result = model(sample_k, sample_L)

        assert result.shape == sample_k.shape[:-1]
        assert torch.all(result >= 0.0)
        assert not torch.isnan(result).any()

    def test_learnable_esm_parameters(self, sample_k, sample_L):
        """Test that the learnable parameters are properly constrained."""
        model = Learnable_ESM()
        p = model._positive(model._raw_p)
        q = model._positive(model._raw_q)

        assert torch.all(p > 0)
        assert torch.all(q > 0)

        result = model(sample_k, sample_L)
        assert not torch.isnan(result).any()

    @pytest.mark.precision
    def test_precision_consistency(self, precision_dtype, device):
        """Test that energy spectrum models work with different precisions."""
        k = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=precision_dtype)
        L = torch.tensor(10.0, dtype=precision_dtype)

        # Move to device
        k = k.to(device)
        L = L.to(device)

        models = [VonKarman_ESM(), Learnable_ESM()]

        for model in models:
            result = model(k, L)
            assert result.dtype == precision_dtype
            assert result.device == k.device
            assert not torch.isnan(result).any()

    def test_extreme_values(self, device):
        """Test behavior with extreme wavevector values."""
        # Very large wavevectors
        k_large = torch.tensor([[1e6, 0.0, 0.0]], dtype=torch.float32)

        # Very small wavevectors
        k_small = torch.tensor([[1e-6, 0.0, 0.0]], dtype=torch.float32)

        L = torch.tensor(10.0)

        # Move to device
        k_large = k_large.to(device)
        k_small = k_small.to(device)
        L = L.to(device)

        models = [VonKarman_ESM(), Learnable_ESM()]

        for model in models:
            for k_test in [k_large, k_small]:
                result = model(k_test, L)
                assert not torch.isnan(result).any()
                assert not torch.isinf(result).any()
                assert result.device == k_test.device
