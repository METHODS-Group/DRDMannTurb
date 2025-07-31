"""Test eddy lifetime models."""

import pytest
import torch

from drdmannturb.nn_modules import TauNet
from drdmannturb.spectra_fitting.spectral_tensor_models import (
    Constant_ELT,
    EddyLifetimeModel,
    Mann_ELT,
    TauNet_ELT,
    TwoThirds_ELT,
)


@pytest.mark.unit
@pytest.mark.spectra_fitting
class TestEddyLifetimeModels:
    """Test eddy lifetime models."""

    @pytest.fixture
    def sample_k(self):
        """Sample wavevector tensor.

        Returns the 3d identity matrix.
        """
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    @pytest.fixture
    def sample_params(self):
        """Sample parameters."""
        return torch.tensor(10.0), torch.tensor(4.0)  # L, gamma

    def test_eddy_lifetime_model_base(self, sample_k, sample_params):
        """Test that the base class raises NotImplementedError."""
        model = EddyLifetimeModel()
        L, gamma = sample_params

        with pytest.raises(NotImplementedError):
            model(sample_k, L, gamma)

    def test_taunet_elt(self, sample_k, sample_params):
        """Test the TauNet ELT model."""
        L, gamma = sample_params
        taunet = TauNet(n_layers=2, hidden_layer_sizes=5)
        model = TauNet_ELT(taunet)

        result = model(sample_k, L, gamma)

        assert result.shape == sample_k.shape[:-1]
        assert torch.all(result >= 0.0)
        assert not torch.isnan(result).any()

    def test_mann_elt(self, sample_k, sample_params):
        """Test Mann ELT model."""
        L, gamma = sample_params
        model = Mann_ELT()

        result = model(sample_k, L, gamma)

        assert result.shape == sample_k.shape[:-1]
        assert torch.all(result >= 0.0)
        assert not torch.isnan(result).any()

    def test_two_thirds_elt(self, sample_k, sample_params):
        """Test TwoThirds ELT model."""
        L, gamma = sample_params
        model = TwoThirds_ELT()

        result = model(sample_k, L, gamma)

        assert result.shape == sample_k.shape[:-1]
        assert torch.all(result >= 0.0)
        assert not torch.isnan(result).any()

        # Test the specific functional form
        k_norm = sample_k.norm(dim=-1)
        expected = gamma * (L * k_norm) ** (-2.0 / 3.0)
        torch.testing.assert_close(result, expected)

    def test_constant_elt(self, sample_k, sample_params):
        """Test Constant ELT model."""
        L, gamma = sample_params
        model = Constant_ELT()

        result = model(sample_k, L, gamma)

        assert result.shape == sample_k.shape[:-1]
        assert torch.all(result == gamma)
        assert not torch.isnan(result).any()

    def test_elt_symmetries(self):
        """Test symmetry properties of the ELT models."""
        k = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]])
        L, gamma = torch.tensor(10.0), torch.tensor(4.0)

        models = [Mann_ELT(), TwoThirds_ELT(), Constant_ELT()]

        for model in models:
            result1 = model(k, L, gamma)
            result2 = model(-k, L, gamma)

            # Should be even with respect to k (tau(k) = tau(-k))
            torch.testing.assert_close(result1, result2)

    @pytest.mark.precision
    def test_precision_consistency(self, precision_dtype, device):
        """Test that eddy lifetime models work with different precisions."""
        k = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=precision_dtype)
        L, gamma = torch.tensor(10.0, dtype=precision_dtype), torch.tensor(4.0, dtype=precision_dtype)

        # Move to device
        k = k.to(device)
        L, gamma = L.to(device), gamma.to(device)

        models = [Mann_ELT(), TwoThirds_ELT(), Constant_ELT()]

        for model in models:
            result = model(k, L, gamma)
            assert result.dtype == precision_dtype
            assert result.device == k.device
            assert not torch.isnan(result).any()

    def test_edge_cases(self, device):
        """Test edge cases for eddy lifetime models."""
        # Zero wavevector
        k_zero = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        L, gamma = torch.tensor(10.0), torch.tensor(4.0)

        # Move to device
        k_zero = k_zero.to(device)
        L, gamma = L.to(device), gamma.to(device)

        models = [Mann_ELT(), TwoThirds_ELT(), Constant_ELT()]

        for model in models:
            result = model(k_zero, L, gamma)
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()

        # Very large wavevector
        k_large = torch.tensor([[1e6, 0.0, 0.0]], dtype=torch.float32)
        k_large = k_large.to(device)

        for model in models:
            result = model(k_large, L, gamma)
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
