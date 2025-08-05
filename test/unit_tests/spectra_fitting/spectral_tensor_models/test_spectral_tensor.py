"""Basic unit tests for the spectral tensor models, including energy spectrum and eddy lifetime models.."""

import pytest  # noqa: I001
import torch
from unittest.mock import Mock

from drdmannturb.spectra_fitting.spectral_tensor_models import (
    # Eddy Lifetime Models
    EddyLifetimeModel,
    TauNet_ELT,
    Mann_ELT,
    TwoThirds_ELT,
    Constant_ELT,
    # Energy Spectrum Models
    EnergySpectrumModel,
    VonKarman_ESM,
    Learnable_ESM,
    # Spectral Tensor Models
    SpectralTensorModel,
    RDT_SpectralTensor,
)
from drdmannturb.nn_modules import TauNet


class TestEddyLifetimeModels:
    """Test the eddy lifetime models."""

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

            torch.testing.assert_close(result1, result2)


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


class TestSpectralTensorModels:
    """Test spectral tensor models."""

    @pytest.fixture
    def sample_k(self):
        """Sample wavevector tensor."""
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    @pytest.fixture
    def mock_elt(self):
        """Mock ELT model."""
        mock = Mock(spec=EddyLifetimeModel)
        mock.return_value = torch.tensor([1.0, 2.0, 3.0])
        return mock

    @pytest.fixture
    def mock_esm(self):
        """Mock energy spectrum model."""
        mock = Mock(spec=EnergySpectrumModel)
        mock.return_value = torch.tensor([0.1, 0.2, 0.3])
        return mock

    def test_spectral_tensor_model_base(self, sample_k, mock_elt, mock_esm):
        """Test that the base class raises NotImplementedError."""
        model = SpectralTensorModel(mock_elt, mock_esm, 10.0, 4.0, 3.0)

        with pytest.raises(NotImplementedError):
            model(sample_k)

    def test_rdt_spectral_tensor_initialization(self, mock_elt, mock_esm):
        """Test RDT spectral tensor initialization."""
        model = RDT_SpectralTensor(mock_elt, mock_esm, 10.0, 4.0, 3.0)

        assert torch.exp(model.log_L) == 10.0
        assert torch.exp(model.log_gamma) == 4.0
        assert torch.exp(model.log_sigma) == 3.0

    def test_rdt_spectral_tensor_invalid_initialization(self, sample_k, mock_elt, mock_esm):
        """Test RDT spectral tensor forward pass."""
        with pytest.raises(ValueError, match="L_init must be positive"):
            RDT_SpectralTensor(mock_elt, mock_esm, -1.0, 4.0, 3.0)

        with pytest.raises(ValueError, match="gamma_init must be positive"):
            RDT_SpectralTensor(mock_elt, mock_esm, 10.0, -1.0, 3.0)

        with pytest.raises(ValueError, match="sigma_init must be positive"):
            RDT_SpectralTensor(mock_elt, mock_esm, 10.0, 4.0, -1.0)

    def test_rdt_spectral_tensor_forward(self, sample_k, mock_elt, mock_esm):
        """Test RDT spectral tensor forward pass."""
        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)

        result = model(sample_k)

        assert len(result) == 6

        for i, phi in enumerate(result):
            assert phi.shape == sample_k.shape[:-1]

            if i == 3:  # phi_3 = uw component is always negative
                assert torch.all(phi < 0.0)
            else:
                assert torch.all(phi > 0.0)

            assert not torch.isnan(phi).any()

    def test_rdt_spectral_tensor_symmetries(self):
        """Test symmetry properties of the RDT spectral tensor."""
        k = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0]])
        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)

        phi1 = model(k)
        phi2 = model(-k)

        # Auto-spectra Phi11, Phi22, Phi33 should be even functions of k
        torch.testing.assert_close(phi1[0], phi2[0])
        torch.testing.assert_close(phi1[1], phi2[1])
        torch.testing.assert_close(phi1[2], phi2[2])

        # Cross-spectra Phi12, Phi13, Phi23 should be odd functions of k
        torch.testing.assert_close(phi1[3], -phi2[3])
        torch.testing.assert_close(phi1[4], -phi2[4])
        torch.testing.assert_close(phi1[5], -phi2[5])

    def test_rdt_spectral_tensor_parameter_gradients(self):
        """Test that parameters are differentiable."""
        k = torch.tensor([[1.0, 0.0, 0.0]])
        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)

        result = model(k)
        loss = torch.stack([phi.sum() for phi in result]).sum()

        loss.backward()

        assert model.log_L.grad is not None
        assert model.log_gamma.grad is not None
        assert model.log_sigma.grad is not None


class TestNumericalStability:
    """Test numerical stability of the spectral tensor models."""

    def test_extreme_wavevectors(self):
        """Test behavior with extreme wavevector values."""
        k_large = torch.tensor([[1e6, 0.0, 0.0]])

        k_small = torch.tensor([[1e-6, 0.0, 0.0]])

        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)

        result_large = model(k_large)
        result_small = model(k_small)

        for phi in list(result_large) + list(result_small):
            assert not torch.isnan(phi).any()
            assert not torch.isinf(phi).any()

    def test_zero_wavevector(self):
        """Test behavior with zero wavevector."""
        k_zero = torch.tensor([[0.0, 0.0, 0.0]])

        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)

        result = model(k_zero)

        for phi in result:
            assert not torch.isnan(phi).any()
            assert not torch.isinf(phi).any()


class TestIntegration:
    """Integration tests for spectral tensor models."""

    @pytest.mark.slow
    def test_full_pipeline(self):
        """Test full pipeline with realistic parameters."""
        k1 = torch.logspace(-2, 2, 20)
        k2 = torch.logspace(-2, 2, 20)
        k3 = torch.logspace(-2, 2, 20)

        k1_grid, k2_grid, k3_grid = torch.meshgrid(k1, k2, k3)
        k = torch.stack([k1_grid.flatten(), k2_grid.flatten(), k3_grid.flatten()])

        # Test with different eddy lifetime models
        elt_models = [Mann_ELT(), TwoThirds_ELT(), Constant_ELT()]
        esm = VonKarman_ESM()

        for elt in elt_models:
            model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)
            result = model(k)

            assert len(result) == 6
            for phi in result:
                assert phi.shape == (k.shape[0],)
                assert torch.all(phi >= 0)
                assert not torch.isnan(phi).any()
                assert not torch.isinf(phi).any()

    @pytest.mark.slow
    def test_learnable_esm_training(self):
        """Test that learnable ESM can be trained."""
        k = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        L = torch.tensor(10.0)

        model = Learnable_ESM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for _ in range(10):
            optimizer.zero_grad()
            output = model(k, L)
            loss = output.sum()
            loss.backward()
            optimizer.step()

        assert model._raw_p.grad is not None
        assert model._raw_q.grad is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_tensor(self):
        """Test with empty tensor."""
        k_empty = torch.empty((0, 3))

        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)

        result = model(k_empty)
        assert len(result) == 6
        for phi in result:
            assert phi.shape == (0,)

    def test_single_point(self):
        """Test with single point."""
        k_single = torch.tensor([[1.0, 1.0, 1.0]])

        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)

        result = model(k_single)
        assert len(result) == 6
        for phi in result:
            assert phi.shape == (1,)
