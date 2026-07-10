"""Basic unit tests for the spectral tensor models, including energy spectrum and eddy lifetime models.."""

import pytest  # noqa: I001
import torch
from unittest.mock import Mock

from drdmannturb.spectra_fitting.spectral_tensor_models import (
    # Eddy Lifetime Models
    EddyLifetimeModel,
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
        """Test RDT spectral tensor forward pass.

        Output order is (Phi11, Phi22, Phi33, Phi13, Phi23, Phi12) = (uu, vv, ww, uw, vw, uv).

        At generic (non-axis-aligned) wavevectors, the auto-spectra must be strictly
        positive. On coordinate axes several components vanish identically (e.g.
        Phi33 ~ k1^2 + k2^2 = 0 at k = (0, 0, k3)); strict sign assertions there
        are NOT correct physics.
        """
        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)

        # Generic wavevectors: strictly positive auto-spectra, all finite
        k_generic = torch.tensor([[0.5, 0.7, 0.9], [1.0, 2.0, 3.0], [2.0, -1.0, 0.5]])
        result = model(k_generic)

        assert len(result) == 6
        for i, phi in enumerate(result):
            assert phi.shape == k_generic.shape[:-1]
            assert torch.isfinite(phi).all()
            if i < 3:
                assert torch.all(phi > 0.0), f"auto-spectrum {i} not strictly positive at generic k"

        # Axis-aligned wavevectors: exact zeros required by the tensor structure.
        # At k = (0, 0, k3) and k = (0, k2, 0), k1 = 0 kills the shear distortion
        # and the isotropic part has -k_i k_j = 0 for all cross terms.
        result_axes = model(sample_k)
        for phi in result_axes:
            assert torch.isfinite(phi).all()

        phi33 = result_axes[2]
        phi13, phi23, phi12 = result_axes[3], result_axes[4], result_axes[5]

        # sample_k rows: (1,0,0), (0,1,0), (0,0,1)
        assert phi33[2] == 0.0  # Phi33 ~ (k1^2 + k2^2) = 0 at k = e3
        for cross in (phi13, phi23, phi12):
            assert cross[1] == 0.0  # k = e2
            assert cross[2] == 0.0  # k = e3

    def test_rdt_spectral_tensor_symmetries(self):
        """Test symmetry properties of the RDT spectral tensor.

        For a real homogeneous velocity field the spectral tensor satisfies
        Phi_ij(-k) = Phi_ij(k)^* = Phi_ji(k); the Mann/RDT tensor is real and
        symmetric, so ALL components are EVEN under full inversion k -> -k.

        Under the reflection k2 -> -k2 the v-involving cross components
        (Phi23, Phi12) are odd while everything else is even; this is what makes
        the one-point spectra F23(k1) and F12(k1) vanish upon integration.
        """
        k = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [0.3, 0.8, -1.7]])
        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)

        phi = model(k)
        phi_inv = model(-k)

        # Full inversion k -> -k: every component is even
        for i in range(6):
            torch.testing.assert_close(phi[i], phi_inv[i])

        # Reflection k2 -> -k2: Phi23 (vw) and Phi12 (uv) are odd, the rest even
        k_reflected = k.clone()
        k_reflected[..., 1] = -k_reflected[..., 1]
        phi_ref = model(k_reflected)

        torch.testing.assert_close(phi[0], phi_ref[0])  # uu even
        torch.testing.assert_close(phi[1], phi_ref[1])  # vv even
        torch.testing.assert_close(phi[2], phi_ref[2])  # ww even
        torch.testing.assert_close(phi[3], phi_ref[3])  # uw even
        torch.testing.assert_close(phi[4], -phi_ref[4])  # vw odd
        torch.testing.assert_close(phi[5], -phi_ref[5])  # uv odd

    def test_rdt_isotropic_limit(self):
        """In the limit gamma -> 0 the RDT tensor reduces to the isotropic von Karman tensor.

        Phi_ij(k) = E(|k|) / (4 pi |k|^4) * (|k|^2 delta_ij - k_i k_j)
        """
        L_val, sigma_val = 10.0, 3.0
        gamma_tiny = 1e-12

        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, L_val, gamma_tiny, sigma_val)

        k = torch.tensor([[0.5, 0.7, 0.9], [1.0, 2.0, 3.0], [2.0, -1.0, 0.5]])
        phi = model(k)

        k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]
        kk = k.norm(dim=-1) ** 2
        E0 = sigma_val * L_val ** (5.0 / 3.0) * esm(k, torch.tensor(L_val)) / (4 * torch.pi)

        iso = {
            0: E0 / kk**2 * (kk - k1**2),
            1: E0 / kk**2 * (kk - k2**2),
            2: E0 / kk**2 * (kk - k3**2),
            3: E0 / kk**2 * (-k1 * k3),
            4: E0 / kk**2 * (-k2 * k3),
            5: E0 / kk**2 * (-k1 * k2),
        }

        for i in range(6):
            torch.testing.assert_close(phi[i], iso[i], rtol=1e-5, atol=1e-8)

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

        k1_grid, k2_grid, k3_grid = torch.meshgrid(k1, k2, k3, indexing="ij")
        k = torch.stack([k1_grid.flatten(), k2_grid.flatten(), k3_grid.flatten()], dim=-1)

        # Test with different eddy lifetime models
        elt_models = [Mann_ELT(), TwoThirds_ELT(), Constant_ELT()]
        esm = VonKarman_ESM()

        for elt in elt_models:
            model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)
            result = model(k)

            assert len(result) == 6
            for i, phi in enumerate(result):
                assert phi.shape == (k.shape[0],)
                if i < 3:
                    # Auto-spectra are mathematically non-negative, but the quadratic-form
                    # expansion suffers float cancellation (~1e-14 absolute) at extreme kL;
                    # allow noise far below the spectrum peak. Cross-spectra take either sign.
                    assert torch.all(phi >= -1e-10 * phi.abs().max())
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
