"""Test precision verification for spectral tensor models."""

import pytest
import torch

from drdmannturb.nn_modules import TauNet
from drdmannturb.spectra_fitting.spectral_tensor_models import (
    Constant_ELT,
    Learnable_ESM,
    Mann_ELT,
    RDT_SpectralTensor,
    TauNet_ELT,
    TwoThirds_ELT,
    VonKarman_ESM,
)


@pytest.mark.precision
@pytest.mark.unit
@pytest.mark.spectra_fitting
class TestPrecisionVerification:
    """Test that all models work correctly across different precisions."""

    def test_precision_accuracy(self, device):
        """Test that higher precision gives more accurate results."""
        # Use a challenging case that might show precision differences
        k = torch.logspace(-3, 3, 100).unsqueeze(-1).expand(-1, 3)

        results = {}
        for dtype in [torch.float32, torch.float64]:
            k_dtype = k.to(dtype)
            _L, _gamma = torch.tensor(10.0, dtype=dtype), torch.tensor(4.0, dtype=dtype)

            elt = Mann_ELT()
            esm = VonKarman_ESM()
            model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)

            # Convert model parameters
            model.log_L.data = model.log_L.data.to(dtype)
            model.log_gamma.data = model.log_gamma.data.to(dtype)
            model.log_sigma.data = model.log_sigma.data.to(dtype)

            results[dtype] = model(k_dtype)

        # Check that float64 results are more stable (fewer NaNs/infs)
        float32_nan_count = sum(torch.isnan(phi).sum().item() for phi in results[torch.float32])
        float64_nan_count = sum(torch.isnan(phi).sum().item() for phi in results[torch.float64])

        # Float64 should have fewer or equal NaNs
        assert float64_nan_count <= float32_nan_count

    def test_precision_consistency_across_models(self, precision_dtype, device):
        """Test that all models work consistently across precisions."""
        k = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=precision_dtype)
        L, gamma = torch.tensor(10.0, dtype=precision_dtype), torch.tensor(4.0, dtype=precision_dtype)

        # Move to device
        k = k.to(device)
        L, gamma = L.to(device), gamma.to(device)

        # Test eddy lifetime models
        elt_models = [Mann_ELT(), TwoThirds_ELT(), Constant_ELT()]
        for model in elt_models:
            result = model(k, L, gamma)
            assert result.dtype == precision_dtype
            assert result.device == k.device
            assert not torch.isnan(result).any()

        # Test energy spectrum models
        esm_models = [VonKarman_ESM(), Learnable_ESM()]
        for model in esm_models:
            result = model(k, L)
            assert result.dtype == precision_dtype
            assert result.device == k.device
            assert not torch.isnan(result).any()

        # Test spectral tensor model
        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)

        # Convert model parameters
        model.log_L.data = model.log_L.data.to(device=device, dtype=precision_dtype)
        model.log_gamma.data = model.log_gamma.data.to(device=device, dtype=precision_dtype)
        model.log_sigma.data = model.log_sigma.data.to(device=device, dtype=precision_dtype)

        result = model(k)
        for phi in result:
            assert phi.dtype == precision_dtype
            assert phi.device == k.device
            assert not torch.isnan(phi).any()

    def test_taunet_precision(self, precision_dtype, device):
        """Test TauNet with different precisions."""
        k = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=precision_dtype)
        L, gamma = torch.tensor(10.0, dtype=precision_dtype), torch.tensor(4.0, dtype=precision_dtype)

        # Move to device
        k = k.to(device)
        L, gamma = L.to(device), gamma.to(device)

        taunet = TauNet(n_layers=2, hidden_layer_sizes=5)
        # Move TauNet to correct device and dtype
        taunet = taunet.to(device=device, dtype=precision_dtype)
        model = TauNet_ELT(taunet)

        result = model(k, L, gamma)
        assert result.dtype == precision_dtype
        assert result.device == k.device
        assert not torch.isnan(result).any()

    def test_learnable_esm_precision(self, precision_dtype, device):
        """Test Learnable ESM with different precisions."""
        k = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=precision_dtype)
        L = torch.tensor(10.0, dtype=precision_dtype)

        # Move to device
        k = k.to(device)
        L = L.to(device)

        model = Learnable_ESM()
        # Move model to correct device and dtype
        model = model.to(device=device, dtype=precision_dtype)

        result = model(k, L)
        assert result.dtype == precision_dtype
        assert result.device == k.device
        assert not torch.isnan(result).any()

        # Check that parameters are also in the correct precision
        p = model._positive(model._raw_p)
        q = model._positive(model._raw_q)
        assert p.dtype == precision_dtype
        assert q.dtype == precision_dtype

    def test_large_scale_precision(self, precision_dtype, device):
        """Test with large-scale inputs at different precisions."""
        k = torch.logspace(-2, 2, 1000, dtype=precision_dtype).unsqueeze(-1).expand(-1, 3)

        # Move to device
        k = k.to(device)

        elt = Constant_ELT()
        esm = VonKarman_ESM()
        model = RDT_SpectralTensor(elt, esm, 10.0, 4.0, 3.0)

        # Convert model parameters
        model.log_L.data = model.log_L.data.to(device=device, dtype=precision_dtype)
        model.log_gamma.data = model.log_gamma.data.to(device=device, dtype=precision_dtype)
        model.log_sigma.data = model.log_sigma.data.to(device=device, dtype=precision_dtype)

        result = model(k)

        # Check memory usage
        if device == "cuda":
            memory_allocated = torch.cuda.memory_allocated()
            assert memory_allocated > 0

        for phi in result:
            assert phi.dtype == precision_dtype
            assert torch.isfinite(phi).all()

    def test_precision_gradient_consistency(self, precision_dtype, device):
        """Test that gradients work correctly with different precisions."""
        k = torch.tensor([[1.0, 0.0, 0.0]], dtype=precision_dtype)
        model = RDT_SpectralTensor(Constant_ELT(), VonKarman_ESM(), 10.0, 4.0, 3.0)

        # Move to device and convert precision
        k = k.to(device)
        # TODO: is this not done with eg. just model.to(device=device, dtype=precision_dtype) ?
        model.log_L.data = model.log_L.data.to(device=device, dtype=precision_dtype)
        model.log_gamma.data = model.log_gamma.data.to(device=device, dtype=precision_dtype)
        model.log_sigma.data = model.log_sigma.data.to(device=device, dtype=precision_dtype)

        result = model(k)
        loss = torch.stack([phi.sum() for phi in result]).sum()

        loss.backward()

        assert model.log_L.grad is not None
        assert model.log_gamma.grad is not None
        assert model.log_sigma.grad is not None

        # Check gradients are in correct precision
        assert model.log_L.grad.dtype == precision_dtype
        assert model.log_gamma.grad.dtype == precision_dtype
        assert model.log_sigma.grad.dtype == precision_dtype

        # Check gradients are finite
        assert torch.isfinite(model.log_L.grad).all()
        assert torch.isfinite(model.log_gamma.grad).all()
        assert torch.isfinite(model.log_sigma.grad).all()
