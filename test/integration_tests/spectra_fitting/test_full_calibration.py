"""Integration tests for full calibration pipeline."""

import pytest
import torch

from drdmannturb.nn_modules import TauNet
from drdmannturb.spectra_fitting.spectral_tensor_models import (
    Constant_ELT,
    RDT_SpectralTensor,
    TauNet_ELT,
    VonKarman_ESM,
)


@pytest.mark.integration
@pytest.mark.spectra_fitting
@pytest.mark.slow
class TestFullCalibration:
    """Test full calibration pipeline."""

    def test_basic_calibration(self, device):
        """Test basic calibration workflow."""
        # Create model
        taunet = TauNet(n_layers=2, hidden_layer_sizes=5)
        model = RDT_SpectralTensor(
            eddy_lifetime_model=TauNet_ELT(taunet),
            energy_spectrum_model=VonKarman_ESM(),
            L_init=10.0,
            gamma_init=4.0,
            sigma_init=3.0,
        )

        # Create test data
        k = torch.logspace(-1, 2, 20, device=device)
        _test_data = torch.randn(20, 6, device=device)

        # Test forward pass
        result = model(k.unsqueeze(-1).expand(-1, 3))

        for phi in result:
            assert phi.device == k.device
            assert torch.isfinite(phi).all()

    @pytest.mark.precision
    def test_calibration_precision(self, precision_dtype, device):
        """Test calibration with different precisions."""
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(precision_dtype)
        try:
            taunet = TauNet(n_layers=2, hidden_layer_sizes=5)
            model = RDT_SpectralTensor(
                eddy_lifetime_model=TauNet_ELT(taunet),
                energy_spectrum_model=VonKarman_ESM(),
                L_init=10.0,
                gamma_init=4.0,
                sigma_init=3.0,
            )

            k = torch.logspace(-1, 2, 20, dtype=precision_dtype, device=device)
            result = model(k.unsqueeze(-1).expand(-1, 3))

            for phi in result:
                assert phi.dtype == precision_dtype
                assert phi.device == k.device
                assert torch.isfinite(phi).all()
        finally:
            torch.set_default_dtype(original_dtype)

    @pytest.mark.gpu
    def test_gpu_calibration(self):
        """Test calibration on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        device = "cuda"
        taunet = TauNet(n_layers=2, hidden_layer_sizes=5)
        model = RDT_SpectralTensor(
            eddy_lifetime_model=TauNet_ELT(taunet),
            energy_spectrum_model=VonKarman_ESM(),
            L_init=10.0,
            gamma_init=4.0,
            sigma_init=3.0,
        )

        k = torch.logspace(-1, 2, 20, device=device)
        result = model(k.unsqueeze(-1).expand(-1, 3))

        for phi in result:
            assert phi.device.type == "cuda"
            assert torch.isfinite(phi).all()

    def test_gradient_computation(self, device):
        """Test gradient computation during calibration."""
        model = RDT_SpectralTensor(
            eddy_lifetime_model=Constant_ELT(),
            energy_spectrum_model=VonKarman_ESM(),
            L_init=10.0,
            gamma_init=4.0,
            sigma_init=3.0,
        )

        k = torch.logspace(-1, 2, 20, device=device)
        k = k.unsqueeze(-1).expand(-1, 3)

        # Test gradient computation
        result = model(k)
        loss = torch.stack([phi.sum() for phi in result]).sum()

        loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert param.grad.device == k.device
                assert torch.isfinite(param.grad).all()
