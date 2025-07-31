"""Test the nn_modules module."""

import pytest
import torch
import torch.nn as nn

from drdmannturb.nn_modules import TauNet


@pytest.mark.unit
class TestTauNet:
    """Test the TauNet module."""

    def test_taunet_constructor_defaults(self):
        """Test the constructor with default parameters."""
        taunet = TauNet()
        assert taunet is not None
        assert taunet.n_layers == 2
        assert taunet.hidden_layer_sizes == [10, 10]
        assert len(taunet.activations) == 2
        assert all(isinstance(act, nn.ReLU) for act in taunet.activations)
        assert taunet.Ra.fg_learn_nu is True

    def test_taunet_constructor_custom_params(self):
        """Test the constructor with custom parameters."""
        taunet = TauNet(n_layers=3, hidden_layer_sizes=[5, 10, 15], learn_nu=False, nu_init=-0.5)
        assert taunet.n_layers == 3
        assert taunet.hidden_layer_sizes == [5, 10, 15]
        assert len(taunet.activations) == 3
        assert taunet.Ra.fg_learn_nu is False

    def test_taunet_constructor_int_hidden_sizes(self):
        """Test constructor with integer hidden_layer_sizes."""
        taunet = TauNet(n_layers=4, hidden_layer_sizes=20)
        assert taunet.hidden_layer_sizes == [20, 20, 20, 20]

    def test_taunet_constructor_custom_activations(self):
        """Test constructor with custom activation functions."""
        activations = [nn.Tanh(), nn.Sigmoid()]
        taunet = TauNet(n_layers=2, activations=activations)
        assert len(taunet.activations) == 2
        assert isinstance(taunet.activations[0], nn.Tanh)
        assert isinstance(taunet.activations[1], nn.Sigmoid)

    def test_taunet_constructor_single_activation(self):
        """Test constructor with single activation function."""
        taunet = TauNet(n_layers=3, activations=nn.Tanh())
        assert len(taunet.activations) == 3
        assert all(isinstance(act, nn.Tanh) for act in taunet.activations)

    def test_taunet_constructor_invalid_n_layers(self):
        """Test constructor with invalid n_layers."""
        with pytest.raises(ValueError, match="n_layers must be a positive integer"):
            TauNet(n_layers=0)

        with pytest.raises(ValueError, match="n_layers must be a positive integer"):
            TauNet(n_layers=-1)

        with pytest.raises(ValueError, match="n_layers must be a positive integer"):
            TauNet(n_layers=2.5)

    def test_taunet_constructor_invalid_hidden_layer_sizes_int(self):
        """Test constructor with invalid integer hidden_layer_sizes."""
        with pytest.raises(ValueError, match="hidden_layer_sizes must be a positive integer"):
            TauNet(hidden_layer_sizes=0)

        with pytest.raises(ValueError, match="hidden_layer_sizes must be a positive integer"):
            TauNet(hidden_layer_sizes=-5)

    def test_taunet_constructor_invalid_hidden_layer_sizes_list(self):
        """Test constructor with invalid list hidden_layer_sizes."""
        with pytest.raises(ValueError, match="hidden_layer_sizes must be a list of integers of length n_layers"):
            TauNet(n_layers=2, hidden_layer_sizes=[5, 10, 15])  # Wrong length

        with pytest.raises(ValueError, match="hidden_layer_sizes must be a list of positive integers"):
            TauNet(n_layers=2, hidden_layer_sizes=[5, 0])

        with pytest.raises(ValueError, match="hidden_layer_sizes must be a list of positive integers"):
            TauNet(n_layers=2, hidden_layer_sizes=[5, -10])

    def test_taunet_constructor_invalid_activations_length(self):
        """Test constructor with invalid activations list length."""
        with pytest.raises(ValueError, match="activations must be a list of nn.Module's of length n_layers"):
            TauNet(n_layers=2, activations=[nn.ReLU()])  # Too short

        with pytest.raises(ValueError, match="activations must be a list of nn.Module's of length n_layers"):
            TauNet(n_layers=2, activations=[nn.ReLU(), nn.Tanh(), nn.Sigmoid()])  # Too long

    def test_taunet_constructor_invalid_activations_type(self):
        """Test constructor with invalid activation types."""
        with pytest.raises(ValueError, match="activations must be a list of nn.Module's"):
            TauNet(n_layers=2, activations=[nn.ReLU(), "not_a_module"])

    def test_taunet_forward_pass(self):
        """Test the forward pass of the TauNet module."""
        taunet = TauNet(n_layers=2, hidden_layer_sizes=[5, 5])
        k = torch.randn(10, 3)  # 10 samples, 3D wave vectors

        result = taunet(k)

        assert result.shape == (10,)  # Should return scalar tau values
        assert not torch.isnan(result).any()
        assert torch.all(result >= 0)  # Tau should be non-negative

    def test_taunet_forward_pass_batch(self):
        """Test forward pass with different batch sizes."""
        taunet = TauNet()

        # Single sample
        k1 = torch.randn(1, 3)
        result1 = taunet(k1)
        assert result1.shape == (1,)

        # Multiple samples
        k2 = torch.randn(100, 3)
        result2 = taunet(k2)
        assert result2.shape == (100,)

    def test_taunet_forward_pass_extreme_values(self):
        """Test forward pass with extreme input values."""
        taunet = TauNet()

        # Very large values
        k_large = torch.tensor([[1e6, 1e6, 1e6]])
        result_large = taunet(k_large)
        assert not torch.isnan(result_large).any()
        assert not torch.isinf(result_large).any()

        # Very small values
        k_small = torch.tensor([[1e-6, 1e-6, 1e-6]])
        result_small = taunet(k_small)
        assert not torch.isnan(result_small).any()
        assert not torch.isinf(result_small).any()

    @pytest.mark.precision
    def test_taunet_precision(self, precision_dtype, device):
        """Test TauNet with different precisions."""
        taunet = TauNet(n_layers=2, hidden_layer_sizes=[5, 5])
        k = torch.randn(10, 3, dtype=precision_dtype)

        # Move model and input to device and dtype
        taunet = taunet.to(device=device, dtype=precision_dtype)
        k = k.to(device)

        result = taunet(k)
        assert result.dtype == precision_dtype
        assert result.device == k.device
        assert not torch.isnan(result).any()

    def test_taunet_gradients(self):
        """Test that TauNet produces gradients."""
        taunet = TauNet(n_layers=2, hidden_layer_sizes=[5, 5])
        k = torch.randn(10, 3, requires_grad=True)

        result = taunet(k)
        loss = result.sum()
        loss.backward()

        assert k.grad is not None
        assert torch.isfinite(k.grad).all()

    def test_taunet_parameters(self):
        """Test that TauNet has the expected parameters."""
        taunet = TauNet(n_layers=2, hidden_layer_sizes=[5, 5])

        # Should have parameters from linear layers and rational kernel
        param_names = [name for name, _ in taunet.named_parameters()]

        # Check for linear layer parameters
        assert any("linears" in name for name in param_names)

        # Check for rational kernel parameters
        assert any("Ra.nu" in name for name in param_names)

    def test_taunet_mlp_forward(self):
        """Test the internal _mlp_forward method."""
        taunet = TauNet(n_layers=2, hidden_layer_sizes=[5, 5])
        x = torch.randn(10, 3)

        result = taunet._mlp_forward(x)

        assert result.shape == x.shape
        assert not torch.isnan(result).any()

    def test_taunet_zero_input(self):
        """Test TauNet with zero input."""
        taunet = TauNet()
        k = torch.zeros(10, 3)

        result = taunet(k)
        assert not torch.isnan(result).any()
        assert torch.all(result >= 0)

    def test_taunet_negative_input(self):
        """Test TauNet with negative input values."""
        taunet = TauNet()
        k = torch.randn(10, 3) * -1  # Negative values

        result = taunet(k)
        assert not torch.isnan(result).any()
        assert torch.all(result >= 0)

    def test_taunet_different_nu_init(self):
        """Test TauNet with different nu_init values."""
        taunet1 = TauNet(nu_init=-0.5)
        taunet2 = TauNet(nu_init=0.0)

        k = torch.randn(10, 3)
        result1 = taunet1(k)
        result2 = taunet2(k)

        # Results should be different due to different nu_init
        assert not torch.allclose(result1, result2, atol=1e-6)

    def test_taunet_learn_nu_false(self):
        """Test TauNet with learn_nu=False."""
        taunet = TauNet(learn_nu=False)
        assert taunet.Ra.fg_learn_nu is False
        assert not isinstance(taunet.Ra.nu, nn.Parameter)

        k = torch.randn(10, 3)
        result = taunet(k)
        assert not torch.isnan(result).any()
