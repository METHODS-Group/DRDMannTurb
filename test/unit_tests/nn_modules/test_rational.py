"""Test the Rational module."""

import pytest
import torch

from drdmannturb.nn_modules import Rational


@pytest.mark.unit
class TestRational:
    """Test the Rational module."""

    def test_rational_constructor(self):
        """Test the constructor for the Rational module."""
        rational = Rational(learn_nu=True, nu_init=-1.0 / 3.0)
        assert rational is not None
        assert rational.fg_learn_nu is True
        assert isinstance(rational.nu, torch.nn.Parameter)

    def test_rational_constructor_no_learn_nu(self):
        """Test the constructor with learn_nu=False."""
        rational = Rational(learn_nu=False, nu_init=-1.0 / 3.0)
        assert rational is not None
        assert rational.fg_learn_nu is False
        assert not isinstance(rational.nu, torch.nn.Parameter)

    def test_rational_forward(self):
        """Test the forward pass of the Rational module."""
        rational = Rational(learn_nu=True, nu_init=-1.0 / 3.0)
        x = torch.tensor([1.0, 2.0, 3.0])

        result = rational(x)

        assert result.shape == x.shape
        assert torch.all(result >= 0)  # Should be non-negative
        assert not torch.isnan(result).any()

    def test_rational_extreme_values(self):
        """Test the Rational module with extreme values."""
        rational = Rational(learn_nu=True, nu_init=-1.0 / 3.0)

        # Very large values
        x_large = torch.tensor([1e6, 1e7, 1e8])
        result_large = rational(x_large)
        assert not torch.isnan(result_large).any()
        assert not torch.isinf(result_large).any()

        # Very small values
        x_small = torch.tensor([1e-6, 1e-7, 1e-8])
        result_small = rational(x_small)
        assert not torch.isnan(result_small).any()
        assert not torch.isinf(result_small).any()

    @pytest.mark.precision
    def test_rational_precision(self, precision_dtype, device):
        """Test the Rational module with different precisions."""
        rational = Rational(learn_nu=True, nu_init=-1.0 / 3.0)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=precision_dtype)

        # Move to device
        x = x.to(device)

        result = rational(x)
        assert result.dtype == precision_dtype
        assert result.device == x.device
        assert not torch.isnan(result).any()

    def test_rational_gradients(self):
        """Test that the Rational module produces gradients."""
        rational = Rational(learn_nu=True, nu_init=-1.0 / 3.0)
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        result = rational(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        if rational.fg_learn_nu:
            assert rational.nu.grad is not None
            assert torch.isfinite(rational.nu.grad).all()
