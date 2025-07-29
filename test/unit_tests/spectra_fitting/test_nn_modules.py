"""Test the nn_modules module."""

from drdmannturb.nn_modules import TauNet


class TestTauNet:
    """Test the TauNet module."""

    def test_taunet_constructor(self):
        """Test the constructor for the TauNet module."""
        _taunet = TauNet(n_layers=2, hidden_layer_sizes=[5, 5])

        raise NotImplementedError("Write tests for the TauNet module.")
