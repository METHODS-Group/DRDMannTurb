"""Tests behavior for the FluctuationFieldGenerator class and methods."""

import numpy as np
import pytest

from drdmannturb.fluctuation_generation.fluctuation_field_generator import FluctuationFieldGenerator

# TODO: Get baseline for current implementation.


@pytest.mark.skip(reason="Not implemented")
def test_generator_class():
    pass


class TestFluctuationFieldGenerator:
    @pytest.fixture
    def basic_params(self):
        """Common parameters for test initialization"""

        return {
            "friction_velocity": 1.0,
            "reference_height": 100.0,
            "grid_dimensions": np.array([1000.0, 500.0, 300.0]),
            "grid_levels": np.array([6, 6, 5]),
            "seed": 42,
        }

    def test_init_invalid_model(self, basic_params):
        """Test initialization with invalid model"""

        params = basic_params.copy()
        params = basic_params.copy()
        params.update({"model": "InvalidModel", "length_scale": 100.0, "time_scale": 3.0, "energy_spectrum_scale": 1.7})

        with pytest.raises(ValueError, match="Model must be one of"):
            FluctuationFieldGenerator(**params)
