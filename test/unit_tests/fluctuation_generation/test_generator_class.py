"""Tests behavior for the FluctuationFieldGenerator class and methods."""

import numpy as np
import pytest

from drdmannturb.fluctuation_generation import FluctuationFieldGenerator, VonKarmanCovariance

# TODO: Get baseline for current implementation.


class TestFluctuationFieldGenerator:
    """Test the FluctuationFieldGenerator class."""

    @pytest.fixture
    def basic_params(self):
        """Create dictionary of common parameters for test initialization."""
        return {
            "grid_dimensions": np.array([1000.0, 500.0, 300.0]),
            "grid_levels": np.array([6, 6, 5]),
            "covariance": VonKarmanCovariance(length_scale=20.0, E0=1.0),
            "seed": 42,
        }

    def test_init_and_generate_basic(self, basic_params):
        """Generator initializes and produces a non-empty field of correct rank."""
        gen = FluctuationFieldGenerator(**basic_params)

        zref = 50.0
        uref = 10.0
        z0 = 0.1
        windprofiletype = "LOG"

        field = gen.generate(
            num_blocks=1,
            zref=zref,
            uref=uref,
            z0=z0,
            windprofiletype=windprofiletype,
        )

        assert field.ndim == 4
        assert field.shape[-1] == 3
        assert field.shape[0] > 0 and field.shape[1] > 0 and field.shape[2] > 0

    def test_repeat_generate_appends(self, basic_params):
        """Second generate() call appends more blocks to the time/x dimension."""
        gen = FluctuationFieldGenerator(**basic_params)

        zref = 50.0
        uref = 10.0
        z0 = 0.1

        field1 = gen.generate(1, zref, uref, z0, "LOG")
        field2 = gen.generate(1, zref, uref, z0, "LOG")

        assert field2.shape[0] > field1.shape[0]
