"""Regression tests for VTK export metadata."""

import numpy as np

from drdmannturb.fluctuation_generation import FluctuationFieldGenerator, MannCovariance


def test_vtk_spacing_uses_plain_numeric_metadata(tmp_path):
    """NumPy scalars must not leak into the VTI Spacing attribute."""
    generator = FluctuationFieldGenerator(
        grid_dimensions=np.array([3.0, 4.0, 5.0]),
        grid_levels=np.array([1, 1, 1]),
        covariance=MannCovariance(length_scale=0.5, E0=1.0, Gamma=1.0),
        seed=1,
    )
    generator.total_fluctuation = np.zeros((2, 2, 2, 3))

    output = tmp_path / "field"
    generator.save_to_vtk(output)

    raw = output.with_suffix(".vti").read_bytes()
    header = raw[: raw.index(b"<AppendedData")].decode()
    assert "np.float" not in header
    assert 'Spacing="1.0 1.3333333333333333 1.6666666666666667 "' in header
