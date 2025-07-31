"""Fixture data for spectral testing."""

from typing import Dict

import torch


def create_kaimal_spectra_data(n_points: int = 100) -> Dict[str, torch.Tensor]:
    """Create synthetic Kaimal spectra data for testing."""
    freq = torch.logspace(-2, 2, n_points)

    # Synthetic spectra following Kaimal form
    uu = 100.0 * freq / (1 + 5.67 * freq) ** (5 / 3)
    vv = 40.0 * freq / (1 + 5.67 * freq) ** (5 / 3)
    ww = 15.0 * freq / (1 + 5.67 * freq) ** (5 / 3)

    # Cross-spectra (simplified)
    uw = -20.0 * freq / (1 + 5.67 * freq) ** (5 / 3)
    vw = 0.0 * torch.ones_like(freq)
    uv = 0.0 * torch.ones_like(freq)

    return {"freq": freq, "uu": uu, "vv": vv, "ww": ww, "uw": uw, "vw": vw, "uv": uv}


def create_test_wavevector_grid(n_points: int = 50, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a test wavevector grid."""
    k1 = torch.logspace(-2, 2, n_points, dtype=dtype)
    k2 = torch.logspace(-2, 2, n_points, dtype=dtype)
    k3 = torch.logspace(-2, 2, n_points, dtype=dtype)

    k1_grid, k2_grid, k3_grid = torch.meshgrid(k1, k2, k3, indexing="ij")
    k = torch.stack([k1_grid.flatten(), k2_grid.flatten(), k3_grid.flatten()], dim=-1)

    return k


def create_test_parameters(dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    """Create test parameters for spectral tensor models."""
    return {
        "L": torch.tensor(10.0, dtype=dtype),
        "gamma": torch.tensor(4.0, dtype=dtype),
        "sigma": torch.tensor(3.0, dtype=dtype),
    }
