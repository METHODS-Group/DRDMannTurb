"""Helper functions for testing."""

from typing import Dict, Tuple

import torch


def create_test_wavevectors(n_points: int = 100, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create test wavevectors for spectral tensor testing."""
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


def assert_tensor_properties(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...] | None = None,
    expected_dtype: torch.dtype | None = None,
    check_finite: bool = True,
) -> None:
    """Assert tensor properties."""
    if expected_shape is not None:
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"

    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"

    if check_finite:
        assert torch.isfinite(tensor).all(), f"Tensor contains non-finite values: {tensor}"


def benchmark_function(func, *args, **kwargs) -> Dict[str, float]:
    """Benchmark a function execution."""
    import time

    # Warmup
    for _ in range(3):
        func(*args, **kwargs)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    for _ in range(10):
        _result = func(*args, **kwargs)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    return {"total_time": end_time - start_time, "avg_time": (end_time - start_time) / 10}
