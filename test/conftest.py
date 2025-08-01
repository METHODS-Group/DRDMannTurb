"""Enhanced test configuration with precision and platform support."""

import pytest
import torch


# Global test configuration
def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption(
        "--precision", action="store", default="both", choices=["float32", "float64", "both"], help="test precision"
    )
    parser.addoption("--platform", action="store", default="both", choices=["cpu", "gpu", "both"], help="test platform")
    parser.addoption("--memory", action="store_true", default=False, help="run memory tests")

    # Group-specific options
    parser.addoption(
        "--group",
        action="store",
        default=None,
        choices=["unit", "integration", "precision", "gpu", "memory", "performance"],
        help="run specific test group",
    )


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "precision: mark test as precision verification")
    config.addinivalue_line("markers", "gpu: mark test as GPU-dependent")
    config.addinivalue_line("markers", "memory: mark test as memory-intensive")
    config.addinivalue_line("markers", "performance: mark test as performance-critical")

    # Group markers
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "spectra_fitting: spectra fitting tests")
    config.addinivalue_line("markers", "fluctuation_generation: fluctuation generation tests")


def pytest_collection_modifyitems(config, items):
    """Modify pytest collection based on options."""
    # Group filtering
    group = config.getoption("--group")
    if group:
        if group == "unit":
            items[:] = [item for item in items if "unit" in item.keywords]
        elif group == "integration":
            items[:] = [item for item in items if "integration" in item.keywords]
        elif group == "precision":
            items[:] = [item for item in items if "precision" in item.keywords]
        elif group == "gpu":
            items[:] = [item for item in items if "gpu" in item.keywords]
        elif group == "memory":
            items[:] = [item for item in items if "memory" in item.keywords]
        elif group == "performance":
            items[:] = [item for item in items if "performance" in item.keywords]

    # Slow test filtering
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Precision filtering - only for precision verification tests
    precision = config.getoption("--precision")
    if precision != "both":
        skip_precision = pytest.mark.skip(reason=f"precision verification test, run with --precision={precision}")
        for item in items:
            if "precision" in item.keywords and precision not in item.name:
                item.add_marker(skip_precision)

    # Platform filtering
    platform = config.getoption("--platform")
    if platform != "both":
        skip_platform = pytest.mark.skip(reason=f"platform test, run with --platform={platform}")
        for item in items:
            if "gpu" in item.keywords and platform == "cpu":
                item.add_marker(skip_platform)


# Fixtures for different precisions
@pytest.fixture(params=[torch.float32, torch.float64])
def precision_dtype(request):
    """Fixture providing different precision dtypes."""
    return request.param


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    """Fixture providing different devices."""
    return request.param


@pytest.fixture
def sample_k(precision_dtype, device="cpu"):
    """Sample wavevector tensor with specified precision and device."""
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=precision_dtype, device=device)


@pytest.fixture
def sample_params(precision_dtype, device="cpu"):
    """Sample parameters with specified precision."""
    return (
        torch.tensor(10.0, dtype=precision_dtype, device=device),
        torch.tensor(4.0, dtype=precision_dtype, device=device),
    )
