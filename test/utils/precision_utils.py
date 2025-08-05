"""Utilities for precision testing."""

import contextlib
from typing import Any, Dict, Tuple

import torch


def compare_precisions(func, *args, **kwargs) -> Dict[torch.dtype, Any]:
    """Run a function with different precisions and return results."""
    results = {}

    for dtype in [torch.float32, torch.float64]:
        # Convert args to dtype
        args_dtype = convert_args_to_dtype(args, dtype)
        kwargs_dtype = convert_kwargs_to_dtype(kwargs, dtype)

        results[dtype] = func(*args_dtype, **kwargs_dtype)

    return results


def convert_args_to_dtype(args: Tuple, dtype: torch.dtype) -> Tuple:
    """Convert tensor arguments to specified dtype."""
    converted = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            converted.append(arg.to(dtype))
        else:
            converted.append(arg)
    return tuple(converted)


def convert_kwargs_to_dtype(kwargs: Dict, dtype: torch.dtype) -> Dict:
    """Convert tensor keyword arguments to specified dtype."""
    converted = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            converted[key] = value.to(dtype)
        else:
            converted[key] = value
    return converted


def check_precision_consistency(results: Dict[torch.dtype, Any], tolerance: float = 1e-5) -> bool:
    """Check that results are consistent across precisions."""
    if torch.float32 not in results or torch.float64 not in results:
        return True

    float32_result = results[torch.float32]
    float64_result = results[torch.float64]

    if isinstance(float32_result, torch.Tensor) and isinstance(float64_result, torch.Tensor):
        # Convert to same dtype for comparison
        float32_result = float32_result.to(torch.float64)
        rel_error = torch.abs(float32_result - float64_result) / (torch.abs(float64_result) + 1e-12)
        return bool(torch.max(rel_error) < tolerance)

    return float32_result == float64_result


@contextlib.contextmanager
def precision_context(dtype: torch.dtype):
    """Context manager for testing with specific precision."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)
