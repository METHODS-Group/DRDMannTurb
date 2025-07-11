"""
Fluctuation generation module.

This module contains the classes and functions for generating
fluctuation fields.

The main class is `FluctuationFieldGenerator`, which generates
a fluctuation field from a given covariance kernel.
"""

__all__ = [
    "Covariance",
    "VonKarmanCovariance",
    "MannCovariance",
    "GaussianRandomField",
    "VectorGaussianRandomField",
    "FluctuationFieldGenerator",
    "NNCovariance",
    "Sampling_method_base",
    "Sampling_method_freq",
    "Sampling_FFTW",
    "Sampling_VF_FFTW",
    "Sampling_FFT",
    "Sampling_DST",
    "Sampling_DCT",
    "create_grid",
    "format_wind_field",
    "plot_velocity_components",
    "plot_velocity_magnitude",
]

from .covariance_kernels import Covariance, MannCovariance, VonKarmanCovariance
from .fluctuation_field_generator import FluctuationFieldGenerator
from .gaussian_random_fields import GaussianRandomField, VectorGaussianRandomField
from .nn_covariance import NNCovariance
from .sampling_methods import (
    Sampling_DCT,
    Sampling_DST,
    Sampling_FFT,
    Sampling_FFTW,
    Sampling_method_base,
    Sampling_method_freq,
    Sampling_VF_FFTW,
)
from .wind_plot import (
    create_grid,
    format_wind_field,
    plot_velocity_components,
    plot_velocity_magnitude,
)
