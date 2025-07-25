"""
Data-driven ABL turbulence modeling software, intended for synthetic inlet data generation.

DRDMannTurb (short for Deep Rapid Distortion theory Mann Turbulence model) is a data-driven
framework for syntetic turbulence generation in Python. The code is based on the original work
of Jacob Mann in 1994 and 1998 as well as in the deep-learning enhancement developed by Keith
et al. in 2021.
"""

from .fluctuation_generation import (
    Covariance,
    FluctuationFieldGenerator,
    GaussianRandomField,
    MannCovariance,
    NNCovariance,
    Sampling_DCT,
    Sampling_DST,
    Sampling_FFT,
    Sampling_FFTW,
    Sampling_method_base,
    Sampling_method_freq,
    Sampling_VF_FFTW,
    VectorGaussianRandomField,
    VonKarmanCovariance,
    create_grid,
    format_wind_field,
    plot_velocity_components,
    plot_velocity_magnitude,
)
from .nn_modules import Rational, TauNet
from .parameters import (
    IntegrationParameters,
    LossParameters,
)
from .spectra_fitting import (
    CalibrationProblem,
    CustomDataLoader,
    LossAggregator,
    OnePointSpectra,
    generate_kaimal_spectra,
    generate_von_karman_spectra,
)

__all__ = [
    # Fluctuation Generation
    "Covariance",
    "FluctuationFieldGenerator",
    "GaussianRandomField",
    "MannCovariance",
    "NNCovariance",
    "Sampling_DCT",
    "Sampling_DST",
    "Sampling_FFT",
    "Sampling_FFTW",
    "Sampling_method_base",
    "Sampling_method_freq",
    "Sampling_VF_FFTW",
    "VectorGaussianRandomField",
    "VonKarmanCovariance",
    "create_grid",
    "format_wind_field",
    "plot_velocity_components",
    "plot_velocity_magnitude",
    # NN Modules
    "Rational",
    "TauNet",
    # Parameters
    "IntegrationParameters",
    "LossParameters",
    # Spectra Fitting
    "CalibrationProblem",
    "CustomDataLoader",
    "generate_kaimal_spectra",
    "generate_von_karman_spectra",
    "LossAggregator",
    "OnePointSpectra",
]
