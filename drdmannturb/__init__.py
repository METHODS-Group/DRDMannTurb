"""
Data-driven ABL turbulence modeling software, intended for synthetic inlet data generation.

DRDMannTurb (short for Deep Rapid Distortion theory Mann Turbulence model) is a data-driven
framework for syntetic turbulence generation in Python. The code is based on the original work
of Jacob Mann in 1994 and 1998 as well as in the deep-learning enhancement developed by Keith
et al. in 2021.
"""

from .common import (
    CPU_Unpickler,
    Mann_linear_exponential_approx,
    MannEddyLifetime,
    VKEnergySpectrum,
    plot_loss_logs,
)
from .enums import EddyLifetimeType
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
from .interpolation import extract_x_spectra, interp_spectra, interpolate
from .nn_modules import CustomMLP, CustomNet, Rational, SimpleNN, TauNet
from .parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)
from .spectra_fitting import (
    CalibrationProblem,
    LossAggregator,
    OnePointSpectra,
)

__all__ = [
    # Common
    "CPU_Unpickler",
    "Mann_linear_exponential_approx",
    "MannEddyLifetime",
    "VKEnergySpectrum",
    "plot_loss_logs",
    # Enums
    "DataType",
    "EddyLifetimeType",
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
    # Interpolation
    "extract_x_spectra",
    "interp_spectra",
    "interpolate",
    # NN Modules
    "CustomMLP",
    "CustomNet",
    "Rational",
    "SimpleNN",
    "TauNet",
    # Parameters
    "LossParameters",
    "NNParameters",
    "PhysicalParameters",
    "ProblemParameters",
    # Spectra Fitting
    "CalibrationProblem",
    "LossAggregator",
    "OnePointSpectra",
    "OnePointSpectraDataGenerator",
]
