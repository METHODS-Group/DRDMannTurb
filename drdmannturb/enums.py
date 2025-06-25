"""Enums determining the fundamental datatypes and choices of models available in the package."""

__all__ = ["DataType", "EddyLifetimeType"]
from enum import Enum

DataType = Enum(
    "DataType",
    [
        "KAIMAL",
        "CUSTOM",
        "VK",
    ],
)

EddyLifetimeType = Enum(
    "EddyLifetimeType",
    ["TWOTHIRD", "CUSTOMMLP", "TAUNET", "MANN", "MANN_APPROX", "CONST"],
)

SamplingMethod = Enum("SamplingMethod", ["DST", "DCT", "FFT", "FFTW", "VF_FFTW"])
