"""
This module defines the Enums determining the fundamental datatypes and choices of models available in the package. 
"""
__all__ = ["DataType", "EddyLifetimeType", "PowerSpectraType"]

from enum import Enum

DataType = Enum("DataType", ["KAIMAL", "CUSTOM", "AUTO", "VK", "IEC"])

EddyLifetimeType = Enum(
    "EddyLifetimeType",
    ["TWOTHIRD", "CUSTOMMLP", "TAUNET", "MANN", "MANN_APPROX", "CONST"],
)

PowerSpectraType = Enum("PowerSpectraType", ["RDT"])

SamplingMethod = Enum("SamplingMethod", ["DST", "DCT", "FFT", "FFTW", "VF_FFTW"])
