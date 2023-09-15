"""
This module defines the Enums for quality of life
"""

from enum import Enum


DataType = Enum(
    "DataType", ["KAIMAL", "CUSTOM", "SIMIU_SCANLAN", "SIMIU_YEO", "AUTO", "VK", "IEC"]
)

EddyLifetimeType = Enum(
    "EddyLifetime", ["TWOTHIRD", "CUSTOMMLP", "TAUNET", "TAURESNET"]
)

PowerSpectraType = Enum("PowerSpectra", ["RDT"])
