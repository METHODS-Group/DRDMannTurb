"""
This module defines the Enums for quality of life
"""
__all__ = ["DataType", "EddyLifetimeType", "PowerSpectraType"]

from enum import Enum

DataType = Enum(
    "DataType", ["KAIMAL", "CUSTOM", "SIMIU_SCANLAN", "SIMIU_YEO", "AUTO", "VK", "IEC"]
)

EddyLifetimeType = Enum(
    "EddyLifetimeType",
    ["TWOTHIRD", "CUSTOMMLP", "TAUNET", "TAURESNET", "MANN", "CONST"],
)

PowerSpectraType = Enum("PowerSpectraType", ["RDT"])
