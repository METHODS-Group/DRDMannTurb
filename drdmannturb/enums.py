"""Enums determining the fundamental datatypes and choices of models available in the package."""

__all__ = ["EddyLifetimeType"]
from enum import Enum

EddyLifetimeType = Enum(
    "EddyLifetimeType",
    ["TAUNET", "MANN", "CONST", "TWOTHIRD"],
)
