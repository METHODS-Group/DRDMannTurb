"""Definitions of various eddy lifetime models."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class EddyLifetimeFunction(nn.Module, ABC):
    """Base class for eddy lifetime functions."""

    def __init__(
        self,
        length_scale: Optional[float] = None,
        time_scale: Optional[float] = None,
    ):
        """Initialize base eddy lifetime function."""
        super().__init__()
        self.length_scale = length_scale
        self.time_scale = time_scale

    @abstractmethod
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Evaluate the eddy lifetime function.

        Parameters
        ----------
        k : torch.Tensor
            Wavevector tensor of shape (..., 3)

        Returns
        -------
        torch.Tensor
            Eddy lifetime evaluation with shape matching k.norm(dim=-1)
        """
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, k: torch.Tensor) -> torch.Tensor:
        """Call forward method.

        This enables the module to be called like a function.

        Parameters
        ----------
        k : torch.Tensor
            Wavevector tensor

        Returns
        -------
        torch.Tensor
            Result of forward pass
        """
        return self.forward(k)
