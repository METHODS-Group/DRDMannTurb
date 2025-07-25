"""Several dataclasses that make it easy to pass around parameters."""

from dataclasses import dataclass
from typing import Union

import numpy as np

__all__ = ["IntegrationParameters", "LossParameters"]


@dataclass
class LossParameters:
    r"""Set coefficients for loss function terms.

    This class provides a convenient method of storing and passing around
    the loss function term coefficients; this also offers default values, which result in the loss function
    consisting purely of an MSE loss.

    Args
    ----
    alpha_pen1 : float
        Set the first-order penalization term coefficient :math:`\alpha_1`, by default 0.0
    alpha_pen2 : float
        Set the second-order penalization term coefficient :math:`\alpha_2`, by default 0.0
    beta_reg : float
        Set the regularization term coefficient :math:`\beta`, by default 0.0
    """

    alpha_pen1: float = 0.0
    alpha_pen2: float = 0.0

    beta_reg: float = 0.0

    gamma_coherence: float = 0.0


#######################################################################################################
# 	Fluctuation field generation parameters
#######################################################################################################


@dataclass
class DomainParameters:
    r"""Define domain parameters for the fluctuation field generation component.

    This class provides a convenient method of storing and passing around
    the domain parameters required for the fluctuation field generation.

    Args
    ----
    grid_dimensions : Tuple[float, float, float]
        The dimensions of the grid in the x, y, and z directions.
    grid_levels : Tuple[int, int, int]
        The number of grid points in the x, y, and z directions.
        These are calculated as :math:`2^{\text{grid\_levels}} + 1`.
    """

    d_dimensions: Union[tuple[float, float, float], "np.ndarray", list[float]]
    d_levels: Union[tuple[int, int, int], "np.ndarray", list[int]]

    def __post_init__(self):
        # Check grid dimensions
        if isinstance(self.d_dimensions, (list, np.ndarray)) and len(self.d_dimensions) != 3:
            raise ValueError("Grid dimensions must be length 3.")

        if isinstance(self.d_dimensions, np.ndarray):
            self._grid_dimensions = self.d_dimensions
        else:
            self._grid_dimensions = np.array(self.d_dimensions)

        # Check grid levels
        if isinstance(self.grid_levels, (list, np.ndarray)) and len(self.grid_levels) != 3:
            raise ValueError("Grid levels must be a tuple of length 3.")

        if isinstance(self.d_levels, np.ndarray):
            self._grid_levels = self.d_levels
        else:
            self._grid_levels = np.array(self.d_levels)

        del self.d_dimensions
        del self.d_levels

    @property
    def grid_dimensions(self):
        """Getter for grid dimensions."""
        return self._grid_dimensions

    @property
    def grid_levels(self):
        """Getter for grid levels."""
        return self._grid_levels


@dataclass
class LowFreqParameters:
    r"""Define parameters for the low-frequency model extension.

    This class provides a convenient method of storing and passing around
    the physical parameters required for the low-frequency model extension.

    The default values are taken from the Syed-Mann (2024) paper.
    """

    L_2D: float = 15_000.0
    sigma2: float = 0.6
    z_i: float = 500.0
    psi_degs: float = 43.0

    c: float | None = None


@dataclass
class IntegrationParameters:
    """Parameters for integration grids in spectra calculations.

    This dataclass defines the parameters for creating log-spaced grids
    used in one-point spectra and coherence integrations.

    Attributes
    ----------
        ops_log_min: Minimum exponent for OPS grid (default -3).
        ops_log_max: Maximum exponent for OPS grid (default 3).
        ops_num_points: Number of points per side for OPS grid (default 100).
        coh_log_min: Minimum exponent for coherence grid (default -3).
        coh_log_max: Maximum exponent for coherence grid (default 3).
        coh_num_points: Number of points per side for coherence grid (default 100).
    """

    ops_log_min: float = -3.0
    ops_log_max: float = 3.0
    ops_num_points: int = 100
    coh_log_min: float = -3.0
    coh_log_max: float = 3.0
    coh_num_points: int = 100
